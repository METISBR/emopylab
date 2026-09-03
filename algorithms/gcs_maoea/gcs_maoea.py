# emopylab 2026
"""GCS-MaOEA — Geometry-Coupled Search for Many-Objective Optimization.

This is the main algorithm module. GCS-MaOEA is designed for unconstrained
many-objective optimization (MAOPs, M > 3), with a single Geometry-Coupled
Search mechanism that links objective-space reference geometry to
decision-space variation:

    1. Adaptive Reference-Vector Repair (ARV)
       -> ``reference_repair.py``
       The uniform weight vectors are projected onto the adaptive
       normal hyperplane estimated from ASF extreme points.  This makes the
       reference structure follow the true geometry of the Pareto front.

    2. Direction-partitioned Decision-Variable Analysis (DVA)
       -> ``dva.py``
       For large-scale problems, binary variable masks are evolved and scored
       by both convergence and their coverage of the repaired reference
       directions. The same repaired geometry partitions variables and drives
       direction-targeted reproduction.

    3. Dynamic Multi-Archive Framework
       -> ``archives.py``
       Three archives (exploration, diversity, convergence) share the
       population budget adaptively.  The diversity archive uses the repaired
       reference vectors for niching.

The class follows the standard EmoPyLab Algorithm interface:
    ``_initialize_infill`` -> ``_initialize_advance`` -> ``_infill`` -> ``_advance``.

For the article, this module maps to:
    - Section 3: overall framework and main loop (Algorithm 1).
    - Section 4: experimental setup and metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from core.population import Population
from util.optimum import filter_optimum

from core.algorithm import Algorithm
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorDE import OperatorDE
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.UniformPoint import UniformPoint

from algorithms.community_utils.moead_family import (
    current_fe,
    fe_ratio,
    max_fe,
    sample_initial,
    tchebycheff_values,
)

from .archives import (
    ConvergenceArchive,
    DiversityArchive,
    DynamicArchiveAllocator,
    ExplorationArchive,
)
from .dva import DVAManager, detect_variable_interactions, directional_variable_analysis
from .reference_repair import (
    adapt_reference_vectors,
    assess_front_degeneracy,
    find_extreme_points,
    generate_manifold_vectors,
    generate_manifold_vectors_clustered,
    normal_hyperplane,
    repair_reference_vectors,
    sine_distance,
    update_ideal_nadir,
    update_nadir_from_extremes,
)

ALGORITHM_FLAGS = {
    "GCS-MaOEA": {"multi", "many"},
    "GCSMaOEA": {"multi", "many"},
}


def _cosine_neighbor_structure(W: np.ndarray, T: int) -> np.ndarray:
    """Neighbor structure B: the T nearest vectors of W by cosine angle.

    The article's warmup builds the MOEA/D neighborhood from cosine
    proximity between reference directions (not Euclidean distance), so
    neighbors share search direction rather than weight-vector position.
    """
    W = np.asarray(W, dtype=float)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    U = W / norms
    similarity = U @ U.T
    order = np.argsort(-similarity, axis=1, kind="stable")
    return order[:, : max(1, min(int(T), W.shape[0]))]


class GCSMaOEA(Algorithm):
    """Geometry-Coupled Search for many-objective optimization.

    Parameters
    ----------
    pop_size : int
        Target population size.  The actual size follows the number of
        generated reference directions.
    ref_dirs : np.ndarray | None
        Optional user-supplied reference directions.  If None, they are
        generated uniformly with ``UniformPoint`` and then repaired online.
    sampling : Sampling | None
        EmoPyLab sampling object for the initial population.
    use_dva : bool
        Enable direction-aware decision-variable analysis.  Automatically
        disabled when ``problem.n_var < 100``.
    dva_masks, dva_samples, dva_generations : int
        Hyperparameters of the DVA optimizer.
    de_cr, de_f : float
        DE crossover probability and scaling factor.
    warmup_generations : int
        Number of decomposition-based generations used as a warm-up before
        the main loop.
    seed : int | None
        Random seed forwarded to EmoPyLab.
    """

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Optional[np.ndarray] = None,
        sampling=None,
        use_dva: bool = True,
        dva_masks: int = 10,
        dva_samples: int = 3,
        dva_generations: int = 20,
        de_cr: float = 1.0,
        de_f: float = 0.5,
        warmup_generations: int = 5,
        analyze_variables: bool = True,
        couple_reproduction: bool = True,
        use_arv: bool = True,
        dynamic_allocation: bool = True,
        convergence_repair_rate: float = 1.0,
        partition_period: int = 5,
        partition_max_probe: int = 128,
        partition_base_count: int = 3,
        nadir_smoothing: float = 0.5,
        adapt_references: bool = True,
        manifold_vectors: bool = True,
        manifold_stable_gens: int = 3,
        manifold_reduction_min: int = 1,
        manifold_gap_min: float = 3.0,
        piecewise_manifold: bool = True,
        manifold_accept_ratio: float = 0.75,
        use_twonn: bool = True,
        convergence_targets: bool = True,
        consensus_targets: bool = False,
        linkage_learning: bool = True,
        linkage_pair_budget: int = 200,
        linkage_refresh_period: int = 20,
        stagnation_window: int = 8,
        seed: Optional[int] = None,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )
        self.pop_size = int(max(pop_size, 3))
        self.ref_dirs_uniform: Optional[np.ndarray] = (
            None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        )
        self.ref_dirs_repaired: Optional[np.ndarray] = None
        self.sampling = sampling

        self.use_dva = bool(use_dva)
        self.dva_masks = int(dva_masks)
        self.dva_samples = int(dva_samples)
        self.dva_generations = int(dva_generations)

        self.de_cr = float(np.clip(de_cr, 0.0, 1.0))
        self.de_f = float(max(0.0, de_f))
        self.warmup_generations = int(max(0, warmup_generations))

        # Geometry-Coupled Search (GCS): direction-partitioned variables drive
        # direction-targeted reproduction. ``couple_reproduction`` is the
        # ablation switch (False -> standard, decoupled variation).
        self.couple_reproduction = bool(couple_reproduction)
        # Ablation switches (all True/on in the full method):
        #   manifold_vectors   -> B1 (-Manifold): off falls back to simplex repair,
        #                         isolating the intrinsic-dimension-adaptive vectors.
        #   analyze_variables  -> B2 (-DVA): off disables direction-aware DVA.
        #   couple_reproduction-> off disables the reproduction direction mask.
        #   use_arv            -> off uses the uniform vectors directly (no repair).
        #   dynamic_allocation -> off uses fixed N/3 archive budgets.
        self.analyze_variables = bool(analyze_variables)
        self.use_arv = bool(use_arv)
        self.dynamic_allocation = bool(dynamic_allocation)
        self.convergence_repair_rate = float(np.clip(convergence_repair_rate, 0.0, 1.0))
        self.partition_period = int(max(1, partition_period))
        self.partition_max_probe = int(max(1, partition_max_probe))
        self.partition_base_count = int(max(1, partition_base_count))
        self.dir_to_vars: Optional[list] = None
        self.var_to_dir: Optional[np.ndarray] = None
        self.var_targets: Optional[np.ndarray] = None
        self._partition_age = 0

        # Online front-geometry adaptation (problem-agnostic; all triggers are
        # measured signals, never the problem identity).
        self.nadir_smoothing = float(np.clip(nadir_smoothing, 0.0, 1.0))
        self.adapt_references = bool(adapt_references)
        # Intrinsic-dimension-adaptive reference generation (B1 ablation switch).
        # When the estimated front dimension d* stays below M-1 for
        # ``manifold_stable_gens`` consecutive generations, reference vectors are
        # generated directly inside the d*-manifold (``generate_manifold_vectors``)
        # rather than projected onto the full (M-1)-simplex.
        self.manifold_vectors = bool(manifold_vectors)
        self.manifold_stable_gens = int(max(1, manifold_stable_gens))
        # Degeneracy gate: engage manifold generation only when the front removes
        # at least ``manifold_reduction_min`` dimensions relative to M-1 AND the
        # singular spectrum shows a sharp gap of at least ``manifold_gap_min``.
        # This rejects regular high-M fronts whose converged cloud looks low-rank
        # under PCA, and avoids engaging on low-M cases where simplex repair is
        # already adequate.
        self.manifold_reduction_min = int(max(0, manifold_reduction_min))
        self.manifold_gap_min = float(max(1.0, manifold_gap_min))
        # Piecewise (per-cluster) manifold generation: local PCA per connected
        # component of the front, covering disconnected/curved degenerate
        # manifolds that a single global linear subspace cannot represent.
        self.piecewise_manifold = bool(piecewise_manifold)
        # Collapse guard for the manifold candidate (see _update_reference_frame).
        self.manifold_accept_ratio = float(np.clip(manifold_accept_ratio, 0.0, 1.0))
        # TwoNN committee member for the intrinsic-dimension estimate
        # (sample-efficient; extends the mechanism to sparse high-M clouds).
        self.use_twonn = bool(use_twonn)
        # B3 ablation switch: convergence-target machinery (learned variable
        # targets, repair/manifold populations, and the selection reserve).
        self.convergence_targets = bool(convergence_targets)
        # B6 ablation switch: population-consensus target filling.  Unlike the
        # radial sweep detector (which requires monotone evidence), the
        # consensus filler infers targets from clustering alone and can pin
        # variables to spurious values on converged regular fronts.
        self.consensus_targets = bool(consensus_targets)
        # Linkage learning (budgeted differential grouping) for nonseparable
        # problems: variable interaction groups constrain direction masking.
        self.linkage_learning = bool(linkage_learning)
        self.linkage_pair_budget = int(max(0, linkage_pair_budget))
        self.linkage_refresh_period = int(max(1, linkage_refresh_period))
        self.var_groups: list = []
        self.var_group_of: Optional[np.ndarray] = None
        self._linkage_age = 10 ** 9  # force detection at first opportunity
        self._low_dim_streak = 0
        # Latches True the first time the manifold path is actually taken, and
        # counts the generations it was active — a faithful record of engagement
        # across the whole run (the instantaneous streak at the final generation
        # is too noisy to use as a diagnostic).
        self._manifold_ever_engaged = False
        self._manifold_active_gens = 0
        self._manifold_last_fraction = 0.0
        self._manifold_fraction_sum = 0.0
        self._manifold_mean_fraction = 0.0
        self._reference_updates = 0
        self.stagnation_window = int(max(1, stagnation_window))

        # State variables.
        self.z_min: Optional[np.ndarray] = None
        self.z_nad: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.front_dim: Optional[int] = None
        self.extreme_points: Optional[np.ndarray] = None
        self.neighbor_indices: Optional[np.ndarray] = None
        self.dva_manager: Optional[DVAManager] = None

        # Stagnation tracking (drives adaptive variation; see ``_operator_de``).
        self._best_convergence: float = float("inf")
        self._stall_count: int = 0

        # Archives and allocator.
        self.exploration_archive: Optional[ExplorationArchive] = None
        self.diversity_archive: Optional[DiversityArchive] = None
        self.convergence_archive: Optional[ConvergenceArchive] = None
        self.allocator: Optional[DynamicArchiveAllocator] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self, problem, **kwargs):
        if self.ref_dirs_uniform is None:
            generated, n_effective = UniformPoint(self.pop_size, int(problem.n_obj))
            self.ref_dirs_uniform = np.asarray(generated, dtype=float)
            self.pop_size = int(max(1, n_effective))
        else:
            self.ref_dirs_uniform = np.asarray(self.ref_dirs_uniform, dtype=float)
            self.pop_size = max(1, len(self.ref_dirs_uniform))

        # Article warmup: neighbor structure B from W^U, 10% nearest by
        # cosine angle.
        self.neighbor_indices = _cosine_neighbor_structure(
            self.ref_dirs_uniform, max(2, self.pop_size // 10)
        )
        self.allocator = DynamicArchiveAllocator(self.pop_size)

        try:
            total_fe = int(max_fe(self))
        except Exception:
            total_fe = None

        self.dva_manager = DVAManager(
            n_var=int(problem.n_var),
            n_masks=self.dva_masks,
            n_samples=self.dva_samples,
            n_generations=self.dva_generations,
            enabled=self.use_dva,
            period=self.partition_period,
            max_fe=total_fe,
            fe_budget_ratio=0.10,
        )

        self.exploration_archive = ExplorationArchive(self.pop_size // 3, self.random_state)
        self.diversity_archive = DiversityArchive(self.pop_size // 3, self.ref_dirs_uniform, self.random_state)
        self.convergence_archive = ConvergenceArchive(self.pop_size // 3, self.random_state)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _eval_budget_limit(self) -> int:
        """Return the active FE limit, or 0 when it is not available."""
        try:
            return int(max_fe(self))
        except Exception:
            return 0

    def _remaining_eval_budget(self) -> int:
        """Return the number of objective evaluations still available."""
        limit = self._eval_budget_limit()
        if limit <= 0:
            return 10**12
        return max(0, limit - current_fe(self))

    def _truncate_to_eval_budget(self, pop: Population) -> Population:
        """Keep only candidates that can still be evaluated under the FE limit."""
        if len(pop) == 0:
            return pop
        remaining = self._remaining_eval_budget()
        if remaining <= 0:
            return Population.empty()
        if len(pop) <= remaining:
            return pop
        return pop[:remaining]

    def _initialize_infill(self):
        n_initial = min(self.pop_size, self._remaining_eval_budget())
        if n_initial <= 0:
            return Population.empty()
        return sample_initial(self.problem, n_initial, self.sampling, self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            self.pop = Population.empty()
            self.opt = self.pop
            return

        self.pop = infills

        # Optional decomposition-based warm-up using the uniform reference vectors.
        if self.warmup_generations > 0:
            self.pop = self._decomposition_warmup(self.pop, self.warmup_generations)

        self._update_reference_frame(self.pop)
        self._update_archives(self.pop)
        self._set_optimum()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _infill(self):
        remaining = self._remaining_eval_budget()
        if remaining <= 0:
            return Population.empty()

        if self.pop is None or len(self.pop) == 0:
            n_initial = min(self.pop_size, remaining)
            if n_initial <= 0:
                return Population.empty()
            return sample_initial(self.problem, n_initial, self.sampling, self.random_state)

        # Use the archives as parent pools.  Fall back to the current
        # population if an archive is empty.
        pool1 = self.exploration_archive.archive if self.exploration_archive else self.pop
        pool2 = self.diversity_archive.archive if self.diversity_archive else self.pop
        pool3 = self.convergence_archive.archive if self.convergence_archive else self.pop

        if len(pool1) == 0:
            pool1 = self.pop
        if len(pool2) == 0:
            pool2 = self.pop
        if len(pool3) == 0:
            pool3 = self.pop

        n_off = max(1, self.pop_size // 3)
        idx1_base = self._sample_parent_indices(len(pool1), n_off)
        idx1_1 = self._sample_parent_indices(len(pool1), n_off)
        idx1_2 = self._sample_parent_indices(len(pool1), n_off)
        idx2_base = self._sample_parent_indices(len(pool2), n_off)
        idx2_1 = self._sample_parent_indices(len(pool2), n_off)
        idx2_2 = self._sample_parent_indices(len(pool2), n_off)
        idx3_base = self._sample_parent_indices(len(pool3), n_off)
        idx3_1 = self._sample_parent_indices(len(pool3), n_off)
        idx3_2 = self._sample_parent_indices(len(pool3), n_off)

        if self.random_state.random() < 0.5:
            off1 = self._operator_de(pool1, idx1_base, idx1_1, idx1_2)
            off2 = self._operator_de(pool2, idx2_base, idx2_1, idx2_2)
            off3 = self._operator_de(pool3, idx3_base, idx3_1, idx3_2)
        else:
            off1 = self._operator_ga(pool1, idx1_base)
            off2 = self._operator_ga(pool2, idx2_base)
            off3 = self._operator_ga(pool3, idx3_base)

        offspring = Population.merge(off1, off2, off3)
        if len(offspring) == 0:
            n_initial = min(self.pop_size, self._remaining_eval_budget())
            if n_initial <= 0:
                return Population.empty()
            return sample_initial(self.problem, n_initial, None, self.random_state)
        return self._truncate_to_eval_budget(offspring)

    def _sample_parent_indices(self, pool_size: int, n_parents: int) -> np.ndarray:
        """Sample an archive pool, using replacement only when it is undersized."""
        pool_size = int(pool_size)
        n_parents = int(max(0, n_parents))
        if pool_size <= 0 or n_parents == 0:
            return np.array([], dtype=int)
        return np.asarray(
            self.random_state.choice(
                pool_size,
                size=n_parents,
                replace=pool_size < n_parents,
            ),
            dtype=int,
        )

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return

        merged = Population.merge(self.pop, infills)
        self._update_reference_frame(merged)
        self._update_stagnation()

        # GCS: refresh the direction-partitioned variable assignment periodically
        # from the current elite, then make its probes available to the search.
        # ``analyze_variables`` is the B2 (-DVA) gate; the reproduction coupling
        # has its own exact switch in ``_direction_targeted``.
        if self.analyze_variables:
            linkage_extra = self._refresh_linkage(merged)
            if len(linkage_extra) > 0:
                merged = Population.merge(merged, linkage_extra)
            part_extra = self._refresh_partition(merged)
            if len(part_extra) > 0:
                merged = Population.merge(merged, part_extra)
            if self.convergence_targets:
                manifold_extra = self._convergence_manifold_population(merged)
                if len(manifold_extra) > 0:
                    merged = Population.merge(merged, manifold_extra)
                repaired_extra = self._convergence_repair_population(merged)
                if len(repaired_extra) > 0:
                    merged = Population.merge(merged, repaired_extra)

        # Direction-aware DVA runs periodically for large-scale problems.
        extra = Population.empty()
        if self.dva_manager is not None and self.dva_manager.enabled:
            # Consult the manager every generation; it executes only on its
            # partition-period schedule and under its 10% FE sub-budget.
            estimated_cost = int(self.dva_manager._cost_estimate())
            if self._remaining_eval_budget() >= estimated_cost:
                extra = self.dva_manager.run(
                    self.problem,
                    merged,
                    self.ref_dirs_repaired,
                    self.z_min,
                    self.z_nad,
                    self.random_state,
                    evaluator=self.evaluator,
                )

        if len(extra) > 0:
            merged = Population.merge(merged, extra)
            self._adopt_dva_partition()

        self._update_archives(merged)
        self.pop = self._environmental_selection(merged)
        self._set_optimum()

    # ------------------------------------------------------------------
    # Reference-vector repair
    # ------------------------------------------------------------------

    def _update_reference_frame(self, pop: Population) -> None:
        """Recompute ideal/nadir, extreme points, hyperplane, and repaired vectors."""
        if len(pop) == 0:
            return

        F_all = np.asarray(pop.get("F"), dtype=float)
        # Every reference-geometry equation in the article is defined on the
        # current non-dominated cloud, not on dominated points in the merged set.
        front_no, _ = NDSort(F_all, np.inf)
        nd = np.where(np.asarray(front_no, dtype=float).reshape(-1) == 1.0)[0]
        F = F_all[nd] if nd.size > 0 else F_all
        self.z_min, self.z_nad = update_ideal_nadir(F, self.z_min, self.z_nad)

        self.extreme_points = find_extreme_points(F, self.z_min, self.z_nad)
        # Temporal fallback: reuse the last valid normal vector when the
        # extreme-point system is singular (degenerate front), instead of
        # resetting the geometry to the uniform vector.
        self.H = normal_hyperplane(
            self.extreme_points,
            self.z_min,
            H_prev=self.H,
            z_nad=self.z_nad,
        )
        # Smoothed, clamped nadir for stable normalization on scaled fronts.
        self.z_nad = update_nadir_from_extremes(
            F, self.z_min, self.z_nad, self.extreme_points,
            smoothing=self.nadir_smoothing,
        )
        # Intrinsic dimension of the current front and the degeneracy decision.
        # ``assess_front_degeneracy`` combines the PCA dimension with a
        # dimensional-reduction floor and a spectral-gap test, so a regular
        # high-M front whose converged cloud merely looks low-rank is not
        # misflagged.  This drives the central design decision: regular fronts
        # use simplex repair, confirmed-degenerate fronts use manifold generation.
        M = int(F.shape[1])
        self.front_dim, degenerate_now = assess_front_degeneracy(
            F, self.z_min, self.z_nad,
            reduction_min=self.manifold_reduction_min,
            gap_min=self.manifold_gap_min,
            use_twonn=self.use_twonn,
        )
        # A degeneracy decision from a single generation is noisy: in early
        # search the cloud is diffuse.  We require the flag to persist for
        # ``manifold_stable_gens`` consecutive generations before switching to
        # manifold generation, and decay the streak otherwise.  This conservative
        # rule is deliberate: it engages the manifold path only once the front has
        # convincingly collapsed onto a low-dimensional structure, and naturally
        # disengages at very high M (M=15), where the sparse objective-space cloud
        # never produces a stable degeneracy signal and the robust simplex path is
        # the safer choice.  The trigger is the measured signal, never the problem
        # identity.
        if degenerate_now:
            self._low_dim_streak += 1
        else:
            self._low_dim_streak = 0
        manifold_candidate = (
            self.manifold_vectors
            and self._low_dim_streak >= self.manifold_stable_gens
        )
        self._reference_updates += 1
        self._manifold_last_fraction = 0.0

        if self.ref_dirs_uniform is not None:
            manifold_accepted = False
            if manifold_candidate:
                # Degenerate front confirmed: generate reference vectors directly
                # inside the estimated d*-dimensional manifold.  This is the core
                # contribution; it cannot be reproduced by simplex repair.  The
                # piecewise variant fits one local subspace per connected
                # component (disconnected or curved manifolds); with a single
                # connected component it reduces to the global generator.
                if self.piecewise_manifold:
                    W_cand = generate_manifold_vectors_clustered(
                        F, self.z_min, self.z_nad, len(self.ref_dirs_uniform),
                    )
                else:
                    W_cand = generate_manifold_vectors(
                        F, self.z_min, self.z_nad,
                        self.front_dim, len(self.ref_dirs_uniform),
                    )
                # The manifold generator works on the [z_min, z_nad]-normalized
                # cloud, so its directions live in normalized coordinates,
                # whereas every downstream consumer (sine_distance for niching,
                # association, and this validation) expects raw-space anchors
                # and re-applies the normalization.  Lift the candidate back to
                # raw space so the standard transformation recovers exactly the
                # designed normalized-space geometry, instead of applying the
                # normalization twice.
                denom_nr = np.maximum(
                    np.asarray(self.z_nad, dtype=float)
                    - np.asarray(self.z_min, dtype=float),
                    1e-12,
                )
                if W_cand.ndim == 2 and W_cand.shape[1] == denom_nr.shape[0]:
                    W_cand = (
                        np.asarray(self.z_min, dtype=float)[None, :]
                        + W_cand * denom_nr[None, :]
                    )
                # Per-direction tournament (union competition).  Once the H3
                # cloud-coverage safety gate below accepts the candidate set,
                # no second whole-set quality switch is used.  Following the
                # two-set principle of RVEA* and the union strategy of
                # DEA-GNG, both direction sets instead compete in a common
                # pool for the fixed budget of K directions: directions that
                # are *active* (some non-dominated solution within the angular
                # band tau_a) are preferred, and spread is preserved by
                # farthest-first selection in normalized direction space.  On
                # a degenerate front most simplex directions are inactive and
                # the manifold grid fills the budget; on a regular front both
                # sets are active and a mixed, well-spread subset survives; a
                # pathological manifold fit contributes inactive directions
                # that are simply not selected.  The decision is per
                # direction and threshold-free within the accepted union.
                W_arv = self._simplex_reference_vectors(F)
                if W_cand.ndim == 2 and W_cand.shape[1] == W_arv.shape[1]:
                    # Coverage-based self-validation (H3): a genuine low-d* front
                    # is covered by the generated manifold at least as well as by
                    # the simplex set, whereas a misfit produced on a *regular*
                    # front — the large-scale LSMOP regime, where a converged
                    # (M-1)-simplex clusters and is misread by PCA as d*=1-2 with
                    # a spurious spectral gap — leaves most of the cloud
                    # uncovered.  Accept the manifold only when its cloud coverage
                    # stays within manifold_accept_ratio of the simplex set's
                    # coverage; otherwise keep the robust simplex path.  This is
                    # the gate the paper attributes to the trigger's safety and
                    # was previously declared but not wired into acceptance.
                    cov_arv = self._coverage_fraction(F, W_arv)
                    cov_cand = self._coverage_fraction(F, W_cand)
                    covers_enough = (
                        cov_arv <= 1e-12
                        or cov_cand >= self.manifold_accept_ratio * cov_arv
                    )
                    if covers_enough:
                        selected, frac_manifold = self._select_union_directions(
                            F, W_arv, W_cand, len(self.ref_dirs_uniform),
                        )
                        self.ref_dirs_repaired = selected
                        manifold_accepted = frac_manifold > 0.0
                        self._manifold_last_fraction = frac_manifold
                    else:
                        self.ref_dirs_repaired = W_arv
                        manifold_accepted = False
                        self._manifold_last_fraction = 0.0
                else:
                    self.ref_dirs_repaired = W_arv
            elif self.use_arv:
                self.ref_dirs_repaired = self._simplex_reference_vectors(F)
            else:
                # B1 (-Manifold) with ARV also off: use the uniform vectors.
                self.ref_dirs_repaired = np.asarray(self.ref_dirs_uniform, dtype=float).copy()

            if manifold_accepted:
                self._manifold_ever_engaged = True
                self._manifold_active_gens += 1
            # Consistency guard: the direction-to-variable partition indexes
            # directions by position, so it becomes stale (and unsafe) if the
            # number of reference vectors ever changes between refreshes.
            if (
                self.dir_to_vars is not None
                and len(self.dir_to_vars) != len(self.ref_dirs_repaired)
            ):
                self.dir_to_vars = None
                self.var_to_dir = None
                self._partition_age = self.partition_period  # force refresh
            if self.diversity_archive is not None:
                self.diversity_archive.set_reference_vectors(self.ref_dirs_repaired)

            self._manifold_fraction_sum += self._manifold_last_fraction
            self._manifold_mean_fraction = (
                self._manifold_fraction_sum / max(self._reference_updates, 1)
            )

        if self.convergence_archive is not None:
            self.convergence_archive.extreme_points = self.extreme_points

    def _select_union_directions(
        self,
        F: np.ndarray,
        W_arv: np.ndarray,
        W_cand: np.ndarray,
        k: int,
        band: float = 0.2,
    ) -> tuple[np.ndarray, float]:
        """Per-direction tournament over the union of ARV and manifold sets.

        Returns exactly ``k`` directions and the fraction of survivors that
        came from the manifold candidate.  Active directions (nearest
        non-dominated solution within the sine band) are selected first;
        spread is enforced by farthest-first traversal over the normalized
        unit directions; inactive directions fill any remaining budget, also
        farthest-first, so the selection degrades gracefully when few
        directions are active.
        """
        pool = np.vstack([np.asarray(W_arv, dtype=float), np.asarray(W_cand, dtype=float)])
        origin = np.concatenate([
            np.zeros(len(W_arv), dtype=bool),
            np.ones(len(W_cand), dtype=bool),
        ])
        k = int(min(max(1, k), len(pool)))

        dist = sine_distance(F, pool, self.z_min, self.z_nad)   # (N, P)
        nearest = dist.min(axis=0)                              # (P,)
        active = nearest <= float(band)

        # Unit directions in normalized space for the spread criterion.
        z = np.asarray(self.z_min, dtype=float).reshape(1, -1)
        zn = np.asarray(self.z_nad, dtype=float).reshape(1, -1)
        U = (pool - z) / np.maximum(zn - z, 1e-12)
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)

        def _fps(cand_idx: np.ndarray, chosen: list, need: int) -> None:
            """Greedy farthest-first over U[cand_idx], appending to chosen."""
            cand = list(int(i) for i in cand_idx if int(i) not in chosen)
            while need > 0 and cand:
                if chosen:
                    ref = U[np.asarray(chosen, dtype=int)]
                    d = np.linalg.norm(
                        U[np.asarray(cand, dtype=int)][:, None, :] - ref[None, :, :],
                        axis=2,
                    ).min(axis=1)
                else:
                    # Seed with the most active direction.
                    d = -nearest[np.asarray(cand, dtype=int)]
                pick = int(np.argmax(d))
                chosen.append(cand.pop(pick))
                need -= 1

        chosen: list = []
        _fps(np.where(active)[0], chosen, k - len(chosen))
        if len(chosen) < k:
            _fps(np.where(~active)[0], chosen, k - len(chosen))

        idx = np.asarray(chosen[:k], dtype=int)
        frac_manifold = float(np.mean(origin[idx])) if idx.size else 0.0
        return pool[idx], frac_manifold

    def _coverage_fraction(self, F: np.ndarray, W: np.ndarray, band: float = 0.2) -> float:
        """Fraction of non-dominated solutions within the angular band of W.

        Used by the manifold candidate self-validation: a direction set that
        leaves a large share of the current cloud uncovered would starve those
        regions in niching, so the better-covering geometry is kept.
        """
        F = np.asarray(F, dtype=float)
        W = np.asarray(W, dtype=float)
        if F.ndim != 2 or W.ndim != 2 or len(F) == 0 or len(W) == 0:
            return 0.0
        dist = sine_distance(F, W, self.z_min, self.z_nad)
        return float(np.mean(dist.min(axis=1) <= float(band)))

    def _simplex_reference_vectors(self, F: np.ndarray) -> np.ndarray:
        """Repaired simplex reference vectors (ARV + adaptive resampling).

        This is the regular-front path and the fallback used both when the front
        is not degenerate and when the manifold candidate fails self-validation.
        """
        W = repair_reference_vectors(self.ref_dirs_uniform, self.H, self.z_min)
        if self.adapt_references:
            W = adapt_reference_vectors(W, F, self.z_min, self.z_nad)
        return W

    # ------------------------------------------------------------------
    # Archives
    # ------------------------------------------------------------------

    def _update_archives(self, pop: Population) -> None:
        """Update archive sizes, then update each archive content."""
        if len(pop) == 0 or self.H is None:
            return

        F = np.asarray(pop.get("F"), dtype=float)
        ratio = fe_ratio(self)
        if self.dynamic_allocation:
            sizes = self.allocator.allocate(F, self.H, self.z_min, ratio)
        else:
            # A4 (-DynAlloc): fixed N/3 budgets.
            third = max(1, self.pop_size // 3)
            sizes = {
                "exploration": third,
                "diversity": third,
                "convergence": max(1, self.pop_size - 2 * third),
            }

        self.exploration_archive.capacity = sizes["exploration"]
        self.diversity_archive.capacity = sizes["diversity"]
        self.convergence_archive.capacity = sizes["convergence"]

        self.exploration_archive.update(pop)
        self.diversity_archive.update(pop, self.z_min, self.z_nad)
        self.convergence_archive.update(pop, self.H, self.z_min)

    # ------------------------------------------------------------------
    # Environmental selection
    # ------------------------------------------------------------------

    def _environmental_selection(self, pop: Population) -> Population:
        """Select pop_size survivors from the merged population.

        Non-dominated sorting is applied first.  The last front is completed
        by niche-counting over the repaired reference vectors, preferring
        under-represented directions.
        """
        if len(pop) <= self.pop_size:
            return pop

        F = np.asarray(pop.get("F"), dtype=float)
        front_no, max_f_no = NDSort(F, self.pop_size)
        front_no = np.asarray(front_no, dtype=float).reshape(-1)

        next_mask = front_no < float(max_f_no)
        last = np.where(front_no == float(max_f_no))[0]

        remaining = int(self.pop_size - np.sum(next_mask))
        if remaining > 0 and last.size > 0:
            chosen_last = self._reference_niching(F[next_mask], F[last], remaining)
            next_mask[last[chosen_last]] = True

        next_mask = self._apply_convergence_reserve(F, next_mask, np.asarray(pop.get("X"), dtype=float))
        return pop[next_mask]

    def _convergence_reserve_indices(self, F: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        """Return elite indices reserved by learned convergence-target evidence."""
        if not self.convergence_targets or self.var_targets is None:
            return np.array([], dtype=int)
        targets = np.asarray(self.var_targets, dtype=float)
        if targets.size == 0 or np.mean(np.isfinite(targets)) < 0.5:
            return np.array([], dtype=int)
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or len(F) == 0:
            return np.array([], dtype=int)
        reserve = max(1, int(0.35 * self.pop_size))
        reserve = min(reserve, len(F), self.pop_size - 1)
        if reserve <= 0:
            return np.array([], dtype=int)
        score = np.linalg.norm(self._normalized_objectives(F), axis=1)
        known = np.isfinite(targets)
        free = np.where(~known)[0]

        X_arr = None if X is None else np.asarray(X, dtype=float)
        if X_arr is not None and X_arr.ndim == 2 and X_arr.shape[0] == len(F) and X_arr.shape[1] == targets.shape[0]:
            target_delta = np.zeros(len(F), dtype=float)
            if known.any():
                target_delta = np.linalg.norm(X_arr[:, known] - targets[known][None, :], axis=1)
            score = target_delta

        order = np.argsort(score)
        if X_arr is None or free.size == 0 or reserve == 1:
            return order[:reserve].astype(int)

        pool_size = min(len(order), max(reserve, reserve * 4))
        pool = order[:pool_size]
        xl = np.asarray(self.problem.xl, dtype=float).reshape(-1) if self.problem is not None else np.zeros(targets.shape[0])
        xu = np.asarray(self.problem.xu, dtype=float).reshape(-1) if self.problem is not None else np.ones(targets.shape[0])
        denom = np.maximum(xu[free] - xl[free], 1e-12)
        X_free = (X_arr[pool][:, free] - xl[free][None, :]) / denom[None, :]

        chosen_local = [int(np.argmin(X_free[:, 0]))]
        while len(chosen_local) < reserve and len(chosen_local) < len(pool):
            chosen_points = X_free[chosen_local]
            dist = np.linalg.norm(X_free[:, None, :] - chosen_points[None, :, :], axis=2)
            min_dist = np.min(dist, axis=1)
            min_dist[chosen_local] = -np.inf
            chosen_local.append(int(np.argmax(min_dist)))
        return pool[chosen_local].astype(int)

    def _apply_convergence_reserve(
        self,
        F: np.ndarray,
        selected_mask: np.ndarray,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Force a convergence elite into survivor selection when targets exist."""
        reserve = self._convergence_reserve_indices(F, X)
        if reserve.size == 0:
            return selected_mask

        selected = set(int(i) for i in np.where(selected_mask)[0])
        reserved = set(int(i) for i in reserve)
        selected.update(reserved)

        score = np.linalg.norm(self._normalized_objectives(F), axis=1)
        while len(selected) > self.pop_size:
            removable = [idx for idx in selected if idx not in reserved]
            if not removable:
                removable = list(selected)
            worst = max(removable, key=lambda idx: score[idx])
            selected.remove(worst)

        if len(selected) < self.pop_size:
            remaining = [idx for idx in np.argsort(score) if int(idx) not in selected]
            for idx in remaining[: self.pop_size - len(selected)]:
                selected.add(int(idx))

        mask = np.zeros(len(F), dtype=bool)
        mask[list(selected)] = True
        return mask

    def _scalarizing_values(self, F: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Tchebycheff-style convergence score for each solution/reference pair."""
        F = np.asarray(F, dtype=float)
        W = np.asarray(W, dtype=float)
        Fn = self._normalized_objectives(F)
        Wn = np.maximum(W, 1e-12)
        Wn = Wn / np.maximum(np.sum(Wn, axis=1, keepdims=True), 1e-12)
        return np.max(Fn[:, None, :] / Wn[None, :, :], axis=2)

    def _normalized_objectives(self, F: np.ndarray) -> np.ndarray:
        """Normalize objective vectors with the current ideal/nadir frame."""
        F = np.asarray(F, dtype=float)
        z = self.z_min if self.z_min is not None else np.min(F, axis=0)
        zn = self.z_nad if self.z_nad is not None else np.max(F, axis=0)
        z = np.asarray(z, dtype=float).reshape(1, -1)
        zn = np.asarray(zn, dtype=float).reshape(1, -1)
        return np.maximum((F - z) / np.maximum(zn - z, 1e-12), 0.0)

    def _reference_niching(
        self,
        F_selected: np.ndarray,
        F_last: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Pick k solutions from the last front using repaired reference vectors."""
        F_selected = np.asarray(F_selected, dtype=float)
        F_last = np.asarray(F_last, dtype=float)
        k = int(max(0, min(k, len(F_last))))
        if k == 0:
            return np.array([], dtype=int)

        if self.ref_dirs_repaired is None or len(self.ref_dirs_repaired) == 0:
            return np.arange(k)

        W = self.ref_dirs_repaired
        dist = sine_distance(
            np.vstack([F_selected, F_last]),
            W,
            self.z_min,
            self.z_nad,
        )
        n_selected = len(F_selected)
        association = np.argmin(dist, axis=1)

        rho = np.bincount(association[:n_selected], minlength=len(W)).astype(int)
        choose = np.zeros(len(F_last), dtype=bool)
        scalar = self._scalarizing_values(F_last, W)
        global_convergence = np.linalg.norm(self._normalized_objectives(F_last), axis=1)

        for _ in range(k):
            active = np.where(~choose)[0]
            if active.size == 0:
                break
            local_refs = association[n_selected + active]
            counts = rho[local_refs]
            local_dist = dist[n_selected + active, local_refs]
            local_scalar = scalar[active, local_refs]
            score = (
                global_convergence[active]
                + 0.05 * np.log1p(local_scalar)
                + 0.05 * local_dist
                + 0.25 * counts
            )
            idx = int(active[np.argmin(score)])
            choose[idx] = True
            rho[association[n_selected + idx]] += 1

        return np.where(choose)[0]

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def _update_stagnation(self) -> None:
        """Track convergence stalls via the ideal-point progress.

        The sum of the ideal point is a problem-agnostic global convergence
        proxy: while it keeps decreasing the search is progressing.  When it
        stops improving for ``stagnation_window`` generations the search is
        likely trapped on a local front (multimodal ``g`` as in DTLZ1/DTLZ3),
        which boosts exploration in ``_de_scale``.
        """
        if self.z_min is None:
            return
        conv = float(np.sum(np.asarray(self.z_min, dtype=float)))
        improved = conv < self._best_convergence - 1e-10 * max(1.0, abs(self._best_convergence))
        if improved:
            self._best_convergence = conv
            self._stall_count = 0
        else:
            self._stall_count += 1

    def _de_scale(self) -> float:
        """Effective DE scaling factor, boosted when the search stalls."""
        if self._stall_count >= self.stagnation_window:
            return float(min(1.0, self.de_f * 1.5))
        return float(self.de_f)

    def _operator_de(
        self,
        pop: Population,
        idx_base: np.ndarray,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> Population:
        n = len(pop)
        if n == 0:
            return Population.empty()

        X = np.asarray(pop.get("X"), dtype=float)
        off = OperatorDE(
            self.problem,
            X[np.asarray(idx_base, dtype=int)],
            X[np.asarray(idx1, dtype=int)],
            X[np.asarray(idx2, dtype=int)],
            Parameter=[self.de_cr, self._de_scale(), 1, 1],
            rng=self.random_state,
        )
        off = np.asarray(off, dtype=float)
        off = self._direction_targeted(pop, idx_base, off)
        off = np.clip(off, self.problem.xl, self.problem.xu)
        return Population.new("X", off)

    def _convergence_repaired_X(self, X: np.ndarray) -> np.ndarray:
        """Return decision vectors with learned convergence targets applied."""
        X = np.asarray(X, dtype=float)
        if self.var_targets is None or self.convergence_repair_rate <= 0.0:
            return X.copy()
        targets = np.asarray(self.var_targets, dtype=float)
        if targets.shape[0] != X.shape[1]:
            return X.copy()
        known = np.isfinite(targets)
        repaired = X.copy()
        if known.any():
            rate = self.convergence_repair_rate
            repaired[:, known] = (
                (1.0 - rate) * repaired[:, known]
                + rate * targets[known][None, :]
            )
        return repaired

    def _convergence_repair_population(self, pop: Population) -> Population:
        """Evaluate repaired copies of promising candidates when targets exist."""
        if not self.convergence_targets or self.var_targets is None or len(pop) == 0:
            return Population.empty()
        targets = np.asarray(self.var_targets, dtype=float)
        known = np.isfinite(targets)
        if np.mean(known) < 0.5:
            return Population.empty()

        X = np.asarray(pop.get("X"), dtype=float)
        F = np.asarray(pop.get("F"), dtype=float)
        if X.ndim != 2 or X.shape[1] != targets.shape[0]:
            return Population.empty()

        n_repair = min(len(X), self.pop_size)
        if len(X) > n_repair and F.ndim == 2 and len(F) == len(X):
            score = np.linalg.norm(self._normalized_objectives(F), axis=1)
            idx = np.argsort(score)[:n_repair]
            X = X[idx]
        elif len(X) > n_repair:
            X = X[:n_repair]

        repaired = self._convergence_repaired_X(X)
        repaired = np.clip(repaired, self.problem.xl, self.problem.xu)
        if np.allclose(repaired, X, rtol=0.0, atol=1e-14):
            return Population.empty()
        candidates = Population.new("X", repaired)
        candidates = self._truncate_to_eval_budget(candidates)
        if len(candidates) == 0:
            return Population.empty()
        self.evaluator.eval(self.problem, candidates)
        return candidates

    def _convergence_manifold_X(self, base_X: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample the free coordinates of a learned convergence manifold."""
        base_X = np.asarray(base_X, dtype=float)
        n_samples = int(max(1, n_samples))
        if self.var_targets is None or base_X.ndim != 2 or base_X.size == 0:
            return np.empty((0, base_X.shape[1] if base_X.ndim == 2 else 0), dtype=float)
        targets = np.asarray(self.var_targets, dtype=float)
        if targets.shape[0] != base_X.shape[1]:
            return np.empty((0, base_X.shape[1]), dtype=float)
        known = np.isfinite(targets)
        free = np.where(~known)[0]
        if not known.any() or free.size == 0:
            return np.empty((0, base_X.shape[1]), dtype=float)

        X = np.repeat(base_X[:1], n_samples, axis=0)
        X[:, known] = targets[known][None, :]

        xl = np.asarray(self.problem.xl, dtype=float).reshape(-1)
        xu = np.asarray(self.problem.xu, dtype=float).reshape(-1)
        if free.size == 1:
            X[:, free[0]] = np.linspace(xl[free[0]], xu[free[0]], n_samples)
        else:
            rng = self.random_state if self.random_state is not None else np.random.default_rng(0)
            grid = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
            for pos, j in enumerate(free):
                values = grid.copy()
                rng.shuffle(values)
                X[:, j] = xl[j] + values * (xu[j] - xl[j])
        return np.clip(X, xl, xu)

    def _convergence_manifold_population(self, pop: Population) -> Population:
        """Evaluate samples on the learned convergence manifold."""
        if not self.convergence_targets or self.var_targets is None or len(pop) == 0:
            return Population.empty()
        targets = np.asarray(self.var_targets, dtype=float)
        known = np.isfinite(targets)
        if np.mean(known) < 0.5:
            return Population.empty()

        X = np.asarray(pop.get("X"), dtype=float)
        F = np.asarray(pop.get("F"), dtype=float)
        if X.ndim != 2 or F.ndim != 2 or len(X) == 0 or len(F) != len(X):
            return Population.empty()
        score = np.linalg.norm(self._normalized_objectives(F), axis=1)
        base = X[[int(np.argmin(score))]]
        manifold = self._convergence_manifold_X(base, n_samples=self.pop_size)
        if len(manifold) == 0:
            return Population.empty()
        candidates = Population.new("X", manifold)
        candidates = self._truncate_to_eval_budget(candidates)
        if len(candidates) == 0:
            return Population.empty()
        self.evaluator.eval(self.problem, candidates)
        return candidates

    def _operator_ga(self, pop: Population, idx: np.ndarray) -> Population:
        n = len(pop)
        if n == 0:
            return Population.empty()

        X = np.asarray(pop.get("X"), dtype=float)
        off = OperatorGA(
            self.problem,
            X[np.asarray(idx, dtype=int)],
            Parameter=[1, 20, 1, 1],
            rng=self.random_state,
        )
        off = np.asarray(off, dtype=float)
        off = self._direction_targeted(pop, idx, off)
        off = np.clip(off, self.problem.xl, self.problem.xu)
        return Population.new("X", off)

    # ------------------------------------------------------------------
    # Geometry-Coupled Search: partition refresh + direction-targeted variation
    # ------------------------------------------------------------------

    def _refresh_linkage(self, pop: Population) -> Population:
        """Periodically re-learn variable interaction groups (linkage learning).

        Budgeted differential grouping around the current elite.  The learned
        groups constrain direction-targeted masking so that interacting
        variables are always perturbed together, addressing the nonseparable
        regime (WFG transformations, LSMOP3/5+) where independent per-variable
        masking is misleading.  All probe evaluations are charged to the FE
        budget and merged back into the search population.
        """
        if (
            not self.linkage_learning
            or self.linkage_pair_budget <= 0
            or self.problem is None
            or int(self.problem.n_var) < 100
            or len(pop) == 0
        ):
            return Population.empty()

        self._linkage_age += 1
        if self._linkage_age < self.linkage_refresh_period:
            return Population.empty()

        # 1 base + <=2*pairs singles + pairs doubles, conservatively bounded.
        cost_bound = 1 + 3 * self.linkage_pair_budget
        if self._remaining_eval_budget() < cost_bound:
            return Population.empty()

        F = np.asarray(pop.get("F"), dtype=float)
        X = np.asarray(pop.get("X"), dtype=float)
        elite = int(np.argmin(np.linalg.norm(self._normalized_objectives(F), axis=1)))

        groups, var_group_of, extra = detect_variable_interactions(
            self.problem,
            X[elite],
            self.random_state,
            evaluator=self.evaluator,
            pair_budget=self.linkage_pair_budget,
        )
        self.var_groups = groups
        self.var_group_of = var_group_of
        self._linkage_age = 0
        return extra

    def _adopt_dva_partition(self) -> None:
        """Install the disjoint direction groups learned by large-scale DVA."""
        if self.dva_manager is None or self.dva_manager.dir_to_vars is None:
            return

        var_to_dir = np.asarray(self.dva_manager.var_to_dir, dtype=int).copy()
        # Linkage closure preserves disjointness by assigning an entire detected
        # interaction component to the direction already owning most members.
        for linkage_group in self.var_groups:
            owners = var_to_dir[linkage_group]
            owners = owners[owners >= 0]
            if owners.size > 0:
                winner = int(np.argmax(np.bincount(owners)))
                var_to_dir[linkage_group] = winner

        if self.var_targets is not None and len(var_to_dir) == len(self.var_targets):
            convergence_vars = np.isfinite(np.asarray(self.var_targets, dtype=float))
            var_to_dir[convergence_vars] = -1

        self.dir_to_vars = [
            np.where(var_to_dir == d)[0].astype(int)
            for d in range(len(self.ref_dirs_repaired))
        ]
        self.var_to_dir = var_to_dir
        self._partition_age = 0

    def _linkage_closure(self, keep: np.ndarray) -> np.ndarray:
        """Expand a variable mask so interacting variables move together."""
        if not self.var_groups:
            return keep
        for g in self.var_groups:
            if keep[g].any():
                keep[g] = True
        return keep

    def _refresh_partition(self, pop: Population) -> Population:
        """Periodically recompute the direction-partitioned variable assignment."""
        self._partition_age += 1
        stale = (self.dir_to_vars is None) or (self._partition_age >= self.partition_period)
        if (not stale) or self.ref_dirs_repaired is None or len(pop) == 0:
            return Population.empty()

        F = np.asarray(pop.get("F"), dtype=float)
        X = np.asarray(pop.get("X"), dtype=float)
        z = self.z_min if self.z_min is not None else F.min(axis=0)
        score = np.linalg.norm(self._normalized_objectives(F), axis=1)
        base_indices = np.argsort(score)[: min(self.partition_base_count, len(X))]

        var_to_dir_votes: list[np.ndarray] = []
        target_votes: list[np.ndarray] = []
        extras = Population.empty()
        for base_idx in base_indices:
            remaining = self._remaining_eval_budget()
            # Each coordinate sweep evaluates five points.  Keep the complete
            # call inside the shared FE budget.
            max_probe = min(self.partition_max_probe, max(0, remaining // 5))
            if max_probe <= 0:
                break
            _dir_to_vars, var_to_dir, var_targets, extra = directional_variable_analysis(
                self.problem,
                X[int(base_idx)],
                self.ref_dirs_repaired,
                z,
                self.random_state,
                max_probe=max_probe,
                evaluator=self.evaluator,
                z_nad=self.z_nad,
            )
            var_to_dir_votes.append(np.asarray(var_to_dir, dtype=int))
            target_votes.append(np.asarray(var_targets, dtype=float))
            if len(extra) > 0:
                extras = Population.merge(extras, extra)

        if not var_to_dir_votes:
            return Population.empty()

        self.var_to_dir = var_to_dir_votes[0].copy()
        for votes in var_to_dir_votes[1:]:
            update = (self.var_to_dir < 0) & (votes >= 0)
            self.var_to_dir[update] = votes[update]

        for linkage_group in self.var_groups:
            owners = self.var_to_dir[linkage_group]
            owners = owners[owners >= 0]
            if owners.size > 0:
                winner = int(np.argmax(np.bincount(owners)))
                self.var_to_dir[linkage_group] = winner
        if target_votes:
            stacked = np.vstack(target_votes)
            new_targets = np.full(stacked.shape[1], np.nan, dtype=float)
            for j in range(stacked.shape[1]):
                finite = stacked[:, j][np.isfinite(stacked[:, j])]
                if finite.size > 0:
                    new_targets[j] = float(np.median(finite))
            self.var_targets = self._merge_var_targets(new_targets)
            if self.consensus_targets:
                self.var_targets = self._population_consensus_targets(X, F, self.var_targets)
        if self.var_targets is not None and len(self.var_targets) == len(self.var_to_dir):
            self.var_to_dir[np.isfinite(self.var_targets)] = -1
        self.dir_to_vars = [
            np.where(self.var_to_dir == d)[0].astype(int)
            for d in range(len(self.ref_dirs_repaired))
        ]
        self._partition_age = 0
        return extras

    def _merge_var_targets(self, new_targets: np.ndarray) -> np.ndarray:
        """Merge newly learned targets without erasing prior inconclusive targets."""
        new_targets = np.asarray(new_targets, dtype=float).reshape(-1)
        if self.var_targets is None:
            return new_targets.copy()

        previous = np.asarray(self.var_targets, dtype=float).reshape(-1)
        if previous.shape != new_targets.shape:
            return new_targets.copy()

        merged = previous.copy()
        known_new = np.isfinite(new_targets)
        merged[known_new] = new_targets[known_new]
        return merged

    def _population_consensus_targets(
        self,
        X: np.ndarray,
        F: np.ndarray,
        current_targets: np.ndarray,
    ) -> np.ndarray:
        """Fill missing coordinate targets when diverse representatives agree."""
        targets = np.asarray(current_targets, dtype=float).reshape(-1).copy()
        if self.ref_dirs_repaired is None or self.problem is None:
            return targets
        if not np.isfinite(targets).any():
            return targets

        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float)
        if X.ndim != 2 or F.ndim != 2 or len(X) != len(F) or X.shape[1] != targets.size:
            return targets

        W = np.asarray(self.ref_dirs_repaired, dtype=float)
        if len(W) == 0:
            return targets

        dist = sine_distance(F, W, self.z_min, self.z_nad)
        scalar = self._scalarizing_values(F, W)
        representatives = []
        for d in range(len(W)):
            assigned = np.where(np.argmin(dist, axis=1) == d)[0]
            if assigned.size == 0:
                continue
            representatives.append(int(assigned[np.argmin(scalar[assigned, d])]))
        if not representatives:
            return targets

        representatives = np.asarray(sorted(set(representatives)), dtype=int)
        min_representatives = min(len(X), max(2, int(np.ceil(0.20 * len(W)))))
        if representatives.size < min_representatives:
            return targets

        X_rep = X[representatives]
        xl = np.asarray(self.problem.xl, dtype=float).reshape(-1)
        xu = np.asarray(self.problem.xu, dtype=float).reshape(-1)
        denom = np.maximum(xu - xl, 1e-12)
        Xn = (X_rep - xl[None, :]) / denom[None, :]

        q10 = np.quantile(Xn, 0.10, axis=0)
        q90 = np.quantile(Xn, 0.90, axis=0)
        spread = q90 - q10
        median = np.median(X_rep, axis=0)

        missing = ~np.isfinite(targets)
        stable = spread <= 0.08
        eligible = missing & stable
        if self.var_to_dir is not None:
            var_to_dir = np.asarray(self.var_to_dir, dtype=int).reshape(-1)
            if var_to_dir.shape == targets.shape:
                eligible &= var_to_dir < 0
        targets[eligible] = median[eligible]
        return targets

    def _direction_targeted(
        self,
        pop: Population,
        idx_base: np.ndarray,
        off: np.ndarray,
    ) -> np.ndarray:
        """Keep offspring perturbations on the variables owned by each parent's
        nearest repaired direction; revert other variables to the base parent.

        This realizes the objective<->variable geometric coupling at reproduction
        time. When the target direction owns no variables (e.g. small-scale
        problems), variation stays global (unchanged), so the mechanism degrades
        gracefully and the ablation (``couple_reproduction=False``) is exact.
        """
        masked = off.copy()
        # Direction-targeted masking has its own exact ablation switch
        # (``couple_reproduction``); the convergence-target repair below is
        # gated ONLY by ``convergence_targets`` (ablation B3), so the two
        # ablations stay independent as stated in the article.
        apply_direction_mask = (
            self.couple_reproduction
            and self._has_reliable_direction_partition()
        )

        idx_base = np.asarray(idx_base, dtype=int)
        Xb = np.asarray(pop.get("X"), dtype=float)[idx_base]
        if Xb.shape[0] != off.shape[0]:
            return off

        if (
            apply_direction_mask
            and self.ref_dirs_repaired is not None
            and self.z_min is not None
            and self.z_nad is not None
        ):
            Fb = np.asarray(pop.get("F"), dtype=float)[idx_base]
            if Fb.shape[0] != off.shape[0]:
                return off
            dist = sine_distance(Fb, self.ref_dirs_repaired, self.z_min, self.z_nad)
            target = np.argmin(dist, axis=1)

            # Convergence/global variables (direction == -1) are always perturbed;
            # position variables are perturbed only for the parent's target direction.
            global_mask = np.zeros(off.shape[1], dtype=bool)
            if self.var_to_dir is not None:
                global_mask[np.where(self.var_to_dir < 0)[0]] = True

            # Soft coupling: instead of hard-reverting non-owned variables to
            # the base parent (which freezes useful variables when the partition
            # is noisy, e.g. on degenerate/non-separable fronts), blend them
            # toward the base.  The coupling strength grows with search progress
            # (more global exploration early, more direction-targeted late).
            try:
                ratio = float(fe_ratio(self))
            except Exception:
                ratio = 0.5
            strength = float(np.clip(0.5 + 0.5 * ratio, 0.0, 1.0))

            for i in range(off.shape[0]):
                t_i = int(target[i])
                if t_i >= len(self.dir_to_vars):
                    continue  # stale partition for this direction -> global variation
                grp = self.dir_to_vars[t_i]
                keep = global_mask.copy()
                keep[grp] = True
                # Linkage closure: interacting variables are perturbed together
                # (a partial perturbation of a nonseparable group is misleading).
                keep = self._linkage_closure(keep)
                if not keep.any():
                    continue  # nothing structured yet -> global variation
                blended = strength * Xb[i] + (1.0 - strength) * off[i]
                masked[i] = np.where(keep, off[i], blended)
        if (
            self.convergence_targets
            and self.var_targets is not None
            and self.convergence_repair_rate > 0.0
        ):
            targets = np.asarray(self.var_targets, dtype=float)
            known = np.isfinite(targets)
            if known.any() and targets.shape[0] == masked.shape[1]:
                rate = self.convergence_repair_rate
                masked[:, known] = (
                    (1.0 - rate) * masked[:, known]
                    + rate * targets[known][None, :]
                )
        return masked

    def _has_reliable_direction_partition(self) -> bool:
        """Return whether direction masking has enough support to be useful."""
        if self.dir_to_vars is None or len(self.dir_to_vars) == 0:
            return False
        nonempty = sum(1 for group in self.dir_to_vars if len(group) > 0)
        min_nonempty = max(2, int(np.ceil(0.10 * len(self.dir_to_vars))))
        return nonempty >= min_nonempty

    # ------------------------------------------------------------------
    # Decomposition-based warm-up
    # ------------------------------------------------------------------

    def _decomposition_warmup(self, pop: Population, n_generations: int) -> Population:
        """Run a short decomposition-based phase with the uniform reference vectors."""
        if len(pop) < 3:
            return pop

        W = self.ref_dirs_uniform
        B = self.neighbor_indices
        if B is None or B.shape[0] != len(pop):
            return pop

        Z = np.asarray(pop.get("F"), dtype=float).min(axis=0)
        current = pop

        for _ in range(n_generations):
            for i in range(len(current)):
                if self._remaining_eval_budget() <= 0:
                    return current
                if self.random_state.random() < 0.9:
                    P = B[i, self.random_state.permutation(B.shape[1])]
                else:
                    P = self.random_state.permutation(len(current))
                if len(P) < 2:
                    return current

                parent_i = current[i]
                parent_j = current[int(P[0])]
                parent_k = current[int(P[1])]

                off = OperatorDE(
                    self.problem,
                    parent_i.get("X").reshape(1, -1),
                    parent_j.get("X").reshape(1, -1),
                    parent_k.get("X").reshape(1, -1),
                    Parameter=[self.de_cr, self.de_f, 1, 1],
                    rng=self.random_state,
                )
                off = np.asarray(off, dtype=float).reshape(1, -1)
                off = np.clip(off, self.problem.xl, self.problem.xu)
                offspring = Population.new("X", off)
                self.evaluator.eval(self.problem, offspring)

                Z = np.minimum(Z, offspring.get("F").reshape(-1))

                g_old = tchebycheff_values(current[P].get("F"), Z, W[P, :])
                g_new = tchebycheff_values(offspring.get("F"), Z, W[P, :])
                replace = np.where(g_old >= g_new)[0]
                idx_array = np.asarray(P, dtype=int)
                current = current.copy()
                current[idx_array[replace]] = offspring[0]

        return current

    # ------------------------------------------------------------------
    # Optimum
    # ------------------------------------------------------------------

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)


ALGORITHMS = {"GCS-MaOEA": GCSMaOEA}
