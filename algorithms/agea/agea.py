# emopylab 2026
"""Adaptive Grid-based Evolutionary Algorithm (AGEA).

Reference:
Z. Liu, F. Han, Q.-H. Ling, H. Han, J. Jiang, and Q. Liu,
"A multi-objective evolutionary algorithm based on a grid with adaptive divisions
for multi-objective optimization with irregular Pareto fronts,"
Applied Soft Computing, vol. 176, p. 113106, 2025.
DOI: 10.1016/j.asoc.2025.113106
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from util.optimum import filter_optimum
from algorithms.community_utils.moead_family import sample_initial, rng_from_algo

ALGORITHM_FLAGS = {"AGEA": {"multi", "many", "real"}}


def calc_grid_parameters(
    z_star: np.ndarray,
    g_nad: np.ndarray,
    div: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate grid cell size (eq. 4) and grid lower boundary (eq. 5).

    Args:
        z_star: Ideal point vector of shape (M,).
        g_nad: Grid nadir point vector of shape (M,).
        div: Number of grid divisions along each objective axis.

    Returns:
        Tuple (gs, lb) where gs is grid step size and lb is lower boundary.
    """
    div_eff = max(2, int(div))
    diff = np.maximum(g_nad - z_star, 1e-12)
    gs = diff / float(div_eff - 1)
    gs = np.maximum(gs, 1e-12)
    lb = z_star - 0.5 * gs
    return gs, lb


def calc_grid_indices(G: np.ndarray, lb: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Calculate grid coordinate indices I = floor((G - lb) / gs) (eq. 7).

    Args:
        G: Objective vectors of shape (K, M).
        lb: Grid lower boundary vector of shape (M,).
        gs: Grid step size vector of shape (M,).

    Returns:
        Integer matrix of grid coordinates of shape (K, M).
    """
    return np.floor((G - lb) / gs).astype(int)


def calc_grid_corners(lb: np.ndarray, I: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Calculate grid cell corner points gc = lb + I * gs (eq. 8).

    Args:
        lb: Grid lower boundary vector of shape (M,).
        I: Integer grid coordinate matrix of shape (K, M).
        gs: Grid step size vector of shape (M,).

    Returns:
        Corner points matrix of shape (K, M).
    """
    return lb + I.astype(float) * gs


def stabilize_g_nad(
    z_nad: np.ndarray,
    g_nad: Optional[np.ndarray],
    z_star: np.ndarray,
    div: int,
) -> np.ndarray:
    """Stabilize grid nadir point using non-dominated set (Algorithm 2).

    The grid nadir coordinates move along each objective axis only when
    the displacement exceeds half of the current grid cell size.

    Args:
        z_nad: Empirical nadir point computed from non-dominated solutions.
        g_nad: Current grid nadir point, or None if uninitialized.
        z_star: Current ideal point.
        div: Number of grid divisions.

    Returns:
        Updated grid nadir point vector of shape (M,).
    """
    div_eff = max(2, int(div))
    if g_nad is None:
        return np.maximum(z_nad.copy(), z_star + 1e-6)

    g_new = g_nad.copy()
    diff = np.maximum(g_new - z_star, 1e-12)
    gs = diff / float(div_eff - 1)
    threshold = 0.5 * gs

    shift = np.abs(z_nad - g_new)
    mask = shift > threshold
    g_new[mask] = z_nad[mask]
    return np.maximum(g_new, z_star + 1e-6)


def calc_environmental_fitness(
    G: np.ndarray,
    gc: np.ndarray,
    I: np.ndarray,
    z_star: np.ndarray,
    g_nad: np.ndarray,
    fn: np.ndarray,
    delta: float = 1e6,
) -> np.ndarray:
    """Compute normalized coordinates and fitness (eq. 9-11).

    Normalized objective vectors and cell corner vectors are evaluated with
    a boundary penalty delta = 1e6 applied when I_ij == 0.

    Args:
        G: Candidate objective matrix of shape (K, M).
        gc: Grid cell corner matrix of shape (K, M).
        I: Grid index matrix of shape (K, M).
        z_star: Ideal point vector of shape (M,).
        g_nad: Grid nadir point vector of shape (M,).
        fn: Pareto non-domination rank vector of shape (K,).
        delta: Boundary penalty multiplier for boundary coordinates.

    Returns:
        Fitness vector of shape (K,).
    """
    diff = np.maximum(g_nad - z_star, 1e-12)
    G_norm = (G - z_star) / diff
    gc_norm = (gc - z_star) / diff

    diff_norm = G_norm - gc_norm
    dist = np.sqrt(np.sum(diff_norm ** 2, axis=1))

    boundary_penalty = float(delta) * np.sum(I == 0, axis=1).astype(float)
    rank_penalty = (fn - 1.0) * 1e12

    return rank_penalty + boundary_penalty + dist


def environmental_selection(
    F: np.ndarray,
    fn: np.ndarray,
    z_star: np.ndarray,
    g_nad: np.ndarray,
    div: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute Environmental Selection (Algorithm 3).

    Partitions candidate solutions into grid subspaces, computes fitness (eq. 11)
    with boundary penalty delta = 1e6 when I_ij == 0, and retains one survivor
    per occupied subspace having minimum fitness.

    Args:
        F: Candidate objective matrix of shape (K, M).
        fn: Pareto non-domination rank vector of shape (K,).
        z_star: Ideal point vector of shape (M,).
        g_nad: Grid nadir point vector of shape (M,).
        div: Number of grid divisions.

    Returns:
        Tuple (survivor_indices, fitness_array, grid_indices).
    """
    gs, lb = calc_grid_parameters(z_star, g_nad, div)
    I = calc_grid_indices(F, lb, gs)
    gc = calc_grid_corners(lb, I, gs)

    fitness = calc_environmental_fitness(F, gc, I, z_star, g_nad, fn)

    subspace_dict: dict[tuple[int, ...], int] = {}
    for idx, idx_tuple in enumerate(map(tuple, I)):
        if idx_tuple not in subspace_dict:
            subspace_dict[idx_tuple] = idx
        else:
            prev_idx = subspace_dict[idx_tuple]
            if fitness[idx] < fitness[prev_idx]:
                subspace_dict[idx_tuple] = idx

    survivors = np.array(list(subspace_dict.values()), dtype=int)
    return survivors, fitness, I


def population_reselection(
    candidates: np.ndarray,
    I_candidates: np.ndarray,
    fitness_candidates: np.ndarray,
    n_select: int,
    reference_I: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Reselect population solutions using crowding degree and fitness (eq. 14).

    Computes crowding degree per subspace-neighborhood based on grid coordinate
    Manhattan distance and sorts by composite fitness (eq. 14).

    Args:
        candidates: Array of integer candidate indices.
        I_candidates: Grid coordinate matrix of candidates of shape (K, M).
        fitness_candidates: Fitness values of candidates of shape (K,).
        n_select: Number of solutions to sample.
        reference_I: Optional reference grid coordinates for crowding computation.
        rng: Optional pseudo-random number generator.

    Returns:
        Array of selected candidate indices of length min(n_select, len(candidates)).
    """
    n_cand = len(candidates)
    if n_cand <= n_select:
        return candidates

    ref = reference_I if reference_I is not None and len(reference_I) > 0 else I_candidates

    diff = np.abs(I_candidates[:, None, :] - ref[None, :, :])
    dist_grid = np.sum(diff, axis=2)

    if reference_I is None or len(reference_I) == 0:
        crowding = np.sum((dist_grid <= 1) & (dist_grid > 0), axis=1).astype(float)
    else:
        crowding = np.sum(dist_grid <= 1, axis=1).astype(float)

    max_fit = np.max(np.abs(fitness_candidates)) if len(fitness_candidates) > 0 else 1.0
    norm_fit = fitness_candidates / (max_fit + 1.0)
    score = crowding + norm_fit

    order = np.argsort(score)
    return candidates[order[:n_select]]


def survivor_indices_clean(survivors: np.ndarray) -> np.ndarray:
    """Return flat integer array of survivor indices."""
    return np.asarray(survivors, dtype=int).reshape(-1)


class AGEA(Algorithm):
    """Adaptive Grid-based Evolutionary Algorithm for multi- and many-objective optimization.

    This solver adapts grid divisions dynamically across generations to balance
    convergence and diversity on regular and irregular Pareto fronts (Liu et al., 2025).
    """

    def __init__(
        self,
        pop_size: int = 100,
        div: int = 10,
        sampling: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.pop_size = int(max(2, pop_size))
        self.div_init = int(max(2, div))
        self.div = self.div_init
        self.sampling = sampling

        self.z_star: Optional[np.ndarray] = None
        self.g_nad: Optional[np.ndarray] = None
        setattr(self, "z*", None)

        self.front_no: Optional[np.ndarray] = None
        self.fitness_vals: Optional[np.ndarray] = None
        self.crowding_score: Optional[np.ndarray] = None

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        super()._setup(problem, **kwargs)
        self.div = self.div_init

    def _initialize_infill(self) -> Population:
        rng = rng_from_algo(self)
        return sample_initial(self.problem, self.pop_size, self.sampling, rng)

    def _initialize_advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        self.pop = infills if infills is not None else Population.empty()
        if len(self.pop) == 0:
            self.opt = self.pop
            return

        F = np.asarray(self.pop.get("F"), dtype=float)
        self.z_star = np.min(F, axis=0)
        setattr(self, "z*", self.z_star)

        fn, _ = NDSort(F, self.pop_size)
        self.front_no = np.asarray(fn, dtype=float).reshape(-1)
        nd_mask = self.front_no == 1.0
        F_nd = F[nd_mask] if np.any(nd_mask) else F
        z_nad = np.max(F_nd, axis=0)

        self.g_nad = stabilize_g_nad(z_nad, None, self.z_star, self.div)

        self._run_survival(self.pop)
        self._set_optimum()

    def _infill(self) -> Population:
        if self.pop is None or len(self.pop) == 0:
            rng = rng_from_algo(self)
            return sample_initial(self.problem, self.pop_size, self.sampling, rng)

        rng = rng_from_algo(self)
        n = self.pop_size

        rank = self.front_no if self.front_no is not None else np.zeros(len(self.pop))
        score = self.fitness_vals if self.fitness_vals is not None else np.zeros(len(self.pop))

        n_needed = 2 * ((n + 1) // 2)
        idx = np.asarray(
            TournamentSelection(2, n_needed, rank, score, rng=rng),
            dtype=int,
        ) - 1
        idx = np.clip(idx, 0, len(self.pop) - 1)

        X = np.asarray(self.pop.get("X"), dtype=float)
        off_x = OperatorGA(self.problem, X[idx], rng=rng)
        off_x = np.asarray(off_x, dtype=float)[:n]

        return Population.new("X", off_x)

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return

        off_F = np.asarray(infills.get("F"), dtype=float)
        self.z_star = np.minimum(
            self.z_star if self.z_star is not None else np.min(off_F, axis=0),
            np.min(off_F, axis=0),
        )
        setattr(self, "z*", self.z_star)

        merged = Population.merge(self.pop, infills)
        self._run_survival(merged)
        self._set_optimum()

    def _run_survival(self, pop: Population) -> None:
        F = np.asarray(pop.get("F"), dtype=float)
        n_candidates = len(F)
        target_n = self.pop_size

        if n_candidates <= target_n:
            self.pop = pop
            self._update_diagnostics(pop)
            return

        self.z_star = np.minimum(
            self.z_star if self.z_star is not None else np.min(F, axis=0),
            np.min(F, axis=0),
        )
        setattr(self, "z*", self.z_star)

        fn, _ = NDSort(F, n_candidates)
        fn = np.asarray(fn, dtype=float).reshape(-1)
        nd_mask = fn == 1.0
        F_nd = F[nd_mask] if np.any(nd_mask) else F
        z_nad = np.max(F_nd, axis=0)

        # Stabilization Alg. 2
        self.g_nad = stabilize_g_nad(z_nad, self.g_nad, self.z_star, self.div)

        # Adaptive division Alg. 4 & Environmental selection Alg. 3
        selected_indices = self._adaptive_division_loop(F, fn, target_n)

        self.pop = pop[selected_indices]
        self._update_diagnostics(self.pop)

    def _adaptive_division_loop(
        self,
        F: np.ndarray,
        fn: np.ndarray,
        target_n: int,
    ) -> np.ndarray:
        rng = rng_from_algo(self)
        self.div = self.div_init

        while True:
            survivors, fitness, I = environmental_selection(
                F, fn, self.z_star, self.g_nad, self.div,
            )
            n_survivors = len(survivors)

            if n_survivors > target_n and self.div > 2:
                self.div -= 1
                continue
            break

        if n_survivors == target_n:
            return survivors

        if n_survivors > target_n:
            chosen = population_reselection(
                survivors,
                I[survivors],
                fitness[survivors],
                target_n,
                rng=rng,
            )
            return chosen

        rem_needed = target_n - n_survivors
        all_idx = np.arange(len(F))
        cand_mask = ~np.isin(all_idx, survivors)
        candidates = all_idx[cand_mask]

        additional = population_reselection(
            candidates,
            I[candidates],
            fitness[candidates],
            rem_needed,
            reference_I=I[survivors],
            rng=rng,
        )
        return np.concatenate([survivor_indices_clean(survivors), additional])

    def _update_diagnostics(self, pop: Population) -> None:
        if pop is None or len(pop) == 0:
            return
        F = np.asarray(pop.get("F"), dtype=float)
        fn, _ = NDSort(F, len(F))
        self.front_no = np.asarray(fn, dtype=float).reshape(-1)

        gs, lb = calc_grid_parameters(self.z_star, self.g_nad, self.div)
        I = calc_grid_indices(F, lb, gs)
        gc = calc_grid_corners(lb, I, gs)
        self.fitness_vals = calc_environmental_fitness(
            F, gc, I, self.z_star, self.g_nad, self.front_no,
        )

        diff = np.abs(I[:, None, :] - I[None, :, :])
        dist_grid = np.sum(diff, axis=2)
        self.crowding_score = np.sum((dist_grid <= 1) & (dist_grid > 0), axis=1).astype(float)

        self.data["div"] = self.div
        self.data["g_nad"] = self.g_nad
        self.data["z_star"] = self.z_star

    def _set_optimum(self) -> None:
        self.opt = filter_optimum(self.pop, least_infeasible=True)
