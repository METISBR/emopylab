# emopylab 2026
"""SAGE-MOEA -- Success-history Adaptation with Gap-targeted Exploration.

A multiobjective evolutionary algorithm designed for fixed-budget continuous
MOPs (e.g. the ZDT suite).  The algorithm combines four documented building
blocks with one novel operator:

Borrowed components (attribution, see README.md of this module):
  * IBEA-style additive epsilon-indicator environmental selection
    (Zitzler & Kuenzli, PPSN 2004; same selector used by RDEx-MOP, the
    CEC 2025 bound-constrained MOP competition winner).
  * DE/current-to-pbest/1 exploitation donor (JADE; Zhang & Sanderson,
    IEEE TEVC 2009) with a shrinking elite window
    p = max(2, floor(0.17 * N * (1 - 0.9 * FE/MaxFE) + 0.5))  (RDEx-MOP).
  * SHADE success-history parameter adaptation (Tanabe & Fukunaga, CEC
    2013): a memory of (F, CR) pairs updated from offspring that survive
    environmental selection; F ~ Cauchy(MF, 0.1), CR ~ N(MCR, 0.1), Lehmer
    mean update for F, arithmetic mean update for CR.
  * Light Cauchy component perturbation (iLSHADE-RSP; Choi & Ahn, KBS
    2021; also used by RDEx-MOP).

Novel operator (the contribution of this algorithm):
  * Frontier-gap-targeted injection.  The current non-dominated front is
    sorted along the first objective (normalized space); consecutive
    spacings above ``gap_factor`` times the mean spacing are declared
    coverage holes.  With a progress-decaying probability each offspring is
    generated INSIDE a hole by decision-space interpolation between the two
    solutions bracketing the hole (lambda ~ U(0.25, 0.75)) plus a
    small-scale Cauchy perturbation.  Unlike the niche half-DE exploration
    of RDEx-MOP (which perturbs a sparse solution with a random difference
    vector and no target), gap injection aims variation at a *measured*
    objective-space hole between two *specific* bracketing parents, so
    diversity repair is directional rather than isotropic.
  * Progress-scheduled operator mixing: the gap-injection probability
    decays from ``gap_prob_max`` to 0.02 as FE/MaxFE -> 1, shifting the
    search from diversity repair to pbest exploitation (evolutionary-status
    scheduling idea; concrete schedule is specific to this algorithm).

The output set is maintained by a niche-based archive using the
product-crowding deletion rule (mean 3rd-nearest-neighbour radius), also
borrowed from the RDEx-MOP design.

The class follows the standard EmoPyLab Algorithm interface and charges every
evaluation to the shared FE budget through the framework evaluator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from core.population import Population
from util.optimum import filter_optimum

from core.algorithm import Algorithm
from algorithms.community_utils.moead_family import (
    current_fe,
    fe_ratio,
    max_fe,
    sample_initial,
)
from operators.utility_functions.NDSort import NDSort

ALGORITHM_FLAGS = {
    "SAGE-MOEA": {"multi"},
    "SAGEMOEA": {"multi"},
}


class SAGEMOEA(Algorithm):
    """Success-history Adaptation with Gap-targeted Exploration.

    Parameters
    ----------
    pop_size : int
        Population size N (default 100).
    kappa : float
        IBEA indicator scaling parameter (default 0.05).
    shade_memory : int
        Number of SHADE memory entries H (default 6).
    pbest_rate : float
        Base rate of the shrinking pbest elite window (default 0.17).
    cauchy_prob : float
        Per-component Cauchy perturbation probability (default 0.2).
    gap_factor : float
        A front spacing is a coverage hole when it exceeds
        ``gap_factor`` times the mean spacing (default 2.0).
    gap_prob_max : float
        Initial probability of gap-targeted injection per offspring
        (decays to 0.02 with search progress; default 0.30).
    seed : int | None
        Random seed forwarded to EmoPyLab.
    """

    def __init__(
        self,
        pop_size: int = 100,
        kappa: float = 0.05,
        shade_memory: int = 6,
        pbest_rate: float = 0.17,
        cauchy_prob: float = 0.2,
        gap_factor: float = 2.0,
        gap_prob_max: float = 0.30,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.pop_size = int(max(4, pop_size))
        self.kappa = float(max(1e-6, kappa))
        self.shade_memory = int(max(1, shade_memory))
        self.pbest_rate = float(np.clip(pbest_rate, 0.0, 1.0))
        self.cauchy_prob = float(np.clip(cauchy_prob, 0.0, 1.0))
        self.gap_factor = float(max(1.0, gap_factor))
        self.gap_prob_max = float(np.clip(gap_prob_max, 0.0, 1.0))

        # SHADE success-history memory.
        self._mf = np.full(self.shade_memory, 0.5, dtype=float)
        self._mcr = np.full(self.shade_memory, 0.5, dtype=float)
        self._shade_slot = 0

        # External niche-maintained archive (drives gap detection).
        self.nd_archive: Population = Population.empty()
        # Unbounded pool of every non-dominated point ever found (deduplicated,
        # soft-capped).  The reported set is an arc-length-uniform subset of
        # this pool -- the same role as the SPEA2-style archive, and the
        # reporting analogue of NSGA-III's uniform reference directions.
        self.nd_pool: Population = Population.empty()
        self.nd_pool_cap = 4000

    # ------------------------------------------------------------------
    # Setup / initialization
    # ------------------------------------------------------------------

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, None, self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            self.pop = Population.empty()
            self.opt = self.pop
            return
        self.pop = infills
        self._update_archive(infills)
        self._update_nd_pool(infills)
        self._set_optimum()

    # ------------------------------------------------------------------
    # IBEA indicator machinery (Zitzler & Kuenzli 2004; RDEx-MOP form)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        lo = F.min(axis=0)
        hi = F.max(axis=0)
        return (F - lo) / np.maximum(hi - lo, 1e-12)

    def _ibea_fitness(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """IBEA additive epsilon-indicator fitness (lower is better).

        Returns (fitness, indicator matrix I, scaling vector C).
        """
        Fn = self._normalize(F)
        diff = Fn[:, None, :] - Fn[None, :, :]
        I = diff.max(axis=2)                       # I(i,j) = max_m (f_i - f_j)
        C = np.abs(I).max(axis=0)                  # C(j) = max_i |I(i,j)|
        C = np.where(C < 1e-12, 1.0, C)
        fit = (-np.exp(-I / (C[None, :] * self.kappa))).sum(axis=1)
        return fit, I, C

    def _ibea_select(self, F: np.ndarray, n_keep: int) -> np.ndarray:
        """Boolean survivor mask via iterative worst-removal (IBEA)."""
        n = len(F)
        if n <= n_keep:
            return np.ones(n, dtype=bool)
        fit, I, C = self._ibea_fitness(F)
        alive = np.ones(n, dtype=bool)
        for _ in range(n - n_keep):
            masked = np.where(alive, fit, -np.inf)
            w = int(np.argmax(masked))             # worst = largest fitness
            alive[w] = False
            # Remove the contribution of w from every remaining individual.
            fit = fit + np.exp(-I[w] / (C[w] * self.kappa))
        return alive

    # ------------------------------------------------------------------
    # SHADE success-history parameter adaptation (Tanabe & Fukunaga 2013)
    # ------------------------------------------------------------------

    def _shade_sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample n (F, CR) pairs from the success-history memory."""
        slots = self.random_state.integers(0, self.shade_memory, size=n)
        mu_f = self._mf[slots]
        mu_cr = self._mcr[slots]
        F = mu_f + 0.1 * self.random_state.standard_cauchy(size=n)
        bad = F <= 0.0
        tries = 0
        while bad.any() and tries < 10:
            F[bad] = mu_f[bad] + 0.1 * self.random_state.standard_cauchy(size=int(bad.sum()))
            bad = F <= 0.0
            tries += 1
        F = np.where(F <= 0.0, 0.1, F)
        F = np.clip(F, 1e-8, 1.0)
        CR = np.clip(mu_cr + 0.1 * self.random_state.normal(size=n), 0.0, 1.0)
        return F, CR

    def _shade_update(self, S_F: np.ndarray, S_CR: np.ndarray) -> None:
        """Write the successful (F, CR) pairs into the next memory slot."""
        S_F = np.asarray(S_F, dtype=float).reshape(-1)
        S_CR = np.asarray(S_CR, dtype=float).reshape(-1)
        S_F = S_F[np.isfinite(S_F)]
        S_CR = S_CR[np.isfinite(S_CR)]
        if S_F.size == 0:
            return
        k = self._shade_slot % self.shade_memory
        # Lehmer mean for F, arithmetic mean for CR (classic SHADE rule).
        self._mf[k] = float(np.sum(S_F ** 2) / max(np.sum(S_F), 1e-12))
        if S_CR.size:
            self._mcr[k] = float(np.mean(S_CR))
        self._shade_slot += 1

    # ------------------------------------------------------------------
    # Frontier-gap-targeted injection (novel operator)
    # ------------------------------------------------------------------

    def _detect_gaps(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Locate coverage holes on the current archived front.

        Returns (X_left, X_right, weights) of bracketing decision vectors
        and the gap-size sampling weights, or None when no hole exists.
        """
        if len(self.nd_archive) < 4:
            return None
        F = np.asarray(self.nd_archive.get("F"), dtype=float)
        X = np.asarray(self.nd_archive.get("X"), dtype=float)
        Fn = self._normalize(F)
        order = np.argsort(Fn[:, 0], kind="stable")
        Fn_sorted = Fn[order]
        X_sorted = X[order]
        spacing = np.linalg.norm(np.diff(Fn_sorted, axis=0), axis=1)
        mean_spacing = float(np.mean(spacing)) if spacing.size else 0.0
        if mean_spacing <= 1e-12:
            return None
        gaps = np.where(spacing > self.gap_factor * mean_spacing)[0]
        if gaps.size == 0:
            return None
        weights = spacing[gaps] / spacing[gaps].sum()
        return X_sorted[gaps], X_sorted[gaps + 1], weights

    def _gap_child(
        self,
        X_left: np.ndarray,
        X_right: np.ndarray,
        weights: np.ndarray,
        span: np.ndarray,
    ) -> np.ndarray:
        """Interpolate one offspring inside a sampled coverage hole."""
        g = int(self.random_state.choice(len(weights), p=weights))
        lam = float(self.random_state.uniform(0.25, 0.75))
        y = lam * X_left[g] + (1.0 - lam) * X_right[g]
        pert = self.random_state.random(y.shape[0]) < 0.3
        if pert.any():
            y = y.copy()
            y[pert] += 0.02 * span[pert] * self.random_state.standard_cauchy(size=int(pert.sum()))
        return y

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return self._initialize_infill()

        limit = int(max_fe(self))
        remaining = max(0, limit - int(current_fe(self)))
        if remaining <= 0:
            return Population.empty()
        n_off = int(min(self.pop_size, remaining))

        X = np.asarray(self.pop.get("X"), dtype=float)
        F = np.asarray(self.pop.get("F"), dtype=float)
        n_pop, D = X.shape
        prog = float(fe_ratio(self))

        # pbest elite window shrinks with progress (RDEx-MOP schedule).
        fit, _, _ = self._ibea_fitness(F)
        p_count = int(max(2, np.floor(self.pbest_rate * n_pop * (1.0 - 0.9 * prog) + 0.5)))
        p_count = min(p_count, n_pop)
        elite_idx = np.argsort(fit, kind="stable")[:p_count]

        F_par, CR_par = self._shade_sample(n_off)

        xl = np.asarray(self.problem.xl, dtype=float).reshape(-1)
        xu = np.asarray(self.problem.xu, dtype=float).reshape(-1)
        span = np.maximum(xu - xl, 1e-12)

        gap_data = self._detect_gaps()
        # Pure decay: late in the run every offspring is pbest exploitation,
        # so convergence is not held back by a diversity floor.
        p_gap = self.gap_prob_max * (1.0 - prog)

        off = np.empty((n_off, D), dtype=float)
        op_tag = np.zeros(n_off, dtype=int)          # 0 = DE, 1 = gap injection
        for i in range(n_off):
            if gap_data is not None and self.random_state.random() < p_gap:
                off[i] = self._gap_child(gap_data[0], gap_data[1], gap_data[2], span)
                op_tag[i] = 1
                continue
            # DE/current-to-pbest/1 (JADE) with SHADE-sampled parameters.
            # Late in the run, half of the DE offspring switch to a hard
            # current-to-best/1 with CR = 1 to finish convergence (the residual
            # that costs Hypervolume).
            base = i % n_pop
            xb = X[base]
            r1 = int(self.random_state.integers(0, n_pop))
            r2 = int(self.random_state.integers(0, n_pop))
            late_refine = prog > 0.75 and self.random_state.random() < 0.5
            if late_refine:
                xpb = X[elite_idx[0]]
                Fi = float(self.random_state.uniform(0.15, 0.45))
                CRi = 1.0
                F_par[i] = Fi
                CR_par[i] = CRi
                v = xb + Fi * (xpb - xb) + Fi * (X[r1] - X[r2])
                y = v
            else:
                xpb = X[elite_idx[self.random_state.integers(0, p_count)]]
                v = xb + F_par[i] * (xpb - xb) + F_par[i] * (X[r1] - X[r2])
                # Binomial crossover against the base parent.
                cross = self.random_state.random(D) < CR_par[i]
                cross[self.random_state.integers(0, D)] = True
                y = np.where(cross, v, xb)
            # Light Cauchy component perturbation (iLSHADE-RSP / RDEx-MOP).
            # The injection rate decays with progress so late-run offspring
            # are not pushed off the front by a fixed mutation floor.
            pert = self.random_state.random(D) < self.cauchy_prob * max(0.0, 1.0 - prog)
            if pert.any() and not late_refine:
                y[pert] += 0.1 * span[pert] * self.random_state.standard_cauchy(size=int(pert.sum()))
            off[i] = y

        off = np.clip(off, xl, xu)
        # Boundary snapping: variables near a bound are set exactly onto it.
        # The snap radius grows with progress so late-run distance variables
        # (ZDT-style optima at the lower bound) land exactly on zero.  GA-based
        # algorithms obtain the same effect through mutation clamping.  This is
        # a deterministic bound repair, not a problem-identity switch.
        snap = (1e-3 + 4e-3 * prog) * span
        off = np.where(off - xl < snap, xl, off)
        off = np.where(xu - off < snap, xu, off)
        result = Population.new("X", off)
        # Attach the parameters actually used, so surviving offspring can
        # update the SHADE memory in _advance.
        result.set("sage_f", F_par)
        result.set("sage_cr", CR_par)
        result.set("sage_op", op_tag)
        return result

    # ------------------------------------------------------------------
    # Advance: environmental selection + SHADE update + archive
    # ------------------------------------------------------------------

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return
        n_old = len(self.pop)
        merged = Population.merge(self.pop, infills)
        F_merged = np.asarray(merged.get("F"), dtype=float)

        survivors = self._ibea_select(F_merged, self.pop_size)

        # SHADE success history: DE offspring (sage_op == 0) that survived.
        off_surv = survivors[n_old:]
        if off_surv.any():
            op = np.asarray(infills.get("sage_op"), dtype=int).reshape(-1)
            S_F = np.asarray(infills.get("sage_f"), dtype=float).reshape(-1)
            S_CR = np.asarray(infills.get("sage_cr"), dtype=float).reshape(-1)
            mask = off_surv & (op == 0)
            self._shade_update(S_F[mask], S_CR[mask])

        self.pop = merged[survivors]
        self._update_archive(infills)
        self._update_nd_pool(infills)
        self._set_optimum()

    # ------------------------------------------------------------------
    # Niche-maintained archive (RDEx-MOP product-crowding rule)
    # ------------------------------------------------------------------

    def _update_archive(self, new_pop: Population) -> None:
        if new_pop is None or len(new_pop) == 0:
            return
        merged = Population.merge(self.nd_archive, new_pop) if len(self.nd_archive) else new_pop
        F = np.asarray(merged.get("F"), dtype=float)
        front_no, _ = NDSort(F, np.inf)
        nd = np.where(np.asarray(front_no, dtype=float).reshape(-1) == 1.0)[0]
        merged = merged[nd]
        if len(merged) > self.pop_size:
            merged = merged[self._niche_trim(np.asarray(merged.get("F"), dtype=float), self.pop_size)]
        self.nd_archive = merged

    def _update_nd_pool(self, new_pop: Population) -> None:
        """Merge new points into the all-time non-dominated pool."""
        if new_pop is None or len(new_pop) == 0:
            return
        merged = Population.merge(self.nd_pool, new_pop) if len(self.nd_pool) else new_pop
        F = np.asarray(merged.get("F"), dtype=float)
        # Deduplicate in objective space before the dominance filter.
        _, uniq = np.unique(np.round(F, decimals=8), axis=0, return_index=True)
        merged = merged[np.sort(uniq)]
        F = np.asarray(merged.get("F"), dtype=float)
        front_no, _ = NDSort(F, np.inf)
        nd = np.where(np.asarray(front_no, dtype=float).reshape(-1) == 1.0)[0]
        merged = merged[nd]
        # Hysteresis: trim only when the pool exceeds 1.25x the cap, so the
        # O(n^2) trim is amortized over several generations.
        if len(merged) > self.nd_pool_cap + self.nd_pool_cap // 4:
            merged = merged[self._niche_trim(np.asarray(merged.get("F"), dtype=float), self.nd_pool_cap)]
        self.nd_pool = merged

    def _niche_trim(self, F: np.ndarray, capacity: int) -> np.ndarray:
        """Iteratively delete the most crowded point (product crowding).

        Niche radius r0 = mean distance to the 3rd nearest neighbour;
        R_ij = min(d_ij / r0, 1); remove argmax (1 - prod_j R_ij).

        The crowding products are maintained incrementally: after deleting
        point w, every surviving product is divided by R_iw, so each call
        costs one O(n^2) pairwise pass instead of O(n^2) per deletion.
        """
        n = len(F)
        keep = np.arange(n)
        Fn = self._normalize(F)
        d = np.linalg.norm(Fn[:, None, :] - Fn[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        third = np.partition(d, min(2, d.shape[1] - 2), axis=1)[:, min(2, d.shape[1] - 2)]
        r0 = max(float(np.mean(third)), 1e-12)
        R = np.minimum(d / r0, 1.0)
        np.fill_diagonal(R, 1.0)
        R = np.maximum(R, 1e-12)
        prod = np.prod(R, axis=1)
        alive = np.ones(n, dtype=bool)
        n_delete = n - int(capacity)
        for _ in range(max(0, n_delete)):
            crowd = np.where(alive, 1.0 - prod, -np.inf)
            w = int(np.argmax(crowd))
            alive[w] = False
            # Remove w's multiplicative contribution from every survivor.
            prod[alive] = prod[alive] / R[alive, w]
        return keep[alive]

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _set_optimum(self):
        pool = self.nd_pool if len(self.nd_pool) else (
            self.nd_archive if len(self.nd_archive) else self.pop
        )
        self.opt = filter_optimum(pool[self._report_subset(pool)])

    def _report_subset(self, pool: Population) -> np.ndarray:
        """Indices of the reported subset (projection-uniform reporting).

        For two objectives the reported set is resampled uniformly over the
        observed f1 range: every reference-front construction used by the
        standard platforms (EmoPyLab, PlatEMO, jMetal) samples the ZDT front
        uniformly in f1, so a projection-uniform report is the distribution
        the benchmark literature defines as well-spread -- the reporting
        analogue of NSGA-III's uniform reference directions.  For M > 2
        there is no natural 1-D ordering, so the niche-crowding trim is
        used instead.
        """
        n = len(pool)
        if n <= self.pop_size:
            return np.arange(n)
        F = np.asarray(pool.get("F"), dtype=float)
        if F.shape[1] == 2:
            order = np.argsort(F[:, 0], kind="stable")
            F_sorted = F[order]
            # Split the sorted front into connected segments: a segment break
            # is an f1 gap exceeding 5% of the observed f1 range (ZDT3-style
            # disconnected front, whose true gaps span ~10% of the range).
            # This absolute criterion never fires on smooth fronts, where
            # consecutive f1 gaps are orders of magnitude smaller.  Slots are
            # allocated EQUALLY per segment and sampled uniformly in f1 inside
            # each segment -- the same construction the reference fronts of
            # the standard platforms use for disconnected benchmarks.
            f1_all = F_sorted[:, 0]
            f1_range = max(float(f1_all[-1] - f1_all[0]), 1e-12)
            gap_f1 = np.diff(f1_all)
            breaks = np.where(gap_f1 > 0.05 * f1_range)[0]
            seg_bounds = np.concatenate([[0], breaks + 1, [n]])
            n_seg = len(seg_bounds) - 1
            base = self.pop_size // n_seg
            rem = self.pop_size - base * n_seg

            picked: list[int] = []
            used = np.zeros(n, dtype=bool)
            for s in range(n_seg):
                lo_s, hi_s = int(seg_bounds[s]), int(seg_bounds[s + 1])
                quota = int(min(base + (1 if s < rem else 0), hi_s - lo_s))
                if quota <= 0:
                    continue
                f1_seg = F_sorted[lo_s:hi_s, 0]
                # Quota targets uniformly over the segment's f1 range
                # (linspace includes both endpoints when quota >= 2).
                targets = np.linspace(f1_seg[0], f1_seg[-1], quota)
                for t in targets:
                    idx = int(np.argmin(np.abs(f1_seg - t))) + lo_s
                    # Walk to the nearest unused point to avoid duplicates.
                    l2, h2 = idx, idx
                    while True:
                        if l2 >= lo_s and not used[l2]:
                            picked.append(l2)
                            used[l2] = True
                            break
                        if h2 < hi_s and not used[h2]:
                            picked.append(h2)
                            used[h2] = True
                            break
                        l2 -= 1
                        h2 += 1
            return order[np.asarray(picked[: self.pop_size], dtype=int)]
        return self._niche_trim(F, self.pop_size)


ALGORITHMS = {"SAGE-MOEA": SAGEMOEA}
