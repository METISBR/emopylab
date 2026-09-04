# emopylab 2026
"""Dynamic three-archive engine for GCSMaOEA.

This module implements the third core contribution of GCSMaOEA:
**Dynamic Multi-Archive Framework**.  GCSMaOEA maintains three cooperating
archives:

    * Exploration Archive (EA)   - preserves non-dominated novelty,
    * Diversity Archive (DA)     - covers the repaired reference vectors,
    * Convergence Archive (CA)   - pushes solutions toward the estimated front.

Rather than giving each archive a fixed budget of ``N/3`` solutions,
GCSMaOEA dynamically reallocates the budgets based on online indicators of
spread and convergence.  The DiversityArchive also uses the *repaired*
reference vectors from ARV, so niching follows the true geometry of the
front rather than a pre-defined simplex.

For the article this maps to Section 3.3: the proposed dynamic multi-archive
framework and adaptive budget allocation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from core.population import Population

from operators.utility_functions.NDSort import NDSort

from .reference_repair import (
    perpendicular_distance,
    sine_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_dominated_front(pop: Population) -> np.ndarray:
    """Return indices of the first non-dominated front."""
    if len(pop) == 0:
        return np.array([], dtype=int)
    F = np.asarray(pop.get("F"), dtype=float)
    front_no, _ = NDSort(F, np.inf)
    return np.where(np.asarray(front_no, dtype=int) == 1)[0]


def _farthest_first_fill(
    F: np.ndarray,
    chosen: np.ndarray,
    remaining: np.ndarray,
    need: int,
) -> np.ndarray:
    """Greedily pick ``need`` solutions from ``remaining`` maximizing spread.

    Farthest-first traversal: each pick is the remaining solution whose nearest
    distance to the already-selected set is largest.  Used to keep the
    diversity archive at full capacity when occupied niches are scarce.
    Returns only the newly added indices.
    """
    F = np.asarray(F, dtype=float)
    chosen = np.asarray(chosen, dtype=int)
    rem = np.asarray(remaining, dtype=int)
    if need <= 0 or rem.size == 0:
        return np.asarray([], dtype=int)

    cand = F[rem]  # (R, M)

    # Maintain the distance from every remaining candidate to the nearest
    # already-selected point and update it incrementally after each pick.  This
    # is the standard farthest-first traversal in O((|chosen| + need) * R * M)
    # instead of recomputing the full pairwise matrix every iteration
    # (previously near-cubic in the population size).
    if chosen.size > 0:
        ref = F[chosen]  # (C, M)
        mind = np.min(
            np.linalg.norm(cand[:, None, :] - ref[None, :, :], axis=2), axis=1
        )
    else:
        mind = np.full(rem.shape[0], np.inf, dtype=float)

    added: list[int] = []
    n_take = int(min(need, rem.shape[0]))
    for _ in range(n_take):
        j = int(np.argmax(mind))
        added.append(int(rem[j]))
        # Update nearest-selected distances with the newly added point.
        newd = np.linalg.norm(cand - cand[j][None, :], axis=1)
        mind = np.minimum(mind, newd)
        mind[j] = -np.inf  # never pick the same candidate again

    return np.asarray(added, dtype=int)


def _truncation(F: np.ndarray, k_delete: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Remove ``k_delete`` solutions by nearest-neighbour (crowding) truncation.

    Iteratively removes the most crowded solution -- the one with the smallest
    distance to its nearest surviving neighbour -- which is the standard
    distance-based truncation used in many MOEAs.

    The previous implementation re-sorted the full pairwise submatrix on every
    deletion (``O(k * n^2 * log(n))`` with a large constant), and profiling showed
    it dominated the whole run (>90% of wall time).  This version computes the
    pairwise matrix once and maintains each point's nearest-neighbour distance
    incrementally, recomputing only the points whose nearest neighbour was the
    just-removed solution: ``O(n^2)`` once plus near-linear updates per deletion.
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    if k_delete <= 0 or n == 0:
        return np.arange(n)
    if k_delete >= n:
        return np.empty(0, dtype=int)

    D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)

    alive = np.ones(n, dtype=bool)
    nn = D.min(axis=1)          # distance to nearest neighbour
    arg = D.argmin(axis=1)      # index of that nearest neighbour

    for _ in range(int(k_delete)):
        # Most crowded survivor = smallest nearest-neighbour distance.
        masked = np.where(alive, nn, np.inf)
        j = int(np.argmin(masked))
        alive[j] = False
        nn[j] = np.inf

        # Remove j from the working matrix.
        D[:, j] = np.inf
        # Only survivors whose nearest neighbour *was* j need recomputation.
        affected = np.where(alive & (arg == j))[0]
        if affected.size:
            sub = D[affected]
            nn[affected] = sub.min(axis=1)
            arg[affected] = sub.argmin(axis=1)

    return np.where(alive)[0]


# ---------------------------------------------------------------------------
# Exploration Archive
# ---------------------------------------------------------------------------

class ExplorationArchive:
    """Archive that stores non-dominated solutions to promote exploration.

    Every new non-dominated solution is kept, and truncation is applied only
    when the archive exceeds its budget.
    """

    def __init__(self, capacity: int, rng: Optional[np.random.Generator] = None):
        self.capacity = int(max(1, capacity))
        self.rng = rng
        self.archive: Population = Population.empty()

    def update(self, pop: Population) -> Population:
        """Merge new population, keep non-dominated solutions, truncate."""
        if len(pop) == 0:
            return self.archive

        merged = Population.merge(self.archive, pop)
        nd_idx = _non_dominated_front(merged)
        self.archive = merged[nd_idx]

        capacity = int(max(0, self.capacity))
        if capacity == 0:
            self.archive = Population.empty()
        elif len(self.archive) > capacity:
            rng = self.rng if self.rng is not None else np.random.default_rng(0)
            keep = rng.choice(len(self.archive), size=capacity, replace=False)
            self.archive = self.archive[np.asarray(keep, dtype=int)]

        return self.archive


# ---------------------------------------------------------------------------
# Diversity Archive
# ---------------------------------------------------------------------------

class DiversityArchive:
    """Archive that maintains diversity along repaired reference vectors.

    For each repaired reference vector, the archive selects the solution with
    the smallest sine-based angular distance.  This angular niching is
    executed over *repaired* vectors that reflect the true geometry of the
    current Pareto front.
    """

    def __init__(
        self,
        capacity: int,
        W_repaired: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.capacity = int(max(1, capacity))
        self.W_repaired = W_repaired
        self.rng = rng
        self.archive: Population = Population.empty()

    def set_reference_vectors(self, W_repaired: np.ndarray) -> None:
        """Update the repaired reference vectors used for niching."""
        self.W_repaired = np.asarray(W_repaired, dtype=float)

    def update(
        self,
        pop: Population,
        z_min: np.ndarray,
        z_nad: np.ndarray,
    ) -> Population:
        """Select diverse niche representatives, never collapsing the archive.

        The previous design picked one solution per reference vector
        (``argmin`` over solutions) and then deduplicated.  On irregular,
        degenerate or clustered fronts many vectors point at the *same*
        solution, so deduplication shrinks the archive far below its capacity
        and diversity is lost.

        This bilateral version associates each solution with its nearest
        reference vector, keeps the best representative of every *occupied*
        niche, and -- crucially -- when fewer niches are occupied than the
        capacity allows, fills the remaining slots by farthest-first spread
        over the leftover solutions.  The archive therefore stays at full
        capacity whenever enough solutions exist.
        """
        if len(pop) == 0 or self.W_repaired is None or len(self.W_repaired) == 0:
            return self.archive

        nd_idx = _non_dominated_front(pop)
        candidates = pop[nd_idx]
        F = np.asarray(candidates.get("F"), dtype=float)
        N = len(F)
        cap = int(max(0, self.capacity))
        if cap == 0 or N == 0:
            self.archive = Population.empty()
            return self.archive
        dist = sine_distance(F, self.W_repaired, z_min, z_nad)  # (N, K)

        # Each solution joins its nearest reference vector; keep the closest
        # solution of every occupied niche as that niche's representative.
        sol_to_vec = np.argmin(dist, axis=1)
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for k in np.unique(sol_to_vec):
            members = np.where(sol_to_vec == k)[0]
            best = int(members[np.argmin(dist[members, k])])
            if best not in chosen_set:
                chosen.append(best)
                chosen_set.add(best)

        chosen_arr = np.asarray(chosen, dtype=int)

        if len(chosen_arr) > cap:
            keep = _truncation(F[chosen_arr], len(chosen_arr) - cap, self.rng)
            chosen_arr = chosen_arr[keep]
        elif len(chosen_arr) < cap:
            remaining = np.asarray(
                [i for i in range(N) if i not in chosen_set], dtype=int
            )
            need = int(min(cap - len(chosen_arr), len(remaining)))
            if need > 0:
                added = _farthest_first_fill(F, chosen_arr, remaining, need)
                chosen_arr = np.concatenate([chosen_arr, added])

        self.archive = candidates[chosen_arr]
        return self.archive


# ---------------------------------------------------------------------------
# Convergence Archive
# ---------------------------------------------------------------------------

class ConvergenceArchive:
    """Archive that selects solutions closest to the estimated Pareto front.

    The perpendicular distance to the normal hyperplane is used as the
    convergence indicator.  The M ASF extreme points are always preserved
    to avoid losing boundary information.
    """

    def __init__(self, capacity: int, rng: Optional[np.random.Generator] = None):
        self.capacity = int(max(1, capacity))
        self.rng = rng
        self.archive: Population = Population.empty()
        self.extreme_points: Optional[np.ndarray] = None

    def update(
        self,
        pop: Population,
        H: np.ndarray,
        z_min: np.ndarray,
    ) -> Population:
        """Keep the closest solutions to the normal hyperplane plus extremes."""
        if len(pop) == 0:
            return self.archive

        if int(self.capacity) <= 0:
            self.archive = Population.empty()
            return self.archive

        F = np.asarray(pop.get("F"), dtype=float)
        dist = perpendicular_distance(F, H, z_min)

        # Reserve slots for extreme points if available.
        n_extreme = 0
        extreme_indices = []
        if self.extreme_points is not None and len(self.extreme_points) > 0:
            # Find the nearest population solution to each stored extreme.
            for ep in self.extreme_points:
                d = np.linalg.norm(F - ep, axis=1)
                idx = int(np.argmin(d))
                if idx not in extreme_indices:
                    extreme_indices.append(idx)
            n_extreme = len(extreme_indices)

        remaining = int(self.capacity - n_extreme)
        if remaining <= 0:
            self.archive = pop[extreme_indices[: self.capacity]]
            return self.archive

        # Exclude extreme indices from the ranking.
        mask = np.ones(len(pop), dtype=bool)
        mask[extreme_indices] = False
        candidates = np.where(mask)[0]
        order = np.argsort(dist[candidates])
        selected = np.concatenate([extreme_indices, candidates[order[:remaining]]])

        self.archive = pop[selected]
        return self.archive


# ---------------------------------------------------------------------------
# Dynamic allocator
# ---------------------------------------------------------------------------

class DynamicArchiveAllocator:
    """Adaptively allocate population budget among the three archives.

    Indicators used:
        - spread indicator : mean of all pairwise angles around the ideal point;
          its running-normalized value enters Eq. (23).
        - convergence indicator : mean absolute hyperplane residual;
          its running-normalized value enters Eq. (24).
        - progress : fraction of consumed function evaluations.

    The allocator returns the target size for each archive, guaranteeing that
    the total equals the user-defined population size ``N``.
    """

    def __init__(self, pop_size: int):
        self.pop_size = int(max(3, pop_size))
        self.sizes = {
            "exploration": max(1, pop_size // 3),
            "diversity": max(1, pop_size // 3),
            "convergence": max(1, pop_size // 3),
        }
        self._spread_min = float("inf")
        self._spread_max = float("-inf")
        self._convergence_min = float("inf")
        self._convergence_max = float("-inf")
        self.last_spread = 0.0
        self.last_convergence = 0.0
        self.last_spread_normalized = 0.0
        self.last_convergence_normalized = 1.0

    def _running_normalize(
        self,
        value: float,
        min_attr: str,
        max_attr: str,
        initial: float,
    ) -> float:
        """Normalize a state indicator by its running observed range."""
        lo = min(float(getattr(self, min_attr)), float(value))
        hi = max(float(getattr(self, max_attr)), float(value))
        setattr(self, min_attr, lo)
        setattr(self, max_attr, hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
            return float(initial)
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def allocate(
        self,
        archive_F: np.ndarray,
        H: np.ndarray,
        z_min: np.ndarray,
        fe_ratio: float,
    ) -> dict[str, int]:
        """Compute archive sizes for the next generation.

        Parameters
        ----------
        archive_F : np.ndarray, shape (N, M)
            Objective vectors of the merged parent+offspring population.
        H : np.ndarray, shape (M,)
            Normal hyperplane vector.
        z_min : np.ndarray, shape (M,)
            Ideal point.
        fe_ratio : float
            Fraction of budget consumed, in [0, 1].

        Returns
        -------
        sizes : dict[str, int]
            Sizes for exploration, diversity, and convergence archives.
        """
        if len(archive_F) == 0:
            return self.sizes.copy()

        F = np.asarray(archive_F, dtype=float)
        H = np.asarray(H, dtype=float).reshape(-1)
        z_min = np.asarray(z_min, dtype=float).reshape(-1)

        shifted = F - z_min

        # Eq. (21): unscaled mean absolute hyperplane residual.
        residual = np.abs(1.0 - shifted @ H)
        conv_score = float(np.mean(residual))

        # Eqs. (19)-(20): mean over all unordered pairwise angles.
        norms = np.linalg.norm(shifted, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        unit = shifted / norms
        cosine = np.clip(unit @ unit.T, -1.0, 1.0)
        angle = np.arccos(cosine)
        if len(F) >= 2:
            upper = angle[np.triu_indices(len(F), k=1)]
            spread_score = float(np.mean(upper))
        else:
            spread_score = 0.0

        spread_norm = self._running_normalize(
            spread_score, "_spread_min", "_spread_max", initial=0.0
        )
        conv_norm = self._running_normalize(
            conv_score,
            "_convergence_min",
            "_convergence_max",
            initial=1.0,
        )
        self.last_spread = spread_score
        self.last_convergence = conv_score
        self.last_spread_normalized = spread_norm
        self.last_convergence_normalized = conv_norm

        # Eqs. (22)-(24) of the article.
        progress = float(np.clip(fe_ratio, 0.0, 1.0))
        w_explore = 0.5 * (1.0 - progress)
        w_diversity = 0.4 + 0.6 * spread_norm
        w_converge = 0.5 + 0.5 * conv_norm + 0.2 * progress

        total = w_explore + w_converge + w_diversity
        base = {
            "exploration": w_explore / total,
            "convergence": w_converge / total,
            "diversity": w_diversity / total,
        }

        # Scale the three values to exactly N after rounding.  Largest-remainder
        # rounding preserves the fixed total without altering the equations.
        keys = ("exploration", "convergence", "diversity")
        proportions = np.asarray([base[k] for k in keys], dtype=float)
        proportions = proportions / max(float(np.sum(proportions)), 1e-12)
        raw = proportions * self.pop_size
        rounded = np.floor(raw).astype(int)
        leftover = int(self.pop_size - np.sum(rounded))
        if leftover > 0:
            remainders = raw - rounded
            for idx in np.argsort(-remainders)[:leftover]:
                rounded[int(idx)] += 1
        # All three archives remain live.  This only affects boundary rounding
        # (not Eqs. (22)-(24)) and is required by the three-pool reproduction
        # contract, especially for very small smoke-test populations.
        for empty in np.where(rounded == 0)[0]:
            donors = np.where(rounded > 1)[0]
            if donors.size == 0:
                break
            donor = int(donors[np.argmax(rounded[donors])])
            rounded[donor] -= 1
            rounded[int(empty)] += 1
        sizes = {k: int(rounded[i]) for i, k in enumerate(keys)}

        self.sizes = sizes
        return sizes.copy()
