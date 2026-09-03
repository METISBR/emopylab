"""EmoPyLab Non-Dominated Sorting and Pareto Front computation (zero-pymoo standalone)."""

from __future__ import annotations

from typing import Any
import numpy as np

from core.nds.ens import efficient_non_dominated_sort


def fast_non_dominated_sort(F: np.ndarray, G: np.ndarray | None = None) -> list[np.ndarray]:
    """Classic Deb's Fast Non-dominated Sorting algorithm O(M N^2)."""
    F = np.asarray(F, dtype=float)
    N, M = F.shape

    if N == 0:
        return []

    # Handle constraints if provided
    if G is not None:
        G = np.asarray(G, dtype=float)
        if G.ndim == 1:
            G = G[:, None]
        cv = np.sum(np.maximum(0.0, G), axis=1)
        infeasible = cv > 0
        if np.any(infeasible):
            penalty = cv[infeasible, None]
            fmax = np.max(F, axis=0, keepdims=True)
            F = F.copy()
            F[infeasible, :] = fmax + penalty

    domination_count = np.zeros(N, dtype=int)
    dominated_solutions: list[list[int]] = [[] for _ in range(N)]
    fronts: list[list[int]] = [[]]

    for p in range(N):
        for q in range(p + 1, N):
            p_dom_q = False
            q_dom_p = False

            # Check dominance
            fp = F[p]
            fq = F[q]
            p_better = False
            q_better = False

            for m in range(M):
                if fp[m] < fq[m]:
                    p_better = True
                elif fq[m] < fp[m]:
                    q_better = True

            if p_better and not q_better:
                p_dom_q = True
            elif q_better and not p_better:
                q_dom_p = True

            if p_dom_q:
                dominated_solutions[p].append(q)
                domination_count[q] += 1
            elif q_dom_p:
                dominated_solutions[q].append(p)
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return [np.array(f, dtype=np.int64) for f in fronts if len(f) > 0]


def find_non_dominated(F: np.ndarray, G: np.ndarray | None = None) -> np.ndarray:
    """Return a boolean mask or indices of non-dominated solutions in F."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    N, M = F.shape
    if N == 0:
        return np.array([], dtype=bool)

    if G is not None:
        G = np.asarray(G, dtype=float)
        if G.ndim == 1:
            G = G[:, None]
        cv = np.sum(np.maximum(0.0, G), axis=1)
        # Any feasible solution dominates all infeasible solutions
        is_feasible = cv <= 0
        if np.any(is_feasible) and not np.all(is_feasible):
            # Non-dominated among feasible only
            feas_indices = np.where(is_feasible)[0]
            sub_mask = find_non_dominated(F[feas_indices])
            mask = np.zeros(N, dtype=bool)
            mask[feas_indices[sub_mask]] = True
            return mask
        elif not np.any(is_feasible):
            # All infeasible: lowest CV is best
            min_cv = np.min(cv)
            min_cv_indices = np.where(cv == min_cv)[0]
            sub_mask = find_non_dominated(F[min_cv_indices])
            mask = np.zeros(N, dtype=bool)
            mask[min_cv_indices[sub_mask]] = True
            return mask

    # Pure Pareto non-domination
    # Vectorized / pairwise check
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_efficient[i]:
            continue
        # i is dominated if there is any j where F[j] <= F[i] and any F[j] < F[i]
        # Equivalently: all(F <= F[i], axis=1) & any(F < F[i], axis=1)
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominated):
            is_efficient[i] = False
        else:
            # i dominates any j where F[i] <= F[j] and any F[i] < F[j]
            dom_by_i = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
            is_efficient[dom_by_i] = False

    return is_efficient


class NonDominatedSorting:
    """Non-dominated sorting implementation supporting ENS, Fast NDS, etc."""

    def __init__(self, method: str = "efficient_non_dominated_sort", **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

    def do(
        self,
        F: np.ndarray,
        return_rank: bool = False,
        only_non_dominated_front: bool = False,
        n_stop_if_ranked: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], np.ndarray]:
        """Perform non-dominated sorting."""
        F = np.asarray(F, dtype=float)
        if F.ndim == 1:
            F = F[:, None]
        N = F.shape[0]

        if N == 0:
            if return_rank:
                return [], np.array([], dtype=int)
            return []

        if only_non_dominated_front:
            nd_mask = find_non_dominated(F)
            indices = np.where(nd_mask)[0]
            return indices

        # Use ENS or fast NDS
        if self.method in ("efficient_non_dominated_sort", "ens"):
            fronts = efficient_non_dominated_sort(F)
        else:
            fronts = fast_non_dominated_sort(F)
        # Truncate if n_stop_if_ranked is specified
        if n_stop_if_ranked is not None and n_stop_if_ranked < N:
            trimmed_fronts = []
            count = 0
            for f in fronts:
                trimmed_fronts.append(f)
                count += len(f)
                if count >= n_stop_if_ranked:
                    break
            fronts = trimmed_fronts

        if return_rank:
            rank = np.full(N, np.inf, dtype=float)
            for r, front in enumerate(fronts):
                rank[front] = r
            return fronts, rank

        return fronts
