"""EmoPyLab Deductive Efficient Non-Dominated Sort (ENS) Implementation."""

from __future__ import annotations

import numpy as np


def efficient_non_dominated_sort(F_matrix: np.ndarray) -> list[np.ndarray]:
    """Deductive ENS with Sequential Search in O(M N sqrt(N))."""
    F = np.asarray(F_matrix, dtype=np.float64)
    N, M = F.shape

    if N == 0:
        return []

    # 1. Lexicographical sorting by f1, then f2, etc.
    order = np.lexsort([F[:, m] for m in reversed(range(M))])
    F_sorted = F[order]

    fronts: list[list[int]] = [[]]

    for orig_idx, i in zip(order, range(N)):
        p = F_sorted[i]
        allocated = False

        for k, front in enumerate(fronts):
            # Check if any member in front k dominates p
            dominated = False
            for member_idx in front:
                q = F[member_idx]
                # q dominates p? (p cannot dominate q because p >= q in f1)
                if np.all(q <= p) and np.any(q < p):
                    dominated = True
                    break

            if not dominated:
                front.append(orig_idx)
                allocated = True
                break

        if not allocated:
            fronts.append([orig_idx])

    return [np.array(f, dtype=np.int64) for f in fronts]
