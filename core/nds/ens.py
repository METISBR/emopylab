"""EmoPyLab Efficient Non-Dominated Sort (ENS with binary search over the fronts)."""

from __future__ import annotations

import numpy as np


def _dominated_by(buf: np.ndarray, p: np.ndarray) -> bool:
    """True when some row of ``buf`` dominates ``p`` (minimisation)."""
    return bool(np.any(np.all(buf <= p, axis=1) & np.any(buf < p, axis=1)))


def efficient_non_dominated_sort(F_matrix: np.ndarray, first_only: bool = False) -> list[np.ndarray]:
    """ENS-BS: solutions are visited in lexicographic order, so a solution can only be dominated by earlier ones and
    the front it belongs to is found by binary search (dominance by a front is monotone in the front index).

    With ``first_only`` just the first front is built (every other solution is compared with it only)."""
    F = np.asarray(F_matrix, dtype=np.float64)
    N, M = F.shape
    if N == 0:
        return []
    order = np.lexsort([F[:, m] for m in reversed(range(M))])
    buf = [np.empty((N, M))]            # per-front row buffers (grown lazily; N is an upper bound on any size)
    cnt = [0]
    idx: list[list[int]] = [[]]
    for oi in order:
        p = F[oi]
        if first_only:
            if cnt[0] == 0 or not _dominated_by(buf[0][: cnt[0]], p):
                buf[0][cnt[0]] = p
                cnt[0] += 1
                idx[0].append(int(oi))
            continue
        lo, hi = 0, len(idx)            # smallest front index not dominating p (hi = new front)
        while lo < hi:
            mid = (lo + hi) // 2
            if cnt[mid] and _dominated_by(buf[mid][: cnt[mid]], p):
                lo = mid + 1
            else:
                hi = mid
        if lo == len(idx):
            buf.append(np.empty((N, M)))
            cnt.append(0)
            idx.append([])
        buf[lo][cnt[lo]] = p
        cnt[lo] += 1
        idx[lo].append(int(oi))
    return [np.array(f, dtype=np.int64) for f in idx if f]
