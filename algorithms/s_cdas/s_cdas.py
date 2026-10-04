# emopylab 2026
"""S-CDAS (self-controlling dominance area of solutions).

Reference:
H. Sato, H. E. Aguirre, and K. Tanaka. Self-controlling dominance area of solutions in evolutionary
many-objective optimization. Proceedings of the Asia-Pacific Conference on Simulated Evolution and
Learning, 2010, 455-465.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'SCDAS': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


def _scdas_sort(F):
    """Fractional rank in (0, 1) from the S-CDAS relaxed dominance (angle-widened Pareto cones)."""
    N, M = F.shape
    F = F - F.min(axis=0)
    L = np.diag(F.max(axis=0) + 1e-2)
    rx = np.sqrt(np.sum(F ** 2, axis=1))
    with np.errstate(all="ignore"):
        norm_f = np.maximum(rx, 0.0)
        cos = (F @ np.eye(M)) / (norm_f[:, None] * 1.0)  # cosine with the unit axes
        wx = np.arccos(np.clip(cos, -1.0, 1.0))
        lx = pdist2(F, L)
        phi = np.arcsin(np.clip(rx[:, None] * np.sin(wx) / lx, -1.0, 1.0))          # (N, M): phi_i
        # F1[i, j, :] = rx_j * sin(wx_j + phi_i) / sin(phi_i)
        F1 = rx[None, :, None] * np.sin(wx[None, :, :] + phi[:, None, :]) / np.sin(phi[:, None, :])
        F1_self = F1[np.arange(N), np.arange(N), :]                                      # F1_i for point i
        dominated = np.all(F1_self[:, None, :] < F1, axis=2)                             # i dominates j
    rank = 1 + dominated.sum(axis=0)
    return rank / (rank.max() + 1.0)


def _environmental_selection(pop, N):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    front_no = front_no.astype(float)
    for i in range(1, int(max_f) + 1):
        sel = front_no == i
        front_no[sel] = front_no[sel] + _scdas_sort(F[sel])
    _, front_no = np.unique(front_no, return_inverse=True)
    front_no = front_no + 1.0
    counts = np.cumsum(np.bincount(front_no.astype(int), minlength=int(front_no.max()) + 1)[1:])
    max_f = int(np.argmax(counts >= N)) + 1
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], cd[nxt]


class SCDAS(LoopAlgorithm):
    def start(self):
        self.pop, self.front_no, self.crowd = _environmental_selection(self.pop, self.N)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop, self.front_no, self.crowd = _environmental_selection(Population.merge(self.pop, off), self.N)
