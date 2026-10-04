# emopylab 2026
"""EMyO-C (evolutionary many-objective optimization algorithm with clustering-based).

Reference:
R. Denysiuk, L. Costa, and I. E. Santo. Clustering-based selection for evolutionary many-objective
optimization. Proceedings of the International Conference on Parallel Problem Solving from Nature,
2014, 538-547.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs
from algorithms.myo_demr.myo_demr import MyODEMR
from core.population import Population

ALGORITHM_FLAGS = {'EMyOC': {'integer', 'many', 'multi', 'real'}}


def _truncation(F, Z, remaining):
    """Average-linkage clustering of the simplex-projected objectives into ``remaining`` clusters; the member
    closest to the ideal point represents each cluster."""
    n = len(F)
    dist = np.linalg.norm(F - Z, axis=1)
    if remaining >= n:
        return np.arange(n)
    with np.errstate(all="ignore"):
        P = np.nan_to_num((F - Z) / np.sum(F - Z, axis=1, keepdims=True))
    labels = fcluster(linkage(P, method="average"), t=remaining, criterion="maxclust")
    idx = [int(np.where(labels == c)[0][np.argmin(dist[labels == c])]) for c in np.unique(labels)]
    if len(idx) < remaining:                       # ties collapsed clusters: complete by closest remaining points
        rest = [i for i in np.argsort(dist, kind="stable") if i not in set(idx)]
        idx += rest[: remaining - len(idx)]
    return np.asarray(idx[:remaining], dtype=int)


def _environmental_selection(pop, N, Z):
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    nxt[last[_truncation(F[last], Z, N - int(nxt.sum()))]] = True
    return pop[nxt]


class EMyOC(LoopAlgorithm):
    _operator = MyODEMR._operator

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.CR, self.proM, self.disM = 0.15, 1.0, 20.0

    def start(self):
        self.Z = objs(self.pop).min(axis=0)

    def step(self):
        N, rng = self.N, self.rng
        off = self._operator(decs(self.pop), decs(self.pop[rng.integers(0, N, size=N)]), decs(self.pop[rng.integers(0, N, size=N)]))
        self.Z = np.minimum(self.Z, objs(off).min(axis=0))
        self.pop = _environmental_selection(Population.merge(self.pop, off), N, self.Z)
