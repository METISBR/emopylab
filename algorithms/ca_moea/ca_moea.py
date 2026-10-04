# emopylab 2026
"""CA-MOEA (clustering based adaptive multi-objective evolutionary algorithm).

Reference:
Y. Hua, Y. Jin, and K. Hao. A clustering-based adaptive evolutionary algorithm for multiobjective
optimization with irregular Pareto fronts. IEEE Transactions on Cybernetics, 2019, 49(7): 2758-2770.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'CAMOEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _reference_generation(Fn, M, K, rng):
    ex = np.unique(np.concatenate([np.argmax(Fn, axis=0), np.argmin(Fn, axis=0)]))
    extreme = np.unique(Fn[ex], axis=0)
    E = len(extreme)
    Nc = K - E
    if Nc > 0:
        T = fcluster(linkage(Fn, method="ward"), t=Nc, criterion="maxclust")
        ref = np.unique(np.array([Fn[T == i].mean(axis=0) for i in np.unique(T)]), axis=0)
        if len(ref) < Nc:
            ref = np.vstack([ref, Fn[rng.permutation(len(Fn))[: Nc - len(ref)]]])
        return np.vstack([ref[:Nc], extreme])
    return extreme[:K]


def _reference_point_selection(Fn, last, ref, K, rng):
    n = len(Fn)
    dis = pdist2(Fn, ref)
    nearest = np.argmin(dis, axis=1)
    dmin = dis[np.arange(n), nearest]
    label = np.full(n, 3)
    for i in range(len(ref)):
        a = np.where(nearest == i)[0]
        if len(a):
            b = np.argsort(dmin[a], kind="stable")
            label[a[b[0]]] = 1
            if len(a) > 3:
                label[a[b[1:]]] = 2
    t1, t2 = int(np.sum(label == 1)), int(np.sum(label == 2))
    if t1 <= K:
        sel = list(np.where(label == 1)[0])
        if t2 >= K - t1:
            d = np.where(label == 2)[0]
            sel += list(d[rng.permutation(len(d))[: K - t1]])
        else:
            sel += list(np.where(label == 2)[0])
            d = np.where(label == 3)[0]
            sel += list(d[rng.permutation(len(d))[: K - t1 - t2]])
    else:
        d = np.where(label == 1)[0]
        sel = list(d[rng.permutation(len(d))[:K]])
    return last[np.asarray(sel, dtype=int)]


class CAMOEA(LoopAlgorithm):
    def step(self):
        rng, N = self.rng, self.N
        perm = rng.permutation(len(self.pop))
        off = self.evaluate(ga(self.problem, decs(self.pop[perm]), rng=rng))
        uni = Population.merge(self.pop, off)
        F = objs(uni)
        front_no, max_f = nd_sort(F, None, N)
        K = N - int(np.sum(front_no < max_f))
        if K != 0:
            pareto = np.where(front_no < max_f)[0]
            last = np.where(front_no == max_f)[0]
            zmin, zmax = F[last].min(axis=0), F[last].max(axis=0)
            with np.errstate(all="ignore"):
                Fn = np.nan_to_num((F[last] - zmin) / (zmax - zmin))
            ref = _reference_generation(Fn, self.M, K, rng)
            refpop = _reference_point_selection(Fn, last, ref, K, rng)
        else:
            pareto, refpop = np.where(front_no <= max_f)[0], np.zeros(0, dtype=int)
        self.pop = uni[np.concatenate([pareto, refpop]).astype(int)]
