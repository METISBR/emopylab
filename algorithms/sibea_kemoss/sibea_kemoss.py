# emopylab 2026
"""SIBEA-kEMOSS (sIBEA with minimum objective subset of size k with minimum error).

Reference:
D. Brockhoff and E. Zitzler. Improving hypervolume-based multiobjective evolutionary algorithms by
using objective reduction methods. Proceedings of the IEEE Congress on Evolutionary Computation,
2007, 2086-2093.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from core.population import Population
from util.hv import hv_contributions

ALGORITHM_FLAGS = {'SIBEAkEMOSS': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _delta_min(F, set1, set2):
    """Largest loss in ``set2`` objectives when dominance is judged on ``set1`` only."""
    dom = np.all(F[:, None, set1] <= F[None, :, set1], axis=2)          # dom[i, j]: i weakly dominates j on set1
    diff = F[:, None, :][:, :, set2] - F[None, :, :][:, :, set2]        # f_i - f_j on the full set
    vals = np.where(dom[:, :, None], diff, -np.inf)
    return max(0.0, float(vals.max()))


def _kemoss(F, k):
    all_obj = list(range(F.shape[1]))
    selected = []
    while len(selected) < k:
        unsel = [o for o in all_obj if o not in selected]
        errors = [_delta_min(F, selected + [u], all_obj) for u in unsel]
        selected.append(unsel[int(np.argmin(errors))])
    return selected


def _environmental_selection(pop, N, obj_set):
    F = objs(pop)[:, obj_set]
    front_no, max_f = nd_sort(F, None, N)
    nxt = front_no < max_f
    last = np.where(front_no == max_f)[0]
    loss = np.zeros(len(last))
    Fl = F[last]
    ref = Fl.max(axis=0) + 0.1
    loss = hv_contributions(Fl, ref)
    rank = np.argsort(-loss, kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt]


class SIBEAkEMOSS(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, G: int = 5, k: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.G, self.k = int(G), int(k)

    def start(self):
        self.it = 0
        self.obj_set = list(range(min(self.k, self.M)))

    def step(self):
        if self.it % self.G == 0:
            self.obj_set = _kemoss(objs(self.pop), min(self.k, self.M))
        self.it += 1
        pool = self.rng.integers(0, len(self.pop), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.pop = _environmental_selection(Population.merge(self.pop, off), self.N, self.obj_set)
