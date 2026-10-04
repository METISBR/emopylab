# emopylab 2026
"""SVR-NSGA-II (support vector regression based NSGA-II).

Reference:
L. Cao, L. Xu, E. D. Goodman, C. Bao, and S. Zhu. Evolutionary dynamic multiobjective optimization
assisted by a support vector regression predictor. IEEE Transactions on Evolutionary Computation,
2019, 24(2): 305-319.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga, nd_sort, objs, tournament
from algorithms.kl_nsga_ii.kl_nsga_ii import changed
from core.population import Population
from util.svm import SVR

ALGORITHM_FLAGS = {'SVRNSGAII': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}

P_HISTORY = 4


def _selection(pop, N):
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def _forecast(data, up, lo):
    """Fit an SVR on (history[:-1] -> latest) and read its prediction for the shifted history (the next step)."""
    svr = SVR(C=1e3, epsilon=0.05, kernel_scale=1.0, standardize=False).fit(data[:, :-1], data[:, -1])
    return float(np.clip(svr.predict(data[:, 1:])[-1], lo, up))


class SVRNSGAII(LoopAlgorithm):
    """NSGA-II that, when a change of the environment is detected, keeps the old population and rebuilds the new one
    variable by variable from an SVR forecast of each individual's decision-value history."""

    def __init__(self, pop_size: int = 100, change_count: int = 0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.change_count0 = int(change_count)

    def start(self):
        self.change_count = self.change_count0
        _, self.front, self.crowd = _selection(self.pop, self.N)
        self.all_pop = None
        self.nds = {0: self.pop}

    def _reinitialise(self):
        cc, P = self.change_count, P_HISTORY
        dec = decs(self.nds[cc]).copy()
        up, lo = self.upper, self.lower
        n, D = dec.shape
        cols = cc + 1 if cc < P else P + 1
        first = 0 if cc < P else cc - P
        data = np.zeros((0, cols))
        for i in range(n):
            row = np.zeros(cols)
            data = np.vstack([data, row])
            for z in range(D):
                for j in range(cols):
                    data[i, j] = decs(self.nds[first + j])[i, z]
                dec[i, z] = _forecast(data, up[z], lo[z])
        return self.evaluate(dec)

    def step(self):
        N, rng = self.N, self.rng
        if changed(self, self.pop):
            self.change_count += 1
            self.nds[self.change_count] = self.pop
            self.all_pop = self.pop if self.all_pop is None else Population.merge(self.all_pop, self.pop)
            self.pop = self._reinitialise()
            _, self.front, self.crowd = _selection(self.pop, len(self.pop))
        mate = tournament(2, N, self.front, -self.crowd, rng=rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[mate]), rng=rng))
        self.pop, self.front, self.crowd = _selection(Population.merge(self.pop, off), N)
        if self.FE >= self.max_FE and self.all_pop is not None:
            self.pop = Population.merge(self.all_pop, self.pop)
