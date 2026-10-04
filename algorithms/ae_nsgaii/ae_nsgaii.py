# emopylab 2026
"""AE-NSGA-II (autoencoding NSGA-II).

Reference:
L. Feng, W. Zhou, W. Liu, Y. S. Ong, and K. C. Tan. Solving dynamic multiobjective problem via
autoencoding evolutionary search. IEEE Transations on Cybernetics, 2020, 52(5): 2649-2662.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga, nd_sort, objs, tournament
from algorithms.kl_nsga_ii.kl_nsga_ii import changed, nsga2_selection
from core.population import Population

ALGORITHM_FLAGS = {'AENSGAII': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _by_crowding(pop, N):
    C = cons(pop)
    front, _ = nd_sort(objs(pop), C if C.size else None, N)
    cd = crowding(objs(pop), front)
    return pop[np.argsort(cd, kind="stable")]


def ae_prediction(algo, curr_nds, his_nds, curr_pop, NP):
    rng = algo.rng
    N1, N2 = len(curr_nds), len(his_nds)
    if N1 > N2:
        curr_nds = _by_crowding(curr_nds, N1)[:N2]
        NL = N2
    else:
        his_nds = _by_crowding(his_nds, N2)[:N1]
        NL = N1
    i1, i2 = decs(curr_nds), decs(his_nds)
    Q, P = i2 @ i2.T, i1 @ i2.T
    reg = 1e-5 * np.eye(NL)
    reg[-1, -1] = 0
    M = np.linalg.solve((Q + reg).T, P.T).T
    var = (i1 - M @ i2) ** 2
    v = var.mean()
    pre = algo.evaluate(M @ i1 + v)
    if len(pre) > NP / 2:
        pre = _by_crowding(pre, N1)[: NP // 2]
    sel = rng.permutation(len(curr_pop))[: NP // 2]
    pop = Population.merge(pre, curr_pop[sel])
    if len(pop) < NP:
        pop = Population.merge(pop, algo.evaluate(algo.random_decs(NP - len(pop))))
    return nsga2_selection(pop, algo.N)


class AENSGAII(LoopAlgorithm):
    """Autoencoding-prediction NSGA-II for dynamic problems: the non-dominated set of the previous environment and
    the current one are related by a closed-form linear (ridge) mapping used to predict the next set; the
    prediction, part of the old population and random solutions form the new population after a change."""

    def start(self):
        self.pop, self.front, self.crowd = nsga2_selection(self.pop, self.N)
        self.nds = [self.pop[self.front == 1]]
        self.count = 0
        self.all_pop = []

    def step(self):
        N, rng = self.N, self.rng
        if changed(self, self.pop):
            self.count += 1
            self.nds.append(self.pop[self.front == 1])
            self.all_pop.append(self.pop)
            self.pop, self.front, self.crowd = ae_prediction(self, self.nds[self.count], self.nds[self.count - 1], self.pop, N)
        pool = tournament(2, N, self.front, -self.crowd, rng=rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=rng))
        self.pop, self.front, self.crowd = nsga2_selection(Population.merge(self.pop, off), N)
        if self.FE >= self.max_FE and self.all_pop:
            self.pop = Population.merge(*self.all_pop, self.pop)
