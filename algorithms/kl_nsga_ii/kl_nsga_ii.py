# emopylab 2026
"""KL-NSGA-II (algorithm).

Reference:
Q. Zhao, B. Yan, Y. Shi, and M. Middendorf. Evolutionary dynamic multiobjective optimization via
learning from historical search process. IEEE Transactions on Cybernetics, 2021, 52(7): 6119-6130.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'KLNSGAII': {'binary', 'constrained', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


def nsga2_selection(pop, N):
    C = cons(pop)
    front, maxf = nd_sort(objs(pop), C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def changed(algo, pop):
    """Environmental change test: re-evaluate a tenth of the population and compare objectives and constraints."""
    sel = algo.rng.permutation(len(pop))[: int(np.ceil(len(pop) / 10))]
    old = pop[sel]
    new = algo.evaluate(decs(old))
    return not (np.array_equal(objs(old), objs(new)) and np.array_equal(cons(old), cons(new)))


def da_two_layer(algo, source_temp, source_gmax, target):
    """Two-layer denoising-autoencoder mapping: learns (ridge regression, closed form) how the previous environment's
    population maps to the current one and applies it to the present population."""
    noisy, clean = source_temp.T, source_gmax.T
    d, n = noisy.shape
    na = np.vstack([noisy, np.ones((1, n))])
    ca = np.vstack([clean, np.ones((1, n))])
    reg = 1e-5 * np.eye(d + 1)
    reg[-1, -1] = 0

    def rdiv(P, Q):
        return np.linalg.solve(Q.T, P.T).T

    W1 = rdiv(ca @ na.T, na @ na.T + reg)
    na2 = np.tanh(W1 @ na)
    W2 = rdiv(ca @ na2.T, na2 @ na2.T + reg)
    b = W1.shape[0] - 1
    W1 = np.delete(np.delete(W1, b, axis=0), b, axis=1)
    W2 = np.delete(np.delete(W2, b, axis=0), b, axis=1)
    off = (W2 @ np.tanh(W1 @ target.T)).T
    lower = np.r_[0.0, -np.ones(algo.D - 1)]
    upper = np.r_[1.0, np.ones(algo.D - 1)]
    off = np.maximum(np.minimum(off, upper), lower)
    return algo.evaluate(off)


class KLNSGAII(LoopAlgorithm):
    """Knowledge-learning NSGA-II for dynamic problems: after a change 20% of the population is re-initialised and
    the rest re-evaluated; once enough generations have passed, offspring proposed by a denoising autoencoder
    trained on the two most recent populations are added to the usual GA offspring."""

    def __init__(self, pop_size: int = 100, zeta: float = 0.2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.zeta = float(zeta)

    def start(self):
        self.pop, self.front, self.crowd = nsga2_selection(self.pop, self.N)
        self.j, self.tau, self.count = 1, 0, 0
        self.all_pop = []
        self.source = [None] * 10
        self.source[0] = self.pop

    def _reinit(self, pop):
        n = int(np.floor(len(pop) * self.zeta / 2) * 2)
        sel = self.rng.permutation(len(pop))[:n]
        new = self.evaluate(self.random_decs(n))
        rest = np.setdiff1d(np.arange(len(pop)), sel)
        re = self.evaluate(decs(pop[rest]))
        out = pop.copy(deep=False)
        for a, ind in zip(sel, new):
            out[int(a)] = ind
        for a, ind in zip(rest, re):
            out[int(a)] = ind
        return nsga2_selection(out, len(out))

    def step(self):
        N, rng = self.N, self.rng
        if changed(self, self.pop):
            self.count += 1
            self.source[0] = self.pop
            self.j = 0
            self.all_pop.append(self.pop)
            self.pop, self.front, self.crowd = self._reinit(self.pop)
        pool = tournament(2, N, self.front, -self.crowd, rng=rng)
        self.j += 1
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=rng))
        if self.tau > 50:
            filled = [k for k, s in enumerate(self.source) if s is not None]
            if len(filled) >= 2 and self.source[-2] is not None and self.source[-1] is not None:
                learn = da_two_layer(self, decs(self.source[-2]), decs(self.source[-1]), decs(self.pop))
                off = Population.merge(off, learn)
        self.pop, self.front, self.crowd = nsga2_selection(Population.merge(self.pop, off), N)
        while len(self.source) < self.j:
            self.source.append(None)
        if self.j <= len(self.source):
            self.source[self.j - 1] = self.pop
        self.tau += 1
        if self.FE >= self.max_FE and self.all_pop:
            self.pop = Population.merge(*self.all_pop, self.pop)
