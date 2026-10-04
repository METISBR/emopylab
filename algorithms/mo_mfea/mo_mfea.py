# emopylab 2026
"""MO-MFEA (multi-objective multifactorial evolutionary algorithm).

Reference:
A. Gupta, Y. Ong, L. Feng, and K. C. Tan. Multiobjective multifactorial optimization in evolutionary
multitasking. IEEE Transactions on Cybernetics, 2017, 47(7): 1652-1665.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MOMFEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _select(pop, N):
    C = cons(pop)
    front, maxf = nd_sort(objs(pop), C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(objs(pop), front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def divide(pop, T):
    skills = np.rint(decs(pop)[:, -1])
    return [pop[skills == i + 1] for i in range(T)]


class MOMFEA(LoopAlgorithm):
    """Multi-objective multifactorial evolutionary algorithm: one subpopulation per task, all tasks share one
    unified decision space (the last variable is the task label); parents from different tasks recombine with
    probability ``rmp`` (assortative mating) and children keep the label of their first / second parent."""

    def __init__(self, pop_size: int = 100, rmp: float = 1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.rmp = float(rmp)

    def start(self):
        self.T = len(getattr(self.problem, "sub_d", [self.problem.n_var]))
        self.sub = divide(self.pop, self.T)

    def _create_off(self, pool):
        rng, pr = self.rng, self.problem
        n = len(pool) // 2
        g1, g2 = [], []
        for i in range(n):
            same = int(round(pool[i].X[-1])) == int(round(pool[i + n].X[-1]))
            (g1 if (same or rng.random() < self.rmp) else g2).append(i)
        blocks = []
        for idx, params in ((g1, None), (g2, [0, 20, 1, 20])):
            if not idx:
                continue
            X = decs(pool)
            P1, P2 = X[idx], X[[i + n for i in idx]]
            off = ga(pr, np.vstack([P1, P2]), params, rng=rng)
            off[:, -1] = np.concatenate([P1[:, -1], P2[:, -1]])
            blocks.append(off)
        return divide(self.evaluate(np.vstack(blocks)), self.T)

    def step(self):
        T, rng = self.T, self.rng
        ranks, sub = [], []
        for i in range(T):
            s = self.sub[i]
            if len(s) == 0:
                sub.append(s)
                ranks.append(np.zeros(0))
                continue
            _, front, cd = _select(s, len(s))
            order = np.lexsort((-cd, front))
            sub.append(s[order])
            ranks.append(np.arange(1, len(s) + 1, dtype=float))
        pop = Population.merge(*[s for s in sub if len(s)])
        pool = pop[tournament(2, len(pop), np.concatenate(ranks), rng=rng)]
        off = self._create_off(pool)
        self.sub = [(_select(Population.merge(sub[i], off[i]), len(sub[i]))[0] if len(sub[i]) and len(off[i]) else sub[i]) for i in range(T)]
        self.pop = Population.merge(*[s for s in self.sub if len(s)])
