# emopylab 2026
"""MO-MFEA-II (multi-objective multifactorial evolutionary algorithm II).

Reference:
K. K. Bali, A. Gupta, Y. Ong, and P. S. Tan. Cognizant multitasking in multiobjective multifactorial
evolution: MO-MFEA-II. IEEE Transactions on Cybernetics, 2021, 51(4): 1784-1796.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, ga_half, tournament
from algorithms.community_utils.optimize1d import fminbnd
from algorithms.mo_mfea.mo_mfea import _select, divide
from core.population import Population

ALGORITHM_FLAGS = {'MOMFEAII': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _normal_pdf(x, mu, sd):
    with np.errstate(all="ignore"):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _loglik(rmp, pm, ntasks):
    f = 0.0
    for i in range(2):
        m = pm[i].copy()
        for j in range(2):
            m[:, j] *= (1 - 0.5 * (ntasks - 1) * rmp / ntasks) if i == j else 0.5 * (ntasks - 1) * rmp / ntasks
        with np.errstate(all="ignore"):
            f += np.sum(-np.log(m.sum(axis=1)))
    return f


def learn_rmp(problem, sub, rng):
    T = len(getattr(problem, "sub_d", [problem.n_var]))
    vars_ = list(getattr(problem, "sub_d", [problem.n_var]))
    data = [np.zeros((0, getattr(problem, "sub_d", [problem.n_var])[0]))] + [None] * (T - 1)
    data = [[] for _ in range(T)]
    for s in sub:
        X = decs(s)
        for x in X:
            data[int(round(x[-1])) - 1].append(x[: getattr(problem, "sub_d", [problem.n_var])[0]])
    data = [np.array(d).reshape(-1, getattr(problem, "sub_d", [problem.n_var])[0]) for d in data]
    maxd = max(vars_)
    rmp = np.eye(T)
    mean, std, ns = [], [], []
    for i in range(T):
        ns.append(len(data[i]))
        nr = int(np.floor(0.1 * ns[i]))
        stack = np.vstack([data[i], rng.random((nr, data[i].shape[1]))])
        mean.append(stack.mean(axis=0))
        std.append(stack.std(axis=0, ddof=1))
    for i in range(T):
        for j in range(i + 1, T):
            d = min(vars_[i], vars_[j])
            pm = [np.ones((ns[i], 2)), np.ones((ns[j], 2))]
            for (side, task) in ((0, i), (1, j)):
                for l in range(d):
                    pm[side][:, 0] *= _normal_pdf(data[task][:, l], mean[i][l], std[i][l])
                    pm[side][:, 1] *= _normal_pdf(data[task][:, l], mean[j][l], std[j][l])
            val = fminbnd(lambda x: _loglik(x, pm, T), 0.0, 1.0)
            rmp[i, j] = min(max(0.0, val + rng.normal(0, 0.01)), 1.0)
            rmp[j, i] = rmp[i, j]
    return rmp


class MOMFEAII(LoopAlgorithm):
    """MO-MFEA with online learning of the random-mating-probability matrix: the task-pair probabilities are the
    maximum-likelihood mixing weights of univariate Gaussian models of the subpopulations; parents of different
    tasks that do not mate get a partner from their own task."""

    def start(self):
        self.T = len(getattr(self.problem, "sub_d", [self.problem.n_var]))
        self.sub = divide(self.pop, self.T)

    def _create_off(self, pool, sub, RMP):
        rng, pr = self.rng, self.problem
        n = len(pool) // 2
        X = decs(pool)
        g1, g2 = [], []
        for i in range(n):
            t1, t2 = int(round(X[i, -1])), int(round(X[i + n, -1]))
            rmp = RMP[t1 - 1, t2 - 1]
            (g1 if (t1 == t2 or rng.random() < rmp) else g2).append(i)
        blocks = []
        if g1:
            P1, P2 = X[g1], X[[i + n for i in g1]]
            off = ga(pr, np.vstack([P1, P2]), rng=rng)
            off[:, -1] = np.concatenate([P1[:, -1], P2[:, -1]])
            blocks.append(off)
        if g2:
            p1, p2 = [], []
            for i in g2:
                for idx in (i, i + n):
                    t = int(round(X[idx, -1])) - 1
                    p1.append(X[idx])
                    p2.append(decs(sub[t])[int(rng.integers(len(sub[t])))])
            p1, p2 = np.array(p1), np.array(p2)
            off = ga_half(pr, np.vstack([p1, p2]), rng=rng)
            off[:, -1] = p1[:, -1]
            blocks.append(off)
        return divide(self.evaluate(np.vstack(blocks)), self.T)

    def step(self):
        T, rng = self.T, self.rng
        RMP = learn_rmp(self.problem, self.sub, rng)
        ranks, sub = [], []
        for i in range(T):
            s = self.sub[i]
            _, front, cd = _select(s, len(s))
            order = np.lexsort((-cd, front))
            sub.append(s[order])
            ranks.append(np.arange(1, len(s) + 1, dtype=float))
        pop = Population.merge(*sub)
        pool = pop[tournament(2, len(pop), np.concatenate(ranks), rng=rng)]
        off = self._create_off(pool, sub, RMP)
        self.sub = [_select(Population.merge(sub[i], off[i]), len(sub[i]))[0] if len(off[i]) else sub[i] for i in range(T)]
        self.pop = Population.merge(*self.sub)
