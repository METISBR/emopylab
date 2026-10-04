# emopylab 2026
"""CoMMEA (coevolutionary multimodal multi-objective evolutionary algorithm).

Reference:
W. Li, X. Yao, K. Li, R. Wang, T. Zhang, and  L. Wang. Coevolutionary framework for generalized
multimodal multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2023, 10(7):
1544-1556.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'CoMMEA': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


def _dominance(F):
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    return k == 1


def cal_fitness(pop, operation):
    F, X = objs(pop), decs(pop)
    Dom = _dominance(F)
    if operation == "LocalC":
        d = pdist2(X, X)
        R = d.mean() / 4 if X.shape[1] <= 8 else d.mean() / 2
        Dom = Dom & (d < R)
    return Dom.sum(axis=1) @ Dom


def crowding_dec(X):
    N = len(X)
    if N < 2:
        return np.full(N, np.inf)
    Z, Zmax = X.min(axis=0), X.max(axis=0)
    with np.errstate(all="ignore"):
        P = (X - Z) / (Zmax - Z)
    v = np.sort(pdist2(P, P), axis=1)
    with np.errstate(all="ignore"):
        return (N - 1) / np.sum(1.0 / v[:, 1:], axis=1)


def _three_nearest(X):
    d = np.sort(pdist2(X, X), axis=0)
    return d[:3].sum(axis=0)


def kdis(pop, K):
    X = decs(pop)
    d = pdist2(X, X)
    np.fill_diagonal(d, np.inf)
    dn = np.sort(d, axis=0)[: int(K)].sum(axis=0)
    avg = dn.mean()
    if avg == 0:
        avg = np.inf
    return 1.0 / (1 + dn / avg)


def env_selection1(pop, N, operation, state):
    fit = cal_fitness(pop, operation)
    nxt = fit < 1
    if nxt.sum() < N:
        rank = np.lexsort((-crowding_dec(decs(pop)), fit))
        pop = pop[rank[:N]]
    else:
        pop = pop[nxt]
        while len(pop) > N:
            pop = pop[np.delete(np.arange(len(pop)), int(np.argmin(_three_nearest(decs(pop)))))]
    if state < 0.5:
        return pop, cal_fitness(pop, operation) - crowding_dec(decs(pop))
    return pop, -crowding_dec(decs(pop))


def env_selection2(pop, N, operation, eps):
    front, _ = nd_sort(objs(pop), None, N)
    nxt = np.where(front == 1)[0]
    remain = np.where(front > 1)[0]
    F = objs(pop)
    is_eps = np.zeros(len(remain), bool)
    for j, r in enumerate(remain):
        tmp = np.vstack([F[r], (1 + eps) * F[nxt]])
        fno, _ = nd_sort(tmp, None, np.inf)
        is_eps[j] = fno[0] == 1
    if len(nxt) + is_eps.sum() < N:
        idx = np.lexsort((-crowding_dec(decs(pop)), front))
        pop = pop[idx[:N]]
    else:
        pop = Population.merge(pop[nxt], pop[remain[is_eps]])
    fit = cal_fitness(pop, operation)
    nxt = fit < 1
    K = 3
    if nxt.sum() < N:
        cd = -np.sort(pdist2(decs(pop), decs(pop)), axis=0)[:K].sum(axis=0)
        pop = pop[np.lexsort((cd, fit))[:N]]
    else:
        pop = pop[nxt]
        while len(pop) > N:
            do = np.sort(pdist2(objs(pop), objs(pop)), axis=0)[:K].sum(axis=0)
            dd = np.sort(pdist2(decs(pop), decs(pop)), axis=0)[:K].sum(axis=0)
            with np.errstate(all="ignore"):
                cd = do / do.max() + dd / dd.max()
            pop = pop[np.delete(np.arange(len(pop)), int(np.argmin(cd)))]
    return pop, kdis(pop, N / 2)


class CoMMEA(LoopAlgorithm):
    """Coevolutionary multimodal EA: a global population (domination, decision-space crowding) and a niche
    population (local domination inside decision-space niches, epsilon-dominance tolerance that decays over the
    run) exchange offspring."""

    def __init__(self, pop_size: int = 100, eps: float = 0.2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.eps = float(eps)

    def start(self):
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(self.N))
        self.f1 = cal_fitness(self.P1, "Normal")
        self.f2 = cal_fitness(self.P2, "LocalC")

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        m1 = tournament(2, int(np.round(N / 4)), self.f1, rng=rng)
        m2 = tournament(2, N, self.f2, rng=rng)
        off1 = self.evaluate(ga(pr, decs(self.P1[m1]), rng=rng))
        off2 = self.evaluate(ga(pr, decs(self.P2[m2]), rng=rng))
        state = self.FE / self.max_FE
        cur = max(-np.log2(1.5 * state), self.eps) if state > 0 else np.inf
        self.P1, self.f1 = env_selection1(Population.merge(self.P1, off1, off2), N, "Normal", state)
        self.P2, self.f2 = env_selection2(Population.merge(self.P2, off1, off2), N, "LocalC", cur)
        self.pop = self.P2
