# emopylab 2026
"""DCNSGA-III (dynamic constrained NSGA-III).

Reference:
R. Jiao, S. Zeng, C. Li, S. Yang, and Y. S. Ong. Handling constrained many-objective optimization
problems via problem transformation. IEEE Transactions on Cybernetics, 2021, 51(10): 4834-4847.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, cosine_distance, decs, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'DCNSGAIII': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def reduce_boundary(eF, k, max_k, cp):
    z, near = 1e-8, 1e-15
    with np.errstate(all="ignore"):
        B = max_k / np.power(np.log((eF + z) / z), 1.0 / cp)
        B = np.where(B == 0, B + near, B)
        f = eF * np.exp(-((k / B) ** cp))
    f = np.where(np.abs(f - z) < near, z, f)
    eps = f - z
    eps[eps <= 0] = 0
    return eps


def _last_selection(F1, F2, K, Z, Zmin, rng):
    F = np.vstack([F1, F2])
    N, M = F.shape
    N1, N2, NZ = len(F1), len(F2), len(Z)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    extreme = np.array([int(np.argmin(np.max(F / w[i], axis=1))) for i in range(M)])
    try:
        a = 1.0 / np.linalg.solve(F[extreme], np.ones(M))
    except np.linalg.LinAlgError:
        a = np.full(M, np.nan)
    if np.any(np.isnan(a)):
        a = F.max(axis=0)
    F = (F - Zmin) / (a - Zmin)
    cosine = 1 - cosine_distance(F, Z)
    dist = np.linalg.norm(F, axis=1)[:, None] * np.sqrt(np.maximum(0, 1 - cosine ** 2))
    pi = np.argmin(dist, axis=1)
    d = dist[np.arange(N), pi]
    rho = np.zeros(NZ, dtype=int)
    if N1 > 0:
        rho += np.bincount(pi[:N1], minlength=NZ)
    choose = np.zeros(N2, bool)
    zchoose = np.ones(NZ, bool)
    while choose.sum() < K:
        temp = np.where(zchoose)[0]
        jmin = np.where(rho[temp] == rho[temp].min())[0]
        j = temp[jmin[rng.integers(len(jmin))]]
        I = np.where(~choose & (pi[N1:] == j))[0]
        if len(I):
            s = int(np.argmin(d[N1 + I])) if rho[j] == 0 else int(rng.integers(len(I)))
            choose[I[s]] = True
            rho[j] += 1
        else:
            zchoose[j] = False
    return choose


class DCNSGAIII(LoopAlgorithm):
    """NSGA-III whose feasibility threshold (epsilon per constraint) shrinks along an exponential schedule."""

    def __init__(self, pop_size: int = 100, cp: float = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.cp = float(cp)

    def initial_size(self):
        self.Z, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        e = np.max(np.maximum(0, _con(self.pop)), axis=0)
        e[e == 0] = 1
        self.initialE = e

    def _select(self, pop, eps, Zmin):
        N, rng = self.N, self.rng
        C = _con(pop)
        nCon = C.shape[1]
        vio = np.maximum(0, C)
        ok = np.sum(vio <= eps, axis=1) == nCon
        if ok.sum() > N:
            pop = pop[ok]
            CV = np.sum(np.maximum(0, _con(pop)) / self.initialE, axis=1) / nCon
            front, maxf = nd_sort(np.hstack([objs(pop), CV[:, None]]), None, N)
            nxt = front < maxf
            last = np.where(front == maxf)[0]
            ch = _last_selection(objs(pop)[nxt], objs(pop)[last], N - int(nxt.sum()), self.Z, Zmin, rng)
            nxt[last[ch]] = True
            return pop[nxt]
        CV = np.sum(np.maximum(0, C) / self.initialE, axis=1) / nCon
        return pop[np.argsort(CV, kind="stable")[:N]]

    def step(self):
        N, pop = self.N, self.pop
        eps = reduce_boundary(self.initialE, int(np.ceil(self.FE / N)), int(np.ceil(self.max_FE / N)) - 1, self.cp)
        pool = tournament(2, N, np.sum(np.maximum(0, _con(pop) - eps), axis=1), rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(pop[pool]), rng=self.rng))
        parts = []
        for p in (pop, off):
            C = _con(p)
            parts.append(objs(p)[np.sum(np.maximum(0, C) <= eps, axis=1) == C.shape[1]])
        parts = np.vstack(parts)
        Zmin = parts.min(axis=0) if len(parts) else np.minimum(objs(pop).min(axis=0), objs(off).min(axis=0))
        self.pop = self._select(Population.merge(pop, off), eps, Zmin)
