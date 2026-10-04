# emopylab 2026
"""MOEA-D-CMT (mOEA/D with competitive multitasking).

Reference:
X. Chu, F. Ming, and W. Gong. Competitive multitasking for computational resource allocation in
evolutionary constrained multi-objective optimization. IEEE Transactions on Evolutionary
Computation, 2025, 29(3): 809-821.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga_half, neighbors_of, objs, truncate_lexi, uniform_point, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'MOEADCMT': {'constrained', 'multi', 'real'}}


def _ocv(c):
    c = np.asarray(c)
    return np.sum(np.abs(np.maximum(c, 0)), axis=1) if c.size else np.zeros(len(c))


def _lg(x, sigma):
    return 1.0 / (1.0 + np.exp(-x / sigma))


def _archive_update(pop, N):
    F = objs(pop)
    fit = cal_fitness(F, cons(pop))
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        d = pdist2(F[nxt], F[nxt])
        np.fill_diagonal(d, np.inf)
        idx = np.where(nxt)[0]
        nxt[idx[truncate_lexi(d, int(nxt.sum()) - N)]] = False
    return pop[nxt], nxt


class MOEADCMT(LoopAlgorithm):
    """MOEA/D with two cooperating tasks (constrained main task and an unconstrained PBI helper); a bandit-style
    reward decides which task drives each generation and offspring are shared between the tasks."""

    def __init__(self, pop_size: int = 100, delta: float = 0.9, nr: int = 2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.nr = float(delta), int(nr)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.P = [self.pop, self.evaluate(self.random_decs(self.N))]
        self.Z = [objs(self.P[0]).min(axis=0), objs(self.P[1]).min(axis=0)]
        self.conmin = float(_ocv(cons(self.P[0])).min())
        self.A = self.P[0]
        self.rwd = np.zeros(2)
        self.RMP, self.Nt, self.beta, self.pmin = 0.2, 2, 0.3, 0.1

    def _update_main(self, P, off, sigma_obj, sigma_cv):
        W = self.W
        fo = objs(off)[0]
        self.Z[0] = np.minimum(self.Z[0], fo)
        cvo = float(_ocv(cons(off))[0])
        self.conmin = min(self.conmin, cvo)
        conmax = max(float(_ocv(cons(self.P[0])).max()), cvo)
        with np.errstate(all="ignore"):
            g_old = np.max(np.abs(objs(self.P[0][P]) - self.Z[0]) / W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z[0]) / W[P], axis=1)
        cv_old = _ocv(cons(self.P[0][P]))
        cv_new = np.full(len(P), cvo)
        if conmax > self.conmin:
            pos = cv_old > 0
            cv_old[pos] = (cv_old[pos] - self.conmin) / (conmax - self.conmin)
            if cvo > 0:
                cv_new[:] = (cvo - self.conmin) / (conmax - self.conmin)
        new_old = _lg(g_old - g_new, sigma_obj) * np.maximum(_lg(cv_old - cv_new, sigma_cv), 0.0001)
        old_new = _lg(g_new - g_old, sigma_obj) * np.maximum(_lg(cv_new - cv_old, sigma_cv), 0.0001)
        for j in np.where(new_old >= old_new)[0][: self.nr]:
            self.P[0][P[j]] = off[0]

    def _update_aux(self, P, off):
        W = self.W
        fo = objs(off)[0]
        self.Z[1] = np.minimum(self.Z[1], fo)
        normW = np.linalg.norm(W[P], axis=1)
        d = objs(self.P[1][P]) - self.Z[1]
        normP = np.linalg.norm(d, axis=1)
        normO = np.linalg.norm(fo - self.Z[1])
        with np.errstate(all="ignore"):
            cP = np.sum(d * W[P], axis=1) / normW / normP
            cO = np.sum((fo - self.Z[1]) * W[P], axis=1) / normW / normO
            g_old = normP * cP + 5 * normP * np.sqrt(1 - cP ** 2)
            g_new = normO * cO + 5 * normO * np.sqrt(1 - cO ** 2)
        for j in np.where(g_old >= g_new)[0]:
            self.P[1][P[j]] = off[0]

    def step(self):
        N, rng, B, W, pr = self.N, self.rng, self.B, self.W, self.problem
        cv = _ocv(cons(self.P[0]))
        fr = np.sum(cv <= 0) / N
        sigma_obj, sigma_cv = 0.3, fr * 10
        Q = []
        if self.FE <= self.beta * self.max_FE or self.rwd.sum() == 0:
            pro = np.full(self.Nt, 1.0 / self.Nt)
        else:
            pro = self.pmin / self.Nt + (1 - self.pmin) * self.rwd / self.rwd.sum()
            pro = pro / pro.sum()
        r = rng.random()
        k = int(np.searchsorted(np.cumsum(pro), r, side="left"))
        k = min(k, self.Nt - 1)
        for i in range(N):
            P = B[i][rng.permutation(B.shape[1])] if rng.random() < self.delta else rng.permutation(N)
            if k == 0:
                src = self.P[0] if rng.random() < self.RMP else self.P[1]
                off = self.evaluate(de(pr, decs(self.P[0][[i]]), decs(src[[P[0]]]), decs(src[[P[1]]]), rng=rng))
                self._update_main(P, off, sigma_obj, sigma_cv)
                self._update_aux(P, off)
            else:
                src = self.P[1] if rng.random() < self.RMP else self.P[0]
                off = self.evaluate(ga_half(pr, decs(src[P[:2]]), rng=rng))
                self._update_aux(P, off)
                self._update_main(P, off, sigma_obj, sigma_cv)
            Q.append(off[0])
        if Q:
            s = len(self.A)
            self.A, nxt = _archive_update(Population.merge(self.A, Population.create(Q)), N)
            if s >= N:
                self.rwd[k] += np.sum(nxt[N:]) / N
        self.pop = self.A
