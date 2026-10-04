# emopylab 2026
"""URCMO (utilizing the relationship between constrained and unconstrained Pareto fronts for constrained multi-objective optimization).

Reference:
J. Liang, K. Qiao, K. Yu, B. Qu, C. Yue, W. Guo, and L. Wang. Utilizing the relationship between
unconstrained and constrained Pareto fronts for constrained multi-objective optimization. IEEE
Transactions on Cybernetics, 2023, 53(6): 3873-3886.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, first_front, ga_half, nd_sort, objs, tournament
from algorithms.emcmms.emcmms import _bound_constraint
from core.population import Population

ALGORITHM_FLAGS = {'URCMO': {'constrained', 'integer', 'multi', 'real'}}

FM = np.array([0.6, 0.8, 1.0])
CRM = np.array([0.1, 0.2, 1.0])


def _select(pop, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return nxt


def _survive(pop, N, is_origin):
    fit = cal_fitness(objs(pop), cons(pop)) if is_origin == 1 else cal_fitness(objs(pop))
    nxt = _select(pop, fit, N)
    p, f = pop[nxt], fit[nxt]
    r = np.argsort(f, kind="stable")
    return p[r], f[r], nxt


def _gn_r123(NP1, r0, rng):
    """Three index vectors r1, r2, r3 (0-based) that differ from r0 and from each other."""
    n0 = len(r0)
    r1 = np.floor(rng.random(n0) * NP1).astype(int)
    for _ in range(1001):
        pos = r1 == r0
        if not pos.any():
            break
        r1[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int)
    r2 = np.floor(rng.random(n0) * NP1).astype(int)
    for _ in range(1001):
        pos = (r2 == r1) | (r2 == r0)
        if not pos.any():
            break
        r2[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int)
    r3 = np.floor(rng.random(n0) * NP1).astype(int)
    for _ in range(1001):
        pos = (r3 == r1) | (r3 == r0) | (r3 == r2)
        if not pos.any():
            break
        r3[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int)
    return r1, r2, r3


class URCMO(LoopAlgorithm):
    """Uncertainty-aware relationship-classified constrained MOEA: after a learning phase the relation between the
    constrained and the unconstrained population (converged / partially feasible / infeasible) selects how the
    helper population is bred (GA, transfer DE from the main population, or DE towards its best solutions)."""

    def __init__(self, pop_size: int = 100, first_FES: int = 10000, beita: float = 0.9, p: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.first_FES, self.beita, self.p = int(first_FES), float(beita), float(p)

    def start(self):
        self.P = [self.pop, self.evaluate(self.random_decs(self.N))]
        self.fit = [cal_fitness(objs(self.P[0]), cons(self.P[0])), cal_fitness(objs(self.P[1]))]
        self.cnt, self.succ, self.flag = 0, [], None

    # -- variation operators ---------------------------------------------------
    def _ga(self, pop, fit, n):
        pool = tournament(2, 2 * n, fit, rng=self.rng)
        return self.evaluate(ga_half(self.problem, decs(pop[pool]), rng=self.rng))

    def _de_current_to_rand(self, pop, n):
        rng, N, D = self.rng, self.N, self.D
        F = FM[rng.integers(0, 3, n)]
        perm = rng.permutation(N)
        r1, r2, r3 = _gn_r123(N, perm, rng)
        arr = perm[:n]
        X = decs(pop)
        p1 = X[:n]
        vi = p1 + rng.random((n, 1)) * (X[r1[arr]] - p1) + F[:, None] * (X[r2[arr]] - X[r3[arr]])
        return self.evaluate(_bound_constraint(vi, p1, self.lower, self.upper))

    def _de_transfer(self, pop1, pop2, n):
        rng, N, D = self.rng, self.N, self.D
        CR = CRM[rng.integers(0, 3, n)]
        index = rng.integers(0, N, n)
        arr = rng.permutation(N)[:n]
        x1, x2 = decs(pop1[arr]), decs(pop2)
        u = x2[index].copy()
        mask = rng.random((n, D)) > CR[:, None]
        mask[np.arange(n), np.floor(rng.random(n) * D).astype(int)] = False
        u[mask] = x1[mask]
        return self.evaluate(u)

    def _de_pbest(self, pop, n, other_fit, other_pop):
        rng, N, D = self.rng, self.N, self.D
        F = FM[rng.integers(0, 3, n)]
        perm = rng.permutation(N)
        r1, r2, r3 = _gn_r123(N, perm, rng)
        arr = perm[:n]
        X = decs(pop)
        p1 = X[arr]
        best = np.argsort(other_fit, kind="stable")
        pNP = max(int(np.floor(self.p * N + 0.5)), 2)
        ri = np.maximum(np.ceil(rng.random(n) * pNP).astype(int), 1) - 1
        pbest = decs(other_pop)[best[ri]]
        vi = p1 + F[:, None] * (pbest - p1 + X[r2[arr]] - X[r3[arr]])
        return self.evaluate(_bound_constraint(vi, p1, self.lower, self.upper))

    @staticmethod
    def _cv(pop):
        C = cons(pop)
        return np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))

    def _classify(self):
        P1, P2 = self.P
        cv1, cv2 = self._cv(P1), self._cv(P2)
        f1, _ = nd_sort(objs(P1), cv1[:, None], np.inf)
        x1 = np.where(f1 == 1)[0]
        f2, _ = nd_sort(objs(P2), None, np.inf)
        x2 = np.where(f2 == 1)[0]
        if not np.any(cv2[x2] <= 0):
            return 3
        if not np.any(cv2[x2] > 0):
            return 1
        front, _ = nd_sort(np.vstack([objs(P1)[x1], objs(P2)[x2]]), None, np.inf)
        ll = np.sum(front[: len(x1)] == 1) / len(x1)
        if ll > self.beita:
            return 1
        return 3 if ll < 1 - self.beita else 2

    def step(self):
        N, rng = self.N, self.rng
        self.cnt += 1
        P1, P2 = self.P
        h = N // 2
        if self.FE < self.first_FES:
            off = []
            for i in range(2):
                a = self._ga(self.P[i], self.fit[i], h)
                b = self._de_current_to_rand(self.P[i], h)
                off.append(Population.merge(a, b))
            new1 = _survive(Population.merge(P1, off[0], off[1]), N, 1)
            new2 = _survive(Population.merge(P2, off[1], off[0]), N, 2)
            n1 = new1[2]
            s1 = n1[N:2 * N]
            s2 = n1[2 * N:]
            self.succ.append([s1[:h].sum(), s1[h:N].sum(), s2[:h].sum(), s2[h:N].sum()])
            self.P[0], self.fit[0] = new1[0], new1[1]
            self.P[1], self.fit[1] = new2[0], new2[1]
            if self.FE >= self.first_FES:
                S = np.array(self.succ, dtype=float)
                a = np.zeros(4)
                for k in range(4):
                    sd = S[:, k].std(ddof=1) if len(S) > 1 else 0.0
                    a[k] = S[:, k].mean() / sd if sd != 0 else 0.0
                self.flag = self._classify()
                if self.flag == 1 and a[2] < a[3]:
                    self.flag = 3
        else:
            o1 = Population.merge(self._ga(P1, self.fit[0], h), self._de_current_to_rand(P1, h))
            flag = self.flag if self.flag is not None else self._classify()
            self.flag = flag
            if flag == 1:
                o2 = Population.merge(self._ga(P2, self.fit[1], h), self._de_transfer(P2, P1, h))
            elif flag == 2:
                num = int(np.ceil(N / 3))
                r = rng.random()
                rest = N - 2 * num
                if r <= 1 / 3:
                    parts = [self._ga(P2, self.fit[1], num), self._de_transfer(P2, P1, num), self._de_pbest(P2, rest, self.fit[0], P1)]
                elif r <= 2 / 3:
                    parts = [self._ga(P2, self.fit[1], num), self._de_pbest(P2, num, self.fit[0], P1), self._de_transfer(P2, P1, rest)]
                else:
                    parts = [self._de_pbest(P2, num, self.fit[0], P1), self._de_transfer(P2, P1, num), self._ga(P2, self.fit[1], rest)]
                o2 = Population.merge(*parts)
            else:
                o2 = self._de_pbest(P2, N, self.fit[0], P1)
            self.P[0], self.fit[0], _ = _survive(Population.merge(P1, o1, o2), N, 1)
            self.P[1], self.fit[1], _ = _survive(Population.merge(P2, o2, o1), N, 2)
        self.pop = self.P[0]
