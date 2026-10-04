# emopylab 2026
"""CMOEA-MSG (multi-stage constrained multi-objective evolutionary algorithm).

Reference:
Y. Tian, J. Chen, and X. Zhang. An optimizer combining evolutionary computation and gradient descent
for constrained multi-objective optimization. Journal of Computer Applications (Chinese), 2024,
44(05): 1386-1392.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, ga, objs, tournament, uniform_point
from algorithms.mscmo.mscmo import cal_fitness
from core.population import Population

ALGORITHM_FLAGS = {'CMOEAMSG': {'constrained', 'integer', 'multi', 'real'}}

POP = 100          # the reference fixes the size of the main population


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def constraint_priority(pop, priority, current):
    C = _con(pop)
    fr = np.array([np.sum(C[:, j] <= 0) / len(pop) for j in range(C.shape[1])])
    if len(priority) == 0:
        return np.argsort(fr, kind="stable"), fr
    if current + 1 < C.shape[1]:
        fp = fr[priority]
        order = np.argsort(fp[current:], kind="stable")
        priority = priority.copy()
        priority[current:] = priority[order + current]
    return priority, fr


def _survive(pop, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    return nxt


def env_selection1(pop, N, priority, count, c):
    fit = cal_fitness(objs(pop), _con(pop), priority, count, c)
    nxt = _survive(pop, fit, N)
    p, f = pop[nxt], fit[nxt]
    r = np.argsort(f, kind="stable")
    return p[r], f[r]


def env_selection(pop, N, priority, count, c):
    """Region-wise ranking: inside each of 100 reference regions the solutions are ranked by fitness, then whole rank
    levels are taken until the population is full and the overflowing level is truncated."""
    F = objs(pop)
    fit = cal_fitness(F, _con(pop), priority, count, c)
    M = F.shape[1]
    L = len(pop)
    rank = np.full(L, np.inf)
    z = F.min(axis=0)
    W, _ = uniform_point(100, M)
    Fz = F - z
    with np.errstate(all="ignore"):
        cosd = 1 - (Fz @ W.T) / (np.linalg.norm(Fz, axis=1)[:, None] * np.linalg.norm(W, axis=1)[None, :])
    region = np.argmin(np.where(np.isnan(cosd), np.inf, cosd), axis=1)
    for i in range(len(W)):
        idx = np.where(region == i)[0]
        if len(idx):
            rank[idx[np.argsort(fit[idx], kind="stable")]] = np.arange(1, len(idx) + 1)
    a, num, keep = 0, 1, []
    top = int(np.max(rank[np.isfinite(rank)])) if np.isfinite(rank).any() else 0
    while num <= top:
        lvl = np.where(rank == num)[0]
        if a + len(lvl) <= N:
            a += len(lvl)
            keep.append(lvl)
            num += 1
        else:
            a += len(lvl)
            break
    chosen = np.concatenate(keep) if keep else np.zeros(0, dtype=int)
    if len(chosen) == N or num > top:
        return pop[chosen], fit[chosen]
    lvl = np.where(rank == num)[0]
    dele = _truncation(F[lvl], a - N)
    chosen = np.concatenate([chosen, lvl[~dele]])
    return pop[chosen], fit[chosen]


def _compare(new, old, priority, current, handling):
    """Return (winner, improved): the new solution wins when it has smaller violation, or equal violation and
    dominates the old one."""
    if current == 0:
        cv = np.zeros(2)
    else:
        C = _con(Population.merge(new, old))[:, priority[:current]]
        cv = np.sum(np.maximum(0, C), axis=1)
    if handling == 0:
        if cv[0] < cv[1]:
            return new, True
        if cv[0] > cv[1]:
            return old, False
        f1, f2 = objs(new)[0], objs(old)[0]
        k = int(np.any(f1 < f2)) - int(np.any(f1 > f2))
        return (new, True) if k == 1 else (old, False)
    return new, True


class CMOEAMSG(LoopAlgorithm):
    """Constrained MOEA with multi-stage gradient guidance: solutions are pushed by the sign of a (conflict-free)
    objective gradient, or by the gradient of the currently active constraint, before the usual GA variation; the
    active constraints are switched on in priority order (hardest first) when the archive stagnates."""

    def __init__(self, pop_size: int = 100, sampling=None, type=1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.type = type

    def _initialize_infill(self):
        rng = self.rng
        P1 = self.evaluate(self.random_decs(POP))
        P2 = self.evaluate(self.random_decs(POP))
        self.priority, _ = constraint_priority(Population.merge(P1, P2), np.array([], dtype=int), 0)
        self.current, self.handling = 0, 0
        self.last_gen = max(30, int(np.floor(200 / (len(self.priority) + 1))))
        self.thr, self.gen, self.ii, self.i, self.ab = 0.1, 0, 0, 0, 10
        self.change = {}
        self.Ar = P1
        P1, _ = env_selection1(P1, POP, self.priority, 0, 0)
        for h in range(len(P1)):
            while True:
                X = P1[[h]]
                g = np.sign(self._gradient(X, 0, 0))
                while True:
                    step = np.abs(np.asarray(P1[h].X, float) - np.asarray(P1[int(rng.integers(POP)) % len(P1)].X, float))
                    self.i += 1
                    H = X
                    X = self.evaluate((np.asarray(X[0].X, float) - step * g * 0.1)[None, :])
                    X, ok = _compare(X, H, self.priority, 0, 0)
                    if self.i == self.ab or not ok:
                        break
                if self.i == 10 or not ok:
                    self.i = 0
                    break
            P1 = Population.merge(P1, X)
        self.P1, _ = env_selection(P1, POP, self.priority, 0, 0)
        return self.P1

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.Ar
        self._set_optimum()

    def _gradient(self, X, current, oc):
        og, cg = self.cal_grad(np.asarray(X[0].X, dtype=float))
        if oc == 1:
            return cg[self.priority[current - 1]]
        g = og.sum(axis=0)
        conflict = np.any(og < 0, axis=0) & np.any(og > 0, axis=0)
        g = g.copy()
        g[conflict] = 0
        return g

    def _selection(self, fit):
        P = self.P1
        C = _con(P)[:, self.priority]
        cur = self.current
        if cur == 0:
            order = np.argsort(fit, kind="stable")
            pick = [order[max(len(order) // 10 - 1, 0)]] + [int(k) for k in self.rng.integers(0, POP, 2)]
            return P[pick], 0
        if cur == 1:
            tmp = np.where(C[:, 0] > 0)[0]
            if len(tmp) == 0:
                return P[[int(np.argsort(fit, kind="stable")[0])]], 0
            return P[[int(tmp[np.argsort(fit[tmp], kind="stable")[-1]])]], 1
        tmp = np.where(C[:, : cur - 1].max(axis=1) < 0)[0]
        idx = np.where(C[tmp, cur - 1] > 0)[0]
        if len(idx) == 0:
            return P[[int(np.argsort(fit, kind="stable")[0])]], 0
        tmp = tmp[idx]
        return P[[int(tmp[np.argsort(fit[tmp], kind="stable")[-1]])]], 1

    def step(self):
        rng, N, pr = self.rng, self.N, self.problem
        self.ii += 1
        ii, nCon = self.ii, _con(self.P1).shape[1]
        if self.current <= nCon and self.handling == 0:
            self.change[ii] = objs(self.Ar).mean(axis=0)
            if ii - self.gen > self.last_gen:
                recent = np.mean([self.change[k] for k in range(ii - 30, ii)], axis=0)
                if np.min(np.abs(self.change[ii] - recent)) <= self.thr:
                    if self.current < nCon:
                        self.priority, _ = constraint_priority(self.P1, self.priority, self.current)
                        self.current += 1
                    elif self.current == nCon:
                        self.handling = 1
                    self.gen = ii + 1
        if self.handling == 0:
            fit1 = cal_fitness(objs(self.P1), _con(self.P1), self.priority, self.current, self.handling)
            Xx, oc = self._selection(fit1)
            m = 3 if self.current == 0 else 1
            for h in range(m):
                X = Xx[[h]]
                g = np.sign(self._gradient(X, self.current, oc))
                while True:
                    self.i += 1
                    step = np.abs(np.asarray(self.P1[h].X, float) - np.asarray(self.P1[int(rng.integers(POP)) % len(self.P1)].X, float))
                    H = X
                    X = self.evaluate((np.asarray(X[0].X, float) - step * g * 0.1)[None, :])
                    X, ok = _compare(X, H, self.priority, self.current, self.handling)
                    if self.i > self.ab or not ok:
                        self.i = 0
                        break
                self.P1 = Population.merge(self.P1, X)
        nP = len(self.priority)
        fit = cal_fitness(objs(self.P1), _con(self.P1), self.priority, nP, 0)
        if self.type == 1:
            off1 = self.evaluate(ga(pr, decs(self.P1[tournament(2, POP, fit, rng=rng)]), rng=rng))
        else:
            a = len(self.P1)
            m1, m2 = tournament(2, a, fit, rng=rng), tournament(2, a, fit, rng=rng)
            off1 = self.evaluate(de(pr, decs(self.P1), decs(self.P1[m1]), decs(self.P1[m2]), rng=rng))
        fit5 = cal_fitness(objs(self.Ar), _con(self.Ar), self.priority, nP, 1)
        if self.type == 1:
            off4 = self.evaluate(ga(pr, decs(self.Ar[tournament(2, N, fit5, rng=rng)]), rng=rng))
            merged = Population.merge(self.P1, off1)
            self.P1, _ = (env_selection if self.handling == 0 else env_selection1)(merged, POP, self.priority, self.current, 0)
        else:
            m1, m2 = tournament(2, POP, fit5, rng=rng), tournament(2, POP, fit5, rng=rng)
            off4 = self.evaluate(de(pr, decs(self.Ar), decs(self.Ar[m1]), decs(self.Ar[m2]), rng=rng))
            merged = Population.merge(self.P1, off1)
            self.P1, _ = (env_selection if self.current == 0 else env_selection1)(merged, POP, self.priority, self.current, 0)
        self.Ar, _ = env_selection1(Population.merge(self.Ar, off4, off1), N, self.priority, nP, 1)
        self.pop = self.Ar
