# emopylab 2026
"""CSEMT (constraints separation based evolutionary multitasking).

Reference:
K. Qiao, J. Liang, K. Yu, X. Ban, C. Yue, B. Qu, and P. N. Suganthan. Constraints separation based
evolutionary multitasking for constrained multi-objective optimization problems. IEEE/CAA Journal of
Automatica Sinica, 2024, 11(8): 1819-1835.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'CSEMT': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _con(pop, n_cons):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), n_cons))


def ical_fitness(F, C, idx):
    N = len(F)
    CV = np.zeros(N) if len(idx) == 0 else np.sum(np.maximum(0, C[:, idx]), axis=1)
    k = (F[:, None, :] < F[None, :, :]).any(axis=2).astype(int) - (F[:, None, :] > F[None, :, :]).any(axis=2).astype(int)
    Dom = (CV[:, None] < CV[None, :]) | ((CV[:, None] == CV[None, :]) & (k == 1))
    R = Dom.sum(axis=1) @ Dom
    d = np.sqrt(np.maximum(np.sum((F[:, None, :] - F[None, :, :]) ** 2, axis=2), 0))
    np.fill_diagonal(d, np.inf)
    D = 1.0 / (np.sort(d, axis=1)[:, max(int(np.floor(np.sqrt(N))) - 1, 0)] + 2)
    return R + D


def env_selection(pop, N, idx, n_cons):
    F = objs(pop)
    fit = ical_fitness(F, _con(pop, n_cons), idx)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        ii = np.where(nxt)[0]
        nxt[ii[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    mask = nxt.copy()
    p, f = pop[nxt], fit[nxt]
    r = np.argsort(f, kind="stable")
    return p[r], f[r], mask


def _front_one(pop, target, n_cons):
    C = np.maximum(0, _con(pop, n_cons))
    sub = C[:, target] if len(target) else np.zeros((len(pop), 0))
    front, _ = nd_sort(objs(pop), sub if sub.size else None, np.inf)
    return np.where(front == 1)[0]


def _alphas(t1, t2, target1, target2, n_cons):
    x1, x2 = _front_one(t1, target1, n_cons), _front_one(t2, target2, n_cons)
    o1, o2 = objs(t1)[x1], objs(t2)[x2]
    front, _ = nd_sort(np.vstack([o1, o2]), None, np.inf)
    a1 = np.sum(front[: len(x1)] == 1) / len(x1)
    a2 = np.sum(front[len(x1):] == 1) / len(x2)
    return a1, a2


def relationship(t1, t2, c1, c2, target1, target2, beta, n_cons):
    a1, a2 = _alphas(t1, t2, target1, target2, n_cons)
    if a1 >= beta and a2 < beta:
        return 0
    if a1 < beta and a2 >= beta:
        return 1
    if a1 < beta and a2 < beta:
        return 2
    if c1 == 1:
        return 1
    if c2 == 1:
        return 0
    return 1 if a1 >= a2 else 0


def relationship2(t1, t2, target1, target2, beta, n_cons):
    a1, a2 = _alphas(t1, t2, target1, target2, n_cons)
    if a1 >= beta and a2 < beta:
        return 0
    if a1 < beta and a2 >= beta:
        return 1
    if a1 < beta and a2 < beta:
        return 2
    return 3


class CSEMT(LoopAlgorithm):
    """Constraint-splitting evolutionary multitasking: the main task keeps all constraints while one auxiliary task
    per constraint (plus an unconstrained one) evolves alongside it; once the unconstrained task settles, the
    relations between the tasks decide which auxiliary tasks are kept and how offspring are transferred."""

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.nc = max(_con(self.P1, 0).shape[1], 0)
        nc = self.nc
        self.allc = list(range(nc))
        self.f1 = ical_fitness(objs(self.P1), _con(self.P1, nc), self.allc)
        self.target = [[i] for i in range(nc)]
        self.archive, self.ar_fit = [], []
        for i in range(nc):
            a = self.evaluate(self.random_decs(N))
            self.archive.append(a)
            self.ar_fit.append(ical_fitness(objs(a), _con(a, nc), self.target[i]))
        self.target.append([])
        a = self.evaluate(self.random_decs(N))
        self.archive.append(a)
        last = [nc - 1] if nc else []
        self.ar_fit.append(ical_fitness(objs(a), _con(a, nc), last))
        self.max_tasks = nc + 1
        self.gen, self.last_gen, self.thr = 0, 100, 0.5
        self.change = {}
        self.cnt, self.flag, self.beta = 0, 0, 0.8
        self.is_flag_ar = None
        self.succ = {}

    def _converged(self):
        G = self.cnt
        if G - self.gen > self.last_gen:
            prev = self.change.get(G - self.last_gen, np.zeros(self.M))
            return bool(np.max(np.abs(self.change[G] - prev)) <= self.thr)
        return False

    def _switch_stage(self):
        nc, mt, beta = self.nc, self.max_tasks, self.beta
        isf = np.zeros((mt, mt), dtype=int)
        for i in range(mt - 1):
            for j in range(i + 1, mt):
                c1, c2 = int(i == mt - 1), int(j == mt - 1)
                v = relationship(self.archive[i], self.archive[j], c1, c2, self.target[i], self.target[j], beta, nc)
                isf[i, j] = v
                if v == 2:
                    isf[j, i] = 2
                elif v == 1:
                    isf[j, i] = 0
                elif v == 0:
                    isf[j, i] = 1
        sd = []
        for i in range(mt):
            if not any(isf[i, j] == 0 for j in range(mt) if j != i):
                sd.append(i)
        self.is_flag_ar = np.array([relationship2(self.archive[s], self.P1, self.target[s], self.allc, beta, nc) for s in sd])
        self.archive = [self.archive[s] for s in sd]
        self.ar_fit = [self.ar_fit[s] for s in sd]
        self.target = [self.target[s] for s in sd]
        self.max_tasks = len(sd)

    def step(self):
        N, rng, pr, nc = self.N, self.rng, self.problem, self.nc
        self.cnt += 1
        if self.flag == 0:
            F = objs(self.archive[self.max_tasks - 1])
            with np.errstate(all="ignore"):
                self.change[self.cnt] = np.mean((F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0)), axis=0)
            if self._converged():
                self.flag = 1
                self._switch_stage()
        off1 = self.evaluate(ga_half(pr, decs(self.P1[tournament(2, N, self.f1, rng=rng)]), rng=rng))
        main = [off1]
        arch_off = []
        succ = {}
        for i in range(self.max_tasks):
            mp = tournament(2, N, self.ar_fit[i], rng=rng)
            ao = self.evaluate(ga_half(pr, decs(self.archive[i][mp]), rng=rng))
            arch_off.append(ao)
            if self.flag == 0:
                main.append(ao)
            else:
                _, _, nxt = env_selection(Population.merge(self.archive[i], ao), N, self.allc, nc)
                la = len(self.archive[i])
                succ[i, 0] = nxt[:la].sum() / la - nxt[la:].sum() / len(ao)
                if self.is_flag_ar[i] != 0:
                    if succ[i, 0] > 0:
                        main.append(self.archive[i][rng.permutation(N)[: N // 2]])
                    else:
                        main.append(ao)
                else:
                    main.append(self.archive[i][rng.permutation(N)[: N // 2]])
                    main.append(ao)
                _, _, nxt = env_selection(Population.merge(self.P1, off1), N, self.target[i], nc)
                l1 = len(self.P1)
                succ[i, 1] = nxt[:l1].sum() / l1 - nxt[l1:].sum() / len(off1)
        fu = []
        for i in range(self.max_tasks):
            parts = []
            if self.flag == 0:
                parts.append(off1)
            elif self.is_flag_ar[i] != 0:
                parts.append(self.P1[rng.permutation(N)[: N // 2]] if succ[i, 1] > 0 else off1)
            else:
                parts.append(self.P1[rng.permutation(N)[: N // 2]])
                parts.append(off1)
            parts.extend(arch_off)
            fu.append(Population.merge(*parts) if len(parts) > 1 else parts[0])
        self.P1, self.f1, _ = env_selection(Population.merge(self.P1, *main), N, self.allc, nc)
        for i in range(self.max_tasks):
            self.archive[i], self.ar_fit[i], _ = env_selection(Population.merge(self.archive[i], fu[i]), N, self.target[i], nc)
        self.pop = self.P1
