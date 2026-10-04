# emopylab 2026
"""MCCMO (multi-population coevolutionary constrained multi-objective optimization).

Reference:
J. Zou, R. Sun, Y. Liu, Y. Hu, S. Yang, J. Zheng, and K. Li. A multi-population evolutionary
algorithm using new cooperative mechanism for solving multi-objective problems with multi-
constraint. IEEE Transactions on Evolutionary Computation, 2024, 28(1): 267-280.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation, cal_fitness as _cal_fitness
from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, de, first_front, ga, ga_half, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MCCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _con(pop, n):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), max(n, 1)))


def _fitness(pop, total, processcon, eps=None):
    """CalFitness / CalFitness1 of the reference: the constraint subset ``processcon`` (list), 0 (ignore all
    constraints) or ``total + 1`` (all constraints)."""
    F = objs(pop)
    C = _con(pop, total)
    if eps is not None:
        return _cal_fitness(F, C[:, [int(p) - 1 for p in np.atleast_1d(processcon)]] if C.shape[1] else C, eps)
    if isinstance(processcon, (int, np.integer)):
        if processcon == 0:
            return _cal_fitness(F)
        if processcon > total:
            return _cal_fitness(F, C)
        return _cal_fitness(F, C[:, [processcon - 1]])
    idx = [p - 1 for p in processcon]
    return _cal_fitness(F, C[:, idx])


def env_selection(pop, N, processcon, total, eps=None):
    fit = _fitness(pop, total, processcon, eps)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(objs(pop)[nxt], int(nxt.sum()) - N)]] = False
    p, f = pop[nxt], fit[nxt]
    r = np.argsort(f, kind="stable")
    return p[r], f[r]


def _overall_cv(pop):
    C = cons(pop)
    return np.sum(np.abs(np.maximum(C, 0)), axis=1) if C.size else np.zeros(len(pop))


def _is_stable(vals, gen, pop, N, thr, M, is_nd):
    if is_nd:
        front, _ = nd_sort(objs(pop), None, len(pop))
        NC = int(np.sum(front == 1))
    else:
        NC = N
    if NC != N:
        return 0
    change = abs(vals.get(gen, 0.0) - vals.get(gen - 1, 0.0))
    chth = thr * abs(vals.get(gen, 0.0) / (N * M)) * 10 ** (M - 2)
    return 1 if change <= chth else 0


class _Sub:
    def __init__(self, pop, fit, con, ident):
        self.pop, self.fit, self.con, self.id = pop, fit, list(con), ident
        self.hist = {1: float(objs(pop).sum())}
        self.stable, self.first = 0, 1


class MCCMO(LoopAlgorithm):
    """Multi-constraint co-evolution: one helper population per constraint (plus an unconstrained one and the
    archive population).  Helper populations that stabilise are merged with the ones they dominate into a
    population handling the union of their constraints, until a single main population remains."""

    def start(self):
        N = self.N
        self.Pop0 = self.pop
        self.total = _con(self.pop, 0).shape[1] if cons(self.pop).size else 0
        total = self.total
        self.fit0 = _cal_fitness(objs(self.Pop0))
        self.obj0 = {1: float(objs(self.Pop0).sum())}
        self.subs = []
        off = [self.Pop0]
        for i in range(1, total + 1):
            p = self.evaluate(self.random_decs(N))
            self.subs.append(_Sub(p, _cal_fitness(objs(p)), [i], i))
            off.append(p)
        self.index = total + 1
        self.UPF = self.dUPF = 0
        self.reinit = 0
        self.gen, self.thr = 2, 1e-2
        self.arch, self.fit_arch = env_selection(Population.merge(*off), N, total + 1, total)
        name = type(self.problem).__name__
        self.de_branch = any(k in name for k in ("RWMOP", "LIRCMOP", "DOC", "DASCMOP"))
        self.pop = self.arch

    def _offspring(self):
        N, rng, pr = self.N, self.rng, self.problem
        subs, off = self.subs, []
        temp = None
        if self.de_branch:
            for s in subs:
                temp = None
                if len(subs) > 1 and not s.first:
                    mp = tournament(2, N, s.fit, rng=rng)
                    X = decs(s.pop)
                    temp = self.evaluate(de(pr, X[: N // 2], X[mp[: len(mp) // 2]][: N // 2], X[mp[len(mp) // 2:]][: N // 2], rng=rng))
                off.append(temp)
            if not self.UPF or not self.dUPF:
                mp = tournament(2, N, self.fit0, rng=rng)
                X = decs(self.Pop0)
                temp = self.evaluate(de(pr, X[: N // 2], X[mp[: len(mp) // 2]][: N // 2], X[mp[len(mp) // 2:]][: N // 2], rng=rng))
                off.append(temp)
            elif self.dUPF and self.reinit:
                m1 = tournament(2, N // 2, self.fit_arch, rng=rng)
                m2 = tournament(2, N // 2, self.fit0, rng=rng)
                X = decs(self.Pop0)
                temp = self.evaluate(de(pr, X[: N // 2], X[m2], decs(self.arch)[m1], rng=rng))
                off.append(temp)
            mp = tournament(2, 2 * N, self.fit_arch, rng=rng)
            A = decs(self.arch)
            off.append(self.evaluate(de(pr, A, A[mp[: len(mp) // 2]], A[mp[len(mp) // 2:]], rng=rng)))
        else:
            temp = None
            for s in subs:
                mp = tournament(2, N, s.fit, rng=rng)
                if len(subs) > 1 and not s.first:
                    temp = self.evaluate(ga_half(pr, decs(s.pop[mp]), rng=rng))
                off.append(temp)
            if not self.UPF or not self.dUPF:
                mp = tournament(2, N, self.fit0, rng=rng)
                temp = self.evaluate(ga_half(pr, decs(self.Pop0[mp]), rng=rng))
                off.append(temp)
            elif self.dUPF and self.reinit:
                m1 = tournament(2, N // 2, self.fit_arch, rng=rng)
                m2 = tournament(2, N // 2, self.fit0, rng=rng)
                temp = self.evaluate(ga_half(pr, np.vstack([decs(self.Pop0[m2]), decs(self.arch[m1])]), rng=rng))
                off.append(temp)
            mp = tournament(2, N, self.fit_arch, rng=rng)
            off.append(self.evaluate(ga(pr, decs(self.arch[mp]), rng=rng)))
        off = [o for o in off if o is not None]
        return Population.merge(*off) if len(off) > 1 else off[0]

    def step(self):
        N, M, total, gen, subs = self.N, self.M, self.total, self.gen, self.subs
        off = self._offspring()
        self.arch, self.fit_arch = env_selection(Population.merge(off, self.arch), N, total + 1, total)
        self.Pop0, self.fit0 = env_selection(Population.merge(self.Pop0, off), N, 0, total)
        self.obj0[gen] = float(np.abs(objs(self.Pop0)).sum())
        if len(subs) > 1:
            for s in subs:
                s.pop, s.fit = env_selection(Population.merge(s.pop, off), N, s.con, total, 0)
                s.hist[gen] = float(objs(s.pop).sum())
                s.stable = _is_stable(s.hist, gen, s.pop, N, self.thr, M, 0 if s.first else 1)
                if s.stable and s.first:
                    s.first, s.stable = 0, 0
        if self.UPF == 0:
            self.UPF = _is_stable(self.obj0, gen, self.Pop0, N, self.thr, M, 0)
        else:
            self.dUPF = _is_stable(self.obj0, gen, self.Pop0, N, self.thr, M, 0)
        if self.dUPF:
            self.reinit = 1 if np.sum(_overall_cv(self.Pop0) <= 0) / N > 0 else 0
        if len(subs) > 1:
            self._merge(gen)
        self.gen += 1
        self.pop = self.arch

    def _merge(self, gen):
        N, total, subs = self.N, self.total, self.subs
        allp = Population.merge(*[s.pop for s in subs])
        both = self.UPF and self.dUPF
        if both:
            allp = Population.merge(allp, self.arch)
        front, _ = nd_sort(objs(allp), None, np.inf)
        blk = lambda j: front[j * N:(j + 1) * N]
        if both:
            mn = front[len(subs) * N:].min()
            for j, s in enumerate(subs):
                if blk(j).max() < mn:
                    s.stable = 1
        i = 0
        while i < len(subs):
            if subs[i].stable:
                mn = blk(i).min()
                for j in range(len(subs)):
                    if i != j and blk(j).max() < mn:
                        subs[j].stable = 1
            i += 1
        merge_idx, merge_pops, merge_con = [], [], []
        for p, s in enumerate(subs):
            if s.stable == 1:
                conp = np.maximum(0, _con(s.pop, total)[:, [c - 1 for c in s.con]])
                if np.any(conp <= 0):
                    merge_pops.append(s.pop)
                    merge_idx.append(p)
                    merge_con += s.con
                else:
                    s.pop = self.evaluate(self.random_decs(N))
                    s.fit = _fitness(s.pop, total, s.con)
        if len(merge_idx) > 1:
            sel = subs[merge_idx[0]]
            sel.con = merge_con
            sel.pop, sel.fit = env_selection(Population.merge(allp, self.arch), N, sel.con, total)
            sel.stable = 0
            sel.hist[gen] = float(objs(sel.pop).sum())
            sel.first = 0
            sel.id = self.index
            self.index += 1
            self.subs = [s for s in subs if s.stable != 1]
        if len(self.subs) == 1:
            self.arch, self.fit_arch = env_selection(Population.merge(self.arch, self.subs[0].pop), N, total + 1, total)
