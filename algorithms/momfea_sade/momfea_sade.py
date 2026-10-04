# emopylab 2026
"""MOMFEA-SADE (multi-objective multifactorial evolutionary algorithm with subspace alignment and adaptive differential evolution).

Reference:
Z. Liang, H. Dong, C. Liu, W. Liang, and Z. Zhu. Evolutionary multitasking for multiobjective
optimization with subspace alignment and adaptive differential evolution. IEEE Transactions on
Cybernetics, 2022, 52(4): 2096-2109.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'MOMFEASADE': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def domain_adaption(T, O, x):
    """Subspace alignment of the population ``O`` onto the population ``T`` (PCA bases, half of the dimension)."""
    dim = O.shape[1]
    k = max(int(np.floor(dim * 0.5)), 1)

    def pca_basis(A):
        C = A - A.mean(axis=0)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        return Vt[:k].T

    ct, co = pca_basis(T), pca_basis(O)
    ot = np.linalg.svd(ct, full_matrices=False)[0][:, :k]
    oo = np.linalg.svd(co, full_matrices=False)[0][:, :k]
    Xb = ot @ oo.T @ ot
    rec = (O @ Xb) @ ct.T
    with np.errstate(all="ignore"):
        return (rec[x] - rec.min()) / (rec.max() - rec.min())


def _rank(pop):
    front, _ = nd_sort(objs(pop), None, np.inf)
    cd = crowding(objs(pop), front)
    return np.lexsort((-cd, front))


class MOMFEASADE(LoopAlgorithm):
    """Multiobjective multifactorial EA with self-adaptive DE operator selection and subspace-alignment knowledge
    transfer between tasks: each task evolves a population of ``N/2`` solutions in the unified [0,1] space."""

    def __init__(self, pop_size: int = 100, RMP: float = 1, LP: int = 30, F1: float = 0.6, F2: float = 0.5, LCR: float = 0.3,
                 UCR: float = 0.9, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.RMP, self.LP, self.F1, self.F2, self.LCR, self.UCR = float(RMP), int(LP), float(F1), float(F2), float(LCR), float(UCR)

    def initial_size(self):
        return 0

    def _initialize_infill(self):
        pr, rng = self.problem, self.rng
        self.T = len(getattr(pr, "sub_m", [pr.n_obj]))
        self.ProbN = self.N // 2
        self.sub = []
        for t in range(self.T):
            dec = rng.random((self.ProbN, max(getattr(pr, "sub_d", [pr.n_var]))))
            if self.T > 1:
                dec = np.hstack([dec, np.full((self.ProbN, 1), t + 1.0)])
            self.sub.append(self.evaluate(dec, child=np.zeros(self.ProbN), de=np.zeros(self.ProbN)))
        self.gen, self.R_used, self.R_succ = 0, [], []
        return Population.merge(*self.sub)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    @staticmethod
    def _de_crossover(off, par, CR, rng):
        rep = rng.random(len(off)) > CR
        rep[int(rng.integers(len(off)))] = False
        off = off.copy()
        off[rep] = par[rep]
        return off

    def _generation(self, best, pool, t):
        rng = self.rng
        sub = self.sub
        n = len(sub[t])
        T = self.T
        others = [k for k in range(T) if k != t]
        has_task_id = T > 1
        pd = [decs(s)[:, :-1] if has_task_id else decs(s) for s in sub]
        adapted = []
        if len(others) > 0:
            for _ in range(3):
                k = others[int(rng.integers(len(others)))]
                a = domain_adaption(pd[t], pd[k], rng.integers(0, n, n))
                adapted.append(np.hstack([a, np.full((n, 1), t + 1.0)]) if has_task_id else a)
        else:
            adapted = [pd[t]] * 3
        Xt = decs(sub[t])
        rows = []
        for i in range(n):
            A = rng.permutation(n)[:4]
            A = A[A != i]
            x1, x2, x3 = A[0], A[1], A[2]
            CR = self.LCR + rng.random() * (self.UCR - self.LCR)
            cur = Xt[i]
            if rng.random() < self.RMP:
                a1, a2, a3 = adapted[0][i], adapted[1][i], adapted[2][i]
            else:
                a1, a2, a3 = Xt[x1], Xt[x2], Xt[x3]
            op = pool[i]
            if op == 1:
                base = Xt[best[t][int(rng.integers(len(best[t])))]]
                off = self._de_crossover(base + self.F1 * (a1 - a2), cur, CR, rng)
            elif op == 2:
                off = self._de_crossover(a1 + self.F1 * (a2 - a3), cur, CR, rng)
            else:
                off = cur + self.F2 * (a1 - cur) + self.F1 * (a2 - a3)
            off = np.clip(off, 0, 1)
            if has_task_id:
                off[-1] = t + 1
            rows.append(off)
        return self.evaluate(np.array(rows), child=np.ones(n), de=np.asarray(pool, dtype=float))

    def step(self):
        rng = self.rng
        self.gen += 1
        ST = 3
        if self.gen <= self.LP:
            pro = np.ones(ST)
        else:
            us = np.sum(self.R_used[self.gen - self.LP - 1:self.gen - 1], axis=0) + self.LP
            sc = np.sum(self.R_succ[self.gen - self.LP - 1:self.gen - 1], axis=0) + self.LP
            pro = sc / us
        best = []
        for s in self.sub:
            front, _ = nd_sort(objs(s), None, np.inf)
            best.append(np.where(front == 1)[0])
        pools = []
        for t in range(self.T):
            n = len(self.sub[t])
            roul = np.cumsum(pro / pro.sum())
            pool = [int(np.searchsorted(roul, rng.random(), side="left")) + 1 for _ in range(n)]
            pool = [min(p, ST) for p in pool]
            pools.append(pool)
            for ind in self.sub[t]:
                ind.set("child", 0.0)
            off = self._generation(best, pool, t)
            merged = Population.merge(self.sub[t], off)
            self.sub[t] = merged[_rank(merged)[: self.ProbN]]
        self.R_used.append(np.bincount(np.concatenate(pools), minlength=ST + 1)[1:ST + 1])
        pop_all = Population.merge(*self.sub)
        flags = np.array([ind.get("child") or 0.0 for ind in pop_all])
        succ = np.array([ind.get("de") for ind in pop_all], dtype=float)[flags == 1]
        self.R_succ.append(np.bincount(succ.astype(int), minlength=ST + 1)[1:ST + 1])
        self.pop = pop_all
