# emopylab 2026
"""MTDE-MKTA (multitasking differential evolution with multiple knowledge types and transfer adaptation).

Reference:
Y. Li and W. Gong. Multiobjective multitask optimization with multiple knowledge types and transfer
adaptation. IEEE Transactions on Evolutionary Computation, 2025, 29(1): 205-216.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import _truncation
from algorithms.community_utils.base import LoopAlgorithm, decs, objs
from algorithms.ccmo.ccmo import cal_fitness
from core.population import Population

ALGORITHM_FLAGS = {'MTDEMKTA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def spea2_selection(pop, N):
    F = objs(pop)
    fit = cal_fitness(F)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], fit[nxt]


def _add(pop, k):
    return np.array([ind.get(k) for ind in pop], dtype=float)


class MTDEMKTA(LoopAlgorithm):
    """Multitask DE with adaptive parameters (F, CR, transfer rate, knowledge-type proportion evolved per
    solution) and mixed knowledge transfer: either an individual of the helper task is copied directly or it is
    mapped through the population means / deviations of both tasks."""

    def __init__(self, pop_size: int = 100, Tau1: float = 0.2, Tau2: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.tau1, self.tau2 = float(Tau1), float(Tau2)

    def _initialize_infill(self):
        pr, rng = self.problem, self.rng
        self.T = len(getattr(pr, "sub_m", [pr.n_obj]))
        self.ProbN = self.N // 2
        self.sub, self.fit, self.model = [], [], []
        for t in range(self.T):
            n = self.ProbN
            dec = rng.random((n, max(getattr(pr, "sub_d", [pr.n_var]))))
            if self.T > 1:
                dec = np.hstack([dec, np.full((n, 1), t + 1.0)])
            pop = self.evaluate(dec, mF=0.2 + rng.random(n), mCR=rng.random(n), mTR=rng.random(n), mKP=rng.random(n))
            pop, fit = spea2_selection(pop, n)
            X = decs(pop)
            self.sub.append(pop)
            self.fit.append(fit)
            self.model.append({"mean": X.mean(axis=0), "std": X.std(axis=0, ddof=1) + 1e-100})
        return Population.merge(*self.sub)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _generation(self, rank, t):
        rng, sub, model, T = self.rng, self.sub, self.model, self.T
        n = len(sub[t])
        X = decs(sub[t])
        rows, params = [], []
        for i in range(n):
            F = float(np.clip(rng.normal(sub[t][i].get("mF"), 0.1), 0.2, 1.2))
            CR = float(np.clip(rng.normal(sub[t][i].get("mCR"), 0.1), 0, 1))
            TR = float(np.clip(rng.normal(sub[t][i].get("mTR"), 0.1), 0, 1))
            KP = float(rng.normal(sub[t][i].get("mKP"), 0.1))
            KP = 1 + KP if KP < 0 else (KP - 1 if KP > 1 else KP)
            if rng.random() < self.tau1:
                F = 0.2 + rng.random()
            if rng.random() < self.tau1:
                CR = rng.random()
            if rng.random() < self.tau2:
                TR = rng.random()
            if rng.random() < self.tau2:
                KP = rng.random()
            x1 = int(rng.integers(n))
            while rng.random() > (n - rank[t][x1]) / n or x1 == i:
                x1 = int(rng.integers(n))
            x2 = int(rng.integers(n))
            while rng.random() > (n - rank[t][x2]) / n or x2 == i or x2 == x1:
                x2 = int(rng.integers(n))
            x3 = int(rng.integers(n))
            while x3 == i or x3 == x1 or x3 == x2:
                x3 = int(rng.integers(n))
            xi, d1, d2, d3 = X[i], X[x1], X[x2], X[x3]
            if rng.random() < TR and T > 1:
                k = int(rng.integers(T))
                while k == t:
                    k = int(rng.integers(T))
                xk = decs(sub[k])[int(rng.integers(len(sub[k])))]
                if KP <= 0.5:
                    xk = model[t]["mean"] + model[t]["std"] * ((xk - model[k]["mean"]) / model[k]["std"])
                d2 = xk
            off = d1 + F * (d2 - d3)
            rep = rng.random(len(off)) > CR
            rep[int(rng.integers(len(off)))] = False
            off = np.where(rep, xi, off)
            off = np.clip(off, 0, 1)
            if T > 1:
                off[-1] = t + 1
            rows.append(off)
            params.append((F, CR, TR, KP))
        P = np.array(params)
        return self.evaluate(np.array(rows), mF=P[:, 0], mCR=P[:, 1], mTR=P[:, 2], mKP=P[:, 3])

    def step(self):
        T = self.T
        rank = []
        for t in range(T):
            r = np.argsort(np.argsort(self.fit[t], kind="stable"), kind="stable") + 1
            rank.append(r)
            X = decs(self.sub[t])
            a = 0.5
            self.model[t]["mean"] = a * self.model[t]["mean"] + (1 - a) * X.mean(axis=0)
            self.model[t]["std"] = a * self.model[t]["std"] + (1 - a) * X.std(axis=0, ddof=1) + 1e-100
        off = [self._generation(rank, t) for t in range(T)]
        for t in range(T):
            self.sub[t], self.fit[t] = spea2_selection(Population.merge(self.sub[t], off[t]), self.ProbN)
        self.pop = Population.merge(*self.sub)
