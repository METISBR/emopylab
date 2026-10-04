# emopylab 2026
"""MGSAEA (multigranularity surrogate-assisted constrained evolutionary algorithm).

Reference:
Y. Zhang, H. Jiang, Y. Tian, H. Ma, and X. Zhang. Multigranularity surrogate modeling for
evolutionary multiobjective optimization with expensive constraints. IEEE Transactions on Neural
Networks and Learning Systems, 2024, 35(3): 2956-2968.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, objs, tournament
from algorithms.community_utils.dace import DaceModel
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'MGSAEA': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


def _fit(F, C=None):
    return spea.cal_fitness(F, None if C is None or np.size(C) == 0 else np.atleast_2d(C.T).T)


def _keep(F_trunc, fit, N):
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[spea.truncation(F_trunc[idx], int(nxt.sum()) - N)]] = False
    return nxt


def _first_unique(F):
    return np.unique(F, axis=0, return_index=True)[1]


def update_archive(A, N):
    A = A[_first_unique(objs(A))]
    if len(A) > N:
        A = A[_keep(objs(A), _fit(objs(A), cons(A)), N)]
    return A


def update_population(P, new, N, status=None):
    Fp, Fn = objs(P), objs(new)
    rows = [i for i in range(len(P)) if not np.any(np.all(Fn == Fp[i], 1))]
    rows = np.array(rows, dtype=int)
    rows = rows[_first_unique(Fp[rows])] if len(rows) else rows
    P = P[rows]
    F, C = objs(P), cons(P)
    if status is None:
        fit = _fit(F)
    elif status == 2:
        fit = _fit(np.column_stack([F, np.maximum(0, C).sum(1)]))
    else:
        fit = _fit(F, C)
    if len(P) > N:
        P = P[_keep(F, fit, N)]
    return Population.merge(P, new)


def _norm_cols(C):
    with np.errstate(all="ignore"):
        C = (C - C.min(0)) / (C.max(0) - C.min(0))
    C[:, np.isnan(C[0])] = 0.0 if len(C) else C[:, :0]
    return C


class MGSAEA(LoopAlgorithm):
    """First stage: Kriging-assisted SPEA2 on the objectives only, until the ideal point stops moving (relative change below
    ``lam`` over ``gap`` iterations). Second stage: the constraints are modelled either through one normalised violation
    (every constraint violated somewhere), through the individually violated constraints, or not at all (all satisfied),
    and the surrogate search ranks by constrained dominance accordingly."""

    def __init__(self, pop_size: int = 100, wmax: int = 20, mu: int = 5, gap: int = 20, lam: float = 1e-3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.mu, self.gap, self.lam = int(wmax), int(mu), int(gap), float(lam)

    def _initialize_infill(self):
        self.NI = 11 * self.D - 1
        return self.evaluate(self.lower + lhs_design(self.rng, self.NI, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.P = infills
        self.pop = update_archive(infills, self.N)
        nc = cons(infills).shape[1]
        self.th_obj, self.th_cv, self.th_con = 5.0 * np.ones((self.M, self.D)), 5.0 * np.ones(self.D), 5.0 * np.ones((nc, self.D))
        self.flag, self.it, self.ideal = 0, 1, []
        self._set_optimum()

    def _models(self, X, Y, thetas):
        D = self.D
        out = []
        for i in range(Y.shape[1]):
            m = DaceModel(X, Y[:, i], "regpoly0", thetas[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            thetas[i] = m.theta
            out.append(m)
        return out

    def _env(self, X, Y, N, status=None):
        M = self.M
        u = _first_unique(Y[:, :M])
        X, Y = X[u], Y[u]
        if status is None or status == 3:
            fit = _fit(Y)
            trunc = Y
        elif status == 1:
            fit = _fit(Y[:, :M], np.maximum(0, Y[:, -1])[:, None])
            trunc = Y[:, :M]
        else:
            fit = _fit(np.column_stack([Y[:, :M], np.maximum(0, Y[:, M:]).sum(1)]))
            trunc = Y[:, :M]
        nxt = _keep(trunc, fit, N)
        return X[nxt], Y[nxt], fit[nxt]

    def step(self):
        rng, NI, M = self.rng, self.NI, self.M
        P = self.P
        self.ideal.append(objs(P).min(0))
        if self.it > self.gap and self.flag == 0:
            a, b = self.ideal[-1], self.ideal[-1 - self.gap]
            if np.max(np.abs((a - b) / np.maximum(b, 1e-6))) <= self.lam:
                self.flag = 1
        X = decs(P)
        status = None
        if self.flag == 0:
            Y = objs(P)
            models = self._models(X, Y, self.th_obj)
            fit = _fit(Y)
        else:
            C = cons(P)
            mc = np.maximum(0, C).max(0) if C.shape[1] else np.zeros(0)
            ninf = int((mc > 0).sum())
            if ninf == C.shape[1]:
                status = 1
                Cn = np.maximum(0, C)
                cv = _norm_cols(Cn).sum(1) if C.shape[1] else np.zeros(len(P))
                Y = np.column_stack([objs(P), cv])
                th = np.vstack([self.th_obj, self.th_cv])
                models = self._models(X, Y, th)
                self.th_obj, self.th_cv = th[:M], th[-1]
                fit = _fit(objs(P), C)
            elif ninf > 0:
                status = 2
                idx = np.where(mc > 0)[0]
                Cn = _norm_cols(np.maximum(0, C)[:, idx])
                Y = np.column_stack([objs(P), Cn])
                th = np.vstack([self.th_obj, self.th_con[idx]])
                models = self._models(X, Y, th)
                self.th_obj, self.th_con[idx] = th[:M], th[M:]
                fit = _fit(np.column_stack([objs(P), np.maximum(0, C).sum(1)]))
            else:
                status = 3
                Y = objs(P)
                models = self._models(X, Y, self.th_obj)
                fit = _fit(Y)
        for _ in range(self.wmax):
            mate = tournament(2, NI, fit, rng=rng)
            X = np.vstack([X, ga(self.problem, X[mate], rng=rng)])
            Y = np.column_stack([m.predict(X) for m in models])
            X, Y, fit = self._env(X, Y, NI, status)
        new_dec, _, _ = self._env(X, Y, self.mu, status)
        new = self.evaluate(new_dec)
        self.P = update_population(P, new, NI - self.mu, status)
        self.pop = update_archive(Population.merge(self.pop, new), self.N)
        self.it += 1
