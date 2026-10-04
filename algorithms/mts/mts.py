# emopylab 2026
"""MTS (multiple trajectory search).

Reference:
L. Y. Tseng and C. Chen. Multiple trajectory search for unconstrained / constrained multi-objective
optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2009, 1951-1958.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'MTS': {'integer', 'multi', 'real'}}


def adjust_app_set(app, N, rng):
    F = objs(app)
    choose = np.zeros(len(app), bool)
    for i in range(F.shape[1]):
        choose[F[:, i] == F[:, i].min()] = True
    if choose.sum() > N:
        sel = np.where(choose)[0]
        choose = np.zeros(len(app), bool)
        choose[sel[rng.permutation(len(sel))[:N]]] = True
    else:
        d = pdist2(F, F)
        np.fill_diagonal(d, np.inf)
        while choose.sum() < N and not choose.all():
            un = np.where(~choose)[0]
            x = int(np.argmax(d[np.ix_(un, np.where(choose)[0])].min(axis=1)))
            choose[un[x]] = True
    return app[choose]


class MTS(LoopAlgorithm):
    """Multiple trajectory search: every solution runs a trajectory of coordinate-wise (LS1), random-subset (LS2) and
    grid (LS3) local searches; the local search that scores best in a short test phase is repeated, and only the
    solutions with the best grades stay active in the next round.  Non-dominated solutions found on the way are
    kept in an approximation set, which is the result."""

    def __init__(self, pop_size: int = 100, popsize: int = 40, ofLocalSearchTest: int = 5, ofLocalSearch: int = 45,
                 ofForeground: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.ps, self.n_test, self.n_ls, self.n_fg = int(popsize), int(ofLocalSearchTest), int(ofLocalSearch), int(ofForeground)

    def _initialize_infill(self):
        rng, ps, D = self.rng, self.ps, self.D
        soa = np.argsort(rng.random((ps, D)), axis=0) + 1
        dec = soa / ps * (self.upper - self.lower) + self.lower
        return self.evaluate(dec)

    def _initialize_advance(self, infills=None, **kwargs):
        self.P = [infills[i] for i in range(len(infills))]
        self.app = infills
        ps = self.ps
        self.enable = np.ones(ps, bool)
        self.improve = np.ones(ps, bool)
        self.grade = np.zeros(ps)
        self.SR = np.tile((self.upper - self.lower) / 2, (ps, 1))
        self.pop = self.app
        self._set_optimum()

    def _eval1(self, dec):
        return self.evaluate(dec[None, :])[0]

    def _grading(self, X, oldX, grade, improve):
        fx = np.asarray(X.F, float)
        if self.app is None or len(self.app) == 0:
            self.app = Population.create([X])
            flag = False
        else:
            cmp = objs(self.app) - fx
            if np.any(np.all(cmp <= 0, axis=1)):
                flag = False
            else:
                self.app = self.app[~np.all(cmp >= 0, axis=1)]
                self.app = Population.merge(self.app, Population.create([X]))
                flag = True
        if len(self.app) > 10 * self.N:
            self.app = adjust_app_set(self.app, 5 * self.N, self.rng)
        if flag:
            grade += 9
        fo = np.asarray(oldX.F, float)
        if np.sum(fx < fo) > np.sum(fx > fo):
            grade += 2
            improve = True
        return grade, improve

    def _ls12(self, X, SR, improve, mode):
        rng, D = self.rng, self.D
        if not improve:
            SR = SR / 2
            if np.all(SR < 1e-8):
                SR = (self.upper - self.lower) * (rng.random(D) / 10 + 0.4)
        improve = False
        grade = 0
        steps = rng.permutation(D) if mode == 1 else range(D)
        for i in steps:
            old = X
            dec = np.asarray(X.X, float).copy()
            if mode == 1:
                chosen = np.zeros(D, bool)
                chosen[i] = True
                dec[i] += SR[i] * (rng.random() * 2 - 1)
            else:
                chosen = rng.random(D) < 1 / 4
                dec[chosen] += SR[chosen] * (rng.random(int(chosen.sum())) * 2 - 1)
            X = self._eval1(dec)
            grade, improve = self._grading(X, old, grade, improve)
            if np.all(np.asarray(old.F) <= np.asarray(X.F)):
                dec = np.asarray(old.X, float).copy()
                if mode == 1:
                    dec[i] -= 0.5 * SR[i] * (rng.random() * 2 - 1)
                else:
                    dec[chosen] -= 0.5 * SR[chosen] * (rng.random(int(chosen.sum())) * 2 - 1)
                X = self._eval1(dec)
                grade, improve = self._grading(X, old, grade, improve)
                if np.all(np.asarray(old.F) <= np.asarray(X.F)):
                    X = old
        return grade, X, SR, improve

    def _ls3(self, X, SR, improve):
        rng, D = self.rng, self.D
        lo, up = self.lower.copy(), self.upper.copy()
        L, U = lo.copy(), up.copy()
        disp = (U - L) / 10
        best = X
        while np.any(disp < 1e-2):
            for i in rng.permutation(D):
                values = np.arange(L[i], U[i] + 1e-12, disp[i])
                decs_ = np.tile(np.asarray(best.X, float), (len(values), 1))
                decs_[:, i] = values
                Y = self.evaluate(decs_)
                for y in Y:
                    self._grading(y, best, 0, improve)
                    if np.all(np.asarray(y.F) <= np.asarray(best.F)):
                        best = y
                L[i] = max(best.X[i] - 2 * disp[i], lo[i])
                U[i] = min(best.X[i] + 2 * disp[i], up[i])
                disp[i] = (U[i] - L[i]) / 10
        return 0, best, SR, improve

    def step(self):
        ps = self.ps
        searches = {1: lambda X, SR, imp: self._ls12(X, SR, imp, 1), 2: lambda X, SR, imp: self._ls12(X, SR, imp, 2), 3: self._ls3}
        for i in np.where(self.enable)[0]:
            self.grade[i] = 0
            test = np.zeros(3)
            for _ in range(self.n_test):
                for k in (1, 2, 3):
                    g, self.P[i], self.SR[i], self.improve[i] = searches[k](self.P[i], self.SR[i], self.improve[i])
                    test[k - 1] += g
            best = int(np.argmax(test)) + 1
            for _ in range(self.n_ls):
                g, self.P[i], self.SR[i], self.improve[i] = searches[best](self.P[i], self.SR[i], self.improve[i])
                self.grade[i] += g
        rank = np.argsort(-self.grade, kind="stable")
        self.enable[:] = False
        self.enable[rank[: self.n_fg]] = True
        if self.FE >= self.max_FE:
            self.app = adjust_app_set(self.app, self.N, self.rng)
        self.pop = self.app
