# emopylab 2026
"""SFA-DE (scalarization function approximation based differential evolution algorithm).

Reference:
Y. Horaguchi, K. Nishihara, and M. Nakata. Evolutionary multiobjective optimization assisted by
scalarization function approximation for high-dimensional expensive problems. Swarm and Evolutionary
Computation, 2024, 86: 101516.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, neighbors_of, objs, pdist2, uniform_point
from algorithms.community_utils.surrogates import RBFExact
from core.population import Population

ALGORITHM_FLAGS = {'SFADE': {'expensive', 'integer', 'many', 'multi', 'real'}}


class SFADE(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, F: float = 0.5, CR: float = 0.9, omega: int = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.F, self.CR, self.omega = float(F), float(CR), int(omega)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def _initialize_infill(self):
        n = self.initial_size()
        w, _ = uniform_point(n, self.D, "Latin")
        X = (self.upper - self.lower) * w[:n] + self.lower if len(w) >= n else self.random_decs(n)
        return Population.new("X", self.cal_dec(X))

    def start(self):
        self.swarm = self.pop
        self.arc = self.pop
        self.Z = objs(self.pop).min(axis=0)
        self.pop = self.arc

    def _de_ctb1(self, X, best, r1, r2):
        N, D = X.shape
        rng = self.rng
        site = rng.random((N, D)) <= self.CR
        site[np.arange(N), rng.integers(0, D, size=N)] = True
        off = X.copy()
        off[site] = (X + self.F * (best - X) + self.F * (r1 - r2))[site]
        return np.fmax(np.fmin(off, self.upper), self.lower)

    def step(self):
        rng, T, N, D = self.rng, self.T, self.N, self.D
        for i in range(N):
            Xa = decs(self.arc)
            _, uid = np.unique(Xa, axis=0, return_index=True)
            uid = np.sort(uid)                                            # stable order
            au = self.arc[uid]
            tch = np.max(np.abs(objs(au) - self.Z) * self.W[i], axis=1)
            srt = np.argsort(tch, kind="stable")[:N]
            tr_x, tr_y = decs(au[srt]), tch[srt]
            spread = pdist2(tr_x, tr_x).max() * (D * N) ** (-1.0 / D)
            model = RBFExact(spread).fit(tr_x, tr_y)
            P = self.B[i][rng.permutation(T)]
            pde = decs(self.swarm[P])
            pobj = model.predict(pde)
            for _ in range(self.omega):
                b = int(np.argmin(pobj))
                idx = np.stack([rng.permutation(np.delete(np.arange(T), j))[:2] for j in range(T)])
                cand = self._de_ctb1(pde, np.tile(pde[b], (T, 1)), pde[idx[:, 0]], pde[idx[:, 1]])
                cobj = model.predict(cand)
                rep = pobj >= cobj
                pde[rep], pobj[rep] = cand[rep], cobj[rep]
            off = self.evaluate(pde[int(np.argmin(pobj))][None, :])
            fo = objs(off)[0]
            self.Z = np.minimum(self.Z, fo)
            g_old = np.max(np.abs(objs(self.swarm[P]) - self.Z) * self.W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z) * self.W[P], axis=1)
            self.swarm[P[g_old >= g_new]] = off[0]
            self.arc = Population.merge(self.arc, off)
        self.pop = self.arc
