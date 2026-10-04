# emopylab 2026
"""LDS-AF (low-dimensional surrogate aggregation function).

Reference:
H. Gu, H. Wang, C. He, B. Yuan, and Y. Jin. Large-scale multiobjective evolutionary algorithm guided
by low-dimensional surrogates of scalarization functions. Evolutionary Computation, 2025, 33(3):
309-334.
"""

from __future__ import annotations

from itertools import combinations, islice

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, neighbors_of, objs, pdist2, uniform_point
from algorithms.community_utils.surrogates import RBFExact
from core.population import Population

ALGORITHM_FLAGS = {'LDSAF': {'expensive', 'integer', 'many', 'multi', 'real'}}


class LDSAF(LoopAlgorithm):
    """MOEA/D driven by one exact-RBF surrogate per subproblem built on a PCA-reduced (D/2 dimensional)
    decision space; the candidate with the best predicted scalarised value is evaluated."""

    def __init__(self, pop_size: int = 100, delta: float = 0.9, N_s: int = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.N_s = float(delta), int(N_s)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def _initialize_infill(self):
        n = self.initial_size()
        w, _ = uniform_point(n, self.D, "Latin")
        return Population.new("X", self.cal_dec((self.upper - self.lower) * w[:n] + self.lower))

    def start(self):
        self.swarm = self.pop
        self.A = self.pop
        self.Z = objs(self.pop).min(axis=0)
        self.span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        self.pop = self.A

    def _model(self, i, center_w):
        X = (decs(self.A) - self.lower) / self.span
        g = np.max(np.abs(objs(self.A) - self.Z) * center_w, axis=1)
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov)
        evals, evecs = np.linalg.eigh(cov)
        vec = evecs[:, np.argsort(-evals)]
        d = max(1, int(np.floor(self.D / 2)))
        train = (X @ vec)[:, :d]
        spread = pdist2(train, train).max() * (self.D * self.N) ** (-1.0 / self.D)
        return vec, d, RBFExact(spread).fit(train, g)

    def _predict(self, model, Xc):
        vec, d, rbf = model
        Xn = ((Xc - self.lower) / self.span) @ vec
        return rbf.predict(Xn[:, :d])

    def step(self):
        rng, N, W = self.rng, self.N, self.W
        for i in range(N):
            if self.FE >= 350:
                pds = pdist2(W, W)[i, self.B[i]]
                far = np.where(pds >= pds.max() - 0.00001)[0]
                center = int(self.B[i][far[rng.permutation(len(far))[0]]])
            else:
                center = int(self.B[i][0])
            model = self._model(i, W[center])
            P = self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < self.delta else rng.permutation(N)
            pairs = list(islice(combinations(P, 2), self.N_s))
            pairs = np.asarray(pairs, dtype=int)
            pairs[self.N_s // 2:] = pairs[self.N_s // 2:][:, ::-1]
            base = decs(self.swarm[i:i + 1])
            cands = np.vstack([de(self.problem, base, decs(self.swarm[pairs[r, :1]]), decs(self.swarm[pairs[r, 1:]]), rng=rng)
                               for r in range(len(pairs))])
            y = cands[int(np.argmin(self._predict(model, cands)))]
            off = self.evaluate(y[None, :])
            fo = objs(off)[0]
            self.Z = np.minimum(self.Z, fo)
            nr = 2 if self.FE <= 2 * N else 4
            g_old = np.max(np.abs(objs(self.swarm[P]) - self.Z) * W[P], axis=1)
            g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
            self.swarm[P[np.where(g_old >= g_new)[0][:nr]]] = off[0]
            self.A = Population.merge(self.A, off)
        self.pop = self.A
