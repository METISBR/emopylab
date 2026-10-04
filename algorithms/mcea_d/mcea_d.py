# emopylab 2026
"""MCEA-D (multiple classifiers-assisted evolutionary algorithm based on decomposition).

Reference:
T. Sonoda and M. Nakata. Multiple classifiers-assisted evolutionary algorithm based on decomposition
for high-dimensional multi-objective problems. IEEE Transactions on Evolutionary Computation, 2022,
26(6): 1581-1595.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, neighbors_of, objs, uniform_point
from algorithms.community_utils.surrogates import SVMClassifier
from core.population import Population

ALGORITHM_FLAGS = {'MCEAD': {'expensive', 'integer', 'many', 'multi', 'real'}}


class MCEAD(LoopAlgorithm):
    """MOEA/D whose offspring are pre-screened by one SVM classifier per subproblem (RBF, C=1, gamma=1)
    trained on the archive: the neighbours' best archived solutions are the positive class."""

    def __init__(self, pop_size: int = 100, delta: float = 0.9, nr: int = 2, R_max: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.nr, self.R_max = float(delta), int(nr), int(R_max)

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
        self.pop = self.A
        self.span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)

    def _uniform(self, X):
        return (X - self.lower) / self.span

    def _model(self, i):
        FA, XA = objs(self.A), decs(self.A)
        label = -np.ones(len(FA))
        chosen = []
        for b in self.B[i]:
            g = np.max(np.abs(FA - self.Z) * self.W[b], axis=1)
            for j in np.argsort(g, kind="stable"):
                if j not in chosen:
                    chosen.append(int(j))
                    label[j] = 1
                    break
        return SVMClassifier(C=1.0, kernel_scale=np.sqrt(1.0 / 2.0)).fit(self._uniform(XA), label)

    def _solution_generation(self, i, P, svm):
        rng = self.rng
        best, best_score = None, -np.inf
        P = P.copy()
        for r in range(self.R_max):
            cand = de(self.problem, decs(self.swarm[i:i + 1]), decs(self.swarm[P[:1]]), decs(self.swarm[P[1:2]]), rng=rng)
            P = P[rng.permutation(len(P))]
            score = float(svm.decision_function(self._uniform(cand))[0])
            if score > 0:
                return cand
            if best is None or score > best_score:
                best, best_score = cand, score
        return best

    def step(self):
        rng, N = self.rng, self.N
        for i in range(N):
            svm = self._model(i)
            P = self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < self.delta else rng.permutation(N)
            y = self.evaluate(self._solution_generation(i, P, svm))
            fy = objs(y)[0]
            self.Z = np.minimum(self.Z, fy)
            g_old = np.max(np.abs(objs(self.swarm[P]) - self.Z) * self.W[P], axis=1)
            g_new = np.max(np.abs(fy - self.Z) * self.W[P], axis=1)
            self.swarm[P[np.where(g_old >= g_new)[0][: self.nr]]] = y[0]
            self.A = Population.merge(self.A, y)
        self.pop = self.A
