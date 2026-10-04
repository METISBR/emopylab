# emopylab 2026
"""MOEA-DVA (multi-objective evolutionary algorithm based on decision variable).

Reference:
X. Ma, F. Liu, Y. Qi, X. Wang, L. Li, L. Jiao, M. Yin, and M. Gong. A multiobjective evolutionary
algorithm based on decision variable analyses for multiobjective optimization problems with large-
scale variables. IEEE Transactions Evolutionary Computation, 2016, 20(2): 275-298.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, nd_sort, objs
from operators.utility_functions._common import _good_lattice_point

ALGORITHM_FLAGS = {'MOEADVA': {'integer', 'large', 'multi', 'real'}}


class MOEADVA(LoopAlgorithm):
    """Decision-variable analysis (control-variable analysis + interaction analysis) followed by
    subcomponent-wise differential evolution.  The analysis evaluates solutions exactly as the reference
    implementation does and is therefore charged to the evaluation budget."""

    def __init__(self, pop_size: int = 100, NCA: int = 20, NIA: int = 6, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.NCA, self.NIA = int(NCA), int(NIA)

    def _control_variable_analysis(self):
        rng, D, lo, up = self.rng, self.D, self.lower, self.upper
        diver, conver = np.zeros(D, bool), np.zeros(D, bool)
        for i in range(D):
            x = rng.uniform(lo, up)
            S = np.tile(x, (self.NCA, 1))
            S[:, i] = ((np.arange(self.NCA) + rng.random(self.NCA)) / self.NCA) * (up[i] - lo[i]) + lo[i]
            pop = self.evaluate(S)
            _, max_f = nd_sort(objs(pop), None, np.inf)
            if max_f == len(pop):
                conver[i] = True
            else:
                diver[i] = True
        return diver, conver

    def _dividing_distance_variables(self, diver, conver):
        rng, D, N, lo, up = self.rng, self.D, self.N, self.lower, self.upper
        X = np.zeros((N, D))
        nd = int(diver.sum())
        if nd == 1:
            X[:, diver] = (np.arange(N) / (N - 1))[:, None]
        elif nd > 4:
            X[:, diver] = rng.random((N, nd))
        elif nd > 1:
            X[:, diver] = _good_lattice_point(N, nd)
        X[:, conver] = rng.random((N, int(conver.sum())))
        pop = self.evaluate(X * (up - lo) + lo)
        inter = np.eye(D, dtype=bool)
        for i in range(D - 1):
            for j in range(i + 1, D):
                for _ in range(self.NIA):
                    x = int(rng.integers(0, N))
                    a2 = rng.random() * (up[i] - lo[i]) + lo[i]
                    b2 = rng.random() * (up[j] - lo[j]) + lo[j]
                    base = decs(pop[x:x + 1])[0]
                    decs3 = np.tile(base, (3, 1))
                    decs3[0, i] = a2
                    decs3[1, j] = b2
                    decs3[2, [i, j]] = [a2, b2]
                    F = self.evaluate(decs3)
                    Fo, Fx = objs(F), objs(pop[x:x + 1])[0]
                    d1, d2 = Fo[0] - Fx, Fo[2] - Fo[1]
                    inter[i, j] = inter[i, j] or bool(np.any(d1 * d2 < 0))
                    inter[j, i] = inter[i, j]
                    if conver[j] and np.all(Fo[1] <= Fx):
                        pop[x] = F[1]
                    if conver[i] and np.all(Fo[0] <= objs(pop[x:x + 1])[0]):
                        pop[x] = F[0]
                    if conver[i] and conver[j] and np.all(Fo[2] <= objs(pop[x:x + 1])[0]):
                        pop[x] = F[2]
        subs, divided = [], np.zeros(D, bool)
        while not np.all(divided[conver]):
            x = np.array([int(np.where(~divided & conver)[0][0])])
            while np.sum(np.any(inter[np.ix_(x, np.where(conver)[0])], axis=0)) > len(x):
                x = np.where(np.any(inter[x], axis=0) & conver)[0]
            subs.append(x)
            divided[x] = True
        return subs, pop

    def _initialize_infill(self):
        # the analysis draws its own samples; the framework only needs a (cheap) placeholder evaluation of size 1
        from core.population import Population
        return Population.new("X", self.random_decs(1))

    def _initialize_advance(self, infills=None, **kwargs):
        self.diver, self.conver = self._control_variable_analysis()
        self.subs, self.pop = self._dividing_distance_variables(self.diver, self.conver)
        X = decs(self.pop)
        dis = np.linalg.norm(X[:, self.diver][:, None, :] - X[:, self.diver][None, :, :], axis=2)
        np.fill_diagonal(dis, np.inf)
        self.neigh = np.argsort(dis, axis=1, kind="stable")[:, : int(np.ceil(self.N / 10))]
        self._set_optimum()

    def step(self):
        rng, N = self.rng, self.N
        for idx in self.subs:
            for i in range(N):
                P = self.neigh[i][rng.permutation(self.neigh.shape[1])[:2]] if rng.random() < 0.9 else rng.permutation(N)[:2]
                base = decs(self.pop[i:i + 1])[0]
                new = de(self.problem, base[None, :], decs(self.pop[P[:1]]), decs(self.pop[P[1:2]]),
                         (1, 0.5, len(base) / len(idx) / 2, 20), rng=rng)[0]
                off = base.copy()
                off[idx] = new[idx]
                child = self.evaluate(off[None, :])
                if objs(child)[0].sum() < objs(self.pop[i:i + 1])[0].sum():
                    self.pop[i] = child[0]
