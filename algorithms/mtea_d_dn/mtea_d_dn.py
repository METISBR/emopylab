# emopylab 2026
"""MTEA-D-DN (multiobjective multitask evolutionary algorithm based on decomposition with dual neighborhoods).

Reference:
X. Wang, Z. Dong, L. Tang, and Q. Zhang. Multiobjective multitask optimization-neighborhood as a
bridge for knowledge transfer. IEEE Transactions on Evolutionary Computation, 2023, 27(1): 155-169.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, de, objs, pdist2, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MTEADDN': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MTEADDN(LoopAlgorithm):
    """Multiobjective multitask decomposition with dual neighbourhoods: each subproblem mates inside its own
    neighbourhood or, with probability ``Beta``, with a neighbourhood of another task (knowledge transfer).

    The decision vector carries the task index in its last variable (tasks are 1-based)."""

    def __init__(self, pop_size: int = 100, Beta: float = 0.2, F: float = 0.5, CR: float = 0.9, MuM: float = 20,
                 sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Beta, self.F, self.CR, self.MuM = float(Beta), float(F), float(CR), float(MuM)

    def initial_size(self):
        return 0

    # -- helpers -------------------------------------------------------------
    def _pick_other(self, t, i):
        rng = self.rng
        pool = [k for k in range(self.T) if k != t]
        if len(pool) == 0:
            self.B2k[t][i] = t
            self.B2[(t, i)] = rng.permutation(len(self.W[t]))[: self.DT[t]]
            return
        k = pool[int(rng.integers(len(pool)))]
        self.B2k[t][i] = k
        self.B2[(t, i)] = rng.permutation(len(self.W[k]))[: self.DT[t]]

    def _initialize_infill(self):
        pr = self.problem
        sub_m, sub_d = getattr(pr, "sub_m", [pr.n_obj]), getattr(pr, "sub_d", [pr.n_var])
        self.T = len(sub_m)
        rng = self.rng
        self.W, self.n, self.DT, self.NB, self.sub, self.Z = [], [], [], [], [], []
        for t in range(self.T):
            W, n = uniform_point(self.N // 2, int(sub_m[t]))
            self.W.append(W)
            self.n.append(n)
            self.DT.append(int(np.ceil(n / 20)))
            self.NB.append(np.argsort(pdist2(W, W), axis=1, kind="stable")[:, : self.DT[t]])
            dec = rng.random((self.N // 2, max(sub_d)))
            if self.T > 1:
                dec = np.hstack([dec, np.full((self.N // 2, 1), t + 1.0)])
            pop = self.evaluate(dec)
            self.sub.append(pop)
            self.Z.append(objs(pop)[:, : int(sub_m[t])].min(axis=0))
        self.sub_m = [int(m) for m in sub_m]
        self.B2k = [np.zeros(self.n[t], dtype=int) for t in range(self.T)]
        self.B2 = {}
        for t in range(self.T):
            for i in range(len(self.NB[t])):
                self._pick_other(t, i)
        return Population.merge(*self.sub)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _F(self, t, idx):
        return objs(self.sub[t][idx])[:, : self.sub_m[t]]

    def _make(self, t, i, ta, pa, tb, pb, task):
        pr, rng = self.problem, self.rng
        p1 = decs(self.sub[t][[i]])
        p2 = decs(self.sub[ta][[int(pa)]])
        p3 = decs(self.sub[tb][[int(pb)]])
        dec = de(pr, p1, p2, p3, [self.CR, self.F, 1, self.MuM], rng=rng)
        if self.T > 1:
            dec[0, -1] = task + 1
        return dec

    def _replace(self, k, P, off_obj, off_ind):
        W = self.W[k]
        self.Z[k] = np.minimum(self.Z[k], off_obj)
        g_old = np.max(np.abs(self._F(k, P) - self.Z[k]) * W[P], axis=1)
        g_new = np.max(np.abs(off_obj - self.Z[k]) * W[P], axis=1)
        for j in np.where(g_old >= g_new)[0]:
            self.sub[k][int(P[j])] = off_ind
        return g_old, g_new

    def step(self):
        rng, T = self.rng, self.T
        for t in range(T):
            for i in range(self.n[t]):
                if rng.random() < self.Beta:
                    P1 = self.NB[t][i]
                    k = int(self.B2k[t][i])
                    P2 = self.B2[(t, i)]
                    tasks = np.concatenate([np.full(len(P1), t), np.full(len(P2), k)])
                    P = np.concatenate([P1, P2])
                    perm = rng.permutation(len(tasks))
                    tasks, P = tasks[perm], P[perm]
                    if rng.random() < 0.5:
                        dec = self._make(t, i, tasks[0], P[0], tasks[1], P[1], k)
                        off = self.evaluate(dec)
                        g_old, g_new = self._replace(k, P2, objs(off)[0, : self.sub_m[k]], off[0])
                        if np.all(g_old < g_new):
                            self._pick_other(t, i)
                        elif np.any(g_old >= g_new):
                            self.B2[(t, i)] = P2[g_old >= g_new]
                    else:
                        dec = self._make(t, i, tasks[0], P[0], tasks[1], P[1], t)
                        off = self.evaluate(dec)
                        self._replace(t, P1, objs(off)[0, : self.sub_m[t]], off[0])
                else:
                    P = rng.permutation(self.n[t])
                    dec = self._make(t, i, t, P[0], t, P[1], t)
                    off = self.evaluate(dec)
                    self._replace(t, P, objs(off)[0, : self.sub_m[t]], off[0])
        self.pop = Population.merge(*self.sub)
