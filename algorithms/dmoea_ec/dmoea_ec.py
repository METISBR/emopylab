# emopylab 2026
"""DMOEA-eC (decomposition-based multi-objective evolutionary algorithm with the).

Reference:
J. Chen, J. Li, and B. Xin. DMOEA-eC: Decomposition-based multiobjective evolutionary algorithm with
the e-constraint framework. IEEE Transactions on Evolutionary Computation, 2017, 21(5): 714-730.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, first_front, ga_half, decs, neighbors_of, objs, pdist2, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'DMOEAeC': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _update_archive(arch, N, rng):
    arch = arch[first_front(objs(arch))]
    A = objs(arch)
    choose = np.zeros(len(arch), bool)
    choose[np.argmax(A, axis=0)] = True
    choose[np.argmin(A, axis=0)] = True
    if choose.sum() > N:
        sel = np.where(choose)[0]
        choose = np.zeros(len(arch), bool)
        choose[sel[rng.permutation(len(sel))[:N]]] = True
    else:
        dist = pdist2(A, A)
        np.fill_diagonal(dist, np.inf)
        while choose.sum() < N and not choose.all():
            un = np.where(~choose)[0]
            x = np.argmax(dist[np.ix_(~choose, choose)].min(axis=1))
            choose[un[x]] = True
    arch = arch[choose]
    return arch, objs(arch).max(axis=0)


class DMOEAeC(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, INm: float = 0.2, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.INm = float(INm)

    def initial_size(self):
        self.N_arch = self.pop_size
        self.W, n = uniform_point(self.pop_size, self.M - 1, "grid")
        self.pop_size = n
        self.T = int(np.ceil(n / 10))
        self.nr = int(np.ceil(n / 100))
        self.B = neighbors_of(self.W, self.T)
        return n

    def start(self):
        self.swarm = self.pop
        self.archive, self.znad = _update_archive(self.pop, self.N_arch, self.rng)
        self.z = objs(self.pop).min(axis=0)
        self.Pi = np.ones(self.N)
        self.gen = 0
        self.s = 0
        self.old_obj = None
        self.pop = self.archive

    def _main(self, F, s):
        others = [j for j in range(self.M) if j != s]
        return F[:, s] + 1e-6 * F[:, others].sum(axis=1)

    def step(self):
        rng, N, M, W = self.rng, self.N, self.M, self.W
        if self.gen % int(np.ceil(self.INm * self.max_FE / N)) == 0:
            s = int(rng.integers(0, M))
            self.s = s
            others = [j for j in range(M) if j != s]
            S = list(range(N))
            while S:
                Fn = (objs(self.swarm[np.asarray(S)]) - self.z) / (self.znad - self.z)
                l = int(rng.integers(0, len(S)))
                k = int(np.argmin(np.abs(Fn[:, others] - W[S[l]]).sum(axis=1)))
                a, b = S[l], S[k]
                tmp = self.swarm[a]
                self.swarm[a] = self.swarm[b]
                self.swarm[b] = tmp
                S.pop(l)
            self.old_obj = self._main(objs(self.swarm), s)
        s = self.s
        others = [j for j in range(M) if j != s]
        if self.gen % 10 == 0:
            new_obj = self._main(objs(self.swarm), s)
            with np.errstate(all="ignore"):
                delta = (self.old_obj - new_obj) / self.old_obj
            temp = delta <= 0.001
            self.Pi[~temp] = 1
            self.Pi[temp] = (0.95 + 0.05 * delta[temp] / 0.001) * self.Pi[temp]
            self.old_obj = new_obj
        for _ in range(5):
            boundary = np.where((np.sum(W == 1, axis=1) == 1) & (np.sum(W < 1e-3, axis=1) == W.shape[1] - 1))[0]
            I = np.concatenate([boundary, tournament(10, N // 5 - len(boundary), -self.Pi, rng=rng)]).astype(int)
            kids = []
            for i in I:
                P = self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < 0.9 else rng.permutation(N)
                child = self.evaluate(ga_half(self.problem, decs(self.swarm[P[:2]]), rng=rng))
                kids.append(child[0])
                fo = objs(child)[0]
                self.z = np.minimum(self.z, fo)
                with np.errstate(all="ignore"):
                    oo = (fo - self.z) / (self.znad - self.z)
                    cv = np.sum(np.maximum(0.0, oo[others] - W), axis=1)
                    zero = cv == 0
                    if zero.any():
                        cv[zero] = 1.0 / np.sum(oo[others] - W[zero], axis=1)
                k = int(np.argmin(cv))
                P = self.B[k][rng.permutation(self.B.shape[1])]
                Fp = objs(self.swarm[P])
                fmain_p = self._main(Fp, s)
                fmain_o = fo[s] + 1e-6 * fo[others].sum()
                with np.errstate(all="ignore"):
                    Pn = (Fp - self.z) / (self.znad - self.z)
                    On = (fo - self.z) / (self.znad - self.z)
                cvp = np.sum(np.maximum(0.0, Pn[:, others] - W[P]), axis=1)
                cvo = np.sum(np.maximum(0.0, On[others] - W[P]), axis=1)
                hit = np.where(((cvo == 0) & (cvp == 0) & (fmain_o < fmain_p)) | (cvo < cvp))[0][: self.nr]
                self.swarm[P[hit]] = child[0]
            self.archive, self.znad = _update_archive(Population.merge(self.archive, Population.create(kids)), self.N_arch, rng)
        self.gen += 1
        self.pop = self.archive
