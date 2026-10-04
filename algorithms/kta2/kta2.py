# emopylab 2026
"""KTA2 (kriging-assisted Two_Arch2).

Reference:
Z. Song, H. Wang, C. He, and Y. Jin. A Kriging-assisted two-archive evolutionary algorithm for
expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6):
1013-1027.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs
from algorithms.community_utils.dace import DaceModel
from algorithms.community_utils.kta import adaptive_sampling, ibea_keep, k_update_ca, k_update_da, lp_greedy
from core.population import Population

ALGORITHM_FLAGS = {'KTA2': {'expensive', 'integer', 'many', 'multi', 'real'}}


def update_ca(CA, new, K):
    CA = new if CA is None else Population.merge(CA, new)
    return CA if len(CA) <= K else CA[ibea_keep(objs(CA), K)]


def update_da(DA, new, K, p, rng):
    DA = Population.merge(DA, new)
    f, _ = nd_sort(objs(DA), None, 1)
    DA = DA[f == 1]
    if len(DA) <= K:
        return DA
    F = objs(DA)
    ch = np.zeros(len(DA), bool)
    ch[np.argmin(F, 0)] = True
    ch[np.argmax(F, 0)] = True
    if ch.sum() > K:
        c = np.where(ch)[0]
        ch[c[rng.permutation(len(c))[: int(ch.sum()) - K]]] = False
    elif ch.sum() < K:
        ch, _ = lp_greedy(F, ch, K, p)
    return DA[ch]


class KTA2(LoopAlgorithm):
    """Two_Arch2 run on Kriging surrogates. Each objective has one global model and two influential-point-insensitive models
    (trained on the best and worst ``tau`` fractions); the prediction switches to the insensitive model whose centre is closer.
    The infill is chosen adaptively: convergence sampling from the convergence archive when it is significantly better (signed-
    rank test), uncertainty sampling when the predicted diversity archive is less diverse, and diversity sampling otherwise."""

    def __init__(self, pop_size: int = 100, tau: float = 0.75, phi: float = 0.1, wmax: int = 10, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.tau, self.phi, self.wmax, self.mu = float(tau), float(phi), int(wmax), int(mu)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        P, _ = UniformPoint(self.N, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.p = 1.0 / self.M
        self.CA = update_ca(None, infills, self.N)
        self.DA = infills
        self.th_s = 5.0 * np.ones((self.M, self.D))
        self.th_is = 5.0 * np.ones((2, self.M, self.D))
        self._set_optimum()

    def _dace(self, X, y, th):
        D = self.D
        m = DaceModel(X, y, "regpoly0", th, 1e-5 * np.ones(D), 100 * np.ones(D))
        return m

    def _mating(self, CAo, CAd, DAo, DAd):
        rng, N = self.rng, self.N
        h = int(np.ceil(N / 2))
        a, b = rng.integers(0, len(CAo), h), rng.integers(0, len(CAo), h)
        dom = (CAo[a] < CAo[b]).any(1).astype(int) - (CAo[a] > CAo[b]).any(1).astype(int)
        pc = np.vstack([CAd[np.concatenate([a[dom == 1], b[dom != 1]])], DAd[rng.integers(0, len(DAd), h)]])
        pm = CAd[rng.integers(0, len(CAd), N)]
        return pc, pm

    def step(self):
        rng, M = self.rng, self.M
        X, F = decs(self.pop), objs(self.pop)
        ms = []
        for i in range(M):
            m = self._dace(X, F[:, i], self.th_s[i])
            self.th_s[i] = m.theta
            ms.append(m)
        centers = np.zeros((M, 2))
        mis = [[None] * M for _ in range(2)]
        num = int(np.ceil(len(self.pop) * self.tau))
        for i in range(M):
            o = np.argsort(F[:, i], kind="stable")
            groups = (o[:num], o[len(o) - num - 1:])
            for j in range(2):
                centers[i, j] = F[groups[j], i].mean()
                m = self._dace(X[groups[j]], F[groups[j], i], self.th_is[j, i])
                self.th_is[j, i] = m.theta
                mis[j][i] = m
        CAo, CAd, DAo, DAd = objs(self.CA), decs(self.CA), objs(self.DA), decs(self.DA)
        DAv = np.zeros_like(DAo)
        for _ in range(self.wmax):
            pc, pm = self._mating(CAo, CAd, DAo, DAd)
            off = np.vstack([ga(self.problem, pc, (1, 20, 0, 0), rng=rng), ga(self.problem, pm, (0, 0, 1, 20), rng=rng)])
            PX = np.vstack([DAd, CAd, off])
            PF, PM = np.zeros((len(PX), M)), np.zeros((len(PX), M))
            for j in range(M):
                g = ms[j].predict(PX)
                near1 = np.abs(g - centers[j, 0]) <= np.abs(g - centers[j, 1])
                for k, sel in ((0, near1), (1, ~near1)):
                    if sel.any():
                        y, s = mis[k][j].predict(PX[sel], mse=True)
                        PF[sel, j], PM[sel, j] = y, s
            CAo, CAd, _ = k_update_ca(PF, PX, PM, self.N)
            DAo, DAd, DAv = k_update_da(PF, PX, PM, self.N, self.p, rng)
        cand = adaptive_sampling(CAo, DAo, CAd, DAd, DAv, objs(self.DA), decs(self.DA), self.mu, self.p, self.phi, rng)
        cand = np.unique(cand, axis=0) if len(cand) else cand
        allx = decs(self.pop)
        cand = np.array([c for c in cand if np.sqrt(((allx - c) ** 2).sum(1)).min() > 1e-5]).reshape(-1, self.D)
        if len(cand):
            off = self.evaluate(cand)
            for i in range(len(off)):
                if np.sqrt(((decs(self.pop) - decs(off)[i]) ** 2).sum(1)).min() > 1e-5:
                    self.pop = Population.merge(self.pop, off[[i]])
            self.CA = update_ca(self.CA, off, self.N)
            self.DA = update_da(self.DA, off, self.N, self.p, rng)
