# emopylab 2026
"""SSCEA (subspace segmentation based co-evolutionary algorithm).

Reference:
G. Liu, Z. Pei, N. Liu, and Y. Tian. Subspace segmentation based co-evolutionary algorithm for
balancing convergence and diversity in many-objective optimization. Swarm and Evolutionary
Computation, 2023, 83: 101410.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, angle_matrix, decs, first_front, ga, objs
from algorithms.lmea.lmea import variable_clustering
from algorithms.two_arch2.two_arch2 import _update_ca
from core.population import Population

ALGORITHM_FLAGS = {'SSCEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _update_da(DA, new, max_size, rng):
    DA = new if DA is None else Population.merge(DA, new)
    DA = DA[first_front(objs(DA))]
    N = len(DA)
    if N <= max_size:
        return DA
    F = objs(DA)
    with np.errstate(all="ignore"):
        Fn = np.nan_to_num((F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0)))
    M = F.shape[1]
    choose = np.zeros(N, bool)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    for i in range(M):
        sub = F[~choose]
        # the reference implementation assigns the index found inside the *remaining* subset directly
        ext = int(np.argmin(np.max(sub / w[i], axis=1) + 0.1 * sub[:, i] / 1e-6))
        choose[ext] = True
    if choose.sum() > max_size:
        ch = np.where(choose)[0]
        choose[ch[rng.permutation(len(ch))[: int(choose.sum()) - max_size]]] = False
    elif choose.sum() < max_size:
        ang = angle_matrix(Fn)
        while choose.sum() < max_size:
            sel, rem = np.where(choose)[0], np.where(~choose)[0]
            choose[rem[int(np.argmax(ang[np.ix_(rem, sel)].min(axis=1)))]] = True
    return DA[choose]


class SSCEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, nSel: int = 5, nPer: int = 50, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.nSel, self.nPer = int(nSel), int(nPer)

    def _ca_size(self):
        lb, ub, beta = self.N / 10, self.N, 1.0
        return int(np.floor(lb + (ub - lb) * self.FE ** 2 * beta / self.max_FE ** 2))

    def start(self):
        self.CA = _update_ca(None, self.pop, self._ca_size())
        self.DA = _update_da(None, self.pop, self.N, self.rng)
        self.DV, self.CV = variable_clustering(self, self.pop, self.nSel, self.nPer)     # (position, distance) variables
        self.pop = self.DA

    def step(self):
        rng, N, D = self.rng, self.N, self.D
        h = int(np.ceil(N / 2))
        a, b = rng.integers(0, len(self.CA), size=h), rng.integers(0, len(self.CA), size=h)
        Fa, Fb = objs(self.CA[a]), objs(self.CA[b])
        dom = np.any(Fa < Fb, axis=1).astype(int) - np.any(Fa > Fb, axis=1).astype(int)
        parent_c = Population.merge(self.CA[np.concatenate([a[dom == 1], b[dom != 1]])], self.DA[rng.integers(0, len(self.DA), size=h)])
        parent_m = self.CA[rng.integers(0, len(self.CA), size=N)]
        off_dec = np.vstack([decs(parent_c), decs(parent_m)])
        vars_ = self.CV if (self.FE / self.max_FE < 0.5 or rng.random() < 0.5) else self.DV
        if len(vars_):
            new = np.vstack([ga(self.problem, decs(parent_c), (1, 15, 0, 0), rng=rng),
                             ga(self.problem, decs(parent_m), (0, 0, D / len(vars_) / 2, 15), rng=rng)])
            m = min(len(off_dec), len(new))
            off_dec[:m, vars_] = new[:m][:, vars_]
        off = self.evaluate(off_dec)
        self.CA = _update_ca(self.CA, off, self._ca_size())
        self.DA = _update_da(self.DA, off, N, rng)
        self.pop = self.DA
