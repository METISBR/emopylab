# emopylab 2026
"""DKCA (dynamic knowledge-guided coevolutionary algorithm).

Reference:
Y. Li, X. Feng, and H. Yu. A dynamic knowledge-guided coevolutionary algorithm for large-scale
sparse multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics:
Systems, 2024, 54(11): 7054-7064.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, ga_half, nd_sort, objs, tournament
from algorithms.sparseea.sparseea import _env_selection
from core.population import Population

ALGORITHM_FLAGS = {'DKCA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


def _ts(rng, f):
    return None if len(f) == 0 else int(tournament(2, 1, f, rng=rng)[0])


def _operator(algo, ParentDec, ParentMask, Fit):
    rng = algo.rng
    n, D = ParentDec.shape
    h = n // 2
    P1, P2 = ParentMask[:h], ParentMask[h:]
    Off = P1.copy()
    for i in range(h):
        if rng.random() < 0.5:
            idx = np.where((P1[i] != 0) & (P2[i] == 0))[0]
            k = _ts(rng, -Fit[idx])
            if k is not None:
                Off[i, idx[k]] = 0
        else:
            idx = np.where((P1[i] == 0) & (P2[i] != 0))[0]
            k = _ts(rng, Fit[idx])
            if k is not None:
                Off[i, idx[k]] = P2[i, idx[k]]
    for i in range(h):
        if rng.random() < 0.5:
            idx = np.where(Off[i] != 0)[0]
            k = _ts(rng, -Fit[idx])
            if k is not None:
                Off[i, idx[k]] = 0
        else:
            idx = np.where(Off[i] == 0)[0]
            k = _ts(rng, Fit[idx])
            if k is not None:
                Off[i, idx[k]] = 1
    return ga_half(algo.problem, ParentDec, rng=rng), Off


class DKCA(LoopAlgorithm):
    """Dimension-knowledge sparse EA: probing selects a promising variable subset; a second, low-dimensional mask
    population over that subset is evolved alongside the main one and the shared sparsity level lowers the score
    of the variables that keep appearing."""

    def __init__(self, pop_size: int = 100, t: float = 0.3, k: int = 4, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.t, self.k = float(t), int(k)

    def _dim_selection(self, Dec, Dec2):
        D = self.D
        Mask = np.eye(D)
        base = np.zeros((1, D))
        Ind = self.evaluate(np.vstack([base, Dec * Mask]))
        Ind2 = self.evaluate(np.vstack([base, Dec2 * Mask]))
        f1 = nd_sort(objs(Ind), None, np.inf)[0]
        f2 = nd_sort(objs(Ind2), None, np.inf)[0]
        dim = np.where(f1 <= f1[0])[0]
        tdim = np.where(f2 <= f2[0])[0]
        T = self.t * D
        return np.union1d(tdim, dim) if len(dim) <= T else np.intersect1d(tdim, dim)      # 0 = the base solution

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        lo, up = self.lower, self.upper
        N2 = 4
        TDec, TMask, TPop = [], [], []
        Fit = np.zeros(D)
        for _ in range(N2):
            Dec = lo + rng.random((D, D)) * (up - lo)
            Dec2 = lo + rng.random((D, D)) * (up - lo)
            Mask = np.eye(D)
            P = self.evaluate(Dec * Mask)
            TDec.append(Dec), TMask.append(Mask), TPop.append(P)
            C = cons(P)
            Fit = Fit + nd_sort(np.hstack([objs(P), C]) if C.size else objs(P), None, np.inf)[0]
        dim = self._dim_selection(Dec, Dec2)
        dim = dim[dim != 0]
        if len(dim) == 0:
            dim = np.arange(1, D + 1)
        d1 = len(dim)
        self.dim, self.fitness, self.N2 = dim, Fit, N2
        TMask1, TPop1 = [], []
        for _ in range(4):
            Mask_dim = np.eye(d1)
            Mask_d1 = np.zeros((d1, D))
            for j in range(d1):
                Mask_d1[j, dim[j] - 1] = 1
            Dec_d1 = lo + rng.random((d1, D)) * (up - lo)
            P1 = self.evaluate(Dec_d1 * Mask_d1)
            TMask1.append(Mask_dim), TPop1.append(P1)
        num_e = min(4 * d1, N)
        m = np.vstack([Mask_dim] + TMask1)
        r = _env_selection(Population.merge(P1, *TPop1), m, m, num_e)
        self.pop_d1, self.Mask_dim, self.front_d1, self.crowd_d1 = r[0], r[1], r[3], r[4]
        Dec = lo + rng.random((N, D)) * (up - lo)
        Mask = np.zeros((N, D))
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), Fit, rng=rng)] = 1
        P = self.evaluate(Dec * Mask)
        pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(P, *TPop), np.vstack([Dec] + TDec), np.vstack([Mask] + TMask), N)
        self.non_zero = np.zeros(N)
        self.generation, self.same, self.num_temp = 1, 1, None
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def step(self):
        N, rng, dim, Fit = self.N, self.rng, self.dim, self.fitness
        pool = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
        OffDec, OffMask = _operator(self, self.Dec[pool], self.Mask[pool], Fit)
        pool1 = tournament(2, 2 * N, self.front_d1, -self.crowd_d1, rng=rng)
        _, Mask_dim_off = _operator(self, self.Dec[pool1], self.Mask_dim[pool1], Fit)
        for j in range(N):
            OffMask[j, dim[np.where(Mask_dim_off[j] != 0)[0]] - 1] = 1
        off = self.evaluate(OffDec * OffMask)
        self.pop_d1, self.Mask_dim, self.front_d1, self.crowd_d1 = (lambda r: (r[0], r[1], r[3], r[4]))(
            _env_selection(Population.merge(self.pop_d1, off), np.vstack([self.Mask_dim, Mask_dim_off]),
                           np.vstack([self.Mask_dim, Mask_dim_off]), N))
        self.pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), N)
        for i in range(len(self.Mask_dim)):
            self.non_zero[i] = np.count_nonzero(self.Mask_dim[i])
        num = int(np.argmax(np.bincount(self.non_zero.astype(int))))
        if self.generation > 1:
            self.same = self.same + 1 if self.num_temp == num else 1
        else:
            self.same = 1
        self.num_temp = num
        self.generation += 1
        if self.same > self.k:
            cnt = np.array([np.count_nonzero(r) for r in self.Mask_dim])
            idx = np.where(cnt == num)[0]
            base = np.unique(np.concatenate([dim[self.Mask_dim[i] != 0] for i in idx])) if len(idx) else np.array([], dtype=int)
            for loc in base:
                Fit[loc - 1] -= np.ceil(1.0 / self.N2 * Fit[loc - 1])
