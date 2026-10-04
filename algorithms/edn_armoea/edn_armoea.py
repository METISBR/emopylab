# emopylab 2026
"""EDN-ARMOEA (efficient dropout neural network based AR-MOEA).

Reference:
D. Guo, X. Wang, K. Gao, Y. Jin, J. Ding, and T. Chai. Evolutionary optimization of high-dimensional
multiobjective and many-objective expensive problems assisted by a dropout neural network. IEEE
Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(4): 2084-2097.
"""

from __future__ import annotations

import numpy as np

from algorithms.avg_saea.avg_saea import AVGSAEA
from algorithms.community_utils import armoea
from algorithms.community_utils.base import LoopAlgorithm, decs, ga, kmeans, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.dropout_net import DropoutNet, mc_estimate, minmax_apply, minmax_fit
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'EDNARMOEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


class EDNARMOEA(LoopAlgorithm):
    """A dropout neural network (Monte-Carlo dropout gives mean and uncertainty) replaces the objectives in AR-MOEA; after
    ``wmax`` surrogate generations ``Ke`` k-means representatives are evaluated, chosen by largest uncertainty while the
    share of useful reference points keeps changing and by smallest norm once it has stabilised."""

    def __init__(self, pop_size: int = 100, delta: float = 0.05, wmax: int = 20, Ke: int = 3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.wmax, self.Ke = float(delta), int(wmax), int(Ke)

    def _initialize_infill(self):
        self.W, _ = uniform_point(self.N, self.M)
        self.NI = 11 * self.D - 1
        return self.evaluate(self.lower + lhs_design(self.rng, self.NI, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_data(decs(infills), objs(infills))
        self.net = DropoutNet(self.D, self.M, self.rng).train(self.txx, self.tyy, 80000)
        self.ratio_old = None
        self._set_optimum()

    def _set_data(self, X, Y):
        self.ps, self.qs = minmax_fit(X), minmax_fit(Y)
        self.txx, self.tyy = minmax_apply(X, self.ps), minmax_apply(Y, self.qs)

    def _mating(self, F, ref, rng_):
        fit = armoea.contribution_fitness(F, ref, rng_, clip=False, noise_rng=self.rng)
        n = len(F)
        return tournament(2, int(np.ceil(n / 2)) * 2, np.zeros(n), -fit, rng=self.rng)

    def _env(self, F, ref, rng_, N):
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        ch = armoea.last_selection(F[last], ref, rng_, N - int(nxt.sum()), clip=False, noise_rng=self.rng)
        nxt[last[ch]] = True
        rng_ = rng_.copy()
        rng_[1] = F.max(0)
        rng_[1, rng_[1] - rng_[0] < 1e-6] = 1
        return nxt, rng_

    def step(self):
        rng, N, D = self.rng, self.N, self.D
        self.net.train(self.txx, self.tyy, 8000)
        X = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        F, S = mc_estimate(self.net, X, self.ps, self.qs)
        arc, ref, rng_, ratio = armoea.update_ref_point(F, self.W, None, clip=False, noise_rng=rng)
        if self.ratio_old is None:
            self.ratio_old = ratio
        for _ in range(1, self.wmax):
            mate = self._mating(F, ref, rng_)
            od = ga(self.problem, X[mate], (1, 20, 1, 20), rng=rng)
            of, os_ = mc_estimate(self.net, od, self.ps, self.qs)
            arc, ref, rng_, ratio = armoea.update_ref_point(np.vstack([arc, of]), self.W, rng_, clip=False, noise_rng=rng)
            mX, mF, mS = np.vstack([X, od]), np.vstack([F, of]), np.vstack([S, os_])
            keep, rng_ = self._env(mF, ref, rng_, N)
            X, F, S = mX[keep], mF[keep], mS[keep]
        flag = (self.ratio_old - ratio) < self.delta
        lab = kmeans(F, self.Ke, rng)
        new = []
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            k = idx[int(np.argmin(np.sqrt((F[idx] ** 2).sum(1))))] if flag else idx[int(np.argmax(S[idx].mean(1)))]
            new.append(X[k])
        self.ratio_old = ratio
        newpop = self.evaluate(np.array(new))
        self.pop = Population.merge(self.pop, newpop)
        tx, ty = AVGSAEA._train_data(self.pop, self.NI, len(newpop))
        self._set_data(tx, ty)
