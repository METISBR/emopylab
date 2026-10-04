# emopylab 2026
"""S-ECSO (enhanced competitive swarm optimizer for sparse optimization).

Reference:
X. Wang, K. Zhang, J. Wang, and Y. Jin. An enhanced competitive swarm optimizer with strongly convex
sparse operator for large-scale multi-objective optimization. IEEE Transactions on Evolutionary
Computation, 2022, 26(5): 859-871.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs
from algorithms.community_utils.spea import truncation
from core.population import Population

ALGORITHM_FLAGS = {'SECSO': {'large', 'multi', 'real', 'sparse'}}


def _sc_sparse(algo, pop, lamb):
    """Soft-threshold every particle towards zero (shrinking each entry by ``lamb``) and keep the shrunk copy when its rank is
    not worse than the original's."""
    x = decs(pop)
    new = np.zeros_like(x)
    flag = lamb > 0
    big = np.abs(x) > lamb
    new[big] = (x - lamb * np.sign(x))[big]
    span = algo.upper - algo.lower
    up = new > algo.upper
    lo = new < algo.lower
    r = algo.rng.random(x.shape) * span
    new = np.where(up | lo, r, new)
    child = algo.evaluate(new)
    both = Population.merge(pop, child)
    N = len(child)
    front, _ = nd_sort(objs(both), None, np.inf)
    out = pop[np.arange(len(pop))]
    for i in range(N):
        if front[i + N] <= front[i]:
            out[i] = both[i + N]
    return out


def _a_get(algo, pop, A, it):
    rng, N, D = algo.rng, algo.N, algo.D
    if it == 1 or A is None:
        joined = pop
    else:
        joined = Population.merge(A, pop)
    front, _ = nd_sort(objs(joined), None, np.inf)
    nd = joined[front == 1]
    nd = nd[np.unique(decs(nd), axis=0, return_index=True)[1]]
    els = decs(nd).copy()
    for i in range(len(nd)):
        j = int(np.ceil(rng.random() * D)) - 1
        els[i, j] += (algo.upper[j] - algo.lower[j]) * rng.standard_normal()
        els[i, j] = min(max(els[i, j], algo.lower[j]), algo.upper[j])
    AA = Population.merge(nd, algo.evaluate(els))
    if len(AA) > N:
        F = objs(AA)
        fr, maxf = nd_sort(F, None, N)
        nxt = fr < maxf
        f1 = fr == 1
        with np.errstate(all="ignore"):
            Fn = (F - F[f1].min(axis=0)) / (F[f1].max(axis=0) - F[f1].min(axis=0))
        last = np.where(fr == maxf)[0]
        dele = truncation(Fn[last], len(last) - N + int(nxt.sum()))
        nxt[last[~dele]] = True
        AA = AA[nxt]
    r = rng.random(algo.M)
    f_g = (objs(AA) * r).sum(axis=1) / r.sum()
    return AA, decs(AA)[int(np.argmin(f_g))]


class SECSO(LoopAlgorithm):
    """Three subswarms of a competitive swarm optimiser run on the solutions after a soft-thresholding (sparsifying) step; an
    elite archive of non-dominated solutions (with a Gaussian single-variable perturbation) supplies the global guide and is
    the result.  The shrinkage threshold decays linearly and the run ends after ``maxFE/(3N)`` iterations."""

    def __init__(self, pop_size: int = 100, l_max: float = 0.35, l_min: float = 0.0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.l_max, self.l_min = float(l_max), float(l_min)

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        span = self.upper - self.lower
        x = rng.random((N, D)) * span + self.lower
        self.v = rng.random((N, D)) * span + self.lower
        sub = N // 3
        self.sub_index = [0, sub, N - sub - 1]                      # zero-based starts of the three subswarms
        self.step_size = (self.l_max - self.l_min) * span / ((self.max_FE / N / 3) - 1)
        self.lamb = self.l_max * span
        return self.evaluate(x)

    def _initialize_advance(self, infills=None, **kwargs):
        self.swarm = infills
        self.pop = infills
        self.archive = None
        self.it = 0
        self._set_optimum()

    def _ecso(self, pop, gbest):
        rng, N = self.rng, self.N
        w, c1, c2 = 0.7968, 1.4962, 1.4962
        x, F = decs(pop), objs(pop)
        a, b, c = self.sub_index
        bounds = [(a, b), (b, c), (c, N)]
        llbest = np.zeros_like(x)
        for lo, hi in bounds:
            fr, _ = nd_sort(F[lo:hi], None, np.inf)
            cand = x[lo:hi][fr == 1]
            llbest[lo:hi] = cand[int(rng.integers(0, len(cand)))]
        s1, s2, s3 = b - a, c - b, N - c
        r1 = rng.permutation(s1)
        r2 = rng.permutation(s2) + s1
        r3 = rng.permutation(s3) + s1 + s2
        front, _ = nd_sort(F, None, np.inf)
        v = self.v
        # (when N is a multiple of three the middle subswarm is one member short; the reference would index past it)
        for i in range(min(N // 3, s1, s2, s3)):
            trio = [r1[i], r2[i], r3[i]]
            win = int(np.argmin(front[trio]))
            winner = trio[win]
            losers = [t for k, t in enumerate(trio) if k != win]
            l1, l2 = losers
            if rng.random() < 0.5:
                v[l1] = rng.random() * v[l1] + rng.random() * (x[winner] - x[l1])
                x[l1] = x[l1] + v[l1]
                v[l2] = w * v[l2] + c1 * rng.random() * (llbest[l2] - x[l2]) + c2 * rng.random() * (gbest - x[l2])
                x[l2] = x[l2] + v[l2]
            else:
                v[l2] = rng.random() * v[l2] + rng.random() * (x[winner] - x[l2])
                x[l2] = x[l2] + v[l2]
                v[l1] = w * v[l1] + c1 * rng.random() * (llbest[l1] - x[l1]) + c2 * rng.random() * (gbest - x[l1])
                x[l1] = x[l1] + v[l1]
        x = np.minimum(np.maximum(x, self.lower), self.upper)
        return self.evaluate(x)

    def step(self):
        self.it += 1
        stop = self.it > self.max_FE / self.N / 3 - 1
        self.swarm = _sc_sparse(self, self.swarm, self.lamb)
        self.archive, gbest = _a_get(self, self.swarm, self.archive, self.it)
        self.pop = self.archive
        self.swarm = self._ecso(self.swarm, gbest)
        self.lamb = self.lamb - self.step_size
        if stop:
            self.evaluator.n_eval = max(int(self.evaluator.n_eval), int(self.max_FE))     # the reference forces FE = maxFE here
