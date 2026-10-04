# emopylab 2026
"""DSSEA (dynamic subspace search-based evolutionary algorithm).

Reference:
X. Ban, J. Liang, K. Yu, B. Qu, K. Qiao, P. N. Suganthan, and Y. Wang. A subspace search-based
evolutionary algorithm for large-scale constrained multi-objective optimization and application.
IEEE Transactions on Cybernetics, 2025, 55(5): 2486-2499.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament
from algorithms.community_utils.eps_ea import cal_fitness_eps, de_pbest_1, gn_r1r2r3
from algorithms.community_utils.spea import overall_cv, truncation
from core.population import Population

ALGORITHM_FLAGS = {'DSSEA': {'constrained', 'integer', 'large', 'many', 'multi', 'real'}}


def _select(pop, N, epsilon):
    F, C = objs(pop), cons(pop)
    fit = cal_fitness_eps(F, C if C.size else None, epsilon)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    pop, fit = pop[nxt], fit[nxt]
    order = np.argsort(fit, kind="stable")
    return pop[order], fit[order]


class DSSEA(LoopAlgorithm):
    """Every variable is ranked by the effect of perturbing it (length of the resulting objective spread plus how close the
    principal direction of that spread is to the convergence direction); variables with better ranks are varied more often
    by DE/GA, the others are recombined variable-wise from random population members, under epsilon-constraint handling."""

    def __init__(self, pop_size: int = 100, n_sel: int = 2, n_per: int = 4, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.n_sel, self.n_per = int(n_sel), int(n_per)

    def _aldva(self, pop):
        rng, N, D, M = self.rng, self.N, self.D, self.M
        X, F = decs(pop), objs(pop)
        nd = nd_sort(F, None, 1)[0] == 1
        fmin, fmax = F[nd].min(axis=0), F[nd].max(axis=0)
        if np.any(fmax == fmin):
            fmax, fmin = np.ones(M), np.zeros(M)
        nsel, nper = self.n_sel, self.n_per
        length = np.zeros((D, nsel))
        angle = np.zeros((D, nsel))
        for i in range(D):
            sample = rng.integers(0, N, nsel)
            Xs = np.tile(X[sample], (nper, 1))
            Xs[:, i] = rng.uniform(self.lower[i], self.upper[i], len(Xs))
            new = self.evaluate(Xs)
            Fn = objs(new)
            for j in range(nsel):
                pts = Fn[j::nsel]
                srt = pts[np.lexsort(pts.T[::-1])]
                length[i, j] = np.linalg.norm(srt[0] - srt[-1])
                pts = (pts - fmin) / (fmax - fmin)
                pts = pts - pts.mean(axis=0)
                _, _, Vt = np.linalg.svd(pts)
                vec = Vt[0] / np.linalg.norm(Vt[0])
                sine = abs(np.sum(vec * np.ones(M))) / np.linalg.norm(vec) / np.linalg.norm(np.ones(M))
                angle[i, j] = np.degrees(np.arcsin(min(sine, 1.0)))
        length, angle = length.mean(axis=1), angle.mean(axis=1)
        with np.errstate(all="ignore"):
            length = (length - length.min()) / (length.max() - length.min())
            angle = (angle - angle.min()) / (angle.max() - angle.min())
        R = angle + length
        order = np.argsort(-np.where(np.isnan(R), -np.inf, R), kind="stable")
        rank = np.empty(D, dtype=int)
        rank[order] = np.arange(1, D + 1)
        return rank

    def start(self):
        pop = self.pop
        self.rank_dv = self._aldva(pop)
        cv = overall_cv(cons(pop))
        e0 = float(cv.sum())
        self.eps0 = e0 if e0 != 0 else 100.0
        self.fit = cal_fitness_eps(objs(pop), cons(pop) if cons(pop).size else None, self.eps0)
        self.x = 0.0

    def _shuffle_nop(self, off_dec, pop, nop):
        """Variables outside the optimised set take values from random population members (one member per variable)."""
        rng = self.rng
        if nop.sum() <= 1:
            return off_dec
        n, D = len(off_dec), self.D
        Xp = decs(pop)
        cols = np.where(nop)[0]
        pick = rng.integers(0, n, (n, D))
        off_dec[:, cols] = Xp[pick[:, : len(cols)], cols[None, :]]
        return off_dec

    def _de_offspring(self, pop, fit, op, nop, p=0.1):
        rng, N = self.rng, self.N
        X = decs(pop)
        perm = rng.permutation(N) + 1
        r1, r2, _ = gn_r1r2r3(rng, N, perm)
        arr = perm - 1
        off_dec = X[arr].copy()
        best = np.argsort(fit, kind="stable")
        pnp = max(int(np.round(p * N)), 2)
        ri = np.maximum(1, np.ceil(rng.random(N) * pnp).astype(int))
        new = de_pbest_1(rng, self.lower, self.upper, X[arr], X[best[ri - 1]], X[r1[:N] - 1], X[r2[:N] - 1])
        off_dec[:, op] = new[:, op]
        off_dec = self._shuffle_nop(off_dec, pop, nop)
        return self.evaluate(off_dec)

    def _ga_offspring(self, pop, parents, op, nop):
        ga_off = ga(self.problem, decs(parents), rng=self.rng)
        off_dec = decs(parents).copy()
        off_dec[:, op] = ga_off[:, op]
        off_dec = self._shuffle_nop(off_dec, pop, nop)
        return self.evaluate(off_dec)

    def step(self):
        rng, N, D = self.rng, self.N, self.D
        pop = self.pop
        cp = (-np.log(self.eps0) - 6) / np.log(1 - 0.75)
        eps = self.eps0 * (1 - self.x) ** cp
        ip = (self.rank_dv - D) / (1 - D)
        op_prob = (1 - ip) * np.tanh(9 * self.FE / self.max_FE) + ip
        r = rng.random(D)
        op, nop = r <= op_prob, r > op_prob
        if rng.random() <= 0.5:
            off = self._de_offspring(pop, self.fit, op, nop)
        else:
            mate = tournament(2, N, self.fit, rng=rng)
            off = self._ga_offspring(pop, pop[mate], op, nop)
        self.pop, self.fit = _select(Population.merge(pop, off), N, eps)
        self.x += 1 / (self.max_FE / N)
