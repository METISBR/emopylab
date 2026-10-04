# emopylab 2026
"""FDV (fuzzy decision variable framework with various internal optimizers).

Reference:
X. Yang, J. Zou, S. Yang, J. Zheng, and Y. Liu. A fuzzy decision variables framework for large-scale
multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2023, 27(3): 445-459.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import nsga3_ref
from algorithms.community_utils.base import LoopAlgorithm, adds, cons, crowding, decs, ga, ga_half, nd_sort, neighbors_of, objs, polynomial_mutation, tournament, uniform_point
from algorithms.lmocso.lmocso import cal_fitness as _sde_fitness, environmental_selection as _lmocso_selection
from core.population import Population

ALGORITHM_FLAGS = {'FDV': {'integer', 'large', 'many', 'multi', 'real'}}


def _nsga2_selection(pop, N):
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def _cmopso_selection(pop, N):
    F = objs(pop)
    front, maxf = nd_sort(F, None, N)
    nxt = front < maxf
    f1 = front == 1
    with np.errstate(invalid="ignore", divide="ignore"):
        Fn = (F - F[f1].min(axis=0)) / (F[f1].max(axis=0) - F[f1].min(axis=0))
    last = np.where(front == maxf)[0]
    from algorithms.community_utils.spea import truncation
    dele = truncation(Fn[last], len(last) - N + int(nxt.sum()))
    nxt[last[~dele]] = True
    return pop[nxt]


class FDV(LoopAlgorithm):
    """Early in the run (while ``FE/maxFE <= rate``) every offspring is snapped to a decision grid whose resolution
    becomes finer in steps (0.1 R, then 0.01 R, ...), which coarsens the landscape at first; afterwards the chosen engine
    (1 NSGA-II, 2 NSGA-III, 3 MOEA/D, 4 CMOPSO, 5 LMOCSO) runs unchanged."""

    def __init__(self, pop_size: int = 100, rate: float = 0.8, acc: float = 0.4, optimizer: int = 5, type: int = 1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.rate, self.acc, self.optimizer, self.type = float(rate), float(acc), int(optimizer), int(type)

    def initial_size(self):
        if self.optimizer in (2, 3, 5):
            self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    # -- the fractional-precision operator ------------------------------------------------------------
    def _fdv(self, X):
        S = int(np.floor(np.sqrt(2 * self.rate * 1 / self.acc)))
        step = np.zeros(S + 2)
        for i in range(1, S + 1):
            step[i] = (S * i - i * i / 2) * self.acc
        step[S + 1] = self.rate
        R = self.upper - self.lower
        it = self.FE / self.max_FE
        X = np.asarray(X, dtype=float)
        for i in range(1, S + 2):
            if step[i - 1] < it <= step[i]:
                with np.errstate(all="ignore"):
                    ga_ = R * 10.0 ** -i * np.floor(10.0 ** i / R * (X - self.lower)) + self.lower
                    gb_ = R * 10.0 ** -i * np.ceil(10.0 ** i / R * (X - self.lower)) + self.lower
                    miu1, miu2 = 1 / (X - ga_), 1 / (gb_ - X)
                    take_a = (miu1 - miu2) > 0
                X = np.where(take_a, ga_, gb_)
        return X

    def _make(self, X, V=None):
        if self.FE / self.max_FE <= self.rate:
            X = self._fdv(X)
        return self.evaluate(X) if V is None else self.evaluate(X, V=V)

    # -- engines --------------------------------------------------------------------------------------
    def start(self):
        pop, N = self.pop, self.N
        opt = self.optimizer
        if opt == 1:
            _, self.front, self.crowd = _nsga2_selection(pop, N)
        elif opt == 2:
            C = cons(pop)
            feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
            self.Zmin = objs(pop)[feas].min(axis=0) if feas.any() else None
        elif opt == 3:
            self.T = int(np.ceil(N / 10))
            self.B = neighbors_of(self.V, self.T)
            self.Z = objs(pop).min(axis=0)
        elif opt == 5:
            self.pop = _lmocso_selection(pop, self.V, (self.FE / self.max_FE) ** 2)

    def _moead(self):
        rng, N, T, W = self.rng, self.N, self.T, self.V
        pop = self.pop
        for i in range(N):
            P = self.B[i][rng.permutation(self.B.shape[1])]
            off = self._make(ga_half(self.problem, decs(pop[P[:2]]), rng=rng))
            fo = objs(off)[0]
            self.Z = np.minimum(self.Z, fo)
            Fp, Z = objs(pop[P]), self.Z
            if self.type == 1:
                normW = np.linalg.norm(W[P], axis=1)
                normP = np.linalg.norm(Fp - Z, axis=1)
                normO = np.linalg.norm(fo - Z)
                with np.errstate(all="ignore"):
                    cP = np.sum((Fp - Z) * W[P], axis=1) / normW / normP
                    cO = np.sum((fo - Z) * W[P], axis=1) / normW / normO
                    g_old = normP * cP + 5 * normP * np.sqrt(1 - cP ** 2)
                    g_new = normO * cO + 5 * normO * np.sqrt(1 - cO ** 2)
            elif self.type == 2:
                g_old = np.max(np.abs(Fp - Z) * W[P], axis=1)
                g_new = np.max(np.abs(fo - Z) * W[P], axis=1)
            elif self.type == 3:
                zmax = objs(pop).max(axis=0)
                with np.errstate(all="ignore"):
                    g_old = np.max(np.abs(Fp - Z) / (zmax - Z) * W[P], axis=1)
                    g_new = np.max(np.abs(fo - Z) / (zmax - Z) * W[P], axis=1)
            else:
                with np.errstate(all="ignore"):
                    g_old = np.max(np.abs(Fp - Z) / W[P], axis=1)
                    g_new = np.max(np.abs(fo - Z) / W[P], axis=1)
            for j in np.where(g_old >= g_new)[0]:
                pop[P[j]] = off[0]

    def _cmopso_operator(self, pop):
        rng, N, D = self.rng, len(pop), self.D
        X, F = decs(pop), objs(pop)
        V = adds(pop, "V", np.zeros((N, D)))
        front, _ = nd_sort(F, None, np.inf)
        cd = crowding(F, front)
        leader = np.lexsort((-cd, front))[:10]
        off_p, off_v = np.zeros((N, D)), np.zeros((N, D))
        for i in range(N):
            w = leader[rng.permutation(len(leader))[:2]]
            with np.errstate(all="ignore"):
                ang = [np.degrees(np.arccos(np.clip(F[i] @ F[k] / (np.linalg.norm(F[i]) * np.linalg.norm(F[k])), -1, 1))) for k in w]
            win = w[1] if ang[0] > ang[1] else w[0]
            r1, r2 = rng.random(D), rng.random(D)
            off_v[i] = r1 * V[i] + r2 * (X[win] - X[i])
            off_p[i] = X[i] + off_v[i]
        return polynomial_mutation(off_p, self.lower, self.upper, rng, 20.0, prob=1.0 / D), off_v

    def _lmocso_operator(self, loser, winner):
        rng, D = self.rng, self.D
        LD, WD = decs(loser), decs(winner)
        n = len(LD)
        LV, WV = adds(loser, "V", np.zeros((n, D))), adds(winner, "V", np.zeros((n, D)))
        r1, r2 = np.repeat(rng.random((n, 1)), D, axis=1), np.repeat(rng.random((n, 1)), D, axis=1)
        vel = r1 * LV + r2 * (WD - LD)
        dec = LD + vel
        if self.FE / self.max_FE < self.rate:
            LV1 = rng.random((n, D))
            vel1 = r1 * LV1 + r2 * (WD - LD)
            dec1 = LD + vel1 + r1 * (vel1 - LV1)
            dec, vel = np.vstack([dec, dec1]), np.vstack([vel, vel1])
        dec, vel = np.vstack([dec, WD]), np.vstack([vel, WV])
        return polynomial_mutation(dec, self.lower, self.upper, rng), vel

    def step(self):
        rng, N, opt = self.rng, self.N, self.optimizer
        pop = self.pop
        if opt == 1:
            mate = tournament(2, N, self.front, -self.crowd, rng=rng)
            off = self._make(ga(self.problem, decs(pop[mate]), rng=rng))
            self.pop, self.front, self.crowd = _nsga2_selection(Population.merge(pop, off), N)
        elif opt == 2:
            C = cons(pop)
            mate = tournament(2, N, np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop)), rng=rng)
            off = self._make(ga(self.problem, decs(pop[mate]), rng=rng))
            Co = cons(off)
            feas = np.all(Co <= 0, axis=1) if Co.size else np.ones(len(off), bool)
            if feas.any():
                f = objs(off)[feas].min(axis=0)
                self.Zmin = f if self.Zmin is None else np.minimum(self.Zmin, f)
            self.pop = nsga3_ref.select(Population.merge(pop, off), N, self.V, self.Zmin, rng)
        elif opt == 3:
            self._moead()
        elif opt == 4:
            x, v = self._cmopso_operator(pop)
            off = self._make(x, v)
            self.pop = _cmopso_selection(Population.merge(pop, off), N)
        else:
            fit = _sde_fitness(objs(pop))
            n = len(pop)
            rank = rng.permutation(n)[: (n // 2) * 2] if n >= 2 else np.array([0, 0])
            h = len(rank) // 2
            loser, winner = rank[:h].copy(), rank[h:].copy()
            change = fit[loser] >= fit[winner]
            loser[change], winner[change] = winner[change].copy(), loser[change].copy()
            x, v = self._lmocso_operator(pop[loser], pop[winner])
            off = self._make(x, v)
            self.pop = _lmocso_selection(Population.merge(pop, off), self.V, (self.FE / self.max_FE) ** 2)
