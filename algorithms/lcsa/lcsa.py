# emopylab 2026
"""LCSA (linear combination-based search algorithm).

Reference:
H. Zille. Large-scale Multi-objective Optimisation: New Approaches and a Classification of the
State-of-the-Art. PhD Thesis, Otto von Guericke University Magdeburg, 2019.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import nsga3_ref
from algorithms.community_utils.base import LoopAlgorithm, adds, cons, crowding, decs, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'LCSA': {'integer', 'large', 'many', 'multi', 'real'}}


def _nsga2_selection(pop, N):
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def _update_gbest(pop, N):
    pop = pop[nd_sort(objs(pop), None, 1)[0] == 1]
    cd = crowding(objs(pop))
    rank = np.argsort(-cd, kind="stable")[: min(N, len(pop))]
    return pop[rank], cd[rank]


def _update_pbest(pbest, pop):
    replace = ~np.all(objs(pop) >= objs(pbest), axis=1)
    out = pbest[np.arange(len(pbest))]
    out[replace] = pop[replace]
    return out


def _poly(X, lower, upper, site, mu, disM=20.0):
    span = upper - lower
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] = X[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lower[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] = X[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (upper[t] - X[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return X


def _smpso_move(rng, X, V, Pb, Gb, lower, upper):
    """Constriction-factor particle move with velocity clamping, bound repair and occasional polynomial mutation."""
    N, D = X.shape
    W = np.repeat(rng.uniform(0.1, 0.5, (N, 1)), D, axis=1)
    r1, r2 = np.repeat(rng.random((N, 1)), D, axis=1), np.repeat(rng.random((N, 1)), D, axis=1)
    C1, C2 = np.repeat(rng.uniform(1.5, 2.5, (N, 1)), D, axis=1), np.repeat(rng.uniform(1.5, 2.5, (N, 1)), D, axis=1)
    off_v = W * V + C1 * r1 * (Pb - X) + C2 * r2 * (Gb - X)
    phi = np.maximum(4, C1 + C2)
    off_v = off_v * 2 / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
    delta = np.tile((upper - lower) / 2, (N, 1))
    off_v = np.maximum(np.minimum(off_v, delta), -delta)
    off_x = X + off_v
    lo, up = np.tile(lower, (N, 1)), np.tile(upper, (N, 1))
    off_v[(off_x < lo) | (off_x > up)] *= 0.001
    off_x = np.maximum(np.minimum(off_x, up), lo)
    site = np.repeat(rng.random((N, 1)) < 0.15, D, axis=1) & (rng.random((N, D)) < 1 / D)
    return _poly(off_x, lo, up, site, rng.random((N, D))), off_v


class LCSA(LoopAlgorithm):
    """Every so often the search switches to the space of linear-combination coefficients of the current population:
    a coefficient vector ``x`` (bounds -10..10) creates the solution ``x @ population``, and 30 generations of the same
    engine (SMPSO ``1``, NSGA-II ``2`` or NSGA-III ``3``) are run on those coefficient vectors."""

    def __init__(self, pop_size: int = 100, optimiser: int = 3, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.optimiser = int(optimiser)

    def initial_size(self):
        if self.optimiser == 3:
            self.Z, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        pop, N = self.pop, self.N
        self.it = 1
        self.xint = max(30, min(100, int(np.floor(0.1 * self.max_FE / N))))
        if self.optimiser == 1:
            self.swarm, self.pbest = pop, pop
            self.gbest, self.crowd = _update_gbest(pop, N)
            self.pop = self.gbest
        elif self.optimiser == 2:
            _, self.front, self.crowd = _nsga2_selection(pop, N)
        else:
            C = cons(pop)
            feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
            self.Zmin = objs(pop)[feas].min(axis=0) if feas.any() else None

    # -- engines on the decision space -------------------------------------------------------------
    def _generation(self):
        rng, N = self.rng, self.N
        if self.optimiser == 1:
            gb = self.gbest[tournament(2, N, -self.crowd, rng=rng)]
            X, V = decs(self.swarm), adds(self.swarm, "V", np.zeros((N, self.D)))
            x, v = _smpso_move(rng, X, V, decs(self.pbest), decs(gb), self.lower, self.upper)
            new = self.evaluate(x, V=v)
            self.swarm = new
            self.gbest, self.crowd = _update_gbest(Population.merge(self.gbest, new), N)
            self.pbest = _update_pbest(self.pbest, new)
            self.pop = self.gbest
        elif self.optimiser == 2:
            mate = tournament(2, N, self.front, -self.crowd, rng=rng)
            off = self.evaluate(ga(self.problem, decs(self.pop[mate]), rng=rng))
            self.pop, self.front, self.crowd = _nsga2_selection(Population.merge(self.pop, off), N)
        else:
            C = cons(self.pop)
            cvv = np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(self.pop))
            mate = tournament(2, N, cvv, rng=rng)
            off = self.evaluate(ga(self.problem, decs(self.pop[mate]), rng=rng))
            self._update_zmin(off)
            self.pop = nsga3_ref.select(Population.merge(self.pop, off), N, self.Z, self.Zmin, rng)

    def _update_zmin(self, pop):
        C = cons(pop)
        feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
        if feas.any():
            f = objs(pop)[feas].min(axis=0)
            self.Zmin = f if self.Zmin is None else np.minimum(self.Zmin, f)

    # -- coefficient space ---------------------------------------------------------------------------
    def _coef_ga(self, parents, lo, up):
        """SBX and polynomial mutation of coefficient vectors with their own bounds."""
        rng = self.rng
        P1, P2 = parents[: len(parents) // 2], parents[len(parents) // 2: (len(parents) // 2) * 2]
        n, D = P1.shape
        mu = rng.random((n, D))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
        beta[rng.random((n, D)) < 0.5] = 1
        off = np.vstack([(P1 + P2) / 2 + beta * (P1 - P2) / 2, (P1 + P2) / 2 - beta * (P1 - P2) / 2])
        lower, upper = np.tile(lo, (2 * n, 1)), np.tile(up, (2 * n, 1))
        off = np.minimum(np.maximum(off, lower), upper)
        return _poly(off, lower, upper, rng.random((2 * n, D)) < 1 / D, rng.random((2 * n, D)))

    def _eval_combos(self, coef, base, **extras):
        return self.evaluate(coef @ base, **extras)

    def _coefficient_phase(self):
        rng, N = self.rng, self.N
        base = decs(self.pop)
        lo, up = np.full(N, -10.0), np.full(N, 10.0)
        coef = lo + rng.random((N, N)) * (up - lo)
        gens = 30
        if self.optimiser == 1:
            xpop = self._eval_combos(coef, base, xd=coef, xv=np.zeros((N, N)))
            xpb = xpop
            xg, xcd = _update_gbest(xpop, N)
            self.gbest, self.crowd = _update_gbest(Population.merge(self.gbest, xg), N)
            self.pop = self.gbest
            for _ in range(gens):
                self.not_terminated(self.gbest)
                gb = xg[tournament(2, N, -xcd, rng=rng)]
                Xp, Pb, Gb = (adds(xpop, "xd", coef), adds(xpb, "xd", coef), adds(gb, "xd", coef))
                x, v = _smpso_move(rng, Xp, Xp, Pb, Gb, lo, up)      # (the reference feeds the positions in as velocities)
                new = self._eval_combos(x, base, xd=x, xv=v)
                xpb = _update_pbest(xpb, new)
                xg, xcd = _update_gbest(Population.merge(xpop, new), N)
                self.gbest, self.crowd = _update_gbest(Population.merge(self.gbest, xg), N)
                self.pop = self.gbest
            return
        xpop = self._eval_combos(coef, base, xd=coef)
        if self.optimiser == 2:
            _, xfront, xcd = _nsga2_selection(xpop, N)
            self.pop = _nsga2_selection(Population.merge(self.pop, xpop), N)[0]
            for _ in range(gens):
                self.not_terminated(self.pop)
                mate = tournament(2, N, xfront, -xcd, rng=rng)
                x = self._coef_ga(adds(xpop[mate], "xd", coef[:1].repeat(N, 0)), lo, up)
                new = self._eval_combos(x, base, xd=x)
                xpop, xfront, xcd = _nsga2_selection(Population.merge(xpop, new), N)
                self.pop, self.front, self.crowd = _nsga2_selection(Population.merge(self.pop, xpop), N)
            return
        self._update_zmin(xpop)
        self.pop = nsga3_ref.select(Population.merge(self.pop, xpop), N, self.Z, self.Zmin, rng)
        for _ in range(gens):
            self.not_terminated(self.pop)
            C = cons(xpop)
            mate = tournament(2, N, np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(xpop)), rng=rng)
            x = self._coef_ga(adds(xpop[mate], "xd", coef[:1].repeat(N, 0)), lo, up)
            new = self._eval_combos(x, base, xd=x)
            self._update_zmin(new)
            xpop = nsga3_ref.select(Population.merge(xpop, new), N, self.Z, self.Zmin, rng)
            self.pop = nsga3_ref.select(Population.merge(self.pop, xpop), N, self.Z, self.Zmin, rng)

    def step(self):
        self._generation()
        self.it += 1
        if self.it % self.xint == 0:
            self.not_terminated(self.pop)
            self._coefficient_phase()
