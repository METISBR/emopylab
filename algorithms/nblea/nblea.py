# emopylab 2026
"""NBLEA (nested bilevel evolutionary algorithm).

Reference:
A. Sinha, P. Malo, and K. Deb. Test problem construction for single-objective bilevel optimization.
Evolutionary Computation, 2014, 22(3): 439-477.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'NBLEA': {'bilevel', 'constrained', 'multi', 'real'}}


def pcx_operator(parent, lower, upper, rng, proC=0.9, proM=1.0, disM=20.0):
    """Parent-centric recombination followed by polynomial mutation (one offspring per parent)."""
    N, D = parent.shape
    W = parent.mean(axis=0)
    p1 = np.r_[np.arange(1, N), 0]
    p2 = np.r_[np.arange(2, N), 0, 1]
    with np.errstate(all="ignore"):
        off = parent + 0.1 * (parent - W) + (D / np.mean(np.abs(parent - W), axis=1))[:, None] * (parent[p2] - parent[p1]) / 2
    keep = np.repeat(rng.random((N, 1)) > proC, D, axis=1)
    off[keep] = parent[keep]
    Lo, Up = np.tile(lower, (N, 1)), np.tile(upper, (N, 1))
    site = rng.random((N, D)) < proM / D
    mu = rng.random((N, D))
    off = np.minimum(np.maximum(off, Lo), Up)
    span = Up - Lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - Lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


def _fitness(obj, con, lower_level, C):
    """Single-objective fitness of one level: feasible solutions by objective, infeasible by violation + 1e10."""
    if con is None or con.size == 0:
        cv = np.zeros(len(obj))
    else:
        c = con[:, C:] if lower_level else con[:, :C]
        cv = np.sum(np.maximum(0, c), axis=1) if c.size else np.zeros(len(obj))
    feas = cv <= 0
    return np.where(feas, obj, cv + 1e10)


class NBLEA(LoopAlgorithm):
    """Nested bilevel EA: every upper-level candidate triggers a lower-level evolutionary search (PCX recombination,
    steady-state replacement of two random members) whose best response completes the solution; the upper level
    evolves the same way with three parents per step."""

    def _lower_fit(self, LD, lo_bounds):
        pr = self.problem
        X = np.clip(LD, pr.xl, pr.xu)
        if hasattr(pr, '_calc_f'):
            F = pr._calc_f(X)
            G = pr._calc_g(X)
        else:
            pop = self.evaluate(X)
            F = np.asarray(pop.get("F"), float)
            G = pop.get("G")
        return X, F[:, 1], (None if G is None or G.shape[1] == 0 else np.asarray(G, float))

    def _ll_search(self, ul, ll_init):
        pr, rng, N, DU = self.problem, self.rng, self.N, getattr(self.problem, "DU", max(1, self.problem.n_var // 2))
        lo, up = self.lower[DU:], self.upper[DU:]
        C = getattr(pr, "C", 0)
        n_rand = N - (0 if ll_init is None else len(ll_init))
        ll = lo + rng.random((n_rand, len(lo))) * (up - lo)
        if ll_init is not None:
            ll = np.vstack([ll_init, ll])
        def evaluate(LL):
            X = np.hstack([np.tile(ul, (len(LL), 1)), LL])
            Xc, f, g = self._lower_fit(X, None)
            return Xc, _fitness(f, g, True, C)
        pop_x, pop_f = evaluate(ll)
        fe = 0
        while fe < getattr(pr, "maxFElower", 1000):
            pool = tournament(2, 3, pop_f, rng=rng)
            off = pcx_operator(pop_x[pool][:, DU:], lo, up, rng)
            ox, of = evaluate(off)
            fe += len(off)
            sel = rng.permutation(len(pop_x))[:2]
            cx, cf = np.vstack([pop_x[sel], ox]), np.concatenate([pop_f[sel], of])
            r = np.argsort(cf, kind="stable")[:2]
            pop_x[sel], pop_f[sel] = cx[r], cf[r]
        return pop_x[int(np.argmin(pop_f))][DU:]

    def _upper_fit(self, pop):
        F, G = np.asarray(pop.get("F"), float), pop.get("G")
        G = None if G is None or np.size(G) == 0 else np.asarray(G, float)
        return _fitness(F[:, 0], G, False, getattr(self.problem, "C", 0))

    def _initialize_infill(self):
        pr, rng, N, DU = self.problem, self.rng, self.N, getattr(self.problem, "DU", max(1, self.problem.n_var // 2))
        lo, up = self.lower[:DU], self.upper[:DU]
        ul = lo + rng.random((N, DU)) * (up - lo)
        ll = np.array([self._ll_search(ul[i], None) for i in range(N)])
        return self.evaluate(np.hstack([ul, ll]))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def step(self):
        pr, rng, DU = self.problem, self.rng, getattr(self.problem, "DU", max(1, self.problem.n_var // 2))
        pop = self.pop
        fit = self._upper_fit(pop)
        pool = tournament(2, 3, fit, rng=rng)
        parents = decs(pop[pool])
        ul = pcx_operator(parents[:, :DU], self.lower[:DU], self.upper[:DU], rng)
        d = ((ul[:, None, :] - parents[None, :, :DU]) ** 2).sum(axis=2)
        closest = np.argmin(d, axis=1)
        ll = np.array([self._ll_search(ul[i], parents[closest[i], DU:][None, :]) for i in range(len(ul))])
        off = self.evaluate(np.hstack([ul, ll]))
        sel = rng.permutation(len(pop))[:2]
        merged = Population.merge(pop[sel], off)
        rank = np.argsort(self._upper_fit(merged), kind="stable")
        for a, r in zip(sel, rank[:2]):
            pop[int(a)] = merged[int(r)]
        self.pop = pop
