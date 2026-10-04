# emopylab 2026
"""ToP (two-phase framework with NSGA-II).

Reference:
Z. Liu and Y. Wang. Handling constrained multiobjective optimization problems with constraints in
both the decision and objective spaces. IEEE Transactions on Evolutionary Computation, 2019, 23(5):
870-884.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, cv, cons, de, decs, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'ToP': {'constrained', 'integer', 'multi', 'real'}}


def _environmental_selection(pop, N):
    F, c = objs(pop), cons(pop)
    front_no, max_f = nd_sort(F, c if c.size else None, N)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], cd[nxt]


def _mutate(algo, off):
    N, D = off.shape
    lo, up = np.broadcast_to(algo.lower, (N, D)), np.broadcast_to(algo.upper, (N, D))
    span = up - lo
    rng = algo.rng
    s2 = rng.random((N, D)) < 1.0 / D
    mu = rng.random((N, D))
    disM = 20.0
    off = np.minimum(np.maximum(off, lo), up)
    t = s2 & (mu <= 0.5)
    off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    t = s2 & (mu > 0.5)
    off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


class ToP(LoopAlgorithm):
    """Two-phase constrained MOEA: constraint/objective-first DE (current-to-rand or rand-to-best) while the
    population is mostly infeasible or still improving, then NSGA-II-style DE."""

    def start(self):
        cvp = cv(self.pop)
        self.Pf = float(np.mean(cvp == 0))
        self.Delta = 1.0

    def step(self):
        rng, N = self.rng, self.N
        if (self.Delta >= 0.2 or self.Pf <= 1 / 3) and self.FE <= 0.9 * self.max_FE:
            X, F = decs(self.pop), objs(self.pop)
            fp = F.sum(axis=1) / self.M
            cvp = cv(self.pop)
            best = int(np.argmin(fp))
            cur = rng.random(N) < 0.5
            r = rng.integers(0, N, size=(N, 3))
            off = np.zeros_like(X)
            ii = np.where(cur)[0]                                           # DE/current-to-rand/1
            off[ii] = X[ii] + 0.5 * (X[r[ii, 0]] - X[ii]) + 0.5 * (X[r[ii, 1]] - X[r[ii, 2]])
            jj = np.where(~cur)[0]                                          # DE/rand-to-best/1
            off[jj] = X[r[jj, 0]] + 0.5 * (X[best] - X[r[jj, 0]]) + 0.5 * (X[r[jj, 1]] - X[r[jj, 2]])
            child = self.evaluate(_mutate(self, off))
            cvo = cv(child)
            fo = objs(child).sum(axis=1) / self.M
            better = (cvo < cvp) | ((cvo == cvp) & (fo < fp))
            self.pop[better] = child[better]
            cvp = cv(self.pop)
            self.Pf = float(np.mean(cvp == 0))
            feas = self.pop[cvp == 0]
            if len(feas):
                Ff = objs(feas)
                with np.errstate(all="ignore"):
                    fn = np.nan_to_num((Ff - Ff.min(axis=0)) / (Ff.max(axis=0) - Ff.min(axis=0)))
                fs = np.sort(fn.sum(axis=1))
                if len(fs) >= 3:
                    self.Delta = fs[len(fs) // 3 - 1] - fs[0]
        else:
            _, front_no, cd = _environmental_selection(self.pop, N)
            pool = tournament(2, N, front_no, -cd, rng=rng)
            off = self.evaluate(de(self.problem, decs(self.pop[pool]), decs(self.pop[rng.integers(0, N, size=N)]),
                                   decs(self.pop[rng.integers(0, N, size=N)]), rng=rng))
            self.pop, _, _ = _environmental_selection(Population.merge(self.pop, off), N)
