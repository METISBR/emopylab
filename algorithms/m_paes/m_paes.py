# emopylab 2026
"""M-PAES (memetic algorithm with Pareto archived evolution strategy).

Reference:
J. D. Knowles and D. W. Corne. M-PAES: A memetic algorithm for multiobjective optimization.
Proceedings of the IEEE Congress on Evolutionary Computation, 2000, 325-332.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, first_front, ga_half, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'MPAES': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _grid_density(div, *mats):
    """Number of solutions sharing the adaptive-grid cell of every row of the stacked matrices."""
    F = np.vstack([np.asarray(m, dtype=float).reshape(-1, mats[0].shape[-1]) for m in mats])
    d = (F.max(axis=0) - F.min(axis=0)) / div
    with np.errstate(all="ignore"):
        loc = np.floor((F - F.min(axis=0)) / d)
    loc[loc >= div] = div - 1
    loc[np.isnan(loc)] = 0
    _, site = np.unique(loc, axis=0, return_inverse=True)
    site = site.reshape(-1)
    crowd = np.bincount(site)[site]
    out, k = [], 0
    for m in mats:
        n = np.asarray(m).reshape(-1, mats[0].shape[-1]).shape[0]
        out.append(crowd[k:k + n])
        k += n
    return out


def _update_archive(G, off, parents, N, div, rng):
    """Returns (G, dominated, G_crowd, off_crowd, parent_crowd). ``off`` is one individual, ``parents`` an array/None."""
    of = np.asarray(off.F, dtype=float)[None, :]
    Gf = objs(G)
    domi = np.any(of <= Gf, axis=1).astype(int) - np.any(of >= Gf, axis=1).astype(int)
    dominated = bool(np.any(domi == -1))
    pf = np.zeros((0, Gf.shape[1])) if parents is None or len(parents) == 0 else objs(parents)
    g_c, o_c, p_c = _grid_density(div, Gf, of, pf)
    if np.any(domi == 1):
        G = Population.merge(G[domi != 1], Population.create([off]))
    elif not dominated:
        if len(G) < N:
            G = Population.merge(G, Population.create([off]))
        elif np.any(o_c < g_c):
            worst = np.where(g_c == g_c.max())[0]
            G = G.copy(deep=False)
            G[int(worst[rng.integers(0, len(worst))])] = off
    return G, dominated, g_c, o_c, p_c


class MPAES(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, l_fails: int = 5, l_opt: int = 10, cr_trials: int = 20, div: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.l_fails, self.l_opt, self.cr_trials, self.div = int(l_fails), int(l_opt), int(cr_trials), int(div)

    def start(self):
        self.P = self.pop
        self.G = self.P[first_front(objs(self.P))]
        self.pop = self.G

    def _paes(self, c, G, H):
        fails = moves = 0
        while fails < self.l_fails and moves < self.l_opt:
            m = self.fep(Population.create([c]))[0]
            fc, fm = np.asarray(c.F), np.asarray(m.F)
            if np.all(fc <= fm):
                fails += 1
            else:
                H, dominated, _, m_c, c_c = _update_archive(H, m, Population.create([c]), self.N, self.div, self.rng)
                if np.all(fc >= fm):
                    c, fails = m, 0
                elif not dominated and m_c[0] < c_c[0]:
                    c = m
            G, *_ = _update_archive(G, m, None, self.N, self.div, self.rng)
            moves += 1
        return c, G

    def step(self):
        rng, N = self.rng, self.N
        for i in range(N):
            Pi = self.P[i]
            keep = ~np.all(objs(self.G) <= np.asarray(Pi.F), axis=1)
            H = Population.merge(self.G[keep], Population.create([Pi]))
            self.P[i], self.G = self._paes(Pi, self.G, H)
        P1 = []
        for i in range(N):
            for _ in range(self.cr_trials):
                comb = Population.merge(self.P, self.G)
                parents = comb[rng.permutation(len(comb))[:2]]
                c = self.evaluate(ga_half(self.problem, decs(parents), (1, 20, 0, 0), rng=rng))[0]
                self.G, dominated, g_c, c_c, p_c = _update_archive(self.G, c, parents, N, self.div, rng)
                if not dominated and np.any(c_c <= p_c):
                    break
            if dominated:
                c = self.G[int(tournament(2, 1, g_c, rng=rng)[0])]
            P1.append(c)
        self.P = Population.create(P1)
        self.pop = self.G
