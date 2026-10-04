# emopylab 2026
"""MOEA-PC (multiobjective evolutionary algorithm based on polar coordinates).

Reference:
R. Denysiuk, L. Costa, I. E. Santo, and J. C. Matos. MOEA/PC: Multiobjective evolutionary algorithm
based on polar coordinates. Proceedings of the International Conference on Evolutionary Multi-
Criterion Optimization, 2015, 141-155.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, neighbors_of, objs, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'MOEAPC': {'integer', 'many', 'multi', 'real'}}


def _polar_coordinates_grid(y, r, nDivs):
    m = len(r)
    rho = np.linalg.norm(y - r)
    theta = np.zeros(m - 1)
    for i in range(m - 1):
        u = (y[m - 1 - i] - r[m - 1 - i]) / rho
        for j in range(i):
            u = u / np.cos(theta[j])
        u = min(max(u, -1.0), 1.0)
        theta[i] = min(max(np.arcsin(u), np.finfo(float).eps), np.pi / 2 - np.finfo(float).eps)
    G = np.ceil(2 * nDivs * theta / np.pi).astype(int)
    idx = G[0]
    for i in range(1, len(G)):
        idx += (G[i] - 1) * nDivs ** i
    return int(idx) - 1, rho


class MOEAPC(LoopAlgorithm):
    """Steady-state MOEA on a polar-coordinate grid: one offspring per step, one solution per grid cell."""

    PER_STEP_OPTIMUM = False

    def __init__(self, pop_size: int = 100, delta: float = 0.8, T: int = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.delta, self.T = float(delta), int(T)

    def initial_size(self):
        m = self.M
        self.nDivs = int(np.ceil(self.pop_size ** (1.0 / (m - 1))))
        n_grids = self.nDivs ** (m - 1)
        G = np.zeros((n_grids, m - 1))
        D = m - 2
        for j in range(D + 1):
            tmp = np.tile(np.arange(1, self.nDivs + 1)[:, None], (1, self.nDivs ** j)).T          # repmat(1:nDivs, nDivs^j, 1)
            col = tmp.reshape(-1, order="F")
            G[:, j] = np.tile(col, self.nDivs ** (D - j))
        self.pop_size = n_grids
        self.B = neighbors_of(G, min(self.T, n_grids))
        return n_grids

    def start(self):
        self.Z = objs(self.pop).min(axis=0)

    def _environmental_selection(self, off):
        pop = self.pop
        for _ in range(10 * len(pop)):
            fo = np.asarray(off.F, dtype=float)
            idx, rho_o = _polar_coordinates_grid(fo, self.Z, self.nDivs)
            inc = pop[idx]
            fi = np.asarray(inc.F, dtype=float)
            pidx, rho_p = _polar_coordinates_grid(fi, self.Z, self.nDivs)
            if idx == pidx:
                if rho_o < rho_p:
                    pop[idx] = off
                return
            elif np.any(fo < fi):
                pop[idx] = off
                off = inc
            else:
                return

    def step(self):
        rng, n = self.rng, self.N
        i = int(rng.integers(0, n))
        P = self.B[i][rng.permutation(self.B.shape[1])[:3]] if rng.random() < self.delta else rng.permutation(n)[:3]
        off = self.evaluate(de(self.problem, decs(self.pop[P[:1]]), decs(self.pop[P[1:2]]), decs(self.pop[P[2:3]]), rng=rng))[0]
        self.Z = np.minimum(self.Z, np.asarray(off.F, dtype=float))
        self._environmental_selection(off)
