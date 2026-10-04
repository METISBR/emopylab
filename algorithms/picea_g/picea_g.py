# emopylab 2026
"""PICEA-g (preference-inspired coevolutionary algorithm with goals).

Reference:
R. Wang, R. C. Purshouse, and P. J. Fleming. Preference-inspired coevolutionary algorithms for many-
objective optimization. IEEE Transactions on Evolutionary Computation, 2013, 17(4): 474-494.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, first_front, ga, objs, pdist2, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'PICEAg': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _gene_goal(F, n_goal, rng):
    return rng.uniform(F.min(axis=0), F.max(axis=0) * 1.2, size=(n_goal, F.shape[1]))


def _update_archive(arch, N):
    arch = arch[first_front(objs(arch))]
    if len(arch) > N:
        dist = pdist2(objs(arch), objs(arch))
        np.fill_diagonal(dist, np.inf)
        arch = arch[~truncate_lexi(dist, len(arch) - N)]
    return arch


def _environment_selection(pop, goal, N):
    F = objs(pop)
    NP, NG = len(pop), len(goal)
    fdg = np.all(F[:, None, :] - goal[None, :, :] <= 0, axis=2)        # solution i satisfies goal g
    ng = fdg.sum(axis=0).astype(float)
    with np.errstate(all="ignore"):
        fs = (fdg / ng[None, :]).sum(axis=1)
        fs = np.nan_to_num(fs)
        fg = np.where(ng == 0, 0.5, 1.0 / (1.0 + (ng - 1.0) / (NP - 1.0)))
    nd = np.where(first_front(F))[0]
    if len(nd) < N:
        fs = fs.copy()
        fs[nd] = np.inf
        nxt = np.argsort(-fs, kind="stable")[:N]
    else:
        nxt = nd[np.argsort(-fs[nd], kind="stable")[:N]]
    return pop[nxt], goal[np.argsort(-fg, kind="stable")[:N]]


class PICEAg(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, NGoal: int | None = None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.NGoal = NGoal

    def start(self):
        self.n_goal = int(self.NGoal) if self.NGoal else 100 * self.M
        self.swarm = self.pop
        self.goal = _gene_goal(objs(self.pop), self.n_goal, self.rng)
        self.archive = _update_archive(self.pop, self.N)
        self.pop = self.archive

    def step(self):
        pool = self.rng.integers(0, len(self.swarm), size=self.N)
        off = self.evaluate(ga(self.problem, decs(self.swarm[pool]), rng=self.rng))
        self.archive = _update_archive(Population.merge(self.archive, off), self.N)
        new_goal = _gene_goal(np.vstack([objs(self.swarm), objs(off)]), self.n_goal, self.rng)
        self.swarm, self.goal = _environment_selection(Population.merge(self.swarm, off), np.vstack([self.goal, new_goal]), self.N)
        self.pop = self.archive
