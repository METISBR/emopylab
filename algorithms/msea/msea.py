# emopylab 2026
"""MSEA (multi-stage multi-objective evolutionary algorithm).

Reference:
Y. Tian, C. He, R. Cheng, and X. Zhang. A multi-stage evolutionary algorithm for better diversity
preservation in multi-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics:
Systems, 2021, 51(9): 5880-5894.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, nd_sort, objs, tournament
from algorithms.moea_dd.moea_dd import update_front
from core.population import Population

ALGORITHM_FLAGS = {'MSEA': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MSEA(LoopAlgorithm):
    """Steady-state multi-stage EA: the current stage (converge non-dominated levels, then spread the front, then
    refine it) is derived from the population's non-domination levels and the nearest-neighbour diversity of the
    normalised objectives, and decides both the mating partners and the replacement rule of every offspring."""

    PER_STEP_OPTIMUM = False

    def start(self):
        self.front = nd_sort(objs(self.pop), None, np.inf)[0]

    def step(self):
        N, rng = self.N, self.rng
        pop = self.pop
        F = objs(pop)
        f1 = self.front == 1
        fmax, fmin = F[f1].max(axis=0), F[f1].min(axis=0)
        with np.errstate(all="ignore"):
            P = (F - fmin) / (fmax - fmin)
        Dist = np.sqrt(np.maximum(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2), 0))
        np.fill_diagonal(Dist, np.inf)
        for _ in range(N):
            sd = np.sort(Dist, axis=1)
            Div = sd[:, 0] + 0.01 * sd[:, 1]
            if self.front.max() > 1:
                stage = 1
            elif Div.min() < Div.max() / 2:
                stage = 2
            else:
                stage = 3
            s = P.sum(axis=1)
            if stage == 1:
                mp = tournament(2, 2, self.front, s, rng=rng)
            elif stage == 2:
                mp = np.array([int(np.argmax(Div)), int(tournament(2, 1, -Div, rng=rng)[0])])
            else:
                mp = np.array([int(tournament(2, 1, s, rng=rng)[0]), int(tournament(2, 1, -Div, rng=rng)[0])])
            off = self.evaluate(ga_half(self.problem, decs(pop[mp]), rng=rng))
            with np.errstate(all="ignore"):
                oo = (objs(off)[0] - fmin) / (fmax - fmin)
            new_front = update_front(np.vstack([P, oo]), self.front)
            if new_front[-1] > 1:
                continue
            od = np.sqrt(((P - oo) ** 2).sum(axis=1))
            if new_front.max() > 1:
                stage = 1
            elif Div.min() < Div.max() / 2:
                stage = 2
            else:
                stage = 3
            replace = False
            if stage == 1:
                worse = np.where(new_front == new_front.max())[0]
                q = int(worse[int(np.argmax(P[worse].sum(axis=1)))])
                od[q] = np.inf
                replace = True
            elif stage == 2:
                q = int(np.argmin(Div))
                od[q] = np.inf
                so = np.sort(od)
                replace = so[0] + 0.01 * so[1] >= Div[q]
            else:
                q = int(np.argmin(od))
                od[q] = np.inf
                so = np.sort(od)
                replace = oo.sum() <= P[q].sum() and so[0] + 0.01 * so[1] >= Div[q]
            if replace:
                fr = update_front(np.vstack([P, oo]), new_front, q)
                self.front = np.concatenate([fr[:q], fr[-1:], fr[q:-1]])
                pop[q] = off[0]
                P[q] = oo
                Dist[q, :] = od
                Dist[:, q] = od
        self.pop = pop
        self._set_optimum()
