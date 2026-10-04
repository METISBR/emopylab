# emopylab 2026
"""MOEA-D-DYTS (mOEA/D with dynamic Thompson sampling).

Reference:
L. Sun and K. Li. Adaptive operator selection based on dynamic Thompson sampling for MOEA/D.
Proceedings of the International Conference on Parallel Problem Solving from Nature, 2020, 271-284.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, neighbors_of, objs, tournament, uniform_point
from algorithms.moea_d_frrmab.moea_d_frrmab import five_ops

ALGORITHM_FLAGS = {'MOEADDYTS': {'integer', 'many', 'multi', 'real'}}


class MOEADDYTS(LoopAlgorithm):
    """Operator choice by dynamic Thompson sampling: every one of the five operators has a Beta(a, b)
    success model whose window is capped at C observations."""

    def initial_size(self):
        self.Weight, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T, self.nr, self.C = 20, 2, 100
        self.B = neighbors_of(self.Weight, min(self.T, self.pop_size))
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.Z = F.min(axis=0)
        self.Pi = np.ones(self.N)
        with np.errstate(all="ignore"):
            self.old_obj = np.max(np.abs((F - self.Z) * self.Weight), axis=1)
        self.ab = np.ones((5, 2))
        self.count_ops = np.zeros(5)

    def step(self):
        rng, N, W = self.rng, self.N, self.Weight
        for _ in range(5):
            boundary = np.where(np.sum(W < 1e-3, axis=1) == self.M - 1)[0]
            I = np.concatenate([boundary, tournament(10, N // 5 - len(boundary), -self.Pi, rng=rng)]).astype(int)
            for i in I:
                op = int(np.argmax(rng.beta(self.ab[:, 0], self.ab[:, 1]))) + 1
                self.count_ops[op - 1] += 1
                P = self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < 0.8 else rng.permutation(N)
                off = five_ops(self, op, i, P)
                fo = objs(off)[0]
                self.Z = np.minimum(self.Z, fo)
                g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * W[P], axis=1)
                g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
                rep = np.where(g_old >= g_new)[0][: self.nr]
                self.pop[P[rep]] = off[0]
                r = 1.0 if len(rep) else 0.0
                a, b = self.ab[op - 1]
                if a + b < self.C:
                    a, b = a + r, b + 1 - r
                else:
                    a, b = (a + r) * self.C / (self.C + 1), (b + 1 - r) * self.C / (self.C + 1)
                self.ab[op - 1] = (a, b)
        if int(np.ceil(self.FE / N)) % 10 == 0:
            with np.errstate(all="ignore"):
                new_obj = np.max(np.abs((objs(self.pop) - self.Z) * W), axis=1)
                delta = (self.old_obj - new_obj) / self.old_obj
            temp = delta < 0.001
            self.Pi[~temp] = 1
            self.Pi[temp] = (0.95 + 0.05 * delta[temp] / 0.001) * self.Pi[temp]
            self.old_obj = new_obj
