# emopylab 2026
"""MOEA-D-FRRMAB (mOEA/D with fitness-rate-rank-based multiarmed bandit).

Reference:
K. Li, A. Fialho, S. Kwong, and Q. Zhang. Adaptive operator selection with bandits for a
multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary
Computation, 2014, 18(1): 114-130.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, neighbors_of, objs, tournament, uniform_point


ALGORITHM_FLAGS = {'MOEADFRRMAB': {'integer', 'many', 'multi', 'real'}}


def _credit_assignment(SW, D):
    reward = np.array([SW[1, SW[0] == i].sum() for i in range(1, 5)])
    rank = np.argsort(np.argsort(-reward, kind="stable"), kind="stable") + 1
    decay = D ** rank * reward
    with np.errstate(all="ignore"):
        return decay / decay.sum()


def _frrmab(FRR, SW, C, rng):
    if np.any(FRR == 0) or np.any(SW[0] == 0):
        return int(rng.integers(1, len(FRR) + 1))
    n = np.bincount(SW[0].astype(int), minlength=len(FRR) + 1)[1:]
    return int(np.argmax(FRR + C * np.sqrt(2 * np.log(n.sum()) / n))) + 1


def five_ops(algo, op, i, P):
    """DE-based reproduction operators 1-4 (ops 1-2 DE/rand-like differences, 3-4 with K-weighted pull) and 5
    (uniform jump); polynomial mutation and one true evaluation."""
    rng = algo.rng
    X = decs(algo.pop[[i] + [int(p) for p in P[:5]]])
    x, x1, x2, x3, x4, x5 = X
    F, K, D = 0.5, 0.5, len(x)
    lo, up = algo.lower, algo.upper
    if op == 1:
        v = x + F * (x1 - x2)
    elif op == 2:
        v = x + F * (x1 - x2) + F * (x3 - x4)
    elif op == 3:
        v = x + K * (x - x1) + F * (x2 - x3) + F * (x4 - x5)
    elif op == 4:
        v = x + K * (x - x1) + F * (x2 - x3)
    else:
        v = x + rng.random(D) * (up - lo)
    off = x.copy()
    site = rng.random(D) < (1 + (op > 2))
    off[site] = v[site]
    s2 = rng.random(D) < 1.0 / D
    mu = rng.random(D)
    off = np.minimum(np.maximum(off, lo), up)
    span = up - lo
    disM = 20.0
    t = s2 & (mu <= 0.5)
    off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    t = s2 & (mu > 0.5)
    off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return algo.evaluate(off[None, :])


class MOEADFRRMAB(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, C: float = 5, Wsize: int | None = None, D: float = 1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.C, self.Wsize, self.Dd = float(C), Wsize, float(D)

    def initial_size(self):
        self.Weight, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T, self.nr = 20, 2
        self.B = neighbors_of(self.Weight, min(self.T, self.pop_size))
        return self.pop_size

    def start(self):
        F = objs(self.pop)
        self.Z = F.min(axis=0)
        self.Pi = np.ones(self.N)
        with np.errstate(all="ignore"):
            self.old_obj = np.max(np.abs((F - self.Z) * self.Weight), axis=1)
        self.FRR = np.zeros(4)
        self.SW = np.zeros((2, int(self.Wsize) if self.Wsize else int(np.ceil(self.N / 2))))

    def _four_de(self, op, i, P):
        return five_ops(self, op, i, P)

    def step(self):
        rng, N, W = self.rng, self.N, self.Weight
        for _ in range(5):
            boundary = np.where(np.sum(W < 1e-3, axis=1) == self.M - 1)[0]
            I = np.concatenate([boundary, tournament(10, N // 5 - len(boundary), -self.Pi, rng=rng)]).astype(int)
            for i in I:
                op = _frrmab(self.FRR, self.SW, self.C, rng)
                P = self.B[i][rng.permutation(self.B.shape[1])] if rng.random() < 0.9 else rng.permutation(N)
                off = self._four_de(op, i, P)
                fo = objs(off)[0]
                self.Z = np.minimum(self.Z, fo)
                g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * W[P], axis=1)
                g_new = np.max(np.abs(fo - self.Z) * W[P], axis=1)
                rep = np.where(g_old >= g_new)[0][: self.nr]
                self.pop[P[rep]] = off[0]
                with np.errstate(all="ignore"):
                    fir = np.sum((g_old[rep] - g_new[rep]) / g_old[rep])
                self.SW = np.column_stack([self.SW[:, 1:], [op, fir]])
                self.FRR = _credit_assignment(self.SW, self.Dd)
        if int(np.ceil(self.FE / N)) % 10 == 0:
            with np.errstate(all="ignore"):
                new_obj = np.max(np.abs((objs(self.pop) - self.Z) * W), axis=1)
                delta = (self.old_obj - new_obj) / self.old_obj
            temp = delta < 0.001
            self.Pi[~temp] = 1
            self.Pi[temp] = (0.95 + 0.05 * delta[temp] / 0.001) * self.Pi[temp]
            self.old_obj = new_obj
