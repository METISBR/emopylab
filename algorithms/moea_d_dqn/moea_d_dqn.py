# emopylab 2026
"""MOEA-D-DQN (mOEA/D based on deep Q-network (Enhanced with proper Target Network)).

Reference:
Y. Tian, X. Li, H. Ma, X. Zhang, K. C. Tan, and Y. Jin, Deep reinforcement learning based adaptive
operator selection for evolutionary multi-objective optimization, IEEE Transactions on Emerging
Topics in Computational Intelligence, 2023, 7(4): 1051-1064.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, objs, tournament, uniform_point
from algorithms.community_utils.dqn import DQN
from core.population import Population

ALGORITHM_FLAGS = {'MOEADDQN': {'integer', 'multi', 'real'}}

_PROB_TABLE = np.array([70, 28, 10, 8], dtype=float)


class MOEADDQN(LoopAlgorithm):
    """MOEA/D (Tchebycheff, neighbourhood replacement, utility-based subproblem selection) whose recombination operator
    (SBX, M2M, DE/rand/1, DE/rand/2) is chosen per offspring by a DQN fed with the parent and its weight vector; the reward
    is the best recent fitness-improvement rate of that operator in a sliding window."""

    def _initialize_infill(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.evaluate(self.random_decs(self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        N, M, D = self.N, self.M, self.D
        self.pop = infills
        T = int(np.ceil(N / 10))
        self.B = np.argsort(np.sqrt(((self.W[:, None] - self.W[None]) ** 2).sum(-1)), 1, kind="stable")[:, :T]
        self.Z = objs(infills).min(0)
        self.Pi = np.ones(N)
        self.old = np.max(np.abs((objs(infills) - self.Z) * self.W), 1)
        self.dqn = DQN(D + M, 4, self.rng)
        self.SW = np.zeros((2, N * 4))
        self._set_optimum()

    # -- operators -------------------------------------------------------------
    def _sbx(self, X, r0, P):
        rng = self.rng
        p0, p1 = X[r0], X[P[rng.integers(0, len(P))]]
        D = len(p0)
        mu = rng.random(D)
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (1 if rng.random() >= 0.5 else -1)
        beta[rng.random(D) < 0.5] = 1
        return (p0 + p1) / 2 + beta * (p0 - p1) / 2 if rng.random() < 0.5 else (p0 + p1) / 2 - beta * (p0 - p1) / 2

    def _m2m(self, X, r0, P):
        rng = self.rng
        N = len(X)
        p1, p2 = X[r0], X[P[rng.integers(0, len(P))]]
        with np.errstate(all="ignore"):
            rc = (2 * rng.random() - 1) * (1 - rng.random() ** (-(1 - self.FE / (self.max_FE + N)) ** 0.7))
        return p1 + rc * (p1 - p2)

    def _choose(self, s):
        q = self.dqn.net.predict(s[None])[0]
        order = np.argsort(q, kind="stable") + 1            # action indices by ascending Q (1-based)
        idxs = 4 + 1 - order
        prob = _PROB_TABLE[idxs - 1] if np.all(idxs < 5) else np.full(4, 5.0)
        prob = prob / prob.sum()
        return int(self.rng.choice(4, p=prob))

    def _polymut(self, x):
        rng, lb, ub = self.rng, self.lower, self.upper
        x = np.clip(x, lb, ub)
        D = len(x)
        mut = rng.random(D) < 1 / D
        u = rng.random(D)
        with np.errstate(all="ignore"):
            d1, d2 = (x - lb) / (ub - lb), (ub - x) / (ub - lb)
            dl = np.where(u <= 0.5, (2 * u + (1 - 2 * u) * (1 - d1) ** 21) ** (1 / 21) - 1,
                          1 - (2 * (1 - u) + (2 * u - 1) * (1 - d2) ** 21) ** (1 / 21))
        return np.clip(x + mut * np.nan_to_num(dl) * (ub - lb), lb, ub)

    def _offspring(self, X, r0, P):
        s = np.concatenate([X[r0], self.W[r0]])
        if self.dqn.counter > 300:
            a = self._choose(s)
            for i in range(4):
                if not np.any(self.SW[0] == i + 1):
                    a = i
                    break
        else:
            a = int(self.rng.integers(0, 4))
        if a == 0:
            off = self._sbx(X, r0, P)
        elif a == 1:
            off = self._m2m(X, r0, P)
        elif a == 2:
            off = X[r0] + 0.5 * (X[P[0]] - X[P[1]])
        else:
            off = X[r0] + 0.5 * (X[P[0]] - X[P[1]] + X[P[2]] - X[P[3]])
        self._a, self._s, self._s2 = a, s, np.concatenate([off, self.W[r0]])
        return off

    def _learn(self, r):
        self.SW = np.column_stack([self.SW[:, 1:], [self._a + 1, r]])
        reward = self.SW[1, self.SW[0] == self._a + 1].max()
        self.dqn.store(self._s, self._a, reward, self._s2)
        if self.dqn.counter > 200 and self.rng.random() < 0.2:
            self.dqn.learn()

    def step(self):
        rng, N, M = self.rng, self.N, self.M
        W, B = self.W, self.B
        for _ in range(5):
            bnd = np.where((W < 1e-3).sum(1) == M - 1)[0]
            k = int(np.floor(N / 5)) - len(bnd)
            cand = np.unique(np.concatenate([bnd, tournament(10, k, -self.Pi, rng=rng) if k > 0 else np.zeros(0, int)]))
            for c in cand:
                P = B[c][rng.permutation(B.shape[1])] if rng.random() < 0.9 else rng.permutation(N)
                if len(P) < 4:
                    # deviation: DE/rand/2 needs four mates; with T = ceil(N/10) < 4 (N <= 30) the reference indexes past the
                    # neighbourhood, so it is completed with random distinct population members
                    perm = rng.permutation(N)
                    rest = perm[~np.isin(perm, P)]
                    P = np.concatenate([P, rest[: 4 - len(P)]])
                X = decs(self.pop)
                off = self.evaluate(self._polymut(self._offspring(X, c, P))[None])
                f = objs(off)[0]
                self.Z = np.minimum(self.Z, f)
                F = objs(self.pop)
                g_old = np.max(np.abs((F[P] - self.Z) * W[P]), 1)
                g_new = np.max(np.abs((f - self.Z) * W[P]), 1)
                upd = g_old >= g_new
                if upd.any():
                    with np.errstate(all="ignore"):
                        fir = (g_old[upd] - g_new[upd]) / g_old[upd]
                    pop = self.pop
                    idx = P[upd]
                    merged = Population.merge(pop, off)
                    sel = np.arange(len(pop))
                    sel[idx] = len(pop)
                    self.pop = merged[sel]
                    self._learn(float(np.nansum(fir)))
        if int(np.ceil(self.FE / N)) % 50 == 0:
            new = np.max(np.abs((objs(self.pop) - self.Z) * W), 1)
            with np.errstate(all="ignore"):
                delta = (self.old - new) / self.old
            tmp = delta < 0.001
            self.Pi[~tmp] = 1
            self.Pi[tmp] = (0.95 + 0.05 * delta[tmp] / 0.001) * self.Pi[tmp]
            self.old = new
