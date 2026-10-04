# emopylab 2026
"""NSGA-II+ARSBX (nSGA-II with adaptive rotation based simulated binary crossover).

Reference:
L. Pan, W. Xu, L. Li, C. He, and R. Cheng. Adaptive simulated binary crossover for rotated multi-
objective optimization. Swarm and Evolutionary Computation, 2021, 60: 100759.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, adds, cons, crowding, decs, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'NSGAIIARSBX': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _environmental_selection(pop, N):
    F = objs(pop)
    c = cons(pop)
    front_no, max_f = nd_sort(F, c if c.size else None, N)
    nxt = front_no < max_f
    cd = crowding(F, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], cd[nxt]


def _sbx_pairs(P1, P2, rng, disC=2.0, proC=1.0):
    n, D = P1.shape
    mu = rng.random((n, D))
    beta = np.zeros((n, D))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta *= np.where(rng.integers(0, 2, size=(n, D)) == 0, -1.0, 1.0)
    beta[rng.random((n, D)) < 0.5] = 1.0
    beta[np.repeat(rng.random((n, 1)) > proC, D, axis=1)] = 1.0
    return np.vstack([(P1 + P2) / 2 + beta * (P1 - P2) / 2, (P1 + P2) / 2 - beta * (P1 - P2) / 2])


def _nthroot(x, n):
    return np.sign(x) * np.abs(x) ** (1.0 / n)


class NSGAIIARSBX(LoopAlgorithm):
    def start(self):
        self.pop, self.front_no, self.crowd = _environmental_selection(self.pop, self.N)
        self.B = np.eye(self.D)
        self.m = 0.5 * (self.upper - self.lower)
        self.ps = 0.5

    def _arsbx(self, parent):
        rng, D = self.rng, self.D
        X = decs(parent)
        N = len(X)
        n_org = int(round(N * self.ps / 2) * 2)
        n_eig = int(round(N * (1 - self.ps) / 2) * 2)
        po = X[:n_org]
        off = _sbx_pairs(po[: n_org // 2], po[n_org // 2:], rng)
        flag = np.ones(n_org)
        if n_eig > 0:
            pe = (X[len(X) - n_eig:] - self.m) @ self.B
            r = _sbx_pairs(pe[: n_eig // 2], pe[n_eig // 2:], rng)
            off = np.vstack([off, r @ self.B.T + self.m])
            flag = np.concatenate([flag, 2 * np.ones(n_eig)])
        n = len(off)
        lo, up = np.broadcast_to(self.lower, (n, D)), np.broadcast_to(self.upper, (n, D))
        span = up - lo
        disM = 20.0
        site = rng.random((n, D)) < 1.0 / D
        mu = rng.random((n, D))
        with np.errstate(all="ignore"):
            t = site & (mu <= 0.5)
            off[t] = off[t] + span[t] * (_nthroot(2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo[t]) / span[t]) ** (disM + 1), disM + 1) - 1)
            t = site & (mu > 0.5)
            off[t] = off[t] + span[t] * (1 - _nthroot(2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - off[t]) / span[t]) ** (disM + 1), disM + 1))
        return self.evaluate(off, flag=flag[:, None])

    def _update_parameter(self):
        X = decs(self.pop)
        flag = adds(self.pop, "flag", np.zeros((len(self.pop), 1))).reshape(-1)
        ori, eig = np.sum(flag == 1), np.sum(flag == 2)
        self.ps = 1.0 / (1.0 + np.exp(-self.M * np.sqrt(self.D) * ((ori + 1) / (eig + ori + 2) - 0.5) * self.FE / self.max_FE))
        C = np.atleast_2d(np.cov(X, rowvar=False))
        E, B = np.linalg.eigh(C)
        order = np.argsort(-np.sqrt(np.maximum(E, 0)), kind="stable")
        self.B = B[:, order]
        self.m = X.mean(axis=0)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self._arsbx(self.pop[pool])
        self.pop, self.front_no, self.crowd = _environmental_selection(Population.merge(self.pop, off), self.N)
        self._update_parameter()
