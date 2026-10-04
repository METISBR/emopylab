# emopylab 2026
"""NNDREA-MO (evolutionary algorithm with neural network-based dimensionality reduction).

Reference:
Y. Tian, L. Wang, S. Yang, J. Ding, Y. Jin, and X. Zhang. Neural network-based dimensionality
reduction for large-scale binary optimization with millions of variables. IEEE Transactions on
Evolutionary Computation, 2025, 29(6): 2328-2342.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, first_front, ga_half, nd_sort, objs, tournament
from algorithms.hype.hype import cal_hv
from algorithms.moea_dd.moea_dd import update_front
from core.population import Population

ALGORITHM_FLAGS = {'NNDREAMO': {'binary', 'constrained', 'large', 'multi', 'sparse'}}


def _sbx_pm(parent, lower, upper, rng, proC=1.0, disC=20, proM=1.0, disM=20):
    h = len(parent) // 2
    P1, P2 = parent[:h], parent[h:2 * h]
    N, D = P1.shape
    mu = rng.random((N, D))
    beta = np.zeros((N, D))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta = beta * (-1.0) ** rng.integers(0, 2, (N, D))
    beta[rng.random((N, D)) < 0.5] = 1
    beta[np.repeat(rng.random((N, 1)) > proC, D, axis=1)] = 1
    off = np.vstack([(P1 + P2) / 2 + beta * (P1 - P2) / 2, (P1 + P2) / 2 - beta * (P1 - P2) / 2])
    Lo, Up = np.tile(lower, (2 * N, 1)), np.tile(upper, (2 * N, 1))
    site = rng.random((2 * N, D)) < proM / D
    mu = rng.random((2 * N, D))
    off = np.minimum(np.maximum(off, Lo), Up)
    span = Up - Lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - Lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


def _leaky(x):
    return np.where(x > 0, x, 0.01 * x)


def fcn_forward(W, instance, s_list):
    """Feed-forward evaluation of every weight vector (rows of ``W``) on the instance matrix; the last layer is a
    step function, so the output is one binary decision vector per weight vector."""
    out = np.zeros((len(W), len(instance)))
    for n in range(len(W)):
        o = instance
        ptr = 0
        for i, (a, b) in enumerate(s_list):
            if b != -1:
                w = W[n, ptr:ptr + a * b]
                ptr += a * b
                o = o @ w.reshape(a, b, order="F")
            else:
                o = o + W[n, ptr:ptr + a]
                ptr += a
                if i == len(s_list) - 1:
                    out[n] = (o > 0).reshape(-1)
                else:
                    o = _leaky(o)
    return out


def _cal_hv_contrib(F, rng):
    return cal_hv(F, F.max(axis=0) * 1.1, 1, 10000, rng)


class NNDREAMO(LoopAlgorithm):
    """Neural-network-based dimensionality reduction for sparse combinatorial problems (knapsack, instance selection,
    community detection / network construction): the search runs on the weights of a tiny feed-forward network that
    maps the instance features to a binary decision vector; the second half of the budget refines the decision
    vectors directly with steady-state hypervolume-contribution reduction."""

    def __init__(self, pop_size: int = 100, lower: float = -1, upper: float = 1, delta: float = 0.5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.lo_w, self.up_w, self.delta = float(lower), float(upper), float(delta)

    def _instance(self):
        pr, rng = self.problem, self.rng
        name = type(pr).__name__
        if name in ("MOKP", "Sparse_KP"):
            return rng.normal(np.vstack([np.asarray(pr.P, dtype=float), np.asarray(pr.W, dtype=float)]).T, 0.1)
        if name == "Sparse_IS":
            return rng.normal(np.asarray(pr.Data, dtype=float), 0.1)
        if name in ("Sparse_CD", "Sparse_CN"):
            A = np.asarray(pr.Adj if name == "Sparse_CD" else pr.A, dtype=float)
            d = A.sum(axis=1)
            with np.errstate(all="ignore"):
                dm = np.where(d > 0, d ** -0.5, 0.0)
            V = np.linalg.eigh(dm[:, None] * A * dm[None, :])[1]          # eigenvectors by ascending eigenvalue
            return V[:, :10]
        return self.rng.random((self.problem.n_var, self.problem.n_var))

    def _initialize_infill(self):
        N, rng = self.N, self.rng
        self.instance = self._instance()
        st = [max(self.problem.n_var, self.instance.shape[1]), 4, 1]
        self.s_list = []
        dim = 0
        for i in range(len(st) - 1):
            self.s_list += [(st[i], st[i + 1]), (st[i + 1], -1)]
            dim += st[i] * st[i + 1] + st[i + 1]
        self.lo = np.full(dim, self.lo_w)
        self.up = np.full(dim, self.up_w)
        self.W = self.lo + rng.random((N, dim)) * (self.up - self.lo)
        return self.evaluate(fcn_forward(self.W, self.instance, self.s_list))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._select_w(self.pop, self.W)
        self._set_optimum()

    def _select_w(self, pop, W):
        N = self.N
        C = cons(pop)
        front, maxf = nd_sort(objs(pop), C if C.size else None, N)
        nxt = front < maxf
        cd = crowding(objs(pop), front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        self.pop, self.W, self.front, self.crowd = pop[nxt], W[nxt], front[nxt], cd[nxt]

    def _reduce(self, pop):
        F = objs(pop)
        self.front = update_front(F, self.front)
        last = np.where(self.front == self.front.max())[0]
        P = F[last]
        n, M = P.shape
        delta = np.full(n, np.inf)
        if M == 2:
            rank = np.lexsort((P[:, 1], P[:, 0]))
            for i in range(1, n - 1):
                delta[rank[i]] = (P[rank[i + 1], 0] - P[rank[i], 0]) * (P[rank[i - 1], 1] - P[rank[i], 1])
        elif n > 1:
            delta = _cal_hv_contrib(P, self.rng)
        worst = int(last[int(np.argmin(delta))])
        self.front = update_front(F, self.front, worst)
        return pop[np.delete(np.arange(len(pop)), worst)]

    def step(self):
        N, rng = self.N, self.rng
        if self.FE <= self.max_FE * self.delta:
            pool = tournament(2, N, self.front, -self.crowd, rng=rng)
            off_w = _sbx_pm(self.W[pool], self.lo, self.up, rng)
            off = self.evaluate(fcn_forward(off_w, self.instance, self.s_list))
            self._select_w(Population.merge(self.pop, off), np.vstack([self.W, off_w]))
        else:
            for _ in range(N):
                pick = rng.permutation(len(self.pop))[:2]
                off = self.evaluate(ga_half(self.problem, np.array([np.asarray(self.pop[int(k)].X, float) for k in pick]), rng=rng))
                C = cons(off)
                if (not C.size) or np.all(C <= 0):
                    self.pop = self._reduce(Population.merge(self.pop, off))
