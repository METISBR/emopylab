# emopylab 2026
"""CMOSMA (constrained multi-objective evolutionary algorithm with self-organizing map).

Reference:
C. He, M. Li, C. Zhang, H. Chen, P. Zhong, Z. Li, and J. Li. A self-organizing map approach for
constrained multi-objective optimization problems. Complex & Intelligent Systems, 2022, 8:
5355-5375.
"""

from __future__ import annotations

import itertools

import numpy as np

from algorithms.ccmo.ccmo import environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, decs, ga, pdist2
from core.population import Population

ALGORITHM_FLAGS = {'CMOSMA': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _initialize_som(D, H):
    grids = [np.arange(1, d + 1, dtype=float) for d in D]
    mesh = np.meshgrid(*grids, indexing="ij")
    Z = np.column_stack([m.reshape(-1, order="F") for m in mesh])
    LDis = pdist2(Z, Z)
    B = np.argsort(LDis, axis=1, kind="stable")[:, 1:min(H + 1, len(Z))]
    return LDis, B


def _update_som(S, W, FE, max_FE, LDis, sigma0, tau0):
    W = W.copy()
    for s in range(len(S)):
        sigma = sigma0 * (1 - (FE + s + 1) / max_FE)
        tau = tau0 * (1 - (FE + s + 1) / max_FE)
        u1 = int(np.argmin(np.linalg.norm(W - S[s], axis=1)))
        U = LDis[u1] < sigma
        W[U] += tau * np.exp(-LDis[u1, U])[:, None] * (S[s] - W[U])
    return W


def _associate(pop, W, N, rng):
    X = decs(pop)
    A, U = list(range(N)), list(range(N))
    XU = np.zeros(N, dtype=int)
    for _ in range(N):
        x = int(rng.integers(len(A)))
        u = int(np.argmin(np.linalg.norm(W[U] - X[A[x]], axis=1)))
        XU[U[u]] = A[x]
        del A[x], U[u]
    return XU


def _mating_pool(XU, N, B, rng):
    pool = np.zeros(N, dtype=int)
    for u in range(N):
        Q = XU[B[u]] if rng.random() < 0.9 else np.arange(N)
        pool[u] = Q[int(rng.integers(len(Q)))]
    return pool


def _setdiff_rows(A, B):
    have = {tuple(r) for r in B}
    new = sorted({tuple(r) for r in A if tuple(r) not in have})
    return np.array(new, dtype=float).reshape(-1, A.shape[1])


class CMOSMA(LoopAlgorithm):
    """Two co-evolving populations (constrained / unconstrained) whose mating is organised by self-organizing
    maps trained on the newly produced decision vectors."""

    def __init__(self, pop_size: int = 100, D=None, tau0: float = 0.7, H: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Dgrid, self.tau0, self.H = D, float(tau0), int(H)

    def initial_size(self):
        M = self.M
        D = np.full(M - 1, int(np.ceil(self.pop_size ** (1.0 / (M - 1))))) if self.Dgrid is None else np.asarray(self.Dgrid)
        self.Dg = D
        self.pop_size = int(np.prod(D))
        self.sigma0 = np.sqrt(np.sum(D.astype(float) ** 2) / (M - 1)) / 2
        return self.pop_size

    def start(self):
        self.FP = self.pop
        self.AP = self.evaluate(self.random_decs(self.N))
        self.S, self.W = decs(self.FP), decs(self.FP)
        self.S2, self.W2 = decs(self.AP), decs(self.AP)
        self.LDis, self.B = _initialize_som(self.Dg, self.H)
        self.LDis2, self.B2 = self.LDis, self.B

    def step(self):
        N, rng, FE, mx = self.N, self.rng, self.FE, self.max_FE
        self.W = _update_som(self.S, self.W, FE, mx, self.LDis, self.sigma0, self.tau0)
        self.W2 = _update_som(self.S2, self.W2, FE, mx, self.LDis2, self.sigma0, self.tau0)
        XU = _associate(self.FP, self.W, N, rng)
        XU2 = _associate(self.AP, self.W2, N, rng)
        mp1, mp2 = _mating_pool(XU, N, self.B, rng), _mating_pool(XU2, N, self.B2, rng)
        A1, A2 = decs(self.FP), decs(self.AP)
        off1 = self.evaluate(ga(self.problem, np.vstack([decs(self.FP[XU]), decs(self.FP[mp1])]), rng=rng))
        off2 = self.evaluate(ga(self.problem, np.vstack([decs(self.AP[XU2]), decs(self.AP[mp2])]), rng=rng))
        self.FP, _ = environmental_selection(Population.merge(self.FP, off1, off2), N, True)
        self.AP, _ = environmental_selection(Population.merge(self.AP, off1, off2), N, False)
        self.S = _setdiff_rows(decs(self.FP), A1)
        self.S2 = _setdiff_rows(decs(self.AP), A2)
        self.pop = self.FP
