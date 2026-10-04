# emopylab 2026
"""MaOEA-IT (many-objective evolutionary algorithms based on an independent two-stage).

Reference:
Y. Sun, B. Xue, M. Zhang, and G. G. Yen. A new two-stage evolutionary algorithm for many-objective
optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(5): 748-761.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, first_front, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MaOEAIT': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _round_half_away(x, d):
    f = 10.0 ** d
    return np.sign(x) * np.floor(np.abs(x) * f + 0.5) / f


def find_subspace(X, eps):
    x = X.T
    x = x - x.mean(axis=0)
    sigma = x @ x.T / x.shape[0]
    s = np.linalg.svd(sigma, compute_uv=False)
    v = np.cumsum(s) / s.sum()
    i = 0
    while i < len(v) - 1 and v[i] < eps:
        i += 1
    x1 = X.copy()
    avg1 = _round_half_away(X.mean(axis=0), 2)
    x1[:, i + 1:] = avg1[i + 1:]
    return x1


def _cos_v(F, v):
    with np.errstate(all="ignore"):
        return -np.sum(F * v, axis=1) / np.sqrt(np.sum(F ** 2, axis=1))


class MaOEAIT(LoopAlgorithm):
    """Many-objective EA with information transfer: a first NDWA stage learns a low-dimensional subspace of the
    decision variables, then single-objective searches inside the subspace locate the extreme points and the
    scalarised solutions of a reference-point set."""

    def __init__(self, pop_size: int = 100, Evaluation1: int = 20000, Evaluation2: int = 6000, epsilon: float = 0.999,
                 sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Ev1, self.Ev2, self.eps = int(Evaluation1), int(Evaluation2), float(epsilon)

    def _operator(self, parents, lower, upper):
        rng = self.rng
        h = len(parents) // 2
        P1, P2 = parents[:h], parents[h:2 * h]
        N, D = P1.shape
        mu = rng.random((N, D))
        beta = np.zeros((N, D))
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / 21)
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / 21)
        beta = beta * (-1.0) ** rng.integers(0, 2, (N, D))
        beta[rng.random((N, D)) < 0.5] = 1
        off = np.vstack([(P1 + P2) / 2 + beta * (P1 - P2) / 2, (P1 + P2) / 2 - beta * (P1 - P2) / 2])
        n = len(off)
        Lo, Up = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
        site = rng.random((n, D)) < 1.0 / D
        mu = rng.random((n, D))
        off = np.minimum(np.maximum(off, Lo), Up)
        span = Up - Lo
        with np.errstate(all="ignore"):
            t = site & (mu <= 0.5)
            off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - Lo[t]) / span[t]) ** 21) ** (1 / 21) - 1)
            t = site & (mu > 0.5)
            off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - off[t]) / span[t]) ** 21) ** (1 / 21))
        return self.evaluate(off)

    def _single_objective(self, l, u, target, budget, sf):
        """One single-objective search inside [l, u] scalarised by ``cos_v(., target)``; returns (best, pop)."""
        N, D, rng = self.N, self.D, self.rng
        cur = self.FE
        pop = self.evaluate(rng.random((N, D)) * (u - l) + l)
        sf[:N] = _cos_v(objs(pop), target)
        while self.not_terminated(pop) and self.FE < cur + budget:
            pool = tournament(2, N, sf[:N], rng=rng)
            off = self._operator(decs(pop[pool]), l, u)
            sf[N:] = _cos_v(objs(off), target)
            rank = np.argsort(sf, kind="stable")
            pop = Population.merge(pop, off)[rank[:N]]
            sf[:N] = sf[rank[:N]]
        return pop

    def step(self):
        N, M, D, rng = self.N, self.M, self.D, self.rng
        pop = self.pop
        W, W_num = uniform_point(N, M)
        UW = np.vstack([W, W[::-1]])
        num_W = len(UW)
        SF = np.zeros(N + (N // 2) * 2)
        SF[:N] = np.sum(objs(pop) * UW[0], axis=1)
        archive = None
        while self.not_terminated(pop) and self.FE < self.Ev1:
            w_index = (int(np.ceil(self.FE / N)) + 1) % num_W
            if w_index == 0:
                w_index = num_W
            pool = tournament(2, N, SF[:N], rng=rng)
            off = self.evaluate(ga(self.problem, decs(pop[pool]), [0.9, 20, 1, 20], rng=rng))
            SF[N:] = np.sum(objs(off) * UW[w_index - 1], axis=1)
            merged = Population.merge(pop, off)
            C = cons(merged)
            front, maxf = nd_sort(objs(merged), C if C.size else None, N)
            nxt = front < maxf
            last = np.where(front == maxf)[0]
            nxt[last[: N - int(nxt.sum())]] = True
            SF[:N] = SF[nxt]
            pop = merged[nxt]
            archive = Population.merge(archive, pop)
            archive = archive[first_front(objs(archive))]
        self.not_terminated(pop)
        learned = find_subspace(decs(archive), self.eps)
        mean_val = learned.mean(axis=0)
        lo, up = self.lower, self.upper
        u_lim, l_lim = np.ones(D), np.zeros(D)
        for i in range(D):
            if abs(learned[0, i] - mean_val[i]) < 0.1:
                l_lim[i] = u_lim[i] = _round_half_away(mean_val[i], 1)
            else:
                l_lim[i], u_lim[i] = lo[i], up[i]
        rf = np.eye(M)
        so_max = self.Ev2 // M
        extreme = []
        for i in range(M):
            pop = self._single_objective(l_lim, u_lim, rf[i], so_max, SF)
            extreme.append(pop[0])
        ext = Population.create(extreme)
        Fe = objs(ext)
        ideal, nadir = Fe.min(axis=0), Fe.max(axis=0)
        ref = W * (nadir - ideal) + ideal
        so_max = (self.max_FE - self.Ev1 - self.Ev2) // W_num
        result = [pop[0]] * len(ref)
        for i in range(W_num):
            cur = self.FE
            pop = self.evaluate(rng.random((N, D)) * (u_lim - l_lim) + l_lim)
            SF[:N] = _cos_v(objs(pop), ref[i])
            while self.not_terminated(pop) and self.FE < cur + so_max:
                pool = tournament(2, N, SF[:N], rng=rng)
                off = self._operator(decs(pop[pool]), l_lim, u_lim)
                SF[N:] = _cos_v(objs(off), ref[i])
                rank = np.argsort(SF, kind="stable")
                pop = Population.merge(pop, off)[rank[:N]]
                SF[:N] = SF[rank[:N]]
                if i == W_num - 1 and self.FE == cur + so_max:
                    result[W_num - 1] = pop[0]
                    pop = Population.create(result)
            result[i] = pop[0]
        self.pop = Population.create(result)
