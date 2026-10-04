# emopylab 2026
"""SSDE (self-organized surrogate-assisted differential evolution).

Reference:
A. F. R. Araújo, L. R. C. Farias, and A. R. C. Gonçalves. Self-organizing surrogate-assisted non-
dominated sorting differential evolution. Swarm and Evolutionary Computation, 2024, 91: 101703.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, de, decs, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'SSDE': {'constrained', 'integer', 'multi', 'real'}}


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


class SSDE(LoopAlgorithm):
    """Self-organizing surrogate-free differential evolution: a 1-D self-organizing map over (decision, objective)
    samples proposes target objective vectors that pre-screen DE offspring before evaluation."""

    def __init__(self, pop_size: int = 100, num_nodes=None, eta0: float = 0.2, sigma0=None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.num_nodes, self.eta0, self.sigma0 = num_nodes, float(eta0), sigma0

    def start(self):
        N, D, M, rng = self.N, self.D, self.M, self.rng
        self.nn = int(N if self.num_nodes is None else self.num_nodes)
        self.sigma = float(N if self.sigma0 is None else self.sigma0)
        self.samples = self.pop
        self.W = 0.001 * rng.standard_normal((self.nn, D + M)) + 0.5
        self.win = np.zeros(N, bool)
        v = np.arange(1, self.nn + 1, dtype=float)[:, None]
        self.LDis = pdist2(v, v)

    def _rescale(self, X):
        return (X - self.lower) / (self.upper - self.lower)

    def _training(self):
        D, M, nn, rng, W = self.D, self.M, self.nn, self.rng, self.W
        win = self.win
        if win.any() and win.sum() < nn:
            nw = int(win.sum())
            Wobj = W[win, D:]
            front, _ = nd_sort(Wobj, None, nw)
            cd = crowding(Wobj, front)
            p = np.argsort(cd, kind="stable")
            factor = np.arange(1, nw + 1, dtype=float)
            factor[p] = np.arange(1, nw + 1, dtype=float)
            n_new = nn - nw
            chosen = rng.choice(nw, size=2 * n_new, replace=True, p=factor / factor.sum())
            c1, c2 = chosen[:n_new], chosen[n_new:]
            new = (W[c1] + W[c2]) / 2 + 0.001 * rng.standard_normal((n_new, D + M))
            W[~win] = new
            W[:, :D] = np.clip(W[:, :D], 0, 1)
        S = np.hstack([decs(self.samples), objs(self.samples)])
        S[:, :D] = self._rescale(S[:, :D])
        nS = len(S)
        count = np.zeros(nn)
        for epoch in range(1, 51):
            randpos = rng.permutation(nS)
            sig = self.sigma * np.exp(-count / nS)
            eta = self.eta0 * np.exp(-count / nS)
            for s in randpos:
                u1 = int(np.argmin(np.linalg.norm(W[:, :D] - S[s, :D], axis=1)))
                if count[u1] == 0:
                    W[u1] = S[s]
                if count[u1] < epoch:
                    count[u1] += 1
                U = self.LDis[u1] < sig
                W[U] += eta[U][:, None] * np.exp(-self.LDis[u1, U])[:, None] * (S[s] - W[U])

    def step(self):
        N, D, M, rng, pop = self.N, self.D, self.M, self.rng, self.pop
        if len(self.samples) >= N:
            self._training()
            self.win = np.zeros(self.nn, bool)
            self.samples = None
        W = self.W
        pool = tournament(2, N, np.sum(np.maximum(0, _con(pop)), axis=1), rng=rng)
        X = decs(pop)
        off = de(self.problem, X[pool], X[rng.integers(0, N, N)], X[rng.integers(0, N, N)], rng=rng)
        dist = pdist2(self._rescale(off), W[:, :D])
        u = np.argmin(dist, axis=1)
        labels = W[u, D:D + M]
        self.win[u] = True
        F = np.vstack([objs(pop), labels])
        C = np.vstack([_con(pop), np.zeros((len(labels), _con(pop).shape[1]))])
        front, maxf = nd_sort(F, C, N)
        nxt = front < maxf
        cd = crowding(F, front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        out, inn = nxt[:N], nxt[N:]
        if inn.any():
            offspring = self.evaluate(off[inn])
            for slot, ind in zip(np.where(~out)[0], offspring):
                pop[slot] = ind
            new_samples = offspring
        else:
            new_samples = pop
        self.samples = Population.merge(self.samples, new_samples)
        self.pop = pop
