# emopylab 2026
"""MMEAPSL (multimodal multi-objective evolutionary algorithm assisted by Pareto set learning).

Reference:
F. Ming, W. Gong, and Y. Jin. Growing neural gas network-based surrogate-assisted Pareto set
learning for multimodal multi-objective optimization. Swarm and Evolutionary Computation, 2024, 87:
101541.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, ga_half, nd_sort, objs, tournament, truncate_lexi
from core.population import Population

ALGORITHM_FLAGS = {'MMEAPSL': {'integer', 'multi', 'multimodal', 'real'}}


def _pd(A, B):
    return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0.0))


def _kth(Dm, k):
    Dm = Dm.copy()
    np.fill_diagonal(Dm, np.inf)
    return np.sort(Dm, axis=1)[:, k - 1]


def cal_fitness(F, X):
    """Strength rank plus objective-space and decision-space density; returns (D_dec, D_pop, fitness)."""
    N = len(F)
    lt = (F[:, None] < F[None]).any(-1)
    gt = (F[:, None] > F[None]).any(-1)
    dom = lt & ~gt
    S = dom.sum(1)
    R = np.array([S[dom[:, i]].sum() for i in range(N)], float)
    k = int(np.floor(np.sqrt(N)))
    d_pop = 1.0 / (_kth(_pd(F, F), k) + 2)
    d_dec = 1.0 / (_kth(_pd(X, X), k) + 2)
    return d_dec, d_pop, R + d_pop + d_dec


class GNG:
    """Growing neural gas state (nodes ``w``, errors ``E``, edges ``C``, edge ages ``t``, stagnation counters)."""

    def __init__(self, w):
        n = len(w)
        self.w, self.E = w.copy(), np.zeros(n)
        self.C, self.t = np.zeros((n, n)), np.zeros((n, n))
        self.nx = 0

    def _drop(self, rm, extra=True):
        keep = ~rm
        self.C, self.t = self.C[np.ix_(keep, keep)], self.t[np.ix_(keep, keep)]
        self.w, self.E = self.w[keep], self.E[keep]
        if extra:
            self.age_before, self.flag = self.age_before[keep], self.flag[keep]

    def _insert(self, q, f, alpha, extra=True):
        n = len(self.w)
        self.w = np.vstack([self.w, (self.w[q] + self.w[f]) / 2])
        C, t = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))
        C[:n, :n], t[:n, :n] = self.C, self.t
        C[q, f] = C[f, q] = 0
        C[q, n] = C[n, q] = C[n, f] = C[f, n] = 1
        self.C, self.t = C, t
        self.E[q] *= alpha
        self.E[f] *= alpha
        self.E = np.append(self.E, self.E[q])
        if extra:
            self.age_before = np.append(self.age_before, 0.0)
            self.flag = np.append(self.flag, 0.0)

    def adapt(self, x, p, extra=True):
        if len(self.w) < 2:
            return
        self.nx += 1
        d = np.sqrt(((self.w - x) ** 2).sum(1))
        order = np.argsort(d, kind="stable")
        s1, s2 = order[0], order[1]
        self.t[s1, :] += 1
        self.t[:, s1] += 1
        self.E[s1] += d[s1] ** 2
        self.w[s1] += p["epsilon_b"] * (x - self.w[s1])
        for j in np.where(self.C[s1] == 1)[0]:
            self.w[j] += p["epsilon_n"] * (x - self.w[j])
        self.C[s1, s2] = self.C[s2, s1] = 1
        self.t[s1, s2] = self.t[s2, s1] = 0
        self.C[self.t > p["T"]] = 0
        alone = self.C.sum(0) == 0
        if alone.any():
            self._drop(alone, extra)

    def grow(self, alpha, extra=True):
        q = int(np.argmax(self.E))
        f = int(np.argmax(self.C[:, q] * self.E))
        self._insert(q, f, alpha, extra)

    def age_sums(self):
        return np.array([self.t[i, self.C[i] == 1].sum() for i in range(len(self.w))])


def init_gng(pop, p, fitness, rng):
    F, X = objs(pop), decs(pop)
    front, _ = nd_sort(F, None, p["N"])
    valid = front == 1
    ref = int(valid.sum())
    if ref > 2:
        P = X[valid]
    else:
        P = X[fitness < 1]
        ref = len(pop)
    net = GNG(P[rng.permutation(ref)[:2]] if ref >= 2 else P)
    for _ in range(p["MaxIt"]):
        for kk in range(2, ref):
            net.adapt(P[kk], p, extra=False)
            if net.nx % p["L"] == 0 and len(net.w) < p["N"]:
                net.grow(p["alpha"], extra=False)
            net.E = p["delta"] * net.E
    net.age_before = net.age_sums()
    net.flag = np.zeros(len(net.w))
    return net


def train_gng(P, net, p, gen, maxgen, gen_flag):
    N = p["N"]
    ages = net.age_sums()
    net.flag = net.flag + (ages == net.age_before)
    net.age_before = ages
    maxN = 1.5
    cap = int(round(maxN * N))
    if gen <= round(0.9 * maxgen):
        max_iter, max_pz = 1, maxN
        if len(net.w) == cap:
            r = np.argsort(-net.flag, kind="stable")[: cap - N]
            rm = np.zeros(len(net.w), bool)
            rm[r] = True
            net._drop(rm)
            net.flag = np.zeros(N)
    else:
        if len(net.w) < cap and gen_flag is None:
            max_pz, max_iter = maxN, 1
        else:
            max_pz, max_iter = 1, 0
        if len(net.w) == cap:
            max_pz, max_iter, gen_flag = 1, 0, gen
    if gen_flag is None:
        L = p["L"]
        for _ in range(max_iter):
            for x in P:
                net.adapt(x, p)
                lim = int(round(max_pz * N))
                if net.nx % L == 0 and len(net.w) < lim:
                    net.grow(p["alpha"])
                    if net.nx % (2 * L) == 0 and len(net.w) < lim:
                        edge = net.C.sum(1) + net.C.sum(0)
                        q = int(np.argmin(edge))
                        Dc = np.abs(net.w[:, None] - net.w[None]).sum(-1)
                        np.fill_diagonal(Dc, np.inf)
                        dq = Dc[q].copy()
                        dq[net.C[q] == 1] = np.inf
                        dq[net.C[:, q] == 1] = np.inf
                        f = int(np.argmin(dq))
                        net._insert(q, f, p["alpha"])
                net.E = p["delta"] * net.E
    return net, gen_flag


class MMEAPSL(LoopAlgorithm):
    """SPEA2-style selection with decision- and objective-space densities; a growing neural gas learns the Pareto set from
    the non-dominated decision vectors. After the first 20% of generations offspring also come from the gas nodes and from
    a second population kept close to the nodes."""

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.p = dict(N=self.N, MaxIt=50, L=30, epsilon_b=0.2, epsilon_n=0.006, alpha=0.5, delta=0.995, T=30)
        self.gen_flag, self.net, self.gen, self.pop1 = None, None, 0, None
        # the reference unpacks (D_dec, D_pop, fitness) as (fitness, D_dec, ~) at this first call
        d_dec, d_pop, _ = cal_fitness(objs(infills), decs(infills))
        self.fitness, self.d_dec = d_dec, d_pop
        self._set_optimum()

    def _env_orig(self, pop):
        N, F, X = self.N, objs(pop), decs(pop)
        front, _ = nd_sort(F, None, N)
        nd = X[front == 1]
        gen, maxgen = int(np.ceil(self.FE / N)), int(np.ceil(self.max_FE / N))
        if len(nd) > 2 and not np.isnan(nd).any() and gen <= maxgen and self.gen_flag is None:
            self.net, self.gen_flag = train_gng(nd, self.net, self.p, gen, maxgen, self.gen_flag)
        d_dec, _, fit = cal_fitness(F, X)
        nxt = fit < 1
        if nxt.sum() < N:
            nxt[np.argsort(fit, kind="stable")[:N]] = True
        elif nxt.sum() > N:
            idx = np.where(nxt)[0]
            Dm = _pd(F[idx], F[idx]) + _pd(X[idx], X[idx])
            np.fill_diagonal(Dm, np.inf)
            nxt[idx[truncate_lexi(Dm, int(nxt.sum()) - N)]] = False
        idx = np.where(nxt)[0]
        o = np.argsort(fit[idx], kind="stable")
        idx = idx[o]
        return pop[idx], fit[idx], d_dec[idx]

    def _env_sup(self, pop):
        N, V, C = self.N, self.net.w, self.net.C
        Dm = _pd(V, V) * (C == 0)
        d = np.array([row[row != 0].min() if np.any(row != 0) else np.nan for row in Dm])
        theta = np.nanmax(d) if np.any(~np.isnan(d)) else np.inf
        X = decs(pop)
        fit = _pd(X, V).min(axis=1)
        nxt = fit < theta
        if nxt.sum() < N:
            nxt[np.argsort(fit, kind="stable")[:N]] = True
        elif nxt.sum() > N:
            idx = np.where(nxt)[0]
            Dx = _pd(X[idx], X[idx])
            np.fill_diagonal(Dx, np.inf)
            nxt[idx[truncate_lexi(Dx, int(nxt.sum()) - N)]] = False
        idx = np.where(nxt)[0]
        idx = idx[np.argsort(fit[idx], kind="stable")]
        return pop[idx], fit[idx]

    def step(self):
        rng, N = self.rng, self.N
        self.gen += 1
        if self.net is None and (self.fitness < 1).sum() >= 2:
            self.net = init_gng(self.pop, self.p, self.fitness, rng)
        maxgen = int(np.ceil(self.max_FE / N))
        if self.net is None or self.gen < 0.2 * maxgen:
            mate = tournament(2, N, self.d_dec, self.fitness, rng=rng)
            off = self.evaluate(ga(self.problem, decs(self.pop)[mate], rng=rng))
            self.pop, self.fitness, self.d_dec = self._env_orig(Population.merge(self.pop, off))
        else:
            if self.pop1 is None:
                self.pop1 = self.pop
                self.fit1 = _pd(decs(self.pop1), self.net.w).min(axis=1)
            m1 = tournament(2, N, self.d_dec, self.fitness, rng=rng)
            o1 = self.evaluate(ga_half(self.problem, decs(self.pop)[m1], rng=rng))
            V = self.net.w
            o2 = self.evaluate(ga_half(self.problem, V[rng.integers(0, len(V), N)], rng=rng))
            m3 = tournament(2, N, -self.fit1, rng=rng)
            o3 = self.evaluate(ga_half(self.problem, decs(self.pop1)[m3], rng=rng))
            off = Population.merge(o1, o2, o3)
            self.pop, self.fitness, self.d_dec = self._env_orig(Population.merge(self.pop, off))
            self.pop1, self.fit1 = self._env_sup(Population.merge(self.pop1, off))
