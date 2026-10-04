# emopylab 2026
"""WOF (weighted optimization framework).

Reference:
H. Zille, H. Ishibuchi, S. Mostaghim, and Y. Nojima. A framework for large-scale multiobjective
optimization based on problem transformation. IEEE Transactions on Evolutionary Computation, 2018,
22(2): 260-275. ----------------------------------------------------------------------- Copyright
(C) 2020 Heiner Zille This work is licensed under the Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International License. (CC BY-NC-SA 4.0). To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/ or see the pdf-file "License-CC-BY-NC-SA-4.0.pdf"
that came with this code. You are free to: * Share ? copy and redistribute the material in any
medium or format * Adapt ? remix, transform, and build upon the material Under the following terms:
* Attribution ? You must give appropriate credit, provide a link to the license, and indicate if
changes were made. You may do so in any reasonable manner, but not in any way that suggests the
licensor endorses you or your use. * NonCommercial ? You may not use the material for commercial
purposes. * ShareAlike ? If you remix, transform, or build upon the material, you must distribute
your contributions under the same license as the original. * No additional restrictions ? You may
not apply legal terms or technological measures that legally restrict others from doing anything the
license permits. Author of this Code: Heiner Zille <heiner.zille@ovgu.de> or
<heiner.zille@gmail.com> This code is based on the following publications: 1) Heiner Zille "Large-
scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art" PhD
Thesis, Otto von Guericke University Magdeburg, 2019 http://dx.doi.org/10.25673/32063 2) Heiner
Zille and Sanaz Mostaghim "Comparison Study of Large-scale Optimisation Techniques on the LSMOP
Benchmark Functions" IEEE Symposium Series on Computational Intelligence (SSCI), IEEE, Honolulu,
Hawaii, November 2017 https://ieeexplore.ieee.org/document/8280974 3) Heiner Zille, Hisao Ishibuchi,
Sanaz Mostaghim and Yusuke Nojima "A Framework for Large-scale Multi-objective Optimization based on
Problem Transformation" IEEE Transactions on Evolutionary Computation, Vol. 22, Issue 2, pp.
260-275, April 2018. http://ieeexplore.ieee.org/document/7929324 4) Heiner Zille, Hisao Ishibuchi,
Sanaz Mostaghim and Yusuke Nojima "Weighted Optimization Framework for Large-scale Mullti-objective
Optimization"
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import nsga3_ref
from algorithms.community_utils.base import LoopAlgorithm, adds, cons, crowding, decs, ga, nd_sort, neighbors_of, objs, tournament, uniform_point
from algorithms.glmo.glmo import create_groups
from core.population import Population

ALGORITHM_FLAGS = {'WOF': {'integer', 'large', 'multi', 'real'}}


# --------------------------------------------------------------------------------------------------
# helpers shared by the four optimisers (SMPSO 1, MOEA/D 2, NSGA-II 3, NSGA-III 4)
# --------------------------------------------------------------------------------------------------
def _transform(xprime, weight, upper, lower, method):
    """Move ``xprime`` by ``weight``: product (1), shift by a share of the range (2) or interval scaling (3)."""
    if method == 1:
        return xprime * weight
    if method == 2:
        return xprime + 0.2 * (weight - 1.0) * (upper - lower)
    lo = lower + weight * (xprime - lower)
    return np.where(weight > 1.0, xprime + (weight - 1.0) * (upper - xprime), lo)


def _nsga2_sel(pop, N):
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


def _update_gbest(pop, N):
    pop = pop[nd_sort(objs(pop), None, 1)[0] == 1]
    cd = crowding(objs(pop))
    rank = np.argsort(-cd, kind="stable")[: min(N, len(pop))]
    return pop[rank], cd[rank]


def _update_pbest(pbest, pop):
    replace = ~np.all(objs(pop) >= objs(pbest), axis=1)
    out = pbest[np.arange(len(pbest))]
    out[replace] = pop[replace]
    return out


def _poly(X, lower, upper, site, mu, disM=20.0):
    span = upper - lower
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] = X[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lower[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] = X[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (upper[t] - X[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return X


def _ga(rng, parents, lower, upper, half=False):
    """SBX (probability 1, index 20) and polynomial mutation (1/D, index 20) within the given bounds."""
    P = np.asarray(parents, dtype=float)
    h = len(P) // 2
    P1, P2 = P[:h], P[h: 2 * h]
    n, D = P1.shape
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
    beta[rng.random((n, D)) < 0.5] = 1
    off = (P1 + P2) / 2 + beta * (P1 - P2) / 2
    if not half:
        off = np.vstack([off, (P1 + P2) / 2 - beta * (P1 - P2) / 2])
    m = len(off)
    lo, up = np.tile(lower, (m, 1)), np.tile(upper, (m, 1))
    off = np.minimum(np.maximum(off, lo), up)
    return _poly(off, lo, up, rng.random((m, D)) < 1 / D, rng.random((m, D)))


class _Space:
    """The space an optimiser works in: the real problem, or the low-dimensional space of group weights (each weight
    vector is evaluated through the transformation of the fixed solution ``xprime``)."""

    def __init__(self, algo, N, lower, upper, dummy=None):
        self.algo, self.N, self.lower, self.upper, self.dummy = algo, N, lower, upper, dummy

    def make(self, X, V=None):
        algo = self.algo
        if self.dummy is None:
            return algo.evaluate(X) if V is None else algo.evaluate(X, V=V)
        d = self.dummy
        W = np.minimum(np.maximum(np.atleast_2d(X), self.lower), self.upper)
        full = _transform(d["xprime"], W[:, d["G"]], algo.upper, algo.lower, d["psi"])
        real = algo.evaluate(full)
        pop = Population.new("X", W, "F", objs(real), *(["G", real.get("G")] if real.get("G") is not None else []))
        if V is not None:
            for ind, v in zip(pop, V):
                ind.set("V", v)
        return pop


def _opt_nsga2(algo, sp, pop, evaluations):
    maximum = algo.FE + evaluations
    _, front, crowd = _nsga2_sel(pop, sp.N)
    while algo.FE < maximum:
        mate = tournament(2, sp.N, front, -crowd, rng=algo.rng)
        off = sp.make(_ga(algo.rng, decs(pop[mate]), sp.lower, sp.upper))
        pop, front, crowd = _nsga2_sel(Population.merge(pop, off), sp.N)
    return pop


def _feasible_min(pop):
    C = cons(pop)
    feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
    return objs(pop)[feas].min(axis=0) if feas.any() else None


def _opt_nsga3(algo, sp, pop, Z, evaluations):
    maximum = algo.FE + evaluations
    zmin = _feasible_min(pop)
    while algo.FE < maximum:
        C = cons(pop)
        mate = tournament(2, sp.N, np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop)), rng=algo.rng)
        off = sp.make(_ga(algo.rng, decs(pop[mate]), sp.lower, sp.upper))
        fo = _feasible_min(off)
        if fo is not None:
            zmin = fo if zmin is None else np.minimum(zmin, fo)
        pop = nsga3_ref.select(Population.merge(pop, off), sp.N, Z, zmin, algo.rng)
    return pop


def _opt_moead(algo, sp, pop, W, evaluations):
    maximum = algo.FE + evaluations
    N = sp.N
    T = max(int(np.ceil(N / 10)), 2)
    B = neighbors_of(W, T)
    F = objs(pop)
    Z = F.min(axis=0)
    with np.errstate(all="ignore"):
        g = np.array([np.max(np.abs(F[i] - Z) / W, axis=1) for i in range(N)])
    rank = np.argsort(g, axis=1, kind="stable")
    assoc = np.zeros(N, dtype=int) - 1
    taken = np.zeros(N, bool)
    for i in range(N):
        slot = rank[i][np.where(~taken[rank[i]])[0][0]]
        assoc[slot], taken[slot] = i, True
    pop = pop[assoc]
    while algo.FE < maximum:
        for i in range(N):
            P = B[i][algo.rng.permutation(B.shape[1])]
            off = sp.make(_ga(algo.rng, decs(pop[P[:2]]), sp.lower, sp.upper, half=True))
            fo = objs(off)[0]
            Z = np.minimum(Z, fo)
            Fp = objs(pop[P])
            nW = np.linalg.norm(W[P], axis=1)
            nP, nO = np.linalg.norm(Fp - Z, axis=1), np.linalg.norm(fo - Z)
            with np.errstate(all="ignore"):
                cP = np.sum((Fp - Z) * W[P], axis=1) / nW / nP
                cO = np.sum((fo - Z) * W[P], axis=1) / nW / nO
                g_old = nP * cP + 5 * nP * np.sqrt(1 - cP ** 2)
                g_new = nO * cO + 5 * nO * np.sqrt(1 - cO ** 2)
            for j in np.where(g_old >= g_new)[0]:
                pop[P[j]] = off[0]
    return pop


def _smpso_operator(algo, sp, particles):
    n = len(particles)
    idx = np.concatenate([np.arange(n), np.arange(int(np.ceil(n / 3)) * 3 - n)])
    particles = particles[idx]
    X = decs(particles)
    n, D = X.shape
    k = n // 3
    V = adds(particles, "V", np.zeros((n, D)))
    Xp, Vp, Pb, Gb = X[:k], V[:k], X[k: 2 * k], X[2 * k:]
    rng = algo.rng
    W = np.repeat(rng.uniform(0.1, 0.5, (k, 1)), D, axis=1)
    r1, r2 = np.repeat(rng.random((k, 1)), D, axis=1), np.repeat(rng.random((k, 1)), D, axis=1)
    C1, C2 = np.repeat(rng.uniform(1.5, 2.5, (k, 1)), D, axis=1), np.repeat(rng.uniform(1.5, 2.5, (k, 1)), D, axis=1)
    nv = W * Vp + C1 * r1 * (Pb - Xp) + C2 * r2 * (Gb - Xp)
    phi = np.maximum(4, C1 + C2)
    nv = nv * 2 / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
    delta = np.tile((sp.upper - sp.lower) / 2, (k, 1))
    nv = np.maximum(np.minimum(nv, delta), -delta)
    nx = Xp + nv
    lo, up = np.tile(sp.lower, (k, 1)), np.tile(sp.upper, (k, 1))
    nv[(nx < lo) | (nx > up)] *= 0.001
    nx = np.maximum(np.minimum(nx, up), lo)
    site = np.repeat(rng.random((k, 1)) < 0.15, D, axis=1) & (rng.random((k, D)) < 1 / D)
    nx = _poly(nx, lo, up, site, rng.random((k, D)))
    return sp.make(nx, V=nv)


def _opt_smpso(algo, sp, pop, evaluations):
    pbest = pop
    gbest, crowd = _update_gbest(pop, sp.N)
    maximum = algo.FE + evaluations
    while algo.FE < maximum:
        pick = gbest[tournament(2, sp.N, -crowd, rng=algo.rng)]
        pop = _smpso_operator(algo, sp, Population.merge(pop, pbest, pick))
        gbest, crowd = _update_gbest(Population.merge(gbest, pop), sp.N)
        pbest = _update_pbest(pbest, pop)
    return gbest


def _select_xprimes(pop, amount, method, rng):
    n = len(pop)
    F = objs(pop)
    if method == 1:
        front, _ = nd_sort(F, None, np.inf)
        out, i = [], 1
        if n < amount:
            return pop
        while len(out) < amount:
            left = amount - len(out)
            sub = pop[front == i]
            f = nd_sort(objs(sub), None, np.inf)[0]
            cd = crowding(objs(sub), f)
            order = np.argsort(-cd, kind="stable")
            out.append(sub[order[: min(left, len(sub))]])
            i += 1
        return Population.merge(*out)
    if method == 2:
        front, _ = nd_sort(F, None, np.inf)
        cd = crowding(F, front)
        return pop[tournament(2, amount, front, -cd, rng=rng)]
    m = F.shape[1]
    pick = []

    def closest(vec):
        with np.errstate(all="ignore"):
            d = 1 - (F @ vec) / (np.linalg.norm(F, axis=1) * np.linalg.norm(vec))
        return int(np.argmin(np.where(np.isnan(d), np.inf, d))) if not np.all(np.isnan(d)) else 0

    for i in range(m):
        v = np.zeros(m)
        v[i] = 1
        pick.append(closest(v))
    if len(pick) < amount:
        pick.append(closest(np.ones(m)))
    while len(pick) < amount:
        pick.append(int(rng.integers(0, n)))
    return pop[np.array(pick)]


class WOF(LoopAlgorithm):
    """Variables are grouped and each group is scaled by one weight; the (few) weights of a chosen solution
    ``xprime`` are optimised with one of SMPSO (1), MOEA/D (2), NSGA-II (3), NSGA-III (4), and the resulting weight
    vectors are applied to the whole population.  The first ``delta`` of the budget alternates normal optimisation
    (``t1`` evaluations) with weight optimisation (``t2``); the remainder is normal optimisation."""

    def __init__(self, pop_size: int = 100, gamma: int = 4, groups: int = 2, psi: int = 3, t1: int = 1000, t2: int = 500,
                 q=None, delta: float = 0.5, optimiser: int = 1, random_optimisers: bool = True, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.gamma, self.group_method, self.psi, self.t1, self.t2 = int(gamma), int(groups), int(psi), int(t1), int(t2)
        self.q, self.delta, self.optimiser, self.dice = q, float(delta), int(optimiser), bool(random_optimisers)

    def initial_size(self):
        if self.optimiser in (2, 4) or self.dice:
            self.uniW, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.qn = self.M + 1 if self.q is None else int(self.q)
        self.dummy_size = 10
        self.restart = True

    # -- pieces of the reference framework -------------------------------------------------------------
    def _rand_type(self):
        return int(self.rng.integers(1, 5))

    def _fill(self, pop):
        N = self.N
        if len(pop) >= N:
            return pop
        amount = N - len(pop)
        F = objs(pop)
        front, _ = nd_sort(F, None, np.inf)
        cd = crowding(F, front)
        mate = tournament(2, amount + 1, front, -cd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(pop[mate]), rng=self.rng))
        return Population.merge(pop, off[:amount])

    def _optimise(self, sp, pop, uniW, evaluations, opt):
        if opt == 4:
            return _opt_nsga3(self, sp, pop, uniW, evaluations)
        if opt == 3:
            return _opt_nsga2(self, sp, pop, evaluations)
        if opt == 2:
            return _opt_moead(self, sp, pop, uniW, evaluations)
        return _opt_smpso(self, sp, pop, evaluations)

    def _groups(self, xprime):
        method, D, g = self.group_method, self.D, self.gamma
        if method == 4:
            fx = objs(xprime)[0]
            base = decs(xprime)[0]
            out = np.zeros(D, dtype=int)
            for i in range(D):
                x = base.copy()
                x[i] = base[i] * 1.05
                if objs(self.evaluate(x[None, :]))[0, 0] < fx[0]:
                    out[i] = 1
            return out, 2
        return create_groups(g, decs(xprime), method, self.rng)[0] - 1, g

    def _extract(self, weight_pop, pop, G, xprime, dummy):
        rng = self.rng
        chosen = _select_xprimes(weight_pop, self.qn, 3, rng)
        X = decs(pop)
        rows = []
        for w in decs(chosen):
            for x in X:
                rows.append(_transform(x, w[G], self.upper, self.lower, self.psi))
        W1 = self.evaluate(np.array(rows))
        xp = decs(xprime)[0]
        rows = [_transform(xp, w[G], self.upper, self.lower, self.psi) for w in decs(weight_pop)]
        W2 = self.evaluate(np.array(rows))
        return Population.merge(W1, W2)

    # -- one round of the framework ---------------------------------------------------------------------
    def step(self):
        if self.FE < self.delta * self.max_FE:
            self._weighted_round()
            return
        while True:
            self.not_terminated(self.pop)
            self.pop = self._fill(self.pop)
            opt = 4 if self.dice else self.optimiser
            sp = _Space(self, self.N, self.lower, self.upper)
            self.pop = self._optimise(sp, self.pop, getattr(self, "uniW", None), self.t1, opt)
            return

    def _weighted_round(self):
        M, N = self.M, self.N
        pop = self.pop
        opt = self._rand_type() if self.dice else self.optimiser
        pop = self._fill(pop)
        real = _Space(self, N, self.lower, self.upper)
        pop = self._optimise(real, pop, getattr(self, "uniW", None), self.t1, opt)
        self.not_terminated(pop)
        xprimes = _select_xprimes(pop, self.qn, 3, self.rng)
        gathered = []
        for c in range(len(xprimes)):
            xprime = xprimes[[c]]
            G, gamma = self._groups(xprime)
            if self.dice:
                opt = self._rand_type()
            if opt in (2, 4):
                uniW, dN = uniform_point(self.dummy_size, M)
            else:
                uniW, dN = None, self.dummy_size
            dummy = {"xprime": decs(xprime)[0], "G": G, "psi": self.psi}
            sp = _Space(self, dN, np.zeros(gamma), np.full(gamma, 2.0), dummy)
            weights = self.rng.random((dN, gamma)) * 2.0
            wpop = sp.make(weights)
            wpop = self._optimise(sp, wpop, uniW, self.t2 - self.dummy_size, opt)
            gathered.append(self._extract(wpop, pop, G, xprime, dummy))
        allp = Population.merge(pop, *gathered)
        allp = allp[np.unique(objs(allp), axis=0, return_index=True)[1]]
        allp = self._fill(allp)
        pop, _, _ = self._survivors(allp, N)
        self.pop = pop
        self.not_terminated(pop)

    def _survivors(self, pop, N):
        F = objs(pop)
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        cd = crowding(F, front)
        last = np.where(front == maxf)[0]
        size = min(N, len(pop))
        nxt[last[np.argsort(-cd[last], kind="stable")[: size - int(nxt.sum())]]] = True
        return pop[nxt], front[nxt], cd[nxt]
