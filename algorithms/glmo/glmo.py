# emopylab 2026
"""GLMO (grouped and linked mutation operator algorithm).

Reference:
H. Zille. Large-scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art. PhD Thesis, Otto von Guericke University Magdeburg, 2019. ----------------------------------------------------------------------- Copyright (C) 2020 Heiner Zille This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. (CC BY-NC-SA 4.0). To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or see the pdf-file "License-CC-BY-NC-SA-4.0.pdf" that came with this code. You are free to: * Share ? copy and redistribute the material in any medium or format * Adapt ? remix, transform, and build upon the material Under the following terms: * Attribution ? You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. * NonCommercial ? You may not use the material for commercial purposes. * ShareAlike ? If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original. * No additional restrictions ? You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits. Author of this Code: Heiner Zille <heiner.zille@ovgu.de> or <heiner.zille@gmail.com> This code is based on the following publications: 1) Heiner Zille "Large-scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art" PhD Thesis, Otto von Guericke University Magdeburg, 2019 http://dx.doi.org/10.25673/32063 2) Heiner Zille, Hisao Ishibuchi, Sanaz Mostaghim and Yusuke Nojima "Mutation Operators Based on Variable Grouping for Multi-objective Large-scale Optimization" IEEE Symposium Series on Computational Intelligence (SSCI), IEEE, Athens, Greece, December 2016 https://ieeexplore.ieee.org/document/7850214 This file is intended to work with the PlatEMO framework version 2.5. Date of publication of this code: 06.04.2020 Last Update of this code: 06.04.2020 A newer version of this algorithm may be available. Please contact the author or see http://www.ci.ovgu.de/Research/Codes.html. The files may have been modified in Feb 2021 by the authors of the Platemo framework to work with the Platemo 3.0 release. ----------------------------------------------------------------------- methods function main(Algorithm,Problem) [optimiser,typeOfGroups,numberOfGroups] = Algorithm.ParameterSet(3,2,4); if optimiser == 1
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, adds, cons, crowding, decs, first_front, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils import nsga3_ref
from core.population import Population

ALGORITHM_FLAGS = {'GLMO': {'integer', 'large', 'multi', 'real'}}


def create_groups(n_groups, X, method, rng):
    """Group index (1..n_groups) of every variable of every solution: linear blocks (1), blocks over the variables
    ordered by value (2) or randomly shuffled blocks (3)."""
    n, D = X.shape
    per = D // n_groups
    if method == 2:
        rank = np.argsort(np.argsort(X, axis=1, kind="stable"), axis=1, kind="stable")
        return np.full((n, D), n_groups) if per == 0 else np.minimum(rank // per, n_groups - 1) + 1
    base = np.concatenate([np.repeat(np.arange(1, n_groups), per), np.full(D - per * (n_groups - 1), n_groups)]) if per else np.full(D, n_groups)
    out = np.tile(base, (n, 1))
    if method == 3:
        for i in range(n):
            out[i] = out[i][rng.permutation(D)]
    return out


def _group_mutation(X, lower, upper, site, mu, disM=20.0):
    """Polynomial mutation of the variables flagged in ``site`` with one random number ``mu`` per solution."""
    X = np.minimum(np.maximum(X, lower), upper)
    span = upper - lower
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] = X[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lower[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] = X[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (upper[t] - X[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return X


def _grouped_ga(algo, parents):
    """SBX with random-sign spread followed by group-wise polynomial mutation (a single group of variables per child)."""
    rng, disC = algo.rng, 20.0
    P = np.asarray(parents, dtype=float)
    h = len(P) // 2
    P1, P2 = P[:h], P[h: 2 * h]
    N, D = P1.shape
    mu = rng.random((N, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (disC + 1)), (2 - 2 * mu) ** (-1 / (disC + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, (N, D))
    beta[rng.random((N, D)) < 0.5] = 1
    off = np.vstack([(P1 + P2) / 2 + beta * (P1 - P2) / 2, (P1 + P2) / 2 - beta * (P1 - P2) / 2])
    lower, upper = np.tile(algo.lower, (2 * N, 1)), np.tile(algo.upper, (2 * N, 1))
    groups = create_groups(algo.n_groups, off, algo.group_type, rng)
    site = groups == rng.integers(1, algo.n_groups + 1, (len(groups), 1))
    mu = np.repeat(rng.random((2 * N, 1)), D, axis=1)
    return _group_mutation(off, lower, upper, site, mu)


def _nsga2_selection(pop, N):
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


class GLMO(LoopAlgorithm):
    """Large-scale search that mutates only one group of variables per child (groups formed linearly, by value or
    at random) on top of an SMPSO (1), NSGA-II (2) or NSGA-III (3) engine."""

    def __init__(self, pop_size: int = 100, optimiser: int = 3, type_of_groups: int = 2, n_groups: int = 4, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.optimiser, self.group_type, self.n_groups = int(optimiser), int(type_of_groups), int(n_groups)

    def initial_size(self):
        if self.optimiser == 3:
            self.Z, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        pop = self.pop
        if self.optimiser == 1:
            self.swarm, self.pbest = pop, pop
            self.gbest, self.crowd = self._update_gbest(pop)
            self.pop = self.gbest
        elif self.optimiser == 2:
            _, self.front, self.crowd = _nsga2_selection(pop, self.N)
        else:
            C = cons(pop)
            feas = np.all(C <= 0, axis=1) if C.size else np.ones(len(pop), bool)
            self.Zmin = objs(pop)[feas].min(axis=0) if feas.any() else None

    def _update_gbest(self, pop):
        pop = pop[first_front(objs(pop))]
        cd = crowding(objs(pop))
        rank = np.argsort(-cd, kind="stable")[: min(self.N, len(pop))]
        return pop[rank], cd[rank]

    def _smpso(self):
        rng, N, D = self.rng, self.N, self.D
        gbest = self.gbest[tournament(2, N, -self.crowd, rng=rng)]
        X, Pb, Gb = decs(self.swarm), decs(self.pbest), decs(gbest)
        V = adds(self.swarm, "V", np.zeros((N, D)))
        W = np.repeat(rng.uniform(0.1, 0.5, (N, 1)), D, axis=1)
        r1, r2 = np.repeat(rng.random((N, 1)), D, axis=1), np.repeat(rng.random((N, 1)), D, axis=1)
        C1, C2 = np.repeat(rng.uniform(1.5, 2.5, (N, 1)), D, axis=1), np.repeat(rng.uniform(1.5, 2.5, (N, 1)), D, axis=1)
        off_v = W * V + C1 * r1 * (Pb - X) + C2 * r2 * (Gb - X)
        phi = np.maximum(4, C1 + C2)
        off_v = off_v * 2 / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
        delta = np.tile((self.upper - self.lower) / 2, (N, 1))
        off_v = np.maximum(np.minimum(off_v, delta), -delta)
        off_x = X + off_v
        lower, upper = np.tile(self.lower, (N, 1)), np.tile(self.upper, (N, 1))
        repair = (off_x < lower) | (off_x > upper)
        off_v[repair] *= 0.001
        off_x = np.maximum(np.minimum(off_x, upper), lower)
        site1 = np.repeat(rng.random((N, 1)) < 0.15, D, axis=1)
        groups = create_groups(self.n_groups, off_x, self.group_type, rng)
        site2 = groups == rng.integers(1, self.n_groups + 1, (N, 1))
        mu = np.repeat(rng.random((N, 1)), D, axis=1)
        off_x = _group_mutation(off_x, lower, upper, site1 & site2, mu)
        new = self.evaluate(off_x, V=off_v)
        self.swarm = new
        self.gbest, self.crowd = self._update_gbest(Population.merge(self.gbest, new))
        replace = ~np.all(objs(new) >= objs(self.pbest), axis=1)
        pb = self.pbest.copy() if hasattr(self.pbest, "copy") else self.pbest
        pb[replace] = new[replace]
        self.pbest = pb
        self.pop = self.gbest

    def step(self):
        rng, N = self.rng, self.N
        if self.optimiser == 1:
            return self._smpso()
        pop = self.pop
        if self.optimiser == 2:
            mate = tournament(2, N, self.front, -self.crowd, rng=rng)
            off = self.evaluate(_grouped_ga(self, decs(pop[mate])))
            self.pop, self.front, self.crowd = _nsga2_selection(Population.merge(pop, off), N)
            return
        C = cons(pop)
        cv = np.sum(np.maximum(0, C), axis=1) if C.size else np.zeros(len(pop))
        mate = tournament(2, N, cv, rng=rng)
        off = self.evaluate(_grouped_ga(self, decs(pop[mate])))
        Co = cons(off)
        feas = np.all(Co <= 0, axis=1) if Co.size else np.ones(len(off), bool)
        if feas.any():
            fo = objs(off)[feas].min(axis=0)
            self.Zmin = fo if self.Zmin is None else np.minimum(self.Zmin, fo)
        self.pop = nsga3_ref.select(Population.merge(pop, off), N, self.Z, self.Zmin, rng)
