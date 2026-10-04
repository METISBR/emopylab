# emopylab 2026
"""NMPSO (novel multi-objective particle swarm optimization).

Reference:
Q. Lin, S. Liu, Q. Zhu, C. Tang, R. Song, J. Chen, C. A. Coello Coello, K. Wong, and J. Zhang.
Particle swarm optimization with a balanceable fitness estimation for many-objective optimization
problems. IEEE Transactions on Evolutionary Computation, 2018, 22(1): 32-46.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, first_front, ga_half, objs, sde_distance, velocity
from core.population import Population

ALGORITHM_FLAGS = {'NMPSO': {'integer', 'many', 'multi', 'real'}}


def _cal_bfe(sde, Cv, d1, d2, rng):
    n = len(Cv)
    SDE = sde.min(axis=1)
    with np.errstate(all="ignore"):
        Cd = np.nan_to_num((SDE - SDE.min()) / (SDE.max() - SDE.min()))
    alpha, beta = np.zeros(n), np.zeros(n)
    mCd, mCv, m1, m2 = Cd.mean(), Cv.mean(), d1.mean(), d2.mean()
    c111 = (Cv > mCv) & (d1 <= m1) & (Cd <= mCd)
    c112 = (Cv > mCv) & (d1 <= m1) & (Cd > mCd)
    c121 = (Cv > mCv) & (d1 > m1) & (Cd <= mCd)
    c122 = (Cv > mCv) & (d1 > m1) & (Cd > mCd)
    c211 = (Cv <= mCv) & (d1 <= m1) & (d2 > m2) & (Cd <= mCd)
    c212 = (Cv <= mCv) & (d1 <= m1) & (d2 > m2) & (Cd > mCd)
    c221 = (Cv <= mCv) & ((d1 > m1) | (d2 <= m2)) & (Cd <= mCd)
    c222 = (Cv <= mCv) & ((d1 > m1) | (d2 <= m2)) & (Cd > mCd)
    alpha[c111], beta[c111] = rng.random(c111.sum()) * 0.3 + 0.8, 1
    alpha[c112], beta[c112] = 1, 1
    alpha[c121], beta[c121] = 0.6, 1
    alpha[c122], beta[c122] = 0.9, 1
    alpha[c211], beta[c211] = rng.random(c211.sum()) * 0.3 + 0.8, rng.random(c211.sum()) * 0.3 + 0.8
    alpha[c212], beta[c212] = 1, 1
    alpha[c221], beta[c221] = 0.2, 0.2
    alpha[c222], beta[c222] = 1, 0.2
    return alpha * Cd + beta * Cv


def _update_archive(A, S, K, rng):
    comb = A if S is None or len(S) == 0 else Population.merge(A, S)
    FA = objs(A)
    fmin, fmax = FA.min(axis=0), FA.max(axis=0)
    with np.errstate(all="ignore"):
        P = np.nan_to_num((objs(comb) - fmin) / (fmax - fmin))
    N, M = P.shape
    sde = sde_distance(P)
    dis = np.linalg.norm(P, axis=1)
    Cv = 1 - dis
    cosine = 1.0 - cosine_distance(P, np.ones((1, M)))[:, 0]
    d1 = dis * cosine
    d2 = dis * np.sqrt(np.maximum(0.0, 1 - cosine ** 2))
    choose = list(range(len(A)))
    nS = 0 if S is None else len(S)
    for i in range(nS):
        Si = P[len(A) + i]
        dominated = False
        keep = []
        for c in choose:
            lt, gt = np.any(Si < P[c]), np.any(Si > P[c])
            flag = int(lt) - int(gt)
            if flag == 1:
                continue                                   # S_i dominates c: drop c
            keep.append(c)
            if flag == -1:
                dominated = True
                break
        if dominated:
            continue
        choose = keep
        choose.append(len(A) + i)
        if len(choose) > K:
            ix = np.asarray(choose)
            worst = int(np.argmin(_cal_bfe(sde[np.ix_(ix, ix)], Cv[ix], d1[ix], d2[ix], rng)))
            choose.pop(worst)
    ix = np.asarray(choose)
    bfe = _cal_bfe(sde[np.ix_(ix, ix)], Cv[ix], d1[ix], d2[ix], rng)
    return comb[ix[np.argsort(-bfe, kind="stable")]]


class NMPSO(LoopAlgorithm):
    def start(self):
        self.swarm = self.pop
        self.pbest = self.pop
        self.archive = _update_archive(self.pop[first_front(objs(self.pop))], None, self.N, self.rng)
        self.pop = self.archive

    def _operator(self, particle, pbest, gbest):
        rng = self.rng
        X, P, G, V = decs(particle), decs(pbest), decs(gbest), velocity(particle)
        N = len(X)
        W = rng.uniform(0.1, 0.5, (N, 1))
        r1, r2, r3 = rng.random((N, 1)), rng.random((N, 1)), rng.random((N, 1))
        C1, C2, C3 = (rng.uniform(1.5, 2.5, (N, 1)) for _ in range(3))
        vel = W * V + C1 * r1 * (P - X) + C2 * r2 * (G - X) + C3 * r3 * (G - P)
        return self.evaluate(X + vel, V=vel)

    def step(self):
        rng, N = self.rng, self.N
        gbest = self.archive[rng.integers(0, int(np.ceil(len(self.archive) / 10)), size=N)]
        self.swarm = self._operator(self.swarm, self.pbest, gbest)
        replace = ~np.all(objs(self.swarm) >= objs(self.pbest), axis=1)
        pb = self.pbest.copy(deep=False)
        pb[replace] = self.swarm[replace]
        self.pbest = pb
        self.archive = _update_archive(self.archive, self.swarm, N, rng)
        n = len(self.archive)
        partner = rng.integers(0, int(np.ceil(n / 2)), size=n)
        S = self.evaluate(ga_half(self.problem, np.vstack([decs(self.archive), decs(self.archive[partner])]), rng=rng))
        self.archive = _update_archive(self.archive, S, N, rng)
        self.pop = self.archive
