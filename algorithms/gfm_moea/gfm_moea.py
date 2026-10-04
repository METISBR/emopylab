# emopylab 2026
"""GFM-MOEA (generic front modeling based multi-objective evolutionary algorithm).

Reference:
Y. Tian, X. Zhang, R. Cheng, C. He, and Y. Jin. Guiding evolutionary multi-objective optimization
with generic front modeling. IEEE Transactions on Cybernetics, 2020, 50(3): 1106-1119.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, ga, nd_sort, objs, pdist2, tournament
from core.population import Population

ALGORITHM_FLAGS = {'GFMMOEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _gfm(X):
    """Levenberg-Marquardt fit of the generalised simplex  sum_i a_i x_i^p_i = 1  to the front."""
    N, M = X.shape
    X = np.maximum(X, 1e-12)
    P, A = np.ones(M), np.ones(M)
    lam = 1.0
    E = np.sum(A * X ** P, axis=1) - 1
    mse = np.mean(E ** 2)
    logX = np.log(X)
    for _ in range(1000):
        J = np.hstack([A * X ** P * logX, X ** P])
        JTJ, JTE = J.T @ J, J.T @ E
        while True:
            delta = -np.linalg.solve(JTJ + lam * np.eye(2 * M), JTE)
            nP, nA = P + delta[:M], A + delta[M:]
            nE = np.sum(nA * X ** nP, axis=1) - 1
            nmse = np.mean(nE ** 2)
            if nmse < mse and np.all(nP > 1e-3) and np.all(nA > 1e-3):
                P, A, E, mse, lam = nP, nA, nE, nmse, lam / 1.1
                break
            elif lam > 1e8:
                return P, A
            lam *= 1.1
    return P, A


def _cal_fitness(F, P, A):
    N, M = F.shape
    r = np.ones(N)
    lam = np.full(N, 0.1)
    with np.errstate(all="ignore"):
        E = np.sum(A * (F * r[:, None]) ** P, axis=1) - 1
        for _ in range(1000):
            newr = r - lam * E * np.sum(A * P * F ** P * r[:, None] ** (P - 1), axis=1)
            newE = np.sum(A * (F * newr[:, None]) ** P, axis=1) - 1
            update = (newr > 0) & (np.sum(newE ** 2) < np.sum(E ** 2))
            r = np.where(update, newr, r)
            E = np.where(update, newE, E)
            lam = np.where(update, lam * 1.1, lam / 1.1)
    F1 = F * r[:, None]
    app = np.linalg.norm(F1, axis=1) - np.linalg.norm(F, axis=1)
    dis = pdist2(F1, F1)
    np.fill_diagonal(dis, np.inf)
    return app, dis


def _last_selection(F, front_no, app, dis, theta, N):
    nds = np.where(front_no == 1)[0]
    M = F.shape[1]
    with np.errstate(all="ignore"):
        pbi = np.linalg.norm(F[nds], axis=1)[:, None] * np.sqrt(np.maximum(0.0, 1 - (1 - cosine_distance(F[nds], np.eye(M))) ** 2))
    non_extreme = np.ones(len(front_no), bool)
    non_extreme[nds[np.argmin(pbi, axis=0)]] = False
    last = front_no == front_no.max()
    choose = np.ones(len(F), bool)
    while choose.sum() > N:
        remain = np.where(choose & last & non_extreme)[0]
        if len(remain) == 0:
            break
        d = np.sort(dis[np.ix_(remain, np.where(choose)[0])], axis=1)
        d = d[:, 0] + 0.1 * d[:, 1]
        choose[remain[int(np.argmin(theta * app[remain] + (1 - theta) * d))]] = False
    return choose


def _environmental_selection(pop, P, A, zmin, theta, N):
    front_no, max_f = nd_sort(objs(pop), None, N)
    nxt = np.where(front_no <= max_f)[0]
    F = objs(pop)[nxt] - zmin
    app, dis = _cal_fitness(F, P, A)
    choose = _last_selection(F, front_no[nxt], app, dis, theta, N)
    sel = nxt[choose]
    d = np.sort(dis[np.ix_(choose, choose)], axis=1)
    return pop[sel], front_no[sel], app[choose], d[:, 0] + 0.1 * d[:, 1]


class GFMMOEA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, theta: float = 0.2, fPFE: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.theta, self.fPFE = float(theta), float(fPFE)

    def start(self):
        F = objs(self.pop)
        self.front_no, _ = nd_sort(F, None, np.inf)
        self.zmin = F.min(axis=0)
        self.P, self.A = np.ones(self.M), np.ones(self.M)
        self.app, dis = _cal_fitness(F - self.zmin, self.P, self.A)
        d = np.sort(dis, axis=1)
        self.crowd = d[:, 0] + 0.1 * d[:, 1]

    def step(self):
        N, theta = self.N, self.theta
        pool = tournament(2, N, self.front_no, -theta * self.app - (1 - theta) * self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), rng=self.rng))
        self.zmin = np.minimum(self.zmin, objs(off).min(axis=0))
        period = int(np.ceil(self.fPFE * np.ceil(self.max_FE / N)))
        if self.fPFE == 0 or (period > 0 and int(np.ceil(self.FE / N)) % period == 0):
            self.P, self.A = _gfm(objs(self.pop)[self.front_no == 1] - self.zmin)
        self.pop, self.front_no, self.app, self.crowd = _environmental_selection(
            Population.merge(self.pop, off), self.P, self.A, self.zmin, theta, N)
