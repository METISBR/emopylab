# emopylab 2026
"""LMPFE (evolutionary algorithm with local model based Pareto front estimation).

Reference:
Y. Tian, L. Si, X. Zhang, K. C. Tan, and Y. Jin. Local model based Pareto front estimation for
multi-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2023,
53(1): 623-634.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'LMPFE': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _pdist(A, B):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0))


def _minmax(F):
    with np.errstate(all="ignore"):
        return (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))


def _adaptive_division(rng, F, K):
    N, M = F.shape
    if K == 1:
        return np.zeros((1, M)), np.array([np.inf])
    Fn = _minmax(F)
    dist = _pdist(Fn, Fn)
    np.fill_diagonal(dist, np.inf)
    radius = dist.min(axis=0).max()
    trans = np.zeros(N, dtype=int)
    rid = 1
    while np.any(trans == 0):
        seeds = np.array([int(np.where(trans == 0)[0][0])])
        trans[seeds] = rid
        remain = np.where(trans == 0)[0]
        while True:
            nb = np.sum(dist[np.ix_(seeds, remain)] <= radius, axis=0) if len(remain) else np.zeros(0, int)
            seeds = remain[nb >= 1]
            trans[seeds] = rid
            remain = np.where(trans == 0)[0]
            if nb.sum() == 0:
                break
        rid += 1
    true_num = len(np.unique(trans))
    center = np.zeros((true_num, M))
    R = np.ones(true_num)
    for i in range(true_num):
        cur = trans == i + 1
        center[i] = Fn[cur].mean(axis=0)
        R[i] = _pdist(Fn[cur], center[i][None, :]).max()
    count = np.bincount(trans, minlength=true_num + 1)[1:].astype(float)     # tabulate
    if true_num > K:
        while np.sum(np.isfinite(count)) > K:
            I = int(np.argmin(count))
            center[I] = np.inf
            count[I] = np.inf
            R[I] = -np.inf
            cur = np.where(trans == I + 1)[0]
            T = np.argmin(_pdist(Fn[cur], center), axis=1)
            trans[cur] = T + 1
            for k in np.where(np.isfinite(count))[0]:
                sel = trans == k + 1
                center[k] = Fn[sel].mean(axis=0)
                R[k] = _pdist(Fn[sel], center[k][None, :]).max() / np.sqrt(M - 1)
    elif true_num < K:
        counts = list(count)
        center = list(center)
        R = list(R)
        while sum(c != -np.inf for c in counts) < K:
            I = int(np.argmax(counts))
            center[I] = np.full(M, -np.inf)
            counts[I] = -np.inf
            R[I] = -np.inf
            cur = np.where(trans == I + 1)[0]
            t1 = int(np.argmax(_pdist(Fn[cur], Fn[[cur[int(rng.integers(0, len(cur)))]]])[:, 0]))
            t2 = int(np.argmax(_pdist(Fn[cur], Fn[[cur[t1]]])[:, 0]))
            T = np.argmin(_pdist(Fn[cur], Fn[cur[[t1, t2]]]), axis=1) + 1
            exist = len(counts)
            trans[cur] = T + exist
            c1, c2 = Fn[trans == exist + 1].mean(axis=0), Fn[trans == exist + 2].mean(axis=0)
            center += [c1, c2]
            r = 0.5 * float(np.linalg.norm(c1 - c2))
            R += [r, r]
            counts += [float(np.sum(T == 1)), float(np.sum(T == 2))]
        center, R, count = np.array(center), np.array(R), np.array(counts)
    keep = np.abs(count) != np.inf
    return np.atleast_2d(center)[keep], np.atleast_1d(R)[keep]


def _allocation(Fn, center, R):
    """Assign each solution to a subregion; with the reference's bookkeeping every solution ends up with its nearest centre."""
    if len(R) == 1:
        return np.zeros(len(Fn), dtype=int)
    return np.argmin(_pdist(Fn, center), axis=1)


def _inter_point(F, P):
    N = len(F)
    P = np.tile(P, (N, 1))
    r = np.ones(N)
    lam = np.full(N, 0.002)
    with np.errstate(all="ignore"):
        E = np.sum((r[:, None] * F) ** P, axis=1) - 1
        for _ in range(1000):
            newr = r - lam * E * np.sum(P * F ** P * r[:, None] ** (P - 1), axis=1)
            newE = np.sum((newr[:, None] * F) ** P, axis=1) - 1
            upd = (newr > 0) & (np.sum(newE ** 2) < np.sum(E ** 2))
            r = np.where(upd, newr, r)
            E = np.where(upd, newE, E)
            lam = np.where(upd, lam * 1.002, lam / 1.002)
    return F * r[:, None]


def _sub_fitness(F, P, center, R):
    N, M = F.shape
    Fn = _minmax(F)
    if len(R) == 1:
        ip = _inter_point(Fn, P[0])
    else:
        ip = np.ones((N, M))
        tr = _allocation(Fn, center, R)
        for i in range(len(R)):
            cur = np.where(tr == i)[0]
            if len(cur):
                ip[cur] = _inter_point(Fn[cur], P[i])
    app = np.min(ip - Fn, axis=1)
    dis = np.max(np.abs(ip[:, None, :] - ip[None, :, :]), axis=2)
    np.fill_diagonal(dis, np.inf)
    return app, dis


def _gfm(X):
    """Levenberg-Marquardt fit of the exponents p such that sum_i x_i^p_i = 1 on the points ``X``."""
    N, M = X.shape
    X = np.maximum(X, 1e-12)
    P = np.ones(M)
    lam = 1.0
    E = np.sum(X ** P, axis=1) - 1
    mse = np.mean(E ** 2)
    for _ in range(1000):
        J = X ** P * np.log(X)
        while True:
            delta = -np.linalg.solve(J.T @ J + lam * np.eye(M), J.T @ E)
            newP = P + delta
            newE = np.sum(X ** newP, axis=1) - 1
            newmse = np.mean(newE ** 2)
            if newmse < mse and np.all(newP > 1e-3):
                P, E, mse = newP, newE, newmse
                lam /= 1.08
                break
            elif lam > 1e8:
                return P
            lam *= 1.08
    return P


def _sub_gfm(F, center, R, front):
    K = len(center)
    N, M = F.shape
    Fn = _minmax(F)
    if K == 1:
        return _gfm(Fn[front == 1])[None, :]
    P = np.ones((K, M))
    tr = _allocation(Fn, center, R)
    sub_first = np.zeros(N, bool)
    for i in range(K):
        cur = np.where(tr == i)[0]
        if len(cur):
            fno, mfno = nd_sort(F[cur], None, len(cur))
            sub_first[cur[(fno < mfno) | (fno == 1)]] = True
    ftr = tr[sub_first]
    remain = Fn[sub_first]
    if sub_first.sum() > M:
        for i in range(K):
            cur = np.where(ftr == i)[0]
            if len(cur):
                if len(cur) < M + 1:
                    cur = np.argsort(_pdist(remain, center[[i]])[:, 0], kind="stable")[: M + 1]
                P[i] = _gfm(Fn[cur])          # (indexes the full normalised set, exactly as the reference does)
    return P


def _selection(pop, P, theta, N, center, R):
    F = objs(pop)
    front, maxf = nd_sort(F, None, N)
    nxt = np.where(front <= maxf)[0]
    app, dis = _sub_fitness(F[nxt], P, center, R)
    fr = front[nxt]
    Fs = F[nxt]
    nds = np.where(fr == 1)[0]
    M = F.shape[1]
    normf = np.linalg.norm(Fs[nds], axis=1)[:, None]
    with np.errstate(all="ignore"):
        cos = (Fs[nds] @ np.eye(M).T) / (normf * 1.0)
    perp = normf * np.sqrt(np.maximum(1 - cos ** 2, 0))
    extreme = np.argmin(np.where(np.isnan(perp), np.inf, perp), axis=0)
    non_extreme = ~np.isin(np.arange(len(fr)), nds[extreme])
    last = fr == fr.max()
    choose = np.ones(len(fr), bool)
    while choose.sum() > N:
        remain = np.where(choose & last & non_extreme)[0]
        d = np.sort(dis[np.ix_(remain, np.where(choose)[0])], axis=1)
        d = d[:, 0] + 0.1 * d[:, 1]
        fit = theta * d + (1 - theta) * app[remain]
        choose[remain[int(np.argmin(fit))]] = False
    idx = np.where(choose)[0]
    d = np.sort(dis[np.ix_(idx, idx)], axis=1)
    return pop[nxt[idx]], fr[idx], app[idx], d[:, 0] + 0.1 * d[:, 1]


class LMPFE(LoopAlgorithm):
    """Pareto-front shape estimation (a generalised Lp-norm surface per objective-space subregion, refitted every ``fPFE`` of the
    run) turns every solution's distance to the estimated front into an 'approximation' score; selection balances that score
    with a crowding measure through an adaptive weight ``theta``."""

    def __init__(self, pop_size: int = 100, f_pfe: float = 0.1, k: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.f_pfe, self.K = float(f_pfe), int(k)

    def start(self):
        pop = self.pop
        F = objs(pop)
        self.front, _ = nd_sort(F, None, np.inf)
        self.center, self.R = _adaptive_division(self.rng, F, self.K)
        self.P = np.ones((self.K, self.M))
        app, dis = _sub_fitness(F, self.P, self.center, self.R)
        d = np.sort(dis, axis=1)
        self.app, self.crowd = app, d[:, 0] + 0.1 * d[:, 1]
        self.theta = 0.8
        self.pre_app, self.pre_crowd = self.app.mean(), self.crowd.mean()

    def step(self):
        rng, N = self.rng, self.N
        pop = self.pop
        mate = tournament(2, N, self.front, -self.theta * self.crowd - (1 - self.theta) * self.app, rng=rng)
        off = self.evaluate(ga(self.problem, decs(pop[mate]), rng=rng))
        gen = int(np.ceil(self.FE / N))
        if self.f_pfe == 0 or gen % int(np.ceil(self.f_pfe * np.ceil(self.max_FE / N))) == 0:
            F = objs(pop)
            self.center, self.R = _adaptive_division(rng, F, self.K)
            self.P = _sub_gfm(F, self.center, self.R, self.front)
        self.pop, self.front, self.app, self.crowd = _selection(Population.merge(pop, off), self.P, self.theta, N, self.center, self.R)
        rate_app = abs(self.app.mean() - self.pre_app) / abs(self.pre_app)
        rate_crowd = abs(self.crowd.mean() - self.pre_crowd) / abs(self.pre_crowd)
        self.theta = 1 - np.exp(-np.exp(rate_app - rate_crowd))
        self.pre_app, self.pre_crowd = self.app.mean(), self.crowd.mean()
