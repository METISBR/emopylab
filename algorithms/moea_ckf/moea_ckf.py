# emopylab 2026
"""MOEA-CKF (multi-objective evolutionary algorithm based on cross-scale knowledge fusion).

Reference:
Z. Ding, L. Chen, D. Sun, and X. Zhang. Efficient sparse large-scale multi-objective optimization
based on cross-scale knowledge fusion. IEEE Transactions on Systems, Man, and Cybernetics: Systems,
2024, 54(11): 6989-7001.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, kmeans, nd_sort, objs, tournament, decs
from algorithms.community_utils.spea import truncation
from core.population import Population

ALGORITHM_FLAGS = {'MOEACKF': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


def _binary_crossover(P1, P2, rng):
    """Exchange differing bits between the parents keeping the share of ones balanced (equal expected 1->0 and 0->1 flips)."""
    off = P1 > 0
    P2 = P2 > 0
    for i in range(len(off)):
        diff = np.where(off[i] != P2[i])[0]
        if len(diff) == 0:
            continue
        m1 = off[i, diff].mean()
        r = min(0.5, 2 * m1, 2 * (1 - m1))
        with np.errstate(all="ignore"):
            rate = np.where(off[i, diff], r / 2 / m1, r / 2 / (1 - m1))
        ex = rng.random(len(diff)) < rate
        off[i, diff[ex]] = ~off[i, diff[ex]]
    return off


def _binary_mutation(off, rng):
    off = off.copy()
    n, D = off.shape
    if D == 0:
        return off
    m1 = off.mean(axis=1)
    r = np.minimum(np.minimum(1.0 / D, 2 * m1), 2 * (1 - m1))
    with np.errstate(all="ignore"):
        rate = np.where(off, (r / 2 / m1)[:, None], (r / 2 / (1 - m1))[:, None])
    flip = rng.random((n, D)) < rate
    off[flip] = ~off[flip]
    return off


def _sbx_half(rng, P1, P2, dis_c=20.0):
    n, D = P1.shape
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (dis_c + 1)), (2 - 2 * mu) ** (-1 / (dis_c + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
    beta[rng.random((n, D)) < 0.5] = 1
    return (P1 + P2) / 2 + beta * (P1 - P2) / 2


def _pm(rng, X, lower, upper, dis_m=20.0):
    X = np.atleast_2d(X)
    n, D = X.shape
    lo, up = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    site, mu = rng.random((n, D)) < 1.0 / max(D, 1), rng.random((n, D))
    X = np.minimum(np.maximum(X, lo), up)
    span = up - lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lo[t]) / span[t]) ** (dis_m + 1)) ** (1 / (dis_m + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - X[t]) / span[t]) ** (dis_m + 1)) ** (1 / (dis_m + 1)))
    return X


def _env_selection(pop, Dec, Mask, N, length, num):
    success = np.zeros(len(pop), bool)
    F = objs(pop)
    uni = np.unique(F, axis=0, return_index=True)[1]
    if len(uni) == 1:
        uni = np.unique(decs(pop), axis=0, return_index=True)[1]
    pop, Dec, Mask = pop[uni], Dec[uni], Mask[uni]
    N = min(N, len(pop))
    F = objs(pop)
    front, maxf = nd_sort(F, None, N)
    nxt = front < maxf
    f1 = front == 1
    with np.errstate(invalid="ignore", divide="ignore"):
        Fn = (F - F[f1].min(axis=0)) / (F[f1].max(axis=0) - F[f1].min(axis=0))
    last = np.where(front == maxf)[0]
    dele = truncation(Fn[last], len(last) - N + int(nxt.sum()))
    nxt[last[~dele]] = True
    success[uni[nxt]] = True
    s1, s2 = success[length: length + num].sum(), success[length + num:].sum()
    ratio = float(np.clip((s1 + 1e-6) / (s1 + s2 + 1e-6), 0.1, 0.9))
    return pop[nxt], Dec[nxt], Mask[nxt], ratio


class MOEACKF(LoopAlgorithm):
    """Two subpopulations share the search: one (chosen with probability ``rho``) exploits sparsity knowledge learned
    from the elite masks (local) and from all masks (global) — variable groups from a clustering of the variable scores
    drive balanced mask crossover/mutation and a PCA-reduced real crossover — while the other reproduces conventionally;
    ``rho`` follows the survival ratio of the knowledge-driven offspring."""

    def _initialize_infill(self):
        rng, N, D = self.rng, self.N, self.D
        lo, up = self.lower, self.upper
        real = True                               # the reference's REAL flag compares an encoding array with a string: always true
        TDec, TMask, TPop = [], [], []
        fit = np.zeros(D)
        for _ in range(1 + 4 * int(real)):
            dec = lo + rng.random((D, D)) * (up - lo)
            mask = np.eye(D)
            p = self.evaluate(dec * mask)
            TDec.append(dec), TMask.append(mask), TPop.append(p)
            C = cons(p)
            fit += nd_sort(np.hstack([objs(p), C]) if C.size else objs(p), None, np.inf)[0]
        dec = lo + rng.random((N, D)) * (up - lo)
        mask = np.zeros((N, D))
        for i in range(N):
            mask[i, tournament(2, int(np.ceil(rng.random() * D)), fit, rng=rng)] = 1
        p = self.evaluate(dec * mask)
        TDec.append(dec), TMask.append(mask), TPop.append(p)
        self.Fitness = fit
        pop, self.Dec, self.Mask, _ = _env_selection(Population.merge(*TPop), np.vstack(TDec), np.vstack(TMask), N, 0, 0)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.rho, self.group = 0.5, None
        self._set_optimum()

    def _sparsity_analysis(self, front):
        D = self.D
        M = self.Mask > 0
        elite = M[front == 1]
        L1, L2 = elite.all(axis=0), (~elite).all(axis=0)
        local = np.vstack([L1, L2, ~L1 & ~L2])
        G1, G2 = M.all(axis=0), (~M).all(axis=0)
        glob = np.vstack([G1, G2, ~G1 & ~G2])
        theta = len(elite) / len(M)
        f = self.Fitness
        f = (f - f.min() + 1e-6) / (f.max() - f.min() + 1e-6) + 1e-6
        if D < 2:
            nsv, sv = self.group[0], self.group[1]
        else:
            cluster = kmeans(f[:, None], 2, self.rng) + 1
            f = f + (1 - elite.mean(axis=0)) * (0.5 * theta + 0.5 * (self.FE / self.max_FE)) * f
            m1 = f[cluster == 1].mean() if np.any(cluster == 1) else np.nan
            m2 = f[cluster == 2].mean() if np.any(cluster == 2) else np.nan
            if m1 > m2:
                nsv, sv = cluster == 2, cluster == 1
            else:
                nsv, sv = cluster == 1, cluster == 2
        self.Fitness = f
        self.group = (nsv, sv)
        return local, glob, nsv, sv, theta

    def _reproduction1(self, pop1, dec1, mask1, front, crowd, site, local, glob, nsv, sv, theta):
        rng, D = self.rng, self.D
        len1 = len(pop1)
        mate = tournament(2, len1 * 2, front[site], -crowd[site], rng=rng)
        P1m, P2m = mask1[mate[: len(mate) // 2]] > 0, mask1[mate[len(mate) // 2:]] > 0
        N = len(P1m)
        off = np.zeros((N, D), bool)
        for i in range(N):
            know = local if rng.random() < theta else glob
            allone, other = know[0], know[2]
            t = np.zeros(D, bool)
            for grp in (nsv, sv):
                idx = np.where(other & grp)[0]
                if len(idx):
                    c = _binary_crossover(P1m[[i]][:, idx], P2m[[i]][:, idx], rng)
                    t[idx] = _binary_mutation(c, rng)[0]
            t[allone] = True
            off[i] = t
        lo, up = self.lower, self.upper
        allone, allzero, other = glob[0], glob[1], glob[2]
        best = dec1[:, ~allzero]
        mean = best.mean(axis=0)
        norm = best.max(axis=0) - best.min(axis=0) + 1e-6
        T = (best - mean) / norm
        U, _, _ = np.linalg.svd((1.0 / N) * (T.T @ T))
        K = int(np.sum(other & nsv) + np.sum(allone))
        Ur = U[:, :K]
        red = T @ Ur
        h = len(mate) // 2
        R = red[mate]
        P1, P2 = R[:h], R[h: 2 * h]
        mu = rng.random(P1.shape)
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (-1.0) ** rng.integers(0, 2, P1.shape)
        beta[rng.random(P1.shape) < 0.5] = 1
        off_red = (P1 + P2) / 2 + beta * (P1 - P2) / 2
        t_dec = (off_red @ Ur.T) * norm + mean
        t_dec = _pm(rng, t_dec, lo[~allzero], up[~allzero])
        if allzero.any():
            dec = np.zeros((N, D))
            base = dec1[mate][:, allzero]
            half = _sbx_half(rng, base[:h], base[h: 2 * h])
            dec[:, allzero] = _pm(rng, half, lo[allzero], up[allzero])
            dec[:, ~allzero] = t_dec
        else:
            dec = t_dec
        return dec, off.astype(float), len1

    def _reproduction2(self, pdec, pmask, nsv, sv):
        rng, D = self.rng, self.D
        h = len(pmask) // 2
        P1m, P2m = pmask[:h] > 0, pmask[h:] > 0
        off = np.zeros((h, D), bool)
        for grp in (sv, nsv):
            idx = np.where(grp)[0]
            if len(idx):
                off[:, idx] = _binary_mutation(_binary_crossover(P1m[:, idx], P2m[:, idx], rng), rng)
        P1d, P2d = pdec[:h], pdec[h: 2 * h]
        dec = np.zeros((h, D))
        lo, up = self.lower, self.upper
        for i in range(h):
            nz = off[i]
            for sel in (nz, ~nz):
                if sel.any():
                    v = _sbx_half(rng, P1d[[i]][:, sel], P2d[[i]][:, sel])
                    dec[i, sel] = _pm(rng, v, lo[sel], up[sel])[0]
        return dec, off.astype(float)

    def step(self):
        rng, N = self.rng, self.N
        pop, Dec, Mask = self.pop, self.Dec, self.Mask
        C = cons(pop)
        front, _ = nd_sort(objs(pop), C if C.size else None, np.inf)
        crowd = crowding(objs(pop), front)
        site = self.rho > rng.random(len(pop))
        if site.sum() >= 2:
            pop1, dec1, mask1, pop2, dec2, mask2 = pop[site], Dec[site], Mask[site], pop[~site], Dec[~site], Mask[~site]
        elif (~site).sum() < 1:
            pop1, dec1, mask1, pop2 = pop, Dec, Mask, None
        else:
            pop1, pop2, dec2, mask2 = None, pop, Dec, Mask
        local, glob, nsv, sv, theta = self._sparsity_analysis(front)
        len1 = 0
        off1 = off2 = None
        if pop1 is not None:
            d1, m1, len1 = self._reproduction1(pop1, dec1, mask1, front, crowd, site if pop2 is not None else np.ones(len(pop), bool), local, glob, nsv, sv, theta)
            off1 = self.evaluate(d1 * m1)
        else:
            d1, m1 = np.zeros((0, self.D)), np.zeros((0, self.D))
        if pop2 is not None:
            sel = ~site if pop1 is not None else np.ones(len(pop), bool)
            mate = tournament(2, (N - len1) * 2, front[sel], -crowd[sel], rng=rng)
            d2, m2 = self._reproduction2(dec2[mate], mask2[mate], sv, nsv)
            off2 = self.evaluate(d2 * m2)
        else:
            d2, m2 = np.zeros((0, self.D)), np.zeros((0, self.D))
        parts = [p for p in (pop, off1, off2) if p is not None]
        self.pop, self.Dec, self.Mask, ratio = _env_selection(Population.merge(*parts), np.vstack([Dec, d1, d2]), np.vstack([Mask, m1, m2]), N, len(pop), len1)
        self.rho = (self.rho + ratio) / 2
