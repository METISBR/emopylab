# emopylab 2026
"""ADSAPSO (adaptive dropout based surrogate-assisted particle swarm optimization).

Reference:
J. Lin, C. He, and R. Cheng. Adaptive dropout for high-dimensional expensive multiobjective
optimization. Complex & Intelligent Systems, 2022, 8(1): 271C285.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs
from algorithms.community_utils.sparse_mask import lhs_design
from algorithms.community_utils.surrogates import RBFPoly
from core.population import Population

ALGORITHM_FLAGS = {'ADSAPSO': {'expensive', 'integer', 'large', 'multi', 'real'}}


def _desc(v):
    """Stable descending order with NaN first (MATLAB ``sort(...,'descend')``)."""
    key = np.where(np.isnan(v), np.inf, v)
    return np.argsort(-key, kind="stable")


def env_select(F, N):
    """Indices kept by NSGA-II truncation, plus their front numbers and crowding distances."""
    front, maxf = nd_sort(F, None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[_desc(cd[last])[: N - int(nxt.sum())]]] = True
    idx = np.where(nxt)[0]
    return idx, front[idx], cd[idx]


def _crowd_same_front(F):
    N, M = F.shape
    cd = np.zeros(N)
    fmax, fmin = F.max(axis=0), F.min(axis=0)
    with np.errstate(all="ignore"):
        for i in range(M):
            r = np.argsort(F[:, i], kind="stable")
            cd[r[0]] = cd[r[-1]] = np.inf
            for j in range(1, N - 1):
                cd[r[j]] += (F[r[j + 1], i] - F[r[j - 1], i]) / (fmax[i] - fmin[i])
    return cd


def _bfe(sde, Cv, d1, d2, rng):
    SDE = sde.min(axis=1)
    with np.errstate(all="ignore"):
        Cd = (SDE - SDE.min()) / (SDE.max() - SDE.min())
    n = len(Cv)
    alpha, beta = np.zeros(n), np.zeros(n)
    mCd, mCv, m1, m2 = Cd.mean(), Cv.mean(), d1.mean(), d2.mean()
    hi, lo = Cv > mCv, Cv <= mCv
    c = {
        "111": hi & (d1 <= m1) & (Cd <= mCd), "112": hi & (d1 <= m1) & (Cd > mCd),
        "121": hi & (d1 > m1) & (Cd <= mCd), "122": hi & (d1 > m1) & (Cd > mCd),
        "211": lo & (d1 <= m1) & (d2 > m2) & (Cd <= mCd), "212": lo & (d1 <= m1) & (d2 > m2) & (Cd > mCd),
        "221": lo & ((d1 > m1) | (d2 <= m2)) & (Cd <= mCd), "222": lo & ((d1 > m1) | (d2 <= m2)) & (Cd > mCd),
    }
    alpha[c["111"]] = rng.random(c["111"].sum()) * 0.3 + 0.8; beta[c["111"]] = 1
    alpha[c["112"]] = 1; beta[c["112"]] = 1
    alpha[c["121"]] = 0.6; beta[c["121"]] = 1
    alpha[c["122"]] = 0.9; beta[c["122"]] = 1
    alpha[c["211"]] = rng.random(c["211"].sum()) * 0.3 + 0.8; beta[c["211"]] = rng.random(c["211"].sum()) * 0.3 + 0.8
    alpha[c["212"]] = 1; beta[c["212"]] = 1
    alpha[c["221"]] = 0.2; beta[c["221"]] = 0.2
    alpha[c["222"]] = 1; beta[c["222"]] = 0.2
    return alpha * Cd + beta * Cv


def bfe_rank(F, rng):
    """Order of an archive by balanced fitness estimation (shift-based density + convergence), best first."""
    with np.errstate(all="ignore"):
        P = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    N, M = P.shape
    S = np.maximum(P[None, :, :], P[:, None, :])          # S[i, j] = max(P_j, P_i)
    sde = np.sqrt(((P[:, None, :] - S) ** 2).sum(-1))
    np.fill_diagonal(sde, np.inf)
    dis = np.sqrt((P ** 2).sum(1))
    with np.errstate(all="ignore"):
        cos = P.sum(1) / (dis * np.sqrt(M))
    d1, d2 = dis * cos, dis * np.sqrt(1 - cos ** 2)
    return _desc(_bfe(sde, 1 - dis, d1, d2, rng))


class ADSAPSO(LoopAlgorithm):
    """Each iteration keeps the best ``N_a`` evaluated solutions, finds the ``beta*D`` variables whose means differ most
    between the ``N_s`` best and ``N_s`` worst of them, builds Gaussian RBF models on those variables only, runs 100
    surrogate PSO generations on them, and writes the ``k`` best predicted particles into copies of the ``k`` best
    evaluated solutions (the other variables are kept)."""

    def __init__(self, pop_size: int = 100, k: int = 5, beta: float = 0.5, init_num: int = 100, n_a: int = 200, n_s: int = 50,
                 sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.k, self.beta, self.init_num, self.n_a, self.n_s = int(k), float(beta), int(init_num), int(n_a), int(n_s)

    def _initialize_infill(self):
        return self.evaluate(self.lower + lhs_design(self.rng, self.init_num, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _reproduction(self, X, F, models, idx_dif):
        rng = self.rng
        N, M = F.shape
        pbX, pbF = X.copy(), F.copy()
        front, _ = nd_sort(F, None, 1)
        nd = np.where(front == 1)[0]
        order = nd[bfe_rank(F[nd], rng)]
        g = order[rng.integers(0, int(np.ceil(len(order) / 10)), N)]
        gX, gF = X[g], F[g]
        vel = np.zeros_like(X)
        pX = X
        for _ in range(100):
            Dd = pX.shape[1]
            r1, r2 = rng.random((N, 1)), rng.random((N, 1))
            vel = 0.5 * vel + r1 * (pbX - pX) + r2 * (gX - pX)
            off = pX + vel
            oF = np.column_stack([m.predict(off) for m in models])
            pX = off
            rep = ~np.all(oF >= pbF, axis=1)
            pbX[rep], pbF[rep] = off[rep], oF[rep]
            f1, _ = nd_sort(gF, None, 1)
            keep = np.where(f1 == 1)[0]
            cdg = _crowd_same_front(gF[keep])
            keep = keep[_desc(cdg)[: min(N, len(keep))]]
            gX, gF = gX[keep], gF[keep]
        if self.FE >= self.max_FE * 0.75:
            Lo, Up = self.lower[idx_dif], self.upper[idx_dif]
            Dd = off.shape[1]
            site = rng.random((N, Dd)) < 1.0 / Dd
            mu = rng.random((N, Dd))
            off = np.minimum(np.maximum(off, Lo), Up)
            L, U = np.broadcast_to(Lo, off.shape), np.broadcast_to(Up, off.shape)
            t = site & (mu <= 0.5)
            off[t] = off[t] + (U[t] - L[t]) * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - L[t]) / (U[t] - L[t])) ** 21) ** (1 / 21) - 1)
            t = site & (mu > 0.5)
            off[t] = off[t] + (U[t] - L[t]) * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (U[t] - off[t]) / (U[t] - L[t])) ** 21) ** (1 / 21))
            oF = np.column_stack([m.predict(off) for m in models])
        return off, oF

    def step(self):
        D, k = self.D, self.k
        arc = self.pop
        idx, front, cd = env_select(objs(arc), min(self.n_a, len(arc)))
        A = arc[idx]
        X, F = decs(A), objs(A)
        order = np.lexsort((-cd, front))
        cand = X[order[:k]].copy()
        well, poor = order[: self.n_s], order[-self.n_s:]
        dif = X[well].mean(axis=0) - X[poor].mean(axis=0)
        thr = np.sort(np.abs(dif))[::-1][int(np.ceil(self.beta * D)) - 1]
        idx_dif = np.where(np.abs(dif) >= thr)[0]
        models = [RBFPoly("gaussian").fit(X[:, idx_dif], F[:, i]) for i in range(self.M)]
        pidx, _, _ = env_select(F, self.N)
        off, oF = self._reproduction(X[pidx][:, idx_dif], F[pidx], models, idx_dif)
        sel, _, _ = env_select(oF, k)
        cand[:, idx_dif] = off[sel]
        self.pop = Population.merge(arc, self.evaluate(cand))
