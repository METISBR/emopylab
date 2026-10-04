# emopylab 2026
"""MOEA-D-2WA (mOEA/D with two-type weight vector adjustments).

Reference:
R. Jiao, S. Zeng, C. Li, and Y. S. Ong. Two-type weight adjustments in MOEA/D for highly constrained
many-objective optimization. Information Sciences, 2021, 578: 592-614.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, cosine_distance, decs, ga_half, objs, pdist2, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'MOEAD2WA': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _con(pop):
    C = cons(pop)
    return C if C.size else np.zeros((len(pop), 1))


def reduce_boundary(eF, k, max_k, cp=4):
    z, near = 1e-8, 1e-15
    with np.errstate(all="ignore"):
        B = max_k / np.power(np.log((eF + z) / z), 1.0 / cp)
        B = np.where(B == 0, B + near, B)
        f = eF * np.exp(-((k / B) ** cp))
    f = np.where(np.abs(f - z) < near, z, f)
    eps = f - z
    eps[eps <= 0] = 0
    return eps


def _lexi_min_row(A):
    """Index of the first row of ``sortrows(A)``."""
    cand = np.arange(len(A))
    for c in range(A.shape[1]):
        col = A[cand, c]
        cand = cand[col == col.min()]
        if len(cand) == 1:
            break
    return int(cand[0])


def weight_generation(N, AdN, M, theta_m, rng):
    W1, _ = uniform_point(N, M)
    W1 = np.hstack([W1, np.zeros((len(W1), 1))])
    W2 = rng.random((5000, M + 1))
    W2 = W2 / W2.sum(axis=1, keepdims=True)
    before, after = 1 - W2[:, M], 1 - W2[:, M] * theta_m
    W2[:, :M] = (after / before)[:, None] * W2[:, :M]
    W2[:, M] = W2[:, M] * theta_m
    W1 = np.vstack([W1, np.r_[np.zeros(M), 1.0]])
    while len(W1) < N + AdN:
        temp = -np.sort(-cosine_distance(W2, W1), axis=1)
        idx = _lexi_min_row(temp)
        W1 = np.vstack([W1, W2[idx]])
        W2 = np.delete(W2, idx, axis=0)
    W1 = np.maximum(W1, 1e-6)
    return W1, len(W1)


def _last_selection(F, K, Zmin, cv, rng):
    N = len(F)
    with np.errstate(all="ignore"):
        F = (F - Zmin) / (F.max(axis=0) - Zmin)
    cos = np.nan_to_num(1 - cosine_distance(F, F))
    cos = cos * (1 - np.eye(N))
    dele = np.zeros(N, bool)
    while dele.sum() < K:
        cols, rows = np.where(cos.T == cos.max())          # column-major order of the maxima
        j = int(rng.integers(len(rows)))
        a, b = rows[j], cols[j]
        v = a if cv[a] > cv[b] else b
        dele[v] = True
        cos[:, v] = 0
        cos[v, :] = 0
    return dele


class MOEAD2WA(LoopAlgorithm):
    """MOEA/D with two weight-vector sets: a decomposition over the objectives plus an extra constraint-violation
    axis whose adaptive epsilon removes weights as the feasible region is approached."""

    def initial_size(self):
        self.orN = self.pop_size
        self.W, self.pop_size = weight_generation(self.pop_size, self.pop_size, self.M, 1.0, self.rng)
        self.T = 20
        self.B = np.argsort(pdist2(self.W, self.W), axis=1, kind="stable")[:, : self.T]
        self.n_cur = self.pop_size
        return self.pop_size

    def start(self):
        pop = self.pop
        C = _con(pop)
        self.initialE = np.maximum(np.max(np.maximum(0, C), axis=0), 1)
        self.nCon = C.shape[1]
        cv = np.sum(np.maximum(0, C) / self.initialE, axis=1) / self.nCon
        self.Z = np.concatenate([objs(pop).min(axis=0), [cv.min()]])

    def _pop_update(self, pop, N, epsn, Zmin):
        C = _con(pop)
        vio = np.maximum(0, C)
        ok = np.sum(vio <= epsn, axis=1) == self.nCon
        if ok.sum() >= N:
            pop = pop[ok]
            pcv = np.sum(np.maximum(0, _con(pop)) / self.initialE, axis=1) / self.nCon
            dele = _last_selection(objs(pop), len(pop) - N, Zmin, pcv, self.rng)
            return pop[~dele]
        cv = np.sum(np.maximum(0, C) / self.initialE, axis=1) / self.nCon
        return pop[np.argsort(cv, kind="stable")[:N]]

    def step(self):
        rng, M, nCon, iE = self.rng, self.M, self.nCon, self.initialE
        pop = self.pop
        epsn = reduce_boundary(iE, self.FE, self.max_FE - self.orN)
        a = (self.W[:, M] - 1e-6) > epsn[0] / iE[0]
        self.W = self.W[~a]
        if self.n_cur > len(self.W):
            self.n_cur = len(self.W)
            self.B = np.argsort(pdist2(self.W, self.W), axis=1, kind="stable")[:, : self.T]
            pop = self._pop_update(pop, self.n_cur, epsn, self.Z[:M])
        W, B, Z = self.W, self.B, self.Z
        for i in range(self.n_cur):
            P = B[i][rng.permutation(B.shape[1])] if rng.random() < 0.9 else rng.permutation(self.n_cur)
            off = self.evaluate(ga_half(self.problem, decs(pop[P[:2]]), rng=rng))
            fo = objs(off)[0]
            Co, Cp = _con(off), _con(pop[P])
            Z[:M] = np.minimum(Z[:M], fo)
            cvO = float(np.sum(np.maximum(0, Co) / iE, axis=1)[0] / nCon)
            Z[M] = min(Z[M], cvO)
            cvo, cvp = np.maximum(0, Co)[0], np.maximum(0, Cp)
            cvP = np.sum(np.maximum(0, Cp) / iE, axis=1) / nCon
            PObj = np.column_stack([objs(pop[P]), cvP])
            OObj = np.concatenate([fo, [cvO]])
            normW = np.linalg.norm(W[P], axis=1)
            normP = np.linalg.norm(PObj - Z, axis=1)
            normO = np.linalg.norm(OObj - Z)
            with np.errstate(all="ignore"):
                cosP = np.sum((PObj - Z) * W[P], axis=1) / normW / normP
                cosO = np.sum((OObj - Z) * W[P], axis=1) / normW / normO
                g_old = normP * cosP + 5 * normP * np.sqrt(1 - cosP ** 2)
                g_new = normO * cosO + 5 * normO * np.sqrt(1 - cosO ** 2)
            of = np.sum(cvo <= epsn) == nCon
            pf = np.sum(cvp <= epsn, axis=1) == nCon
            cond = (of & pf & (g_old >= g_new)) | (of & ~pf) | (~of & ~pf & (np.sum(np.maximum(0, Cp), axis=1) > np.sum(cvo)))
            for j in np.where(cond)[0][:2]:
                pop[P[j]] = off[0]
        self.pop = pop
