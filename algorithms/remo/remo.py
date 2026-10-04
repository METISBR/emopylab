# emopylab 2026
"""REMO (expensive multiobjective optimization by relation learning and prediction).

Reference:
H. Hao, A. Zhou, H. Qian, and H. Zhang. Expensive multiobjective optimization by relation learning
and prediction. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 1157-1170.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, objs
from algorithms.community_utils.nn import PatternNet, minmax_apply, minmax_fit
from algorithms.community_utils.sparse_mask import lhs_design
from algorithms.csea.csea import ref_select
from core.population import Population

ALGORITHM_FLAGS = {'REMO': {'expensive', 'integer', 'many', 'multi', 'real'}}


def _split(P, R, delt):
    N = len(P)
    out = np.ones(N, bool)
    with np.errstate(all="ignore"):
        cs = (P @ R.T) / (np.linalg.norm(P, axis=1)[:, None] * np.linalg.norm(R, axis=1)[None])
    ref = np.argmax(np.where(np.isnan(cs), -np.inf, cs), axis=1)
    Z = P.min(axis=0)
    for i in range(len(R)):
        sub = np.where(ref == i)[0]
        if not len(sub):
            continue
        w = R[i] - Z
        W = w / np.sqrt((w ** 2).sum())
        nP = np.sqrt(((P[sub] - Z) ** 2).sum(1))
        nR = np.sqrt((w ** 2).sum())
        with np.errstate(all="ignore"):
            c = ((P[sub] - Z) @ W) / np.linalg.norm(W) / nP - 1e-6
            g = (nP * c + delt * nP * np.sqrt(1 - c ** 2)) / nR
        out[sub[g > 1]] = False
    return out, out.mean()


def pbi_output(P, R):
    """Label solutions inside the PBI contour of their nearest reference solution; the penalty is bisected so 30-70% are inside."""
    lo, hi, r, l = -20.0, 20.0, 0.0, None
    while r > 0.7 or r < 0.3:
        c = (lo + hi) / 2
        if abs(lo - hi) < 1e-1:
            break
        l, r = _split(P, R, c)
        if r > 0.7:
            lo = c
        elif r < 0.3:
            hi = c
    return l


def _pairs(A, B):
    """Rows [a_i, b_j] for every pair, first operand varying fastest."""
    return np.hstack([np.tile(A, (len(B), 1)), np.repeat(B, len(A), axis=0)])


def relation_pairs(X, cat, rng):
    c1, c2 = X[cat], X[~cat]
    n1, n2 = len(c1), len(c2)
    C11 = _pairs(c1, c1)[~np.eye(n1, dtype=bool).T.ravel()] if n1 else np.zeros((0, 2 * X.shape[1]))
    C22 = _pairs(c2, c2)[~np.eye(n2, dtype=bool).T.ravel()] if n2 else np.zeros((0, 2 * X.shape[1]))
    C12, C21 = _pairs(c1, c2), _pairs(c2, c1)
    t = int(np.ceil(len(C12) / 2))
    if len(C11) > t and len(C22) > t:
        C11, C22 = C11[rng.permutation(len(C11))[:t]], C22[rng.permutation(len(C22))[:t]]
    elif len(C11) < t:
        C22 = C22[rng.permutation(len(C22))[: min(len(C22), 2 * t - len(C11))]]
    elif len(C22) < t:
        C11 = C11[rng.permutation(len(C11))[: min(len(C11), 2 * t - len(C22))]]
    XX = np.vstack([C11, C22, C12, C21])
    L = np.concatenate([np.zeros(len(C11) + len(C22)), np.ones(len(C12)), -np.ones(len(C21))])
    return XX, L


def onehot(L):
    O = np.zeros((len(L), 3))
    O[L == 1, 0], O[L == 0, 1], O[L == -1, 2] = 1, 1, 1
    return O


class REMO(LoopAlgorithm):
    """A neural network learns the relation (better / similar / worse) between pairs of solutions, labelled by whether each
    lies inside the PBI contour of ``k`` reference solutions; GA offspring are scored against the two categories of evaluated
    solutions and the most promising after ``gmax`` surrogate predictions are evaluated."""

    def __init__(self, pop_size: int = 100, k: int = 6, gmax: int = 3000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.k, self.gmax = int(k), int(gmax)

    def _initialize_infill(self):
        n = 11 * self.D - 1 if self.D <= 10 else 100
        return self.evaluate(self.lower + lhs_design(self.rng, n, self.D) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.P = infills
        self.pop = infills
        self._set_optimum()

    def _scores(self, X, cat, st, net, Nxt):
        c1, c2 = X[cat], X[~cat]
        n1, n2 = len(c1), len(c2)
        blocks = []
        for x in Nxt:
            blocks += [np.hstack([c1, np.tile(x, (n1, 1))]), np.hstack([np.tile(x, (n1, 1)), c1]),
                       np.hstack([c2, np.tile(x, (n2, 1))]), np.hstack([np.tile(x, (n2, 1)), c2])]
        P = net.predict_proba(minmax_apply(np.vstack(blocks), st))
        per = 2 * (n1 + n2)
        sc = np.zeros(len(Nxt))
        with np.errstate(all="ignore"):
            for i in range(len(Nxt)):
                o = i * per
                a = P[o:o + n1].sum(0) / n1
                b = P[o + n1:o + 2 * n1].sum(0) / n1
                c = P[o + 2 * n1:o + 2 * n1 + n2].sum(0) / n2
                d = P[o + 2 * n1 + n2:o + per].sum(0) / n2
                s1 = a[1] + a[2] + b[1] + b[0] + c[2] + d[0]
                s2 = a[0] + b[2] + c[1] + c[0] + d[1] + d[2]
                sc[i] = s1 - s2
        return sc

    def step(self):
        rng = self.rng
        ref = ref_select(self.P, self.k)
        X = decs(self.P)
        cat = pbi_output(objs(self.P), objs(ref))
        if cat is None:
            cat = np.ones(len(X), bool)
        XX, L = relation_pairs(X, cat, rng)
        K = np.concatenate([rng.permutation(np.where(L == v)[0])[: int(np.ceil(0.75 * (L == v).sum()))] for v in (0, 1, -1)])
        te = rng.permutation(np.setdiff1d(np.arange(len(L)), K))
        K = K[rng.permutation(len(K))]
        st = minmax_fit(XX[K])
        net = PatternNet([int(np.ceil(self.D * 2 * 1.5)), self.D * 2, int(np.ceil(self.D * 2 / 2))], rng).fit(minmax_apply(XX[K], st), onehot(L[K]))
        rdec = decs(ref)
        nxt = ga(self.problem, np.vstack([X, rdec]), (1, 15, 1, 5), rng=rng)
        i = 0
        while i < self.gmax:
            sc = self._scores(X, cat, st, net, nxt)
            inp = nxt[np.argsort(-sc, kind="stable")[: len(ref)]]
            nxt = ga(self.problem, np.vstack([inp, rdec]), (1, 15, 1, 5), rng=rng)
            i += len(nxt)
        sc = self._scores(X, cat, st, net, nxt)
        nxt = nxt[np.argsort(-sc, kind="stable")[:4]] if (sc > 3.9).sum() < 4 else nxt[sc > 3.9]
        if len(nxt):
            self.pop = Population.merge(self.pop, self.evaluate(nxt))
        self.P = ref_select(self.pop, self.N)
