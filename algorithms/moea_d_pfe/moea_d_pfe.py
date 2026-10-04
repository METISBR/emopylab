# emopylab 2026
"""MOEA-D-PFE (mOEA/D with Pareto front estimation).

Reference:
T. Takagi, K. Takadama, and H. Sato. A multi-objective evolutionary algorithm using weight vector
arrangement based on Pareto front estimation. Transaction of the Japanese Society for Evolutionary
Computation (Japanese), 2021, 12(2): 45-60.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Delaunay

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, nd_sort, objs, uniform_point
from algorithms.community_utils.surrogates import RBFExact
from core.population import Population

ALGORITHM_FLAGS = {'MOEADPFE': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _proj(M):
    i = np.arange(1, M)
    A = np.tril(np.tile(np.sqrt(1 / i / (i + 1)), (M - 1, 1)), -1) + np.diag(np.sqrt((i + 1) / i))
    return np.vstack([np.zeros(M - 1), A])                      # M x (M-1): simplex -> (M-1)-D coordinates


def cluster_validator(W, X, alpha):
    """Keep the directions whose projection falls inside a single-linkage cluster (cutoff ``alpha``) of the front."""
    A = _proj(W.shape[1])
    xa = np.sort((X @ A)[:, 0])
    if len(xa) < 2:
        return W
    T = fcluster(linkage(xa[:, None], "single"), t=alpha, criterion="distance")
    first = {t: np.where(T == t)[0][0] for t in np.unique(T)}
    last = {t: np.where(T == t)[0][-1] for t in np.unique(T)}
    f, l = set(first.values()), set(last.values())
    edges = xa[np.array(sorted(f ^ l), dtype=int)]
    if len(edges) < 2:
        return W[:0]
    wa = (W @ A)[:, 0]
    b = np.searchsorted(edges, wa, side="right")               # bin k = [edges[k-1], edges[k])
    b[wa == edges[-1]] = len(edges) - 1
    inside = (wa >= edges[0]) & (wa <= edges[-1])
    return W[inside & (b % 2 == 1)]


def alpha_shape_validator(W, X, alpha):
    """Keep the directions inside the alpha shape (Delaunay simplices of circumradius <= alpha) of the front, applied
    only when ``alpha`` exceeds the smallest critical radius (as the reference tests with the alpha spectrum)."""
    A = _proj(W.shape[1])
    P = X @ A
    try:
        tri = Delaunay(P)
    except Exception:  # noqa: BLE001 - degenerate point set
        return W
    radii = []
    for s in tri.simplices:
        V = P[s]
        d = V.shape[1]
        Mx = 2 * (V[1:] - V[0])
        rhs = (V[1:] ** 2).sum(1) - (V[0] ** 2).sum()
        try:
            c = np.linalg.solve(Mx, rhs)
            radii.append(np.sqrt(((c - V[0]) ** 2).sum()))
        except np.linalg.LinAlgError:
            radii.append(np.inf)
    radii = np.array(radii)
    if not (alpha > radii.min()):
        return W
    keep = radii <= alpha
    loc = tri.find_simplex(W @ A)
    return W[(loc >= 0) & keep[np.maximum(loc, 0)]]


def estimate_pf(F, alpha, M):
    lo, hi = F.min(0), F.max(0)
    with np.errstate(all="ignore"):
        O = np.unique(np.nan_to_num((F - lo) / (hi - lo)), axis=0)
    Y = np.abs(O).sum(1)
    ok = Y > 0
    O, Y = O[ok], Y[ok]
    X = O / Y[:, None]
    W, _ = uniform_point(20000, M, "ILD")
    if M > 4 or np.isinf(alpha):
        pass
    elif M == 2:
        W = cluster_validator(W, X, alpha)
    else:
        W = alpha_shape_validator(W, X, alpha)
    if len(W) == 0:
        return np.zeros((0, M))
    hat = W * np.asarray(RBFExact(1.0).fit(X, Y).predict(W)).reshape(-1, 1)
    f, _ = nd_sort(hat, None, 1)
    return hat[f == 1]


def update_weight(F, Fhat, N):
    fmin, fmax = F.min(0), F.max(0)
    with np.errstate(all="ignore"):
        O = np.unique(np.vstack([np.nan_to_num((F - fmin) / (fmax - fmin)), Fhat]), axis=0)
    M = O.shape[1]
    lp = lambda a: (np.abs(O - a) ** 0.5).sum(1) ** 2          # Minkowski p = 0.5 distance to one point
    with np.errstate(all="ignore"):
        cd = 1 - (O @ np.eye(M)) / (np.linalg.norm(O, axis=1)[:, None])
    ch = np.zeros(len(O), bool)
    ch[np.argmin(np.where(np.isnan(cd), np.inf, cd), 0)] = True
    dmin = np.full(len(O), np.inf)
    for k in np.where(ch)[0]:
        dmin = np.minimum(dmin, lp(O[k]))
    while ch.sum() < min(N, len(O)):                            # greedy max-min, distances kept incrementally
        rem = np.where(~ch)[0]
        k = rem[int(np.argmax(dmin[rem]))]
        ch[k] = True
        dmin = np.minimum(dmin, lp(O[k]))
    W = O[ch] * (fmax - fmin)
    B = np.argsort(np.sqrt(((W[:, None] - W[None]) ** 2).sum(-1)), 1, kind="stable")[:, : int(np.ceil(len(W) / 10))]
    with np.errstate(all="ignore"):
        W = W / np.abs(W).sum(1, keepdims=True)
    return W, B


class MOEADPFE(LoopAlgorithm):
    """MOEA/D (Tchebycheff, full neighbourhood replacement) whose weight vectors are periodically re-derived from an
    estimated Pareto front: the non-dominated archive is mapped to (direction, L1 norm) pairs, an exact RBF network
    predicts the norm along a dense set of directions restricted to the front's support (clustering for two objectives,
    alpha shape for three or four), and new weights are chosen by greedy max-min L0.5 distance."""

    def __init__(self, pop_size: int = 100, phi: float = 0.1, alpha: float = 0.1, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.phi, self.alpha = float(phi), float(alpha)

    def _initialize_infill(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M, "ILD")
        return self.evaluate(self.random_decs(self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        N = self.N
        self.T = int(np.ceil(N / 10))
        self.B = np.argsort(np.sqrt(((self.W[:, None] - self.W[None]) ** 2).sum(-1)), 1, kind="stable")[:, : self.T]
        self.pop = infills
        self.Z = objs(infills).min(0)
        self.update_fe = int(np.ceil(self.phi * self.max_FE / N)) * N
        self.BP = int(np.floor((self.max_FE - 1) / self.update_fe)) * self.update_fe
        self.EP = infills
        self._set_optimum()

    def step(self):
        rng, N = self.rng, self.N
        offs = []
        for i in range(N):
            P = self.B[i][rng.permutation(self.B.shape[1])]
            o = self.evaluate(ga_half(self.problem, decs(self.pop)[P[:2]], rng=rng))
            offs.append(o)
            f = objs(o)[0]
            self.Z = np.minimum(self.Z, f)
            F = objs(self.pop)
            with np.errstate(all="ignore"):
                g_old = np.max(np.abs(F[P] - self.Z) / self.W[P], 1)
                g_new = np.max(np.abs(f - self.Z) / self.W[P], 1)
            rep = P[g_old >= g_new]
            if len(rep):
                merged = Population.merge(self.pop, o)
                sel = np.arange(len(self.pop))
                sel[rep] = len(self.pop)
                self.pop = merged[sel]
        if self.FE <= self.BP:
            self.EP = Population.merge(self.EP, *offs)
            f, _ = nd_sort(objs(self.EP), None, 1)
            self.EP = self.EP[f == 1]
            if len(self.EP) > 5000:
                self.EP = self.EP[len(self.EP) - 5000:]
            if self.FE % self.update_fe == 0:
                hat = estimate_pf(objs(self.EP), self.alpha, self.M)
                self.W, self.B = update_weight(objs(self.EP), hat, N)
                Fe = np.abs(objs(self.EP) - self.Z)
                with np.errstate(all="ignore"):
                    pick = [int(np.nanargmin(np.max(Fe / w, 1))) for w in self.W]
                self.pop = self.EP[np.array(pick)]
