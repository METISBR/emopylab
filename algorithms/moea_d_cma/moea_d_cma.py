# emopylab 2026
"""MOEA-D-CMA (mOEA/D with covariance matrix adaptation evolution strategy).

Reference:
H. Li, Q. Zhang, and J. Deng. Biased multiobjective optimization and decomposition algorithm. IEEE
Transactions on Cybernetics, 2017, 47(1): 52-66.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, de, decs, kmeans, neighbors_of, objs, uniform_point

ALGORITHM_FLAGS = {'MOEADCMA': {'integer', 'many', 'multi', 'real'}}


def _fresh(n):
    return {"s": None, "x": None, "sigma": 0.5, "C": np.eye(n), "pc": np.zeros(n), "ps": np.zeros(n)}


def _update_cma(X, S, gen):
    """One CMA-ES generation on the ranked candidates ``X`` (best first); returns the new state (or a reset one)."""
    n = X.shape[1]
    mu = 4 + int(np.floor(3 * np.log(n)))
    mu1 = mu // 2
    w = np.log((mu + 1) / 2) - np.log(np.arange(1, mu1 + 1))
    w = w / w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cs = (mueff + 2) / (n + mueff + 5)
    ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    ENI = np.sqrt(n) * (1 - 1 / 4 / n + 1 / 21 / n ** 2)
    y = (X[:mu1] - S["x"]) / S["sigma"]
    yw = w @ y
    S["x"] = S["x"] + S["sigma"] * yw
    evals, evecs = np.linalg.eigh((S["C"] + S["C"].T) / 2)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-300))) @ evecs.T
    S["ps"] = (1 - cs) * S["ps"] + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt @ yw
    hs = float(np.linalg.norm(S["ps"]) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * ENI)
    deltahs = 1 - hs
    S["pc"] = (1 - cc) * S["pc"] + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
    S["sigma"] = S["sigma"] * np.exp(cs / ds * (np.linalg.norm(S["ps"]) / ENI - 1))
    S["C"] = (1 - c1 - cmu) * S["C"] + c1 * (np.outer(S["pc"], S["pc"]) + deltahs * S["C"]) + cmu * y.T @ np.diag(w) @ y
    S["C"] = np.triu(S["C"]) + np.triu(S["C"], 1).T
    D_, B_ = np.linalg.eigh(S["C"])
    diagC = np.diag(S["C"])
    cond = D_.max() > 1e14 * D_.min()
    k = gen % n
    no_coord = np.any(S["x"] == S["x"] + 0.2 * S["sigma"] * np.sqrt(diagC))
    no_axis = np.all(S["x"] == S["x"] + 0.1 * S["sigma"] * np.sqrt(max(D_[k], 0.0)) * B_[:, k])
    tolx = np.any(S["sigma"] * np.sqrt(diagC) > 1e4)
    if cond or no_coord or no_axis or tolx or not np.all(np.isfinite(S["C"])):
        return _fresh(n)
    return S


class MOEADCMA(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, K: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.K = int(K)

    def initial_size(self):
        W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        W = 1.0 / W / np.sum(1.0 / W, axis=1, keepdims=True)
        self.W = W
        return self.pop_size

    def start(self):
        rng = self.rng
        lab = kmeans(self.W, self.K, rng)
        self.G = [np.where(lab == k)[0] for k in range(self.K) if np.any(lab == k)]
        self.K = len(self.G)
        self.B = neighbors_of(self.W, self.T)
        self.Z = objs(self.pop).min(axis=0)
        self.sig = []
        for g in self.G:
            st = _fresh(self.D)
            st["s"] = int(g[rng.integers(0, len(g))])
            st["x"] = decs(self.pop[st["s"]:st["s"] + 1])[0]
            self.sig.append(st)

    def step(self):
        rng, N, W = self.rng, self.N, self.W
        lam = 4 + int(np.floor(3 * np.log(self.D)))
        gen = int(np.ceil(self.FE / N))
        for s in range(N):
            k = next((i for i, st in enumerate(self.sig) if st["s"] == s), None)
            if k is not None:
                P = self.B[s][rng.permutation(self.B.shape[1])]
                st = self.sig[k]
                Xs = rng.multivariate_normal(st["x"], st["sigma"] ** 2 * st["C"], size=lam, check_valid="ignore")
                off = self.evaluate(Xs)
                cand_x = np.vstack([decs(off), decs(self.pop[s:s + 1])])
                cand_f = np.vstack([objs(off), objs(self.pop[s:s + 1])])
                order = np.argsort(np.max(np.abs(cand_f - self.Z) * W[s], axis=1), kind="stable")
                self.sig[k] = _update_cma(cand_x[order], st, gen)
                if self.sig[k]["s"] is None:
                    g = self.G[k]
                    sk = int(g[rng.integers(0, len(g))])
                    self.sig[k]["s"], self.sig[k]["x"] = sk, decs(self.pop[sk:sk + 1])[0]
            else:
                P = self.B[s][rng.permutation(self.B.shape[1])] if rng.random() < 0.9 else rng.permutation(N)
                off = self.evaluate(de(self.problem, decs(self.pop[s:s + 1]), decs(self.pop[P[:1]]), decs(self.pop[P[1:2]]), rng=rng))
            for x in range(len(off)):
                fx = objs(off[x:x + 1])[0]
                self.Z = np.minimum(self.Z, fx)
                g_old = np.max(np.abs(objs(self.pop[P]) - self.Z) * W[P], axis=1)
                g_new = np.max(np.abs(fx - self.Z) * W[P], axis=1)
                self.pop[P[np.where(g_old >= g_new)[0][:2]]] = off[x]
