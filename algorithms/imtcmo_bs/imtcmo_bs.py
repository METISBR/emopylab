# emopylab 2026
"""IMTCMO-BS (IMTCMO with convergence/diversity directed boundary sampling)."""

from __future__ import annotations

import numpy as np

from algorithms.apsea.apsea import _epsilon_selection
from algorithms.community_utils.base import decs, kmeans, objs, uniform_point
from algorithms.imtcmo.imtcmo import IMTCMO, _main_selection
from core.population import Population

ALGORITHM_FLAGS = {'IMTCMO_BS': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def representative(F, R, rng):
    """One solution per direction: a random associate (largest cosine), or the best-aligned unused solution."""
    n = len(F)
    with np.errstate(all="ignore"):
        Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
        cos = (Fn @ R.T) / (np.linalg.norm(Fn, axis=1)[:, None] * np.linalg.norm(R, axis=1)[None])
    cos = np.nan_to_num(cos, nan=-np.inf)
    assoc = np.argmax(cos, 1)
    best = np.zeros(len(R), dtype=int)
    used = np.zeros(n, bool)
    empty = []
    for i in range(len(R)):
        cur = np.where(assoc == i)[0]
        if len(cur):
            b = cur[int(np.ceil(rng.random() * len(cur))) - 1] if len(cur) > 1 else cur[0]
            best[i], used[b] = b, True
        else:
            empty.append(i)
    for i in empty:
        order = np.argsort(-cos[:, i], kind="stable")
        if n > len(R):
            k = 0
            while used[order[k]]:
                k += 1
            best[i], used[order[k]] = order[k], True
        else:
            best[i] = order[0]
    return best


class IMTCMO_BS(IMTCMO):
    """IMTCMO whose two tasks additionally receive, every ``g`` generations (and at the first), solutions sampled along
    lines in decision space: from the bounds towards representative solutions of the boundary and clustered reference
    directions (convergence), or between pairs of such representatives (diversity), chosen with probability 1/2."""

    def __init__(self, pop_size: int = 100, Nw: int = 10, Ns: int = 30, g: int = 50, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.Nw, self.Ns, self.g = int(Nw), int(Ns), int(g)
        self.cnt = 0

    def _directions(self):
        M = self.M
        RefV, _ = uniform_point(self.N, M)
        lab = kmeans(RefV, self.Nw, self.rng)
        centres = np.array([RefV[lab == i].mean(0) for i in np.unique(lab)])
        B = np.eye(M)
        B[B == 0] = 10e-7
        return np.vstack([B, centres])

    def _clip_eval(self, X):
        return self.evaluate(np.clip(X, self.lower, self.upper))

    def _convergence(self, pop):
        rng, D = self.rng, self.D
        R = self._directions()
        bx = decs(pop)[representative(objs(pop), R, rng)]
        lo, up = self.lower, self.upper
        a, b = bx - lo, bx - up
        dirs = np.vstack([a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)])
        nw = len(bx)
        span = np.sqrt(((up - lo) ** 2).sum())
        out = []
        for _ in range(self.Ns):
            r = rng.random(2 * nw) * span
            X = np.vstack([lo + r[:nw, None] * dirs[:nw], up + r[nw:, None] * dirs[nw:]])
            out.append(self._clip_eval(np.nan_to_num(X)))
        return Population.merge(*out)

    def _diversity(self, pop):
        rng = self.rng
        R = self._directions()
        best = representative(objs(pop), R, rng)
        X = decs(pop)
        if len(best) == 1:
            s, e = np.repeat(X[best], 2, 0), np.vstack([self.lower, self.upper])
        else:
            if len(best) % 2 == 1:
                best = best[:-1][rng.permutation(len(best) - 1)]
            bx = X[best]
            h = len(bx) // 2
            s, e = bx[:h], bx[h:]
        d = s - e
        with np.errstate(all="ignore"):
            dirs = d / np.linalg.norm(d, axis=1, keepdims=True)
        span = np.sqrt(((self.upper - self.lower) ** 2).sum())
        out = []
        for _ in range(self.Ns):
            r = rng.random(len(s)) * span
            out.append(self._clip_eval(np.nan_to_num(e + r[:, None] * dirs)))
        return Population.merge(*out)

    def step(self):
        N = self.N
        cp = (-np.log(self.var0) - 6) / np.log(1 - 0.5)
        var = self.var0 * (1 - self.x) ** cp
        self.cnt += 1
        if self.cnt == 1 or self.cnt % self.g == 0:
            sample = self._diversity(self.pop1) if self.rng.random() > 0.5 else self._convergence(self.pop1)
            self.pop1, self.fit1 = _main_selection(Population.merge(self.pop1, sample), N)
            self.pop2, self.fit2 = _epsilon_selection(Population.merge(self.pop2, sample), N, var)
        super().step()
