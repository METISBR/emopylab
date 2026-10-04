# emopylab 2026
"""MOEA-D-MRDL (mOEA/D with maximum relative diversity loss).

Reference:
S. B. Gee, K. C. Tan, V. A. Shim, and N. R. Pal. Online diversity assessment in evolutionary
multiobjective optimization: A geometrical perspective. IEEE Transactions on Evolutionary
Computation, 2015, 19(4): 542-559.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cosine_distance, decs, ga_half, neighbors_of, objs, pdist2, uniform_point

ALGORITHM_FLAGS = {'MOEADMRDL': {'integer', 'multi', 'real'}}


def _operator_adaption(etaC, Pn, all_e, e, nmov):
    if len(all_e) == 0:
        if not np.isnan(e):
            all_e = [e]
        return etaC, Pn, all_e
    all_e = all_e + [e if not np.isnan(e) else all_e[-1]]
    a = np.asarray(all_e)
    ma = np.array([a[max(0, i - nmov + 1): i + 1].mean() for i in range(len(a))])
    with np.errstate(all="ignore"):
        Y = np.log(ma[:-1])
    if len(Y) == 0 or not np.all(np.isfinite(Y)):
        return etaC, Pn, all_e
    Phi = np.column_stack([np.ones(len(Y)), np.arange(1, len(Y) + 1)])
    lam, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    pred = np.exp(lam[0] + lam[1] * len(a))
    if a[-1] > pred:
        return max(etaC - 2, 2), 0.5 * (a[-1] - pred), all_e
    return etaC + 2, 0.0, all_e


class MOEADMRDL(LoopAlgorithm):
    def __init__(self, pop_size: int = 100, gamma: float = 20, nmov: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.gamma, self.nmov = float(gamma), int(nmov)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.T = int(np.ceil(self.pop_size / 10))
        self.B = neighbors_of(self.W, self.T)
        return self.pop_size

    def start(self):
        self.Z = objs(self.pop).min(axis=0)
        self.disC, self.Pn, self.all_e = 20.0, 0.0, []

    def _selection(self, off):
        rng, W, B, gamma = self.rng, self.W, self.B, self.gamma
        Fo = objs(off)
        conv = None
        egam = []
        for i in rng.permutation(len(self.pop)):
            nb = B[i]
            g_old = np.max(np.abs(objs(self.pop[nb]) - self.Z) * W[nb], axis=1)
            g_new = np.max(np.abs(Fo[i] - self.Z) * W[nb], axis=1)
            PM = nb[g_old > g_new]
            if len(PM) == 0:
                continue
            Fpm = objs(self.pop[PM])
            nearest = int(np.argmin(pdist2(Fo[i][None, :], Fpm)[0]))
            if conv is None:
                conv = (Fpm[nearest] - Fo[i])[None, :]
                self.pop[PM[nearest]] = off[i]
            else:
                sine1 = np.sqrt(np.maximum(0, 1 - (1 - cosine_distance(Fpm, conv)) ** 2))
                sine2 = np.sqrt(np.maximum(0, 1 - (1 - cosine_distance(Fo[i][None, :], conv)) ** 2))
                with np.errstate(all="ignore"):
                    rdl = np.linalg.norm(Fpm, axis=1)[:, None] * sine1 / (np.linalg.norm(Fo[i]) * sine2)
                mrdl = np.nanmax(np.where(np.isnan(rdl), -np.inf, rdl), axis=1)
                if np.all(mrdl <= gamma):
                    conv = np.vstack([conv, Fpm[nearest] - Fo[i]])
                    self.pop[PM[nearest]] = off[i]
                    egam.append(mrdl[nearest])
        return float(np.mean(egam)) if egam else float("nan")

    def step(self):
        rng, N, T = self.rng, self.N, self.T
        p1 = self.B[np.arange(N), rng.integers(0, T, size=N)]
        p2 = self.B[np.arange(N), rng.integers(0, T, size=N)]
        dec = ga_half(self.problem, np.vstack([decs(self.pop[p1]), decs(self.pop[p2])]), (1, self.disC, 1, 20), rng=rng)
        dec = dec + rng.standard_normal(dec.shape) * self.Pn
        off = self.evaluate(dec)
        self.Z = np.minimum(self.Z, objs(off).min(axis=0))
        egamma = self._selection(off)
        self.disC, self.Pn, self.all_e = _operator_adaption(self.disC, self.Pn, self.all_e, egamma, self.nmov)
