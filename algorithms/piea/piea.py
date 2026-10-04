# emopylab 2026
"""PIEA (performance indicator-based evolutionary algorithm).

Reference:
Y. Li, W. Li, S. Li, and Y. Zhao. A performance indicator-based evolutionary algorithm for expensive
high-dimensional multi-/many- objective optimization. Information Sciences, 2024: 121045.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, de, nd_sort, objs, tournament, uniform_point
from algorithms.community_utils.robust import cosine_dist
from core.population import Population
from util.svm import SVR

ALGORITHM_FLAGS = {'PIEA': {'expensive', 'many', 'multi', 'real'}}

_CP = np.array([0.27, 0.36, 0.43, 0.5, 0.57, 0.66, 0.75, 0.86, 1, 1.15, 1.35, 1.6, 2, 2.4, 3.1, 4.2, 6.5])


def _minmax(F):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))


def _shape_estimate(F, N):
    front, _ = nd_sort(F, None, N)
    F = F[front <= 1]
    if len(F) < 20:
        return 1.0
    n = len(F)
    F = _minmax(F)
    k = 1.5
    vp = np.zeros(len(_CP))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, p in enumerate(_CP):
            gp = np.sum(F ** p, axis=1) ** (1.0 / p)
            t = np.sort(gp)
            q1, q3 = t[max(int(n * 0.25), 1) - 1], t[max(int(n * 0.75), 1) - 1]
            gp = gp[~(gp > q3 + k * (q3 - q1))]      # denoise with a box plot
            vp[i] = np.std(gp / np.max(gp), ddof=1) if len(gp) > 1 else 0.0
    return float(_CP[0] if np.all(np.isnan(vp)) else _CP[int(np.nanargmin(vp))])


def _minkowski_to_ideal(F, p):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(np.abs(F - F.min(axis=0)) ** p, axis=1) ** (1.0 / p)


def _fitness_sde(F, Lp):
    N = len(F)
    F = _minmax(F)
    with np.errstate(invalid="ignore", divide="ignore"):
        shifted = np.maximum(F[None, :, :], F[:, None, :])          # [i, j] = max(F_j, F_i)
        dis = np.linalg.norm(F[:, None, :] - shifted, axis=2)
        np.fill_diagonal(dis, np.inf)
        fit = dis.min(axis=1)
        fit = 3 / (fit.max() + np.finfo(float).eps - fit.min()) * (fit - fit.min())
        d = _minkowski_to_ideal(F, Lp)
        d = -3 / (np.nanmax(d) + np.finfo(float).eps - np.nanmin(d)) * (d - np.nanmin(d))
    small = fit < 1e-4
    fit[small] = d[small]
    return np.tanh(fit)                                              # tansig


def _fitness_epsilon(F, kappa):
    F = _minmax(F)
    I = np.max(F[:, None, :] - F[None, :, :], axis=2)                 # I[i, j] = max_k(F_ik - F_jk)
    C = np.max(np.abs(I), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(-np.exp(-I / C[None, :] / kappa), axis=0) + 1


def _fitness_md(F, Lp):
    return -_minkowski_to_ideal(_minmax(F), Lp)


def _nd_sort_sdr(F, n_sort):
    """Non-dominated sorting where dominance is relaxed by the angle between solutions (SDR)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        N = len(Fn)
        normp = Fn.sum(axis=1)
        cosine = 1.0 - cosine_dist(Fn, Fn)
        np.fill_diagonal(cosine, 0.0)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        temp = np.unique(np.min(angle, axis=1))
        temp = temp[~np.isnan(temp)]
        min_a = temp[min(int(np.ceil(N / 2)), len(temp)) - 1] if len(temp) else np.nan
        theta = np.fmax(1.0, angle / min_a)
        dominate = normp[:, None] * theta < normp[None, :]
    front = np.full(N, np.inf)
    max_f = 0
    while np.sum(front != np.inf) < min(n_sort, N):
        max_f += 1
        current = ~dominate.any(axis=0) & (front == np.inf)
        front[current] = max_f
        dominate[current, :] = False
    return front, max_f


class PIEA(LoopAlgorithm):
    """Surrogate-assisted search on a support vector regression of one of three performance indicators
    (shift-based density, I_epsilon+, Minkowski distance to the ideal point); the indicator is drawn with a
    probability that adapts to how often its infill points enter the non-dominated set."""

    def __init__(self, pop_size: int = 100, eta: int = 5, r_max: int = 20, tau: int = 20, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.eta, self.r_max, self.tau = int(eta), int(r_max), int(tau)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        NI, D = self.pop_size, self.D
        P, _ = UniformPoint(NI, D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.choose = np.ones((3, self.tau))
        self.win = np.ones((3, self.tau))
        self.Pw = np.full(3, 1 / 3)
        self._set_optimum()

    def _update_information(self, flag, score):
        self.choose = np.hstack([self.choose[:, 1:], np.eye(3)[:, [flag]]])
        add = np.zeros((3, 1))
        if score != 0:
            add[flag, 0] = score / 2
        self.win = np.hstack([self.win[:, 1:], add])
        eps = np.finfo(float).eps
        p = (eps + self.win.sum(axis=1)) / (eps + self.choose.sum(axis=1))
        self.Pw = p / p.sum()

    def step(self):
        rng, N, NI = self.rng, self.N, self.pop_size
        A = self.pop
        F = objs(A)
        Lp = _shape_estimate(F, N)
        r = rng.random()
        if r < self.Pw[0]:
            fit, flag = _fitness_sde(F, Lp), 0
        elif r < self.Pw[0] + self.Pw[1]:
            fit, flag = _fitness_epsilon(F, 0.05), 1
        else:
            fit, flag = _fitness_md(F, Lp), 2
        model = SVR().fit(decs(A), fit)
        Dec = decs(A)
        Arc = Dec[rng.permutation(len(Dec))[:NI]]
        for k in range(self.r_max):
            pool = tournament(2, N, -fit, rng=rng)
            off = de(self.problem, Dec[pool], Arc, Dec[rng.permutation(len(Dec))[:NI]], rng=rng)
            of = model.predict(off)
            if k == 0:
                Arc, arc_fit = off, of
            else:
                better = arc_fit < of
                Arc = np.where(better[:, None], off, Arc)
                arc_fit = np.where(better, of, arc_fit)
        Arc = Arc[np.argsort(-arc_fit, kind="stable")[: self.eta]]
        rng_x = self.upper - self.lower
        dist = np.min(np.linalg.norm(((Arc - self.lower) / rng_x)[:, None, :] - ((Dec - self.lower) / rng_x)[None, :, :], axis=2), axis=1)
        new = self.evaluate(Arc[[int(np.argmax(dist))]])
        A = Population.merge(A, new)
        FA = objs(A)
        front, _ = nd_sort(FA, None, 1)
        score = 0
        if front[-1] == 1:
            score = 1
            sub, _ = _nd_sort_sdr(FA[front == 1], 1)
            if sub[-1] == 1:
                score = 2
        self._update_information(flag, score)
        self.pop = A
