# emopylab 2026
"""SLMEA (super-large-scale multi-objective evolutionary algorithm).

Reference:
Y. Tian, Y. Feng, X. Zhang, and C. Sun. A fast clustering based evolutionary algorithm for super-
large-scale sparse multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2022, 9(4):
1-16.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import nsga2_mask_selection
from core.population import Population

ALGORITHM_FLAGS = {'SLMEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


def _calculate_fitness(archive_mask, D, rng):
    """Score of every variable from the Hamming/And distances between its archive column and a reference column (the one
    whose usage ratio is closest to one half); also returns the mixed / never-used / always-used variable sets."""
    n = len(archive_mask)
    A = archive_mask > 0
    ratio = A.sum(axis=0) / n
    R = np.abs(ratio - 0.5)
    el = np.where(R == R.min())[0]
    vec = A[:, el[int(rng.integers(0, len(el)))]][:, None]
    hd = np.sum(vec ^ A, axis=0)
    ad = np.sum(vec & A, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fit = hd / (hd + ad)
    return fit, np.where((ratio > 0) & (ratio < 1))[0], np.where(ratio == 0)[0], np.where(ratio == 1)[0]


def _cpu_group(n_groups, xprime, n_vars):
    """Group index (1-based) of ``n_vars`` variables ordered by ``xprime`` into equal blocks (the remainder is its own)."""
    per = n_vars // n_groups
    if per == 1:
        return np.arange(1, n_vars + 1), n_vars
    base = np.concatenate([np.repeat(np.arange(1, n_groups + 1), per), np.full(n_vars - per * n_groups, n_groups + 1)])
    order = np.argsort(np.where(np.isnan(xprime), np.inf, xprime), kind="stable")   # NaN sorts last
    index = np.zeros(n_vars, dtype=int)
    index[order] = base
    return index, (n_groups if n_vars % n_groups == 0 else n_groups + 1)


def _encode_mask(mask, index, n_groups, rng):
    cols = []
    for u in range(1, n_groups + 1):
        c = np.where(index == u)[0]
        with np.errstate(invalid="ignore"):
            m = mask[:, c].mean(axis=1) if len(c) else np.full(len(mask), np.nan)
        cols.append(m > rng.random(len(mask)))
    return np.column_stack(cols).astype(float)


def _encode_dec(dec, index, n_groups):
    cols = []
    for u in range(1, n_groups + 1):
        c = np.where(index == u)[0]
        cols.append(dec[:, c].mean(axis=1) if len(c) else np.full(len(dec), np.nan))
    return np.column_stack(cols)


def _ga_half(rng, parent, lower, upper, encoding):
    """One child per parent pair: one-point crossover + bit flip (binary) or SBX + polynomial mutation (real)."""
    h = len(parent) // 2
    P1, P2 = parent[:h], parent[h: 2 * h]
    n, D = P1.shape
    if encoding == "binary":
        k = np.tile(np.arange(1, D + 1), (n, 1)) > rng.integers(1, D + 1, (n, 1))
        off = P1.copy()
        off[k] = P2[k]
        site = rng.random((n, D)) < 1.0 / D
        off[site] = 1 - off[site]
        return off
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
    beta[rng.random((n, D)) < 0.5] = 1
    off = (P1 + P2) / 2 + beta * (P1 - P2) / 2
    lo, up = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    site, mu = rng.random((n, D)) < 1.0 / D, rng.random((n, D))
    off = np.minimum(np.maximum(off, lo), up)
    span = up - lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo[t]) / span[t]) ** 21) ** (1 / 21) - 1)
        t = site & (mu > 0.5)
        off[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - off[t]) / span[t]) ** 21) ** (1 / 21))
    return np.minimum(np.maximum(off, lo), up)


class SLMEA(LoopAlgorithm):
    """Variables that are sometimes on and sometimes off (mixed) are clustered by an archive-based score into groups
    whose masks/values are encoded by their mean; a share ``P`` of the mating pairs is varied in that reduced group space
    (cluster offspring) and the rest in the original space, and ``P`` and the number of groups adapt to which kind of
    offspring reaches the first front."""

    def __init__(self, pop_size: int = 100, use_gpu: bool = False, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.use_gpu = bool(use_gpu)    # the vectorised NumPy operators are device independent; kept for API compatibility

    def _initialize_infill(self):
        rng, N, D, enc = self.rng, self.N, self.D, self.encoding
        Mask = np.zeros((N, D))
        for i in range(N):
            Mask[i, tournament(2, int(np.ceil(rng.random() * D)), np.ones(D), rng=rng)] = 1
        Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        Dec[:, enc == 4] = 1
        pop = self.evaluate(Dec * Mask)
        self.pop_, self.Dec, self.Mask, self.front, self.crowd = nsga2_mask_selection(pop, Dec, Mask, N)
        return self.pop_

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.P, self.num_group, self.Lr, self.RC, self.T, self.gen = 0.5, 10, 0.0, 1.0, 20, 1
        self.arc_obj = objs(self.pop)[self.front == 1]
        self.arc_mask = self.Mask[self.front == 1]
        self.fitness, self.mix, self.zero, self.one = _calculate_fitness(self.arc_mask, self.D, self.rng)
        self._set_optimum()

    def _operator(self, pdec, pmask):
        rng, D = self.rng, self.D
        n = len(pdec)
        h = n // 2
        P1m, P2m, P1d, P2d = pmask[:h], pmask[h:], pdec[:h], pdec[h:]
        lower, upper = self.lower, self.upper
        fit = self.fitness[self.mix]
        Index, MAX, index = None, 0, None
        if self.gen >= self.T:
            with np.errstate(invalid="ignore"):
                grow = np.ceil(self.num_group * self.RC)
            # (at least one group: with no cluster offspring in the last generation the reference would divide by zero)
            self.num_group = max(1, int(np.fmin(200, np.fmin(len(self.mix), grow))))
            if len(self.mix):
                index, MAX = _cpu_group(self.num_group, fit, len(self.mix))
            else:
                index, MAX = np.zeros(0, dtype=int), 0
            Index = np.zeros(len(lower), dtype=int)
            if len(self.zero) and len(self.one):
                Index[self.zero], Index[self.one] = MAX + 1, MAX + 2
                MAX += 2
            elif len(self.one):
                Index[self.one] = MAX + 1
                MAX += 1
            elif len(self.zero):
                Index[self.zero] = MAX + 1
                MAX += 1
            Index[self.mix] = index
        loc = rng.random(h) < self.P
        cluster = bool(loc.any()) and self.gen >= self.T
        long = 0
        cl_mask = np.zeros((0, D))
        if cluster:
            m11 = _encode_mask(P1m[loc], Index, MAX, rng)
            m12 = _encode_mask(P2m[loc], Index, MAX, rng)
            or1m, or2m = P1m[~loc], P2m[~loc]
            span = upper - lower
            d11 = _encode_dec((P1d[loc] - lower) / span, Index, MAX)
            d12 = _encode_dec((P2d[loc] - lower) / span, Index, MAX)
            or1d, or2d = P1d[~loc], P2d[~loc]
            glow, gup = np.zeros(MAX), np.ones(MAX)
            cl = _ga_half(rng, np.vstack([m11, m12]), glow, gup, "binary")
            cl_mask = cl[:, Index - 1]
            long = len(cl_mask)
        else:
            or1m, or2m, or1d, or2d = P1m, P2m, P1d, P2d
        or_mask = _ga_half(rng, np.vstack([or1m, or2m]), lower, upper, "binary")
        off_mask = np.vstack([cl_mask, or_mask])
        if np.any(self.encoding != 4):
            cl_dec = np.zeros((0, D))
            if cluster:
                cl_dec = _ga_half(rng, np.vstack([d11, d12]), glow, gup, "real")[:, Index - 1]
                cl_dec = cl_dec * (upper - lower) + lower
            or_dec = _ga_half(rng, np.vstack([or1d, or2d]), lower, upper, "real")
            off_dec = np.vstack([cl_dec, or_dec])
            off_dec = np.minimum(np.maximum(off_dec, lower), upper)
            off_dec[:, self.encoding == 4] = 1
        else:
            off_dec = np.ones((h, D))
        return off_dec, off_mask, long

    def step(self):
        N, rng = self.N, self.rng
        mate = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
        off_dec, off_mask, long = self._operator(self.Dec[mate], self.Mask[mate])
        off = self.evaluate(off_dec * off_mask)
        self.gen += 1
        self.pop, self.Dec, self.Mask, self.front, self.crowd = nsga2_mask_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, off_dec]), np.vstack([self.Mask, off_mask]), N)
        if self.gen >= self.T + 1:
            fn, _ = nd_sort(objs(off), None, 1)
            loc = np.where(fn == 1)[0]
            a, b = np.float64(np.sum(loc < long)), np.float64(np.sum(loc >= long))
            with np.errstate(all="ignore"):
                ratio = a * (N - long) / (a * (N - long) + b * long)
                self.P = float(np.fmin(0.95, np.fmax(0.05, 0.5 * (self.P + ratio))))
                self.RC = float(np.exp((a / (long + 0.00001) - self.Lr) / self.num_group))
                self.Lr = a / np.float64(long)
        F = objs(self.pop)
        arc_obj = np.vstack([self.arc_obj, F[self.front == 1]])
        arc_mask = np.vstack([self.arc_mask, self.Mask[self.front == 1]])
        arc_obj, ia = np.unique(arc_obj, axis=0, return_index=True)
        f1, _ = nd_sort(arc_obj, None, 1)
        arc_mask = arc_mask[ia][f1 == 1]
        arc_obj = arc_obj[f1 == 1]
        if len(arc_obj) > 100:
            keep = rng.permutation(len(arc_obj))[:100]
            arc_obj, arc_mask = arc_obj[keep], arc_mask[keep]
        self.arc_obj, self.arc_mask = arc_obj, arc_mask
        self.fitness, self.mix, self.zero, self.one = _calculate_fitness(arc_mask, self.D, rng)
