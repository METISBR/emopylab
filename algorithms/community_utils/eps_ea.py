"""Differential-evolution helpers shared by the epsilon-constrained algorithms (DVCEA, DSSEA, DBEMTO)."""

from __future__ import annotations

import numpy as np

__all__ = ["gn_r1r2r3", "de_pbest_1", "de_rand_1", "cal_fitness_eps"]

_FM = np.array([0.6, 0.8, 1.0])
_CRM = np.array([0.1, 0.2, 1.0])


def gn_r1r2r3(rng, NP1, r0):
    """Random index vectors (1-based like the reference) with r1 != r0, r2 not in {r0, r1}, r3 not in {r0, r1, r2}."""
    n0 = len(r0)
    r1 = np.floor(rng.random(n0) * NP1).astype(int) + 1
    for _ in range(1001):
        pos = r1 == r0
        if not pos.any():
            break
        r1[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int) + 1
    r2 = np.floor(rng.random(n0) * NP1).astype(int) + 1
    for _ in range(1001):
        pos = (r2 == r1) | (r2 == r0)
        if not pos.any():
            break
        r2[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int) + 1
    r3 = np.floor(rng.random(n0) * NP1).astype(int) + 1
    for _ in range(1001):
        pos = (r3 == r1) | (r3 == r0) | (r3 == r2)
        if not pos.any():
            break
        r3[pos] = np.floor(rng.random(int(pos.sum())) * NP1).astype(int) + 1
    return r1, r2, r3


def de_pbest_1(rng, lower, upper, P1, P2, P3, P4):
    """DE/current-to-pbest style move ``x + F (pbest - x + p3 - p4)`` on randomly chosen variables (CR from {.1,.2,1}, F from
    {.6,.8,1}) followed by polynomial mutation."""
    N, D = P1.shape
    F = np.repeat(_FM[rng.integers(0, 3, N)][:, None], D, axis=1)
    CR = _CRM[rng.integers(0, 3, N)][:, None]
    site = rng.random((N, D)) < CR
    off = P1.copy()
    off[site] = off[site] + F[site] * (P2[site] - off[site] + P3[site] - P4[site])
    lo, up = np.tile(lower, (N, 1)), np.tile(upper, (N, 1))
    site, mu = rng.random((N, D)) < 1.0 / D, rng.random((N, D))
    off = np.minimum(np.maximum(off, lo), up)
    span = up - lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo[t]) / span[t]) ** 21) ** (1 / 21) - 1)
        t = site & (mu > 0.5)
        off[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - off[t]) / span[t]) ** 21) ** (1 / 21))
    return off


def de_rand_1(rng, lower, upper, P1, P2, P3):
    """DE/rand/1 (F from {.6,.8,1}, CR from {.1,.2,1} per individual) followed by polynomial mutation."""
    N, D = P1.shape
    F = np.repeat(_FM[rng.integers(0, 3, N)][:, None], D, axis=1)
    CR = _CRM[rng.integers(0, 3, N)][:, None]
    site = rng.random((N, D)) < CR
    off = P1.copy()
    off[site] = off[site] + F[site] * (P2[site] - P3[site])
    lo, up = np.tile(lower, (N, 1)), np.tile(upper, (N, 1))
    site, mu = rng.random((N, D)) < 1.0 / D, rng.random((N, D))
    off = np.minimum(np.maximum(off, lo), up)
    span = up - lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] += span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo[t]) / span[t]) ** 21) ** (1 / 21) - 1)
        t = site & (mu > 0.5)
        off[t] += span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - off[t]) / span[t]) ** 21) ** (1 / 21))
    return off


def cal_fitness_eps(F, C=None, epsilon=None):
    """Strength/density fitness where violations up to ``epsilon`` (inclusive) count as feasible."""
    from algorithms.community_utils.spea import cal_fitness
    if C is None or epsilon is None:
        return cal_fitness(F, C, None)
    cv = np.sum(np.maximum(0.0, C), axis=1) if np.size(C) else np.zeros(len(F))
    return cal_fitness(F, np.where(cv <= epsilon, 0.0, cv)[:, None], None)
