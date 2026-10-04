# emopylab 2026
"""Inverse-modelling reproduction operator shared by IM-MOEA and IM-MOEA/D."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.surrogates import gp_linear_predict


def inverse_model_offspring(algo, pop, L: int = 3) -> np.ndarray:
    """Offspring decisions of one sub-population: for each objective m a random subset is used to learn
    ``x_d = GP(f_m)`` on ``L`` random variables and new values are sampled along an extended f_m range."""
    rng = algo.rng
    X = np.asarray(pop.get("X"), dtype=float)
    F = np.asarray(pop.get("F"), dtype=float)
    N, D = X.shape
    M = algo.M
    if N < 2 * M:
        off = X.copy()
    else:
        fmin = 1.5 * F.min(axis=0) - 0.5 * F.max(axis=0)
        fmax = 1.5 * F.max(axis=0) - 0.5 * F.min(axis=0)
        blocks = []
        for m in range(M):
            parents = rng.permutation(N)[: N // M]
            block = X[parents].copy()
            for d in rng.permutation(D)[: min(L, D)]:
                try:
                    mu, var = gp_linear_predict(F[parents, m], X[parents, d], np.linspace(fmin[m], fmax[m], len(block)))
                    block[:, d] = mu + rng.random() * np.sqrt(var) * rng.standard_normal(len(var))
                except np.linalg.LinAlgError:
                    pass
            blocks.append(block)
        off = np.vstack(blocks)
    lo, up = algo.lower, algo.upper
    n, D = off.shape
    invalid = (off < lo) | (off > up)
    rand_dec = rng.uniform(lo, up, size=(n, D))
    off[invalid] = rand_dec[invalid]
    site = rng.random((n, D)) < 1.0 / D
    mu = rng.random((n, D))
    off = np.minimum(np.maximum(off, lo), up)
    span = np.broadcast_to(up - lo, (n, D))
    lo_b, up_b = np.broadcast_to(lo, (n, D)), np.broadcast_to(up, (n, D))
    disM = 20.0
    t = site & (mu <= 0.5)
    off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - lo_b[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    t = site & (mu > 0.5)
    off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up_b[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off
