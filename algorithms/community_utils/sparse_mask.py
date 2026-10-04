"""Building blocks shared by the mask-based sparse evolutionary algorithms (decision vector = real part x binary mask)."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import cons, crowding, decs, nd_sort, objs, tournament
from algorithms.community_utils.spea import cal_fitness, truncation

__all__ = ["probe_variables", "unique_by_objs", "nsga2_mask_selection", "spea2_mask_selection", "group_operator_half", "ts", "lhs_design"]


def ts(fit, rng):
    """One binary tournament over a fitness vector (lower wins); ``None`` for an empty vector."""
    return None if len(fit) == 0 else int(tournament(2, 1, fit, rng=rng)[0])


def probe_variables(algo, with_standard: bool = False):
    """Single-variable probes: for each of ``1 + 4*any(non-binary)`` repetitions evaluate the identity-masked random
    decision matrix; returns ``(dec, mask, pop, score)`` with the non-domination ranks summed per variable.

    With ``with_standard`` the all-zero solution is probed as well (its rank is the first score entry)."""
    rng, D, enc = algo.rng, algo.D, algo.encoding
    lo, up = algo.lower, algo.upper
    Dec, Mask, Pop = [], [], []
    score = np.zeros(D + int(with_standard))
    for _ in range(1 + 4 * int(np.any(enc != 4))):
        dec = lo + rng.random((D, D)) * (up - lo)
        dec[:, enc == 4] = 1
        mask = np.eye(D)
        X = dec * mask
        if with_standard:
            X = np.vstack([np.zeros((1, D)), X])
        p = algo.evaluate(X)
        Dec.append(dec), Mask.append(mask)
        Pop.append(p[1:] if with_standard else p)
        C = cons(p)
        score += nd_sort(np.hstack([objs(p), C]) if C.size else objs(p), None, np.inf)[0]
    return Dec, Mask, Pop, score


def unique_by_objs(pop, *arrays):
    """Keep the first solution of every distinct objective vector (rows in lexicographic order)."""
    idx = np.unique(objs(pop), axis=0, return_index=True)[1]
    return (pop[idx],) + tuple(a[idx] for a in arrays)


def nsga2_mask_selection(pop, Dec, Mask, N):
    pop, Dec, Mask = unique_by_objs(pop, Dec, Mask)
    N = min(N, len(pop))
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], Dec[nxt], Mask[nxt], front[nxt], cd[nxt]


def group_operator_half(algo, P1, P2, n_groups, rng):
    """One SBX child per parent pair followed by polynomial mutation of a single group of variables (variables grouped
    by value).  Returns ``(offspring, group_index, chosen_group)``."""
    from algorithms.glmo.glmo import _group_mutation, create_groups
    n, D = P1.shape
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
    beta[rng.random((n, D)) < 0.5] = 1
    off = (P1 + P2) / 2 + beta * (P1 - P2) / 2
    lower, upper = np.tile(algo.lower, (n, 1)), np.tile(algo.upper, (n, 1))
    groups = create_groups(n_groups, off, 2, rng)
    chosen = rng.integers(1, n_groups + 1, (n, 1))
    site = groups == chosen
    mu = np.repeat(rng.random((n, 1)), D, axis=1)
    return _group_mutation(off, lower, upper, site, mu), groups, chosen


def spea2_mask_selection(pop, Dec, Mask, N):
    """Unique objective vectors, then SPEA2 environmental selection carrying the decision parts along.
    Returns ``(pop, Dec, Mask, fitness)``."""
    pop, Dec, Mask = unique_by_objs(pop, Dec, Mask)
    N = min(N, len(pop))
    F = objs(pop)
    fit = cal_fitness(F)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return pop[nxt], Dec[nxt], Mask[nxt], fit[nxt]


def lhs_design(rng, n, p):
    """Latin hypercube sample of ``n`` points in ``[0,1]^p`` (one point per stratum and column, random inside it)."""
    return (np.column_stack([rng.permutation(n) for _ in range(p)]) + rng.random((n, p))) / n
