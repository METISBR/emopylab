# emopylab 2026
"""S-NSGA-II (sparse nondominated sorting genetic algorithm II).

Reference:
I. Kropp, A. Pouyan Nejadhashemi, and K. Deb. Improved evolutionary operators for sparse large-scale
multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2024, 28(2):
460-473.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, nd_sort, objs, tournament
from core.population import Population

ALGORITHM_FLAGS = {'SNSGAII': {'constrained', 'large', 'multi', 'real', 'sparse'}}


def _round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def _poly_core(g, lb, ub, eta, rng):
    """Polynomial mutation of the entries ``g`` (bounds ``lb``/``ub`` broadcast per entry)."""
    d1, d2 = (g - lb) / (ub - lb), (ub - g) / (ub - lb)
    e = 1.0 / (eta + 1)
    ran = rng.random(g.shape)
    dq = np.zeros_like(g)
    left, right = ran < 0.5, ran >= 0.5
    with np.errstate(all="ignore"):
        dl = ((2.0 * ran + (1.0 - 2.0 * ran) * (1 - d1) ** (eta + 1.0)) ** e) - 1.0
        dr = 1.0 - ((2.0 * (1.0 - ran) + 2.0 * (ran - 0.5) * (1.0 - d2) ** (eta + 1.0)) ** e)
    dq[left], dq[right] = dl[left], dr[right]
    return np.minimum(np.maximum(g + dq * (ub - lb), lb), ub)


def _sm2target(P, lb, ub, new_sp, rng):
    """Add or remove non-zero entries so every solution reaches its target sparsity (fraction of zeros)."""
    if P.size == 0:
        return P
    P = P.copy()
    N, D = P.shape
    sp = np.sum(P == 0, axis=1) / D
    for i in np.where(new_sp != sp)[0]:
        k = int(_round_half_up(D * (sp[i] - new_sp[i])))
        if k > 0:
            zeros = np.where(P[i] == 0)[0]
            flip = zeros[rng.permutation(len(zeros))[:k]]
            P[i, flip] = lb[flip] + rng.random(len(flip)) * (ub[flip] - lb[flip])
        else:
            nz = np.where(P[i] != 0)[0]
            flip = nz[rng.permutation(len(nz))[:-k]]
            P[i, flip] = 0.0
    return P


def _spm(P, lb, ub, rng, prob_mut=1.0, distr_mut=20.0, prob_smut=1.0, distr_smut=20.0):
    """Sparse polynomial mutation: non-zero variables mutate with probability 1/D, then the sparsity level itself mutates."""
    P = P.copy()
    N, D = P.shape
    site = (P != 0) & (rng.random((N, D)) < prob_mut / D)
    rows, cols = np.where(site)
    if len(rows):
        P[rows, cols] = _poly_core(P[rows, cols], lb[cols], ub[cols], distr_mut, rng)
    m = rng.random(N) < prob_smut / D
    sp = np.sum(P == 0, axis=1) / D
    new_sp = sp.copy()
    if m.any():
        new_sp[m] = _poly_core(sp[m], np.zeros(int(m.sum())), np.ones(int(m.sum())), distr_smut, rng)
    new_sp = np.minimum(np.maximum(new_sp, 0), 1)
    return P if np.all(new_sp == sp) else _sm2target(P, lb, ub, new_sp, rng)


def _ssbx(P, lb, ub, rng, pro_c=1.0, dis_c=20.0):
    """SBX on the variables where both parents are zero or both are non-zero; the others are exchanged between the
    parents for a random share of the positions."""
    h = len(P) // 2
    P1, P2 = P[:h], P[h: 2 * h]
    match = (P1 == 0) == (P2 == 0)
    O1, O2 = np.full(P1.shape, -99.0), np.full(P2.shape, -99.0)
    rows, cols = np.where(match)
    a, b = P1[rows, cols], P2[rows, cols]
    mu = rng.random(a.shape)
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (dis_c + 1)), (2 - 2 * mu) ** (-1 / (dis_c + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, a.shape)
    beta[rng.random(a.shape) < 0.5] = 1
    if rng.random() > pro_c:
        beta[:] = 1
    O1[rows, cols] = np.minimum(np.maximum((a + b) / 2 + beta * (a - b) / 2, lb[cols]), ub[cols])
    O2[rows, cols] = np.minimum(np.maximum((a + b) / 2 - beta * (a - b) / 2, lb[cols]), ub[cols])
    nr, nc = np.where(~match)
    k = len(nr)
    if k:
        ratio = rng.random()
        zero_n = int(_round_half_up(k * ratio))
        keep = np.ones(k, bool)                           # True = swap the parents' values
        if ratio != 0:
            keep[rng.permutation(k)[:zero_n]] = False
        p1, p2 = P1[nr, nc].copy(), P2[nr, nc].copy()
        q1, q2 = np.where(keep, p2, p1), np.where(keep, p1, p2)
        O1[nr, nc], O2[nr, nc] = q1, q2
    return np.vstack([O1, O2])


def _vssps(algo, s_lower, s_upper):
    """Variable-sparsity sampling: individuals get decreasing numbers of active variables packed in blocks."""
    N, D = algo.N, algo.D
    pop = algo.evaluate(algo.random_decs(N))              # the random start is charged, only its decisions are reused
    density = 1 - np.linspace(s_lower, s_upper, N)
    width = _round_half_up(density * D).astype(int)
    lb = int(np.floor((1 - s_lower) * D))
    width[width > lb] = lb
    cum = np.cumsum(width)
    processed = N if np.sum(width == 0) == N else 0
    cycles = []
    while processed < N:
        fit = (cum <= D) & (cum != 0)
        n_fit = int(fit.sum())
        if n_fit == 0:
            break
        largest = cum[fit].max()
        cum = np.maximum(cum - largest, 0)
        processed += n_fit
        cycles.append(np.where(fit)[0])
    mask = np.zeros((N, D), bool)
    cur = 0
    for c, cyc in enumerate(cycles):
        w = width[cyc]
        gap_to_fill = D - w.sum()
        gap_size = int(np.ceil((D - w.sum()) / len(w)))
        pos = 0
        for wi in w:
            gap = 0
            if gap_to_fill > 0:
                gap = gap_size
                gap_to_fill -= gap
            end = pos + wi - 1 + (0 if c == len(cycles) - 1 else gap)
            end = min(end, D - 1)
            mask[cur, pos: end + 1] = True
            pos += wi + gap
            cur += 1
    X = decs(pop).copy()
    X[~mask] = 0
    return algo.solutions_uncharged(X)


def _env_selection(pop, N):
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], front[nxt], cd[nxt]


class SNSGAII(LoopAlgorithm):
    """NSGA-II whose start population has graded sparsity (``vssps``) and whose variation keeps the sparsity pattern:
    sparse SBX (``ssbx``) and sparse polynomial mutation that also mutates each solution's sparsity (``spm``)."""

    UNCHARGED_EVALS = True    # the sparsified start population is evaluated without being charged (as in the reference)

    def __init__(self, pop_size: int = 100, sparsity_lower: float = 0.75, sparsity_upper: float = 1.0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.s_lower, self.s_upper = float(sparsity_lower), float(sparsity_upper)

    def _initialize_infill(self):
        return _vssps(self, self.s_lower, self.s_upper)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        _, self.front, self.crowd = _env_selection(infills, self.N)
        self._set_optimum()

    def step(self):
        rng, N = self.rng, self.N
        if np.any(self.encoding != 1):
            raise ValueError("S-NSGA-II supports real encoding only")
        pop = self.pop
        mate = tournament(2, N, self.front, -self.crowd, rng=rng)
        off = _ssbx(decs(pop[mate]), self.lower, self.upper, rng)
        off = _spm(off, self.lower, self.upper, rng)
        self.pop, self.front, self.crowd = _env_selection(Population.merge(pop, self.evaluate(off)), N)
