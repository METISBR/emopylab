# emopylab 2026
"""ICMA (indicator-based constrained multi-objective algorithm).

Reference:
J. Yuan, H. Liu, Y. Ong, and Z. He. Indicator-based evolutionary algorithm for solving constrained
multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2022, 26(2):
379-391.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, de, objs, uniform_point
from core.population import Population

ALGORITHM_FLAGS = {'ICMA': {'constrained', 'integer', 'multi', 'real'}}


def _neighbors_update(values, nb, mat, deleted):
    need = np.where(nb == deleted)[0]
    if len(need):
        values[need], nb[need] = mat[need].min(axis=1), mat[need].argmin(axis=1)


def _prea_selection(F, IM, N):
    M = F.shape[1]
    ir = IM.min(axis=1)
    lvl1 = np.where(ir >= 0)[0]
    n1 = len(lvl1)
    if n1 <= N:
        return np.argsort(-ir, kind="stable")[:N]
    alli = lvl1.copy()
    F, IM = F[lvl1], IM[np.ix_(lvl1, lvl1)]
    mid = IM.copy()
    values, nb = mid.min(axis=1), mid.argmin(axis=1)
    deleted = []
    for _ in range(n1 - N):
        d = int(np.argmin(values))
        deleted.append(d)
        mid[d, :] = np.inf
        mid[:, d] = np.inf
        _neighbors_update(values, nb, mid, d)
        values[d] = np.inf
    best = np.delete(np.arange(n1), deleted)
    zmax = F[best].max(axis=0)
    out = np.where(np.min(zmax - F, axis=1) < 0)[0]
    alli = np.delete(alli, out)
    F = np.delete(F, out, axis=0)
    IM = np.delete(np.delete(IM, out, axis=0), out, axis=1)
    num = len(alli)
    F = F / zmax
    ir_val, ir_nb = IM.min(axis=1), IM.argmin(axis=1)
    delta = F[None, :, :] - F[:, None, :]
    D = np.sqrt(np.maximum(np.sum(delta ** 2, axis=2) - np.sum(delta, axis=2) ** 2 / M, 0.0))
    np.fill_diagonal(D, np.inf)
    dv, dn = D.min(axis=1), D.argmin(axis=1)
    dele = []
    for _ in range(num - N):
        i1 = int(np.argmin(dv))
        i2 = int(dn[i1])
        k = i1 if ir_val[i1] < ir_val[i2] else i2
        dele.append(k)
        D[k, :] = np.inf
        D[:, k] = np.inf
        _neighbors_update(dv, dn, D, k)
        dv[dele] = np.inf
        IM[k, :] = np.inf
        IM[:, k] = np.inf
        _neighbors_update(ir_val, ir_nb, IM, k)
    return np.delete(alli, dele)


def _indicator_cht(F, IM, W, N):
    ir = IM.min(axis=1)
    lvl1 = np.where(ir >= 0)[0]
    if len(lvl1) <= N:
        return np.argsort(-ir, kind="stable")[:N]
    sel = lvl1.copy()
    F, IM = F[lvl1], IM[np.ix_(lvl1, lvl1)]
    num = len(F)
    nw = W / np.linalg.norm(W, axis=1, keepdims=True)
    nf = F / np.linalg.norm(F, axis=1, keepdims=True)
    zone_of = np.argmax(nf @ nw.T, axis=1)
    NW = len(W)
    density = np.bincount(zone_of, minlength=NW).astype(float)
    order = np.argsort(-density, kind="stable")
    dens = density[order]
    zones = {z: list(np.where(zone_of == z)[0]) for z in range(NW)}
    values, nb = IM.min(axis=1), IM.argmin(axis=1)
    have = []
    for _ in range(num - N):
        md = int(np.argmax(dens))
        zi = order[md]
        cand = zones[zi]
        now = int(np.argmin(values[cand]))
        d = cand[now]
        cand.pop(now)
        have.append(d)
        IM[d, :] = np.inf
        IM[:, d] = np.inf
        _neighbors_update(values, nb, IM, d)
        values[d] = np.inf
        dens[md] -= 1
    return np.delete(sel, have)


def icma_update(pop, N, W, zmin, fmin):
    F = objs(pop)
    num = len(F)
    C = np.maximum(0, cons(pop))
    CV = (C / np.maximum(1, C.max(axis=0))).sum(axis=1) if C.size else np.zeros(num)
    F = F - zmin + 1e-6
    with np.errstate(all="ignore"):
        A = np.log(F[:, None, :] / F[None, :, :])                    # A[i, j] = log(F_i / F_j)
        mx, mn = A.max(axis=2), A.min(axis=2)
        cva = np.where(mx <= 0, mn, mx)
        ic = (CV[:, None] + 1e-6) / (CV[None, :] + 1e-6)
        hi, lo = np.maximum(F[:, None, :], F[None, :, :]), np.minimum(F[:, None, :], F[None, :, :])
        cvf = np.max(hi / lo, axis=2)
        infe = np.log(np.maximum(cvf, ic))
    V = np.where((CV == 0)[:, None], cva, infe)
    IM = V.T.copy()
    np.fill_diagonal(IM, np.inf)
    feas = np.where(CV == 0)[0]
    if len(feas) <= N:
        archive = pop[np.argsort(CV, kind="stable")[:N]]
    else:
        FP = F[feas] + zmin - fmin
        sel = _prea_selection(FP, IM[np.ix_(feas, feas)], N)
        archive = pop[feas[sel]]
    return pop[_indicator_cht(F, IM, W, N)], archive


class ICMA(LoopAlgorithm):
    """Indicator-based constrained multi-objective algorithm: a population guided by a constraint-aware binary
    indicator and an archive of feasible solutions cooperate, with a neighbour-pairing DE that shifts its mating
    source from the population to the archive as the run advances."""

    PER_STEP_OPTIMUM = True

    def start(self):
        N = self.N
        self.Zmin = objs(self.pop).min(axis=0)
        C = cons(self.pop)
        feas = np.all(C <= 0, axis=1) if C.size else np.ones(N, bool)
        self.Fmin = objs(self.pop)[feas].min(axis=0) if feas.any() else None
        self.archive = self.pop
        self.W, _ = uniform_point(N, self.M)
        self.Ra = 1.0

    def _neighbor_pairing(self, mating):
        rng = self.rng
        F = objs(mating) - self.Zmin
        n = len(F)
        with np.errstate(all="ignore"):
            F = F / np.linalg.norm(F, axis=1, keepdims=True)
            cosv = F @ F.T - 3 * np.eye(n)
        sind = np.argsort(-np.where(np.isnan(cosv), -np.inf, cosv), axis=1, kind="stable")
        nb = sind[:, :10]
        P = np.zeros((n, 2), dtype=int)
        for i in range(n):
            P[i] = nb[i, rng.permutation(10)[:2]]
            if rng.random() > 0.7:
                P[i, 1] = int(rng.integers(n))
        return mating, mating[P[:, 0]], mating[P[:, 1]]

    def step(self):
        N, rng = self.N, self.rng
        nt = int(np.floor(self.Ra * N))
        pool = Population.merge(self.pop[rng.permutation(N)[:nt]], self.archive[rng.permutation(N)[: N - nt]])
        m1, m2, m3 = self._neighbor_pairing(pool)
        params = None if rng.random() > 0.5 else [0.5, 0.5, 0.5, 0.75]
        off = self.evaluate(de(self.problem, decs(m1), decs(m2), decs(m3), params, rng=rng))
        Co = cons(off)
        fo = np.all(Co <= 0, axis=1) if Co.size else np.ones(len(off), bool)
        if fo.any():
            fm = objs(off)[fo].min(axis=0)
            self.Fmin = fm if self.Fmin is None else np.minimum(self.Fmin, fm)
        self.Zmin = np.minimum(self.Zmin, objs(off).min(axis=0))
        self.pop, self.archive = icma_update(Population.merge(self.pop, off, self.archive), N, self.W, self.Zmin, self.Fmin)
        self.Ra = 1 - self.FE / self.max_FE

    def _set_optimum(self) -> None:
        from util.optimum import filter_optimum
        self.opt = filter_optimum(getattr(self, "archive", self.pop), least_infeasible=True)
