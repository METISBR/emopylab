# emopylab 2026
"""CGLP (correlation-guided layered prediction).

Reference:
K. Yu, D. Zhang, J. Liang, K. Chen, C. Yue, K. Qiao, and L. Wang. A correlation-guided layered
prediction approach for evolutionary dynamic multiobjective optimization. IEEE Transactions on
Evolutionary Computation, 2025, 27(5): 1398-1412.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs
from algorithms.kl_nsga_ii.kl_nsga_ii import changed
from algorithms.rm_meda.rm_meda import _local_pca
from core.population import Population

ALGORITHM_FLAGS = {'CGLP': {'binary', 'dynamic', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _pdist(A, B):
    return np.sqrt(np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0))


def _sph(norm, fi):
    """Hyperspherical coordinates (radius ``norm``, angles ``fi``) -> Cartesian vector of length len(fi) + 1."""
    D = len(fi) + 1
    u = np.zeros(D)
    for i in range(D):
        pre = np.prod(np.sin(fi[:i])) if i else 1.0
        u[i] = norm * pre * (np.cos(fi[i]) if i < D - 1 else 1.0)
    return u


def _angles(v):
    D = len(v)
    out = np.zeros(D - 1)
    with np.errstate(all="ignore"):
        for i in range(D - 2):
            out[i] = np.arctan(np.sqrt(np.sum(v[i + 1:] ** 2)) / v[i])
        out[D - 2] = np.arctan(v[D - 1] / v[D - 2])
    return out


def _dlcm(rng, K1, K2, pop1, pop2, tipe):
    """Predict with the displacement of the group centroid and per-individual hyperspherical displacement (columns = individuals)."""
    D, n = K1.shape
    vec = K1 - K2
    Cw11, Cw12 = K1.mean(axis=1), K2.mean(axis=1)
    Dw = Cw11 - Cw12
    with np.errstate(all="ignore"):
        bi = np.linalg.norm(K1 - K2, axis=0) / np.linalg.norm(Dw)
        bi_o = bi.mean() / n
        bi = bi + rng.normal(0, bi_o)
    bian = np.abs(bi)[:, None] * Dw[None, :]
    pie = np.column_stack([_angles(vec[:, j]) for j in range(n)])
    dw_pie = np.zeros(D - 1)
    with np.errstate(all="ignore"):
        for i in range(D - 2):
            dw_pie[i] = np.arctan(np.sqrt(np.sum(Dw[i + 1:] ** 2)) / Dw[i])
            if dw_pie[i] < 0:
                dw_pie[i] += np.pi
        dw_pie[D - 2] = np.arctan(Dw[D - 1] / Dw[D - 2])
    sample = np.column_stack([_sph(np.linalg.norm(vec[:, j]), pie[:, j]) for j in range(n)])
    d412 = _sph(np.linalg.norm(Dw), dw_pie)
    pam = np.sum(K2 + sample - K1, axis=0)
    d41pam = np.sum(Cw12 + d412 - Cw11)
    for k in range(n):
        if abs(pam[k]) > 1:
            pie[-1, k] += -np.pi if pie[-1, k] > 0 else np.pi
    if abs(d41pam) > 1:
        dw_pie[-1] += -np.pi if dw_pie[-1] > 0 else np.pi
    sample = np.column_stack([_sph(np.linalg.norm(bian[j]), 0.5 * (dw_pie + pie[:, j])) for j in range(n)])
    pop = np.mod(np.abs(K1 + sample), 1).T
    if tipe == 0:
        selecta = n / 2
        tipe = selecta
    else:
        selecta = tipe
    cp = np.where(rng.random(n) < selecta / n)[0]
    dcm = np.setdiff1d(np.arange(n), cp)
    pop[cp] = K1[:, cp].T + bian[cp]
    pop1 = np.vstack([pop1, pop[cp]])
    pop2 = np.vstack([pop2, pop[dcm]])
    return pop, pop1, pop2, tipe


def _selfadjust(px, pop1, pop2, tipe):
    def mean_min(P):
        if len(P) == 0:
            return np.nan, 0
        m = _pdist(P, px).min(axis=1)
        m = m[~np.isnan(m)]
        return (m.sum() / len(m) if len(m) else np.nan), len(m)

    j1, g1 = mean_min(pop1)
    j2, g2 = mean_min(pop2)
    if j1 < j2:
        if g2 > 3:
            tipe += 1
    elif g1 > 3:
        tipe -= 1
    return tipe


class CGLP(LoopAlgorithm):
    """RM-MEDA search in the first two environments; after every detected change the previous two fronts are compared: the
    individuals are grouped by how well their displacement follows the centroid displacement (grey relational analysis) and the
    groups are predicted by translation, by a hyperspherical linear/nonlinear model, or taken from the previous elite set."""

    def __init__(self, pop_size: int = 100, fe_init: int = 10000, taut: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.fe_init, self.taut = int(fe_init), int(taut)

    def _operator(self, pop, K=5):
        rng, M, D = self.rng, self.M, self.D
        X = decs(pop)
        N = len(X)
        means, evec, eval_, a, b, prob = _local_pca(X, M, K, rng)
        off = np.zeros((N, D))
        for i in range(N):
            k = int(np.argmax(rng.random() <= prob))
            if evec[k] is not None:
                lo, up = a[k] - 0.25 * (b[k] - a[k]), b[k] + 0.25 * (b[k] - a[k])
                trial = rng.random(M - 1) * (up - lo) + lo
                sigma = np.sum(np.abs(eval_[k][M - 1:D])) / (D - M + 1)
                off[i] = means[k] + trial @ evec[k][:, : M - 1].T + rng.standard_normal(D) * np.sqrt(sigma)
            else:
                off[i] = means[k] + rng.standard_normal(D)
        return self.evaluate(off)

    def _select(self, pop, N):
        F = objs(pop)
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        while len(last) > N - int(nxt.sum()):
            last = np.delete(last, int(np.argmin(crowding(F[last]))))
        nxt[last] = True
        return pop[nxt]

    def _rmmeda(self, init=None):
        N = self.N
        if init is None:
            pop = self.evaluate(self.random_decs(N))
        else:
            X = np.clip(init, self.lower, self.upper)
            pop = self.evaluate(X)
        while self.FE < self.fe_init:
            off = self._operator(pop)
            pop = self._select(Population.merge(pop, off), N)
        self.pop_x = decs(pop)
        return pop

    def _record(self, pop):
        T = self.T
        F, X = objs(pop), decs(pop)
        order = np.argsort(F[:, 0], kind="stable")
        F, X = F[order], X[order]
        self.his_F[T], self.his_X[T] = F, X
        front, _ = nd_sort(F, None, np.inf)
        idx = np.where(front == 1)[0]
        cd = crowding(F[idx])
        self.his_pareto[T] = X[idx][np.argsort(-cd, kind="stable")]

    def _initialize_infill(self):
        self.his_F, self.his_X, self.his_pareto = {}, {}, {}
        self.T, self.tipe = 1, 0
        self.all_pop = None
        pop = None
        for _ in range(2):
            pop = self._rmmeda()
            self.all_pop = pop if self.all_pop is None else Population.merge(self.all_pop, pop)
            self._record(pop)
            self.T += 1
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _predict(self):
        rng, N, D = self.rng, self.N, self.D
        T = self.T
        F1, X1, F2, X2 = self.his_F[T - 1], self.his_X[T - 1], self.his_F[T - 2], self.his_X[T - 2]
        ran = self.lower + rng.random((len(X1), D)) * (self.upper - self.lower)
        mkl = np.argmin(_pdist(F1[:N], F2), axis=1)
        X2 = X2[mkl]
        X1 = X1[:N]
        D41 = X1 - X2
        D11 = X1.mean(axis=0) - X2.mean(axis=0)
        x = np.vstack([D41, D11])
        with np.errstate(all="ignore"):
            x = x / (x[:, [0]] + 0.0001)
        ck, bj = x[-1], x[:-1]
        te = bj - ck
        jc1, jc2 = np.min(np.abs(te)), np.max(np.abs(te))
        ksi = (jc1 + 0.5 * jc2) / (np.abs(te) + 0.5 * jc2)
        r = ksi.sum(axis=1) / ksi.shape[1]
        rind = np.argsort(-r, kind="stable")
        n1, n2 = int(round(0.6 * N)), int(round(0.9 * N))
        num1, num2, num3 = rind[:n1], rind[n1:n2], rind[n2:]
        pre = np.zeros((N, D))
        dw = X1[num1].mean(axis=0) - X2[num1].mean(axis=0)
        pre[num1] = X1[num1] + dw
        pre[num2], pop_lcm, pop_dcm, self.tipe = _dlcm(rng, X1[num2].T, X2[num2].T, np.zeros((0, D)), np.zeros((0, D)), self.tipe)
        hp = self.his_pareto[T - 1]
        if len(hp) > len(num3):
            pre[num3] = hp[: len(num3)]
        else:
            pre[num3[: len(hp)]] = hp
        POP = pre[:N]
        bad = (POP < self.lower) | (POP > self.upper)
        POP[bad] = ran[:N][bad]
        return POP, pop_lcm, pop_dcm

    def step(self):
        if changed(self, self.pop):
            init, lcm, dcm = self._predict()
            self.pop = self._rmmeda(init)
            self.tipe = _selfadjust(self.pop_x, lcm, dcm, self.tipe)
            self.all_pop = Population.merge(self.all_pop, self.pop)
            self._record(self.pop)
            self.T += 1
        if self.FE >= self.max_FE:
            self.pop = Population.merge(self.all_pop, self.pop)
