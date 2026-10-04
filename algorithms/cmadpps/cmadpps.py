# emopylab 2026
"""CMaDPPs (constrained many-objective optimization with determinantal point processes).

Reference:
F. Ming, W. Gong, S. Li, L. Wang, and Z. Liao. Handling constrained many-objective optimization
problems via determinantal point processes. Information Sciences, 2023, 643: 119260.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs
from algorithms.community_utils.dpp import decompose_kernel, sample_dpp
from algorithms.community_utils.robust import cosine_dist
from algorithms.community_utils.spea import overall_cv
from core.population import Population

ALGORITHM_FLAGS = {'CMaDPPs': {'constrained', 'integer', 'many', 'multi', 'real'}}


def _unique_first(F):
    return np.unique(F, axis=0, return_index=True)[1]


def _indicator(pop, z):
    """Pairwise dominance-strength indicator ``I[j, i]`` of solution ``i`` over solution ``j`` (log ratio of objectives
    for feasible ``i``, violation-aware for infeasible ``i``)."""
    F = objs(pop)
    Num = len(F)
    C = np.maximum(0.0, cons(pop))
    norm = C / np.maximum(1.0, C.max(axis=0)) if C.size else C
    CV = norm.sum(axis=1) if C.size else np.zeros(Num)
    F = F - z + 1e-6
    I = np.ones((Num, Num))
    with np.errstate(all="ignore"):
        for i in range(Num):
            Fi = F[i]
            if CV[i] == 0:
                ir = np.log(Fi / F)
                mx, mn = ir.max(axis=1), ir.min(axis=1)
                v = np.where(mx <= 0, mn, mx)
            else:
                ic = (CV[i] + 1e-6) / (CV + 1e-6)
                cvf = np.max(np.maximum(Fi, F) / np.minimum(Fi, F), axis=1)
                v = np.log(np.maximum(cvf, ic))
            I[:, i] = v
            I[i, i] = np.inf
    return I


def _norm_obj(F, z, znad):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (F - z) / (znad - z)


def _cos_matrix(A):
    D = cosine_dist(A, A)
    np.fill_diagonal(D, 0.0)
    return D


def _dpp_choose(kernel, k):
    V, w = decompose_kernel(np.nan_to_num(kernel, nan=0.0, posinf=1e12, neginf=-1e12))
    return sample_dpp(V, w, k)


def _update_archive(pop_off, max_size, z, znad, da_obj, theta, epsilon):
    F, C = objs(pop_off), cons(pop_off).copy()
    cv = overall_cv(C)
    if C.size:
        C[cv <= epsilon, :] = 0
    front, _ = nd_sort(F, C if C.size else None, 1)
    pop = pop_off[front == 1]
    pop = pop[_unique_first(objs(pop))]
    N = len(pop)
    if N <= max_size:
        return pop
    F = objs(pop)
    F2 = _norm_obj(F, z, znad)
    if len(da_obj):
        nad = da_obj.max(axis=0) + 1e-6
        nn = np.sum((nad - F) < 0, axis=1)
        da = _norm_obj(da_obj, z, znad)
        big = np.max(np.sqrt(np.sum(da ** 2, axis=1))) + 1e-6
        nn = nn + (np.sqrt(np.sum(F2 ** 2, axis=1)) > big)
    else:
        nn = np.zeros(N)
    H = (1 - theta) * np.exp(-_cos_matrix(F2))
    value = np.sum(F2 ** 2, axis=1)
    value[nn == 0] = value.min() / 2
    value = np.maximum(value / value.max(), 1e-12)
    H = H / np.outer(value, value)
    return pop[_dpp_choose(H, max_size)]


def _update_csa(csa, off, max_size, epsilon):
    csa = Population.merge(off, csa)
    csa = csa[overall_cv(cons(csa)) <= epsilon]
    if len(csa) <= max_size:
        return csa
    F = objs(csa)
    M = F.shape[1]
    cunum = int(np.ceil(max_size / (3 * M)))
    F3 = F ** 2
    dao = F3.sum(axis=1)
    ch, mi = [], []
    for i in range(M):
        f = F[:, i]
        t = f.min() + 1e-6
        if np.sum(f < t) > cunum:
            idx = np.where(f < t)[0]
            mi.extend(idx[np.argsort(dao[idx], kind="stable")[:cunum]])
        else:
            mi.extend(np.argsort(f, kind="stable")[:cunum])
        f2 = dao - F3[:, i]
        t = f2.min() + 1e-6
        if np.sum(f2 < t) > 2 * cunum:
            idx = np.where(f2 < t)[0]
            ch.extend(idx[np.argsort(dao[idx], kind="stable")[: 2 * cunum]])
        else:
            ch.extend(np.argsort(f2, kind="stable")[: 2 * cunum])
    return csa[np.array(ch + mi, dtype=int)]


class CMaDPPs(LoopAlgorithm):
    """Population and archive are thinned by sampling a determinantal point process whose kernel mixes a quality term
    (dominance-strength indicator) with a diversity term (angles between normalised objective vectors); a constraint
    handling set (CSA) and an epsilon that follows the search stage steer feasibility."""

    def __init__(self, pop_size: int = 100, theta: float = 0.0, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.theta = float(theta)

    def start(self):
        pop, N = self.pop, self.N
        self.csa = pop
        F = objs(pop)
        self.z, self.znad = F.min(axis=0), F.max(axis=0)
        self.CVmax = float(overall_cv(cons(pop)).max()) if cons(pop).size else 0.0
        self.G = int(np.ceil(self.max_FE / N))
        self.Tc, self.cp, self.alpha, self.tao = 0.8 * self.G, 2, 0.95, 0.05
        self.epsilon = np.inf
        self.last_gen, self.change_threshold, self.max_change = 20, 1e-1, 1.0
        self.ideal, self.nadir = {}, {}
        self.archive = None
        self.pop, self.archive = self._env_selection(pop, None, None)

    def _env_selection(self, pop, off, archive):
        N, z, znad, theta, eps = self.N, self.z, self.znad, self.theta, self.epsilon
        pop_off = Population.merge(*[p for p in (off, pop, archive) if p is not None and len(p)])
        F, C = objs(pop_off), cons(pop_off)
        front, _ = nd_sort(F, C if C.size else None, np.inf)
        p1 = pop_off[front == 1]
        ir1 = _indicator(pop_off, z).min(axis=1)
        p2 = pop_off[ir1 >= 0]
        new = Population.merge(p1, p2)
        new = new[_unique_first(objs(new))]
        if len(new) <= N:
            return new, archive
        ir = _indicator(new, z).min(axis=1)
        L = np.outer(ir, ir)
        F2 = _norm_obj(objs(new), z, znad)
        H = (1 - theta) * np.exp(-_cos_matrix(F2))
        chosen = new[_dpp_choose(H * L, N)]
        feas = overall_cv(cons(pop_off)) <= eps
        archive = _update_archive(pop_off[feas], N, z, znad, objs(self.csa), theta, eps)
        return chosen, archive

    def _mating(self, ca, da):
        rng, N = self.rng, self.N
        ea = Population.merge(ca, da) if da is not None and len(da) else ca
        F2 = _norm_obj(objs(ea), self.z, self.znad)
        n1 = len(F2)
        D = cosine_dist(F2, F2) + np.eye(n1)
        Dn = np.where(np.isnan(D), np.inf, D)
        mincos = np.argmin(Dn, axis=0)
        cosv = Dn[mincos, np.arange(n1)]
        cao = np.sum(F2 ** 2, axis=1)
        ch = (cao[mincos] - cao) > 0
        ch3 = np.where(ch, np.arange(n1), mincos)
        cosv = 1 - cosv
        span = cosv.max() - cosv.min()
        cosv = (cosv - cosv.min()) / span if span > 0 else np.zeros(n1)
        k = rng.integers(0, n1, 2 * N)
        take = rng.random(2 * N) < cosv[k]
        return ea[np.where(take, ch3[k], k)]

    def _update_epsilon(self, eps0, rf, gen):
        if gen > self.Tc:
            return 0.0
        if rf < self.alpha:
            return (1 - self.tao) ** self.cp * self.epsilon
        return self.cp ** self.M * eps0 * ((1 - gen / self.Tc) ** self.cp)

    def _max_change(self, gen):
        d = 1e-6
        z0 = np.zeros(self.M)                  # generations that were never visited read as zeros (each step spends 2N)
        a, b = self.ideal.get(gen, z0), self.ideal.get(gen - self.last_gen + 1, z0)
        c, e = self.nadir.get(gen, z0), self.nadir.get(gen - self.last_gen + 1, z0)
        return float(np.max(np.concatenate([np.abs((a - b) / np.maximum(b, d)), np.abs((c - e) / np.maximum(e, d))])))

    def step(self):
        N, rng = self.N, self.rng
        pop = self.pop
        gen = int(np.ceil(self.FE / N))
        cv = overall_cv(cons(pop))
        self.CVmax = max(float(cv.max()), self.CVmax)
        rf = np.sum(cv <= 1e-6) / N
        self.ideal[gen], self.nadir[gen] = self.z.copy(), self.znad.copy()
        if gen >= self.last_gen:
            self.max_change = self._max_change(gen)
        if self.max_change > self.change_threshold and gen < 0.4 * self.G:
            self.epsilon = np.inf
        else:
            self.epsilon = self._update_epsilon(self.CVmax, rf, gen)
        parents = self._mating(pop, self.archive)
        off = self.evaluate(ga(self.problem, decs(parents), rng=rng))
        self.z = np.minimum(self.z, objs(off).min(axis=0))
        self.csa = _update_csa(self.csa, off, N, self.epsilon)
        pop, self.archive = self._env_selection(pop, off, self.archive)
        self.znad = np.maximum(self.znad, objs(pop).max(axis=0))
        self.pop = self.archive if self.FE >= self.max_FE and self.archive is not None and len(self.archive) else pop
