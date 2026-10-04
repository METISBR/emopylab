# emopylab 2026
"""K-RVEA (surrogate-assisted RVEA).

Reference:
T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya. A surrogate- assisted reference vector
guided evolutionary algorithm for computationally expensive many-objective optimization. IEEE
Transactions on Evolutionary Computation, 2018, 22(1): 129-142.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, kmeans, objs, uniform_point
from algorithms.community_utils.dace import DaceModel
from core.population import Population

ALGORITHM_FLAGS = {'KRVEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


def _angles(P, V):
    with np.errstate(all="ignore"):
        cos = (P @ V.T) / (np.linalg.norm(P, axis=1)[:, None] * np.linalg.norm(V, axis=1)[None, :])
    return np.arccos(np.clip(cos, -1, 1))


def _argmin_rows(A):
    return np.argmin(np.where(np.isnan(A), np.inf, A), axis=1)


def _gamma(V):
    c = np.cos(_angles(V, V))
    np.fill_diagonal(c, 0.0)
    return np.arccos(c).min(axis=1)


def _no_active(F, V):
    F = F - F.min(axis=0)
    assoc = _argmin_rows(_angles(F, V))
    active = np.unique(assoc)
    return len(V) - len(active), active


def _env_selection(F, V, theta):
    N, M = F.shape
    F = F - F.min(axis=0)
    ang = _angles(F, V)
    assoc = _argmin_rows(ang)
    gamma = _gamma(V)
    nxt = []
    for i in np.unique(assoc):
        cur = np.where(assoc == i)[0]
        apd = (1 + M * theta * ang[cur, i] / gamma[i]) * np.sqrt(np.sum(F[cur] ** 2, axis=1))
        nxt.append(cur[int(np.argmin(apd))])
    return np.array(nxt, dtype=int)


def _kriging_select(rng, PopDec, F, MSE, V, V0, num_v1, delta, mu, theta):
    num_v2, _ = _no_active(F, V0)
    nva, va = _no_active(F, V)
    ncluster = int(min(mu, len(V) - nva))
    Va = V[va]
    idx = kmeans(Va, ncluster, rng)
    F = F - F.min(axis=0)
    gamma = _gamma(Va)
    ang = _angles(F, Va)
    assoc = _argmin_rows(ang)
    M = F.shape[1]
    apd_s = np.ones(len(F))
    for i in np.unique(assoc):
        cur = np.where(assoc == i)[0]
        apd_s[cur] = (1 + M * theta * ang[cur, i] / gamma[i]) * np.sqrt(np.sum(F[cur] ** 2, axis=1))
    cidx = idx[assoc]
    flag = num_v2 - num_v1
    nxt = []
    for i in np.unique(cidx):
        cur = np.where(cidx == i)[0]
        if flag <= delta:
            best_sol = []
            for t in np.unique(assoc[cur]):
                cs = np.where(assoc == t)[0]
                best_sol.append(cs[int(np.argmin(apd_s[cs]))])
            best_sol = np.array(best_sol)
            nxt.append(best_sol[int(np.argmin(apd_s[best_sol]))])
        else:
            nxt.append(cur[int(np.argmax(np.mean(MSE[cur], axis=1)))])
    return PopDec[np.array(nxt, dtype=int)]


def _update_archive(rng, A1, New, V, mu, NI):
    all_dec = np.vstack([decs(A1), decs(New)])
    index = np.unique(all_dec, axis=0, return_index=True)[1]
    total = Population.merge(A1, New)[index]
    if len(total) > NI:
        _, active = _no_active(objs(New), V)
        Vi = V[np.setdiff1d(np.arange(len(V)), active)]
        newset = {tuple(r) for r in decs(New)}
        keep = np.array([tuple(r) not in newset for r in decs(total)], dtype=bool)
        total = total[keep]
        F = objs(total)
        F = F - F.min(axis=0)
        assoc = _argmin_rows(_angles(F, Vi))
        Via = Vi[np.unique(assoc)]
        k = NI - mu
        nxt = []
        if len(Via) > k:
            idx = kmeans(Via, k, rng)
        else:
            idx = kmeans(objs(total), k, rng)
        for i in np.unique(idx):
            cur = np.where(idx == i)[0]
            nxt.append(cur[int(rng.integers(0, len(cur)))] if len(cur) > 1 else cur[0])
        return Population.merge(total[np.array(nxt, dtype=int)], New)
    return total


class KRVEA(LoopAlgorithm):
    """Reference-vector guided search on Kriging surrogates: every generation ``wmax`` surrogate-only RVEA steps produce
    candidates; up to ``mu`` of them are evaluated for real, chosen by angle-penalised distance while the reference vectors
    are still adapting and by model uncertainty once they have settled; a bounded archive trains the models."""

    def __init__(self, pop_size: int = 100, alpha: float = 2, wmax: int = 20, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.wmax, self.mu = float(alpha), int(wmax), int(mu)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.V0, self.pop_size = uniform_point(self.pop_size, self.M)
        self.V = self.V0.copy()
        NI = 11 * self.D - 1
        P, _ = UniformPoint(NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills          # A2: every solution evaluated so far
        self.A1 = infills
        self.theta = 5.0 * np.ones((self.M, self.D))
        self._set_optimum()

    def step(self):
        rng, D, M, N = self.rng, self.D, self.M, self.N
        A1Dec, A1Obj = decs(self.A1), objs(self.A1)
        models = []
        for i in range(M):
            dm = DaceModel(A1Dec, A1Obj[:, i], "regpoly1", self.theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            models.append(dm)
            self.theta[i] = dm.theta
        pop_dec = A1Dec
        w = 1
        while w <= self.wmax:
            pop_dec = np.vstack([pop_dec, ga(self.problem, pop_dec, rng=rng)])
            preds = [m.predict(pop_dec, mse=True) for m in models]
            F = np.column_stack([p[0] for p in preds])
            MSE = np.column_stack([p[1] for p in preds])
            index = _env_selection(F, self.V, (w / self.wmax) ** self.alpha)
            pop_dec, F = pop_dec[index], F[index]
            if w % int(np.ceil(self.wmax * 0.1)) == 0:
                self.V = self.V0 * (F.max(axis=0) - F.min(axis=0))
            w += 1
        num_vf, _ = _no_active(A1Obj, self.V0)
        new_dec = _kriging_select(rng, pop_dec, F, MSE[index], self.V, self.V0, num_vf, 0.05 * N, self.mu, (w / self.wmax) ** self.alpha)
        new = self.evaluate(new_dec)
        self.pop = Population.merge(self.pop, new)
        self.A1 = _update_archive(rng, self.A1, new, self.V, self.mu, 11 * D - 1)
