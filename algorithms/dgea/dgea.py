# emopylab 2026
"""DGEA (direction guided evolutionary algorithm).

Reference:
C. He, R. Cheng, and D. Yazdani. Adaptive offspring generation for evolutionary large-scale
multiobjective optimization. IEEE Transactions on System, Man, and Cybernetics: Systems, 2022,
52(2): 786-798.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, nd_sort, objs, polynomial_mutation, uniform_point
from algorithms.community_utils.spea import cal_fitness, overall_cv, truncation
from core.population import Population

ALGORITHM_FLAGS = {'DGEA': {'integer', 'large', 'many', 'multi', 'real'}}


def _angle_matrix(A, B):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :])
    return np.arccos(np.clip(c, -1.0, 1.0))


def _gamma(V):
    ang = _angle_matrix(V, V)
    np.fill_diagonal(ang, np.arccos(0.0))              # diagonal cosine is set to 0 in the reference
    return ang.min(axis=1)


def _argmin_nan(A):
    return np.argmin(np.where(np.isnan(A), np.inf, A), axis=1)


def _pre_selection(pop, V, theta, ref_no):
    NV = len(V)
    F = objs(pop)
    front, maxf = nd_sort(F, None, min(NV, len(pop)))
    if np.sum(front == 1) >= ref_no:
        next0 = front < maxf
        last = np.where(front == maxf)[0]
    else:
        next0 = front == 1
        last = np.where(front > 1)[0]
    P = F[last] - F.min(axis=0)
    M = P.shape[1]
    gamma = _gamma(V)
    ang = _angle_matrix(P, V)
    assoc = _argmin_nan(ang) if len(P) else np.zeros(0, int)
    chosen = []
    for i in np.unique(assoc):
        cur = np.where(assoc == i)[0]
        apd = (1 + M * theta * ang[cur, i] / gamma[i]) * np.sqrt(np.sum(P[cur] ** 2, axis=1))
        chosen.append(cur[int(np.argmin(apd))])
    next0[last[np.array(chosen, dtype=int)]] = True
    return pop[next0], front[next0]


def _sub_rvea(pop, V, theta):
    F = objs(pop)
    M = F.shape[1]
    P = F - F.min(axis=0)
    cv = overall_cv(cons(pop))
    gamma = _gamma(V)
    ang = _angle_matrix(P, V)
    assoc = _argmin_nan(ang)
    nxt = np.zeros(len(V), dtype=int) - 1
    for i in np.unique(assoc):
        c1 = np.where((assoc == i) & (cv == 0))[0]
        c2 = np.where((assoc == i) & (cv != 0))[0]
        if len(c1):
            apd = (1 + M * theta * ang[c1, i] / gamma[i]) * np.sqrt(np.sum(P[c1] ** 2, axis=1))
            nxt[i] = c1[int(np.argmin(apd))]
        elif len(c2):
            nxt[i] = c2[int(np.argmin(cv[c2]))]
    return pop[nxt[nxt >= 0]]


def _sub_nsga2(pop, N):
    CC = min(N, len(pop))
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.size else None, CC)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: CC - int(nxt.sum())]]] = True
    return pop[nxt]


def _sub_ibea(pop, N, kappa):
    F = objs(pop)
    Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    I = np.max(Fn[:, None, :] - Fn[None, :, :], axis=2)
    C = np.max(np.abs(I), axis=0)
    with np.errstate(all="ignore"):
        fit = np.sum(-np.exp(-I / C[None, :] / kappa), axis=0) + 1
        nxt = list(range(len(pop)))
        while len(nxt) > N:
            x = int(np.argmin(fit[nxt]))
            fit = fit + np.exp(-I[nxt[x], :] / C[nxt[x]] / kappa)
            del nxt[x]
    return pop[np.array(nxt)]


def _sub_spea2(pop, N):
    F = objs(pop)
    CC = min(len(pop), N)
    fit = cal_fitness(F)
    nxt = fit < 1
    if nxt.sum() < CC:
        nxt[np.argsort(fit, kind="stable")[:CC]] = True
    elif nxt.sum() > CC:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - CC)]] = False
    return pop[nxt]


class DGEA(LoopAlgorithm):
    """Directions from one non-dominated solution to dominated (or other non-dominated) ones guide the sampling of
    offspring (Gaussian steps along those directions); a small pre-selection keeps the parents and an archive keeps the
    result with the selection operator chosen by ``operation`` (1 RVEA, 2 NSGA-II, 3 IBEA, 4 SPEA2, else none)."""

    def __init__(self, pop_size: int = 100, operation: int = 1, ref_no: int = 10, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.operation, self.ref_no = int(operation), int(ref_no)

    def initial_size(self):
        self.V, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.offspring = self.evaluate(self.random_decs(self.N))
        self.arc = self.pop
        self.parents = self.pop

    def _direction_reproduction(self, pop, front, ref_no):
        rng, D, Ntot = self.rng, self.D, self.N
        n = len(pop)
        X = decs(pop)
        idxD = np.where(front > 1)[0]
        domiN = len(idxD)
        nondec = X[front == 1]
        domidec = X[idxD]
        sub_n = max(int(np.floor(Ntot / ref_no)), 10)
        lo, up = self.lower, self.upper
        start = nondec[rng.integers(0, n - domiN)]
        if domiN < ref_no:
            k = (n - domiN) if n <= ref_no else (ref_no - domiN)
            end = np.vstack([domidec, nondec[rng.permutation(n - domiN)[:k]]])
        else:
            end = domidec[rng.permutation(domiN)[:ref_no]]
        ref_no = len(end)
        vec = end - start
        with np.errstate(invalid="ignore", divide="ignore"):
            direct = vec / np.sqrt(np.sum(vec ** 2, axis=1))[:, None]
        chunks = []
        for i in range(ref_no):
            lam = (nondec - start) @ direct[i]
            sigma = (np.std(lam, ddof=1) if len(lam) > 1 else 0.0) * (1 + (domiN + 1) / n)
            off = rng.normal(0.0, sigma, (sub_n, 1)) * direct[i] + start
            chunks.append(np.fmax(np.fmin(off, up), lo))
        off = np.vstack(chunks)
        off = polynomial_mutation(off, lo, up, rng, 20.0, prob=1.0 / D)
        return self.evaluate(off)

    def step(self):
        theta = (self.FE / self.max_FE) ** 2
        pop, front = _pre_selection(Population.merge(self.parents, self.offspring), self.V, theta, self.ref_no)
        self.parents = pop
        self.offspring = self._direction_reproduction(pop, front, self.ref_no)
        joined = Population.merge(self.arc, self.offspring)
        op = self.operation
        if op == 1:
            self.arc = _sub_rvea(joined, self.V, (self.FE / self.max_FE) ** 2)
        elif op == 2:
            self.arc = _sub_nsga2(joined, self.N)
        elif op == 3:
            self.arc = _sub_ibea(joined, self.N, 0.05)
        elif op == 4:
            self.arc = _sub_spea2(joined, self.N)
        else:
            self.arc = Population.merge(pop, self.offspring)
        self.pop = self.arc
