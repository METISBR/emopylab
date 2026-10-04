# emopylab 2026
"""TriMOEA-TA&R (multi-modal MOEA using two-archive and recombination strategies).

Reference:
Y. Liu, G. G. Yen, and D. Gong. A multi-modal multi-objective evolutionary algorithm using two-
archive and recombination strategies. IEEE Transactions on Evolutionary Computation, 2019, 23(4):
660-674.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import (LoopAlgorithm, angle_matrix, chebyshev_dist, cosine_distance, decs, ga, nd_sort,
                                            objs, tournament, uniform_point)
from core.population import Population

ALGORITHM_FLAGS = {'TriMOEATAR': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


def _update_convergence_archive(AC, pop, NC, Z, Xic, sigma, lo, up):
    pop = pop if AC is None else Population.merge(pop, AC)
    F, X = objs(pop) - Z, (decs(pop) - lo) / (up - lo)
    N = len(pop)
    rank = np.full(N, np.inf)
    nrank = 1
    fS = F.mean(axis=1)
    d = chebyshev_dist(X[:, Xic])
    choose, Q, Q1 = np.zeros(N, bool), np.ones(N, bool), np.zeros(N, bool)
    while choose.sum() < NC:
        if Q.sum() == 0:
            Q, Q1, nrank = Q1, np.zeros(N, bool), nrank + 1
        xmin = int(np.where((fS == fS[Q].min()) & Q)[0][0])
        rank[xmin], choose[xmin], Q[xmin] = nrank, True, False
        dele = (d[xmin] < sigma) & Q
        Q[dele], Q1[dele] = False, True
    return pop[choose], rank[choose], fS[choose]


def _last_selection(F, X, NQ, R, Z, Xre, sigma, lo, up):
    N, NR = len(F), len(R)
    F = F - Z
    X = (X - lo) / (up - lo)
    theta = cosine_distance(R, F)
    d = chebyshev_dist(X[:, Xre])
    label = np.argmin(theta, axis=0)
    thmin = theta[label, np.arange(N)]
    C1, C2 = np.zeros((NR, N), bool), np.zeros((NR, N), bool)
    for j in range(NR):
        member = np.where(label == j)[0]
        for i in np.argsort(thmin[member], kind="stable"):
            m = member[i]
            if np.any(d[m, C1[j] & (label == j)] < sigma):
                C2[j, m] = True
            else:
                C1[j, m] = True
    while C1.sum() > NQ:
        cnt = C1.sum(axis=1)
        jmax = cnt == cnt.max()
        cols = np.where(C1[jmax].sum(axis=0) > 0)[0]
        xmax = cols[int(np.argmax(thmin[cols]))]
        C1[label[xmax], xmax] = False
    while C1.sum() < NQ:
        c2 = C2.sum(axis=1) > 0
        if not c2.any():
            break
        cmin = C1[c2].sum(axis=1).min()
        jmin = (C1.sum(axis=1) == cmin) & c2
        cols = np.where(C2[jmin].sum(axis=0) > 0)[0]
        xmin = cols[int(np.argmin(thmin[cols]))]
        C2[label[xmin], xmin] = False
        C1[label[xmin], xmin] = True
    return C1.sum(axis=0) > 0


def _update_diversity_archive(AD, pop, ND, R, Z, Xre, sigma, lo, up):
    pop = pop if AD is None else Population.merge(pop, AD)
    F = objs(pop)
    front_no, max_f = nd_sort(F, None, ND)
    if np.sum(front_no <= max_f) == ND:
        nxt = front_no <= max_f
    else:
        nxt = front_no < max_f
        last = np.where(front_no == max_f)[0]
        ch = _last_selection(F[last], decs(pop)[last], ND - int(nxt.sum()), R, Z, Xre, sigma, lo, up)
        nxt[last[ch]] = True
    return pop[nxt], front_no[nxt]


class TriMOEATAR(LoopAlgorithm):
    UNCHARGED_EVALS = True   # the decision-variable analysis uses raw objective calls, as in the reference

    def __init__(self, pop_size: int = 100, p_con: float = 0.5, sigma_niche: float = 0.1, eps_peak: float = 0.01,
                 NR: int = 100, NCA: int = 20, NIA: int = 6, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.p_con, self.sigma, self.eps_peak, self.NR, self.NCA, self.NIA = float(p_con), float(sigma_niche), float(eps_peak), int(NR), int(NCA), int(NIA)

    def _decision_variable_analysis(self):
        rng, D, N, lo, up = self.rng, self.D, self.N, self.lower, self.upper
        Xic, Xre = np.zeros(D, bool), np.zeros(D, bool)
        for i in range(D):
            x = rng.random(D) * (up - lo) + lo
            S = np.tile(x, (self.NCA, 1))
            S[:, i] = ((np.arange(self.NCA) + rng.random(self.NCA)) / self.NCA) * (up[i] - lo[i]) + lo[i]
            F = np.unique(self.cal_obj(S), axis=0)
            _, max_f = nd_sort(F, None, np.inf)
            if max_f == len(F):
                Xic[i] = True
            else:
                Xre[i] = True
        PopDec = rng.random((N, D)) * (up - lo) + lo
        PopObj = self.cal_obj(PopDec)
        inter = np.eye(D, dtype=bool)
        for i in range(D - 1):
            for j in range(i + 1, D):
                for _ in range(self.NIA):
                    x = int(rng.integers(0, N))
                    a2 = rng.random() * (up[i] - lo[i]) + lo[i]
                    b2 = rng.random() * (up[j] - lo[j]) + lo[j]
                    dd = np.tile(PopDec[x], (3, 1))
                    dd[0, i], dd[1, j] = a2, b2
                    dd[2, [i, j]] = [a2, b2]
                    F = self.cal_obj(dd)
                    inter[i, j] = inter[i, j] or bool(np.any((F[0] - PopObj[x]) * (F[2] - F[1]) < 0))
                    inter[j, i] = inter[i, j]
        while inter[np.ix_(Xic, Xre)].sum() > 0:
            for i in np.where(Xic)[0]:
                if inter[i, Xre].sum() > 0:
                    Xic[i], Xre[i] = False, True
        return Xic, Xre

    def initial_size(self):
        self.R, _ = uniform_point(self.NR, self.M)
        return self.pop_size

    def start(self):
        self.Xic, self.Xre = self._decision_variable_analysis()
        self.NC = self.ND = self.N
        self.Pc = int(np.floor(self.p_con * self.N))
        self.Pd = self.N - self.Pc
        self.Z = objs(self.pop).min(axis=0)
        lo, up = self.lower, self.upper
        self.AC, self.RankC, self.fS = _update_convergence_archive(None, self.pop, self.NC, self.Z, self.Xic, self.sigma, lo, up)
        self.AD, self.RankD = _update_diversity_archive(None, self.pop, self.ND, self.R, self.Z, self.Xre, self.sigma, lo, up)
        self.pop = self.AD

    def step(self):
        rng, lo, up = self.rng, self.lower, self.upper
        mc = tournament(2, self.Pc, self.RankC, rng=rng)
        md = tournament(2, self.Pd, self.RankD, rng=rng)
        parents = Population.merge(self.AC[mc], self.AD[md])
        parents = parents[rng.permutation(self.N)]
        off = self.evaluate(ga(self.problem, decs(parents), rng=rng))
        self.Z = np.minimum(self.Z, objs(off).min(axis=0))
        self.AC, self.RankC, self.fS = _update_convergence_archive(self.AC, off, self.NC, self.Z, self.Xic, self.sigma, lo, up)
        self.AD, self.RankD = _update_diversity_archive(self.AD, off, self.ND, self.R, self.Z, self.Xre, self.sigma, lo, up)
        if self.FE >= self.max_FE and self.Xic.sum() > 0:
            peak = np.where(((self.fS - self.fS.min()) < self.eps_peak) & (self.RankC == 1))[0]
            if len(peak):
                ACd, ADd = decs(self.AC), decs(self.AD)
                n = len(ADd)
                FS = np.full((len(peak) * n, self.D), np.nan)
                FS[:, self.Xic] = np.repeat(ACd[peak][:, self.Xic], n, axis=0)
                FS[:, self.Xre] = np.tile(ADd[:, self.Xre], (len(peak), 1))
                self.AD = self.evaluate(FS)
        self.pop = self.AD
