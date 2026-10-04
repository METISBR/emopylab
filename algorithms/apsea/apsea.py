# emopylab 2026
"""APSEA (adaptive population sizing based evolutionary algorithm).

Reference:
Y. Tian, R. Wang, Y. Zhang, and X. Zhang. Adaptive population sizing for multi-population based
constrained multi-objective optimization. Neurocomputing, 2025: 129296.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, objs, tournament
from algorithms.community_utils.spea import cal_fitness, overall_cv, select, truncation
from core.population import Population

ALGORITHM_FLAGS = {'APSEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _reduce_boundary(eF, k, max_k, cp):
    z, near = 1e-8, 1e-15
    B = max_k / np.power(np.log((eF + z) / z), 1.0 / cp)
    if B == 0:
        B += near
    f = eF * np.exp(-((k / B) ** cp))
    if abs(f - z) < near:
        f = z
    e = f - z
    return e if e > 0 else 0.0


def _fill(pop, fit, N, F):
    """Non-dominated first; fewer than ``N`` -> best ranks; more -> nearest-neighbour truncation (mask over ``pop``)."""
    nxt = fit < 1
    if nxt.sum() <= N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    else:
        idx = np.where(nxt)[0]
        nxt[idx[truncation(F[nxt], int(nxt.sum()) - N)]] = False
    return nxt


def _epsilon_selection(pop, N, var):
    """Keep every solution within violation ``var`` ranked with the violation as an extra objective, and fill the rest of
    the ``N`` places with the least violating ones; both groups are ordered by fitness."""
    cv = overall_cv(cons(pop))
    f_pop, i_pop = pop[cv <= var], pop[cv > var]
    parts, fits = [], []
    if len(f_pop) == 0:
        F = objs(i_pop)
        fit = cal_fitness(F, cons(i_pop))
        nxt = _fill(i_pop, fit, N, F)
        p, fi = i_pop[nxt], fit[nxt]
        o = np.argsort(fi, kind="stable")
        return p[o], fi[o]
    Ff = np.hstack([objs(f_pop), overall_cv(cons(f_pop))[:, None]])
    if len(f_pop) <= N:
        ffit = cal_fitness(Ff)
        o = np.argsort(ffit, kind="stable")
        parts, fits = [f_pop[o]], [ffit[o]]
        rest = N - len(f_pop)
        if len(i_pop):
            F = objs(i_pop)
            ifit = cal_fitness(F, cons(i_pop))
            nxt = _fill(i_pop, ifit, rest, F)
            p, fi = i_pop[nxt], ifit[nxt] + ffit.max()
            o = np.argsort(fi, kind="stable")
            parts.append(p[o]), fits.append(fi[o])
    else:
        ffit = cal_fitness(Ff)
        nxt = _fill(f_pop, ffit, N, objs(f_pop))
        p, fi = f_pop[nxt], ffit[nxt]
        o = np.argsort(fi, kind="stable")
        parts, fits = [p[o]], [fi[o]]
    return Population.merge(*parts), np.concatenate(fits)


class APSEA(LoopAlgorithm):
    """Two populations: the main one handles constraints by feasibility rules, an auxiliary one of adaptive size
    ignores them and, once the main population stagnates, follows an epsilon-relaxation that shrinks over the run."""

    def __init__(self, pop_size: int = 100, alpha: float = 0.05, beta: float = 0.05, cp: float = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.beta, self.cp = float(alpha), float(beta), float(cp)

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.pop_size))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop1 = infills
        self.pop2 = self.evaluate(self.random_decs(self.pop_size))
        self.fit1 = cal_fitness(objs(self.pop1), cons(self.pop1))
        self.fit2 = cal_fitness(objs(self.pop2))
        self.last_gen = 20
        self.ideal, self.nadir = {}, {}
        cv = overall_cv(cons(self.pop2))
        self.eps0 = float(cv.max()) if len(cv) else 0.0
        if self.eps0 == 0:
            self.eps0 = 1.0
        self.gen, self.max_change = 1, 0.0
        self.pop = self.pop1
        self._set_optimum()

    def _calc_maxchange(self, gen):
        d = 1e-6
        a, b = self.ideal[gen], self.ideal[gen - self.last_gen + 1]
        c, e = self.nadir[gen], self.nadir[gen - self.last_gen + 1]
        return float(np.max(np.concatenate([np.abs((a - b) / np.maximum(b, d)), np.abs((c - e) / np.maximum(e, d))])))

    def step(self):
        rng, N = self.rng, self.N
        gen = self.gen
        P1 = self.pop1
        FR = np.sum(overall_cv(cons(P1)) == 0) / len(P1)
        self.ideal[gen], self.nadir[gen] = objs(P1).min(axis=0), objs(P1).max(axis=0)
        if gen > self.last_gen:
            self.max_change = self._calc_maxchange(gen)
        Np = max(int(np.ceil(N / 2 * (1 - np.log2(1 + FR)) + N / 2 * (1 - np.log2(1 + self.FE / self.max_FE)))), 2)
        mp1 = tournament(2, N, self.fit1, rng=rng)
        off1 = self.evaluate(ga(self.problem, decs(P1[mp1]), rng=rng))
        if FR <= self.alpha or gen <= self.last_gen or self.max_change > self.beta:
            mp2 = tournament(2, Np, self.fit2, rng=rng)
            off2 = self.evaluate(ga(self.problem, decs(self.pop2[mp2]), rng=rng))
            self.pop1, self.fit1 = select(Population.merge(P1, off1, off2), N, True)
            if FR <= self.alpha or gen <= self.last_gen:
                self.pop2, self.fit2 = select(Population.merge(self.pop2, off1, off2), Np, False)
            else:
                eps = _reduce_boundary(self.eps0, gen, np.ceil(self.max_FE / N) - 1, self.cp)
                self.pop2, self.fit2 = _epsilon_selection(Population.merge(self.pop2, off1, off2), Np, eps)
        else:
            self.pop1, self.fit1 = select(Population.merge(P1, off1), N, True)
        self.gen += 1
        self.pop = self.pop1
