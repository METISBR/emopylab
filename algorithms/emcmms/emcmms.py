# emopylab 2026
"""EMCMMS (evolutionary multitasking with a cooperative multistep mutation strategy).

Reference:
K. Qiao, K. Yu, C. Yue, B. Qu, M. Liu, and J. Liang. A cooperative multistep mutation strategy for
multiobjective optimization problems with deceptive constraints. IEEE Transactions on Systems, Man,
and Cybernetics, 2024, 54(11): 6670-6682.
"""

from __future__ import annotations

import numpy as np

from algorithms.ccmo.ccmo import cal_fitness, environmental_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga_half, objs, tournament
from algorithms.mtcmo.mtcmo import auxiliary_selection
from core.population import Population

ALGORITHM_FLAGS = {'EMCMMS': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


def _bound_constraint(vi, pop, lo, up):
    vi = vi.copy()
    pos = vi < lo
    vi[pos] = (pop[pos] + np.broadcast_to(lo, vi.shape)[pos]) / 2
    pos = vi > up
    vi[pos] = (pop[pos] + np.broadcast_to(up, vi.shape)[pos]) / 2
    return vi


class EMCMMS(LoopAlgorithm):
    """Evolutionary multitasking for constrained problems with a multi-stage search step: during the first half of
    the run, opposition-based line search between selected main-task solutions and helper-task solutions produces
    extra offspring; the auxiliary task tolerates a shrinking amount of violation."""

    def __init__(self, pop_size: int = 100, sampling=None, run_rate: float = 0.5, select_rate: float = 0.1, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.run_rate, self.select_rate = float(run_rate), float(select_rate)

    def start(self):
        N = self.N
        self.P1 = self.pop
        self.P2 = self.evaluate(self.random_decs(N))
        self.f1 = cal_fitness(objs(self.P1), cons(self.P1))
        self.f2 = cal_fitness(objs(self.P2), cons(self.P2))
        C = [c for c in (cons(self.P1), cons(self.P2)) if c.size]
        v0 = float(np.max(np.sum(np.maximum(np.vstack(C), 0), axis=1))) if C else 0.0
        self.VAR0 = v0 if v0 != 0 else 1.0
        self.X = 0.0

    def _cmms(self, pop, other, p, flag_index, direc):
        rng, N0, D = self.rng, self.N, self.D
        lo, up = self.lower, self.upper
        Nw = min(D, N0)
        pop = pop[rng.permutation(N0)]
        n = int(np.ceil(N0 * p))
        r1 = np.zeros(n, dtype=int)
        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            r1[i] = rng.integers(N0)
            while r1[i] == i:
                r1[i] = rng.integers(N0)
            r2[i] = rng.integers(N0)
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = rng.integers(N0)
            r3 = rng.integers(N0)
            while r3 == r2[i] or r3 == r1[i] or r3 == i:
                r3 = rng.integers(N0)
        rows = []
        for j in direc:
            start = decs(pop[:n])
            end = decs(other[r1]) if flag_index == 2 else decs(pop[r2])
            if j == 2:
                end = lo + up - end
            imax = np.sqrt(np.sum((up - lo) ** 2))
            rs = rng.random((n, Nw)) * imax
            for i in range(Nw):
                res = end + rs[:, [i]] * (start - end)
                rows.append(_bound_constraint(res, decs(pop[:n]), lo, up))
        Off = np.vstack(rows)
        R = len(Off)
        site = (rng.random((n, D)) < 1.0 / D).reshape(-1, order="F")
        mu = rng.random((n, D)).reshape(-1, order="F")
        flat = Off.reshape(-1, order="F").copy()
        col = np.arange(n * D) // n                           # column of the (n x D) mask layout
        Lo, Up = lo[col], up[col]
        with np.errstate(all="ignore"):
            t = np.where(site & (mu <= 0.5))[0]
            flat[t] = flat[t] + (Up[t] - Lo[t]) * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (flat[t] - Lo[t]) / (Up[t] - Lo[t])) ** 21) ** (1 / 21) - 1)
            t = np.where(site & (mu > 0.5))[0]
            flat[t] = flat[t] + (Up[t] - Lo[t]) * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - flat[t]) / (Up[t] - Lo[t])) ** 21) ** (1 / 21))
        return self.evaluate(flat.reshape(R, D, order="F"))

    def step(self):
        N, rng, pr = self.N, self.rng, self.problem
        cp = (-np.log(self.VAR0) - 6) / np.log(1 - 0.5)
        VAR = self.VAR0 * max(1 - self.X, 0.0) ** cp
        if self.FE / self.max_FE < self.run_rate:
            off3 = self._cmms(self.P1, self.P2, self.select_rate, 2, [2])
            self.P1, self.f1 = environmental_selection(Population.merge(self.P1, off3), N, True)
            self.P2, self.f2 = auxiliary_selection(Population.merge(self.P2, off3), N, VAR)
        off1 = self.evaluate(ga_half(pr, decs(self.P1[tournament(2, N, self.f1, rng=rng)]), rng=rng))
        off2 = self.evaluate(ga_half(pr, decs(self.P2[tournament(2, N, self.f2, rng=rng)]), rng=rng))
        self.P1, self.f1 = environmental_selection(Population.merge(self.P1, off1, off2), N, True)
        self.P2, self.f2 = auxiliary_selection(Population.merge(self.P2, off2, off1), N, VAR)
        self.X += 1 / (self.max_FE / N)
        self.pop = self.P1
