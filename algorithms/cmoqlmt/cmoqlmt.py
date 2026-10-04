# emopylab 2026
"""CMOQLMT (constrained multi-objective optimization based on Q-learning and multitasking).

Reference:
F. Ming, W. Gong, and L. Gao. Adaptive auxiliary task selection for multitasking-assisted constrained multi-objective optimization [feature]. IEEE Computational Intelligence Magazine, 2023, 18(2): 18-30.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, cons, de, decs, ga_half, objs, tournament, truncate_lexi, uniform_point
from algorithms.icma.icma import icma_update
from algorithms.imtcmo.imtcmo import _neighbour_pairing
from core.population import Population

ALGORITHM_FLAGS = {'CMOQLMT': {'constrained', 'integer', 'multi', 'real'}}


def _kth_nn(F, k):
    Dm = np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1))
    np.fill_diagonal(Dm, np.inf)
    return np.sort(Dm, 1)[:, k - 1]


def spea_selection(pop, N, use_cons):
    """SPEA2 selection; returns (survivors sorted by fitness, their fitness, survival mask in input order)."""
    F, C = objs(pop), cons(pop)
    fit = spea.cal_fitness(F, C if (use_cons and C.shape[1]) else None)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[spea.truncation(F[idx], int(nxt.sum()) - N)]] = False
    idx = np.where(nxt)[0]
    o = idx[np.argsort(fit[idx], kind="stable")]
    return pop[o], fit[o], nxt


def selection_t2(pop, N, alpha, gamma, para):
    """Ranking by a blend of a convergence rank (strength of a region-wise adaptively penalised objective vector) and a
    diversity rank (weighted-sum order inside every reference region), both tie-broken by the k-th neighbour distance."""
    F, C = objs(pop), cons(pop)
    n, M = F.shape
    z = F.min(0)
    W, _ = uniform_point(N, M)
    with np.errstate(all="ignore"):
        cosd = 1 - ((F - z) @ W.T) / (np.linalg.norm(F - z, axis=1)[:, None] * np.linalg.norm(W, axis=1)[None])
    region = np.argmin(np.where(np.isnan(cosd), np.inf, cosd), 1)
    cv = np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(n)
    infeas_all = np.any(C > 0, 1) if C.shape[1] else np.zeros(n, bool)
    phi_max = cv[infeas_all].max() if infeas_all.any() else np.nan
    F2 = F.copy()
    for i in range(len(W)):
        idx = np.where(region == i)[0]
        if len(idx):
            inf = infeas_all[idx]
            if inf.any():
                fmax = F[idx].max(0)
                with np.errstate(all="ignore"):
                    w = (cv[idx][inf] / phi_max) ** (np.exp(para) / max(gamma, 1e-6))
                F2[idx[inf]] = F[idx][inf] + w[:, None] * (fmax - F[idx][inf])
    lt = (F2[:, None] < F2[None]).any(-1)
    gt = (F2[:, None] > F2[None]).any(-1)
    dom = lt & ~gt
    R = dom.sum(1) @ dom
    front = R + 1
    cd = _kth_nn(F, int(np.floor(np.sqrt(n))))
    mid = np.zeros(n)
    n1 = front == 1
    if n1.sum() <= N:
        ic = np.lexsort((-cd, front))
    else:
        t = np.where(n1)[0]
        Dm = np.sqrt(((F[t][:, None] - F[t][None]) ** 2).sum(-1))
        np.fill_diagonal(Dm, np.inf)
        mid[t[truncate_lexi(Dm, int(n1.sum()) - N)]] = 1
        ic = np.lexsort((-cd, mid, front))
    rc = np.empty(n)
    rc[ic] = np.arange(1, n + 1)
    fd = np.ones(n)
    for i in range(len(W)):
        idx = np.where(region == i)[0]
        if len(idx):
            g = ((F2[idx] - z) * W[i]).sum(1)
            fd[idx[np.argsort(g, kind="stable")]] = np.arange(1, len(idx) + 1)
    idv = np.lexsort((-cd, fd))
    rd = np.empty(n)
    rd[idv] = np.arange(1, n + 1)
    rs = alpha * rc + (1 - alpha) * rd
    r = np.argsort(rs, kind="stable")[:N]
    return pop[r], rs[r]


class CMOQLMT(LoopAlgorithm):
    """Four populations: the constrained main task (SPEA2 with constrained dominance), an adaptively penalised helper
    (convergence/diversity rank blend), an unconstrained helper, and an indicator-based constraint-handling population
    evolved by DE between objective-direction neighbours. A Q-table (states = actions = helpers) learns which helper's
    offspring survive best in the main task; in the second half of the budget only the chosen helper is evolved."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        rng, N = self.rng, self.N
        self.P = [infills] + [self.evaluate(self.random_decs(N)) for _ in range(3)]
        F4, C4 = objs(self.P[3]), cons(self.P[3])
        self.zmin = F4.min(0)
        feas = np.all(C4 <= 0, 1) if C4.shape[1] else np.ones(len(F4), bool)
        self.fmin = F4[feas].min(0) if feas.any() else np.full(self.M, np.inf)
        self.fit = [spea.cal_fitness(objs(self.P[0]), cons(self.P[0]) if cons(self.P[0]).shape[1] else None), None,
                    spea.cal_fitness(objs(self.P[2]))]
        self.W, _ = uniform_point(N, self.M)
        self.Ra = 1.0
        self.Q = np.zeros((3, 3))
        self.state = int(rng.integers(0, 3))
        self.alpha1 = 2 / (1 + np.exp(-self.FE * 10 / self.max_FE)) - 1
        self.para = np.ceil(self.max_FE / N) / 2 - np.ceil(self.FE / N)
        self.gamma = 1.0
        self.offs = [None] * 4
        self.pop = self.P[0]
        self._set_optimum()

    def _choose(self):
        rng, Q, s = self.rng, self.Q, self.state
        if rng.random() > 0.9 or (Q[s, 0] == Q[s, 1] == Q[s, 2]):
            return int(rng.integers(0, 3))
        return int(np.argmax(Q[s]))

    def _half(self, k, fit):
        mate = tournament(2, 2 * (self.N // 2), fit, rng=self.rng)
        return self.evaluate(ga_half(self.problem, decs(self.P[k])[mate], rng=self.rng))

    def _refresh_t2(self):
        rng, N = self.rng, self.N
        self.P[0] = self.P[0][rng.permutation(N)]
        self.P[1] = self.P[1][rng.permutation(N)]
        F1, F2 = objs(self.P[0]), objs(self.P[1])
        lia = np.array([np.any(np.all(F1 == f, 1)) for f in F2])
        self.gamma = 1 - lia.sum() / N
        self.P[1], self.fit[1] = selection_t2(self.P[1], N, self.alpha1, self.gamma, self.para)

    def _t4_offspring(self):
        rng, N = self.rng, self.N
        nt = int(np.floor(self.Ra * N))
        if len(self.P[0]) > N - nt:
            pool = Population.merge(self.P[3][rng.permutation(N)[:nt]], self.P[0][rng.permutation(N)[: N - nt]])
        else:
            pool = self.P[3][rng.permutation(N)]
        m1, m2, m3 = _neighbour_pairing(rng, pool, self.zmin)
        params = None if rng.random() > 0.5 else (0.5, 0.5, 0.5, 0.75)
        return self.evaluate(de(self.problem, decs(m1), decs(m2), decs(m3), params, rng=rng))

    def _learn(self, a, off1):
        pool = Population.merge(self.P[0], off1, self.P[a + 1], self.offs[a + 1])
        _, _, nxt = spea_selection(pool, self.N, True)
        start = len(self.P[0]) + len(off1)
        succ = nxt[start:].sum() / (len(self.P[a + 1]) + len(self.offs[a + 1]))
        s = self.state
        self.Q[s, a] += 0.8 * (succ + 0.9 * self.Q[a].max() - self.Q[s, a])
        self.state = a

    def _update_t4(self):
        o = self.offs[3]
        F, C = objs(o), cons(o)
        feas = np.all(C <= 0, 1) if C.shape[1] else np.ones(len(F), bool)
        if feas.any():
            self.fmin = np.minimum(self.fmin, F[feas].min(0))
        self.zmin = np.minimum(self.zmin, F.min(0))

    def _update_main(self, off1):
        a = self.state
        self.P[0], self.fit[0], _ = spea_selection(Population.merge(self.P[0], off1, self.P[a + 1], self.offs[a + 1]), self.N, True)

    def step(self):
        N = self.N
        a = self._choose()
        if self.FE < 0.5 * self.max_FE:
            self._refresh_t2()
            off1 = self._half(0, self.fit[0])
            self.offs[1] = self._half(1, self.fit[1])
            self.offs[2] = self._half(2, self.fit[2])
            self.offs[3] = self._t4_offspring()
            self._learn(a, off1)
            self.alpha1 = 2 / (1 + np.exp(-self.FE * 10 / self.max_FE)) - 1
            self.para = np.ceil(self.max_FE / N) / 2 - np.ceil(self.FE / N)
            self._update_t4()
            self._update_main(off1)
            self.P[1], _ = selection_t2(Population.merge(self.P[1], self.offs[1]), N, self.alpha1, self.gamma, self.para)
            self.P[2], self.fit[2], _ = spea_selection(Population.merge(self.P[2], self.offs[2]), N, False)
            self.P[3], _ = icma_update(Population.merge(self.P[3], self.offs[3]), N, self.W, self.zmin, self.fmin)
        else:
            off1 = self._half(0, self.fit[0])
            if a == 0:
                self._refresh_t2()
                self.offs[1] = self._half(1, self.fit[1])
                self._learn(a, off1)
                self.alpha1 = 2 / (1 + np.exp(-self.FE * 10 / self.max_FE)) - 1
                self.para = np.ceil(self.max_FE / N) / 2 - np.ceil(self.FE / N)
                self._update_main(off1)
                self.P[1], _ = selection_t2(Population.merge(self.P[1], self.offs[1]), N, self.alpha1, self.gamma, self.para)
            elif a == 1:
                self.offs[2] = self._half(2, self.fit[2])
                self._learn(a, off1)
                self._update_main(off1)
                self.P[2], self.fit[2], _ = spea_selection(Population.merge(self.P[2], self.offs[2]), N, False)
            else:
                self.offs[3] = self._t4_offspring()
                self._learn(a, off1)
                self._update_t4()
                self._update_main(off1)
                self.P[3], _ = icma_update(Population.merge(self.P[3], self.offs[3]), N, self.W, self.zmin, self.fmin)
        self.Ra = 1 - self.FE / self.max_FE
        self.pop = self.P[0]
