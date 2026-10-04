# emopylab 2026
"""DRL-SAEA (deep reinforcement learning-based expensive constrained evolutionary algorithm).

Reference:
S. Shao, Y. Tian, and Y. Zhang. Deep reinforcement learning assisted surrogate model management for
expensive constrained multi-objective optimization. Swarm and Evolutionary Computation, 2025, 92:
101817.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament
from algorithms.community_utils.nn import LMNet
from algorithms.community_utils.sparse_mask import lhs_design
from algorithms.mgsaea.mgsaea import MGSAEA, _fit, _norm_cols, update_archive, update_population
from algorithms.osp_nsde.osp_nsde import _hv
from core.population import Population

ALGORITHM_FLAGS = {'DRLSAEA': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


def _best_hv(pop, ref):
    """Hypervolume of the feasible non-dominated solutions (NaN when none is feasible)."""
    F, C = objs(pop), cons(pop)
    feas = np.all(C <= 0, 1) if C.shape[1] else np.ones(len(F), bool)
    if not feas.any():
        return np.nan
    F = F[feas]
    f, _ = nd_sort(F, None, 1)
    return _hv(F[f == 1], ref)


class DDQN:
    """Double DQN whose agent/target are Levenberg-Marquardt trained tanh networks; replay memory of ``cap`` records."""

    def __init__(self, nS, nA, rng, layers=(5, 5), cap=100):
        self.rng, self.nS, self.nA, self.cap = rng, nS, nA, cap
        self.buf = np.zeros((cap, 2 * nS + 2))
        self.index, self.overflow = 0, False
        self.agent = LMNet(rng, hidden=layers).configure(rng.random((5, nS)), -rng.random((5, nA)))
        self.target = LMNet(rng, hidden=layers).configure(rng.random((5, nS)), -rng.random((5, nA)))

    def copy(self):
        self.target.theta = self.agent.theta.copy()

    def action(self, s):
        v = self.agent.sim(np.asarray(s, float)[None])[0]
        best = np.where(v == v.max())[0]
        pool = np.concatenate([np.tile(best, self.nA), np.arange(self.nA)])
        return int(pool[self.rng.integers(0, len(pool))])

    def store(self, s, a, r, s2):
        if self.index >= self.cap:
            self.index, self.overflow = 0, True
        self.buf[self.index] = np.concatenate([s, [a, r], s2])
        self.index += 1

    def replay(self, bs, gamma=0.999):
        n = self.cap if self.overflow else self.index
        b = self.buf[self.rng.integers(0, n, bs)]
        S, A, R, S2 = b[:, : self.nS], b[:, self.nS].astype(int), b[:, self.nS + 1], b[:, self.nS + 2:]
        Q = self.agent.sim(S)
        nxt = np.argmax(self.agent.sim(S2), 1)
        tq = self.target.sim(S2)
        Q[np.arange(bs), A] = R + gamma * tq[np.arange(bs), nxt]
        self.agent.train(S, Q, epochs=1000, goal=0.0)


class DRLSAEA(MGSAEA):
    """Kriging-assisted SPEA2 (as in the staged constrained SAEA) whose constraint handling - one normalised violation model,
    one model per violated constraint, or objectives only - is chosen every iteration by a double DQN; the state is the
    archive's objective variance, objective sum, total violation and budget fraction, the reward the relative hypervolume
    gain of the feasible archive plus the relative reduction of its violation."""

    def __init__(self, pop_size: int = 100, wmax: int = 20, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, wmax=wmax, mu=mu, sampling=sampling, **kwargs)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.last = self.pop
        self.state, _, _ = self._sample(self.last, self.pop)
        self.ddqn = DDQN(4, 3, self.rng)
        self.step_n = 0

    def _st(self, A):
        F, C = objs(A), cons(A)
        return np.array([F.var(0, ddof=1).sum() if len(F) > 1 else 0.0, F.sum(), np.maximum(0, C).sum() if C.shape[1] else 0.0,
                         self.FE / self.max_FE])

    def _sample(self, last, cur):
        ref = np.vstack([objs(last), objs(cur)]).max(0)
        h0, h1 = _best_hv(last, ref), _best_hv(cur, ref)
        with np.errstate(all="ignore"):
            r1 = (h1 - h0) / h0
        r1 = 0.0 if np.isnan(r1) else r1
        s0, s1 = self._st(last), self._st(cur)
        if s0[2] == 0:
            r2 = 0.0
        elif s1[2] < s0[2]:
            r2 = abs((s1[2] - s0[2]) / s0[2])
        else:
            r2 = -abs((s1[2] - s0[2]) / s0[2])
        return s0, s1, r1 + r2

    def step(self):
        rng, NI, M = self.rng, self.NI, self.M
        a = self.ddqn.action(self.state)
        self.step_n += 1
        P = self.P
        X, C = decs(P), cons(P)
        if a == 0:
            status = 1
            cv = _norm_cols(np.maximum(0, C)).sum(1) if C.shape[1] else np.zeros(len(P))
            th = np.vstack([self.th_obj, self.th_cv])
            models = self._models(X, np.column_stack([objs(P), cv]), th)
            self.th_obj, self.th_cv = th[:M], th[-1]
            fit = _fit(objs(P), C)
        elif a == 1:
            status = 2
            mc = np.maximum(0, C).max(0) if C.shape[1] else np.zeros(0)
            idx = np.where(mc > 0)[0]
            Cn = _norm_cols(np.maximum(0, C)[:, idx]) if len(idx) else np.zeros((len(P), 0))
            th = np.vstack([self.th_obj, self.th_con[idx]]) if len(idx) else self.th_obj.copy()
            models = self._models(X, np.column_stack([objs(P), Cn]), th)
            self.th_obj = th[:M]
            if len(idx):
                self.th_con[idx] = th[M:]
            fit = _fit(np.column_stack([objs(P), np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(len(P))]))
        else:
            status = 3
            models = self._models(X, objs(P), self.th_obj)
            fit = _fit(objs(P))
        for _ in range(self.wmax):
            mate = tournament(2, NI, fit, rng=rng)
            X = np.vstack([X, ga(self.problem, X[mate], rng=rng)])
            Y = np.column_stack([m.predict(X) for m in models])
            X, Y, fit = self._env(X, Y, NI, status)
        nd, _, _ = self._env(X, Y, self.mu, status)
        new = self.evaluate(nd)
        self.P = update_population(P, new, NI - self.mu, status)
        self.pop = update_archive(Population.merge(self.pop, new), self.N)
        s0, s1, r = self._sample(self.last, self.pop)
        self.ddqn.store(s0, a, r, s1)
        self.state = s1
        self.last = self.pop
        if self.step_n % 8 == 0:
            self.ddqn.replay(16)
        if self.step_n % 11 == 0:
            self.ddqn.copy()
