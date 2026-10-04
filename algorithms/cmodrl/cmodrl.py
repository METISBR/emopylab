# emopylab 2026
"""CMODRL (constrained multiobjective optimization via deep reinforcement learning).

Reference:
F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. Automated configuration of evolutionary algorithms
via deep reinforcement learning for constrained multiobjective optimization. IEEE Transactions on
Cybernetics, 2025, 55(12): 5877-5890.
"""

from __future__ import annotations

import numpy as np

from algorithms.cmoqlmt.cmoqlmt import spea_selection
from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, cons, de, decs, ga, objs, tournament
from algorithms.community_utils.dqn import TanhMLP
from algorithms.community_utils.nn import DAE
from core.population import Population

ALGORITHM_FLAGS = {'CMODRL': {'constrained', 'integer', 'multi', 'real'}}


def _cv(C):
    return np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(len(C))


def fe_selection(pop, N, fe):
    """SPEA2 selection in which violations not above ``fe`` count as feasible."""
    F, C = objs(pop), cons(pop)
    cv = _cv(C)
    cv = np.where(cv <= fe, 0.0, cv)
    k = (F[:, None] < F[None]).any(-1).astype(int) - (F[:, None] > F[None]).any(-1).astype(int)
    dom = (cv[:, None] < cv[None]) | ((cv[:, None] == cv[None]) & (k == 1))
    S = dom.sum(1)
    R = S @ dom
    Dm = np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1))
    np.fill_diagonal(Dm, np.inf)
    fit = R + 1 / (np.sort(Dm, 1)[:, max(int(np.floor(np.sqrt(len(F)))) - 1, 0)] + 2)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[spea.truncation(F[idx], int(nxt.sum()) - N)]] = False
    return pop[nxt], fit[nxt]


class CMODRL(LoopAlgorithm):
    """Constrained SPEA2 with a relaxed-feasibility main population, an unconstrained helper and an archive. After 100
    generations the relaxation threshold and the operator (DE or GA) are set every ``reward_step`` generations: first at
    random while transitions (DAE-encoded population state, action, reward, next state) are collected, then (after 600
    generations) by an actor-critic network for the threshold and a Q-network for the operator, refreshed every 200
    generations."""

    def __init__(self, pop_size: int = 100, reward_step: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.reward_step = int(reward_step)

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        N = self.N
        self.P1 = infills
        self.h = int(np.ceil(N / 2))
        self.P2 = self.evaluate(self.random_decs(self.h))
        self.A, _, _ = spea_selection(Population.merge(self.P1, self.P2), N, True)
        self.init_cv = _cv(cons(self.P1)).sum() / N
        self.Fe = self.rng.random() / 5 * self.init_cv
        self.fit1 = spea.cal_fitness(objs(self.P1), cons(self.P1) if cons(self.P1).shape[1] else None)
        self.fit2 = spea.cal_fitness(objs(self.P2))
        self.g, self.step_, self.acc = 0, 0, 0.0
        self.data, self.op_data, self.pop_data = [], [], np.zeros((0, self.M))
        self.built = False
        self.action = 1
        self.pop = self.P1
        self._set_optimum()

    def _arc_state(self):
        F, C = objs(self.A), cons(self.A)
        return F.sum() / self.N, _cv(C).sum() / len(self.A), (F.max(0) - F.min(0)).sum()

    def _offspring(self, op):
        rng, N = self.rng, self.N
        X1, X2 = decs(self.P1), decs(self.P2)
        if op == 1:
            m1 = tournament(2, 2 * N, self.fit1, rng=rng)
            m2 = tournament(2, len(X2) * 2, self.fit2, rng=rng)
            o1 = de(self.problem, X1, X1[m1[:N]], X1[m1[N:]], rng=rng)
            o2 = de(self.problem, X2, X2[m2[: len(X2)]], X2[m2[len(X2):]], rng=rng)
        else:
            o1 = ga(self.problem, X1[tournament(2, N, self.fit1, rng=rng)], rng=rng)
            o2 = ga(self.problem, X2[tournament(2, len(X2), self.fit2, rng=rng)], rng=rng)
        return self.evaluate(o1), self.evaluate(o2)

    def _update(self, o1, o2):
        N = self.N
        self.A, _, nxt = spea_selection(Population.merge(self.A, o1, o2), N, True)
        self.acc += nxt[len(self.A): len(self.A) + len(o1)].sum() / len(self.A)
        self.P1, self.fit1 = fe_selection(Population.merge(self.P1, o1, o2), N, self.Fe)
        self.P2, self.fit2, _ = spea_selection(Population.merge(self.P2, o1, o2), self.h, False)

    def _random_fe(self):
        avg = _cv(cons(self.P1)).sum() / self.N
        fe = self.rng.random() / 5 * avg
        if fe <= 0:
            fe = self.rng.random() / 5 * self.init_cv * (1 - self.g / np.ceil(self.max_FE / 2 * self.N))
        return fe

    def _state(self):
        return self.dae.reduce(objs(self.P1))[:, 0]

    def _begin_block(self):
        self.acc = 0.0
        self.r0 = self._arc_state()
        self.dae = DAE(self.M, 1, 10, self.N, 0.5, 0.5, 0.1, self.rng)
        self.dae.train(self.pop_data)
        self.s = self._state()

    def _reward(self):
        f1, cv1, d1 = self._arc_state()
        f0, cv0, d0 = self.r0
        return 1000 * ((f0 + cv0 + d1) - (f1 + cv1 + d0) + self.acc / self.reward_step)

    def _sample(self, D, cols):
        rng = self.rng
        D = np.array(D)
        use = np.arange(len(D)) if len(D) < 120 else rng.permutation(len(D))[:120]
        return D[use]

    def _train_models(self, update=False):
        rng, N = self.rng, self.N
        D = self._sample(self.data, None)
        if not update:
            self.critic = TanhMLP([N + 1, 30, 30, 1], rng).fit_regression(D[:, : N + 1], D[:, N + 1])
            self.actor = TanhMLP([N, 30, 30, 1], rng)
        else:
            self.critic.fit_regression(D[:, : N + 1], D[:, N + 1], half_mse=False)
        self._train_actor(D[:, :N])
        O = np.array(self.op_data)
        O = O if len(O) <= 120 else O[rng.permutation(len(O))[:120]]
        if not update:
            self.opnet = TanhMLP([N + 1, 30, 30, 1], rng).fit_regression(O[:, : N + 1], O[:, N + 1])
        else:
            self.opnet.fit_regression(O[:, : N + 1], O[:, N + 1], half_mse=False)

    def _train_actor(self, X, epochs=100, batch=60):
        self.actor.reset_adam()
        for ep in range(1, epochs + 1):
            for s in range(0, len(X), batch):
                x = X[s:s + batch]
                aa = self.actor.forward(x)
                ca = self.critic.forward(np.hstack([x, aa[-1]]))
                _, _, dIn = self.critic.backward(ca, -np.ones_like(ca[-1]))        # d(sum(-Q))/d(critic input)
                gW, gb, _ = self.actor.backward(aa, dIn[:, -1:])
                self.actor.adam(gW, gb, ep)

    def _policy(self, s):
        fe = float(self.actor.predict(s[None])[0, 0])
        q1 = float(self.opnet.predict(np.concatenate([s, [1.0]])[None])[0, 0])
        q2 = float(self.opnet.predict(np.concatenate([s, [2.0]])[None])[0, 0])
        op = 1 if q1 > q2 else 2 if q1 < q2 else int(self.rng.integers(1, 3))
        return fe, op

    def step(self):
        rng, N = self.rng, self.N
        self.g += 1
        g = self.g
        if g <= 100:
            o1 = self.evaluate(ga(self.problem, decs(self.P1)[tournament(2, N, self.fit1, rng=rng)], rng=rng))
            o2 = self.evaluate(ga(self.problem, decs(self.P2)[tournament(2, self.h, self.fit2, rng=rng)], rng=rng))
            self._update(o1, o2)
        elif g <= 600:
            self.step_ += 1
            if self.step_ == 1:
                self.Fe = self._random_fe()
                self.action = int(rng.integers(1, 3))
                self._begin_block()
            self._update(*self._offspring(self.action))
            if self.step_ >= self.reward_step:
                s2 = self._state()
                y = self._reward()
                self.data.append([*self.s, self.Fe, y, *s2])
                self.op_data.append([*self.s, self.action, y, *s2])
                self.step_ = 0
        else:
            self.step_ += 1
            if self.step_ == 1:
                self._begin_block()
            if not self.built:
                self._train_models()
                self.Fe, self.action = self._policy(self.s)
                self.built = True
                self._update(*self._offspring(self.action))
            else:
                self._update(*self._offspring(self.action))
                if self.step_ >= self.reward_step:
                    s2 = self._state()
                    r = self._reward()
                    ta = float(self.actor.predict(s2[None])[0, 0])
                    y = r + 0.95 * float(self.critic.predict(np.concatenate([s2, [ta]])[None])[0, 0])
                    self.data.append([*self.s, self.Fe, y, *s2])
                    self.data = self.data[-200:]
                    yq = r + 0.95 * float(self.opnet.predict(np.concatenate([s2, [self.action]])[None])[0, 0])
                    self.op_data.append([*self.s, self.action, yq, *s2])
                    self.op_data = self.op_data[-200:]
                    if rng.random() > 0.95:
                        self.Fe = self._random_fe()
                        self.action = int(rng.integers(1, 3))
                    else:
                        self.Fe, self.action = self._policy(s2)
                    self.step_ = 0
        self.pop_data = np.vstack([self.pop_data, objs(self.P1)])
        if len(self.pop_data) >= 1000:
            self.pop_data = self.pop_data[100:]
        if self.built and g % 200 == 0:
            self._train_models(update=True)
        self.pop = self.A if self.FE >= self.max_FE else self.P1
