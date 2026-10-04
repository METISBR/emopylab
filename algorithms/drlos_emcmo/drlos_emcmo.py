# emopylab 2026
"""DRLOS-EMCMO (eMCMO with deep reinforcement learning-assisted operator selection).

Reference:
F. Ming, W. Gong, L. Wang, and Y. Jin. Constrained multi-objective optimization with deep
reinforcement learning assisted operator selection. IEEE/CAA Journal of Automatica Sinica, 2024,
11(4): 919-931.
"""

from __future__ import annotations

import numpy as np

from algorithms.cmoqlmt.cmoqlmt import spea_selection
from algorithms.community_utils.base import LoopAlgorithm, cons, de, decs, ga_half, objs, tournament
from algorithms.community_utils.dropout_net import DropoutNet, minmax_apply, minmax_fit, minmax_reverse
from core.population import Population

ALGORITHM_FLAGS = {'DRLOSEMCMO': {'constrained', 'integer', 'multi', 'real'}}


def _state(pop):
    F, C = objs(pop), cons(pop)
    n = len(F)
    cv = np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(n)
    return np.array([F.sum() / n, cv.sum() / n, (F.max(0) - F.min(0)).sum()])


class DRLOSEMCMO(LoopAlgorithm):
    """Two tasks (constrained main problem and its unconstrained helper) evolved by GA or DE, the operator chosen at random
    for the first 200 generations and then (95%) by a dropout network trained on transitions (population state, operator)
    -> (reward, next state) with a discounted target refreshed every 50 generations. After 20% of the budget the tasks
    exchange offspring or half of a population depending on the transferred solutions' survival."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        N = self.N
        P2 = self.evaluate(self.random_decs(N))
        self.P = [infills, P2]
        C1 = cons(infills)
        self.fit = [None, None]
        self.P[0], self.fit[0], _ = spea_selection(infills, N, True)
        self.P[1], self.fit[1], _ = spea_selection(P2, N, False)
        self.transfer, self.cnt, self.data = 0, 0, []
        self.net, self.count = None, 0
        self.pop = self.P[0]
        self._set_optimum()

    def _operator_choice(self):
        rng, N = self.rng, self.N
        s = _state(self.P[0])
        gen = int(np.ceil(self.FE / (2 * N)))
        if gen <= 200 or len(self.data) < 200:
            return int(rng.integers(1, 3)), s
        D = np.array(self.data)
        if self.net is None:
            use = rng.permutation(len(D))[:200]
            self.ps, self.qs = minmax_fit(D[use, :4]), minmax_fit(D[use, 4:8])
            self.net = DropoutNet(4, 4, rng).train(minmax_apply(D[use, :4], self.ps), minmax_apply(D[use, 4:8], self.qs), 80000)
            return int(rng.integers(1, 3)), s
        if rng.random() > 0.95:
            return int(rng.integers(1, 3)), s
        x = minmax_apply(np.array([[*s, 1.0], [*s, 2.0]]), self.ps)
        succ = minmax_reverse(self.net.predict(x), self.qs)
        return int(np.argmax(succ[:, 0])) + 1, s

    def _offspring(self, i, op, tour):
        rng, N = self.rng, self.N
        X = decs(self.P[i])
        if op == 1:
            par = X[tournament(2, N, self.fit[i], rng=rng)] if tour else X[rng.integers(0, N, N)]
            return self.evaluate(ga_half(self.problem, par, rng=rng))
        if tour:
            m = tournament(2, 2 * N, self.fit[i], rng=rng)
            return self.evaluate(de(self.problem, X, X[m[:N]], X[m[N:]], rng=rng))
        return self.evaluate(de(self.problem, X, X[rng.integers(0, N, N)], X[rng.integers(0, N, N)], rng=rng))

    def step(self):
        rng, N = self.rng, self.N
        op, s0 = self._operator_choice()
        self.cnt += 1
        if self.transfer == 0:
            off = [self._offspring(i, op, False) for i in range(2)]
            self.P[0], self.fit[0], _ = spea_selection(Population.merge(self.P[0], off[0], off[1]), N, True)
            self.P[1], self.fit[1], _ = spea_selection(Population.merge(self.P[1], off[1], off[0]), N, False)
            if self.FE / self.max_FE >= 0.2:
                self.transfer = 1
        else:
            off = [self._offspring(i, op, True) for i in range(2)]
            _, _, nx = spea_selection(Population.merge(self.P[1], off[1]), N, True)
            s1 = nx[:N].sum() / 100 - nx[N:].sum() / 50
            _, _, nx = spea_selection(Population.merge(self.P[0], off[0]), N, False)
            s2 = nx[:N].sum() / 100 - nx[N:].sum() / 50
            for i, sr in ((0, s1), (1, s2)):
                other = 1 - i
                if sr > 0:
                    extra = self.P[other][rng.permutation(N)[: N // 2]]
                else:
                    extra = off[other]
                self.P[i], self.fit[i], _ = spea_selection(Population.merge(self.P[i], off[i], extra), N, i == 0)
        s1_ = _state(self.P[0])
        reward = s1_.sum() - s0.sum()
        self.data.append([*s0, float(op), reward, *s1_])
        if len(self.data) > 500:
            del self.data[-1]                              # literal: the newest record is dropped
        if self.net is not None:
            self.count += 1
            if self.count > 50:
                D = np.array(self.data)
                use = rng.permutation(len(D))[:200]
                self.ps = minmax_fit(D[use, :4])
                x = minmax_apply(D[use, :4], self.ps)
                succ = minmax_reverse(self.net.predict(x), self.qs)[:, 0]
                y = D[use, 4:5] + 0.9 * succ.max()
                self.qs = minmax_fit(y)
                self.net.train(x, minmax_apply(y, self.qs), 8000)
                self.count = 0
        self.pop = self.P[0]
