# emopylab 2026
"""Izui (an aggregative gradient based multi-objective optimizer proposed by Izui et al.).

Reference:
K. Izui, T. Yamada, S. Nishiwaki, and K. Tanaka. Multiobjective optimization using an aggregative
gradient-based method. Structural and Multidisciplinary Optimization, 2015, 51: 173-182.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, objs

ALGORITHM_FLAGS = {'Izui': {'constrained', 'large', 'many', 'multi', 'real'}}


class Izui(LoopAlgorithm):
    """Every solution is moved by two LPs: aggregation weights ``w`` (from the current
    objective cloud) and a step minimising the weighted objective gradient inside the box
    ``[0.05*lower, 1.05*upper]``.  Gradients are forward finite differences, so each move
    charges ``D + 2`` function evaluations (as in the reference implementation)."""

    def _gradient(self, x, infeasible):
        og, cg = self.cal_grad(x)
        return (cg if infeasible and cg.shape[0] == self.M else og).T   # (D, M)

    def _weights(self, i):
        A = objs(self.pop)
        r = linprog(c=objs(self.pop[i:i + 1])[0], A_ub=-A, b_ub=-np.ones(len(A)), bounds=(0, None), method="highs")
        return r.x if r.success else None

    def step(self):
        for i in range(self.N):
            w = self._weights(i)
            if w is None:
                continue
            g = self._gradient(decs(self.pop[i:i + 1])[0], bool(np.any(cons(self.pop[i:i + 1]) > 0)))
            r = linprog(c=g @ w, bounds=list(zip(0.05 * self.lower, 1.05 * self.upper)), method="highs")
            if not r.success:
                continue
            self.pop[i] = self.evaluate(r.x[None, :])[0]
