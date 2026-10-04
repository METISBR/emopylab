# emopylab 2026
"""FRCG-M (fletcher-Reeves conjugate gradient (for multi-objective optimization)).

Reference:
R. Fletcher and C. M. Reeves. Function minimization by conjugate gradients. The Computer Journal,
1964, 7(2): 149-154.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, gradient_direction, objs, uniform_point

ALGORITHM_FLAGS = {'FRCGM': {'constrained', 'large', 'many', 'multi', 'real'}}


class FRCGM(LoopAlgorithm):
    """Fletcher-Reeves conjugate-gradient descent of every weighted subproblem (Armijo backtracking)."""

    def __init__(self, pop_size: int = 100, beta: float = 0.6, sigma: float = 0.4, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.beta, self.sigma = float(beta), float(sigma)

    def initial_size(self):
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.pop_size

    def start(self):
        self.k = 0
        self.g0 = [None] * self.N
        self.d0 = [None] * self.N

    def step(self):
        D = self.D
        for i in range(self.N):
            gk, _ = gradient_direction(self, self.pop[i], self.W[i])
            itern = self.k - (D + 1) * (self.k // (D + 1)) + 1
            if itern <= 1:
                dk = -gk
            else:
                betak = (gk @ gk) / (self.g0[i] @ self.g0[i])
                dk = -gk + betak * self.d0[i]
                if gk @ dk >= 0:
                    dk = -gk
            f_old = objs(self.pop[i:i + 1])[0] @ self.W[i]
            x0 = np.asarray(self.pop[i].X, dtype=float)
            for m in range(21):
                X = self.evaluate((x0 + self.beta ** m * dk)[None, :])
                if objs(X)[0] @ self.W[i] <= f_old + self.sigma * self.beta ** m * (gk @ dk):
                    break
            self.pop[i] = X[0]
            self.g0[i], self.d0[i] = gk, dk
        self.k += 1
