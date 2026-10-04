# emopylab 2026
"""SGECF (sparsity-guided elitism co-evolutionary framework).

Reference:
C. Wu, Y. Tian, Y. Zhang, H. Jiang, and X. Zhang. A sparsity-guided elitism co-evolutionary
framework for sparse large-scale multi-objective optimization. Proceedings of the IEEE Congress on
Evolutionary Computation, 2023.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm
from algorithms.mskea.mskea import _spea2_selection
from algorithms.scea.scea import initial_population, operator_max, operator_min, operator_win, SCEA
from core.population import Population

ALGORITHM_FLAGS = {'SGECF': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class SGECF(SCEA):
    """Sparse EA with a group-based evolution of winners and losers: the non-dominated masks (winners) recombine
    among themselves, while the losers on each side of the median winner sparsity are pulled toward it."""

    def _initialize_infill(self):
        rng = self.rng
        D = self.D

        def mask_size(VMask, front, Fit):
            return lambda: int(np.ceil(rng.random() * D))
        return initial_population(self, mask_size)

    def _offspring(self, sub, tm):
        (pw, dw, mw, rw), (p1, d1, m1, r1), (p2, d2, m2, r2) = sub
        F = self.fitness
        parts = [operator_win(self, dw, mw, rw)]
        if len(p1) > 0:
            parts.append(operator_min(self, dw, mw, rw, d1, m1, r1, F, tm))
        if len(p2) > 0:
            parts.append(operator_max(self, dw, mw, rw, d2, m2, r2, F, tm))
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])
