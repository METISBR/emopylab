"""
Standalone implementation of Repair and NoRepair (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.operator import Operator

__all__ = [
    "Repair",
    "NoRepair",
]


class Repair(Operator):
    """Base class for repairing invalid candidate solutions."""

    def do(self, problem, pop, **kwargs: Any):
        X = np.array([ind.X for ind in pop])
        if self.vtype is not None:
            X = X.astype(self.vtype)

        Xp = self._do(problem, X, **kwargs)
        pop.set("X", Xp)
        return pop

    def _do(self, problem, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return X


class NoRepair(Repair):
    """No-op repair operator."""
    pass
