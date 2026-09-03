"""EmoPyLab Dominance and Pareto relations (zero-pymoo standalone)."""

from __future__ import annotations

from typing import Any
import numpy as np


class Dominator:
    """Dominator class for computing Pareto dominance relations."""

    @staticmethod
    def get_relation(a: Any, b: Any, c_a: float = 0.0, c_b: float = 0.0) -> int:
        """Return 1 if a dominates b, -1 if b dominates a, 0 if non-dominated."""
        # Feasibility check if CV (constraint violation) is present
        val_a = float(c_a) if c_a is not None else 0.0
        val_b = float(c_b) if c_b is not None else 0.0

        if val_a < 0:
            val_a = 0.0
        if val_b < 0:
            val_b = 0.0

        if val_a == 0.0 and val_b > 0.0:
            return 1
        elif val_a > 0.0 and val_b == 0.0:
            return -1
        elif val_a > 0.0 and val_b > 0.0:
            if val_a < val_b:
                return 1
            elif val_b < val_a:
                return -1
            else:
                return 0

        # Both feasible: Pareto dominance comparison
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)

        a_better = False
        b_better = False

        for x, y in zip(arr_a, arr_b):
            if x < y:
                a_better = True
                if b_better:
                    return 0
            elif y < x:
                b_better = True
                if a_better:
                    return 0

        if a_better and not b_better:
            return 1
        elif b_better and not a_better:
            return -1
        else:
            return 0

    @classmethod
    def calc_domination_matrix(cls, F: np.ndarray, G: np.ndarray | None = None) -> np.ndarray:
        """Compute NxN matrix M where M[i, j] is 1 if i dominates j, -1 if j dominates i, 0 otherwise."""
        F = np.asarray(F, dtype=float)
        n = F.shape[0]
        M = np.zeros((n, n), dtype=int)

        CV = None
        if G is not None:
            G = np.asarray(G, dtype=float)
            if G.ndim == 1:
                G = G[:, None]
            CV = np.sum(np.maximum(0.0, G), axis=1)

        for i in range(n):
            for j in range(i + 1, n):
                c_i = CV[i] if CV is not None else 0.0
                c_j = CV[j] if CV is not None else 0.0
                rel = cls.get_relation(F[i], F[j], c_a=c_i, c_b=c_j)
                M[i, j] = rel
                M[j, i] = -rel

        return M


def get_relation(a: Any, b: Any, c_a: float = 0.0, c_b: float = 0.0) -> int:
    return Dominator.get_relation(a, b, c_a=c_a, c_b=c_b)
