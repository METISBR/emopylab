"""
Standalone implementation of DuplicateElimination (EmoPyLab 2026).
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union

__all__ = [
    "DuplicateElimination",
    "DefaultDuplicateElimination",
    "ElementwiseDuplicateElimination",
    "HashDuplicateElimination",
    "NoDuplicateElimination",
]


def default_attr(pop):
    return pop.get("X")


def _cdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between matrices A and B."""
    try:
        from scipy.spatial.distance import cdist
        return cdist(A.astype(float), B.astype(float))
    except Exception:
        A = np.atleast_2d(A.astype(float))
        B = np.atleast_2d(B.astype(float))
        # Efficient euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        d = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1, keepdims=True).T - 2.0 * np.dot(A, B.T)
        d = np.maximum(d, 0.0)
        return np.sqrt(d)


class DuplicateElimination:
    """Base class for duplicate elimination."""

    def __init__(self, func: Optional[Callable] = None) -> None:
        self.func = func if func is not None else default_attr

    def do(
        self,
        pop,
        *args,
        return_indices: bool = False,
        to_itself: bool = True,
    ):
        original = pop

        if len(pop) == 0:
            return (pop, [], []) if return_indices else pop

        if to_itself:
            mask = self._do(pop, None, np.full(len(pop), False))
            pop = pop[~mask]

        for arg in args:
            if len(arg) > 0:
                if len(pop) == 0:
                    break
                elif len(arg) == 0:
                    continue
                else:
                    mask = self._do(pop, arg, np.full(len(pop), False))
                    pop = pop[~mask]

        if return_indices:
            no_duplicate, is_duplicate = [], []
            H = set(pop)

            for i, ind in enumerate(original):
                if ind in H:
                    no_duplicate.append(i)
                else:
                    is_duplicate.append(i)

            return pop, no_duplicate, is_duplicate
        else:
            return pop

    def _do(self, pop, other, is_duplicate: np.ndarray) -> np.ndarray:
        return is_duplicate


class DefaultDuplicateElimination(DuplicateElimination):
    """Default duplicate elimination based on Euclidean distance in decision/attribute space."""

    def __init__(self, epsilon: float = 1e-16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def calc_dist(self, pop, other=None) -> np.ndarray:
        X = self.func(pop)

        if other is None:
            D = _cdist(X, X)
            D[np.triu_indices(len(X))] = np.inf
        else:
            _X = self.func(other)
            D = _cdist(X, _X)

        return D

    def _do(self, pop, other, is_duplicate: np.ndarray) -> np.ndarray:
        D = self.calc_dist(pop, other)
        D[np.isnan(D)] = np.inf

        is_duplicate[np.any(D <= self.epsilon, axis=1)] = True
        return is_duplicate


def to_float(val: Any) -> float:
    if isinstance(val, (bool, np.bool_)):
        return 0.0 if val else 1.0
    return float(val)


class ElementwiseDuplicateElimination(DefaultDuplicateElimination):
    """Duplicate elimination using elementwise comparison function."""

    def __init__(self, cmp_func: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if cmp_func is None:
            cmp_func = self.is_equal
        self.cmp_func = cmp_func

    def is_equal(self, a: Any, b: Any) -> bool:
        return a == b

    def _do(self, pop, other, is_duplicate: np.ndarray) -> np.ndarray:
        if other is None:
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    val = to_float(self.cmp_func(pop[i], pop[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break
        else:
            for i in range(len(pop)):
                for j in range(len(other)):
                    val = to_float(self.cmp_func(pop[i], other[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break

        return is_duplicate


def to_hash(x: Any) -> int:
    try:
        return hash(x)
    except Exception:
        try:
            return hash(str(x))
        except Exception as e:
            raise ValueError(
                "Hash could not be calculated for duplicate elimination."
            ) from e


class HashDuplicateElimination(DuplicateElimination):
    """Duplicate elimination by hashing."""

    def __init__(self, func: Callable = to_hash) -> None:
        super().__init__()
        self.func = func

    def _do(self, pop, other, is_duplicate: np.ndarray) -> np.ndarray:
        H = set()

        if other is not None:
            for o in other:
                val = self.func(o)
                H.add(self.func(val))

        for i, ind in enumerate(pop):
            val = self.func(ind)
            h = self.func(val)

            if h in H:
                is_duplicate[i] = True
            else:
                H.add(h)

        return is_duplicate


class NoDuplicateElimination(DuplicateElimination):
    """No-op duplicate elimination."""

    def do(self, pop, *args, **kwargs):
        if kwargs.get("return_indices", False):
            return pop, list(range(len(pop))), []
        return pop
