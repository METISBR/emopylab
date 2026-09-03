"""
Standalone implementation of Crossover base operator (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.operator import Operator, default_random_state
from core.population import Population
from core.variable import Real, get

__all__ = [
    "Crossover",
]


class Crossover(Operator):
    """Base class for crossover operators."""

    def __init__(
        self,
        n_parents: int = 2,
        n_offsprings: int = 2,
        prob: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.prob = Real(prob, bounds=(0.5, 1.0), strict=(0.0, 1.0))

    @default_random_state
    def do(self, problem, pop, parents=None, *args: Any, random_state=None, **kwargs: Any):
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var

        X = np.swapaxes(
            np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1
        )
        if self.vtype is not None:
            X = X.astype(self.vtype)

        Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=X.dtype)

        prob = get(self.prob, size=n_matings)
        cross = random_state.random(n_matings) < prob

        if np.any(cross):
            Q = self._do(problem, X, *args, random_state=random_state, **kwargs)
            assert Q.shape == (n_offsprings, n_matings, problem.n_var), (
                f"Shape {Q.shape} is incorrect of crossover implementation, "
                f"expected {(n_offsprings, n_matings, problem.n_var)}"
            )
            Xp[:, cross] = Q[:, cross]

        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = random_state.choice(
                    np.arange(self.n_parents), size=n_offsprings, replace=False
                )
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(random_state.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = np.copy(X[s, k])

        Xp = Xp.reshape(-1, X.shape[-1])
        off = Population.new("X", Xp)
        return off

    def _do(self, problem, X: np.ndarray, *args: Any, random_state=None, **kwargs: Any) -> np.ndarray:
        return X
