"""
Standalone implementation of ReplacementSurvival and ImprovementReplacement (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from core.duplicate import DefaultDuplicateElimination
from core.individual import Individual
from core.population import Population
from core.survival import Survival

__all__ = [
    "ReplacementSurvival",
    "ImprovementReplacement",
    "is_better",
    "parameter_less",
    "hierarchical_sort",
]


def is_better(_new: Individual, _old: Individual, eps: float = 0.0) -> bool:
    both_infeasible = not _old.feas and not _new.feas
    both_feasible = _old.feas and _new.feas

    if both_infeasible and _old.CV[0] - _new.CV[0] > eps:
        return True
    elif not _old.FEAS and _new.FEAS:
        return True
    elif both_feasible and _old.F[0] - _new.F[0] > eps:
        return True

    return False


class ReplacementSurvival(Survival):
    """Survival based on parent-offspring replacement."""

    def do(
        self,
        problem,
        pop: Union[Population, Individual],
        off: Union[Population, Individual, int],
        return_indices: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ):
        if isinstance(off, int):
            k = off
            off = pop[k:]
            pop = pop[:k]

        if len(off) == 0:
            return pop

        assert len(pop) == len(off), "For replacement, pop and off must have same length."

        pop = Population.create(pop) if isinstance(pop, Individual) else pop
        off = Population.create(off) if isinstance(off, Individual) else off

        I = self._do(problem, pop, off, **kwargs)

        if return_indices:
            return I
        else:
            if not inplace:
                pop = pop.copy()
            pop[I] = off[I]
            return pop

    def _do(self, problem, pop: Population, off: Population, **kwargs: Any) -> np.ndarray:
        return np.full(len(pop), False)


class ImprovementReplacement(ReplacementSurvival):
    """Replaces individual if offspring is better (feasibility + objective improvement)."""

    def _do(self, problem, pop: Population, off: Population, **kwargs: Any) -> np.ndarray:
        ret = np.full((len(pop), 1), False)

        pop_F, pop_CV, pop_feas = pop.get("F", "CV", "FEAS")
        off_F, off_CV, off_feas = off.get("F", "CV", "FEAS")

        if problem.has_constraints():
            ret[(~pop_feas & ~off_feas) & (off_CV < pop_CV)] = True
            ret[~pop_feas & off_feas] = True
            ret[(pop_feas & off_feas) & (off_F < pop_F)] = True
        else:
            ret[off_F < pop_F] = True

        _, _, is_duplicate = DefaultDuplicateElimination(epsilon=0.0).do(
            off, pop, return_indices=True
        )
        ret[is_duplicate] = False

        return ret[:, 0]


def parameter_less(f: np.ndarray, cv: np.ndarray) -> np.ndarray:
    v = np.copy(f)
    infeas = cv > 0
    if np.any(infeas):
        v[infeas] = f.max() + cv[infeas]
    return v


def hierarchical_sort(f: np.ndarray, cv: Optional[np.ndarray] = None) -> np.ndarray:
    if cv is not None:
        f = parameter_less(f, cv)
    return np.argsort(f)
