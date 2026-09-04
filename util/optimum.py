"""EmoPyLab Optimum and Pareto Front filtering helpers (zero-pymoo standalone)."""

from __future__ import annotations

from typing import Any
import numpy as np

from util.nds.non_dominated_sorting import find_non_dominated


def filter_optimum(pop: Any, least_infeasible: bool = True) -> Any:
    """Filter the optimal / non-dominated individuals from a population or array of solutions.

    Args:
        pop: A Population instance, list of Individual instances, or NumPy array of objective values F.
        least_infeasible: If True and no feasible solutions exist, returns the solution(s) with minimal CV.

    Returns:
        Filtered population or non-dominated subset.
    """
    if pop is None or len(pop) == 0:
        return pop

    # Check if pop is a Population-like object
    if hasattr(pop, "get"):
        F = pop.get("F")
        G = pop.get("G")
        CV = pop.get("CV")
        feasible = pop.get("feasible")
    else:
        # Array-like or list
        if isinstance(pop, np.ndarray) and pop.ndim == 2:
            mask = find_non_dominated(pop)
            return pop[mask]
        # List of individuals
        F = np.array([ind.F for ind in pop if hasattr(ind, "F")])
        G = None
        CV = None
        feasible = None

    if F is None or len(F) == 0:
        return pop

    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]

    # Calculate CV if not directly present
    if CV is None:
        if G is not None:
            G_arr = np.asarray(G, dtype=float)
            if G_arr.ndim == 1:
                G_arr = G_arr[:, None]
            CV = np.sum(np.maximum(0.0, G_arr), axis=1)
        elif feasible is not None:
            CV = np.where(np.asarray(feasible, dtype=bool), 0.0, 1.0)

    n = len(F)
    if CV is not None:
        CV = np.asarray(CV, dtype=float).reshape(-1)
        is_feas = CV <= 0.0
        if np.any(is_feas):
            feas_indices = np.where(is_feas)[0]
            nd_sub = find_non_dominated(F[feas_indices])
            best_indices = feas_indices[nd_sub]
        else:
            if least_infeasible:
                min_cv = np.min(CV)
                min_cv_indices = np.where(CV <= min_cv + 1e-12)[0]
                nd_sub = find_non_dominated(F[min_cv_indices])
                best_indices = min_cv_indices[nd_sub]
            else:
                best_indices = np.array([], dtype=int)
    else:
        nd_mask = find_non_dominated(F)
        best_indices = np.where(nd_mask)[0]

    # Return filtered population or slice
    if hasattr(pop, "__getitem__"):
        try:
            return pop[best_indices]
        except Exception:
            return [pop[i] for i in best_indices]

    return best_indices
