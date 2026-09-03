# emopylab 2026

# Reference:
# K. Deb and H. Jain. An evolutionary many-objective optimization algorithm
# using reference-point based non-dominated sorting approach, part I:
# Solving problems with box constraints. IEEE TEC, 2014.

from __future__ import annotations

from typing import Any

import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from util.optimum import filter_optimum

from algorithms.community_utils.moead_family import rng_from_algo, sample_initial
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from operators.utility_functions.UniformPoint import UniformPoint


ALGORITHM_FLAGS = {
    "NSGA3Local": {
        "multi",
        "many",
        "real",
        "integer",
        "binary",
        "permutation",
        "label",
        "constrained",
    }
}


def _as_2d(values: Any, *, n_rows: int, dtype: Any = float) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    target_rows = int(n_rows)
    if arr.ndim == 0:
        if target_rows <= 1:
            arr = arr.reshape(1, 1)
        else:
            raise ValueError(f"Expected {target_rows} rows, got scalar.")
    elif arr.ndim == 1:
        if target_rows <= 1:
            arr = arr.reshape(1, -1)
        elif arr.shape[0] == target_rows:
            arr = arr.reshape(-1, 1)
        else:
            raise ValueError(f"Expected {target_rows} rows, got vector with {arr.shape[0]} values.")
    if arr.shape[0] != target_rows:
        if target_rows == 0 and arr.size == 0:
            return np.zeros((0, 0), dtype=dtype)
        raise ValueError(f"Expected {target_rows} rows, got {arr.shape[0]}.")
    return arr


def _population_objectives(pop: Population) -> np.ndarray:
    if pop is None or len(pop) == 0:
        return np.zeros((0, 0), dtype=float)
    if hasattr(pop, "get"):
        values = pop.get("F")
    else:
        values = np.array([getattr(ind, "F", None) for ind in pop])
    if values is None:
        raise RuntimeError("Population objective matrix 'F' is required by NSGA3Local.")
    return _as_2d(values, n_rows=len(pop), dtype=float)


def _population_constraints(pop: Population) -> np.ndarray:
    if pop is None or len(pop) == 0:
        return np.zeros((0, 0), dtype=float)
    if hasattr(pop, "get"):
        values = pop.get("G")
    else:
        values = np.array([getattr(ind, "G", None) for ind in pop])
    if values is None:
        return np.zeros((len(pop), 0), dtype=float)
    return _as_2d(values, n_rows=len(pop), dtype=float)

def _population_feasible_mask(pop: Population) -> np.ndarray:
    cons = _population_constraints(pop)
    if cons.size == 0:
        return np.ones(len(pop), dtype=bool)
    return np.all(cons <= 0.0, axis=1)


def _constraint_violation(pop: Population) -> np.ndarray:
    cons = _population_constraints(pop)
    if cons.size == 0:
        return np.zeros(len(pop), dtype=float)
    return np.sum(np.maximum(0.0, cons), axis=1)


def _update_zmin(
    current_zmin: np.ndarray | None,
    pop: Population | None,
    n_obj: int,
) -> np.ndarray:
    if pop is None or len(pop) == 0:
        if current_zmin is not None:
            return np.asarray(current_zmin, dtype=float).reshape(-1)
        return np.ones(int(n_obj), dtype=float)

    F = _population_objectives(pop)
    feasible = _population_feasible_mask(pop)
    if not np.any(feasible):
        if current_zmin is not None:
            return np.asarray(current_zmin, dtype=float).reshape(-1)
        return np.ones(int(n_obj), dtype=float)

    candidate = np.min(F[feasible], axis=0)
    if current_zmin is None:
        return np.asarray(candidate, dtype=float).reshape(-1)
    return np.minimum(np.asarray(current_zmin, dtype=float).reshape(-1), candidate)


def _row_norms(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=float), axis=1)


def _last_selection(
    pop_obj_1: np.ndarray,
    pop_obj_2: np.ndarray,
    k: int,
    ref_dirs: np.ndarray,
    zmin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n2 = int(pop_obj_2.shape[0])
    choose = np.zeros(n2, dtype=bool)
    if n2 == 0 or int(k) <= 0:
        return choose

    k = int(max(0, min(int(k), n2)))
    pop_obj_1 = np.asarray(pop_obj_1, dtype=float)
    pop_obj_2 = np.asarray(pop_obj_2, dtype=float)
    ref_dirs = np.asarray(ref_dirs, dtype=float)
    zmin = np.asarray(zmin, dtype=float).reshape(-1)

    if ref_dirs.ndim != 2 or ref_dirs.shape[0] == 0:
        choose[:k] = True
        return choose

    pop_obj = np.vstack([pop_obj_1, pop_obj_2]) - zmin[None, :]
    n, m = pop_obj.shape
    n1 = int(pop_obj_1.shape[0])
    n_ref = int(ref_dirs.shape[0])

    # MATLAB ASF-based extreme point detection with small epsilon weights.
    w = np.eye(m, dtype=float) + 1e-6
    extreme = np.zeros(m, dtype=int)
    for i in range(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            asf = np.max(pop_obj / w[i][None, :], axis=1)
        asf = np.where(np.isfinite(asf), asf, np.inf)
        extreme[i] = int(np.argmin(asf))

    # MATLAB uses backslash; lstsq provides similar least-squares behavior.
    intercept = None
    try:
        intercept = np.linalg.lstsq(pop_obj[extreme, :], np.ones(m, dtype=float), rcond=None)[0]
    except Exception:
        intercept = None

    if intercept is None:
        a = np.max(pop_obj, axis=0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = 1.0 / intercept
        if np.any(~np.isfinite(a)) or np.any(a <= 1e-12):
            a = np.max(pop_obj, axis=0)

    a = np.where(np.isfinite(a) & (a > 1e-12), a, 1.0)
    pop_obj = pop_obj / a[None, :]

    pop_norm = _row_norms(pop_obj)
    ref_norm = _row_norms(ref_dirs)
    denom = pop_norm[:, None] * ref_norm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.divide(
            pop_obj @ ref_dirs.T,
            denom,
            out=np.zeros((n, n_ref), dtype=float),
            where=denom > 0.0,
        )
    cosine = np.clip(cosine, -1.0, 1.0)
    distance = pop_norm[:, None] * np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))

    pi = np.argmin(distance, axis=1).astype(int, copy=False)
    d = distance[np.arange(n), pi]
    rho = np.bincount(pi[:n1], minlength=n_ref).astype(int, copy=False)

    zchoose = np.ones(n_ref, dtype=bool)
    while int(np.sum(choose)) < k:
        active_refs = np.where(zchoose)[0]
        if active_refs.size == 0:
            break

        active_rho = rho[active_refs]
        min_rho = int(np.min(active_rho))
        tied = active_refs[active_rho == min_rho]
        if tied.size == 1:
            j = int(tied[0])
        else:
            j = int(tied[int(rng.integers(0, tied.size))])

        candidate = np.where((~choose) & (pi[n1:] == j))[0]
        if candidate.size > 0:
            if rho[j] == 0:
                pick_local = int(np.argmin(d[n1 + candidate]))
            else:
                pick_local = int(rng.integers(0, candidate.size))
            choose[int(candidate[pick_local])] = True
            rho[j] += 1
        else:
            zchoose[j] = False

    # Fallback for degenerate associations: fill remaining slots by smallest distance.
    remaining_needed = int(k - np.sum(choose))
    if remaining_needed > 0:
        remaining_idx = np.where(~choose)[0]
        if remaining_idx.size > 0:
            order = np.argsort(d[n1 + remaining_idx], kind="mergesort")
            pick = remaining_idx[order[:remaining_needed]]
            choose[pick] = True

    return choose


def _environmental_selection(
    pop: Population,
    n_survive: int,
    ref_dirs: np.ndarray,
    zmin: np.ndarray,
    rng: np.random.Generator,
) -> Population:
    n_survive = int(max(1, n_survive))
    F = _population_objectives(pop)
    G = _population_constraints(pop)
    front_no, max_f_no = NDSort(F, G, n_survive)

    front_no = np.asarray(front_no, dtype=float).reshape(-1)
    next_mask = front_no < float(max_f_no)
    last = np.where(front_no == float(max_f_no))[0]

    remaining = int(n_survive - np.sum(next_mask))
    if remaining > 0 and last.size > 0:
        choose_last = _last_selection(
            F[next_mask],
            F[last],
            remaining,
            np.asarray(ref_dirs, dtype=float),
            np.asarray(zmin, dtype=float).reshape(-1),
            rng,
        )
        next_mask[last[choose_last]] = True

    selected = np.where(next_mask)[0]
    if selected.size < n_survive:
        remain_idx = np.where(~next_mask)[0]
        if remain_idx.size > 0:
            deficit = int(min(n_survive - selected.size, remain_idx.size))
            remain_front = front_no[remain_idx]
            order = np.argsort(remain_front, kind="mergesort")
            selected = np.concatenate([selected, remain_idx[order[:deficit]]])

    if selected.size > n_survive:
        selected = selected[:n_survive]

    return pop[selected]


class NSGA3Local(Algorithm):
    """NSGA-III local port with MATLAB-style reference-point niching."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs=None,
        sampling=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(max(2, pop_size))
        self.ref_dirs = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.sampling = sampling
        self.zmin: np.ndarray | None = None

    def _setup(self, problem, **kwargs):
        valid_ref_dirs = (
            isinstance(self.ref_dirs, np.ndarray)
            and self.ref_dirs.ndim == 2
            and self.ref_dirs.shape[0] > 0
            and self.ref_dirs.shape[1] == int(problem.n_obj)
        )
        if not valid_ref_dirs:
            generated_ref_dirs, n_effective = UniformPoint(self.pop_size, int(problem.n_obj))
            self.ref_dirs = np.asarray(generated_ref_dirs, dtype=float)
            self.pop_size = int(max(1, n_effective))
        else:
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills if infills is not None else Population.empty()
        self.zmin = _update_zmin(None, self.pop, int(self.problem.n_obj))

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))
        rng = rng_from_algo(self)
        cv = _constraint_violation(self.pop)
        mating = np.asarray(TournamentSelection(2, self.pop_size, cv, rng=rng), dtype=int) - 1
        mating = np.clip(mating, 0, len(self.pop) - 1)
        return OperatorGA(self.problem, self.pop[mating], rng=rng)

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return
        self.zmin = _update_zmin(self.zmin, infills, int(self.problem.n_obj))
        self.pop = _environmental_selection(
            Population.merge(self.pop, infills),
            self.pop_size,
            np.asarray(self.ref_dirs, dtype=float),
            np.asarray(self.zmin, dtype=float),
            rng_from_algo(self),
        )

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)
