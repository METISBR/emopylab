# emopylab 2026
"""Canonical NSGA-III with deterministic hardware-accelerated survival kernels.

Algorithmic reference:
K. Deb and H. Jain. An evolutionary many-objective optimization algorithm
using reference-point based non-dominated sorting approach, part I:
Solving problems with box constraints. IEEE TEC, 2014.

Semantics match the established Python implementation used by the scientific
community: constrained tournament selection (CV first, random among feasible),
SBX(eta=30, prob=1), polynomial mutation(eta=20), persistent hyperplane
normalization, reference-direction association, and canonical niching.
The main survival tensor operations (dominance matrix, perpendicular distance,
normalization) dispatch through Torch CUDA/MPS, CuPy, MLX, or NumPy with a
transparent deterministic NumPy fallback.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from core.survival import Survival
from util.optimum import filter_optimum
from util.nds.non_dominated_sorting import NonDominatedSorting
from algorithms.community_utils.moead_family import rng_from_algo, sample_initial
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.UniformPoint import UniformPoint
from operators.sampling.lhs import LatinHypercubeSampling

try:
    from core.nds.gpu_nds import boolean_matrix_nds
except Exception:  # pragma: no cover
    boolean_matrix_nds = None


ALGORITHM_FLAGS = {
    "NSGA-III": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
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
    values = pop.get("F") if hasattr(pop, "get") else np.array([getattr(ind, "F", None) for ind in pop])
    if values is None:
        raise RuntimeError("Population objective matrix 'F' is required by NSGA3.")
    return _as_2d(values, n_rows=len(pop), dtype=float)


def _population_constraints(pop: Population) -> np.ndarray:
    if pop is None or len(pop) == 0:
        return np.zeros((0, 0), dtype=float)
    values = pop.get("G") if hasattr(pop, "get") else np.array([getattr(ind, "G", None) for ind in pop])
    if values is None:
        return np.zeros((len(pop), 0), dtype=float)
    return _as_2d(values, n_rows=len(pop), dtype=float)


def _population_feasible_mask(pop: Population) -> np.ndarray:
    cons = _population_constraints(pop)
    return np.ones(len(pop), dtype=bool) if cons.size == 0 else np.all(cons <= 0.0, axis=1)


def _constraint_violation(pop: Population) -> np.ndarray:
    cons = _population_constraints(pop)
    return np.zeros(len(pop), dtype=float) if cons.size == 0 else np.sum(np.maximum(0.0, cons), axis=1)


def _update_zmin(current_zmin: np.ndarray | None, pop: Population | None, n_obj: int) -> np.ndarray:
    if pop is None or len(pop) == 0:
        return np.asarray(current_zmin, dtype=float).reshape(-1) if current_zmin is not None else np.ones(int(n_obj))
    F = _population_objectives(pop)
    feasible = _population_feasible_mask(pop)
    if not np.any(feasible):
        return np.asarray(current_zmin, dtype=float).reshape(-1) if current_zmin is not None else np.ones(int(n_obj))
    candidate = np.min(F[feasible], axis=0)
    return np.asarray(candidate, dtype=float).reshape(-1) if current_zmin is None else np.minimum(current_zmin, candidate)


def _device_array(values: np.ndarray, backend: str):
    """Materialize a float32 tensor on the requested accelerator, else NumPy."""
    arr = np.ascontiguousarray(values, dtype=np.float32)
    if backend == "torch":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        return torch.as_tensor(arr, device=dev)
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(arr)
    if backend == "mlx":
        import mlx.core as mx
        return mx.array(arr)
    return arr


def _to_numpy(value: Any) -> np.ndarray:
    mod = type(value).__module__
    if "torch" in mod:
        return value.detach().cpu().numpy()
    if "cupy" in mod:
        return value.get()
    if "mlx" in mod:
        import mlx.core as mx
        mx.eval(value)
        return np.asarray(value)
    return np.asarray(value)


def _perpendicular_distance(F: np.ndarray, ref_dirs: np.ndarray, backend: str = "numpy") -> np.ndarray:
    """Perpendicular distances; same float32 equations on every backend."""
    F32 = np.ascontiguousarray(F, dtype=np.float32)
    Z32 = np.ascontiguousarray(ref_dirs, dtype=np.float32)
    try:
        A = _device_array(F32, backend)
        Z = _device_array(Z32, backend)
        if backend == "torch":
            import torch
            z_unit = Z / torch.clamp(torch.linalg.vector_norm(Z, dim=1, keepdim=True), min=1e-12)
            proj = A @ z_unit.T
            p2 = torch.sum(A * A, dim=1, keepdim=True)
            dist = torch.sqrt(torch.clamp(p2 - proj * proj, min=0.0))
            if A.device.type == "cuda": torch.cuda.synchronize()
            elif A.device.type == "mps": torch.mps.synchronize()
        elif backend == "mlx":
            import mlx.core as mx
            z_unit = Z / mx.maximum(mx.sqrt(mx.sum(Z * Z, axis=1, keepdims=True)), 1e-12)
            proj = A @ mx.transpose(z_unit)
            p2 = mx.sum(A * A, axis=1, keepdims=True)
            dist = mx.sqrt(mx.maximum(p2 - proj * proj, 0.0)); mx.eval(dist)
        else:
            xp = np
            if backend == "cupy":
                import cupy as cp
                xp = cp
            z_unit = Z / xp.maximum(xp.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            proj = A @ z_unit.T
            p2 = xp.sum(A * A, axis=1, keepdims=True)
            dist = xp.sqrt(xp.maximum(p2 - proj * proj, 0.0))
        return np.asarray(_to_numpy(dist), dtype=np.float64)
    except Exception:
        z_norm = np.maximum(np.linalg.norm(Z32, axis=1, keepdims=True), 1e-12)
        z_unit = Z32 / z_norm
        proj = F32 @ z_unit.T
        return np.sqrt(np.maximum(np.sum(F32 * F32, axis=1, keepdims=True) - proj * proj, 0.0)).astype(float)


class HyperplaneNormalization:
    """Persistent ideal/extreme/nadir state matching canonical NSGA-III survival."""
    def __init__(self, n_dim: int):
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.nadir_point: np.ndarray | None = None
        self.extreme_points: np.ndarray | None = None

    def update(self, F: np.ndarray, nds: np.ndarray | None = None) -> None:
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)
        if nds is None:
            nds = np.arange(len(F))
        self.extreme_points = _get_extreme_points(F[nds], self.ideal_point, self.extreme_points)
        self.nadir_point = _get_nadir_point(
            self.extreme_points, self.ideal_point, self.worst_point,
            np.max(F[nds], axis=0), np.max(F, axis=0),
        )


def _get_extreme_points(F: np.ndarray, ideal: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    work = F if previous is None else np.concatenate([previous, F], axis=0)
    shifted = work - ideal
    shifted[shifted < 1e-3] = 0.0
    weights = np.eye(F.shape[1]); weights[weights == 0] = 1e6
    asf = np.max(shifted[None, :, :] * weights[:, None, :], axis=2)
    return work[np.argmin(asf, axis=1)]


def _get_nadir_point(extreme: np.ndarray, ideal: np.ndarray, worst: np.ndarray,
                     worst_front: np.ndarray, worst_population: np.ndarray) -> np.ndarray:
    try:
        M = extreme - ideal
        b = np.ones(M.shape[1])
        plane = np.linalg.solve(M, b)
        intercepts = 1.0 / plane
        if not np.allclose(M @ plane, b) or np.any(intercepts <= 1e-6):
            raise np.linalg.LinAlgError()
        nadir = ideal + intercepts
        mask = nadir > worst
        nadir[mask] = worst[mask]
    except np.linalg.LinAlgError:
        nadir = worst_front.copy()
    small = nadir - ideal <= 1e-6
    nadir[small] = worst_population[small]
    return nadir


def _fronts(F: np.ndarray, backend: str, CV: np.ndarray | None = None) -> list[np.ndarray]:
    if boolean_matrix_nds is not None and backend in {"torch", "cupy", "mlx"}:
        try:
            F_dev = _device_array(F, backend)
            CV_dev = None if CV is None else _device_array(np.asarray(CV).reshape(-1, 1), backend)
            return boolean_matrix_nds(F_dev, CV_dev)
        except Exception:
            pass
    if CV is not None:
        from operators.utility_functions.NDSort import NDSort
        front_no, max_no = NDSort(F, np.asarray(CV).reshape(-1, 1), len(F))
        return [np.sort(np.where(front_no == rank)[0]) for rank in range(1, int(max_no) + 1)]
    from util.nds.non_dominated_sorting import NonDominatedSorting
    return [np.sort(np.asarray(front, dtype=int)) for front in NonDominatedSorting().do(F)]


def _niching(last: np.ndarray, n_remaining: int, niche_count: np.ndarray,
             niche: np.ndarray, dist: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected: list[int] = []
    mask = np.ones(len(last), dtype=bool)
    while len(selected) < n_remaining:
        avail_niches = np.unique(niche[mask])
        counts = niche_count[avail_niches]
        candidates = avail_niches[counts == counts.min()]
        candidates = candidates[rng.permutation(len(candidates))[: n_remaining - len(selected)]]
        for target in candidates:
            idx = np.where((niche == target) & mask)[0]
            rng.shuffle(idx)
            pick = int(idx[np.argmin(dist[idx])]) if niche_count[target] == 0 else int(idx[0])
            mask[pick] = False
            selected.append(pick)
            niche_count[target] += 1
    return np.asarray(selected, dtype=int)


def _last_selection(pop_obj_1: np.ndarray, pop_obj_2: np.ndarray, k: int,
                    ref_dirs: np.ndarray, zmin: np.ndarray,
                    rng: np.random.Generator, norm: HyperplaneNormalization | None = None,
                    backend: str = "numpy") -> np.ndarray:
    """Compatibility wrapper returning a boolean mask over the critical front."""
    n2 = len(pop_obj_2)
    choose = np.zeros(n2, dtype=bool)
    if n2 == 0 or k <= 0:
        return choose
    all_F = np.vstack([pop_obj_1, pop_obj_2])
    if norm is None:
        norm = HyperplaneNormalization(all_F.shape[1])
        norm.ideal_point = np.asarray(zmin, dtype=float).copy()
        norm.worst_point = np.max(all_F, axis=0)
    norm.update(all_F, np.arange(len(all_F)))
    denom = norm.nadir_point - norm.ideal_point
    denom[denom == 0] = 1e-12
    normalized = (all_F - norm.ideal_point) / denom
    D = _perpendicular_distance(normalized, ref_dirs, backend)
    niche = np.argmin(D, axis=1)
    dist = D[np.arange(len(all_F)), niche]
    n1 = len(pop_obj_1)
    count = np.bincount(niche[:n1], minlength=len(ref_dirs))
    local = _niching(np.arange(n2), int(k), count, niche[n1:], dist[n1:], rng)
    choose[local] = True
    return choose


def _environmental_selection(pop: Population, n_survive: int, ref_dirs: np.ndarray,
                             zmin: np.ndarray, rng: np.random.Generator,
                             norm: HyperplaneNormalization | None = None,
                             backend: str = "numpy") -> Population:
    n_survive = int(max(1, n_survive))
    F = _population_objectives(pop)
    CV = _constraint_violation(pop)
    fronts = _fronts(F, backend, CV if np.any(CV > 0.0) else None)
    if not fronts:
        return pop[:0]
    first, last = fronts[0], fronts[-1]
    if norm is None:
        norm = HyperplaneNormalization(F.shape[1])
        norm.ideal_point = np.asarray(zmin, dtype=float).copy()
    norm.update(F, first)
    selected_fronts: list[np.ndarray] = []
    count = 0
    for front in fronts:
        if count + len(front) >= n_survive:
            last = front
            break
        selected_fronts.append(front)
        count += len(front)
    until = np.concatenate(selected_fronts) if selected_fronts else np.empty(0, dtype=int)
    if count == n_survive:
        return pop[until]
    I = np.concatenate([*selected_fronts, last]) if selected_fronts else np.asarray(last)
    F_use = F[I]
    rank_fronts = []
    offset = 0
    for front in [*selected_fronts, last]:
        rank_fronts.append(np.arange(offset, offset + len(front)))
        offset += len(front)
    last_local = rank_fronts[-1]
    until_local = np.concatenate(rank_fronts[:-1]) if len(rank_fronts) > 1 else np.empty(0, dtype=int)
    denom = norm.nadir_point - norm.ideal_point
    denom[denom == 0] = 1e-12
    normalized = (F_use - norm.ideal_point) / denom
    D = _perpendicular_distance(normalized, ref_dirs, backend)
    niche = np.argmin(D, axis=1)
    dist = D[np.arange(len(F_use)), niche]
    niche_count = np.bincount(niche[until_local], minlength=len(ref_dirs))
    picks = _niching(last_local, n_survive - len(until_local), niche_count,
                     niche[last_local], dist[last_local], rng)
    survivors_local = np.concatenate([until_local, last_local[picks]])
    return pop[I[survivors_local]]


def associate_to_niches(
    F: np.ndarray,
    ref_dirs: np.ndarray,
    ideal_point: np.ndarray | None = None,
    nadir_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associates objective vectors to reference directions/niches."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    N, M = F.shape

    ref_dirs = np.asarray(ref_dirs, dtype=float)
    if ref_dirs.ndim == 1:
        ref_dirs = ref_dirs.reshape(1, -1)

    if ideal_point is None:
        ideal_point = np.min(F, axis=0)
    else:
        ideal_point = np.asarray(ideal_point, dtype=float).reshape(-1)

    if nadir_point is None:
        nadir_point = np.max(F, axis=0)
    else:
        nadir_point = np.asarray(nadir_point, dtype=float).reshape(-1)

    # Shift and normalize
    F_shifted = F - ideal_point[None, :]
    span = np.maximum(nadir_point - ideal_point, 1e-12)
    F_norm = F_shifted / span[None, :]

    # Perpendicular distance to reference directions
    # Distance from point p to line through origin along ref_dir w:
    # d_perp = || p - (p . w / ||w||^2) w ||
    ref_norms = np.linalg.norm(ref_dirs, axis=1)
    ref_norms = np.maximum(ref_norms, 1e-12)
    w_unit = ref_dirs / ref_norms[:, None]

    # Projections: (N, n_ref)
    projections = F_norm @ w_unit.T  # (N, n_ref)
    
    # Distance matrix: (N, n_ref)
    # ||p||^2 - (p.w_unit)^2
    p_norms_sq = np.sum(F_norm ** 2, axis=1, keepdims=True)  # (N, 1)
    dist_sq = np.maximum(0.0, p_norms_sq - projections ** 2)
    distances = np.sqrt(dist_sq)

    closest_niche = np.argmin(distances, axis=1)
    closest_dist = np.min(distances, axis=1)

    return closest_niche, closest_dist, distances


class ReferenceDirectionSurvival(Survival):
    """Reference Direction-based environmental selection for NSGA-III."""

    def __init__(self, ref_dirs: np.ndarray, filter_infeasible: bool = True) -> None:
        super().__init__(filter_infeasible=filter_infeasible)
        self.ref_dirs = np.asarray(ref_dirs, dtype=float)
        self.nds = NonDominatedSorting()

    def _do(
        self,
        problem: Any,
        pop: Population,
        *args: Any,
        n_survive: int | None = None,
        random_state=None,
        **kwargs: Any,
    ) -> Population:
        if n_survive is None or len(pop) <= n_survive:
            return pop

        F = np.asarray(pop.get("F"), dtype=float)
        N, M = F.shape

        fronts = self.nds.do(F)
        survivors = []
        last_front = None

        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front)
            else:
                last_front = front
                break

        if len(survivors) == n_survive or last_front is None:
            return pop[np.array(survivors, dtype=int)]

        remaining = n_survive - len(survivors)
        chosen_from_last = self._select_from_last_front(
            F, survivors, last_front, remaining, random_state=random_state
        )
        survivors.extend(chosen_from_last)
        return pop[np.array(survivors, dtype=int)]

    def _select_from_last_front(
        self,
        F: np.ndarray,
        survivor_indices: list[int],
        last_front: np.ndarray,
        n_needed: int,
        random_state=None,
    ) -> list[int]:
        rng = random_state if random_state is not None else np.random.default_rng()
        all_candidates = list(survivor_indices) + list(last_front)
        F_sub = F[all_candidates]
        ideal = np.min(F_sub, axis=0)
        nadir = np.max(F_sub, axis=0)

        niche_of_ind, dist_to_niche, _ = associate_to_niches(F, self.ref_dirs, ideal, nadir)

        # Count niche counts in current survivors
        n_ref = len(self.ref_dirs)
        niche_counts = np.zeros(n_ref, dtype=int)
        for idx in survivor_indices:
            niche_counts[niche_of_ind[idx]] += 1

        chosen = []
        available_last = list(last_front)

        while len(chosen) < n_needed and len(available_last) > 0:
            min_count = np.min(niche_counts)
            candidate_niches = np.where(niche_counts == min_count)[0]
            target_niche = int(rng.choice(candidate_niches))

            # Find individuals in available_last associated with target_niche
            matching = [idx for idx in available_last if niche_of_ind[idx] == target_niche]

            if len(matching) == 0:
                niche_counts[target_niche] = 1000000000
                continue

            if niche_counts[target_niche] == 0:
                # Pick individual with minimum distance
                dists = [dist_to_niche[idx] for idx in matching]
                best_idx = matching[int(np.argmin(dists))]
            else:
                best_idx = int(rng.choice(matching))

            chosen.append(best_idx)
            available_last.remove(best_idx)
            niche_counts[target_niche] += 1

        # Fallback if any remaining
        while len(chosen) < n_needed and len(available_last) > 0:
            pick = available_last.pop(0)
            chosen.append(pick)

        return chosen


class NSGA3(Algorithm):
    """Canonical NSGA-III solver with accelerator-backed survival kernels."""
    ALGO_FLAGS = {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"}
    OBJECTIVE_SCOPE = "many"
    def __init__(self, pop_size: int = 100, ref_dirs=None, sampling=None,
                 seed=None, use_gpu: bool = False, array_backend: str = "auto", **kwargs: Any) -> None:
        super().__init__(seed=seed, use_gpu=use_gpu, array_backend=array_backend, **kwargs)
        self.pop_size = int(max(2, pop_size))
        self.ref_dirs = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.sampling = LatinHypercubeSampling() if sampling is None else sampling
        self.norm: HyperplaneNormalization | None = None
        self.zmin: np.ndarray | None = None
    def _setup(self, problem, **kwargs):
        if self.ref_dirs is None or self.ref_dirs.ndim != 2 or self.ref_dirs.shape[1] != int(problem.n_obj):
            self.ref_dirs, n_eff = UniformPoint(self.pop_size, int(problem.n_obj))
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)
            self.pop_size = int(n_eff)
        if self.pop_size < len(self.ref_dirs):
            warnings.warn("pop_size is smaller than the number of reference directions", RuntimeWarning)
        self.norm = HyperplaneNormalization(int(problem.n_obj))

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills if infills is not None else Population.empty()
        if len(self.pop) and self.norm is not None:
            F = _population_objectives(self.pop)
            self.norm.update(F, _fronts(F, self.array_backend_effective)[0])
            self.zmin = np.asarray(self.norm.ideal_point, dtype=float).copy()
        self._set_optimum()

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return self._initialize_infill()
        rng = rng_from_algo(self)
        cv = _constraint_violation(self.pop)
        # Canonical comparator: infeasible -> lower CV; feasible -> random.
        parents = np.empty(self.pop_size, dtype=int)
        draws = rng.integers(0, len(self.pop), size=(2, self.pop_size))
        for j in range(self.pop_size):
            a, b = int(draws[0, j]), int(draws[1, j])
            if cv[a] > 0 or cv[b] > 0:
                parents[j] = a if cv[a] < cv[b] else b if cv[b] < cv[a] else int(rng.choice([a, b]))
            else:
                parents[j] = int(rng.choice([a, b]))
        # Community defaults: SBX eta=30/prob=1, PM eta=20.
        return OperatorGA(self.problem, self.pop[parents], Parameter=[1.0, 30.0, 1.0, 20.0], rng=rng)

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return
        merged = Population.merge(self.pop, infills)
        self.pop = _environmental_selection(
            merged, self.pop_size, self.ref_dirs,
            self.norm.ideal_point if self.norm is not None else np.min(_population_objectives(merged), axis=0),
            rng_from_algo(self), norm=self.norm, backend=self.array_backend_effective,
        )
        self.zmin = np.asarray(self.norm.ideal_point, dtype=float).copy() if self.norm is not None else None
        self._set_optimum()

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True) if self.pop is not None else None


NSGAIII = NSGA3

ALGORITHMS = {
    "NSGA-III": NSGA3,
}

__all__ = [
    "NSGA3",
    "NSGAIII",
    "ALGORITHM_FLAGS",
    "ALGORITHMS",
    "HyperplaneNormalization",
    "_environmental_selection",
    "_last_selection",
    "_perpendicular_distance",
    "_fronts",
]
