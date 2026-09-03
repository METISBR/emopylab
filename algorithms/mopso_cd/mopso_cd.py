# emopylab 2026
"""MOPSO-CD (MOPSO with crowding distance archive management).

Reference:
C. R. Raquel and P. C. Naval Jr. An effective use of crowding distance in
multiobjective particle swarm optimization. GECCO, 2005, 257-264.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from util.optimum import filter_optimum

from algorithms.community_utils.moead_family import rng_from_algo, sample_initial
from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort


ALGORITHM_FLAGS = {"MOPSO_CD": {"multi", "real", "integer"}}


def _row_vector_bounds(x: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(n, float(arr[0]), dtype=float)
    return arr


def _resolve_max_fe(algo: "MOPSOCD") -> int:
    if getattr(algo, "max_fe", None) is not None:
        try:
            parsed = int(algo.max_fe)
            if parsed > 0:
                return parsed
        except Exception:
            pass

    term = getattr(algo, "termination", None)
    for name in ("n_max_evals", "n_max_eval"):
        value = getattr(term, name, None)
        if value is None:
            continue
        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except Exception:
            continue
    return int(1e9)


def _repair_by_vtype(problem: Any, X: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=float).copy()
    vtype = getattr(problem, "vtype", float)

    int_types = (int, np.int32, np.int64)
    bool_types = (bool, np.bool_)

    if isinstance(vtype, np.ndarray) or isinstance(vtype, (list, tuple)):
        vtypes = np.asarray(vtype, dtype=object).reshape(-1)
        if vtypes.size == 1:
            vtype = vtypes[0]
        else:
            upper = min(vtypes.size, out.shape[1])
            int_mask = np.zeros(out.shape[1], dtype=bool)
            bool_mask = np.zeros(out.shape[1], dtype=bool)
            for i in range(upper):
                token = vtypes[i]
                int_mask[i] = token in int_types
                bool_mask[i] = token in bool_types
            if np.any(int_mask):
                out[:, int_mask] = np.rint(out[:, int_mask])
            if np.any(bool_mask):
                out[:, bool_mask] = (out[:, bool_mask] >= 0.5).astype(float)
            return out

    if vtype in int_types:
        out = np.rint(out)
    elif vtype in bool_types:
        out = (out >= 0.5).astype(float)
    return out


def _update_archive(archive: Population, n_archive: int, rng: np.random.Generator) -> Population:
    if len(archive) == 0:
        return Population.empty()

    F = np.asarray(archive.get("F"), dtype=float)
    front_no, _ = NDSort(F, 1)
    nd_idx = np.where(np.asarray(front_no, dtype=float) == 1)[0]
    archive = archive[nd_idx]

    while len(archive) > n_archive:
        f_arc = np.asarray(archive.get("F"), dtype=float)
        crowd = np.asarray(CrowdingDistance(f_arc), dtype=float).reshape(-1)
        order = np.argsort(crowd)  # smaller crowding removed first
        k = max(1, int(math.ceil(order.size * 0.1)))
        drop_local = int(order[int(rng.integers(0, k))])
        keep_mask = np.ones(len(archive), dtype=bool)
        keep_mask[drop_local] = False
        archive = archive[np.where(keep_mask)[0]]

    if len(archive) > 0:
        f_arc = np.asarray(archive.get("F"), dtype=float)
        crowd = np.asarray(CrowdingDistance(f_arc), dtype=float).reshape(-1)
        rank = np.argsort(-crowd)  # descending crowding
        archive = archive[rank]

    return archive


def _update_pbest(pbest: Population, pop: Population) -> Population:
    out = pbest.copy()
    pbest_f = np.asarray(out.get("F"), dtype=float)
    pop_f = np.asarray(pop.get("F"), dtype=float)

    temp = pbest_f - pop_f
    dominate = np.any(temp < 0.0, axis=1).astype(int) - np.any(temp > 0.0, axis=1).astype(int)
    replace = np.where(dominate == -1)[0]
    if replace.size > 0:
        out[replace] = pop[replace]
    return out


class MOPSOCD(Algorithm):
    """Local PlatEMO-style MOPSO-CD implementation."""

    ALGO_FLAGS = {"multi", "real", "integer"}
    OBJECTIVE_SCOPE = "multi"

    def __init__(
        self,
        pop_size: int = 100,
        sampling: Any = None,
        max_fe: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling
        self.max_fe = max_fe

        self.swarm = Population.empty()
        self.archive_pop = Population.empty()
        self.pbest = Population.empty()
        self._max_fe_cache: int | None = None

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.swarm = infills

        n = len(self.swarm)
        d = int(self.problem.n_var)
        if self.swarm.get("V") is None:
            self.swarm.set("V", np.zeros((n, d), dtype=float))

        self.archive_pop = _update_archive(self.swarm, self.pop_size, rng_from_algo(self))
        self.pbest = self.swarm.copy()
        self.pop = self.archive_pop
        self._max_fe_cache = _resolve_max_fe(self)

    def _infill(self):
        if len(self.swarm) == 0:
            return Population.empty()

        rng = rng_from_algo(self)
        n = len(self.swarm)
        d = int(self.problem.n_var)

        particle_dec = np.asarray(self.swarm.get("X"), dtype=float)
        pbest_dec = np.asarray(self.pbest.get("X"), dtype=float)
        particle_vel_raw = self.swarm.get("V")
        if particle_vel_raw is None:
            particle_vel = np.zeros((n, d), dtype=float)
        else:
            particle_vel = np.asarray(particle_vel_raw, dtype=float)
            if particle_vel.shape != (n, d):
                particle_vel = np.zeros((n, d), dtype=float)

        archive_size = len(self.archive_pop)
        if archive_size == 0:
            gbest_dec = pbest_dec
        else:
            k = max(1, int(math.ceil(archive_size * 0.1)))
            pick = rng.integers(0, k, size=n)
            gbest_dec = np.asarray(self.archive_pop.get("X"), dtype=float)[pick]

        w = rng.uniform(0.1, 0.5, size=(n, 1))
        r1 = rng.random((n, 1))
        r2 = rng.random((n, 1))
        c1 = rng.uniform(1.5, 2.5, size=(n, 1))
        c2 = rng.uniform(1.5, 2.5, size=(n, 1))

        off_vel = (
            w * particle_vel
            + c1 * r1 * (pbest_dec - particle_dec)
            + c2 * r2 * (gbest_dec - particle_dec)
        )
        off_dec = particle_dec + off_vel

        xl = _row_vector_bounds(self.problem.xl, d)
        xu = _row_vector_bounds(self.problem.xu, d)
        lower = np.tile(xl[None, :], (n, 1))
        upper = np.tile(xu[None, :], (n, 1))

        repair = (off_dec < lower) | (off_dec > upper)
        off_vel[repair] = -off_vel[repair]
        off_dec = np.minimum(np.maximum(off_dec, lower), upper)

        fe = int(getattr(getattr(self, "evaluator", None), "n_eval", 0) or 0)
        max_fe = int(self._max_fe_cache if self._max_fe_cache is not None else _resolve_max_fe(self))
        if fe <= 0.5 * max_fe:
            dis_m = 20.0
            site = rng.random((n, d)) < (1.0 / max(1, d))
            mu = rng.random((n, d))

            span = upper - lower
            safe = span > 0.0

            left = site & (mu <= 0.5) & safe
            if np.any(left):
                term = (
                    2.0 * mu[left]
                    + (1.0 - 2.0 * mu[left])
                    * (1.0 - (off_dec[left] - lower[left]) / span[left]) ** (dis_m + 1.0)
                )
                off_dec[left] = off_dec[left] + span[left] * (term ** (1.0 / (dis_m + 1.0)) - 1.0)

            right = site & (mu > 0.5) & safe
            if np.any(right):
                term = (
                    2.0 * (1.0 - mu[right])
                    + 2.0 * (mu[right] - 0.5)
                    * (1.0 - (upper[right] - off_dec[right]) / span[right]) ** (dis_m + 1.0)
                )
                off_dec[right] = off_dec[right] + span[right] * (1.0 - term ** (1.0 / (dis_m + 1.0)))

            off_dec = np.minimum(np.maximum(off_dec, lower), upper)

        off_dec = _repair_by_vtype(self.problem, off_dec)
        off_dec = np.minimum(np.maximum(off_dec, lower), upper)

        return Population.new("X", off_dec, "V", off_vel)

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return

        self.swarm = infills
        self.archive_pop = _update_archive(
            Population.merge(self.archive_pop, self.swarm),
            self.pop_size,
            rng_from_algo(self),
        )
        self.pbest = _update_pbest(self.pbest, self.swarm)
        self.pop = self.archive_pop

    def _set_optimum(self):
        source = self.archive_pop if len(self.archive_pop) > 0 else self.swarm
        self.opt = filter_optimum(source, least_infeasible=True)


ALGORITHMS = {
    "MOPSO_CD": MOPSOCD,
}
