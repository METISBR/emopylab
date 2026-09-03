# emopylab 2026
"""Shared helpers for local MOEA/D-family algorithm ports in EmoPyLab."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from core.population import Population
from operators.sampling.rnd import (
    BinaryRandomSampling,
    FloatRandomSampling,
    IntegerRandomSampling,
)
from util.optimum import filter_optimum

from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorDE import OperatorDE
from operators.utility_functions.OperatorGAhalf import OperatorGAhalf
from operators.utility_functions.RouletteWheelSelection import RouletteWheelSelection
from operators.utility_functions.TournamentSelection import TournamentSelection
from operators.utility_functions.UniformPoint import UniformPoint


@dataclass
class Task:
    i: int
    parents_pool: np.ndarray
    ns_idx: int | None = None


def rng_from_algo(algo):
    rng = getattr(algo, "random_state", None)
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        rng = np.random.default_rng()
        algo.random_state = rng
        return rng
    return np.random.default_rng(int(rng))


def default_sampling(problem):
    vtype = getattr(problem, "vtype", float)
    if vtype in (bool, np.bool_):
        return BinaryRandomSampling()
    if vtype in (int, np.int32, np.int64):
        return IntegerRandomSampling()
    return FloatRandomSampling()


def sample_initial(problem, n, sampling=None, rng=None):
    sampling = default_sampling(problem) if sampling is None else sampling
    return sampling.do(problem, int(n), random_state=rng)


def pop_F(pop: Population) -> np.ndarray:
    return np.asarray(pop.get("F"), dtype=float)


def pop_G(pop: Population) -> np.ndarray:
    g = pop.get("G")
    if g is None:
        return np.zeros((len(pop), 0), dtype=float)
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    return g


def ind_F(ind) -> np.ndarray:
    return np.asarray(ind.get("F") if hasattr(ind, "get") else ind.F, dtype=float).reshape(-1)


def ind_G(ind) -> np.ndarray:
    g = ind.get("G") if hasattr(ind, "get") else getattr(ind, "G", None)
    if g is None:
        return np.zeros(0, dtype=float)
    return np.asarray(g, dtype=float).reshape(-1)


def cv_from_cons(cons: np.ndarray) -> float:
    if cons.size == 0:
        return 0.0
    return float(np.sum(np.maximum(0.0, cons)))


def population_cv(pop: Population) -> np.ndarray:
    cv = pop.get("CV")
    if cv is not None:
        arr = np.asarray(cv, dtype=float).reshape(-1)
        if arr.size == len(pop):
            return np.maximum(arr, 0.0)
    g = pop_G(pop)
    if g.size == 0:
        return np.zeros(len(pop), dtype=float)
    return np.sum(np.maximum(0.0, g), axis=1)


def weight_vectors(n_target: int, n_obj: int) -> tuple[np.ndarray, int]:
    W, n_eff = UniformPoint(int(n_target), int(n_obj))
    W = np.asarray(W, dtype=float)
    return np.maximum(W, 1e-12), int(n_eff)


def neighbors(W: np.ndarray, T: int) -> np.ndarray:
    diff = W[:, None, :] - W[None, :, :]
    D = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    order = np.argsort(D, axis=1)
    return order[:, : max(1, min(int(T), W.shape[0]))]


def boundary_indices(W: np.ndarray) -> np.ndarray:
    M = W.shape[1]
    return np.where(np.sum(W < 1e-3, axis=1) == M - 1)[0]


def choose_parent_pool(i: int, B: np.ndarray, N: int, rng, delta: float = 0.9, local_size: int | None = None) -> np.ndarray:
    if rng.random() < float(delta):
        k = B.shape[1] if local_size is None else min(int(local_size), B.shape[1])
        return B[i, rng.permutation(k)]
    return rng.permutation(N)


def de_offspring(problem, pop: Population, i: int, P: np.ndarray, rng):
    off = OperatorDE(problem, pop[[i]], pop[[int(P[0])]], pop[[int(P[1])]], rng=rng)
    return off[0] if isinstance(off, Population) else off


def gahalf_offspring(problem, pop: Population, p1: int, p2: int, rng):
    off = OperatorGAhalf(problem, pop[[int(p1), int(p2)]], rng=rng)
    return off[0] if isinstance(off, Population) else off


def tchebycheff_values(F: np.ndarray, Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    return np.max(np.abs(F - Z) * W, axis=1)


def pbi_values(F: np.ndarray, Z: np.ndarray, W: np.ndarray, theta: float = 5.0) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    V = F - Z
    normW = np.maximum(np.linalg.norm(W, axis=1), 1e-32)
    normV = np.maximum(np.linalg.norm(V, axis=1), 1e-32)
    cosine = np.sum(V * W, axis=1) / (normW * normV)
    cosine = np.clip(cosine, -1.0, 1.0)
    return normV * cosine + theta * normV * np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))


def normalized_tchebycheff_values(F: np.ndarray, Z: np.ndarray, Zmax: np.ndarray, W: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    den = np.maximum(Zmax - Z, 1e-32)
    return np.max(np.abs(F - Z) / den / np.maximum(W, 1e-12), axis=1)


def normalize_du(PopObj: np.ndarray, z: np.ndarray, znad: np.ndarray):
    PopObj = np.asarray(PopObj, dtype=float)
    N, M = PopObj.shape
    z = np.minimum(np.asarray(z, dtype=float), np.min(PopObj, axis=0))

    W = np.full((M, M), 1e-6, dtype=float)
    np.fill_diagonal(W, 1.0)

    den0 = np.maximum(np.asarray(znad, dtype=float) - z, 1e-32)
    ASF = np.zeros((N, M), dtype=float)
    for i in range(M):
        ASF[:, i] = np.max(np.abs((PopObj - z) / den0) / W[i], axis=1)
    extreme = np.argmin(ASF, axis=0)
    try:
        hyperplane = np.linalg.solve(PopObj[extreme, :] - z, np.ones(M))
        a = 1.0 / hyperplane + z
        if np.any(~np.isfinite(a)) or np.any(a <= z):
            a = np.max(PopObj, axis=0)
    except Exception:
        a = np.max(PopObj, axis=0)
    znad = a
    den = np.maximum(znad - z, 1e-32)
    return (PopObj - z) / den, z, znad


def set_weight_dcwv(W: np.ndarray, p: float):
    W = np.asarray(W, dtype=float).copy()
    if 0.0 <= p <= 1.0:
        W0 = np.empty((0, W.shape[1]), dtype=float)
        M = W.shape[1]
        TF = W < 1.0 / M
        W[TF] = W[TF] * p * M
        W[~TF] = 1.0 - (1.0 - W[~TF]) * (1.0 - p) * M / max(M - 1, 1)
    else:
        W0 = W.copy()
    return np.maximum(W, 1e-12), W0


def update_weight_dcwv(objs: np.ndarray, W0: np.ndarray):
    W = np.asarray(W0, dtype=float).copy()
    objs = np.asarray(objs, dtype=float)
    if objs.size == 0:
        return np.maximum(W, 1e-12)
    M = objs.shape[1]
    omin, omax = np.min(objs, axis=0), np.max(objs, axis=0)
    nobj = (objs - omin) / np.maximum(omax - omin, 1e-32)
    normP = np.maximum(np.linalg.norm(nobj, axis=1), 1e-32)
    cosineP = np.sum(nobj / M, axis=1) * np.sqrt(M) / normP
    idx = int(np.argmin(normP * np.sqrt(np.maximum(0.0, 1.0 - cosineP * cosineP))))
    p = normP[idx] * cosineP[idx] / np.sqrt(M)
    TF = W < 1.0 / M
    W[TF] = W[TF] * p * M
    W[~TF] = 1.0 - (1.0 - W[~TF]) * (1.0 - p) * M / max(M - 1, 1)
    return np.maximum(W, 1e-12)


def current_fe(algo) -> int:
    return int(getattr(getattr(algo, "evaluator", None), "n_eval", 0) or 0)


def max_fe(algo) -> int:
    term = getattr(algo, "termination", None)
    for attr in ("n_max_evals", "n_max_eval", "max_evals"):
        v = getattr(term, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return max(1, int(getattr(algo, "pop_size", 100) * 100))


def fe_ratio(algo) -> float:
    return min(1.0, current_fe(algo) / max(max_fe(algo), 1))


def update_pi_dra(PopF: np.ndarray, W: np.ndarray, Z: np.ndarray, Pi: np.ndarray, old_obj: np.ndarray):
    new_obj = tchebycheff_values(PopF, Z, W)
    den = np.where(np.abs(old_obj) <= 1e-32, 1.0, old_obj)
    delta = (old_obj - new_obj) / den
    temp = delta < 0.001
    Pi = np.asarray(Pi, dtype=float).copy()
    Pi[~temp] = 1.0
    Pi[temp] = (0.95 + 0.05 * delta[temp] / 0.001) * Pi[temp]
    return Pi, new_obj


def choose_dra_indices(W: np.ndarray, Pi: np.ndarray, rng, N: int) -> np.ndarray:
    bnd = boundary_indices(W)
    target = max(1, int(math.floor(N / 5)) - len(bnd))
    if target > 0:
        rest = np.asarray(TournamentSelection(10, target, -np.asarray(Pi, dtype=float), rng=rng), dtype=int) - 1
        I = np.concatenate([bnd, rest])
    else:
        I = bnd
    if I.size == 0:
        I = np.arange(N)
    return I.astype(int)


def ensure_population(offspring_list) -> Population:
    if not offspring_list:
        return Population.empty()
    pop = Population.empty()
    for off in offspring_list:
        if isinstance(off, Population):
            pop = Population.merge(pop, off)
        else:
            pop = Population.merge(pop, Population.create(off))
    return pop


def set_optimum_from_pop(algo):
    algo.opt = filter_optimum(algo.pop, least_infeasible=True)

