# emopylab 2026
"""FLEA (Fast sampling based evolutionary algorithm).

Reference:
L. Li, C. He, R. Cheng, H. Li, L. Pan, and Y. Jin.
Swarm and Evolutionary Computation, 2022, 75:101181.
"""

from __future__ import annotations

import math
import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from util.optimum import filter_optimum

from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.UniformPoint import UniformPoint
from algorithms.community_utils.moead_family import fe_ratio, pop_F, rng_from_algo, sample_initial


ALGORITHM_FLAGS = {"FLEA": {"multi", "many", "real"}}


def _pairwise(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if metric == "euclidean":
        d2 = np.maximum(np.sum(A * A, axis=1, keepdims=True) + np.sum(B * B, axis=1)[None, :] - 2 * A @ B.T, 0.0)
        return np.sqrt(d2)
    if metric == "chebychev":
        return np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    if metric == "cosine":
        na = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-32)
        nb = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-32)
        c = (A @ B.T) / (na * nb.T)
        return 1.0 - np.clip(c, -1.0, 1.0)
    raise ValueError(metric)


def _nsga2_environmental_selection(pop: Population, N: int) -> Population:
    F = np.asarray(pop.get("F"), dtype=float)
    G = pop.get("G")
    if G is None:
        front, maxf = NDSort(F, N)
    else:
        front, maxf = NDSort(F, np.asarray(G, dtype=float), N)
    front = np.asarray(front, dtype=float)
    next_mask = front < maxf
    crowd = np.asarray(CrowdingDistance(F, front), dtype=float)
    last = np.where(front == float(maxf))[0]
    need = int(N - np.sum(next_mask))
    if need > 0 and last.size:
        rank = np.argsort(-crowd[last])
        next_mask[last[rank[:need]]] = True
    return pop[np.where(next_mask)[0]]


def _ref_selection(problem, pop: Population, RN: int, theta: float) -> Population:
    V, _ = UniformPoint(RN, problem.n_obj)
    V = np.asarray(V, dtype=float)
    F = np.asarray(pop.get("F"), dtype=float)
    V = V * (np.max(F, axis=0) - np.min(F, axis=0) + np.finfo(float).eps)

    N, M = F.shape
    NV = V.shape[0]
    PopObj = F - np.min(F, axis=0)

    cosine = 1.0 - _pairwise(V, V, "cosine")
    np.fill_diagonal(cosine, 0.0)
    gamma = np.min(np.arccos(np.clip(cosine, -1.0, 1.0)), axis=1)
    gamma = np.maximum(gamma, 1e-12)

    angle = np.arccos(np.clip(1.0 - _pairwise(PopObj, V, "cosine"), -1.0, 1.0))
    chosen = np.zeros(NV, dtype=int)
    for i in range(NV):
        APD = (1.0 + M * theta * angle[:, i] / gamma[i]) * np.sqrt(np.sum(PopObj**2, axis=1))
        chosen[i] = int(np.argmin(APD))
        angle[chosen[i], :] = np.inf
    Ref = pop[chosen]
    order = np.argsort(np.asarray(Ref.get("F"), dtype=float)[:, 0])
    return Ref[order]


def _neighborhood_association(pop: Population, ref: Population):
    RefObj = np.asarray(ref.get("F"), dtype=float)
    PopObj = np.asarray(pop.get("F"), dtype=float)
    OCid = np.argmin(_pairwise(RefObj, PopObj, "chebychev"), axis=0)
    RefDec = np.asarray(ref.get("X"), dtype=float)
    PopDec = np.asarray(pop.get("X"), dtype=float)
    DCid = np.argmin(_pairwise(RefDec, PopDec, "chebychev"), axis=0)
    return OCid, DCid


def _direction_calculation(problem, pop: Population, ref: Population, ids: np.ndarray, front_no: np.ndarray, rng) -> np.ndarray:
    RefDec = np.asarray(ref.get("X"), dtype=float)
    RN = len(ref)
    dirV = np.zeros_like(RefDec)
    PopDec = np.asarray(pop.get("X"), dtype=float)
    for i in range(RN):
        idx = np.where(ids == i)[0]
        if idx.size:
            fvals = front_no[idx]
            minf = np.min(fvals)
            if minf > 1:
                dirV[i] = np.mean(PopDec[idx[fvals == minf]], axis=0) - RefDec[i]
                continue
        if rng.random() > 0.5:
            dirV[i] = (RefDec[i] - np.asarray(problem.xl, dtype=float)) / max(problem.n_var, 1)
        else:
            dirV[i] = (np.asarray(problem.xu, dtype=float) - RefDec[i]) / max(problem.n_var, 1)
    return dirV


def _reproduction(problem, ref: Population, cm: float, dirV_obj: np.ndarray, dirV_dec: np.ndarray, total_pop_n: int, rng):
    RefDec = np.asarray(ref.get("X"), dtype=float)
    RN = len(ref)
    CN = int(np.ceil(cm * 0.5 * total_pop_n / max(RN, 1)))
    DN = max(0, total_pop_n - CN * RN * 2)
    OffDec = np.zeros((CN * RN * 2 + DN, problem.n_var), dtype=float)

    for i in range(RN):
        mu = np.repeat(RefDec[i][None, :], CN, axis=0)
        OffDec[i * CN : (i + 1) * CN, :] = mu + rng.standard_normal((CN, 1)) * np.repeat(dirV_obj[i][None, :], CN, axis=0)
        s = CN * RN + i * CN
        OffDec[s : s + CN, :] = mu + rng.standard_normal((CN, 1)) * np.repeat(dirV_dec[i][None, :], CN, axis=0)

    if DN > 0:
        x1 = rng.integers(0, RN, size=DN)
        x2 = rng.integers(0, RN, size=DN)
        dirV = RefDec[x1] - RefDec[x2]
        OffDec[-DN:, :] = RefDec[x1] + rng.standard_normal((DN, 1)) * dirV

    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    OffDec = np.clip(OffDec, xl, xu)
    return Population.new("X", OffDec)


class FLEA(Algorithm):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills

    def _infill(self):
        rng = rng_from_algo(self)
        F = np.asarray(self.pop.get("F"), dtype=float)
        G = self.pop.get("G")
        if G is None:
            front_no, _ = NDSort(F, np.inf)
        else:
            front_no, _ = NDSort(F, np.asarray(G, dtype=float), np.inf)
        front_no = np.asarray(front_no, dtype=float)

        RN = int(np.ceil(np.sqrt(max(len(self.pop), 1))))
        ref = _ref_selection(self.problem, self.pop, RN, fe_ratio(self) ** 2)
        id_obj, id_dec = _neighborhood_association(self.pop, ref)
        cm = 0.1 * math.floor(10 * fe_ratio(self))
        dirV_obj = _direction_calculation(self.problem, self.pop, ref, id_obj, front_no, rng)
        dirV_dec = _direction_calculation(self.problem, self.pop, ref, id_dec, front_no, rng)
        return _reproduction(self.problem, ref, cm, dirV_obj, dirV_dec, self.pop_size, rng)

    def _advance(self, infills=None, **kwargs):
        self.pop = _nsga2_environmental_selection(Population.merge(self.pop, infills), self.pop_size)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

