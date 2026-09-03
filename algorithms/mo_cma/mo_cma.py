# emopylab 2026
"""MO-CMA (multi-objective covariance matrix adaptation evolution strategy).

Reference:
C. Igel, N. Hansen, and S. Roth. Covariance matrix adaptation for
multi-objective optimization. Evolutionary Computation, 2007, 15(1): 1-28.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.algorithm import Algorithm
from core.population import Population
from util.optimum import filter_optimum

from algorithms.community_utils.moead_family import rng_from_algo, sample_initial
from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort


ALGORITHM_FLAGS = {"MOCMA": {"multi", "real", "integer"}}


def _rank_by_front_and_crowding(front_no: np.ndarray, crowd_dis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Local MATLAB-style ordering used by MO-CMA:
    sortrows([FrontNo, -CrowdingDistance]) -> lower front first, then higher crowding.
    """
    front = np.asarray(front_no, dtype=float).reshape(-1)
    crowd = np.asarray(crowd_dis, dtype=float).reshape(-1)
    order = np.lexsort((-crowd, front))
    fitness = np.empty(order.size, dtype=int)
    fitness[order] = np.arange(order.size, dtype=int)
    return order, fitness


@dataclass
class _MOCMANode:
    x: np.ndarray
    psucc: float
    sigma: float
    pc: np.ndarray
    C: np.ndarray

    @classmethod
    def create(cls, x: np.ndarray, n_var: int, ptarget: float, sigma: float) -> "_MOCMANode":
        return cls(
            x=np.asarray(x, dtype=float).copy(),
            psucc=float(ptarget),
            sigma=float(sigma),
            pc=np.zeros(int(n_var), dtype=float),
            C=np.eye(int(n_var), dtype=float),
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        n_var = int(self.x.size)
        Csym = 0.5 * (self.C + self.C.T)

        try:
            vals, vecs = np.linalg.eigh(Csym)
            vals = np.maximum(np.asarray(vals, dtype=float), 1e-16)
            z = rng.standard_normal(n_var)
            step = vecs @ (np.sqrt(vals) * z)
        except Exception:
            step = rng.standard_normal(n_var)
            self.C = np.eye(n_var, dtype=float)

        return self.x + self.sigma * step

    def update_step_size(
        self,
        is_success: bool,
        ptarget: float,
        min_sigma: float,
        max_sigma: float,
    ) -> None:
        cp = ptarget / (2.0 + ptarget)
        d_damp = 1.0 + self.x.size / 2.0

        self.psucc = (1.0 - cp) * self.psucc + cp * float(is_success)
        sigma_new = self.sigma * math.exp((self.psucc - ptarget) / (d_damp * (1.0 - ptarget)))
        self.sigma = float(min(max(sigma_new, min_sigma), max_sigma))

    def update_covariance(self, xstep: np.ndarray) -> None:
        n_var = float(self.x.size)
        cc = 2.0 / (n_var + 2.0)
        ccov = 2.0 / (n_var**2 + 6.0)

        if self.psucc < 0.44:
            self.pc = (1.0 - cc) * self.pc + math.sqrt(cc * (2.0 - cc)) * np.asarray(xstep, dtype=float)
            self.C = (1.0 - ccov) * self.C + ccov * np.outer(self.pc, self.pc)
        else:
            self.pc = (1.0 - cc) * self.pc
            self.C = (1.0 - ccov) * self.C + ccov * (
                np.outer(self.pc, self.pc) + cc * (2.0 - cc) * self.C
            )

        self.C = 0.5 * (self.C + self.C.T)
        if not np.all(np.isfinite(self.C)):
            self.C = np.eye(self.x.size, dtype=float)
            return

        # Keep covariance positive definite enough for stable eigendecomposition.
        min_diag = float(np.min(np.diag(self.C)))
        if min_diag <= 0.0:
            self.C = self.C + (abs(min_diag) + 1e-12) * np.eye(self.x.size, dtype=float)


class MOCMA(Algorithm):
    """Local MO-CMA implementation compatible with EmoPyLab."""

    ALGO_FLAGS = {"multi", "real", "integer"}
    OBJECTIVE_SCOPE = "multi"

    def __init__(
        self,
        pop_size: int = 100,
        ptarget: float = 1.0 / 5.5,
        sigma0: float = 0.5,
        min_sigma: float = 1e-12,
        max_sigma: float = 1e6,
        sampling: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.ptarget = float(min(max(ptarget, 1e-8), 1.0 - 1e-8))
        self.sigma0 = float(sigma0)
        self.min_sigma = float(max(1e-16, min_sigma))
        self.max_sigma = float(max(self.min_sigma, max_sigma))
        self.sampling = sampling

        self.nodes: list[_MOCMANode] = []
        self._child_raw: np.ndarray | None = None

    def _repair_candidate(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float).copy()

        has_bounds = hasattr(self.problem, "has_bounds") and callable(self.problem.has_bounds)
        if has_bounds and bool(self.problem.has_bounds()):
            out = np.clip(out, self.problem.xl, self.problem.xu)

        vtype = getattr(self.problem, "vtype", float)
        int_types = (int, np.int32, np.int64)
        bool_types = (bool, np.bool_)

        if isinstance(vtype, np.ndarray) or isinstance(vtype, (list, tuple)):
            vtypes = np.asarray(vtype, dtype=object).reshape(-1)
            if vtypes.size == 1:
                vtype = vtypes[0]
            else:
                upper = min(vtypes.size, out.size)
                int_mask = np.zeros(out.size, dtype=bool)
                bool_mask = np.zeros(out.size, dtype=bool)
                for i in range(upper):
                    token = vtypes[i]
                    int_mask[i] = token in int_types
                    bool_mask[i] = token in bool_types
                if np.any(int_mask):
                    out[int_mask] = np.rint(out[int_mask])
                if np.any(bool_mask):
                    out[bool_mask] = (out[bool_mask] >= 0.5).astype(float)
                return out

        if vtype in int_types:
            out = np.rint(out)
        elif vtype in bool_types:
            out = (out >= 0.5).astype(float)
        return out

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, rng_from_algo(self))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        X = np.asarray(self.pop.get("X"), dtype=float)
        n_var = int(self.problem.n_var)
        sigma0 = max(self.min_sigma, min(self.sigma0, self.max_sigma))
        self.nodes = [_MOCMANode.create(X[i], n_var, self.ptarget, sigma0) for i in range(len(self.pop))]
        self._child_raw = None

    def _infill(self):
        rng = rng_from_algo(self)
        n = len(self.pop)
        n_var = int(self.problem.n_var)

        X_eval = np.zeros((n, n_var), dtype=float)
        X_raw = np.zeros((n, n_var), dtype=float)

        for k, node in enumerate(self.nodes):
            x_raw = node.sample(rng)
            X_raw[k] = x_raw
            X_eval[k] = self._repair_candidate(x_raw)

        self._child_raw = X_raw
        return Population.new("X", X_eval)

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return

        n = len(self.pop)
        merged = Population.merge(self.pop, infills)

        pop_obj = np.asarray(merged.get("F"), dtype=float)
        pop_dec = np.asarray(merged.get("X"), dtype=float)
        parent_x = np.asarray([node.x for node in self.nodes], dtype=float)
        child_raw = (
            np.asarray(self._child_raw, dtype=float)
            if self._child_raw is not None and len(self._child_raw) == n
            else np.asarray(infills.get("X"), dtype=float)
        )

        # MATLAB MO-CMA applies a small penalty for repaired/adjusted offspring.
        x_model = np.vstack([parent_x, child_raw])
        if x_model.shape == pop_dec.shape:
            penalty = 1e-6 * np.sum((x_model - pop_dec) ** 2, axis=1)
            pop_obj = pop_obj + penalty[:, None]

        # PlatEMO MOCMA.m uses unconstrained NDSort on penalized objectives.
        front_no, _ = NDSort(pop_obj, np.inf)
        crowd_dis = np.asarray(CrowdingDistance(pop_obj, front_no), dtype=float).reshape(-1)

        order, fitness = _rank_by_front_and_crowding(front_no, crowd_dis)

        parent_nodes: list[_MOCMANode] = []
        child_nodes: list[_MOCMANode] = []
        for k, parent in enumerate(self.nodes):
            child = copy.deepcopy(parent)
            child.x = np.asarray(child_raw[k], dtype=float).copy()

            success = bool(fitness[n + k] < fitness[k])
            parent.update_step_size(
                success,
                self.ptarget,
                self.min_sigma,
                self.max_sigma,
            )
            child.update_step_size(
                success,
                self.ptarget,
                self.min_sigma,
                self.max_sigma,
            )
            denom = max(parent.sigma, self.min_sigma)
            xstep = (child.x - parent.x) / denom
            child.update_covariance(xstep)

            parent_nodes.append(parent)
            child_nodes.append(child)

        survivors = order[:n]
        self.pop = merged[survivors]

        # Q = [parents, children] in PlatEMO. Keep this same index mapping.
        next_nodes: list[_MOCMANode] = []
        for idx_raw in survivors:
            idx = int(idx_raw)
            if idx < n:
                next_nodes.append(parent_nodes[idx])
            else:
                next_nodes.append(child_nodes[idx - n])
        self.nodes = next_nodes

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)
