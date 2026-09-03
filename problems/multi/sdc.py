from __future__ import annotations

import numpy as np
from core.problem import Problem

from ._qiao_cec2006 import (
    as_2d,
    cec2006_fitness,
    cec2006_information,
    distance_function,
    nd_mask,
    quarter_sphere_points_2d,
    quarter_sphere_points_3d,
    simplex_points_2d,
)


def _information(index: int) -> np.ndarray:
    cec_problem = [1, 2, 3, 6, 9, 10, 11, 12, 14, 18, 19, 24, 15, 5, 1]
    shape_problem = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1]
    b = [10, 100, 15, 115, 19, 125, 10, 100, 15, 115, 19, 115, 125, 15, 10]
    distance_problem = [2, 1, 4, 4, 3, 5, 5, 3, 3, 2, 1, 3, 5, 1, 2]

    HCT = 0.5
    high_type = [1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1]
    DCT = 0.5
    dis_type = [2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2]

    i = int(index) - 1
    if i < 0 or i >= len(cec_problem):
        raise ValueError(f"Unsupported SDC index: {index}")
    return np.array(
        [
            cec_problem[i],
            distance_problem[i],
            HCT,
            high_type[i],
            DCT,
            dis_type[i],
            shape_problem[i],
            b[i],
        ],
        dtype=float,
    )


def _transformation_operator(P: np.ndarray, lu: np.ndarray, high_D_C: int, hct_type: int) -> np.ndarray:
    P = as_2d(P, float)
    n = P.shape[0]
    high_D_C = int(high_D_C)
    lower = lu[0]
    upper = lu[1]
    new_P = np.zeros((n, high_D_C), dtype=float)

    if P.shape[1] > high_D_C:
        chushu = P.shape[1] // high_D_C
        yushu = P.shape[1] % high_D_C
        groups: list[list[int]] = [[] for _ in range(high_D_C)]
        for j in range(high_D_C):
            for m in range(chushu):
                groups[j].append(m * high_D_C + j)
            if yushu >= (j + 1):
                groups[j].append(chushu * high_D_C + j)
        for j in range(high_D_C):
            cols = groups[j]
            if len(cols) > 1:
                PP = P[:, cols]
                a2 = np.mod(np.sum(PP, axis=1), 0.5)
                if int(hct_type) == 1:
                    tempa = -2.0 * a2 + 1.0
                else:
                    tempa = np.cos(a2 * np.pi)
                new_P[:, j] = lower[j] + (upper[j] - lower[j]) * tempa
            elif len(cols) == 1:
                col = cols[0]
                new_P[:, j] = lower[j] + (upper[j] - lower[j]) * P[:, col]
            else:
                new_P[:, j] = lower[j]
    else:
        cols = min(P.shape[1], high_D_C)
        new_P[:, :cols] = lower[:cols][None, :] + (upper[:cols] - lower[:cols])[None, :] * P[:, :cols]
        if cols < high_D_C:
            new_P[:, cols:] = lower[cols:][None, :]
    return new_P


def _deformed_sdc_pf(n: int, shape_problem: int, b: float, need_nd_filter: bool) -> np.ndarray:
    R = simplex_points_2d(max(50, n))
    dn = np.linalg.norm(R, axis=1, keepdims=True)
    dn[dn == 0] = 1.0
    R = R / dn

    if int(shape_problem) == 1:
        cc = float(b) / 10.0
        invalid = cc * R[:, 0] + R[:, 1] < cc
        while np.any(invalid):
            R[invalid] = R[invalid] * 1.001
            invalid = cc * R[:, 0] + R[:, 1] < cc
    else:
        cc = float(b) / 100.0
        theta0 = np.arctan2(R[:, 1], R[:, 0])
        invalid = (cc - 0.2 * np.sin(4.0 * theta0) ** 8) ** 2 - R[:, 0] ** 2 - R[:, 1] ** 2 > 0
        while np.any(invalid):
            R[invalid] = R[invalid] * 1.001
            theta0 = np.arctan2(R[:, 1], R[:, 0])
            invalid = (cc - 0.2 * np.sin(4.0 * theta0) ** 8) ** 2 - R[:, 0] ** 2 - R[:, 1] ** 2 > 0

    theta = np.arctan2(R[:, 1], R[:, 0])
    hx = (1.0 - np.sum(R**2, axis=1)) ** 2
    R = np.column_stack([np.cos(theta) * (1.0 + hx), np.sin(theta) * (1.0 + hx)])

    if need_nd_filter:
        R = R[nd_mask(R)]
    return R


class _SDCBase(Problem):
    PROBLEM_INDEX = 1
    DEFAULT_M = 2

    def __init__(self, n_var: int = 30, **kwargs):
        self.problem_index = int(self.PROBLEM_INDEX)
        m = int(self.DEFAULT_M)
        d = max(3, int(n_var))

        a = _information(self.problem_index)
        self.CEC_Problem = int(a[0])
        self.Distance_problem = int(a[1])
        self.HCT = float(a[2])
        self.HCT_type = int(a[3])
        self.DCT = float(a[4])
        self.DCT_type = int(a[5])
        self.Shape_problem = int(a[6])
        self.b = float(a[7])

        self.lu, self.high_D_C, self.aaa, self.optial_f = cec2006_information(self.CEC_Problem)
        self.max_D_con = int(self.high_D_C + np.ceil((d - m - self.high_D_C) * self.HCT))
        self.max_D_con = max(int(self.high_D_C), self.max_D_con)

        if self.Shape_problem == 1:
            self.q1_upper = np.full(m, 4.0)
            self.q1_lower = np.zeros(m)
        else:
            self.q1_upper = np.full(m, 2.0)
            self.q1_lower = np.zeros(m)

        self.THETA_ = None
        self.h = None
        self.HC = None

        super().__init__(n_var=d, n_obj=m, n_ieq_constr=3, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        X = np.clip(as_2d(x, float), 0.0, 1.0)
        N, D = X.shape
        M = self.n_obj

        hi_end = min(D, M + int(self.max_D_con))
        new_P = _transformation_operator(X[:, M:hi_end], self.lu, int(self.high_D_C), int(self.HCT_type))
        self.h, self.HC = cec2006_fitness(new_P, self.CEC_Problem, self.aaa, self.optial_f)

        P = X.copy()
        control_D = int(np.ceil((D - int(self.high_D_C) - M) * self.DCT))
        control_D = max(0, min(control_D, max(0, D - (M + int(self.high_D_C)))))
        if control_D > 0:
            tail = np.arange(D - control_D, D, dtype=int)
            frac = (np.arange(1, control_D + 1, dtype=float) / max(1, control_D))[None, :]
            if self.DCT_type == 1:
                P[:, tail] = (1.0 + frac) * P[:, tail] - P[:, [0]]
            else:
                P[:, tail] = (1.0 + np.cos(0.5 * np.pi * frac)) * P[:, tail] - P[:, [0]]

        dis_start = min(D, M + int(self.high_D_C))
        dis_P = distance_function(P[:, dis_start:], self.Distance_problem)

        if M == 2:
            angle = np.arctan(np.abs(X[:, 1]) / X[:, 0])
            angle[np.isnan(angle)] = 1.0
            self.THETA_ = 2.0 / np.pi * angle[:, None]
        else:
            Sx = np.cumsum((X[:, :M][:, ::-1] ** 2), axis=1)[:, ::-1]
            angle = np.arctan(np.sqrt(Sx[:, 1:]) / X[:, : M - 1])
            angle[np.isnan(angle)] = 1.0
            self.THETA_ = 2.0 / np.pi * angle

        Pop = (self.q1_upper - self.q1_lower)[None, :] * X[:, :M] + self.q1_lower[None, :]
        T_ = (1.0 - np.sum(Pop**2, axis=1)) ** 2 + self.h + dis_P

        theta = self.THETA_
        G1 = np.column_stack([np.ones(N), np.cumprod(np.sin(np.pi / 2.0 * theta), axis=1)])
        G2 = np.column_stack([np.cos(np.pi / 2.0 * theta), np.ones(N)])
        PopObj = G1 * G2 * (1.0 + T_)[:, None]

        PopCon = np.zeros((N, 2), dtype=float)
        if M == 2:
            if self.Shape_problem == 1:
                cc = self.b / 10.0
                PopCon[:, 0] = cc**2 * Pop[:, 0] ** 2 + Pop[:, 1] ** 2 - cc**2
                PopCon[:, 1] = -cc * (Pop[:, 0] - 1.0) - Pop[:, 1]
            else:
                l = np.arctan(Pop[:, 1] / Pop[:, 0])
                l[np.isnan(l)] = 1.0
                cc = self.b / 100.0
                PopCon[:, 0] = Pop[:, 0] ** 2 + Pop[:, 1] ** 2 - (cc + 0.05 + 0.4 * np.sin(4 * l) ** 16) ** 2
                PopCon[:, 1] = (cc - 0.2 * np.sin(4 * l) ** 8) ** 2 - Pop[:, 0] ** 2 - Pop[:, 1] ** 2
        else:
            s2 = np.sum(Pop[:, :3] ** 2, axis=1)
            PopCon[:, 0] = 0.5 - s2
            PopCon[:, 1] = -2.0 - s2

        out["F"] = PopObj
        out["G"] = np.column_stack([PopCon, self.HC])

    def _calc_pareto_front(self, n_pareto_points=200):
        n = int(max(50, n_pareto_points))
        if self.n_obj == 3:
            return quarter_sphere_points_3d(n)

        deformed_shape1 = {3, 11, 14}
        deformed_shape2 = {4, 10, 12, 13}
        if self.problem_index in deformed_shape1:
            return _deformed_sdc_pf(n, self.Shape_problem, self.b, need_nd_filter=False)
        if self.problem_index in deformed_shape2:
            return _deformed_sdc_pf(n, self.Shape_problem, self.b, need_nd_filter=True)
        return quarter_sphere_points_2d(n)


class SDC1(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 1
    DEFAULT_M = 2


class SDC2(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 2
    DEFAULT_M = 2


class SDC3(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 3
    DEFAULT_M = 2


class SDC4(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 4
    DEFAULT_M = 2


class SDC5(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 5
    DEFAULT_M = 2


class SDC6(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 6
    DEFAULT_M = 2


class SDC7(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 7
    DEFAULT_M = 2


class SDC8(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 8
    DEFAULT_M = 2


class SDC9(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 9
    DEFAULT_M = 2


class SDC10(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 10
    DEFAULT_M = 2


class SDC11(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 11
    DEFAULT_M = 2


class SDC12(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 12
    DEFAULT_M = 2


class SDC13(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 13
    DEFAULT_M = 2


class SDC14(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 14
    DEFAULT_M = 2


class SDC15(_SDCBase):
    """Ref: Qiao et al., IEEE Transactions on Evolutionary Computation, 2024."""
    PROBLEM_INDEX = 15
    DEFAULT_M = 3


_CPU = [f"SDC{i}" for i in range(1, 16)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]

