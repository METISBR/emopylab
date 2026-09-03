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


def _duan_yueshu(index: int):
    idx = int(index)
    if idx == 1:
        nk = [5, 4, 6, 6, 3, 5, 5, 5, 5, 2]
        c_index = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        cct, dct, dis_function, pf_form, q_form = 1, 2, [1, 2], 1, 1
    elif idx == 2:
        nk = [5, 4, 6, 6, 3, 5, 5, 5, 5, 2]
        c_index = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        cct, dct, dis_function, pf_form, q_form = 1, 1, [1, 1], 3, 1
    elif idx == 3:
        nk = [5, 4, 6, 6, 3, 5, 5, 5, 5, 2]
        c_index = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        cct, dct, dis_function, pf_form, q_form = 1, 1, [1, 3], 2, 2
    elif idx == 4:
        nk = [5, 4, 3, 5, 4, 3, 5, 4, 3, 5]
        c_index = [4, 11, 4, 11, 4, 11, 4, 11, 4, 11]
        cct, dct, dis_function, pf_form, q_form = 2, 2, [4, 3], 3, 1
    elif idx == 5:
        nk = [5, 4, 3, 5, 4, 3, 5, 4, 3, 5]
        c_index = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        cct, dct, dis_function, pf_form, q_form = 1, 1, [2, 3], 1, 3
    elif idx == 6:
        nk = [5, 3, 5, 4, 3, 5, 4, 3, 5, 2]
        c_index = [7, 8, 7, 8, 7, 8, 7, 8, 7, 8]
        cct, dct, dis_function, pf_form, q_form = 2, 1, [1, 5], 1, 1
    elif idx == 7:
        nk = [3, 6, 5, 6, 3, 6, 5, 6, 3, 6]
        c_index = [3, 11, 4, 3, 11, 4, 3, 11, 4, 3]
        cct, dct, dis_function, pf_form, q_form = 1, 1, [4, 5], 1, 2
    elif idx == 8:
        nk = [4, 4, 5, 6, 3, 5, 5, 5, 5, 2]
        c_index = [10, 11, 9, 10, 11, 9, 10, 11, 9, 10]
        cct, dct, dis_function, pf_form, q_form = 2, 2, [1, 1], 2, 1
    elif idx == 9:
        nk = [4, 5, 4, 5, 4, 5, 4, 5, 4, 5]
        c_index = [4, 7, 11, 4, 7, 11, 4, 7, 11, 4]
        cct, dct, dis_function, pf_form, q_form = 1, 1, [1, 4], 2, 2
    elif idx == 10:
        nk = [4, 5, 4, 5, 4, 5, 4, 5, 4, 5]
        c_index = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        cct, dct, dis_function, pf_form, q_form = 2, 2, [1, 2], 1, 1
    elif idx == 11:
        c_index = [4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6]
        nk = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
        cct, dct, dis_function, pf_form, q_form = 1, 2, [2, 3], 2, 2
    elif idx == 12:
        c_index = [4, 5, 6, 4, 4, 5, 6, 4, 4, 5]
        nk = [11, 4, 3, 11, 4, 3, 11, 4, 3, 11]
        cct, dct, dis_function, pf_form, q_form = 1, 1, [1, 1], 3, 1
    else:
        raise ValueError(f"Unsupported LSCM index: {index}")

    cec_problem = [1, 2, 3, 6, 9, 10, 11, 12, 14, 18, 24]
    out2 = [cec_problem[i - 1] for i in c_index]
    return np.asarray(nk, dtype=int), np.asarray(out2, dtype=int), int(cct), int(dct), [int(dis_function[0]), int(dis_function[1])], int(q_form), int(pf_form)


def _pf_function(pop_dec: np.ndarray, G: np.ndarray, pf_form: int, q_form: int) -> np.ndarray:
    N, M = G.shape
    if pf_form == 1:
        base = np.fliplr(np.cumprod(np.column_stack([np.ones(N), pop_dec[:, : M - 1]]), axis=1)) * np.column_stack(
            [np.ones(N), 1.0 - pop_dec[:, M - 1 :: -1][:, : M - 1]]
        )
        if q_form == 1:
            return (1.0 + G) * base
        if q_form == 2:
            return (1.0 + G + np.column_stack([G[:, 1:], np.zeros(N)])) * base
        if q_form == 3:
            gsum = np.sum(G, axis=1)
            return (1.0 + gsum[:, None]) * base
    elif pf_form == 2:
        cpart = np.fliplr(np.cumprod(np.column_stack([np.ones(N), np.cos(pop_dec[:, : M - 1] * np.pi / 2.0)]), axis=1))
        spart = np.column_stack([np.ones(N), np.sin(pop_dec[:, M - 1 :: -1][:, : M - 1] * np.pi / 2.0)])
        base = cpart * spart
        if q_form == 1:
            return (1.0 + G) * base
        if q_form == 2:
            return (1.0 + G + np.column_stack([G[:, 1:], np.zeros(N)])) * base
        if q_form == 3:
            gsum = np.sum(G, axis=1)
            return (1.0 + gsum[:, None]) * base
    elif pf_form == 3:
        obj = np.zeros((N, M), dtype=float)
        obj[:, : M - 1] = pop_dec[:, : M - 1]
        g_last = 1.0 + G[:, -1]
        term = np.sum(obj[:, : M - 1] / (1.0 + g_last[:, None]) * (1.0 + np.sin(3.0 * np.pi * obj[:, : M - 1])), axis=1)
        obj[:, M - 1] = (1.0 + g_last) * (M - term)
        return obj
    raise ValueError(f"Unsupported PF/Q form combination: pf_form={pf_form}, q_form={q_form}")


def _simplex_points_3d(n: int) -> np.ndarray:
    g = max(5, int(np.ceil(np.sqrt(max(4, n)))))
    a = np.linspace(0.0, 1.0, g)
    x, y = np.meshgrid(a, a)
    z = 1.0 - x - y
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return pts[np.all(pts >= 0.0, axis=1)]


class _LSCMBase(Problem):
    KU_INDEX = 1
    DEFAULT_M = 2

    def __init__(self, n_var: int = 100, **kwargs):
        self.ku_index = int(self.KU_INDEX)
        self.base = None
        self.duan = None

        m = int(self.DEFAULT_M)
        d_req = int(n_var)
        if d_req <= 0:
            d_req = 100
        if d_req == 100:
            base = 100
            d = 100
        else:
            base = int(np.ceil(d_req / 1000.0) * 100)
            d = int(round(d_req / base) * base)
            d = max(base, d)
        self.base = int(base)
        self.duan = int(d // base)

        self.CCT = None
        self.DCT = None
        self.Dis_function = None
        self.Q_form = None
        self.PF_form = None
        self.CEC_Problem = None
        self.lu = []
        self.high_D_C = None
        self.aaa = []
        self.optial_f = None
        self.nk = None
        self.sublen: list[np.ndarray] = []
        self.seg_len: list[np.ndarray] = []
        self.CV: list[np.ndarray] = []
        self.Dis: list[list[list[np.ndarray]]] = []

        nk_all, cec_all, self.CCT, self.DCT, self.Dis_function, self.Q_form, self.PF_form = _duan_yueshu(self.ku_index)
        self.nk = nk_all[: self.duan].astype(int)
        self.CEC_Problem = cec_all[: self.duan].astype(int)
        self.high_D_C = np.zeros(self.duan, dtype=int)
        self.optial_f = np.zeros(self.duan, dtype=float)

        for j in range(self.duan):
            lu_j, high_d_c_j, aaa_j, opt_j = cec2006_information(int(self.CEC_Problem[j]))
            self.lu.append(np.asarray(lu_j, dtype=float))
            self.high_D_C[j] = int(high_d_c_j)
            self.aaa.append(np.asarray(aaa_j, dtype=float))
            self.optial_f[j] = float(opt_j)

        c = [3.8 * 0.1 * (1 - 0.1)]
        for _ in range(m - 1):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        self._build_variable_information(np.asarray(c, dtype=float), m)

        super().__init__(n_var=int(d), n_obj=int(m), n_ieq_constr=1, xl=0.0, xu=1.0, vtype=float, **kwargs)

    def _build_variable_information(self, c: np.ndarray, m: int) -> None:
        self.sublen.clear()
        self.seg_len.clear()
        self.CV.clear()
        self.Dis.clear()

        for j in range(self.duan):
            if j == 0:
                total = self.base - m - int(self.high_D_C[j])
                cv_start = m
            else:
                total = self.base - int(self.high_D_C[j])
                cv_start = j * self.base
            total = max(0, int(total))
            nk_j = max(1, int(self.nk[j]))
            sub = np.floor(c / np.sum(c) * total / nk_j).astype(int)
            seg = np.concatenate([[0], np.cumsum(sub * nk_j)])
            cv_idx = np.arange(cv_start, cv_start + int(self.high_D_C[j]), dtype=int)
            cv_end = int(cv_idx[-1]) if cv_idx.size else (cv_start - 1)

            dis_j: list[list[np.ndarray]] = []
            for mm in range(m):
                subsets: list[np.ndarray] = []
                for i in range(nk_j):
                    start = cv_end + 1 + int(seg[mm]) + i * int(sub[mm])
                    end = cv_end + int(seg[mm + 1]) + i * int(sub[mm])  # exclusive target computed below
                    stop = start + int(sub[mm])
                    if int(sub[mm]) <= 0:
                        subsets.append(np.empty(0, dtype=int))
                    else:
                        subsets.append(np.arange(start, stop, dtype=int))
                dis_j.append(subsets)

            self.sublen.append(sub)
            self.seg_len.append(seg)
            self.CV.append(cv_idx)
            self.Dis.append(dis_j)

    def _variable_linkage(self, pop_dec: np.ndarray) -> np.ndarray:
        pop_dec = np.asarray(pop_dec, dtype=float).copy()
        PPP = pop_dec.copy()
        N = pop_dec.shape[0]

        for i in range(self.duan):
            cv_idx = self.CV[i]
            if cv_idx.size == 0:
                continue
            if i == 0:
                last_P = PPP[:, 0]
            else:
                prev_last_subset = self.Dis[i - 1][self.n_obj - 1][int(self.nk[i - 1]) - 1]
                last_P = PPP[:, int(prev_last_subset[-1])] if prev_last_subset.size else PPP[:, 0]

            P = pop_dec[:, cv_idx]
            PP = last_P[:, None] + P
            a2 = np.mod(PP, 0.5)
            if self.CCT == 1:
                pop_dec[:, cv_idx] = -2.0 * a2 + 1.0
            else:
                pop_dec[:, cv_idx] = np.cos(a2 * np.pi)

            first_subset = self.Dis[i][0][0]
            last_subset = self.Dis[i][self.n_obj - 1][int(self.nk[i]) - 1]
            if first_subset.size == 0 or last_subset.size == 0:
                continue
            control = np.arange(int(first_subset[0]), int(last_subset[-1]) + 1, dtype=int)
            if control.size == 0:
                continue
            frac = (np.arange(1, control.size + 1, dtype=float) / max(1, control.size))[None, :]
            if self.DCT == 1:
                pop_dec[:, control] = (1.0 + frac) * pop_dec[:, control] - last_P[:, None]
            else:
                pop_dec[:, control] = (1.0 + np.cos(0.5 * np.pi * frac)) * pop_dec[:, control] - last_P[:, None]

        return pop_dec

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.clip(as_2d(x, float), 0.0, 1.0)
        N = x.shape[0]
        M = self.n_obj

        pop_dec = self._variable_linkage(x)

        G = np.zeros((N, M), dtype=float)
        Con = np.zeros(N, dtype=float)

        for mm in range(self.duan):
            cv_idx = self.CV[mm]
            P = pop_dec[:, cv_idx]
            lu = self.lu[mm]
            new_P = lu[0][None, :] + (lu[1] - lu[0])[None, :] * P
            h, hc = cec2006_fitness(new_P, int(self.CEC_Problem[mm]), self.aaa[mm], float(self.optial_f[mm]))
            G += h[:, None] / M
            Con += hc

            for i in range(M):
                temp = np.zeros(N, dtype=float)
                for j in range(int(self.nk[mm])):
                    subset = self.Dis[mm][i][j]
                    temp += distance_function(pop_dec[:, subset], int(self.Dis_function[0 if i % 2 == 0 else 1]))
                denom = int(self.sublen[mm][i]) * int(self.nk[mm])
                if denom > 0:
                    G[:, i] += temp / denom

        out["F"] = _pf_function(pop_dec, G, int(self.PF_form), int(self.Q_form))
        out["G"] = Con[:, None]

    def _calc_pareto_front(self, n_pareto_points=200):
        n = int(max(20, n_pareto_points))
        if self.PF_form == 1:
            if self.n_obj == 2:
                return simplex_points_2d(n)
            if self.n_obj == 3:
                return _simplex_points_3d(n)
            return None
        if self.PF_form == 2:
            if self.n_obj == 2:
                return quarter_sphere_points_2d(n)
            if self.n_obj == 3:
                return quarter_sphere_points_3d(n)
            return None
        if self.PF_form == 3:
            if self.n_obj == 2:
                x = np.linspace(0.0, 1.0, n)
                y = 2.0 * (2.0 - x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)))
                pf = np.column_stack([x, y])
                return pf[nd_mask(pf)]
            if self.n_obj == 3:
                g = max(15, int(np.ceil(np.sqrt(n))))
                x, y = np.meshgrid(np.linspace(0.0, 1.0, g), np.linspace(0.0, 1.0, g))
                z = 2.0 * (3.0 - x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)) - y / 2.0 * (1.0 + np.sin(3.0 * np.pi * y)))
                pf = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                return pf[nd_mask(pf)]
        return None


class LSCM1(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 1
    DEFAULT_M = 2


class LSCM2(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 2
    DEFAULT_M = 2


class LSCM3(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 3
    DEFAULT_M = 2


class LSCM4(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 4
    DEFAULT_M = 2


class LSCM5(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 5
    DEFAULT_M = 2


class LSCM6(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 6
    DEFAULT_M = 2


class LSCM7(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 7
    DEFAULT_M = 2


class LSCM8(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 8
    DEFAULT_M = 2


class LSCM9(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 9
    DEFAULT_M = 2


class LSCM10(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 10
    DEFAULT_M = 3


class LSCM11(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 11
    DEFAULT_M = 3


class LSCM12(_LSCMBase):
    """Ref: Qiao et al., Swarm and Evolutionary Computation, 2024."""
    KU_INDEX = 12
    DEFAULT_M = 3


_CPU = [f"LSCM{i}" for i in range(1, 13)]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]

