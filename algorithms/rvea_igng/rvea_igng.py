# emopylab 2026
"""RVEA-iGNG: RVEA based on an improved Growing Neural Gas.

Faithful port of the PlatEMO implementation (written by Qiqi Liu).

Reference:
Q. Liu, Y. Jin, M. Heiderich, T. Rodemann, and G. Yu. An adaptive
reference vector-guided evolutionary algorithm using growing neural gas
for many-objective optimization of irregular problems. IEEE Transactions
on Cybernetics, 2022, 52(5): 2698-2711.

An improved GNG is trained on the normalized union of population and an
IBEA-style external archive; its nodes replace the reference-vector set of
RVEA (APD selection).  Corner nodes are pushed outward, node count is
capped at 1.5N with staleness-based pruning, and after 90% of the budget
the vector set is frozen and augmented with archive directions.

Backend: emopylab array facade; MLX selected automatically by
``core.algorithm.Algorithm`` on Apple Silicon when available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from core.population import Population
from util.optimum import filter_optimum

from core.algorithm import Algorithm
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.UniformPoint import UniformPoint
from util.array_backend import backend_cdist

from algorithms.community_utils.moead_family import (
    current_fe,
    fe_ratio,
    max_fe,
    sample_initial,
)

ALGORITHM_FLAGS = {"RVEA-iGNG": {"multi", "many"}, "RVEAiGNG": {"multi", "many"}}

_EPS = 1e-12


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    na = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), _EPS)
    nb = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), _EPS)
    return np.clip((A @ B.T) / (na * nb.T), -1.0, 1.0)


class _GGNet:
    """State of the improved growing gas net (PlatEMO struct)."""

    def __init__(self):
        self.w = np.zeros((0, 0))
        self.E = np.zeros(0)
        self.C = np.zeros((0, 0))
        self.t = np.zeros((0, 0))
        self.nx = 0
        self.age_sum_before = np.zeros(0)
        self.flag = np.zeros(0)


# GNG hyperparameters (params struct in PlatEMO)
_MAXIT = 50
_L = 50
_EPS_B = 0.2
_EPS_N = 0.006
_ALPHA = 0.5
_DELTA = 0.995
_T_AGE = 50


def _gng_insert(w, E, C, t, extra_rows):
    """Append one node created between the two max-error nodes."""
    q = int(np.argmax(E))
    f = int(np.argmax(C[:, q] * E))
    new = 0.5 * (w[q] + w[f])
    w = np.vstack([w, new[None, :]])
    C = np.pad(C, ((0, 1), (0, 1)))
    t = np.pad(t, ((0, 1), (0, 1)))
    E = np.append(E, 0.0)
    r = w.shape[0] - 1
    C[q, f] = C[f, q] = 0.0
    C[q, r] = C[r, q] = 1.0
    C[r, f] = C[f, r] = 1.0
    t[r, :] = 0.0
    t[:, r] = 0.0
    E[q] *= _ALPHA
    E[f] *= _ALPHA
    E[r] = E[q]
    extra = [np.append(arr, 0.0) for arr in extra_rows]
    return w, E, C, t, extra


def initialize_gng(V: np.ndarray, F: np.ndarray, N: int, rng: np.random.Generator) -> _GGNet:
    """Initialize the net from the reference vectors hit by the population."""
    F = np.asarray(F, dtype=float)
    M = F.shape[1]
    Fs = F - F.min(axis=0, keepdims=True)
    ang = np.arccos(_cosine_matrix(Fs, V))
    associate = np.argmin(ang, axis=1)
    valid = np.unique(associate)
    ref_size = len(valid)

    w = V[valid[:2]].astype(float).copy() if ref_size >= 2 else V[rng.permutation(len(V))[:2]].astype(float).copy()
    E = np.zeros(2)
    C = np.zeros((2, 2))
    t = np.zeros((2, 2))
    nx = 0

    for _ in range(_MAXIT):
        for kk in range(2, ref_size):
            nx += 1
            x = V[valid[kk]].astype(float)

            d = np.linalg.norm(w - x[None, :], axis=1)
            order = np.argsort(d)
            s1, s2 = int(order[0]), int(order[1])

            t[s1, :] += 1.0
            t[:, s1] += 1.0
            E[s1] += d[s1] ** 2

            w[s1] += _EPS_B * (x - w[s1])
            for j in np.where(C[s1] == 1)[0]:
                w[j] += _EPS_N * (x - w[j])

            C[s1, s2] = C[s2, s1] = 1.0
            t[s1, s2] = t[s2, s1] = 0.0

            C[t > _T_AGE] = 0.0
            alone = C.sum(axis=0) == 0
            if alone.any() and (~alone).sum() >= 2:
                keep = ~alone
                w, E = w[keep], E[keep]
                C = C[np.ix_(keep, keep)]
                t = t[np.ix_(keep, keep)]

            if nx % _L == 0 and w.shape[0] < N:
                w, E, C, t, _ = _gng_insert(w, E, C, t, [])

            E = _DELTA * E

    net = _GGNet()
    net.w, net.E, net.C, net.t, net.nx = w, E, C, t, nx
    net.age_sum_before = np.array([t[i, C[i] == 1].sum() for i in range(w.shape[0])])
    net.flag = np.zeros(w.shape[0])
    return net


def _corner_adjust(w: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Push corner nodes outward and clamp negatives (PlatEMO corner step)."""
    w = w.copy()
    output = []
    for i in range(w.shape[0]):
        nb = np.where(C[i] == 1)[0]
        group = np.vstack([w[i][None, :], w[nb]]) if nb.size else w[i][None, :]
        if np.any(np.argmin(group, axis=0) == 0) or np.any(np.argmax(group, axis=0) == 0):
            output.append(i)
    for s in output:
        nb = np.where(C[s] == 1)[0]
        if nb.size == 0:
            continue
        x = w[nb].mean(axis=0)
        w[s] = w[s] - 1.0 * (x - w[s])
    neg = np.any(w < 0.0, axis=1)
    for i in np.where(neg)[0]:
        w[i, w[i] < 0.0] = 0.0
    return w


def train_gng(
    V: np.ndarray,
    signals: np.ndarray,
    net: _GGNet,
    scale: np.ndarray,
    N: int,
    gen: int,
    max_gen: int,
    arc_obj: np.ndarray,
    gen_flag: Optional[int],
    z_min: np.ndarray,
) -> tuple[np.ndarray, _GGNet, Optional[int]]:
    """PlatEMO TrainGrowingGasNet."""
    w, E, C, t = net.w, net.E, net.C, net.t
    nx = net.nx
    asb, flag = net.age_sum_before, net.flag

    age_sum = np.array([t[i, C[i] == 1].sum() for i in range(w.shape[0])])
    same = age_sum == asb[: len(age_sum)] if len(asb) >= len(age_sum) else np.zeros(len(age_sum), dtype=bool)
    flag = flag[: len(age_sum)].copy() if len(flag) >= len(age_sum) else np.zeros(len(age_sum))
    flag[same] += 1.0
    asb = age_sum.copy()
    max_n = 1.5
    cap = int(round(max_n * N))

    if gen <= round(0.9 * max_gen):
        max_iter, max_pz = 1, max_n
        if w.shape[0] == cap:
            r = np.argsort(-flag)[: cap - N]
            keep = np.ones(w.shape[0], dtype=bool)
            keep[r] = False
            w, E = w[keep], E[keep]
            C = C[np.ix_(keep, keep)]
            t = t[np.ix_(keep, keep)]
            asb = asb[keep]
            flag = np.zeros(w.shape[0])
    else:
        if w.shape[0] < cap and gen_flag is None:
            max_pz, max_iter = max_n, 1
        else:
            max_pz, max_iter = 1.0, 0
        if w.shape[0] == cap:
            max_pz, max_iter = 1.0, 0
            gen_flag = gen

    if gen_flag is None:
        for _ in range(max_iter):
            for x in signals:
                nx += 1
                w = w / np.maximum(w.sum(axis=1, keepdims=True), _EPS)
                d = np.linalg.norm(w - x[None, :], axis=1)
                order = np.argsort(d)
                s1, s2 = int(order[0]), int(order[1]) if w.shape[0] > 1 else (int(order[0]),) * 2

                t[s1, :] += 1.0
                t[:, s1] += 1.0
                E[s1] += d[s1] ** 2

                w[s1] += _EPS_B * (x - w[s1])
                for j in np.where(C[s1] == 1)[0]:
                    w[j] += _EPS_N * (x - w[j])

                C[s1, s2] = C[s2, s1] = 1.0
                t[s1, s2] = t[s2, s1] = 0.0

                C[t > _T_AGE] = 0.0
                alone = C.sum(axis=0) == 0
                if alone.any() and (~alone).sum() >= 2:
                    keep = ~alone
                    w, E = w[keep], E[keep]
                    C = C[np.ix_(keep, keep)]
                    t = t[np.ix_(keep, keep)]
                    asb = asb[keep]
                    flag = flag[keep]

                if nx % _L == 0 and w.shape[0] < int(round(max_pz * N)):
                    w, E, C, t, (asb, flag) = _gng_insert(w, E, C, t, [asb, flag])

                E = _DELTA * E

        w = w / np.maximum(w.sum(axis=1, keepdims=True), _EPS)
        net.w, net.E, net.C, net.t, net.nx = w, E, C, t, nx
        net.age_sum_before, net.flag = asb, flag

        w_adj = _corner_adjust(w, C)
        V = w_adj * scale[None, :]

    if gen_flag is not None and gen == gen_flag:
        whole = np.asarray(arc_obj, dtype=float) - z_min[None, :]
        temp2 = whole / np.maximum(whole.sum(axis=1, keepdims=True), _EPS)
        w_adj = _corner_adjust(w, C)
        V = np.vstack([w_adj * scale[None, :], temp2])

    return V, net, gen_flag


def update_archive(pop: Population, archive: Population, max_size: int) -> Population:
    """IBEA-fitness archive with duplicate/outlier removal (PlatEMO)."""
    merged = Population.merge(archive, pop) if len(archive) else pop
    objs = np.asarray(merged.get("F"), dtype=float)
    _, ia = np.unique(objs, axis=0, return_index=True)
    merged = merged[np.sort(ia)]
    objs = np.asarray(merged.get("F"), dtype=float)
    fn, _ = NDSort(objs, 1)
    merged = merged[np.asarray(fn).reshape(-1) == 1]
    N = len(merged)
    if N > max_size:
        objs = np.asarray(merged.get("F"), dtype=float)
        o_min, o_max = objs.min(axis=0), objs.max(axis=0)
        CA = (objs - o_min[None, :]) / np.maximum(o_max - o_min, _EPS)[None, :]
        I = np.max(CA[:, None, :] - CA[None, :, :], axis=2)
        Cn = np.max(np.abs(I), axis=0)
        Fit = np.sum(-np.exp(-I / np.maximum(Cn[None, :], _EPS) / 0.05), axis=0) + 1.0
        choose = list(range(N))
        while len(choose) > max_size:
            x = int(np.argmin(Fit[choose]))
            removed = choose[x]
            Fit = Fit + np.exp(-I[removed, :] / max(Cn[removed], _EPS) / 0.05)
            choose.pop(x)
        merged = merged[np.asarray(choose, dtype=int)]
    # outlier removal
    o = np.asarray(merged.get("F"), dtype=float)
    o = o - o.min(axis=0, keepdims=True)
    d = np.linalg.norm(o, axis=1)
    keep = d <= 10.0 * max(d.mean(), _EPS)
    return merged[keep] if keep.any() else merged


class RVEAiGNG(Algorithm):
    """RVEA based on improved growing neural gas (Liu et al., 2022)."""

    def __init__(
        self,
        pop_size: int = 100,
        alpha: float = 2.0,
        sampling=None,
        seed: Optional[int] = None,
        use_gpu: bool = False,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(
            seed=seed, use_gpu=use_gpu, array_backend=array_backend,
            gpu_dtype=gpu_dtype, **kwargs,
        )
        self.pop_size = int(max(pop_size, 3))
        self.alpha = float(alpha)
        self.sampling = sampling

        self.V: Optional[np.ndarray] = None
        self.net: Optional[_GGNet] = None
        self.ext_archive: Population = Population.empty()
        self.scale: Optional[np.ndarray] = None
        self.z_min: Optional[np.ndarray] = None
        self.gen_flag: Optional[int] = None

    def _setup(self, problem, **kwargs):
        V, n_eff = UniformPoint(self.pop_size, int(problem.n_obj))
        self.V = np.asarray(V, dtype=float)
        self.pop_size = int(max(1, n_eff))
        self.scale = np.ones(int(problem.n_obj))

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills if infills is not None else Population.empty()
        if len(self.pop) == 0:
            self.opt = self.pop
            return
        F = np.asarray(self.pop.get("F"), dtype=float)
        self.z_min = F.min(axis=0)
        self.net = initialize_gng(self.V, F, self.pop_size, self.random_state)
        self.ext_archive = update_archive(self.pop, Population.empty(), self.pop_size)
        self._set_optimum()

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)
        idx = self.random_state.integers(0, len(self.pop), self.pop_size)
        X = np.asarray(self.pop.get("X"), dtype=float)
        off = OperatorGA(self.problem, X[idx], rng=self.random_state)
        return Population.new("X", np.asarray(off, dtype=float))

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return
        self.z_min = np.minimum(
            self.z_min, np.asarray(infills.get("F"), dtype=float).min(axis=0)
        )
        merged = Population.merge(self.pop, infills)
        theta = float(fe_ratio(self)) ** self.alpha
        self._environmental_selection(merged, theta)
        self._set_optimum()

    # -- environmental selection (includes GNG training) ------------------
    def _environmental_selection(self, pop: Population, theta: float) -> None:
        F = np.asarray(pop.get("F"), dtype=float)
        fn, _ = NDSort(F, 1)
        pop = pop[np.asarray(fn).reshape(-1) == 1]
        F = np.asarray(pop.get("F"), dtype=float)

        shifted = F - self.z_min[None, :]
        d = np.linalg.norm(shifted, axis=1)
        keep = d <= 10.0 * max(d.mean(), _EPS)
        pop = pop[keep]

        self.ext_archive = update_archive(pop, self.ext_archive, 2 * self.pop_size)
        arc_obj = np.asarray(self.ext_archive.get("F"), dtype=float)

        whole = Population.merge(pop, self.ext_archive)
        whole_obj = np.asarray(whole.get("F"), dtype=float)
        _, ia = np.unique(whole_obj, axis=0, return_index=True)
        ia = np.sort(ia)
        whole = whole[ia]
        whole_obj = whole_obj[ia] - self.z_min[None, :]

        whole1 = whole_obj / np.maximum(self.scale[None, :], _EPS)
        temp1 = whole1 / np.maximum(whole1.sum(axis=1, keepdims=True), _EPS)

        gen = int(np.ceil(current_fe(self) / self.pop_size))
        try:
            max_gen = int(np.ceil(max_fe(self) / self.pop_size))
        except Exception:
            max_gen = 10 ** 9

        fr = 0.1
        if gen % max(1, int(np.ceil(fr * max_gen))) == 0 and gen <= max_gen and len(arc_obj):
            self.scale = arc_obj.max(axis=0) - arc_obj.min(axis=0)
            self.scale = np.maximum(self.scale, _EPS)

        if (
            len(temp1) > 2
            and np.all(np.isfinite(temp1))
            and gen <= max_gen
            and self.gen_flag is None
        ):
            self.V, self.net, self.gen_flag = train_gng(
                self.V, temp1, self.net, self.scale, self.pop_size,
                gen, max_gen, arc_obj, self.gen_flag, self.z_min,
            )
        elif self.gen_flag is not None and gen == self.gen_flag:
            self.V, self.net, self.gen_flag = train_gng(
                self.V, temp1, self.net, self.scale, self.pop_size,
                gen, max_gen, arc_obj, self.gen_flag, self.z_min,
            )

        # APD-based selection over V
        V = np.asarray(self.V, dtype=float)
        NV = len(V)
        P = whole_obj
        M = P.shape[1]

        cos_vv = _cosine_matrix(V, V)
        np.fill_diagonal(cos_vv, 0.0)
        gamma = np.maximum(np.min(np.arccos(np.clip(cos_vv, -1.0, 1.0)), axis=1), _EPS)

        angle = np.arccos(_cosine_matrix(P, V))
        associate = np.argmin(angle, axis=1)

        next_idx = []
        for i in np.unique(associate):
            current = np.where(associate == i)[0]
            apd = (
                1.0 + M * theta * angle[current, i] / gamma[i]
            ) * np.linalg.norm(P[current], axis=1)
            next_idx.append(int(current[np.argmin(apd)]))

        selected = whole[np.asarray(sorted(set(next_idx)), dtype=int)]

        if len(selected) > self.pop_size:
            sel_obj = np.asarray(selected.get("F"), dtype=float) - self.z_min[None, :]
            choose = np.zeros(len(selected), dtype=bool)
            choose[np.argmin(sel_obj, axis=0)] = True
            choose[np.argmax(sel_obj, axis=0)] = True
            while choose.sum() < self.pop_size:
                rest = np.where(~choose)[0]
                sel = np.where(choose)[0]
                dist = backend_cdist(sel_obj[rest], sel_obj[sel])
                pick = rest[int(np.argmax(dist.min(axis=1)))]
                choose[pick] = True
            selected = selected[choose]

        self.pop = selected

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)


ALGORITHMS = {"RVEA-iGNG": RVEAiGNG}
