# emopylab 2026
"""DEA-GNG: Decomposition-based EA guided by Growing Neural Gas.

Faithful port of the PlatEMO implementation (Copyright Yiping Liu).

Reference:
Y. Liu, H. Ishibuchi, N. Masuyama, and Y. Nojima. Adapting reference
vectors and scalarizing functions by growing neural gas to handle
irregular Pareto fronts. IEEE Transactions on Evolutionary Computation,
2020, 24(3): 439-453.

The GNG net learns the topology of the non-dominated front from an input
signal archive; its (expanded) nodes act as additional reference vectors,
uniform vectors too close to the learned nodes are removed, and the PBI
theta of each node is tuned from the local edge angles.  This is the main
non-linear/topology-learning competitor of intrinsic-dimension approaches
such as GCS-MaOEA.

Backend: uses the emopylab array facade (``util.array_backend``); the MLX
backend is selected automatically by ``core.algorithm.Algorithm`` on Apple
Silicon when available (``array_backend='auto'``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from core.population import Population
from util.optimum import filter_optimum

from core.algorithm import Algorithm
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from operators.utility_functions.UniformPoint import UniformPoint
from util.array_backend import backend_cdist

from algorithms.community_utils.moead_family import (
    current_fe,
    max_fe,
    sample_initial,
)

ALGORITHM_FLAGS = {"DEA-GNG": {"multi", "many"}, "DEAGNG": {"multi", "many"}}

_EPS = 1e-12


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of A and rows of B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    na = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), _EPS)
    nb = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), _EPS)
    return np.clip((A @ B.T) / (na * nb.T), -1.0, 1.0)


def _connected_components(edge: np.ndarray) -> np.ndarray:
    """Label connected components of an undirected adjacency matrix (BFS)."""
    n = edge.shape[0]
    labels = -np.ones(n, dtype=int)
    comp = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = comp
        while stack:
            u = stack.pop()
            for v in np.where(edge[u] > 0)[0]:
                if labels[v] < 0:
                    labels[v] = comp
                    stack.append(int(v))
        comp += 1
    return labels


class _GNGNet:
    """State container of the growing neural gas network (PlatEMO struct)."""

    def __init__(self, max_age: int, max_node: int, lam: int, max_hp: int):
        self.max_iter = 1
        self.max_age = int(max_age)
        self.max_node = int(max_node)
        self.lam = max(1, int(lam))
        self.max_hp = int(max_hp)
        self.node = np.zeros((0, 0))       # training-state nodes
        self.node_s = np.zeros((0, 0))     # expanded nodes (Algorithm 3)
        self.err = np.zeros(0)
        self.edge = np.zeros((0, 0))
        self.age = np.zeros((0, 0))
        self.hp = np.zeros(0)
        self._sig_count = 0


def gng_update(
    o_signals: np.ndarray,
    net: _GNGNet,
    rng: np.random.Generator,
    signal_cap: int = 0,
) -> _GNGNet:
    """One training epoch of the GNG on the signal archive (PlatEMO GNGUpdate).

    ``signal_cap > 0`` subsamples the training signals per epoch (uniformly,
    without replacement).  The default 0 keeps the PlatEMO-faithful behavior of
    iterating over the whole archive (up to M*N signals per generation), which
    dominates DEA-GNG's wall time in Python; a cap of ~2N gives a large
    speed-up at a small fidelity cost and must be reported as an
    implementation deviation when used in comparisons.
    """
    o_signals = np.asarray(o_signals, dtype=float)
    n_sig, D = o_signals.shape
    if n_sig < 2:
        return net

    signals = o_signals[rng.permutation(n_sig)]
    if signal_cap and n_sig > int(signal_cap) > 2:
        signals = signals[: int(signal_cap)]
    o_min = signals.min(axis=0)
    o_rng = np.maximum(signals.max(axis=0) - o_min, _EPS)
    signals = (signals - o_min) / o_rng

    node, err, edge, age, hp = net.node, net.err, net.edge, net.age, net.hp

    for _ in range(net.max_iter):
        if node.shape[0] <= 2 or node.shape[1] != D:
            x_min = signals.min(axis=0)
            x_max = signals.max(axis=0)
            node = rng.uniform(x_min, x_max, size=(2, D))
            err = np.zeros(2)
            edge = np.zeros((2, 2))
            age = np.zeros((2, 2))
            hp = np.full(2, float(net.max_hp))

        for pattern in signals:
            if node.shape[0] < 2:  # safety: re-seed from data
                node = np.vstack([node, pattern[None, :]])
                err = np.append(err, 0.0)
                edge = np.pad(edge, ((0, 1), (0, 1)))
                age = np.pad(age, ((0, 1), (0, 1)))
                hp = np.append(hp, float(net.max_hp))
                continue

            d = np.linalg.norm(node - pattern[None, :], axis=1)
            order = np.argsort(d)
            ra, rb = int(order[0]), int(order[1])

            hp = hp - 1.0
            hp[ra] = float(net.max_hp)
            hp[rb] = hp[rb] + 1.0

            age[ra, :] += 1.0
            age[:, ra] += 1.0
            err[ra] += d[ra] ** 2

            node[ra] += 0.2 * (pattern - node[ra])          # epsilon_a
            for j in np.where(edge[ra] == 1)[0]:
                node[j] += 0.01 * (pattern - node[j])       # epsilon_nb

            edge[ra, rb] = edge[rb, ra] = 1.0
            age[ra, rb] = age[rb, ra] = 0.0

            edge[age > net.max_age] = 0.0

            dead = hp <= 0.0
            if dead.any():
                keep = ~dead
                node, err, hp = node[keep], err[keep], hp[keep]
                edge = edge[np.ix_(keep, keep)]
                age = age[np.ix_(keep, keep)]
                if node.shape[0] < 2:
                    continue

            net._sig_count += 1
            if net._sig_count % net.lam == 0 and node.shape[0] < net.max_node:
                r1 = int(np.argmax(err))
                r2 = int(np.argmax(edge[:, r1] * err))
                new = 0.5 * (node[r1] + node[r2])
                node = np.vstack([node, new[None, :]])
                edge = np.pad(edge, ((0, 1), (0, 1)))
                age = np.pad(age, ((0, 1), (0, 1)))
                err = np.append(err, 0.0)
                hp = np.append(hp, float(net.max_hp))
                rn = node.shape[0] - 1
                edge[r1, r2] = edge[r2, r1] = 0.0
                edge[r1, rn] = edge[rn, r1] = 1.0
                edge[rn, r2] = edge[r2, rn] = 1.0
                age[rn, :] = 0.0
                age[:, rn] = 0.0
                err[r1] *= 0.5                              # alpha
                err[r2] *= 0.5
                err[rn] = err[r1]

            err = 0.9 * err                                 # delta

    net.node = node.copy()
    net.err, net.edge, net.age, net.hp = err, edge, age, hp

    # Expansion (Algorithm 3): rescale each sub-network to its data range.
    if node.shape[0] >= 1:
        dist = backend_cdist(node, signals)
        pi = np.argmin(dist, axis=0)
        labels = _connected_components(edge)
        node_exp = node.copy()
        for c in range(labels.max() + 1):
            sub_net = np.where(labels == c)[0]
            sub_data = np.where(labels[pi] == c)[0]
            if len(sub_data) <= 1 or len(sub_net) <= 1:
                continue
            data_min = signals[sub_data].min(axis=0)
            data_rng = signals[sub_data].max(axis=0) - data_min
            net_min = node[sub_net].min(axis=0)
            net_rng = np.maximum(node[sub_net].max(axis=0) - net_min, _EPS)
            ratio = data_rng / net_rng
            node_exp[sub_net] = (node[sub_net] - net_min) * ratio + data_min
        net.node_s = node_exp
    return net


def reference_combination(Ru: np.ndarray, net: _GNGNet) -> np.ndarray:
    """Remove uniform vectors too close to GNG nodes (PlatEMO Algorithm 4)."""
    node_s = net.node_s
    sums = np.maximum(np.sum(node_s, axis=1, keepdims=True), _EPS)
    node_p = node_s / sums

    d1 = backend_cdist(node_p, node_p) * net.edge
    n_edges = float(np.sum(net.edge))
    avg_dis = float(np.sum(d1)) / n_edges if n_edges > 0 else np.inf

    d2 = backend_cdist(Ru, Ru)
    d2 = np.where(d2 == 0.0, 1.0, d2)
    min_dis = float(d2.min())
    avg_dis = min(avg_dis, min_dis)

    d3 = backend_cdist(Ru, node_p)
    choose = np.all(d3 > avg_dis, axis=1)
    Ruq = Ru[choose]
    return Ruq if len(Ruq) > 0 else Ru[:1]


def tune_pbi(net: _GNGNet, eps: float) -> np.ndarray:
    """Tune the PBI theta of each GNG node from local edge angles."""
    n = net.node_s.shape[0]
    theta = np.zeros(n)
    for i in range(n):
        nb = np.where(net.edge[i] == 1)[0]
        if nb.size == 0:
            theta[i] = np.inf
            continue
        ev = net.node_s[i][None, :] - net.node_s[nb]
        if np.any(np.all(ev == 0.0, axis=1)):
            theta[i] = np.inf
            continue
        cos = _cosine_matrix(net.node_s[i][None, :], ev).max()
        ang = max(np.arccos(np.clip(cos, -1.0, 1.0)) - eps, 0.0)
        theta[i] = np.inf if ang <= 0.0 else max(1.0 / np.tan(ang), 0.0)
    return theta


def archive_update(
    data: np.ndarray,
    n_max: int,
    Z1: np.ndarray,
    Z2: np.ndarray,
    z_min: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Update the input-signal archive (objective vectors only)."""
    data = np.unique(np.asarray(data, dtype=float), axis=0)
    front_no, _ = NDSort(data, len(data))
    data = data[np.asarray(front_no).reshape(-1) == 1]
    if len(data) <= n_max:
        return data

    Z = np.vstack([Z1, Z2]) if len(Z2) else np.asarray(Z1, dtype=float)
    N, M = data.shape
    nz1 = len(Z1)
    z_max = data.max(axis=0)
    norm = (data - z_min[None, :]) / np.maximum(z_max - z_min, _EPS)[None, :]

    dist = 1.0 - _cosine_matrix(norm, Z)          # cosine distance
    pi = np.argmin(dist, axis=1)
    d2 = dist[np.arange(N), pi]

    cn = np.zeros(len(Z), dtype=int)
    choose = np.zeros(N, dtype=bool)
    z_choose = np.ones(len(Z), dtype=bool)
    while choose.sum() < n_max:
        cand = np.where(z_choose)[0]
        if cand.size == 0:
            rest = np.where(~choose)[0]
            take = rng.permutation(rest)[: n_max - int(choose.sum())]
            choose[take] = True
            break
        cand = cand[cn[cand] == cn[cand].min()]
        pref = cand[cand < nz1]
        j = int(rng.choice(pref if pref.size else cand))
        I = np.where(~choose & (pi == j))[0]
        if I.size:
            if cn[j] == 0:
                s = I[np.argmin(d2[I])]
            elif cn[j] < M:
                s = I[np.argmax(d2[I])]
            else:
                s = rng.choice(I)
            choose[int(s)] = True
            cn[j] += 1
        else:
            z_choose[j] = False
    return data[choose]


class DEAGNG(Algorithm):
    """Decomposition-based EA guided by growing neural gas (Liu et al., 2020)."""

    def __init__(
        self,
        pop_size: int = 100,
        aph: float = 0.1,
        eps: float = 0.314,
        gng_signal_cap: int = 0,
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
        self.aph = float(aph)
        self.eps = float(eps)
        self.gng_signal_cap = int(max(0, gng_signal_cap))
        self.sampling = sampling

        self.Ru: Optional[np.ndarray] = None
        self.Ruq: Optional[np.ndarray] = None
        self.net: Optional[_GNGNet] = None
        self.theta: np.ndarray = np.zeros(0)
        self.AS: Optional[np.ndarray] = None
        self.z_min: Optional[np.ndarray] = None
        self.front_no: Optional[np.ndarray] = None
        self.crd: Optional[np.ndarray] = None
        self.max_gen: int = 10 ** 9
        self.nog: float = 0.0
        self.archive_size: int = 0

    # -- setup ----------------------------------------------------------
    def _setup(self, problem, **kwargs):
        Ru, n_eff = UniformPoint(self.pop_size, int(problem.n_obj))
        self.Ru = np.asarray(Ru, dtype=float)
        self.pop_size = int(max(1, n_eff))
        self.Ruq = self.Ru.copy()
        self.archive_size = int(problem.n_obj) * self.pop_size
        self.net = _GNGNet(
            max_age=self.pop_size,
            max_node=self.pop_size,
            lam=int(0.2 * self.pop_size),
            max_hp=2 * self.archive_size,
        )
        self.net.node_s = np.zeros((0, int(problem.n_obj)))

    def _initialize_infill(self):
        return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills if infills is not None else Population.empty()
        if len(self.pop) == 0:
            self.opt = self.pop
            return
        F = np.asarray(self.pop.get("F"), dtype=float)
        self.z_min = F.min(axis=0)
        fn, _ = NDSort(F, self.pop_size)
        self.front_no = np.asarray(fn, dtype=float).reshape(-1)
        self.crd = np.zeros(len(self.pop))
        try:
            self.max_gen = int(np.ceil(max_fe(self) / self.pop_size))
        except Exception:
            self.max_gen = 10 ** 9
        self.nog = self.aph * self.max_gen
        self._set_optimum()

    # -- main loop ------------------------------------------------------
    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)
        # emopylab TournamentSelection returns MATLAB-style 1-based indices.
        idx = np.asarray(
            TournamentSelection(2, self.pop_size, self.front_no, self.crd), dtype=int,
        ) - 1
        X = np.asarray(self.pop.get("X"), dtype=float)
        off = OperatorGA(self.problem, X[idx], rng=self.random_state)
        return Population.new("X", np.asarray(off, dtype=float))

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return
        off_F = np.asarray(infills.get("F"), dtype=float)
        self.z_min = np.minimum(self.z_min, off_F.min(axis=0))

        gen = int(np.ceil(current_fe(self) / self.pop_size))
        if gen <= self.max_gen - self.nog:
            AS = off_F if self.AS is None else np.vstack([self.AS, off_F])
            self.AS = archive_update(
                AS, self.archive_size, self.Ruq, self.net.node_s, self.z_min, self.random_state,
            )
            n_as = len(self.AS)
            self.net.max_node = int(min(self.pop_size, n_as // 2))
            self.net.max_hp = 2 * n_as
            self.net = gng_update(
                self.AS, self.net, self.random_state,
                signal_cap=self.gng_signal_cap,
            )
            if self.net.node_s.shape[0] > 2:
                self.Ruq = reference_combination(self.Ru, self.net)
            self.theta = tune_pbi(self.net, self.eps)

        merged = Population.merge(self.pop, infills)
        self._environmental_selection(merged)
        self._set_optimum()

    # -- environmental selection ----------------------------------------
    def _environmental_selection(self, pop: Population) -> None:
        F = np.asarray(pop.get("F"), dtype=float)
        fn, max_fno = NDSort(F, self.pop_size)
        fn = np.asarray(fn, dtype=float).reshape(-1)
        next_mask = fn < float(max_fno)
        F1 = F[fn == 1.0]
        z_max = F1.max(axis=0)

        last = np.where(fn == float(max_fno))[0]
        k = int(self.pop_size - next_mask.sum())
        choose, crd = self._last_selection(F[next_mask], F[last], k, z_max)
        next_mask[last[choose]] = True

        self.pop = pop[next_mask]
        self.front_no = fn[next_mask]
        self.crd = crd

    def _last_selection(
        self, F1: np.ndarray, F2: np.ndarray, K: int, z_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        M = F2.shape[1] if F2.size else F1.shape[1]
        node_s = self.net.node_s if self.net.node_s.shape[0] else np.zeros((0, M))
        R = np.vstack([self.Ruq, node_s]) if node_s.shape[0] else self.Ruq
        nr1 = len(self.Ruq)
        theta_full = np.concatenate([np.full(nr1, np.inf), self.theta[: max(0, len(R) - nr1)]])

        P = np.vstack([F1, F2]) if F1.size else np.asarray(F2, dtype=float)
        N = len(P)
        n1 = len(F1)
        n2 = len(F2)
        Pn = (P - self.z_min[None, :]) / np.maximum(z_max - self.z_min, _EPS)[None, :]

        cos = _cosine_matrix(Pn, R)
        norm_p = np.linalg.norm(Pn, axis=1)
        d2_mat = norm_p[:, None] * np.sqrt(np.maximum(1.0 - cos ** 2, 0.0))
        pi = np.argmin(d2_mat, axis=1)
        d2 = d2_mat[np.arange(N), pi]

        rho = np.bincount(pi[:n1], minlength=len(R)).astype(int)

        g = np.zeros(N)
        for i in range(n1, N):
            th = theta_full[pi[i]]
            if np.isinf(th):
                g[i] = d2[i]
            else:
                d1 = norm_p[i] * cos[i, pi[i]]
                g[i] = d1 + th * d2[i]

        choose = np.zeros(n2, dtype=bool)
        z_choose = np.ones(len(R), dtype=bool)
        rng = self.random_state
        while choose.sum() < K:
            cand = np.where(z_choose)[0]
            if cand.size == 0:
                rest = np.where(~choose)[0]
                take = rng.permutation(rest)[: K - int(choose.sum())]
                choose[take] = True
                break
            cand = cand[rho[cand] == rho[cand].min()]
            j = int(rng.choice(cand))
            I = np.where(~choose & (pi[n1:] == j))[0]
            if I.size:
                if rho[j] == 0:
                    s = I[np.argmin(g[n1 + I])]
                else:
                    s = rng.choice(I)
                choose[int(s)] = True
                rho[j] += 1
            else:
                z_choose[j] = False

        crd_all = rho[pi]
        crd = np.concatenate([crd_all[:n1], crd_all[n1 + np.where(choose)[0]]]).astype(float)
        return np.where(choose)[0], crd

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)


ALGORITHMS = {"DEA-GNG": DEAGNG}
