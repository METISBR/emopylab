from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.problem import Problem

try:
    import jax.numpy as jnp
except Exception:  # noqa: BLE001
    jnp = None

try:
    import scipy.io as sio
except Exception as _scipy_io_exc:  # noqa: BLE001
    sio = None
    _SCIPY_IO_IMPORT_ERROR = _scipy_io_exc
else:
    _SCIPY_IO_IMPORT_ERROR = None

try:
    from scipy import sparse
except Exception as _scipy_sparse_exc:  # noqa: BLE001
    sparse = None
    _SCIPY_SPARSE_IMPORT_ERROR = _scipy_sparse_exc
else:
    _SCIPY_SPARSE_IMPORT_ERROR = None


def _xp(use_jax: bool):
    if use_jax and jnp is not None:
        return jnp
    return np


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_2d(x: Any, xp):
    arr = xp.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[xp.newaxis, :]
    return arr


def _clip_x(x: Any, xl: Any, xu: Any, xp):
    arr = _as_2d(x, xp)
    return xp.clip(arr, xp.asarray(xl, dtype=float), xp.asarray(xu, dtype=float))


def _problems_matlab_community_root() -> Path:
    here = Path(__file__).resolve()
    problems_root = here.parents[1]
    for cand in problems_root.iterdir():
        if cand.is_dir() and (cand / "Single-objective optimization").exists():
            return cand
    raise RuntimeError("Could not locate MATLAB community problems folder.")


def _single_family_dir(family_name: str) -> Path:
    return _problems_matlab_community_root() / "Single-objective optimization" / family_name


def _require_scipy_io() -> None:
    if sio is None:
        raise RuntimeError(f"scipy.io is required for this problem: {_SCIPY_IO_IMPORT_ERROR}")


def _require_scipy_sparse() -> None:
    if sparse is None:
        raise RuntimeError(f"scipy.sparse is required for this problem: {_SCIPY_SPARSE_IMPORT_ERROR}")


def _load_mat_struct(path: Path, key: str = "Dataset"):
    _require_scipy_io()
    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if key not in data:
        raise RuntimeError(f"Key '{key}' not found in MAT file: {path}")
    return data[key]


def _community_decode(labels: np.ndarray) -> list[np.ndarray]:
    dec = np.asarray(labels, dtype=int).ravel().copy()
    communities: list[np.ndarray] = []
    while np.any(dec):
        idx0 = int(np.flatnonzero(dec)[0])
        value = int(dec[idx0])
        current = np.flatnonzero(dec == value)
        communities.append(current)
        dec[current] = 0
    return communities


def _community_canonical_relabel(PopDec: np.ndarray) -> np.ndarray:
    X = np.asarray(PopDec, dtype=int).copy()
    n, d = X.shape
    for i in range(n):
        labels = X[i]
        out = np.zeros(d, dtype=int)
        next_label = 1
        while np.any(out == 0):
            x = int(np.flatnonzero(out == 0)[0])
            out[labels == labels[x]] = next_label
            next_label += 1
        X[i] = out
    return X


def _modularity(adj: np.ndarray, communities: list[np.ndarray]) -> float:
    adj = np.asarray(adj, dtype=float)
    M = float(np.sum(adj) / 2.0)
    if M <= 0:
        return 0.0
    q = 0.0
    for c in communities:
        sub = adj[np.ix_(c, c)]
        q += float(np.sum(sub) / 2.0 / M)
        q -= float((np.sum(adj[c, :]) / 2.0 / M) ** 2)
    return q


def _rank_to_permutation(row: np.ndarray) -> np.ndarray:
    d = row.size
    try_row = np.asarray(row).astype(int)
    if np.array_equal(np.sort(try_row), np.arange(1, d + 1)):
        return try_row
    order = np.argsort(np.asarray(row, dtype=float), kind="mergesort")
    return (order + 1).astype(int)


def _pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


@dataclass(frozen=True)
class _RealworldRef:
    name: str
    reference: str


class _BaseSingleLocal(Problem):
    _USE_JAX = False

    def _xp(self):
        return _xp(self._USE_JAX)

    def _prepare(self, x):
        xp = self._xp()
        x = _clip_x(x, self.xl, self.xu, xp)
        return xp, x


class CommunityDetection(_BaseSingleLocal):
    """
    Community detection (label encoding).

    Reference:
    Y. Tian, S. Yang, and X. Zhang. An evolutionary multiobjective optimization
    based fuzzy method for overlapping community detection. IEEE Transactions
    on Fuzzy Systems, 2020, 28(11): 2841-2855.
    """

    _DATASET_NAMES = ("Karate", "Dolphin", "Polbook", "Football")

    def __init__(self, data_no: int = 1, **kwargs):
        self.data_no = int(data_no)
        self._family_dir = _single_family_dir("Real-world SOPs")
        ds = _load_mat_struct(self._family_dir / "Dataset_CD.mat", "Dataset")
        if not (1 <= self.data_no <= len(self._DATASET_NAMES)):
            raise ValueError(f"data_no must be in [1,{len(self._DATASET_NAMES)}].")
        key = self._DATASET_NAMES[self.data_no - 1]
        self.Adj = np.asarray(getattr(ds, key), dtype=float)
        self.D = int(self.Adj.shape[1])
        super().__init__(
            n_var=self.D,
            n_obj=1,
            xl=np.ones(self.D, dtype=float),
            xu=np.full(self.D, self.D, dtype=float),
            vtype=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        x_np = np.clip(np.rint(x_np), 1, self.D).astype(int)
        x_np = _community_canonical_relabel(x_np)
        f = np.zeros(x_np.shape[0], dtype=float)
        for i in range(x_np.shape[0]):
            q = _modularity(self.Adj, _community_decode(x_np[i]))
            f[i] = 1.0 - q
        out["F"] = f.reshape(-1, 1)


class KP(_BaseSingleLocal):
    """
    Knapsack problem.

    Reference:
    E. Zitzler and L. Thiele. Multiobjective evolutionary algorithms:
    A comparative case study and the strength Pareto approach. IEEE TEC,
    1999, 3(4): 257-271.
    """

    def __init__(self, n_var: int = 250, seed: int | None = None, **kwargs):
        self._family_dir = _single_family_dir("Real-world SOPs")
        d = int(max(1, n_var))
        self._seed = int(seed) if seed is not None else (10_000 + d)
        self.P, self.W = self._load_or_create_data(d)
        super().__init__(n_var=d, n_obj=1, n_ieq_constr=1, xl=np.zeros(d), xu=np.ones(d), vtype=int, **kwargs)

    def _load_or_create_data(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        file = self._family_dir / f"KP-D{d}.mat"
        if file.exists():
            _require_scipy_io()
            data = sio.loadmat(str(file), squeeze_me=True)
            P = np.asarray(data["P"], dtype=float).reshape(-1)
            W = np.asarray(data["W"], dtype=float).reshape(-1)
            return P, W
        rng = np.random.default_rng(self._seed)
        P = rng.integers(10, 101, size=d).astype(float)
        W = rng.integers(10, 101, size=d).astype(float)
        if sio is not None:
            try:
                sio.savemat(str(file), {"P": P.reshape(1, -1), "W": W.reshape(1, -1)})
            except Exception:  # noqa: BLE001
                pass
        return P, W

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        x_bin = (x_np > 0.5).astype(float)
        total_profit = float(np.sum(self.P))
        f = total_profit - x_bin @ self.P.reshape(-1, 1)
        g = x_bin @ self.W.reshape(-1, 1) - float(np.sum(self.W) / 2.0)
        out["F"] = f.reshape(-1, 1)
        out["G"] = g.reshape(-1, 1)


class MaxCut(_BaseSingleLocal):
    """
    Max-cut problem (objective is negative cut, consistent with minimization convention).

    Reference:
    Y. Tian, L. Wang, S. Yang, J. Ding, Y. Jin, and X. Zhang. Neural
    network-based dimensionality reduction for large-scale binary optimization
    with millions of variables. IEEE TEC, 2025, 29(6): 2328-2342.
    """

    _DATASET_NAMES = ("D941", "D2344", "D5000")

    def __init__(self, data_no: int = 1, **kwargs):
        _require_scipy_sparse()
        self.data_no = int(data_no)
        self._family_dir = _single_family_dir("Real-world SOPs")
        ds = _load_mat_struct(self._family_dir / "Dataset_MC.mat", "Dataset")
        if not (1 <= self.data_no <= len(self._DATASET_NAMES)):
            raise ValueError(f"data_no must be in [1,{len(self._DATASET_NAMES)}].")
        key = self._DATASET_NAMES[self.data_no - 1]
        self.Adj_mat = np.asarray(getattr(ds, key), dtype=float)
        self.Tri_mat = sparse.triu(sparse.csr_matrix(self.Adj_mat), k=1).tocsr()
        self._tri_colsum = np.asarray(self.Tri_mat.sum(axis=0)).ravel()
        self._tri_rowsum = np.asarray(self.Tri_mat.sum(axis=1)).ravel()
        d = int(self.Adj_mat.shape[0])
        super().__init__(n_var=d, n_obj=1, xl=np.zeros(d), xu=np.ones(d), vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        X = (x_np > 0.5).astype(float)
        XT = X @ self.Tri_mat
        term1 = 2.0 * np.sum(XT * X, axis=1)
        term2 = X @ self._tri_colsum.reshape(-1, 1)
        term3 = X @ self._tri_rowsum.reshape(-1, 1)
        f = term1.reshape(-1, 1) - term2.reshape(-1, 1) - term3.reshape(-1, 1)
        out["F"] = np.asarray(f, dtype=float)


class TSP(_BaseSingleLocal):
    """
    Traveling salesman problem.

    Reference:
    D. Corne and J. Knowles. Techniques for highly multiobjective optimisation:
    some nondominated points are better than others. GECCO 2007, 773-780.
    """

    def __init__(self, n_var: int = 30, seed: int | None = None, **kwargs):
        self._family_dir = _single_family_dir("Real-world SOPs")
        d = int(max(2, n_var))
        self._seed = int(seed) if seed is not None else (20_000 + d)
        self.R = self._load_or_create_coords(d)
        self.C = _pairwise_euclidean(self.R)
        super().__init__(n_var=d, n_obj=1, xl=np.ones(d), xu=np.full(d, d, dtype=float), vtype=float, **kwargs)

    def _load_or_create_coords(self, d: int) -> np.ndarray:
        file = self._family_dir / f"TSP-D{d}.mat"
        if file.exists():
            _require_scipy_io()
            data = sio.loadmat(str(file), squeeze_me=True)
            return np.asarray(data["R"], dtype=float)
        rng = np.random.default_rng(self._seed)
        R = rng.random((d, 2), dtype=float)
        if sio is not None:
            try:
                sio.savemat(str(file), {"R": R})
            except Exception:  # noqa: BLE001
                pass
        return R

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        perms = np.zeros_like(x_np, dtype=int)
        for i in range(x_np.shape[0]):
            perms[i] = _rank_to_permutation(x_np[i])
        idx = perms - 1
        rolled = np.roll(idx, -1, axis=1)
        f = np.sum(self.C[idx, rolled], axis=1)
        out["F"] = np.asarray(f, dtype=float).reshape(-1, 1)


_CPU = ["CommunityDetection", "KP", "MaxCut", "TSP"]
for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {"_USE_JAX": True})


__all__ = _CPU + [f"{name}_JAX" for name in _CPU]
