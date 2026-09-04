from __future__ import annotations

"""
Sparse real-world multi/many-objective benchmarks (community problems).

Included
--------
- Sparse_CD
- Sparse_CN
- Sparse_FS
- Sparse_IS
- Sparse_KP
- Sparse_NN
- Sparse_PM
- Sparse_PO
- Sparse_SR
"""

from pathlib import Path

import numpy as np
from core.problem import Problem

try:
    from scipy.io import loadmat, savemat
    from scipy import sparse as scipy_sparse
except Exception:  # pragma: no cover - optional dependency
    loadmat = None
    savemat = None
    scipy_sparse = None

try:
    from sklearn.svm import SVC
except Exception:  # pragma: no cover - optional dependency
    SVC = None


def _require_scipy():
    if loadmat is None or savemat is None:
        raise RuntimeError("scipy is required for sparse_realworld_mops problems (scipy.io).")


def _require_sklearn():
    if SVC is None:
        raise RuntimeError("scikit-learn is required for Sparse_IS (sklearn.svm.SVC).")


def _as_2d(x, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _module_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / "problems"
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        target = candidate / "Multi-objective optimization" / "Real-world MOPs"
        if target.is_dir():
            return target
    raise FileNotFoundError("MATLAB source folder for Real-world MOPs not found.")


def _rng():
    # Deterministic cache generation when MATLAB cache is absent.
    return np.random.default_rng(1)


def _euclidean_cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a**2, axis=1, keepdims=True)
    bb = np.sum(b**2, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
    return np.sqrt(d2)


def _load_dataset_field(mat_file: str, field: str):
    _require_scipy()
    data = loadmat(_module_dir() / mat_file, squeeze_me=True, struct_as_record=False)
    dataset = data["Dataset"]
    return getattr(dataset, field)


def _to_dense(a) -> np.ndarray:
    if scipy_sparse is not None and scipy_sparse.issparse(a):
        return np.asarray(a.toarray(), dtype=float)
    if hasattr(a, "toarray"):
        return np.asarray(a.toarray(), dtype=float)
    return np.asarray(a, dtype=float)


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    den = mx - mn
    den[den == 0] = 1.0
    return (x - mn[None, :]) / den[None, :]


def _pairwise_connectivity(a: np.ndarray) -> float:
    a = np.asarray(a != 0, dtype=bool)
    n = a.shape[0]
    remain = np.ones(n, dtype=bool)
    f = 0.0
    while np.any(remain):
        c = np.zeros(n, dtype=bool)
        c[np.argmax(remain)] = True
        while True:
            c1 = np.any(a[c, :], axis=0) & remain
            if np.array_equal(c, c1):
                break
            c = c1
        remain &= ~c
        s = int(np.sum(c))
        f += s * (s - 1) / 2.0
    return float(f)


def _sparse_cd_random_walk_distance(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=float)
    n = adj.shape[0]
    d = np.diag(np.sum(adj, axis=1))
    laplace = d - adj
    e = np.ones((n, 1), dtype=float)
    degree = float(np.sum(adj))
    proj = (e @ e.T) / n
    lp = np.linalg.inv(laplace - proj) + proj
    act = np.zeros((n, n), dtype=float)
    for i in range(n):
        ai = lp[i, :]
        for j in range(n):
            bi = lp[j, :]
            ci = ai - bi
            act[i, j] = degree * (ci[i] - ci[j])
    return act


def _sparse_cd_decode(pop_dec: np.ndarray, act: np.ndarray, n_nodes: int) -> list[np.ndarray]:
    x = np.asarray(pop_dec, dtype=bool).reshape(-1)
    all_idx = np.arange(n_nodes, dtype=int)
    if np.sum(x) <= 1:
        return [all_idx]

    seed_idx = all_idx[x]
    other_idx = all_idx[~x]
    tmp = act[np.ix_(seed_idx, other_idx)]
    den = np.sum(tmp, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.divide(tmp, den, out=np.zeros_like(tmp), where=np.abs(den) > 0).T
    assign = np.argmax(u, axis=1)
    communities: list[np.ndarray] = []
    for k in range(seed_idx.size):
        members = other_idx[assign == k]
        communities.append(np.concatenate([[seed_idx[k]], members]).astype(int))
    return communities


def _sparse_cd_kkm(adj: np.ndarray, communities: list[np.ndarray]) -> float:
    adj = np.asarray(adj, dtype=float)
    num_var = adj.shape[0]
    cf = 0.0
    for idx in communities:
        idx = np.asarray(idx, dtype=int)
        s = adj[np.ix_(idx, idx)]
        card = s.shape[0]
        if card > 0:
            kins_sum = 0.0
            for j in range(card):
                kins_sum += float(np.sum(s[j, :]))
            cf += kins_sum / card
    return float(2.0 * (num_var - len(communities)) - cf)


def _sparse_cd_rc(adj: np.ndarray, communities: list[np.ndarray]) -> float:
    adj = np.asarray(adj, dtype=float)
    cs = 0.0
    for idx in communities:
        idx = np.asarray(idx, dtype=int)
        s = adj[np.ix_(idx, idx)]
        card = s.shape[0]
        if card > 0:
            kouts_sum = 0.0
            for j in range(card):
                kins = float(np.sum(s[j, :]))
                ksum = float(np.sum(adj[idx[j], :]))
                kouts_sum += (ksum - kins)
            cs += kouts_sum / card
    return float(cs)


def _load_or_create_sparse_kp(m: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    _require_scipy()
    path = _module_dir() / f"Dataset_KP-M{m}-D{d}.mat"
    if path.exists():
        data = loadmat(path)
        return np.asarray(data["P"], dtype=float), np.asarray(data["W"], dtype=float)
    rng = _rng()
    p = rng.integers(10, 101, size=(m, d)).astype(float)
    w = rng.integers(10, 101, size=(m, d)).astype(float)
    savemat(path, {"P": p, "W": w})
    return p, w


def _nn_predict(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = 1.0 - 2.0 / (1.0 + np.exp(2.0 * (np.column_stack([np.ones(x.shape[0]), x]) @ w1)))
    z = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(y.shape[0]), y]) @ w2)))
    return z, y


def _nn_train_one_epoch(x: np.ndarray, t: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z, y = _nn_predict(x, w1, w2)
    p = (z - t) * z * (1.0 - z)
    q = (p @ w2[1:, :].T) * (1.0 - y**2)
    d1 = np.zeros_like(w1)
    d2 = np.zeros_like(w2)
    for i in range(x.shape[0]):
        d2 += np.outer(np.concatenate([[1.0], y[i, :]]), p[i, :])
        d1 += np.outer(np.concatenate([[1.0], x[i, :]]), q[i, :])
    w1 = w1 - d1 / max(1, x.shape[0])
    w2 = w2 - d2 / max(1, x.shape[0])
    return w1, w2


def _load_or_create_sparse_pm_dataset(
    num_tran: int, len_tran: int, num_pa: int, len_pa: int, num_item: int
) -> np.ndarray:
    _require_scipy()
    path = _module_dir() / (
        f"Dataset_PM-D{num_tran}-T{len_tran}-L{num_pa}-I{len_pa}-N{num_item}.mat"
    )
    if path.exists():
        data = loadmat(path)
        return np.asarray(data["Data"], dtype=bool)

    rng = _rng()
    pa_set: list[np.ndarray] = []
    rand1 = np.clip(rng.poisson(len_pa, size=num_pa), 1, num_item).astype(int)
    rand2 = np.clip(rng.exponential(0.5, size=num_pa), 0.0, 1.0)
    pa_set.append(rng.permutation(num_item)[: rand1[0]])
    for i in range(1, num_pa):
        prev = pa_set[i - 1]
        keep = min(prev.size, int(np.round(rand2[i] * rand1[i])))
        head = prev[rng.permutation(prev.size)[:keep]] if keep > 0 else np.empty(0, dtype=int)
        tail_n = int(np.round((1.0 - rand2[i]) * rand1[i]))
        tail = rng.permutation(num_item)[:tail_n]
        pa_set.append(np.concatenate([head, tail]).astype(int))

    rand3 = np.maximum(0.0, rng.exponential(1.0, size=num_pa))
    if np.sum(rand3) <= 0:
        rand3 = np.linspace(1, num_pa, num_pa, dtype=float)
    rand3 = np.cumsum(rand3) / np.sum(rand3)
    rand4 = rng.normal(0.5, 0.1, size=num_pa)
    universe = np.unique(np.concatenate(pa_set)) if pa_set else np.arange(num_item)
    max_len = min(num_item, max(1, universe.size))
    rand5 = np.clip(rng.poisson(len_tran, size=num_tran), 1, max_len).astype(int)

    data = np.zeros((num_tran, num_item), dtype=bool)
    for i in range(num_tran):
        while int(np.sum(data[i])) < int(rand5[i]):
            current = int(np.searchsorted(rand3, rng.random(), side="left"))
            current = min(max(current, 0), num_pa - 1)
            pat = pa_set[current]
            if pat.size == 0:
                continue
            mask = rng.random(pat.size) > rand4[current]
            if np.any(mask):
                data[i, pat[mask]] = True
    savemat(path, {"Data": data.astype(np.uint8)})
    return data


def _orthonormal_rows(a: np.ndarray) -> np.ndarray:
    # MATLAB uses orth(A')'. This keeps row-orthonormal shape (M x N) for M<=N.
    q, _ = np.linalg.qr(a.T)
    return q.T


def _load_or_create_sparse_sr_dataset(
    n_sig: int, n_obs: int, sparsity: int, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_scipy()
    path = _module_dir() / f"Dataset_SR-N{n_sig}-M{n_obs}-K{sparsity}-sigma{sigma:.2f}.mat"
    if path.exists():
        data = loadmat(path)
        a = np.asarray(data["A"], dtype=float)
        b = np.asarray(data["b"], dtype=float).reshape(-1)
        x_true = np.asarray(data["x_true"], dtype=float).reshape(-1)
        return a, b, x_true

    rng = _rng()
    x_true = np.zeros(n_sig, dtype=float)
    q = rng.permutation(n_sig)
    x_true[q[: sparsity]] = 2.0 * rng.standard_normal(sparsity)
    err = float(sigma) * rng.standard_normal(n_obs)
    a = rng.standard_normal((n_obs, n_sig))
    a = _orthonormal_rows(a)
    b = a @ x_true
    norm_a = np.linalg.norm(a)
    if norm_a <= 0:
        norm_a = 1.0
    a = a / norm_a
    b = b / norm_a + err
    savemat(
        path,
        {"A": a, "b": b[:, None], "x_true": x_true[:, None], "N": n_sig, "M": n_obs, "K": sparsity, "sigma": sigma},
    )
    return a, b, x_true


class Sparse_CD(Problem):
    """Community detection problem.

    Ref: Tian, Lu, Zhang, Tan, Jin, IEEE Transactions on Cybernetics, 2021.
    """

    _DATASETS = ("Karate", "Dolphin", "Polbook", "Football")

    def __init__(self, data_no: int = 1, **kwargs):
        data_no = int(data_no)
        name = self._DATASETS[min(max(data_no, 1), len(self._DATASETS)) - 1]
        self.Adj = _to_dense(_load_dataset_field("Dataset_CD.mat", name))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.ACT = np.power(_sparse_cd_random_walk_distance(self.Adj), -2.0)
        d = int(self.Adj.shape[1])

        c = _sparse_cd_decode(np.array([1] + [0] * (d - 1), dtype=bool), self.ACT, d)
        self.MaxKKM = _sparse_cd_kkm(self.Adj, c)
        self.MinRC = _sparse_cd_rc(self.Adj, c)
        c = _sparse_cd_decode(np.ones(d, dtype=bool), self.ACT, d)
        self.MinKKM = _sparse_cd_kkm(self.Adj, c)
        self.MaxRC = _sparse_cd_rc(self.Adj, c)

        super().__init__(n_var=d, n_obj=2, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (_as_2d(x, float) >= 0.5)
        f = np.zeros((x.shape[0], 2), dtype=float)
        for i in range(x.shape[0]):
            c = _sparse_cd_decode(x[i], self.ACT, self.n_var)
            f[i, 0] = _sparse_cd_kkm(self.Adj, c)
            f[i, 1] = _sparse_cd_rc(self.Adj, c)
        den1 = self.MaxKKM - self.MinKKM
        den2 = self.MaxRC - self.MinRC
        if abs(den1) > 0:
            f[:, 0] = (f[:, 0] - self.MinKKM) / den1
        if abs(den2) > 0:
            f[:, 1] = (f[:, 1] - self.MinRC) / den2
        out["F"] = f


class Sparse_CN(Problem):
    """Critical node detection problem.

    Ref: Tian, Zhang, Wang, Jin, IEEE Transactions on Evolutionary Computation, 2020.
    """

    _DATASETS = ("Movies", "GD99", "GD01", "GD97")

    def __init__(self, data_no: int = 1, **kwargs):
        data_no = int(data_no)
        name = self._DATASETS[min(max(data_no, 1), len(self._DATASETS)) - 1]
        self.A = _to_dense(_load_dataset_field("Dataset_CN.mat", name))
        self._base_pc = max(_pairwise_connectivity(self.A), 1e-30)
        d = int(self.A.shape[0])
        super().__init__(n_var=d, n_obj=2, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (_as_2d(x, float) >= 0.5)
        f = np.zeros((x.shape[0], 2), dtype=float)
        for i in range(x.shape[0]):
            keep = ~x[i]
            f[i, 0] = np.mean(x[i])
            f[i, 1] = _pairwise_connectivity(self.A[np.ix_(keep, keep)]) / self._base_pc
        out["F"] = f


class Sparse_FS(Problem):
    """Feature selection problem.

    Ref: Tian, Zhang, Wang, Jin, IEEE Transactions on Evolutionary Computation, 2020.
    """

    _DATASETS = ("MUSK1", "Semeion_handwritten_digit", "LSVT_voice_rehabilitation")

    def __init__(self, data_no: int = 1, **kwargs):
        data_no = int(data_no)
        name = self._DATASETS[min(max(data_no, 1), len(self._DATASETS)) - 1]
        data = _to_dense(_load_dataset_field("Dataset_FS.mat", name))
        data[:, :-1] = _normalize_columns(data[:, :-1])
        self.Category = np.unique(data[:, -1])
        split = int(np.ceil(data.shape[0] * 0.8))
        self.TrainIn = data[:split, :-1]
        self.TrainOut = data[:split, -1]
        self.ValidIn = data[split:, :-1]
        self.ValidOut = data[split:, -1]
        super().__init__(n_var=self.TrainIn.shape[1], n_obj=2, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (_as_2d(x, float) >= 0.5)
        f = np.zeros((x.shape[0], 2), dtype=float)
        for i in range(x.shape[0]):
            sel = x[i]
            dists = _euclidean_cdist(self.ValidIn[:, sel], self.TrainIn[:, sel]) if self.ValidIn.size else np.zeros((0, self.TrainIn.shape[0]))
            rank = np.argsort(dists, axis=1)[:, : min(3, self.TrainIn.shape[0])]
            neigh = self.TrainOut[rank]
            votes = np.column_stack([(neigh == c).sum(axis=1) for c in self.Category]) if neigh.size else np.zeros((self.ValidIn.shape[0], len(self.Category)))
            pred = self.Category[np.argmax(votes, axis=1)] if votes.shape[0] else np.array([], dtype=self.ValidOut.dtype)
            f[i, 0] = np.mean(sel)
            f[i, 1] = np.mean(pred != self.ValidOut) if self.ValidOut.size else 0.0
        out["F"] = f


class Sparse_IS(Problem):
    """Instance selection problem.

    Ref: Tian, Lu, Zhang, Tan, Jin, IEEE Transactions on Cybernetics, 2021.
    """

    _DATASETS = ("Fourclass", "Abalone", "Phishing")

    def __init__(self, data_no: int = 1, **kwargs):
        _require_sklearn()
        data_no = int(data_no)
        name = self._DATASETS[min(max(data_no, 1), len(self._DATASETS)) - 1]
        data = _to_dense(_load_dataset_field("Dataset_IS.mat", name))
        data[:, :-1] = _normalize_columns(data[:, :-1])
        self.Data = data
        super().__init__(n_var=self.Data.shape[0], n_obj=2, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (_as_2d(x, float) >= 0.5)
        f = np.zeros((x.shape[0], 2), dtype=float)
        feat = self.Data[:, :-1]
        label = self.Data[:, -1]
        for i in range(x.shape[0]):
            sel = x[i]
            ratio = float(np.mean(sel))
            train_in = feat[sel]
            train_out = label[sel]
            valid_in = feat[~sel]
            valid_out = label[~sel]
            if ratio == 1.0:
                f[i, 0] = 1.0
                f[i, 1] = 0.0
            elif np.unique(train_out).size == 1 and train_out.size > 0:
                f[i, 0] = ratio
                f[i, 1] = 1.0
            elif ratio > 0.0 and train_out.size > 0:
                model = SVC()
                model.fit(train_in, train_out)
                pred = model.predict(valid_in) if valid_in.shape[0] > 0 else np.array([], dtype=label.dtype)
                f[i, 0] = ratio
                f[i, 1] = float(np.mean(pred != valid_out)) if valid_out.size else 0.0
            else:
                f[i, 0] = 0.0
                f[i, 1] = 1.0
        out["F"] = f


class Sparse_KP(Problem):
    """Sparse multi/many-objective knapsack problem.

    Ref: Su, Jin, Tian, Zhang, Tan, IEEE Computational Intelligence Magazine, 2022.
    """

    def __init__(self, n_var: int = 250, n_obj: int = 2, **kwargs):
        n_var = int(n_var)
        n_obj = int(max(2, n_obj))
        self.P, self.W = _load_or_create_sparse_kp(n_obj, n_var)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (_as_2d(x, float) >= 0.5).astype(float)
        f = np.sum(self.P, axis=1)[None, :] - x @ self.P.T
        viol = x @ self.W.T - 0.1 * np.sum(self.W, axis=1)[None, :]
        penalty = 10.0 * np.max(np.maximum(0.0, viol), axis=1, keepdims=True)
        out["F"] = f + penalty


class Sparse_NN(Problem):
    """Neural network training problem.

    Ref: Tian, Zhang, Wang, Jin, IEEE Transactions on Evolutionary Computation, 2020.
    """

    _DATASETS = ("Statlog_Australian", "Climate", "Statlog_German", "Connectionist_bench_Sonar")

    def __init__(self, data_no: int = 1, n_hidden: int = 20, **kwargs):
        data_no = int(data_no)
        self.nHidden = int(n_hidden)
        name = self._DATASETS[min(max(data_no, 1), len(self._DATASETS)) - 1]
        data = _to_dense(_load_dataset_field("Dataset_NN.mat", name))
        mean = np.mean(data[:, :-1], axis=0)
        std = np.std(data[:, :-1], axis=0)
        std[std == 0] = 1.0
        inp = (data[:, :-1] - mean[None, :]) / std[None, :]
        category = np.unique(data[:, -1])
        if category.size <= 2:
            out_m = (data[:, -1] == category[0]).astype(float)[:, None]
        else:
            out_m = (data[:, -1][:, None] == category[None, :]).astype(float)

        split = int(np.ceil(inp.shape[0] * 0.8))
        self.TrainIn = inp[:split, :]
        self.TrainOut = out_m[:split, :]
        self.TestIn = inp[split:, :]
        self.TestOut = out_m[split:, :]
        if category.size <= 2:
            self.TrainLabel = self.TrainOut.reshape(-1)
            self.TestLabel = self.TestOut.reshape(-1)
        else:
            self.TrainLabel = np.argmax(self.TrainOut, axis=1) + 1
            self.TestLabel = np.argmax(self.TestOut, axis=1) + 1

        d = (self.TrainIn.shape[1] + 1) * self.nHidden + (self.nHidden + 1) * self.TrainOut.shape[1]
        super().__init__(n_var=int(d), n_obj=2, xl=-1.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = _as_2d(x, float)
        f = np.zeros((x.shape[0], 2), dtype=float)
        n_in = self.TrainIn.shape[1]
        n_out = self.TrainOut.shape[1]
        cut = (n_in + 1) * self.nHidden
        for i in range(x.shape[0]):
            w1 = x[i, :cut].reshape(n_in + 1, self.nHidden)
            w2 = x[i, cut:].reshape(self.nHidden + 1, n_out)
            z, _ = _nn_predict(self.TrainIn, w1, w2)
            f[i, 0] = np.mean(x[i] != 0)
            if z.shape[1] == 1:
                pred = np.round(z.reshape(-1))
            else:
                pred = np.argmax(z, axis=1) + 1
            f[i, 1] = np.mean(pred != self.TrainLabel)
        out["F"] = f


class Sparse_PM(Problem):
    """Pattern mining problem.

    Ref: Tian, Zhang, Wang, Jin, IEEE Transactions on Evolutionary Computation, 2020.
    """

    def __init__(
        self,
        num_tran: int = 10000,
        len_tran: int = 50,
        num_pa: int = 100,
        len_pa: int = 5,
        num_item: int = 100,
        **kwargs,
    ):
        self.Data = _load_or_create_sparse_pm_dataset(
            int(num_tran), int(len_tran), int(num_pa), int(len_pa), int(num_item)
        )
        super().__init__(n_var=int(num_item), n_obj=2, xl=0.0, xu=1.0, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = (_as_2d(x, float) >= 0.5)
        f = np.zeros((x.shape[0], 2), dtype=float)
        for i in range(x.shape[0]):
            sel = x[i]
            tx = np.all(self.Data[:, sel], axis=1)
            if not np.any(tx):
                f[i, :] = 1.0
            else:
                f[i, 0] = 1.0 - np.mean(tx)
                occ_den = np.sum(self.Data[tx, :], axis=1)
                occ_den = np.maximum(occ_den, 1)
                f[i, 1] = 1.0 - np.mean(np.sum(sel) / occ_den)
        out["F"] = f


class Sparse_PO(Problem):
    """Portfolio optimization problem.

    Ref: Tian, Lu, Zhang, Tan, Jin, IEEE Transactions on Cybernetics, 2021.
    """

    _DATASETS = ("data1000", "data5000")

    def __init__(self, data_no: int = 1, **kwargs):
        data_no = int(data_no)
        name = self._DATASETS[min(max(data_no, 1), len(self._DATASETS)) - 1]
        data = _to_dense(_load_dataset_field("Dataset_PO.mat", name))
        self.Yield = np.log(data[:, 1:]) - np.log(data[:, :-1])
        self.Risk = np.cov(self.Yield)
        super().__init__(n_var=self.Yield.shape[0], n_obj=2, xl=-1.0, xu=1.0, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = _as_2d(x, float)
        den = np.maximum(np.sum(np.abs(x), axis=1, keepdims=True), 1.0)
        x = x / den
        f = np.zeros((x.shape[0], 2), dtype=float)
        for i in range(x.shape[0]):
            xi = x[i, :]
            f[i, 0] = xi @ self.Risk @ xi.T
            f[i, 1] = 1.0 - np.sum(xi @ self.Yield)
        out["F"] = f


class Sparse_SR(Problem):
    """Sparse signal reconstruction problem.

    Ref: Tian, Lu, Zhang, Tan, Jin, IEEE Transactions on Cybernetics, 2021.
    """

    def __init__(
        self,
        len_sig: int = 1024,
        len_obs: int = 480,
        sparsity: int = 260,
        sigma: float = 0.0,
        **kwargs,
    ):
        self.A, self.b, self.x_true = _load_or_create_sparse_sr_dataset(
            int(len_sig), int(len_obs), int(sparsity), float(sigma)
        )
        nz = self.x_true[self.x_true != 0]
        mu = float(np.mean(nz)) if nz.size else 0.0
        sd = float(np.std(nz)) if nz.size else 1.0
        lower = float(np.floor(mu - 3.0 * sd))
        upper = float(np.ceil(mu + 3.0 * sd))
        self.Total = float(np.linalg.norm(self.A @ np.zeros(self.A.shape[1]) - self.b))
        if self.Total <= 0:
            self.Total = 1.0
        super().__init__(n_var=self.A.shape[1], n_obj=2, xl=lower, xu=upper, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = _as_2d(x, float)
        xb = (x != 0).astype(float)  # MATLAB casts to logical before objective computation.
        f = np.zeros((x.shape[0], 2), dtype=float)
        f[:, 0] = np.sum(xb != 0, axis=1) / self.n_var
        for i in range(x.shape[0]):
            f[i, 1] = np.linalg.norm(self.A @ xb[i, :] - self.b) / self.Total
        out["F"] = f


_CPU = [
    "Sparse_CD",
    "Sparse_CN",
    "Sparse_FS",
    "Sparse_IS",
    "Sparse_KP",
    "Sparse_NN",
    "Sparse_PM",
    "Sparse_PO",
    "Sparse_SR",
]

for _name in _CPU:
    _base = globals()[_name]
    globals()[f"{_name}_JAX"] = type(f"{_name}_JAX", (_base,), {})


__all__ = _CPU + [f"{n}_JAX" for n in _CPU]
