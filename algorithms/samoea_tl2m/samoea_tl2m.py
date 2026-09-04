# -*- coding: utf-8 -*-
"""
SAMOEA-TL2M (EmoPyLab-compatible)
==============================

Reference publication:
Y. Liu, J. Ding, Q. Li, F. Li, and J. Liu,
"A Two-Level Model Management-Based Surrogate-Assisted Evolutionary
Algorithm for Medium-Scale Expensive Multiobjective Optimization,"
IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2025.
DOI: 10.1109/TSMC.2025.3604949

Core elements:
1) RBF-assisted evolutionary multi-objective search
2) Two-level model management for infill selection
3) ARI (accuracy rate indicator) for state switching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from util.array_backend import xp as np

from algorithms.moo.sms import cv_and_dom_tournament
from core.algorithm import Algorithm
from core.mating import Mating
from core.population import Population
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from operators.selection.tournament import TournamentSelection
from util.nds.non_dominated_sorting import NonDominatedSorting

ALGORITHM_FLAGS = {
    "SAMOEA_TL2M": {"multi", "many"},
}

try:
    from sklearn.cluster import KMeans

    _SKLEARN_KMEANS = True
except Exception:  # pragma: no cover
    _SKLEARN_KMEANS = False


def _force_cpu_backend(algorithm, reason: str) -> None:
    if not bool(getattr(algorithm, "use_gpu", False)):
        return
    algorithm.use_gpu = False
    algorithm.array_backend_effective = "numpy"
    state = dict(getattr(algorithm, "backend_state", {}) or {})
    state["forced_cpu_reason"] = str(reason)
    algorithm.backend_state = state


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Lightweight Latin Hypercube Sampling in [0, 1]^n_dim.
    """
    if n_samples <= 0:
        return np.empty((0, n_dim), dtype=float)
    X = np.empty((n_samples, n_dim), dtype=float)
    inv_n = 1.0 / n_samples
    base = np.arange(n_samples, dtype=float) * inv_n
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        X[:, j] = base[perm] + rng.random(n_samples) * inv_n
    return X


def _radbas(z: np.ndarray) -> np.ndarray:
    return np.exp(-(z**2))


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Euclidean distances with numeric floor.
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    D2 = np.maximum(AA + BB - 2.0 * (A @ B.T), 0.0)
    return np.sqrt(D2)


@dataclass
class _RBFModel:
    centers: Optional[np.ndarray] = None
    spreads: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    b: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_obj: int,
        n_var: int,
        rng: np.random.Generator,
    ) -> "_RBFModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(X)
        if n == 0:
            self.centers = np.empty((0, X.shape[1]), dtype=float)
            self.spreads = np.empty(0, dtype=float)
            self.w = np.empty(0, dtype=float)
            self.b = 0.0
            return self

        cluster_num = int(round(np.sqrt(float(n_obj + n_var)) + 3.0))
        cluster_num = int(np.clip(cluster_num, 2, n))

        if _SKLEARN_KMEANS and n >= cluster_num:
            km = KMeans(
                n_clusters=cluster_num,
                n_init=3,
                random_state=int(rng.integers(1, 2**31 - 1)),
            )
            km.fit(X)
            centers = km.cluster_centers_
        else:
            idx = rng.choice(np.arange(n), size=cluster_num, replace=False)
            centers = X[idx].copy()

        dcc = _pairwise_dist(centers, centers)
        np.fill_diagonal(dcc, np.inf)
        spreads = np.min(dcc, axis=1)
        good = spreads[np.isfinite(spreads) & (spreads > 1e-12)]
        fallback = float(np.median(good)) if len(good) > 0 else 1.0
        spreads = np.where(np.isfinite(spreads) & (spreads > 1e-12), spreads, fallback)
        spreads = np.maximum(spreads, 1e-8)

        dist = _pairwise_dist(centers, X)  # [k, n]
        hidden = _radbas(dist / spreads[:, None])
        H = np.hstack([hidden.T, np.ones((n, 1), dtype=float)])  # [n, k+1]

        coef, *_ = np.linalg.lstsq(H, y, rcond=None)
        self.centers = centers
        self.spreads = spreads
        self.w = coef[:-1]
        self.b = float(coef[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.centers is None or self.w is None or len(self.w) == 0:
            return np.zeros(len(X), dtype=float) + float(self.b)
        dist = _pairwise_dist(self.centers, X)  # [k, n]
        hidden = _radbas(dist / self.spreads[:, None])
        return hidden.T @ self.w + self.b


class SAMOEA_TL2M(Algorithm):
    """
    Two-level model management based surrogate-assisted MOEA.
    """
    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        pop_size: int = 100,
        initial_sample_factor: int = 11,
        initial_sample_offset: int = -1,
        infill_size: int = 5,
        surrogate_generations: int = 20,
        ari_alpha: float = 0.4,
        duplicate_tol: float = 1e-5,
        seed: Optional[int] = None,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )

        _force_cpu_backend(
            self,
            "SAMOEA_TL2M relies on sklearn/CPU surrogate stack; GPU backend is pending migration.",
        )

        self.ref_dirs = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.pop_size = int(max(pop_size, 10))
        self.initial_sample_factor = int(max(initial_sample_factor, 1))
        self.initial_sample_offset = int(initial_sample_offset)
        self.infill_size = int(max(infill_size, 1))
        self.surrogate_generations = int(max(surrogate_generations, 1))
        self.ari_alpha = float(np.clip(ari_alpha, 0.0, 1.0))
        self.duplicate_tol = float(max(duplicate_tol, 0.0))

        self.nds = NonDominatedSorting()

        self.selection = TournamentSelection(func_comp=cv_and_dom_tournament)
        self.sbx = SBX(prob=0.9, eta=20, vtype=float)
        self.pm = PolynomialMutation(prob=0.1, eta=20)
        self.mating = Mating(
            self.selection,
            self.sbx,
            self.pm,
            n_max_iterations=100,
        )

        self.ds_X: Optional[np.ndarray] = None
        self.ds_F: Optional[np.ndarray] = None
        self.ari = 1.0
        self.last_infill_X: Optional[np.ndarray] = None

    def _setup(self, problem, **kwargs):
        self.pm = PolynomialMutation(prob=min(1.0, 1.0 / max(problem.n_var, 1)), eta=20)
        self.mating = Mating(
            self.selection,
            self.sbx,
            self.pm,
            n_max_iterations=100,
        )

    def _initialize_infill(self):
        n_var = self.problem.n_var
        ni = self.initial_sample_factor * n_var + self.initial_sample_offset
        ni = int(max(ni, self.pop_size))
        unit = _lhs(ni, n_var, self.random_state)
        X = self.problem.xl + unit * (self.problem.xu - self.problem.xl)
        return Population.new("X", X)

    def _initialize_advance(self, infills=None, **kwargs):
        self._append_archive(infills.get("X"), infills.get("F"))

        ni = len(infills)
        n_pick = min(self.pop_size, ni)
        idx = self.random_state.choice(np.arange(ni), size=n_pick, replace=False)
        self.pop = infills[np.asarray(idx, dtype=int)]
        self._set_optimum()

    def _infill(self):
        if self.ds_X is None or len(self.ds_X) < max(self.pop_size, 20):
            X = self._sample_random(self.infill_size)
            self.last_infill_X = X.copy()
            return Population.new("X", X)

        cand_X, cand_F_hat, model_cd = self._surrogate_search()
        cand_X, cand_F_hat = self._remove_archive_duplicates(cand_X, cand_F_hat)

        if len(cand_X) == 0:
            X = self._sample_random(self.infill_size)
            self.last_infill_X = X.copy()
            return Population.new("X", X)

        if self.ari >= self.ari_alpha:
            pick_idx = self._level1_select(cand_F_hat, model_cd.predict(cand_X), self.infill_size)
        else:
            nu = int(np.floor(self.infill_size * (1.0 - self.ari)))
            nu = int(np.clip(nu, 1, max(self.infill_size - 1, 1)))
            nc = self.infill_size - nu

            idx_c = self._level1_select(cand_F_hat, model_cd.predict(cand_X), nc)
            mask = np.ones(len(cand_X), dtype=bool)
            mask[idx_c] = False
            rem_X = cand_X[mask]
            rem_F = cand_F_hat[mask]

            if len(rem_X) == 0:
                pick_idx = idx_c
            else:
                idw_u = self._idw_uncertainty(rem_X)
                idx_u_local = np.argsort(-idw_u)[: min(nu, len(rem_X))]
                pick_X = np.vstack([cand_X[idx_c], rem_X[idx_u_local]])
                pick_X = self._dedup_decisions(pick_X)
                if len(pick_X) < self.infill_size:
                    extra = self._sample_random(self.infill_size - len(pick_X))
                    pick_X = np.vstack([pick_X, extra])
                self.last_infill_X = pick_X[: self.infill_size].copy()
                return Population.new("X", self.last_infill_X)

        pick_X = cand_X[pick_idx]
        pick_X = self._dedup_decisions(pick_X)
        if len(pick_X) < self.infill_size:
            extra = self._sample_random(self.infill_size - len(pick_X))
            pick_X = np.vstack([pick_X, extra])
        self.last_infill_X = pick_X[: self.infill_size].copy()
        return Population.new("X", self.last_infill_X)

    def _advance(self, infills=None, **kwargs):
        if infills is None or len(infills) == 0:
            return

        self._append_archive(infills.get("X"), infills.get("F"))

        pool_X = np.vstack([self.pop.get("X"), infills.get("X")])
        pool_F = np.vstack([self.pop.get("F"), infills.get("F")])
        next_idx = self._survival_nsga2_sde(pool_F, self.pop_size)
        self.pop = Population.new("X", pool_X[next_idx], "F", pool_F[next_idx])

        self.ari = self._compute_ari(self.pop.get("X"), self.pop.get("F"))
        self._set_optimum()

    def _set_optimum(self):
        if self.pop is None or len(self.pop) == 0:
            self.opt = self.pop
            return
        nd = self.nds.do(self.pop.get("F"), only_non_dominated_front=True)
        self.opt = self.pop[np.asarray(nd, dtype=int)]

    def _surrogate_search(self) -> Tuple[np.ndarray, np.ndarray, _RBFModel]:
        ds_X = self.ds_X
        ds_F = self.ds_F
        m = self.problem.n_obj
        d = self.problem.n_var
        n = len(self.pop)

        y_min = np.min(ds_F, axis=0)
        y_max = np.max(ds_F, axis=0)
        y_span = np.maximum(y_max - y_min, 1e-12)
        y_norm = (ds_F - y_min) / y_span

        obj_models = []
        for j in range(m):
            mdl = _RBFModel().fit(ds_X, y_norm[:, j], m, d, self.random_state)
            obj_models.append(mdl)

        sde_ds = self._calc_sde(ds_F)
        sde_ds_norm = self._normalize01(sde_ds)
        model_cd = _RBFModel().fit(ds_X, sde_ds_norm, m, d, self.random_state)

        pop_dec = np.asarray(self.pop.get("X"), dtype=float).copy()
        pop_obj = self._predict_objectives(pop_dec, obj_models)

        for _ in range(self.surrogate_generations):
            off_dec = self._operator_ga(pop_dec, n_off=n)
            off_obj = self._predict_objectives(off_dec, obj_models)

            comb_dec = np.vstack([pop_dec, off_dec])
            comb_obj = np.vstack([pop_obj, off_obj])
            comb_cd = model_cd.predict(comb_dec)

            sel = self._survival_with_cd(comb_obj, comb_cd, n)
            pop_dec = comb_dec[sel]
            pop_obj = comb_obj[sel]

        return pop_dec, pop_obj, model_cd

    def _predict_objectives(self, X: np.ndarray, models: list[_RBFModel]) -> np.ndarray:
        F = np.zeros((len(X), len(models)), dtype=float)
        for j, mdl in enumerate(models):
            F[:, j] = mdl.predict(X)
        return F

    def _operator_ga(self, pop_dec: np.ndarray, n_off: int) -> np.ndarray:
        pop = Population.new("X", pop_dec)
        off = self.mating.do(
            self.problem,
            pop,
            n_off,
            algorithm=self,
            random_state=self.random_state,
        )
        if off is None or len(off) == 0:
            return self._sample_random(n_off)
        X = np.asarray(off.get("X"), dtype=float)
        if len(X) > n_off:
            X = X[:n_off]
        if len(X) < n_off:
            X = np.vstack([X, self._sample_random(n_off - len(X))])
        return np.clip(X, self.problem.xl, self.problem.xu)

    def _survival_with_cd(self, F: np.ndarray, cd: np.ndarray, n_survive: int) -> np.ndarray:
        fronts, _rank = self.nds.do(F, return_rank=True, n_stop_if_ranked=n_survive)
        selected = []
        for front in fronts:
            front = np.asarray(front, dtype=int)
            if len(selected) + len(front) <= n_survive:
                selected.extend(front.tolist())
            else:
                rem = n_survive - len(selected)
                order = np.argsort(-cd[front])  # higher CD first
                selected.extend(front[order[:rem]].tolist())
                break
        return np.asarray(selected, dtype=int)

    def _level1_select(self, F_hat: np.ndarray, cd_hat: np.ndarray, n_pick: int) -> np.ndarray:
        n_pick = int(max(n_pick, 0))
        if n_pick == 0:
            return np.empty(0, dtype=int)
        return self._survival_with_cd(F_hat, cd_hat, min(n_pick, len(F_hat)))

    def _compute_ari(self, pop_X: np.ndarray, pop_F: np.ndarray) -> float:
        if self.last_infill_X is None or len(self.last_infill_X) == 0:
            return 1.0

        preserved_mask = np.array(
            [self._is_in_matrix(x, pop_X, self.duplicate_tol) for x in self.last_infill_X],
            dtype=bool,
        )
        np_count = int(np.count_nonzero(preserved_mask))
        if np_count == 0:
            return 0.0

        nd_idx = self.nds.do(pop_F, only_non_dominated_front=True)
        nd_X = pop_X[np.asarray(nd_idx, dtype=int)]

        preserved_X = self.last_infill_X[preserved_mask]
        nn_count = int(
            np.sum([self._is_in_matrix(x, nd_X, self.duplicate_tol) for x in preserved_X], dtype=int)
        )

        rp = np_count / max(len(self.last_infill_X), 1)
        rn = nn_count / max(np_count, 1)
        return float(min(rp, rn))

    def _idw_uncertainty(self, Xq: np.ndarray) -> np.ndarray:
        # Based on Eq. (5)-(9) in the paper.
        Xq = np.asarray(Xq, dtype=float)
        d = _pairwise_dist(Xq, self.ds_X)  # [nq, nds]
        d2 = np.maximum(d * d, 1e-16)

        w = np.exp(-d2) / d2
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_sum = np.where(w_sum <= 1e-16, 1.0, w_sum)
        v = w / w_sum

        m = self.ds_F.shape[1]
        idw_total = np.zeros(len(Xq), dtype=float)

        for k in range(m):
            values = self.ds_F[:, k]
            pred = v @ values
            var = np.sum(v * ((values[None, :] - pred[:, None]) ** 2), axis=1)
            idw_total += var
        return idw_total

    def _calc_sde(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        if len(F) <= 1:
            return np.ones(len(F), dtype=float)

        Fn = self._normalize_obj(F)
        n = len(Fn)
        sde = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            shifted = np.maximum(Fn, Fn[i])
            diff = Fn[i][None, :] - shifted
            dist = np.linalg.norm(diff, axis=1)
            sde[i, :] = dist
            sde[i, i] = np.inf
        return np.min(sde, axis=1)

    def _survival_nsga2_sde(self, F: np.ndarray, n_survive: int) -> np.ndarray:
        fronts, _rank = self.nds.do(F, return_rank=True, n_stop_if_ranked=n_survive)
        selected = []
        sde = self._calc_sde(F)
        sde = self._normalize01(sde)
        for front in fronts:
            front = np.asarray(front, dtype=int)
            if len(selected) + len(front) <= n_survive:
                selected.extend(front.tolist())
            else:
                rem = n_survive - len(selected)
                order = np.argsort(-sde[front])  # best SDE first
                selected.extend(front[order[:rem]].tolist())
                break
        return np.asarray(selected, dtype=int)

    def _append_archive(self, X_new: np.ndarray, F_new: np.ndarray):
        X_new = np.asarray(X_new, dtype=float)
        F_new = np.asarray(F_new, dtype=float)
        if self.ds_X is None:
            self.ds_X = X_new.copy()
            self.ds_F = F_new.copy()
            return

        X = np.vstack([self.ds_X, X_new])
        F = np.vstack([self.ds_F, F_new])
        X_key = np.round(X, decimals=12)
        _, idx = np.unique(X_key, axis=0, return_index=True)
        idx = np.sort(idx)
        self.ds_X = X[idx]
        self.ds_F = F[idx]

    def _remove_archive_duplicates(
        self, X: np.ndarray, F: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep = []
        for i, x in enumerate(X):
            if self._is_in_matrix(x, self.ds_X, self.duplicate_tol):
                continue
            ok = True
            for j in keep:
                if np.linalg.norm(x - X[j]) <= self.duplicate_tol:
                    ok = False
                    break
            if ok:
                keep.append(i)
        if len(keep) == 0:
            return np.empty((0, X.shape[1]), dtype=float), np.empty((0, F.shape[1]), dtype=float)
        keep = np.asarray(keep, dtype=int)
        return X[keep], F[keep]

    def _sample_random(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.problem.n_var), dtype=float)
        X = self.problem.xl + self.random_state.random((n, self.problem.n_var)) * (
            self.problem.xu - self.problem.xl
        )
        return np.asarray(X, dtype=float)

    def _dedup_decisions(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return X
        out = [X[0]]
        for i in range(1, len(X)):
            if not self._is_in_matrix(X[i], np.asarray(out), self.duplicate_tol):
                out.append(X[i])
        return np.asarray(out, dtype=float)

    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo <= 1e-14:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    @staticmethod
    def _normalize_obj(F: np.ndarray) -> np.ndarray:
        lo = np.min(F, axis=0)
        hi = np.max(F, axis=0)
        span = np.maximum(hi - lo, 1e-12)
        return (F - lo) / span

    @staticmethod
    def _is_in_matrix(x: np.ndarray, X: np.ndarray, tol: float) -> bool:
        if X is None or len(X) == 0:
            return False
        d = np.linalg.norm(X - x[None, :], axis=1)
        return bool(np.min(d) <= tol)
