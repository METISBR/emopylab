# -*- coding: utf-8 -*-
"""NSGA-III with Expected Hypervolume Improvement (NSGA-III-EHVI).

Reference:
Y. Pang, Y. Wang, S. Zhang, X. Lai, W. Sun, and X. Song.
"An expensive many-objective optimization algorithm based on efficient
expected hypervolume improvement." IEEE Transactions on Evolutionary
Computation, 2023, 27(6): 1822-1836.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from util.array_backend import to_numpy
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.ref_dirs import get_reference_directions
from algorithms.community_utils.moead_family import sample_initial
from operators.utility_functions.UniformPoint import UniformPoint


ALGORITHM_FLAGS = {"NSGAIIIEHVI": {"expensive", "many", "multi", "real"}}


class _KrigingModel:
    """Zero-dependency Gaussian Process (Kriging) regressor for objective modeling."""

    def __init__(self, nugget: float = 1e-5) -> None:
        self.nugget = float(nugget)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.mu: float = 0.0
        self.sigma2: float = 1.0
        self.theta: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.denom: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self.X = X
        self.y = y

        std_x = np.std(X, axis=0)
        self.theta = 1.0 / np.maximum(std_x ** 2, 1e-3)

        diff = X[:, None, :] - X[None, :, :]
        dist_sq = np.sum(diff ** 2 * self.theta[None, None, :], axis=-1)
        R = np.exp(-dist_sq) + (self.nugget + 1e-8) * np.eye(n)

        jitter = 1e-8
        for _ in range(6):
            try:
                self.L = np.linalg.cholesky(R)
                break
            except np.linalg.LinAlgError:
                R += jitter * np.eye(n)
                jitter *= 10
        else:
            self.L = np.linalg.cholesky(R + 1e-3 * np.eye(n))

        ones = np.ones(n, dtype=float)
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, ones))
        beta = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        self.denom = max(float(np.dot(ones, alpha)), 1e-12)
        self.mu = float(np.dot(ones, beta) / self.denom)

        res = y - self.mu
        self.w = np.linalg.solve(self.L.T, np.linalg.solve(self.L, res))
        self.sigma2 = max(float(np.dot(res, self.w) / n), 1e-10)

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_star = np.asarray(X_star, dtype=float)
        m, d = X_star.shape
        diff = self.X[:, None, :] - X_star[None, :, :]
        dist_sq = np.sum(diff ** 2 * self.theta[None, None, :], axis=-1)
        R_star = np.exp(-dist_sq)

        y_hat = self.mu + R_star.T @ self.w
        V = np.linalg.solve(self.L, R_star)
        quad = np.sum(V ** 2, axis=0)
        u = 1.0 - np.sum(V * np.linalg.solve(self.L, np.ones(len(self.X)))[:, None], axis=0)
        s2 = np.maximum(self.sigma2 * (1.0 - quad + (u ** 2) / self.denom), 0.0)
        return y_hat, s2


class NSGAIIIEHVI(Algorithm):
    """NSGA-III with Expected Hypervolume Improvement.

    Combines Kriging surrogates, uncertainty-aware selection (SelectionMSE),
    importance-sampling expected hypervolume improvement (CalEHVI), and
    reference-direction niching.
    """

    ALGORITHM_FLAGS = {"NSGAIIIEHVI": {"expensive", "many", "multi", "real"}}

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Optional[np.ndarray] = None,
        sampling=None,
        n_samples: int = 1000,
        randp: float = 0.3,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.pop_size = int(max(4, pop_size))
        self.ref_dirs = ref_dirs
        self.sampling = sampling
        self.n_samples = int(n_samples)
        self.randp = float(randp)
        self.W: Optional[np.ndarray] = None
        self.archive: Optional[Population] = None
        self.models: List[_KrigingModel] = []

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        M = int(problem.n_obj)
        if self.ref_dirs is not None:
            self.W = np.asarray(to_numpy(self.ref_dirs), dtype=float)
            self.pop_size = len(self.W)
        else:
            try:
                self.W = get_reference_directions("energy", n_obj=M, n_points=self.pop_size)
            except Exception:
                w_init, n_eff = UniformPoint(self.pop_size, M)
                self.W = np.asarray(w_init, dtype=float)
                self.pop_size = len(self.W)

        self.archive = Population.empty()
        self.models = [_KrigingModel() for _ in range(M)]

    def _subsample_archive(
        self, X: np.ndarray, F: np.ndarray, max_points: int = 250
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        if n <= max_points:
            return X, F

        nds = NonDominatedSorting()
        fronts = nds.do(F)
        selected = list(fronts[0]) if len(fronts) > 0 else []

        # Extreme boundary points per objective
        M = F.shape[1]
        for m in range(M):
            selected.append(int(np.argmin(F[:, m])))
            selected.append(int(np.argmax(F[:, m])))
        selected = list(set(selected))

        if len(selected) < max_points:
            remaining = [i for i in range(n) if i not in selected]
            needed = max_points - len(selected)
            rng = self.random_state if hasattr(self, "random_state") and self.random_state is not None else np.random.default_rng(42)
            if hasattr(rng, "choice"):
                picks = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
            else:
                picks = np.random.choice(remaining, size=min(needed, len(remaining)), replace=False)
            selected.extend(picks)

        selected = np.array(selected[:max_points], dtype=int)
        return X[selected], F[selected]

    def _infill(self) -> Optional[Population]:
        if self.pop is None or len(self.pop) == 0:
            init_pop = sample_initial(self.problem, self.pop_size, self.sampling, self.random_state)
            return init_pop

        X_all = np.asarray(self.archive.get("X"), dtype=float)
        F_all = np.asarray(self.archive.get("F"), dtype=float)
        M = int(self.problem.n_obj)
        D = int(X_all.shape[1])
        xl = np.asarray(to_numpy(self.problem.xl), dtype=float)
        xu = np.asarray(to_numpy(self.problem.xu), dtype=float)

        # 1. Train Kriging models on subsampled archive
        X_train, F_train = self._subsample_archive(X_all, F_all, max_points=200)
        for m in range(M):
            self.models[m].fit(X_train, F_train[:, m])

        # 2. Generate candidate pool via differential mutation + polynomial mutation
        N_cand = self.pop_size
        X_curr = np.asarray(self.pop.get("X"), dtype=float)
        if len(X_curr) < N_cand:
            X_curr = X_all[-N_cand:]

        rng = self.random_state if hasattr(self, "random_state") and self.random_state is not None else np.random.default_rng(42)
        if hasattr(rng, "integers"):
            r1 = rng.integers(0, len(X_curr) - 1, size=len(X_curr))
            r2 = rng.integers(0, len(X_curr) - 2, size=len(X_curr))
        else:
            r1 = np.random.randint(0, len(X_curr) - 1, size=len(X_curr))
            r2 = np.random.randint(0, len(X_curr) - 2, size=len(X_curr))

        arange_L = np.arange(len(X_curr))
        r1[r1 >= arange_L] += 1
        min_r = np.minimum(arange_L, r1)
        max_r = np.maximum(arange_L, r1)
        r2[r2 >= min_r] += 1
        r2[r2 >= max_r] += 1

        delta = 0.5 * (X_curr[r1] - X_curr[r2])
        offspring_X = np.clip(X_curr + delta, xl, xu)

        # Polynomial mutation
        prob_m = 1.0 / max(D, 1)
        if hasattr(rng, "random"):
            mutate_mask = rng.random(offspring_X.shape) < prob_m
            u = rng.random(offspring_X.shape)
        else:
            mutate_mask = np.random.random(offspring_X.shape) < prob_m
            u = np.random.random(offspring_X.shape)

        diff = xu - xl
        delta_q = np.where(u <= 0.5, (2.0 * u) ** (1.0 / 21.0) - 1.0, 1.0 - (2.0 * (1.0 - u)) ** (1.0 / 21.0))
        mutated = np.clip(offspring_X + delta_q * diff, xl, xu)
        offspring_X[mutate_mask] = mutated[mutate_mask]

        candidate_X = np.vstack([X_curr, offspring_X])

        # 3. Surrogate predictions
        F_hat = np.zeros((len(candidate_X), M), dtype=float)
        MSE = np.zeros((len(candidate_X), M), dtype=float)
        for m in range(M):
            f_m, s2_m = self.models[m].predict(candidate_X)
            F_hat[:, m] = f_m
            MSE[:, m] = s2_m

        # 4. Uncertainty-aware Selection (SelectionMSE)
        nds = NonDominatedSorting()
        real_fronts = nds.do(F_all)
        F_real_nd = F_all[real_fronts[0]] if len(real_fronts) > 0 else F_all

        mse_scalar = np.prod(np.maximum(MSE, 1e-14), axis=1) ** (1.0 / M)
        diff_r = F_real_nd[:, None, :] - F_hat[None, :, :]
        is_dom = np.any(np.all(diff_r <= 0, axis=-1), axis=0)

        obj_eval = np.zeros((len(candidate_X), M + 1))
        obj_eval[:, :M] = F_hat
        obj_eval[~is_dom, M] = mse_scalar[~is_dom]
        obj_eval[is_dom, M] = -mse_scalar[is_dom]

        surrogate_fronts = nds.do(obj_eval)
        selected_cand = []
        for fr in surrogate_fronts:
            if len(selected_cand) + len(fr) <= self.pop_size:
                selected_cand.extend(fr)
            else:
                needed = self.pop_size - len(selected_cand)
                selected_cand.extend(fr[:needed])
                break

        selected_cand = np.array(selected_cand, dtype=int)
        X_survivors = candidate_X[selected_cand]
        F_survivors = F_hat[selected_cand]
        MSE_survivors = MSE[selected_cand]

        # 5. Expected Hypervolume Improvement (CalEHVI)
        ehvi_scores = self._cal_ehvi(F_survivors, MSE_survivors, F_real_nd)

        # 6. Infill Batch Selection (Diversity + EHVI)
        W_norm = self.W / np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-12)
        span = np.maximum(np.max(F_survivors, axis=0) - np.min(F_survivors, axis=0), 1e-12)
        F_norm = (F_survivors - np.min(F_survivors, axis=0)) / span
        norm_f = np.maximum(np.linalg.norm(F_norm, axis=1, keepdims=True), 1e-12)
        cos_sim = (F_norm / norm_f) @ W_norm.T
        assoc = np.argmax(cos_sim, axis=1)

        infill_idx = []
        for w_i in range(len(self.W)):
            members = np.where(assoc == w_i)[0]
            if len(members) > 0:
                best_m = members[np.argmax(ehvi_scores[members])]
                infill_idx.append(best_m)

        if len(infill_idx) < self.pop_size:
            remaining = [i for i in range(len(X_survivors)) if i not in infill_idx]
            rem_sorted = sorted(remaining, key=lambda i: ehvi_scores[i], reverse=True)
            needed = self.pop_size - len(infill_idx)
            infill_idx.extend(rem_sorted[:needed])

        infill_X = X_survivors[np.array(infill_idx[:self.pop_size], dtype=int)]
        return Population.new("X", infill_X)

    def _cal_ehvi(self, F_hat: np.ndarray, MSE: np.ndarray, F_real_nd: np.ndarray) -> np.ndarray:
        N, M = F_hat.shape
        combined = np.vstack([F_real_nd, F_hat])
        z_min = np.min(combined, axis=0)
        z_max = np.max(combined, axis=0)
        span = np.maximum(z_max - z_min, 1e-6)

        F_hat_norm = (F_hat - z_min) / span
        MSE_norm = MSE / (span ** 2)
        sigma = np.sqrt(np.maximum(MSE_norm, 1e-12))
        F_real_norm = (F_real_nd - z_min) / span

        rng = self.random_state if hasattr(self, "random_state") and self.random_state is not None else np.random.default_rng(42)
        if hasattr(rng, "uniform"):
            S = rng.uniform(-0.2, 1.2, size=(self.n_samples, M))
        else:
            S = np.random.uniform(-0.2, 1.2, size=(self.n_samples, M))

        diff = F_real_norm[:, None, :] - S[None, :, :]
        dom_by_any = np.any(np.all(diff <= 0, axis=-1), axis=0)
        S_nd = S[~dom_by_any]
        if len(S_nd) == 0:
            return np.zeros(N)

        diff_s = S_nd[None, :, :] - F_hat_norm[:, None, :]
        log_p = -0.5 * np.sum(np.log(2 * np.pi) + 2 * np.log(sigma[:, None, :]) + (diff_s ** 2) / (sigma[:, None, :] ** 2), axis=-1)
        max_log = np.max(log_p, axis=0, keepdims=True)
        P = np.exp(np.clip(log_p - max_log, -50, 0))
        return np.sum(P, axis=1)

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> Any:
        if infills is None or len(infills) == 0:
            return

        merged = Population.merge(self.pop, infills) if (self.pop is not None and len(self.pop) > 0) else infills
        F_merged = np.asarray(merged.get("F"), dtype=float)
        nds = NonDominatedSorting()
        fronts = nds.do(F_merged)

        selected = []
        for fr in fronts:
            if len(selected) + len(fr) <= self.pop_size:
                selected.extend(fr)
            else:
                needed = self.pop_size - len(selected)
                selected.extend(fr[:needed])
                break

        self.pop = merged[np.array(selected[:self.pop_size], dtype=int)]

        # Maintain bounded training archive (non-dominated + random reservoir <= 500)
        self.archive = Population.merge(self.archive, infills) if (self.archive is not None and len(self.archive) > 0) else infills
        if len(self.archive) > 500:
            F_arch = np.asarray(self.archive.get("F"), dtype=float)
            arch_fronts = nds.do(F_arch)
            nd_arch = np.asarray(arch_fronts[0], dtype=int) if len(arch_fronts) > 0 else np.arange(len(F_arch))
            rng = self.random_state if hasattr(self, "random_state") and self.random_state is not None else np.random.default_rng(42)
            if len(nd_arch) > 400:
                keep_idx = rng.choice(nd_arch, size=400, replace=False)
            else:
                rem = [i for i in range(len(self.archive)) if i not in nd_arch]
                needed = 500 - len(nd_arch)
                picks = rng.choice(rem, size=min(needed, len(rem)), replace=False) if len(rem) > 0 else np.empty(0, dtype=int)
                keep_idx = np.concatenate([nd_arch, picks])
            self.archive = self.archive[np.asarray(keep_idx, dtype=int)]

        self._set_optimum()

    def _set_optimum(self) -> None:
        if self.pop is None or len(self.pop) == 0:
            return
        F = np.asarray(self.pop.get("F"), dtype=float)
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        if len(fronts) > 0 and len(fronts[0]) > 0:
            self.opt = self.pop[np.array(fronts[0], dtype=int)]
        else:
            self.opt = self.pop


NSGA3_EHVI = NSGAIIIEHVI

__all__ = ["NSGAIIIEHVI", "NSGA3_EHVI"]
