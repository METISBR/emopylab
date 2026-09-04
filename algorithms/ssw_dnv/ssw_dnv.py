# -*- coding: utf-8 -*-
# Author: Prof. Thiago Santos, 2026
"""
SSW-DNV — Schaeffler-SDE-Weighted · Reference Directions · PBI · Adaptive, 2026
=====================================================================

Many-objective evolutionary algorithm (m > 3) combining:
  1. Dynamic epsilon adaptation via CSA path (coverage/success/stagnation signals)
  2. Phase-aware SDE/SBX/RVX allocation driven by search-mode state-switch
  3. Diversity-sensitive parent bias (rarity + sparse-niche boost)
  4. PBI-score niching with adaptive θ (θ-dominance, Yuan et al. 2015)
  5. SDR-based front ranking for higher selection pressure at m ≥ 4
  6. Two-archive: nd_archive (diversity) + conv_archive (Tchebycheff per niche)
  7. Tchebycheff blending into SDE drift direction for m > 4
  8. Latin Hypercube Sampling initialisation
  9. Q-learning for SBX η selection
 10. CMA-style auto-calibration of path/covariance hyperparameters
 11. EEJ (Evolutionary Ensemble Jacobian): Jacobiano multi-objetivo global
     estimado sobre a fronteira não-dominada via mínimos quadrados
     regularizados (Tikhonov), sem custo adicional de avaliação.
     Ref: Nesterov & Spokoiny, Found. Comput. Math. 2017
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np
from util.array_backend import (
    CUPY_AVAILABLE,
    cp,
    get_array_module as _backend_get_array_module,
    to_device as _backend_to_device,
    to_numpy as _backend_to_numpy,
    resolve_backend_config,
    get_cupy_device_name,
)


_CUPY_RUNTIME_READY: Optional[bool] = None


def _cupy_runtime_ready() -> bool:
    """
    Return True only when CuPy can execute basic kernels on the current host.

    `CUPY_AVAILABLE` may be true even if runtime JIT deps (e.g. nvrtc) are
    missing. This probe avoids hard failures and enables clean CPU fallback.
    """
    global _CUPY_RUNTIME_READY
    if _CUPY_RUNTIME_READY is not None:
        return _CUPY_RUNTIME_READY

    if not CUPY_AVAILABLE or cp is None:
        _CUPY_RUNTIME_READY = False
        return False

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            _CUPY_RUNTIME_READY = False
            return False
        probe = cp.asarray([0.0], dtype=cp.float64)
        _ = probe * 1.0 + 1.0
        cp.cuda.Stream.null.synchronize()
        _CUPY_RUNTIME_READY = True
        return True
    except Exception:  # noqa: BLE001
        _CUPY_RUNTIME_READY = False
        return False


def _get_xp(x: Any):
    """Return the array module (numpy or cupy) for the given input."""
    return _backend_get_array_module(x)

def _to_numpy(x: Any) -> np.ndarray:
    """Safely convert cupy or numpy array to numpy."""
    return _backend_to_numpy(x)

def _to_device(x: Any, use_gpu: bool = False) -> Any:
    """Move array to GPU if requested and available, otherwise return numpy."""
    return _backend_to_device(x, use_gpu=use_gpu)

from core.algorithm import Algorithm
from core.population import Population
from core.duplicate import DefaultDuplicateElimination
from algorithms.moo.sms import cv_and_dom_tournament
from core.mating import Mating
from operators.selection.tournament import TournamentSelection
from operators.crossover.sbx import SBX
from operators.mutation.pm import PolynomialMutation
from util.nds.non_dominated_sorting import NonDominatedSorting
from util.dominator import Dominator

ALGORITHM_FLAGS = {
    "SSW_DNV": {"multi", "many"},
}


# ---------------------------------------------------------------------------
# [NOVO] Latin Hypercube Sampling para inicialização (melhora cobertura
# inicial do espaço de decisão vs. amostragem uniforme aleatória pura).
# Referência: McKay et al. 1979; amplamente adotado em MaOEA recentes.
# ---------------------------------------------------------------------------
def _latin_hypercube_sample(rng: Any, n: int, d: int, xp: Any = np) -> np.ndarray:
    """
    Retorna matriz (n x d) com valores em [0, 1] via LHS.
    """
    cut = xp.linspace(0.0, 1.0, n + 1)
    u = xp.asarray(rng.random((n, d)))
    points = xp.empty((n, d), dtype=float)
    for j in range(d):
        perm = xp.asarray(rng.permutation(n))
        points[:, j] = cut[perm] + u[:, j] * (cut[1] - cut[0])
    return xp.clip(points, 0.0, 1.0)


# ---------------------------------------------------------------------------
# [NOVO] θ-dominância escalar (Yuan et al. 2015; θ-NSGA-III / θ-DEA).
# Calcula o score PBI(x, w) = d1 + θ * d2 para um vetor x normalizado
# e direção de referência w normalizada.
#   d1 = componente paralela (convergência)
#   d2 = componente perpendicular (diversidade)
# Scores menores indicam soluções de maior qualidade para o subproblema w.
# ---------------------------------------------------------------------------
def _pbi_score(
    f_norm: np.ndarray,   # shape (n_obj,) normalizado [0,1]^m
    ref_dir: np.ndarray,  # shape (n_obj,) vetor de referência unitário
    theta: float,
) -> float:
    """PBI escalar: d1 + theta * d2. Supports GPU."""
    xp = _get_xp(f_norm)
    w = ref_dir / xp.maximum(float(xp.linalg.norm(ref_dir)), 1e-12)
    d1 = float(xp.dot(f_norm, w))
    diff = f_norm - d1 * w
    d2 = float(xp.linalg.norm(diff))
    return d1 + theta * d2


# ---------------------------------------------------------------------------
# Normalization and niche-association — native implementations replacing
# all core NSGA-III dependencies.
# ---------------------------------------------------------------------------

class _IdealNadirTracker:
    """
    Adaptive ideal/nadir tracker with exponential moving average (EMA).

    The NSGA-III HyperplaneNormalization fits an intercept hyperplane through
    the extreme points of the current non-dominated front, which is unstable
    in early generations and for m > 5.  This replacement:

    • Maintains a running ideal via element-wise minimum over all seen F
      (monotonically non-increasing — never forgets a good value).
    • Maintains a running nadir via EMA of the per-generation max on the
      non-dominated front, smoothed with decay `nadir_ema` (default 0.2).
      This avoids the hyperplane fit while tracking the true nadir reliably.
    • Falls back to population min/max when the front is degenerate.

    Benefit for diversity: the EMA nadir is less susceptible to outlier
    extreme points that NSGA-III's hyperplane intercept amplifies in high-m
    spaces, giving more uniform normalisation across all reference vectors.
    """

    def __init__(self, n_obj: int, nadir_ema: float = 0.20):
        self.n_obj = int(n_obj)
        self.nadir_ema = float(nadir_ema)
        self.ideal_point: np.ndarray = np.full(n_obj, np.inf, dtype=float)
        self.nadir_point: np.ndarray = np.full(n_obj, -np.inf, dtype=float)
        self._nadir_ema: Optional[np.ndarray] = None

    def update(self, F: np.ndarray, nds: Optional[np.ndarray] = None, progress: float = 0.0) -> None:
        """Update ideal and nadir from population F (shape n × m)."""
        xp = _get_xp(F)
        F = xp.asarray(F, dtype=float)
        # Ideal: running minimum over ALL evaluated solutions
        self.ideal_point = _to_device(self.ideal_point, xp == cp)
        self.ideal_point = xp.minimum(self.ideal_point, xp.min(F, axis=0))

        # Nadir: EMA of per-generation max on non-dominated front (or full pop)
        if nds is not None and len(nds) > 0:
            nds_arr = xp.asarray(nds, dtype=int)
            nd_F = F[nds_arr]
        else:
            nd_F = F
        gen_nadir = xp.max(nd_F, axis=0)

        # [P6b] Adaptive EMA: alpha_max*(1-t) + alpha_min*t
        alpha_max = 0.50
        alpha_min = 0.05
        ema_rate  = (alpha_max * (1.0 - float(progress))
                     + alpha_min * float(progress))
        
        if self._nadir_ema is None:
            self._nadir_ema = gen_nadir.copy()
        else:
            self._nadir_ema = _to_device(self._nadir_ema, xp == cp)
            self._nadir_ema = ((1.0 - ema_rate) * self._nadir_ema
                               + ema_rate * gen_nadir)
        self.nadir_point = self._nadir_ema.copy()

        # Safety: ensure nadir > ideal
        bad = self.nadir_point <= self.ideal_point + 1e-12
        if xp.any(bad):
            self.nadir_point[bad] = self.ideal_point[bad] + 1.0


def _associate_to_niches_angular(
    F: np.ndarray,
    ref_dirs: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate each solution in F to the closest reference direction using
    **acute-angle distance**. Supports GPU (CuPy) or CPU (NumPy).
    """
    xp = _get_xp(F)
    F = xp.asarray(F, dtype=float)
    ideal = xp.asarray(ideal, dtype=float)
    nadir = xp.asarray(nadir, dtype=float)
    ref_dirs = xp.asarray(ref_dirs, dtype=float)

    span = xp.maximum(nadir - ideal, 1e-12)
    F_norm = (F - ideal) / span                   # translate + scale to [0,1]^m

    # Add a small floor so zero vectors don't cause division by zero
    F_norm = xp.maximum(F_norm, 1e-12)
    F_mag = xp.linalg.norm(F_norm, axis=1, keepdims=True)   # (n, 1)
    F_unit = F_norm / xp.maximum(F_mag, 1e-12)              # (n, m) unit vectors

    R = ref_dirs
    R_mag = xp.linalg.norm(R, axis=1, keepdims=True)
    R_unit = R / xp.maximum(R_mag, 1e-12)                   # (n_ref, m)

    # Cosine similarity matrix (n × n_ref)
    cos_mat = F_unit @ R_unit.T
    cos_mat = xp.clip(cos_mat, -1.0, 1.0)

    # Best niche = reference direction with maximum cosine (minimum angle)
    niche = xp.argmax(cos_mat, axis=1).astype(int)

    # Perpendicular distance = ||f_norm|| * sin(angle) = ||f_norm|| * sqrt(1 - cos²)
    best_cos = cos_mat[xp.arange(len(F)), niche]
    sin2 = xp.maximum(1.0 - best_cos ** 2, 0.0)
    d_perp = F_mag.ravel() * xp.sqrt(sin2)

    return niche, d_perp


def _calc_niche_count(n_niches: int, niche_of: np.ndarray) -> np.ndarray:
    """Simple bincount wrapper — replaces calc_niche_count from nsga3. Supports GPU."""
    xp = _get_xp(niche_of)
    return xp.bincount(niche_of, minlength=n_niches).astype(int)


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex. Supports GPU."""
    xp = _get_xp(v)
    x = xp.asarray(v, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
    n = len(x)
    if n == 1:
        return xp.array([1.0], dtype=float)

    u = xp.sort(x)[::-1]
    cssv = xp.cumsum(u) - 1.0
    idx = xp.arange(1, n + 1, dtype=float)
    cond = u - cssv / idx > 0.0
    if not xp.any(cond):
        return xp.full(n, 1.0 / n, dtype=float)
    rho = int(xp.where(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    w = xp.maximum(x - theta, 0.0)
    s = float(xp.sum(w))
    if s <= 0.0:
        return xp.full(n, 1.0 / n, dtype=float)
    return w / s


def solve_schaeffler_qp(
    G: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10,
) -> np.ndarray:
    """
    Solve Schaeffler simplex QP:
        min_{lambda in simplex} || sum_i lambda_i * grad(f_i) ||^2

    Supports GPU (CuPy) or CPU (NumPy).
    """
    xp = _get_xp(G)
    G = xp.asarray(G, dtype=float)
    m = int(G.shape[0])

    if m == 1:
        return xp.array([1.0], dtype=float)

    if m == 2:
        u, v = G[0], G[1]
        w = u - v
        ww = float(xp.dot(w, w))
        if ww < 1e-15:
            return xp.array([0.5, 0.5], dtype=float)
        lam = float(xp.dot(v, v - u) / ww)
        lam = float(xp.clip(lam, 0.0, 1.0))
        return xp.array([lam, 1.0 - lam], dtype=float)

    H = G @ G.T
    x = xp.full(m, 1.0 / m, dtype=float)

    # Safe Lipschitz upper bound for quadratic gradient step.
    L = float(xp.linalg.norm(H, ord=xp.inf))
    step = 1.0 / max(L, 1e-12)

    prev = x
    for _ in range(max(max_iter, 1)):
        grad = H @ x
        x = _project_to_simplex(x - step * grad)
        if float(xp.linalg.norm(x - prev)) <= max(tol, 1e-9):
            break
        prev = x

    return x


# =========================================================================
# EEJ — EVOLUTIONARY ENSEMBLE JACOBIAN (substitui Broyden individual)
# =========================================================================

def _compute_eej(
    X_archive: np.ndarray,
    F_archive: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """
    Estima o Jacobiano multi-objetivo coletivamente sobre um arquivo de
    soluções não-dominadas.

    O EEJ (Evolutionary Ensemble Jacobian) usa TODOS os indivíduos do
    arquivo como amostras coletivas para estimar o Jacobiano local da
    fronteira Pareto estimada, via mínimos quadrados regularizados:

        dX = X - mean(X)          (n_pop, n_var)
        dF = F - mean(F)          (n_pop, n_obj)
        G  = dX^T @ dX + reg·I   (n_var, n_var)
        J_hat = (dF^T @ dX) @ G^{-1}   (n_obj, n_var)

    Propriedade: E[J_hat] → J_f(μ) quando |archive| → ∞
    Ref: Nesterov & Spokoiny, Found. Comput. Math. 2017
         Evensen, 1994 (Ensemble Kalman Filter)

    Custo: O(n_var² · |archive|) operações — zero avaliações extras.

    Args:
        X_archive: matriz de decisão do arquivo (n_pop, n_var)
        F_archive: matriz de objetivos (n_pop, n_obj)
        reg:       regularização Tikhonov

    Returns:
        J_hat: Jacobiano ensemble, forma (n_obj, n_var)
    """
    n_pop, n_var = X_archive.shape
    n_obj = F_archive.shape[1]

    if n_pop < 2:
        return np.zeros((n_obj, n_var))

    dX = X_archive - X_archive.mean(axis=0)
    dF = F_archive - F_archive.mean(axis=0)

    G = dX.T @ dX + reg * np.eye(n_var)

    try:
        Jt, _, _, _ = np.linalg.lstsq(G, dX.T @ dF, rcond=None)
        J_hat = Jt.T
    except np.linalg.LinAlgError:
        J_hat = np.zeros((n_obj, n_var))

    return J_hat


def _compute_eej_descent(J: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    """
    Calcula a direção de descida comum via QP no simplex.

    Resolve:  min_{α ∈ Δ}  α^T (J J^T + reg·I) α
    onde      Δ = {α ≥ 0 : Σα_i = 1}

    A direção resultante é:  q = J^T α* / ||J^T α*||

    Args:
        J:   Jacobiano EEJ, forma (n_obj, n_var)
        reg: Regularização

    Returns:
        Vetor normalizado q de forma (n_var,)
    """
    m, n = J.shape

    if m == 1:
        q = J[0]
        norm = float(np.linalg.norm(q))
        return q / norm if norm > 1e-12 else np.zeros(n)

    H = J @ J.T + reg * np.eye(m)
    L = float(np.linalg.norm(H, ord=np.inf))
    step = 1.0 / (L + 1e-12)

    alpha = np.full(m, 1.0 / m)
    for _ in range(300):
        g = H @ alpha
        alpha_new = alpha - step * g
        alpha_new = np.maximum(alpha_new, 0.0)
        s = float(np.sum(alpha_new))
        if s < 1e-12:
            alpha_new = np.full(m, 1.0 / m)
        else:
            alpha_new /= s
        if float(np.linalg.norm(alpha_new - alpha)) < 1e-10:
            alpha = alpha_new
            break
        alpha = alpha_new

    q = J.T @ alpha
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else np.zeros(n)


def _spherical_kmeans(X, k, max_iter=20, xp=np):
    n = len(X)
    k = min(k, n)
    indices = xp.random.choice(n, k, replace=False) if xp != np else np.random.choice(n, k, replace=False)
    centroids = X[indices]
    for _ in range(max_iter):
        sim = X @ centroids.T
        labels = xp.argmax(sim, axis=1)
        new_c = xp.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if xp.any(mask):
                c = xp.sum(X[mask], axis=0)
                nrm = xp.linalg.norm(c)
                if nrm > 1e-12:
                    new_c[i] = c / nrm
                else:
                    new_c[i] = centroids[i]
            else:
                new_c[i] = centroids[i]
        if xp.allclose(centroids, new_c):
            break
        centroids = new_c
    return centroids


class SSW_DNV(Algorithm):
    """
    SSW-DNV: Schaeffler-SDE-Weighted · Reference Directions · PBI · Adaptive.

    Many-objective variant (m > 3) with:
    - Schaeffler QP + Tchebycheff blended SDE local search
    - PBI-score niching with adaptive θ (θ-dominance)
    - SDR-based non-dominated ranking (higher selection pressure for m ≥ 4)
    - Two-archive survival (diversity + convergence)
    - Latin Hypercube Sampling initialisation
    - Q-learning for SBX η, CMA-style path/covariance adaptation
    """
    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    @property
    def is_using_gpu(self) -> bool:
        """Returns True if the algorithm is currently using GPU acceleration."""
        return getattr(self, "use_gpu", False)

    @property
    def backend_code(self) -> str:
        return "gpu" if self.is_using_gpu else "cpu"

    @property
    def backend_info(self) -> str:
        if not self.is_using_gpu:
            return "CPU only"
        gpu_name = get_cupy_device_name(0) or "CUDA GPU"
        return f"GPU CUDA via CuPy ({gpu_name}, {self.gpu_dtype})"

    @property
    def xp(self):
        """
        Dynamic backend module accessor.

        Avoids storing module objects (numpy/cupy) in instance state, which
        breaks deepcopy/pickling used internally by EmoPyLab runner.
        """
        return cp if self.use_gpu else np

    def __init__(
        self,
        ref_dirs: np.ndarray,
        sigma: float = 0.1,
        epsilon: float = 0.10,
        epsilon_min_factor: float = 0.1,
        epsilon_max_factor: float = 3.5,
        epsilon_adapt_lr: float = 0.18,
        ratio_sde: float = 0.14,
        prob_crossover: float = 0.9,
        adaptive_ratio_lr: float = 0.16,
        ratio_bounds: Tuple[float, float] = (0.02, 0.40),
        phase_ratio_targets: Tuple[float, float, float] = (0.06, 0.12, 0.25),
        coverage_target: float = 0.60,
        diversity_boost: float = 0.75,
        c_path: float = 0.20,
        c_cov: float = 0.12,
        c_sigma: float = 0.14,
        target_success: float = 0.22,
        eej_reg: float = 1e-6,
        ucb_exploration: float = 0.08,
        drift_scale: float = 0.20,
        de_fallback_scale: float = 0.14,
        mirror_rate: float = 0.65,
        use_cma_auto: bool = True,
        epsilon_csa_gain: float = 0.28,
        sparse_quantile: float = 0.30,
        elite_keep_frac: float = 0.12,
        angle_adapt_gain: float = 0.30,
        qlearn_alpha: float = 0.22,
        qlearn_gamma: float = 0.90,
        qlearn_eps: float = 0.08,
        rvx_share_base: float = 0.40,
        rvx_mu_f: float = 0.55,
        rvx_mu_cr: float = 0.85,
        rvx_adapt_lr: float = 0.12,
        stagnation_tol: float = 2e-4,
        telemetry_limit: int = 600,
        # ---------------------------------------------------------------
        # [NOVO] Parâmetros adicionados na revisão MaOP 2025
        # ---------------------------------------------------------------
        # θ para PBI-score em niching (θ-dominância); range típico [2, 10].
        # Valores maiores reforçam convergência; menores reforçam diversidade.
        # Ref: Yuan et al. 2015 (θ-NSGA-III); caps-NSGA-III (Symmetry 2024).
        theta_pbi: float = 5.0,
        # Adaptar θ_pbi ao longo do progresso (True = diminui de θ_pbi p/ 1.5)
        theta_adapt: bool = True,
        # σ para SDR — fração do span do objetivo usada como margem de tolerância.
        # Ref: NSGA-II/SDR (Tian et al. 2018); 3DEA (Zhang et al. 2024).
        sdr_sigma: float = 0.02,
        # Ativar SDR no lugar de Pareto puro em _hybrid_survival (recomendado m>3)
        use_sdr: bool = True,
        # Inicialização por LHS em vez de uniforme aleatória.
        # Ref: McKay 1979; melhora cobertura e convergência inicial.
        use_lhs: bool = True,
        # Dois arquivos separados: convergência (conv_archive) + diversidade (pop).
        # A sobrevivência faz swap explícito quando arquivo de convergência evolui.
        # Ref: Wang et al. 2015 (Two-archive); MaOEA-MS (ScienceDirect 2022).
        use_two_archive: bool = True,
        conv_archive_frac: float = 0.20,
        prob_neighbor_mating: float = 0.70,
        archive_injection_rate: float = 0.12,
        seed: Optional[int] = None,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        use_gpu: bool = True,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        cfg = resolve_backend_config(use_gpu=use_gpu, array_backend=array_backend, gpu_dtype=gpu_dtype)
        self.array_backend_requested = str(cfg["requested_backend"])
        self.array_backend = str(cfg["effective_backend"])
        self.gpu_dtype = str(cfg["gpu_dtype"])
        self.use_gpu = bool(cfg["use_gpu"]) and _cupy_runtime_ready()
        if not self.use_gpu:
            self.array_backend = "numpy"
        elif self.array_backend != "cupy":
            self.array_backend = "cupy"

        # Keep CPU copy during construction: EmoPyLab runner deep-copies the
        # algorithm instance before _setup(), and numpy arrays are safe there.
        self.ref_dirs_np = np.asarray(_to_numpy(ref_dirs), dtype=float)
        self.ref_dirs = None
        self.pop_size = len(self.ref_dirs_np)

        self.sigma_base = float(sigma)
        self.epsilon_base = float(epsilon)
        self.epsilon_state = float(epsilon)
        self.epsilon_min = max(1e-14, float(epsilon) * float(epsilon_min_factor))
        self.epsilon_max = max(self.epsilon_min, float(epsilon) * float(epsilon_max_factor))
        self.epsilon_adapt_lr = float(np.clip(epsilon_adapt_lr, 0.01, 0.80))

        self.ratio_sde = float(np.clip(ratio_sde, ratio_bounds[0], ratio_bounds[1]))
        self.ratio_bounds = tuple(ratio_bounds)
        self.adaptive_ratio_lr = float(adaptive_ratio_lr)
        self.phase_ratio_targets = tuple(float(x) for x in phase_ratio_targets)
        if len(self.phase_ratio_targets) != 3:
            raise ValueError("phase_ratio_targets must contain exactly 3 values")
        self.coverage_target = float(np.clip(coverage_target, 0.10, 1.0))
        self.diversity_boost = float(np.clip(diversity_boost, 0.0, 2.0))

        self.c_path = float(c_path)
        self.c_cov = float(c_cov)
        self.c_sigma = float(c_sigma)
        self.target_success = float(target_success)
        self.eej_reg = float(max(eej_reg, 1e-12))
        self.ucb_exploration = float(np.clip(ucb_exploration, 0.0, 1.0))
        self.drift_scale = float(np.clip(drift_scale, 0.05, 0.6))
        self.de_fallback_scale = float(np.clip(de_fallback_scale, 0.0, 0.8))
        self.mirror_rate = float(np.clip(mirror_rate, 0.0, 1.0))
        self.use_cma_auto = bool(use_cma_auto)
        self.epsilon_csa_gain = float(np.clip(epsilon_csa_gain, 0.0, 1.0))
        self.sparse_quantile = float(np.clip(sparse_quantile, 0.05, 0.90))
        self.elite_keep_frac = float(np.clip(elite_keep_frac, 0.0, 0.5))
        self.angle_adapt_gain = float(np.clip(angle_adapt_gain, 0.0, 1.0))
        self.qlearn_alpha = float(np.clip(qlearn_alpha, 0.01, 1.0))
        self.qlearn_gamma = float(np.clip(qlearn_gamma, 0.0, 1.0))
        self.qlearn_eps = float(np.clip(qlearn_eps, 0.0, 0.5))
        self.rvx_share_base = float(np.clip(rvx_share_base, 0.0, 1.0))
        self.rvx_mu_f = float(np.clip(rvx_mu_f, 0.05, 1.0))
        self.rvx_mu_cr = float(np.clip(rvx_mu_cr, 0.05, 1.0))
        self.rvx_adapt_lr = float(np.clip(rvx_adapt_lr, 0.01, 0.5))
        self.stagnation_tol = float(max(stagnation_tol, 0.0))
        self.telemetry_limit = int(max(50, telemetry_limit))
        self.sigma_damp = 1.0
        self.entropy_diversity_threshold = 0.82
        self.niche_cv_diversity_threshold = 1.30
        # [NOVO] Parâmetros da revisão
        self.theta_pbi = float(np.clip(theta_pbi, 0.5, 20.0))
        self.theta_adapt = bool(theta_adapt)
        self.sdr_sigma = float(np.clip(sdr_sigma, 0.0, 0.15))
        self.use_sdr = bool(use_sdr)
        self.use_lhs = bool(use_lhs)
        self.use_two_archive = bool(use_two_archive)
        self.conv_archive_frac = float(np.clip(conv_archive_frac, 0.05, 0.40))
        self.prob_neighbor_mating = float(np.clip(prob_neighbor_mating, 0.0, 1.0))
        self.archive_injection_rate = float(np.clip(archive_injection_rate, 0.0, 0.50))
        self.low_obj_mode = False
        self.conv_archive: Optional[Population] = None  # arquivo de convergência
        self._theta_current = float(theta_pbi)  # valor corrente adaptativo
        self._ref_nbr: Optional[np.ndarray] = None   # [P2] grafo angular

        # Baselines used by runtime auto-calibration (per problem/instance).
        self._user_ratio_bounds = tuple(self.ratio_bounds)
        self._user_ratio_sde = float(self.ratio_sde)
        self._user_phase_ratio_targets = tuple(self.phase_ratio_targets)
        self._user_coverage_target = float(self.coverage_target)
        self._user_sparse_quantile = float(self.sparse_quantile)
        self._user_target_success = float(self.target_success)
        self._user_ucb_exploration = float(self.ucb_exploration)
        self._user_rvx_share_base = float(self.rvx_share_base)
        self._user_drift_scale = float(self.drift_scale)
        self._user_de_fallback_scale = float(self.de_fallback_scale)
        self._user_qlearn_eps = float(self.qlearn_eps)
        self._user_mirror_rate = float(self.mirror_rate)
        self._user_epsilon_adapt_lr = float(self.epsilon_adapt_lr)
        self._user_epsilon_csa_gain = float(self.epsilon_csa_gain)
        self._user_use_sdr = bool(self.use_sdr)
        self._user_use_two_archive = bool(self.use_two_archive)
        self._user_prob_neighbor_mating = float(self.prob_neighbor_mating)
        self._user_archive_injection_rate = float(self.archive_injection_rate)
        self._user_theta_pbi = float(self.theta_pbi)

        self.dup_elim = DefaultDuplicateElimination()
        self.sbx = SBX(prob=prob_crossover, eta=20, vtype=float)
        self.pm = PolynomialMutation(prob=0.1, eta=20)
        self.selection = TournamentSelection(func_comp=cv_and_dom_tournament)
        self.mating = Mating(
            self.selection,
            self.sbx,
            self.pm,
            eliminate_duplicates=self.dup_elim,
            n_max_iterations=100,
        )

        self.nds = NonDominatedSorting()
        self.dominator = Dominator()

        self.hyp_norm = None

        # EEJ — Jacobiano Ensemble Evolutivo global (atualizado 1x/geração)
        self._eej: Optional[np.ndarray] = None
        self._eej_q: Optional[np.ndarray] = None  # direção de descida comum
        self.nd_archive = None

        self.sigma_min = None
        self.sigma_max = None
        self.domain_scale = None

        self.credit_sde = 0.25
        self.credit_sbx = 0.25
        self.operator_trials_sde = 1
        self.operator_trials_sbx = 1
        self.total_operator_events = 2
        self.last_rate_sde = 0.0
        self.last_rate_sbx = 0.0
        self.last_coverage = 1.0
        self.last_archive_improved = True
        self.prev_archive_ideal = None
        self.prev_archive_size = 0
        self.prev_archive_score = None
        self.eps_path = None
        self.eps_path_target = None
        self.last_sde_step_norm = 0.0
        self.last_total_sde = 0
        self.last_success_sde = 0
        self.last_sde_z_mean = None
        self.last_conv_metric = None
        self.last_div_metric = None
        self.last_entropy = 1.0
        self.last_niche_cv = 0.0
        self.stagnation_streak = 0
        self.search_mode = "balanced"
        self.last_q_state = 1
        self.q_table = self.xp.zeros((3, 3), dtype=float)
        self.sbx_eta_candidates = (15.0, 20.0, 28.0)
        self.sbx_mode = 1
        self.telemetry = []
        self._backend_cache: dict[tuple[int, str, str], Any] = {}

    def _sync_backend_arrays(self) -> None:
        """
        Prepare backend-specific cached arrays and enforce safe fallback.
        """
        if self.ref_dirs is not None:
            self.ref_dirs_np = np.asarray(_to_numpy(self.ref_dirs), dtype=float)
        elif self.ref_dirs_np is not None:
            self.ref_dirs = self.xp.asarray(self.ref_dirs_np, dtype=float)
        if self.use_gpu:
            if not _cupy_runtime_ready():
                self.use_gpu = False
                self.array_backend = "numpy"
                self.ref_dirs_xp = self.ref_dirs
                return
            try:
                if self.ref_dirs_np is not None:
                    self.ref_dirs_xp = _to_device(self.ref_dirs_np, use_gpu=True)
                self.array_backend = "cupy"
            except Exception:  # noqa: BLE001
                self.use_gpu = False
                self.array_backend = "numpy"
                self.ref_dirs_xp = self.ref_dirs_np
                return
        else:
            self.array_backend = "numpy"
            self.ref_dirs_xp = self.ref_dirs_np

    def _clear_backend_cache(self) -> None:
        self._backend_cache.clear()

    def _as_backend_array(
        self,
        value: Any,
        *,
        dtype: Any | None = None,
        prefer_gpu: Optional[bool] = None,
        cache_key: Optional[str] = None,
    ) -> Any:
        """
        Convert an arbitrary value to the active backend with lightweight caching.

        This avoids repeated NumPy<->CuPy conversions in the same generation.
        """
        use_gpu = self.use_gpu if prefer_gpu is None else bool(prefer_gpu)
        if value is None:
            return None

        if not use_gpu:
            arr_np = np.asarray(_to_numpy(value))
            if dtype is not None:
                arr_np = arr_np.astype(dtype, copy=False)
            return arr_np

        if cache_key is not None:
            key = (id(value), str(cache_key), str(dtype))
            cached = self._backend_cache.get(key)
            if cached is not None:
                return cached
        else:
            key = None

        arr_dev = _to_device(value, use_gpu=True)
        if dtype is not None:
            arr_dev = self.xp.asarray(arr_dev, dtype=dtype)

        if key is not None:
            self._backend_cache[key] = arr_dev
        return arr_dev

    def _nds_do(self, F: Any, **kwargs):
        """
        Run non-dominated sorting on CPU (NumPy) to avoid CuPy/Python-loop
        slowdown and keep deterministic behavior across backends.
        """
        F_np = np.asarray(_to_numpy(F), dtype=float)
        return self.nds.do(F_np, **kwargs)

    def _setup(self, problem, **kwargs):
        self._sync_backend_arrays()
        self._clear_backend_cache()

        self.ref_dirs = self.xp.asarray(self.ref_dirs_np, dtype=float)
        self.ref_dirs_xp = self.ref_dirs
        self.ref_dirs_base = self.ref_dirs.copy()

        if self.ref_dirs.shape[1] != problem.n_obj:
            raise ValueError(
                f"ref_dirs shape mismatch: expected n_obj={problem.n_obj}, got {self.ref_dirs.shape[1]}"
            )
        self._auto_calibrate_runtime_controls(problem.n_var, problem.n_obj)
        if self.use_cma_auto:
            self._configure_cma_hyperparams(problem.n_var, self.pop_size)

        n = float(max(problem.n_var, 1))
        self.eps_path = self.xp.zeros(problem.n_var, dtype=float)
        self.eps_path_target = float(np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)))
        self.hyp_norm = _IdealNadirTracker(problem.n_obj)
        self.pm = PolynomialMutation(prob=min(1.0, 1.0 / max(problem.n_var, 1)), eta=20)
        self.mating = Mating(
            self.selection,
            self.sbx,
            self.pm,
            eliminate_duplicates=self.dup_elim,
            n_max_iterations=100,
        )
        self._apply_policy_action(self.sbx_mode)

        # [P2] Grafo de vizinhos angulares entre direcoes de referencia.
        # self._ref_nbr[i] = top-K vizinhos angulares da direcao i.
        _k  = max(4, min(12, problem.n_obj + 2))
        _R  = self.ref_dirs_xp
        _Ru = _R / self.xp.maximum(self.xp.linalg.norm(_R, axis=1, keepdims=True), 1e-12)
        _C  = _Ru @ _Ru.T
        self.xp.fill_diagonal(_C, -self.xp.inf)
        _ke = min(_k, _C.shape[1] - 1)
        # Keep neighborhood graph on CPU for cheap Python-side set operations.
        self._ref_nbr = np.asarray(_to_numpy(self.xp.argsort(_C, axis=1)[:, -_ke:]), dtype=int)  # (n_ref, k)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_infill(self):
        xl, xu = self.problem.xl, self.problem.xu
        n_var = self.problem.n_var

        self.domain_scale = float(np.mean(xu - xl))
        self.sigma_min = 5e-5 * self.domain_scale
        self.sigma_max = 0.35 * self.domain_scale

        if self.use_lhs:
            unit_samples = _latin_hypercube_sample(self.random_state, self.pop_size, n_var, xp=self.xp)
            X = xl + _to_numpy(unit_samples) * (xu - xl)
        else:
            X = xl + self.random_state.random((self.pop_size, n_var)) * (xu - xl)
        pop = Population.new("X", X)
        self._attach_initial_state(pop)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self._hybrid_survival(infills, n_survive=self.pop_size)
        self.nd_archive = self._update_archive(self.pop)
        # [NOVO] Inicializar arquivo de convergência (two-archive)
        if self.use_two_archive:
            self.conv_archive = self._build_conv_archive(self.pop)
        # EEJ inicial sobre a população
        self._update_eej(self.pop)
        self.opt = self.nd_archive

    def _attach_initial_state(self, pop: Population):
        n = len(pop)
        n_var = self.problem.n_var
        n_obj = self.problem.n_obj

        # Keep per-individual state in NumPy. These tensors are updated mostly
        # in Python loops, where host memory is consistently faster and avoids
        # GPU ping-pong during every generation.
        C = np.repeat(np.eye(n_var, dtype=float)[None, :, :], n, axis=0)
        pc = np.zeros((n, n_var), dtype=float)
        sigma_i = np.full(n, self.sigma_base * self.domain_scale, dtype=float)

        X = np.asarray(pop.get("X"), dtype=float)
        parent_X = X.copy()
        parent_F = np.full((n, n_obj), np.nan, dtype=float)
        parent_idx = np.full(n, -1, dtype=int)
        origin = np.full(n, "init", dtype="<U4")
        anti = np.where(np.asarray(self.random_state.random(n)) < 0.5, -1.0, 1.0).astype(float)
        rvx_F = np.full(n, np.nan, dtype=float)
        rvx_CR = np.full(n, np.nan, dtype=float)

        pop.set(
            "C", C,
            "pc", pc,
            "sigma_i", sigma_i,
            "parent_X", parent_X,
            "parent_F", parent_F,
            "parent_idx", parent_idx,
            "origin", origin,
            "anti", anti,
            "rvx_F", rvx_F,
            "rvx_CR", rvx_CR,
        )

    # ------------------------------------------------------------------
    # EEJ — Atualização do Jacobiano Ensemble Evolutivo
    # ------------------------------------------------------------------
    def _update_eej(self, pop: Population) -> None:
        """
        Atualiza o EEJ global sobre a fronteira não-dominada da população.

        Calcula J_hat = lstsq(dX^T dX + reg·I, dX^T dF)^T sobre os
        indivíduos não-dominados. Chamado uma vez por geração em _advance().

        O EEJ e a direção q são armazenados em self._eej e self._eej_q,
        compartilhados por todos os SDE offspring na próxima geração.
        """
        if pop is None or len(pop) < 2:
            return

        X = np.asarray(pop.get("X", to_numpy=True), dtype=float)
        F = np.asarray(pop.get("F", to_numpy=True), dtype=float)

        # Usa fronteira não-dominada para estimar o Jacobiano
        try:
            nd_idx = self._nds_do(F, only_non_dominated_front=True)
            nd_idx = np.asarray(_to_numpy(nd_idx), dtype=int)
            if len(nd_idx) >= 2:
                X_nd = X[nd_idx]
                F_nd = F[nd_idx]
            else:
                X_nd = X
                F_nd = F
        except Exception:  # noqa: BLE001
            X_nd = X
            F_nd = F

        self._eej = _compute_eej(X_nd, F_nd, reg=self.eej_reg)
        self._eej_q = _compute_eej_descent(self._eej)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _infill(self):
        self._clear_backend_cache()
        progress = self._search_progress()
        F_pop = self._as_backend_array(self.pop.get("F"), dtype=float, cache_key="pop_F")
        dist_to_niche = None
        if self.pop.has("niche"):
            niche = self._as_backend_array(self.pop.get("niche"), dtype=int, cache_key="pop_niche")
            if self.pop.has("dist_to_niche"):
                dist_to_niche = self._as_backend_array(
                    self.pop.get("dist_to_niche"),
                    dtype=float,
                    cache_key="pop_dist_to_niche",
                )
        else:
            niche, dist_to_niche = self._associate_niches(F_pop)
        niche_count = self.xp.bincount(niche, minlength=len(self.ref_dirs))
        self.last_coverage = float(self.xp.count_nonzero(niche_count > 0) / max(len(self.ref_dirs), 1))
        entropy, niche_cv = self._niche_diversity_metrics(niche_count)
        self.last_entropy = float(entropy)
        self.last_niche_cv = float(niche_cv)

        scarcity = max(0.0, self.coverage_target - self.last_coverage)
        
        # [P7] Lambda sigmoid: substitui flags binarias diversity/convergence.
        # Elimina histerese; todos os parametros downstream sao interpolados
        # suavemente por lambda em vez de chavear abruptamente.
        _H_thr = float(self.entropy_diversity_threshold)
        _dH    = 0.15
        _beta  = 6.0
        _H_eff = (entropy
                  - 1.5 * scarcity
                  + 0.05 * min(self.stagnation_streak, 4))
        _z     = _beta * ((_H_eff - _H_thr) / _dH)
        lam    = float(1.0 / (1.0 + self.xp.exp(-_z)))
        lam    = float(np.clip(
            lam + 0.12 * max(0.0, progress - 0.60), 0.0, 1.0))

        # search_mode derivado continuamente de lambda
        if lam < 0.35:
            self.search_mode = "diversity"
        elif lam > 0.65:
            self.search_mode = "convergence"
        else:
            self.search_mode = "balanced"

        # [P7] phase_target interpolado suavemente via lambda
        t0, t1, t2   = self.phase_ratio_targets
        phase_target = ((t0 + 2.0*lam*(t1-t0))
                        if lam <= 0.5
                        else (t1 + 2.0*(lam-0.5)*(t2-t1)))
        
        self.ratio_sde = float(
            np.clip(
                0.9 * self.ratio_sde + 0.1 * phase_target,
                self.ratio_bounds[0],
                self.ratio_bounds[1],
            )
        )

        f_span = float(self.xp.mean(self.xp.max(F_pop, axis=0) - self.xp.min(F_pop, axis=0)))
        f_scale = float(
            self.xp.mean(self.xp.maximum(self.xp.abs(self.xp.mean(F_pop, axis=0)), 1.0))
        )
        if f_scale > 0 and (f_span / f_scale) > 5.0:
            self.ratio_sde = float(
                np.clip(0.70 * self.ratio_sde, self.ratio_bounds[0], self.ratio_bounds[1])
            )

        # Local model readiness gate: when EEJ is still weak (near-zero),
        # keep a larger share of global variation.
        if len(self.pop) > 0:
            eej_norm = float(np.linalg.norm(self._eej)) if self._eej is not None else 0.0
            model_ready = 1.0 if eej_norm > 1e-6 else 0.0
            if model_ready < 0.35:
                shrink = 0.55 + model_ready
                self.ratio_sde = float(
                    np.clip(
                        self.ratio_sde * shrink,
                        self.ratio_bounds[0],
                        self.ratio_bounds[1],
                    )
                )
        else:
            model_ready = 0.0

        # Reactive correction: if local branch underperforms while coverage is low,
        # reduce local-search pressure to avoid over-exploitation in hard MaOP fronts.
        if self.last_coverage < self.coverage_target and self.last_rate_sde < self.last_rate_sbx:
            gap = (self.last_rate_sbx - self.last_rate_sde) + (self.coverage_target - self.last_coverage)
            self.ratio_sde = float(
                np.clip(self.ratio_sde - 0.05 * gap, self.ratio_bounds[0], self.ratio_bounds[1])
            )

        # Additional damping when local branch repeatedly underperforms.
        if self.last_total_sde > 0 and self.last_rate_sde < max(0.04, 0.55 * self.last_rate_sbx):
            damp = 0.35 if self.low_obj_mode else 0.70
            self.ratio_sde = float(
                np.clip(self.ratio_sde * damp, self.ratio_bounds[0], self.ratio_bounds[1])
            )

        # Two-state control inspired by recent MaOP dynamic-selection works:
        # prioritize diversity recovery before intensification.
        # [P7] Damping suave proporcional a (1 - lambda)
        # Em lambda=0 (diversidade): fator = 0.45 (forte damping)
        # Em lambda=1 (convergencia): fator = 1.00 (sem damping)
        diversity_damp = (1.0 - lam) * 0.55 + 0.45   # [0.45, 1.00]
        self.ratio_sde = float(np.clip(
            self.ratio_sde * diversity_damp,
            self.ratio_bounds[0], self.ratio_bounds[1]))

        if lam > 0.65 and self.stagnation_streak >= 2 \
                and model_ready > 0.40:
            self.ratio_sde = float(np.clip(
                self.ratio_sde + 0.03 + 0.03 * (1.0 - entropy),
                self.ratio_bounds[0], self.ratio_bounds[1]))

        if self.low_obj_mode:
            sde_cap = 0.06 + 0.04 * max(0.0, progress - 0.75)
            sde_cap = float(np.clip(sde_cap, self.ratio_bounds[0], self.ratio_bounds[1]))
            self.ratio_sde = float(min(self.ratio_sde, sde_cap))

        # Activation gate for local branch: keep pure global evolution when
        # archive is improving and SBX is healthy.
        local_pressure = 0.0
        if not self.last_archive_improved:
            local_pressure += 0.6
        if self.last_rate_sbx < max(0.08, 0.60 * self.target_success):
            local_pressure += 0.4
        if self.last_coverage < 0.85 * self.coverage_target:
            local_pressure -= 0.2
        if self.stagnation_streak >= 2:
            local_pressure += 0.35
        # [P7] Penalidade continua pelo grau de diversidade necessario
        local_pressure -= (1.0 - lam) * 0.55

        if local_pressure <= 0.0 and progress < 0.65:
            n_sde = 0
        else:
            n_sde = int(round(self.pop_size * self.ratio_sde))
        # [P7] Cap proporcional a lambda em vez de flag binario
        if lam < 0.35 and progress < 0.75:
            cap_frac = 0.05 + 0.10 * lam    # 5% em div puro -> 8.5%
            n_sde = min(n_sde, max(0,
                        int(round(cap_frac * self.pop_size))))
        if self.low_obj_mode and progress < 0.90 and self.last_rate_sde < max(0.08, 0.70 * self.last_rate_sbx):
            n_sde = min(n_sde, max(0, int(round(0.04 * self.pop_size))))
        n_sde = int(np.clip(n_sde, 0, self.pop_size - 1))

        positive = niche_count[niche_count > 0]
        if len(positive) > 0:
            sparse_th = max(1, int(self.xp.quantile(positive, self.sparse_quantile)))
        else:
            sparse_th = 1
        sparse_mask = niche_count <= sparse_th
        weights_conv, weights_div = self._parent_weights(
            self.pop,
            niche=niche,
            niche_count=niche_count,
            sparse_mask=sparse_mask,
            F_dev=F_pop,
            d_perp_dev=dist_to_niche,
        )

        n_global = self.pop_size - n_sde
        stall = 0.0 if self.last_archive_improved else 1.0
        rvx_share = 0.05 + 0.30 * self.rvx_share_base + 0.32 * stall + 0.18 * scarcity + 0.12 * (1.0 - entropy)
        rvx_share += 0.06 * max(0.0, self.last_rate_sbx - self.last_rate_sde)
        if progress > 0.70:
            rvx_share -= 0.15
        # [P7] rvx_share modulado por lambda em vez de flag binario
        if lam < 0.35:
            rvx_share *= (0.40 + 0.60 * lam)
        rvx_share = float(np.clip(rvx_share, 0.05, 0.70))
        n_rvx = int(round(n_global * rvx_share))
        n_rvx = int(np.clip(n_rvx, 0, n_global))
        n_sbx = n_global - n_rvx

        offsprings = []

        if n_sbx > 0:
            offsprings.append(self._create_sbx_offspring(n_sbx, [weights_conv, weights_div]))

        if n_rvx > 0:
            offsprings.append(self._create_rvx_offspring(n_rvx, [weights_conv, weights_div], niche))

        if n_sde > 0:
            offsprings.append(self._create_sde_offspring(n_sde, [weights_conv, weights_div], niche, niche_count))

        if len(offsprings) == 1:
            return offsprings[0]
        return Population.merge(*offsprings)

    def _advance(self, infills=None, **kwargs):
        # [NOVO] DNV para Local Search: Atualiza vetores normais dinâmicos baseados no Arquivo
        if self.nd_archive is not None and len(self.nd_archive) > 0:
            F_nd = np.asarray(_to_numpy(self.nd_archive.get("F", to_numpy=True)), dtype=float)
            ideal = np.asarray(_to_numpy(self.hyp_norm.ideal_point), dtype=float) if self.hyp_norm else np.min(F_nd, axis=0)
            nadir = np.asarray(_to_numpy(self.hyp_norm.nadir_point), dtype=float) if self.hyp_norm else np.max(F_nd, axis=0)
            span = np.maximum(nadir - ideal, 1e-12)
            F_norm = (F_nd - ideal) / span
            F_norm = F_norm / np.maximum(np.linalg.norm(F_norm, axis=1, keepdims=True), 1e-12)
            # Run k-means strictly on CPU to avoid kernel launch overhead inside loop
            sde_ref_np = _spherical_kmeans(F_norm, self.pop_size, max_iter=20, xp=np)
            self.sde_ref_dirs = self._as_backend_array(sde_ref_np, dtype=float, cache_key="sde_ref_dirs")
        else:
            self.sde_ref_dirs = self.ref_dirs_xp

        if infills is None or len(infills) == 0:
            return

        self._clear_backend_cache()

        self._update_offspring_state(infills)

        pool = Population.merge(self.pop, infills)
        pool = self.dup_elim.do(pool)
        self.pop = self._hybrid_survival(pool, n_survive=self.pop_size)

        # EEJ: atualiza Jacobiano global sobre a população sobrevivente
        self._update_eej(self.pop)

        self.nd_archive = self._update_archive(self.pop)

        # [NOVO] Two-archive: manter arquivo de convergência explícito.
        # O arquivo de convergência preserva os melhores indivíduos por ASF
        # no nicho, complementando o arquivo de diversidade (nd_archive).
        # Ref: Wang et al. 2015 (Two-archive); MaOEA-MS (ScienceDirect 2022).
        if self.use_two_archive:
            pool_full = Population.merge(self.nd_archive, self.conv_archive) \
                if self.conv_archive is not None else self.nd_archive
            self.conv_archive = self._build_conv_archive(pool_full)
            # Opt expõe união dos dois arquivos sem duplicatas
            combined = Population.merge(self.nd_archive, self.conv_archive)
            combined = self.dup_elim.do(combined)
            F_comb = combined.get("F")
            nd_idx = self._nds_do(F_comb, only_non_dominated_front=True)
            self.opt = combined[np.asarray(_to_numpy(nd_idx), dtype=int)]
        else:
            self.opt = self.nd_archive

        self._clear_backend_cache()

        progress = self._search_progress()
        self.last_archive_improved = self._update_archive_improvement()
        if self.last_archive_improved:
            self.stagnation_streak = 0
        else:
            self.stagnation_streak += 1

        # [AJUSTE] Adaptação de θ_pbi com proteção de diversidade.
        # Em PBI, θ muito baixo reduz a penalização perpendicular e pode colapsar
        # múltiplos nichos em uma única direção. Aqui, θ mantém piso mais alto em
        # modo diversity/escassez e só reduz mais em modo convergence.
        if self.theta_adapt:
            scarcity_now = max(0.0, self.coverage_target - self.last_coverage)
            if self.low_obj_mode:
                theta_tail = max(1.00, 0.65 * self.theta_pbi)
                theta_sched = self.theta_pbi - (self.theta_pbi - theta_tail) * progress
                theta_floor = 1.00 if self.problem.n_obj <= 2 else 1.20
                theta_cap = 2.60 if self.problem.n_obj <= 2 else 3.20
                theta_scaled = theta_sched + 0.10 * scarcity_now
                self._theta_current = float(np.clip(theta_scaled, theta_floor, theta_cap))
            else:
                weak_diversity = (
                    self.last_coverage < self.coverage_target
                    or self.last_entropy < self.entropy_diversity_threshold
                    or self.last_niche_cv > self.niche_cv_diversity_threshold
                )
                theta_tail = max(2.0, 0.45 * self.theta_pbi)
                theta_sched = self.theta_pbi - (self.theta_pbi - theta_tail) * progress

                if self.search_mode == "diversity":
                    mode_scale = 1.00 + 0.55 * scarcity_now
                    theta_floor = max(2.10, 0.45 * self.theta_pbi)
                    if weak_diversity:
                        theta_floor = max(theta_floor, 0.55 * self.theta_pbi)
                    theta_cap_scale = 1.20
                elif self.search_mode == "balanced":
                    mode_scale = 0.95 + 0.25 * scarcity_now
                    theta_floor = max(1.80, 0.38 * self.theta_pbi)
                    theta_cap_scale = 1.10
                else:
                    mode_scale = 0.85 + 0.15 * scarcity_now
                    theta_floor = max(1.40, 0.30 * self.theta_pbi)
                    theta_cap_scale = 1.00

                theta_scaled = theta_sched * mode_scale + 0.20 * scarcity_now
                theta_cap = max(self.theta_pbi * theta_cap_scale, theta_floor + 0.25)
                self._theta_current = float(np.clip(theta_scaled, theta_floor, theta_cap))
        else:
            self._theta_current = float(self.theta_pbi)

        conv_metric, div_metric = self._population_metrics(
            self.pop.get("F"),
            niche=self.pop.get("niche") if self.pop.has("niche") else None,
        )
        self._adapt_epsilon(progress)
        reward = self._policy_reward(conv_metric, div_metric)
        next_state = self._policy_state(conv_metric, div_metric)
        self._update_q_policy(reward, next_state)
        action = self._select_policy_action(next_state)
        self._apply_policy_action(action)
        self.last_conv_metric = float(conv_metric)
        self.last_div_metric = float(div_metric)
        self.last_q_state = int(next_state)
        self._record_telemetry(progress)

    # ------------------------------------------------------------------
    # Offspring creation
    # ------------------------------------------------------------------
    def _create_sbx_offspring(self, n_sbx: int, weights: Tuple[np.ndarray, np.ndarray]) -> Population:
        off = self.mating.do(
            self.problem,
            self.pop,
            n_sbx,
            algorithm=self,
            random_state=self.random_state,
        )
        if len(off) == 0:
            off = Population.new("X", self.pop.get("X")[:n_sbx].copy())

        if len(off) > n_sbx:
            off = off[:n_sbx]

        # [Dr. Evolve] POCEA Selection + TPCMaO Injection
        # Use dual weights for Convergence (p1) and Diversity (p2)
        n_pop = len(self.pop)
        replace = n_pop < len(off)
        
        # Generate twice as many indices because we might discard some or need mating pairs
        # But here we just need donors for differential components (C, J, etc)
        # Ideally we want the "Convergence" parents to donate the state (J, C, sigma)
        # to drive the search towards the front.
        
        # Helper to select standard parents using POCEA logic
        # Note: _select_parents_pocea returns a flat array [p1, p2, p1, p2...]
        use_archive_injection = bool(
            self.use_two_archive
            and (not self.low_obj_mode)
            and (self._search_progress() >= 0.45)
            and (self.search_mode == "convergence" or self.stagnation_streak >= 2)
        )

        parents_indices = self._select_parents_pocea(
            len(off),
            weights[0],
            weights[1],
            self.pop.get("niche"),
            archive_injection=use_archive_injection,
        )
        
        # We use the convergence parents (even indices) as the "primary" donors for state
        # to ensure good properties are inherited.
        primary_parents = np.asarray(parents_indices[0::2], dtype=int)
        if len(primary_parents) < len(off): # Should not happen if math is right
             extra = np.asarray(
                 self.random_state.choice(n_pop, size=len(off)-len(primary_parents), p=weights[0]),
                 dtype=int,
             )
             primary_parents = np.concatenate((primary_parents, extra))
        
        p1_rep = primary_parents[:len(off)]
        
        # Second parent for crossover can be the diversity partner
        secondary_parents = np.asarray(parents_indices[1::2], dtype=int)
        if len(secondary_parents) < len(off):
             extra = np.asarray(
                 self.random_state.choice(n_pop, size=len(off)-len(secondary_parents), p=weights[1]),
                 dtype=int,
             )
             secondary_parents = np.concatenate((secondary_parents, extra))
             
        p2_rep = secondary_parents[:len(off)]


        C = self.pop.get("C", to_numpy=True)
        pc = self.pop.get("pc", to_numpy=True)
        sigma_i = self.pop.get("sigma_i", to_numpy=True)
        anti = self.pop.get("anti", to_numpy=True)
        X = self.pop.get("X", to_numpy=True)
        F = self.pop.get("F", to_numpy=True)

        C_child = 0.5 * (C[p1_rep] + C[p2_rep])
        pc_child = 0.5 * (pc[p1_rep] + pc[p2_rep])
        sigma_child = 0.5 * (sigma_i[p1_rep] + sigma_i[p2_rep])
        anti_child = anti[p1_rep].copy()
        parent_X = 0.5 * (X[p1_rep] + X[p2_rep])
        parent_F = 0.5 * (F[p1_rep] + F[p2_rep])
        parent_idx = p1_rep.copy()
        origin = np.full(len(off), "sbx", dtype="<U4")
        rvx_F = np.full(len(off), np.nan, dtype=float)
        rvx_CR = np.full(len(off), np.nan, dtype=float)

        off.set(
            "C",
            C_child,
            "pc",
            pc_child,
            "sigma_i",
            sigma_child,
            "parent_X",
            parent_X,
            "parent_F",
            parent_F,
            "parent_idx",
            parent_idx,
            "origin",
            origin,
            "anti",
            anti_child,
            "rvx_F",
            rvx_F,
            "rvx_CR",
            rvx_CR,
        )
        return off

    def _create_sde_offspring(
        self,
        n_sde: int,
        weights: Tuple[np.ndarray, np.ndarray],
        niche: np.ndarray,
        niche_count: np.ndarray,
    ) -> Population:
        n_pop = len(self.pop)
        replace = n_pop < n_sde
        weights_div = np.asarray(_to_numpy(weights[1]), dtype=float)
        niche = np.asarray(_to_numpy(niche), dtype=int)
        niche_count = np.asarray(_to_numpy(niche_count), dtype=int)

        # NSGA-III niche rarity biases the local branch.
        rarity = 1.0 / (1.0 + niche_count[niche])
        scarcity = max(0.0, self.coverage_target - self.last_coverage)
        rarity_gain = 0.35 + self.diversity_boost * scarcity
        # [Dr. Evolve] SDE uses Diversity weights for local exploration
        local_w = weights_div * (0.30 + rarity_gain * rarity)
        local_w = np.clip(local_w, 1e-16, None)
        local_w = local_w / max(float(local_w.sum()), 1e-16)

        parent_idx = self.random_state.choice(np.arange(n_pop), size=n_sde, replace=replace, p=local_w)

        parents = self.pop[parent_idx]

        X = parents.get("X", to_numpy=True)
        F = parents.get("F", to_numpy=True)
        C = parents.get("C", to_numpy=True)
        pc = parents.get("pc", to_numpy=True)
        sigma_i = parents.get("sigma_i", to_numpy=True)
        anti = parents.get("anti", to_numpy=True)

        xl, xu = self.problem.xl, self.problem.xu
        n_var = self.problem.n_var
        n_obj = self.problem.n_obj

        X_new = np.empty_like(X)

        progress = self._search_progress()
        coverage_relax = max(0.0, self.coverage_target - self.last_coverage)
        max_step = (0.06 - 0.025 * progress) * (1.0 + 0.35 * coverage_relax) * (xu - xl)
        noise_scale = np.sqrt(max(self.epsilon_state, 1e-14)) * (1.0 - 0.35 * progress)
        drift_gain = self.drift_scale * (0.60 + 0.40 * progress) * (1.0 - 0.30 * coverage_relax)

        # Vectorized stochastic components
        z = self.random_state.standard_normal((n_sde, n_var))
        mirror_mask = self.random_state.random(n_sde) < self.mirror_rate
        z[mirror_mask] *= anti[mirror_mask, None]
        anti[mirror_mask] *= -1.0

        # EEJ global: direção de descida q compartilhada por todos os offspring
        J_eej = self._eej if self._eej is not None else np.zeros((n_obj, n_var))
        q_global = self._eej_q if self._eej_q is not None else _compute_eej_descent(J_eej)

        # Usa Schaeffler QP sobre o EEJ global (mesma lambda para todos)
        lam = solve_schaeffler_qp(J_eej, max_iter=8)

        # Direção de descida via EEJ global — inicializa q com q_global
        q = np.empty((n_sde, n_var), dtype=float)
        q_norm = np.linalg.norm(q_global)
        if q_norm < 1e-14:
            q[:] = 0.0
        else:
            q[:] = q_global

        # [NOVO] Blending Schaeffler + Tchebycheff para direção de descida.
        # Para m > 4, aplica blending Tchebycheff por indivíduo sobre a
        # direção EEJ global para especializar por nicho.
        # Ref: MaOEA-DISC (ScienceDirect 2024); θ-DEA (Yuan et al. 2015).
        if n_obj > 4:
            alpha_blend = float(np.clip(progress * 0.60 + 0.10, 0.10, 0.65))
            niche_local = niche[parent_idx] if len(parent_idx) > 0 else niche[:n_sde]
            ref_dirs_np = self.ref_dirs_np
            for i in range(n_sde):
                ni = int(np.clip(niche_local[i], 0, len(self.ref_dirs) - 1))
                w_ref = np.maximum(ref_dirs_np[ni], 1e-8)
                f_norm_i = (F[i] - np.min(F, axis=0)) / np.maximum(
                    np.max(F, axis=0) - np.min(F, axis=0), 1e-12
                )
                tch_grad = f_norm_i / w_ref
                _tau     = 3.0
                _r_shift = tch_grad - tch_grad.max()
                _exp_r   = np.exp(_tau * _r_shift)
                tch_dir  = _exp_r / max(float(_exp_r.sum()), 1e-14)
                lam_i = (1.0 - alpha_blend) * lam + alpha_blend * _project_to_simplex(tch_dir)
                lam_i = np.maximum(lam_i, 1e-12)
                lam_i /= np.sum(lam_i)
                q_i = J_eej.T @ lam_i
                q_n = float(np.linalg.norm(q_i))
                q[i] = q_i / q_n if q_n > 1e-12 else q_global

        # Confiança do EEJ: baseada na norma do Jacobiano
        jac_norm = float(np.linalg.norm(J_eej))
        jac_ref = max(jac_norm, 1e-14)
        model_conf_scalar = float(np.clip(jac_norm / jac_ref, 0.0, 1.0))
        model_conf = np.full(n_sde, model_conf_scalar, dtype=float)

        # Differential fallback when Jacobian-induced q is weak/noisy.
        # Preserves global exploration capacity on difficult many-objective landscapes.
        X_pop = self.pop.get("X", to_numpy=True)
        de_i = self.random_state.choice(np.arange(n_pop), size=n_sde, replace=True, p=weights_div)
        de_j = self.random_state.choice(np.arange(n_pop), size=n_sde, replace=True, p=weights_div)
        de_dir = X_pop[de_i] - X_pop[de_j]
        de_norm = np.linalg.norm(de_dir, axis=1, keepdims=True)
        de_dir[de_norm.ravel() > 1e-14] /= de_norm[de_norm.ravel() > 1e-14]

        Lz = np.empty_like(z)
        for i in range(n_sde):
            C_i = self._stabilize_cov(C[i])
            C_i, L_i = self._cholesky_with_jitter(C_i)
            C[i] = C_i
            Lz[i] = L_i @ z[i]

        sigma_vec = np.sqrt(np.maximum(sigma_i, 1e-14))[:, None]
        drift_conf = (0.30 + 0.70 * model_conf)[:, None]
        drift = -drift_gain * drift_conf * sigma_i[:, None] * q
        de_gain = 0.75 * self.de_fallback_scale * (1.0 - progress) * (0.45 + 0.55 * coverage_relax)
        drift += de_gain * sigma_i[:, None] * (1.0 - model_conf)[:, None] * de_dir
        diffusion_conf = (0.45 + 0.55 * model_conf)[:, None]
        diffusion = diffusion_conf * noise_scale * sigma_vec * Lz
        momentum = 0.05 * sigma_i[:, None] * np.tanh(pc)
        step = np.clip(drift + diffusion + momentum, -max_step, max_step)
        X_new = np.clip(X + step, xl, xu)

        off = Population.new("X", X_new)

        origin = np.full(n_sde, "sde", dtype="<U4")
        rvx_F = np.full(n_sde, np.nan, dtype=float)
        rvx_CR = np.full(n_sde, np.nan, dtype=float)
        off.set(
            "C",
            C.copy(),
            "pc",
            pc.copy(),
            "sigma_i",
            sigma_i.copy(),
            "parent_X",
            X.copy(),
            "parent_F",
            F.copy(),
            "parent_idx",
            parent_idx.copy(),
            "origin",
            origin,
            "anti",
            anti.copy(),
            "rvx_F",
            rvx_F,
            "rvx_CR",
            rvx_CR,
        )
        return off

    def _create_rvx_offspring(
        self,
        n_rvx: int,
        weights: Tuple[np.ndarray, np.ndarray],
        niche: np.ndarray,
    ) -> Population:
        n_pop = len(self.pop)
        replace = n_pop < n_rvx
        weights_conv = np.asarray(_to_numpy(weights[0]), dtype=float)
        weights_div = np.asarray(_to_numpy(weights[1]), dtype=float)
        niche = np.asarray(_to_numpy(niche), dtype=int)

        # [Dr. Evolve] POCEA Selection for RVX
        # Select target vectors using Convergence weights to drive towards front
        target_idx = self.random_state.choice(np.arange(n_pop), size=n_rvx, replace=replace, p=weights_conv)
        parents = self.pop[target_idx]

        X_pop = self.pop.get("X", to_numpy=True)
        F_pop = self.pop.get("F", to_numpy=True)
        C = parents.get("C", to_numpy=True)
        pc = parents.get("pc", to_numpy=True)
        sigma_i = parents.get("sigma_i", to_numpy=True)
        anti = parents.get("anti", to_numpy=True)
        X_t = parents.get("X", to_numpy=True)
        F_t = parents.get("F", to_numpy=True)

        xl, xu = self.problem.xl, self.problem.xu
        n_var = self.problem.n_var

        f_min = np.min(F_pop, axis=0)
        f_max = np.max(F_pop, axis=0)
        f_span = np.maximum(f_max - f_min, 1e-12)
        conv = np.sum((F_pop - f_min) / f_span, axis=1)
        global_best = int(np.argmin(conv))

        niche_best = np.full(len(self.ref_dirs), -1, dtype=int)
        for k in np.unique(niche):
            idx = np.where(niche == k)[0]
            if len(idx) > 0:
                niche_best[int(k)] = int(idx[np.argmin(conv[idx])])

        X_new = np.empty_like(X_t)
        used_f = np.empty(n_rvx, dtype=float)
        used_cr = np.empty(n_rvx, dtype=float)

        for i in range(n_rvx):
            xi_idx = int(target_idx[i])
            xi = X_pop[xi_idx]
            ni = int(niche[xi_idx]) if xi_idx < len(niche) else 0
            bi = niche_best[ni] if 0 <= ni < len(niche_best) and niche_best[ni] >= 0 else global_best

            F_norm = (F_pop - f_min) / f_span

            # [Dr. Evolve] Neighborhood Mating Restriction (POCEA/MOEA/D logic).
            # In high-dim (m>3), global mating produces poor offspring.
            # Select r1, r2 from neighbors (similar niche angle) with prob 0.9.
            if self.random_state.random() < self.prob_neighbor_mating and niche is not None:
                # Simple neighborhood: same niche or random neighbor in ref_dirs?
                # Using same-niche candidates from population if available.
                # Fast approx: pick random index, check if niche is close?
                # Better: Pre-compute niche neighbors? Too slow.
                # Heuristic: Pick r1, r2 from general pool, but prioritize those with
                # positive cosine similarity to current 'xi' parents.
                
                # Implementation: Re-sample candidates if angle is bad (simple rejection).
                # 3 tries to find a "neighbor" (same niche or dot product > 0.8).
                r1_cand = -1
                for _ in range(3):
                    c = int(self.random_state.choice(np.arange(n_pop), p=weights_div))
                    if c == xi_idx or c == bi: continue
                    if niche[c] == ni: # Same niche = perfect neighbor
                        r1_cand = c
                        break
                    # Angle check
                    u = F_norm[xi_idx] / (np.linalg.norm(F_norm[xi_idx]) + 1e-12)
                    v = F_norm[c] / (np.linalg.norm(F_norm[c]) + 1e-12)
                    if np.dot(u, v) > 0.7: # Approx < 45 degrees
                        r1_cand = c
                        break
                if r1_cand == -1:
                    r1_cand = int(self.random_state.choice(np.arange(n_pop), p=weights_div))
                r1 = r1_cand
                
                r2 = int(self.random_state.choice(np.arange(n_pop), p=weights_div))
                if r1 == r2: r2 = (r2 + 1) % n_pop
            else:
                # Global mating
                r = self.random_state.choice(np.arange(n_pop), size=2, replace=(n_pop < 2), p=weights_div)
                r1, r2 = int(r[0]), int(r[1])
                if r1 == r2:
                    r2 = (r2 + 1) % n_pop

            # Success-history style parameters for DE-like branch.
            fj = np.nan
            for _ in range(5):
                cand = self.rvx_mu_f + 0.1 * np.tan(np.pi * (self.random_state.random() - 0.5))
                if cand > 0:
                    fj = min(cand, 1.0)
                    break
            if not np.isfinite(fj):
                fj = self.rvx_mu_f
            crj = float(np.clip(self.random_state.normal(self.rvx_mu_cr, 0.1), 0.0, 1.0))

            mutant = xi + fj * (X_pop[bi] - xi) + fj * (X_pop[r1] - X_pop[r2])

            mask = self.random_state.random(n_var) < crj
            jrand = int(self.random_state.integers(0, n_var))
            mask[jrand] = True
            trial = xi.copy()
            trial[mask] = mutant[mask]
            X_new[i] = np.clip(trial, xl, xu)
            used_f[i] = fj
            used_cr[i] = crj

        off = Population.new("X", X_new)
        origin = np.full(n_rvx, "rvx", dtype="<U4")
        off.set(
            "C",
            C.copy(),
            "pc",
            pc.copy(),
            "sigma_i",
            sigma_i.copy(),
            "parent_X",
            X_t.copy(),
            "parent_F",
            F_t.copy(),
            "parent_idx",
            target_idx.copy(),
            "origin",
            origin,
            "anti",
            anti.copy(),
            "rvx_F",
            used_f,
            "rvx_CR",
            used_cr,
        )
        return off

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def _update_offspring_state(self, off: Population):
        n = len(off)
        if n == 0:
            return

        # This branch is dominated by Python loops and small linear-algebra ops.
        # NumPy is consistently faster than CuPy here due to lower kernel-launch overhead.
        X = np.asarray(off.get("X", to_numpy=True), dtype=float)
        F = np.asarray(off.get("F", to_numpy=True), dtype=float)
        parent_X = np.asarray(off.get("parent_X", to_numpy=True), dtype=float)
        parent_F = np.asarray(off.get("parent_F", to_numpy=True), dtype=float)
        C = np.asarray(off.get("C", to_numpy=True), dtype=float)
        pc = np.asarray(off.get("pc", to_numpy=True), dtype=float)
        sigma_i = np.asarray(off.get("sigma_i", to_numpy=True), dtype=float)
        origin = np.asarray(off.get("origin"), dtype=str)
        anti = np.asarray(off.get("anti", to_numpy=True), dtype=float)
        parent_idx = np.asarray(off.get("parent_idx"), dtype=int)
        rvx_F = np.asarray(off.get("rvx_F", to_numpy=True), dtype=float)
        rvx_CR = np.asarray(off.get("rvx_CR", to_numpy=True), dtype=float)
        ref_dirs_np = self.ref_dirs_np

        n_var = self.problem.n_var

        success_sde = 0
        total_sde = 0
        success_sbx = 0
        total_sbx = 0
        sde_z_success = []
        succ_rvx_f = []
        succ_rvx_cr = []

        parent_niche = None
        if self.pop.has("niche"):
            p_niche = np.asarray(_to_numpy(self.pop.get("niche")), dtype=int)
            if p_niche is not None and len(p_niche) == len(self.pop):
                safe_idx = np.clip(parent_idx.astype(int), 0, max(len(self.pop) - 1, 0))
                parent_niche = p_niche[safe_idx]

        all_finite_parent = np.all(np.isfinite(parent_F), axis=1)
        if np.any(all_finite_parent):
            F_stack = np.vstack((F, parent_F[all_finite_parent]))
        else:
            F_stack = F

        ideal = np.min(F_stack, axis=0)
        nadir = np.max(F_stack, axis=0)
        span = np.maximum(nadir - ideal, 1e-12)

        F_norm = (F - ideal) / span
        parent_norm = np.full_like(parent_F, np.inf, dtype=float)
        parent_norm[all_finite_parent] = (parent_F[all_finite_parent] - ideal) / span
        child_mean = np.mean(F_norm, axis=1)
        child_norm_l2 = np.linalg.norm(F_norm, axis=1)
        parent_mean = np.full(n, np.inf, dtype=float)
        parent_norm_l2 = np.full(n, np.inf, dtype=float)
        parent_mean[all_finite_parent] = np.mean(parent_norm[all_finite_parent], axis=1)
        parent_norm_l2[all_finite_parent] = np.linalg.norm(parent_norm[all_finite_parent], axis=1)
        eye_n = np.eye(n_var, dtype=float)

        for i in range(n):
            branch = str(origin[i])
            if branch == "sde":
                total_sde += 1
            elif branch in {"sbx", "rvx"}:
                total_sbx += 1

            f_parent = parent_F[i]
            success = False
            if all_finite_parent[i]:
                rel = self.dominator.get_relation(F[i], f_parent)
                success = (rel == 1) or (
                    rel == 0
                    and (
                        float(child_mean[i]) < float(parent_mean[i]) - 1e-4
                        or float(child_norm_l2[i]) < float(parent_norm_l2[i]) - 1e-4
                    )
                )
                if (not success) and rel == 0 and parent_niche is not None:
                    niche_i = int(np.clip(parent_niche[i], 0, len(ref_dirs_np) - 1))
                    w = np.maximum(ref_dirs_np[niche_i], 1e-8)
                    asf_child = float(np.max(F_norm[i] / w))
                    asf_parent = float(np.max(parent_norm[i] / w))
                    asf_improved = asf_child < (asf_parent - 1e-4)
                    mean_improved = float(child_mean[i]) < float(parent_mean[i]) - 1e-4
                    if branch == "sde":
                        success = asf_improved and mean_improved
                    else:
                        success = asf_improved or mean_improved

            if success:
                if branch == "sde":
                    success_sde += 1
                elif branch == "sbx":
                    success_sbx += 1
                elif branch == "rvx":
                    success_sbx += 1
                    if np.isfinite(rvx_F[i]):
                        succ_rvx_f.append(float(rvx_F[i]))
                    if np.isfinite(rvx_CR[i]):
                        succ_rvx_cr.append(float(rvx_CR[i]))

            s = X[i] - parent_X[i]
            sigma_eff = max(float(sigma_i[i]), 1e-14)
            z = s / sigma_eff
            if success and branch == "sde":
                sde_z_success.append(z.copy())

            if all_finite_parent[i]:
                pass  # EEJ é atualizado globalmente em _advance(), não por indivíduo

            if success:
                pc[i] = (
                    (1.0 - self.c_path) * pc[i]
                    + np.sqrt(self.c_path * (2.0 - self.c_path)) * z
                )
                C[i] = (
                    (1.0 - self.c_cov) * C[i]
                    + self.c_cov * np.outer(pc[i], pc[i])
                    + 0.03 * self.c_cov * np.outer(z, z)
                )
            else:
                relax = 0.18 if branch == "sde" else 0.07
                C[i] = (1.0 - relax * self.c_cov) * C[i] + relax * self.c_cov * eye_n
                pc[i] = (1.0 - self.c_path) * pc[i]

            C[i] = self._stabilize_cov(C[i])

            target = self.target_success if branch == "sde" else max(0.14, 0.85 * self.target_success)
            gain = 1.0 if success else (0.0 if branch == "sbx" else -0.10)
            sigma_new = sigma_eff * np.exp((self.c_sigma / max(self.sigma_damp, 1e-8)) * (gain - target))
            sigma_i[i] = np.clip(sigma_new, self.sigma_min, self.sigma_max)
            anti[i] = -anti[i] if success and branch == "sde" else anti[i]

        off.set(
            "C",
            C,
            "pc",
            pc,
            "sigma_i",
            sigma_i,
            "anti",
            anti,
        )
        self.last_rate_sde = float(success_sde / total_sde) if total_sde > 0 else 0.0
        self.last_rate_sbx = float(success_sbx / total_sbx) if total_sbx > 0 else 0.0
        self.last_total_sde = int(total_sde)
        self.last_success_sde = int(success_sde)
        if len(succ_rvx_f) > 0:
            f_arr = np.asarray(succ_rvx_f, dtype=float)
            num = np.sum(f_arr * f_arr)
            den = np.sum(f_arr)
            lehmer_f = float(num / max(den, 1e-12))
            self.rvx_mu_f = (1.0 - self.rvx_adapt_lr) * self.rvx_mu_f + self.rvx_adapt_lr * np.clip(lehmer_f, 0.05, 1.0)
        if len(succ_rvx_cr) > 0:
            cr_mean = float(np.mean(succ_rvx_cr))
            self.rvx_mu_cr = (1.0 - self.rvx_adapt_lr) * self.rvx_mu_cr + self.rvx_adapt_lr * np.clip(cr_mean, 0.05, 1.0)
        if len(sde_z_success) > 0:
            arr = np.asarray(sde_z_success, dtype=float)
            self.last_sde_step_norm = float(np.mean(np.linalg.norm(arr, axis=1)))
            self.last_sde_z_mean = np.mean(arr, axis=0)
        else:
            self.last_sde_step_norm = 0.0
            self.last_sde_z_mean = None
        self._update_operator_credit(success_sde, total_sde, success_sbx, total_sbx)

    def _update_operator_credit(
        self, success_sde: int, total_sde: int, success_sbx: int, total_sbx: int
    ):
        alpha = 0.2

        if total_sde > 0:
            rate_sde = success_sde / total_sde
            self.credit_sde = (1.0 - alpha) * self.credit_sde + alpha * rate_sde

        if total_sbx > 0:
            rate_sbx = success_sbx / total_sbx
            self.credit_sbx = (1.0 - alpha) * self.credit_sbx + alpha * rate_sbx

        if self.low_obj_mode and total_sde > 0 and total_sbx > 0:
            if rate_sde < max(0.05, 0.55 * rate_sbx):
                self.credit_sde *= 0.90

        self.operator_trials_sde += max(total_sde, 0)
        self.operator_trials_sbx += max(total_sbx, 0)
        self.total_operator_events += max(total_sde + total_sbx, 0)

        log_term = np.log(max(2.0, float(self.total_operator_events)))
        bonus_sde = self.ucb_exploration * np.sqrt(log_term / float(self.operator_trials_sde))
        bonus_sbx = self.ucb_exploration * np.sqrt(log_term / float(self.operator_trials_sbx))

        score_sde = self.credit_sde + bonus_sde
        score_sbx = self.credit_sbx + bonus_sbx

        span = self.ratio_bounds[1] - self.ratio_bounds[0]
        target_ratio = self.ratio_bounds[0] + span / (1.0 + np.exp(-6.0 * (score_sde - score_sbx)))
        self.ratio_sde = float(
            np.clip(
                (1.0 - self.adaptive_ratio_lr) * self.ratio_sde + self.adaptive_ratio_lr * target_ratio,
                self.ratio_bounds[0],
                self.ratio_bounds[1],
            )
        )

    # ------------------------------------------------------------------
    # Selection and survival
    # ------------------------------------------------------------------
    def _parent_weights(
        self,
        pop: Population,
        niche: Optional[np.ndarray] = None,
        niche_count: Optional[np.ndarray] = None,
        sparse_mask: Optional[np.ndarray] = None,
        F_dev: Optional[Any] = None,
        rank_dev: Optional[Any] = None,
        d_perp_dev: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate p1 (convergence) and p2 (diversity) selection weights. Supports GPU."""
        xp = self.xp
        F = F_dev if F_dev is not None else self._as_backend_array(pop.get("F"), dtype=float, cache_key="pw_F")
        F_np = np.asarray(_to_numpy(F), dtype=float)
        ref_dirs = self.ref_dirs_xp
        if rank_dev is not None:
            rank = xp.asarray(rank_dev, dtype=int)
        elif pop.has("rank"):
            rank = self._as_backend_array(pop.get("rank"), dtype=int, cache_key="pw_rank")
        else:
            _, rank_np = self._nds_do(F_np, return_rank=True)
            rank = _to_device(rank_np, self.use_gpu)

        if niche is None:
            niche, d_perp = self._associate_niches(F)
        else:
            niche = xp.asarray(niche, dtype=int)
            if d_perp_dev is not None:
                d_perp = xp.asarray(d_perp_dev, dtype=float)
            elif pop.has("dist_to_niche"):
                d_perp = self._as_backend_array(pop.get("dist_to_niche"), dtype=float, cache_key="pw_dist")
            else:
                _, d_perp = self._associate_niches(F)

        if niche_count is None:
            niche_count = xp.bincount(niche, minlength=len(self.ref_dirs))
        else:
            niche_count = xp.asarray(niche_count, dtype=float)

        rarity = 1.0 / (1.0 + niche_count[niche])
        scarcity = max(0.0, self.coverage_target - self.last_coverage)

        dist_score = 1.0 - self._normalize01(d_perp)
        
        f_min = xp.min(F, axis=0)
        f_max = xp.max(F, axis=0)
        f_span = xp.maximum(f_max - f_min, 1e-12)
        F_norm = (F - f_min) / f_span

        # [P1] ASF-Tchebycheff per-niche convergence score.
        w_ref = xp.maximum(ref_dirs[niche], 1e-8)
        asf_vals = xp.max(F_norm / w_ref, axis=1)
        conv_score = 1.0 - self._normalize01(asf_vals)

        # Unit-vector angle term
        f_unit = F_norm / xp.maximum(xp.linalg.norm(F_norm, axis=1, keepdims=True), 1e-12)
        ref = ref_dirs[niche]
        ref = ref / xp.maximum(xp.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        cosv = xp.sum(f_unit * ref, axis=1)
        cosv = xp.clip(cosv, -1.0, 1.0)
        angle = xp.arccos(cosv) / self.xp.pi
        angle_score = 1.0 - self._normalize01(angle)

        sparse_boost = xp.ones(len(pop), dtype=float)
        if sparse_mask is not None and scarcity > 0.0:
            sparse_mask_dev = xp.asarray(sparse_mask, dtype=bool)
            sparse_boost += (0.25 + 0.25 * scarcity) * sparse_mask_dev[niche].astype(float)

        angle_gain = self.angle_adapt_gain * (0.25 + 0.75 * scarcity)
        rank_gain = xp.exp(-1.2 * rank.astype(float))

        # [Dr. Evolve] Dual-mode weights
        w_conv = rank_gain * (0.15 + 0.55 * conv_score + 0.15 * dist_score + 0.15 * angle_score)
        w_div = rank_gain * (
            0.10 + 0.15 * conv_score + 0.40 * dist_score + 
            (0.25 + 0.35 * scarcity) * rarity + angle_gain * angle_score
        )
        
        if sparse_mask is not None and scarcity > 0.0:
            w_div *= sparse_boost

        w_conv = xp.clip(w_conv, 1e-16, None)
        w_conv /= w_conv.sum()
        w_div = xp.clip(w_div, 1e-16, None)
        w_div /= w_div.sum()

        return _to_numpy(w_conv), _to_numpy(w_div)

    def _conv_archive_anchor_indices(self, n_pick: int) -> np.ndarray:
        """Map convergence-archive elites to current population indices. Supports GPU."""
        if n_pick <= 0 or self.conv_archive is None or len(self.conv_archive) == 0 or len(self.pop) == 0:
            return self.xp.array([], dtype=int)

        xp = self.xp
        X_pop = self._as_backend_array(self.pop.get("X"), dtype=float, cache_key="conv_anchor_X_pop")
        X_arc = self._as_backend_array(self.conv_archive.get("X"), dtype=float, cache_key="conv_anchor_X_arc")
        
        picked = []
        used = set()
        order = self.random_state.permutation(len(X_arc))

        for ai in order:
            d = xp.linalg.norm(X_pop - X_arc[int(ai)], axis=1)
            idx = int(xp.argmin(d))
            if idx not in used:
                used.add(idx)
                picked.append(idx)
            if len(picked) >= n_pick:
                break

        if len(picked) < n_pick:
            F = self._as_backend_array(self.pop.get("F"), dtype=float, cache_key="conv_anchor_F")
            f_min = xp.min(F, axis=0)
            f_span = xp.maximum(xp.max(F, axis=0) - f_min, 1e-12)
            conv = xp.sum((F - f_min) / f_span, axis=1)
            for idx in _to_numpy(xp.argsort(conv)):
                j = int(idx)
                if j not in used:
                    used.add(j)
                    picked.append(j)
                if len(picked) >= n_pick:
                    break

        return self.xp.asarray(picked, dtype=int)

    def _select_parents_pocea(
        self,
        n_parents: int,
        w_conv: np.ndarray,
        w_div: np.ndarray,
        niche: np.ndarray,
        archive_injection: bool = True
    ) -> np.ndarray:
        """
        [Dr. Evolve] POCEA-style Pairwise Offspring Generation (2021).
        Selects pairs (p1, p2) where:
          - p1 is selected for CONVERGENCE (exploitation).
          - p2 is selected for DIVERSITY (exploration) or is a Niche Neighbor of p1.
        
        Also implements TPCMaO Archive Injection:
          - With prob ~10%, p1 is replaced by a high-quality solution from conv_archive.
        """
        n_pop = len(self.pop)
        w_conv = np.asarray(_to_numpy(w_conv), dtype=float)
        w_div = np.asarray(_to_numpy(w_div), dtype=float)
        niche = np.asarray(_to_numpy(niche), dtype=int)
        
        # Ensure we generate pairs (even number), then trim if needed
        n_gen = n_parents
        if n_gen % 2 != 0:
            n_gen += 1
            
        parents_gen = np.empty(n_gen, dtype=int)
        
        # Half of parents are p1 (convergence), half are p2 (diversity/neighbor)
        n_couples = n_gen // 2
        
        # 1. Select P1 (Convergence Leaders)
        p1_idx = self.random_state.choice(n_pop, size=n_couples, replace=True, p=w_conv)
        
        # [TPCMaO] Archive Injection
        # Inject elitist solutions from convergence archive to accelerate evolution
        if archive_injection and self.conv_archive is not None and len(self.conv_archive) > 0:
            n_inject = int(self.random_state.binomial(n_couples, self.archive_injection_rate))
            if n_inject > 0:
                inject_slots = self.random_state.choice(n_couples, size=n_inject, replace=False)
                anchors = self._conv_archive_anchor_indices(max(n_inject, 2))
                if len(anchors) > 0:
                    replace = len(anchors) < n_inject
                    repl = self.random_state.choice(anchors, size=n_inject, replace=replace)
                    p1_idx[inject_slots] = repl

        # 2. Select P2 (Diversity Partners)
        # For each p1, we want a p2 that is either:
        #   a) A random diversity parent (Global Exploration)
        #   b) A neighbor in the same niche (Local Exploitation/Mating Restriction)
        
        p2_idx = np.empty(n_couples, dtype=int)
        
        _has_nbr = hasattr(self, "_ref_nbr") and self._ref_nbr is not None
        for i in range(n_couples):
            lead     = p1_idx[i]
            my_niche = int(niche[lead]) if lead < len(niche) else 0

            if self.random_state.random() < self.prob_neighbor_mating:
                # [P3] Pool = niche_proprio U vizinhos angulares.
                # Para m >= 6 o nicho exato esta vazio na maioria das
                # geracoes; o pool ampliado garante mating restriction
                # funcional em qualquer dimensionalidade.
                if _has_nbr:
                    nbr_row = np.asarray(self._ref_nbr[my_niche], dtype=int).tolist()
                    nbr_niches = set(nbr_row) | {my_niche}
                else:
                    nbr_niches = {my_niche}
                cand_pool = np.array(
                    [j for j in range(n_pop)
                     if j != lead and niche[j] in nbr_niches],
                    dtype=int,
                )
                if len(cand_pool) >= 1:
                    wl   = np.clip(w_div[cand_pool], 1e-16, None)
                    cand = int(self.random_state.choice(cand_pool,
                                                        p=wl / wl.sum()))
                else:
                    wfb       = w_div.copy()
                    wfb[lead] = 0.0
                    s         = wfb.sum()
                    cand = int(self.random_state.choice(
                               n_pop, p=wfb / max(s, 1e-16)))
            else:
                wfb       = w_div.copy()
                wfb[lead] = 0.0
                s         = wfb.sum()
                cand = int(self.random_state.choice(
                           n_pop, p=wfb / max(s, 1e-16)))
            p2_idx[i] = cand

        parents_gen[0::2] = p1_idx
        parents_gen[1::2] = p2_idx
        
        return parents_gen[:n_parents]

    def _hybrid_survival(
        self, pop: Population, n_survive: int, ideal: Optional[np.ndarray] = None, nadir: Optional[np.ndarray] = None
    ) -> Population:
        F = pop.get("F")

        # [AJUSTE] SDR com gate adaptativo para evitar colapso de diversidade.
        # Em cenários de escassez/diversity-mode, SDR pode gerar fronts muito
        # granulares (muitos singletons) e reduzir cobertura de nichos.
        progress = self._search_progress()
        scarcity = max(0.0, self.coverage_target - self.last_coverage)
        weak_diversity = (
            self.search_mode == "diversity"
            or self.last_coverage < self.coverage_target
            or self.last_entropy < self.entropy_diversity_threshold
            or self.last_niche_cv > self.niche_cv_diversity_threshold
        )
        use_sdr_now = bool(
            self.use_sdr
            and self.problem is not None
            and self.problem.n_obj > 3
            and progress >= 0.05  # Enable SDR slightly earlier
            and (not weak_diversity)
        )

        if use_sdr_now:
            # [P5] Sigma SDR modulado CONTINUAMENTE pela entropia.
            # Remove o gate binario (singleton_ratio > 0.60) que
            # desativava SDR quando a diversidade colapsava.
            # Com H baixo -> sigma_eff cai -> SDR ~ Pareto classico
            # Com H alto  -> sigma_eff = sigma_base -> SDR completo
            _H          = float(self.last_entropy)
            _ent_factor = float(np.clip((_H - 0.40) / 0.60, 0.0, 1.0))
            sigma_eff   = float(
                self.sdr_sigma * (0.20 + 0.80 * _ent_factor))
            sigma_eff   = float(np.clip(
                sigma_eff * (1.0 - 0.40 * scarcity),
                0.0, self.sdr_sigma))
            all_fronts, rank = self._sdr_fronts(F, sigma=sigma_eff)

        if not use_sdr_now:
            all_fronts, rank = self._nds_do(F, return_rank=True, n_stop_if_ranked=n_survive)

        # Guardrail: ensure we only keep fronts needed to reach n_survive.
        # Without this, using all SDR fronts can make the population grow.
        fronts = []
        ranked = 0
        for front in all_fronts:
            fronts.append(np.asarray(_to_numpy(front), dtype=int))
            ranked += len(front)
            if ranked >= n_survive:
                break

        if len(fronts) == 0:
            return pop[:0]

        picked = np.concatenate(fronts).astype(int)
        pop = pop[picked]
        F = pop.get("F")
        rank = np.asarray(rank, dtype=int)[picked]

        mapped_fronts = []
        cursor = 0
        for front in fronts:
            size = len(front)
            mapped_fronts.append(np.arange(cursor, cursor + size, dtype=int))
            cursor += size

        nd_for_norm = mapped_fronts[0] if len(mapped_fronts) > 0 else None
        niche_of, dist_to_niche = self._associate_niches(F, nd=nd_for_norm)
        pop.set("rank", rank, "niche", niche_of, "dist_to_niche", dist_to_niche)

        if len(pop) <= n_survive:
            return pop

        if len(mapped_fronts) == 1:
            n_remaining = n_survive
            until_last_front = np.array([], dtype=int)
            niche_count = self.xp.zeros(len(self.ref_dirs), dtype=int)
        else:
            until_last_front = np.concatenate(mapped_fronts[:-1]).astype(int)
            niche_count = _calc_niche_count(len(self.ref_dirs), niche_of[until_last_front])
            n_remaining = int(n_survive - len(until_last_front))

        split_front = mapped_fronts[-1].astype(int)
        selected = self._niching(
            split_front,
            n_pick=n_remaining,
            niche_of=niche_of,
            dist_to_niche=dist_to_niche,
            niche_count=niche_count,
            F=F,
        )
        selected_np = np.asarray(_to_numpy(selected), dtype=int)
        survivors = np.concatenate((until_last_front, selected_np)).astype(int)
        survivors = self._elite_refinement(survivors, pop, niche_of, F)
        return pop[survivors]

    def _sdr_fronts(self, F: np.ndarray, sigma: Optional[float] = None):
        """
        [NOVO] Classificação em frontes usando SDR (Strengthened Dominance Relation).
        Normaliza F e computa σ proporcional ao span médio.
        Totalmente vetorizado via NumPy (sem loops lentos de Python).
        Ref: Tian et al. 2018; 3DEA (Zhang et al. 2024).
        """
        F_np = np.asarray(_to_numpy(F), dtype=float)
        n = len(F_np)
        f_min = np.min(F_np, axis=0)
        f_max = np.max(F_np, axis=0)
        span = np.maximum(f_max - f_min, 1e-12)
        F_norm = (F_np - f_min) / span
        sigma = self.sdr_sigma if sigma is None else float(np.clip(sigma, 0.0, 0.20))

        # diff[i, j] = F_norm[j] - F_norm[i]  (positivo = i é melhor que j)
        diff = F_norm[None, :, :] - F_norm[:, None, :]
        
        pareto_dom = np.logical_and(
            np.all(diff >= 0.0, axis=-1),
            np.any(diff > 0.0, axis=-1)
        )
        
        sdr_relax = np.logical_and(
            np.any(diff > sigma, axis=-1),
            np.all(diff >= -sigma, axis=-1)
        )
        
        dominates = np.logical_or(pareto_dom, sdr_relax)
        np.fill_diagonal(dominates, False)
        
        dom_count = np.sum(dominates, axis=0).astype(int)
        dominated_by = [list(np.where(dominates[i])[0]) for i in range(n)]

        rank = np.empty(n, dtype=int)
        fronts = []
        current_front = list(np.where(dom_count == 0)[0])
        r = 0
        while current_front:
            fronts.append(np.array(current_front, dtype=int))
            for i in current_front:
                rank[i] = r
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            r += 1

        return fronts, rank

    def _niching(
        self,
        candidates: np.ndarray,
        n_pick: int,
        niche_of: np.ndarray,
        dist_to_niche: np.ndarray,
        niche_count: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        candidates = np.asarray(_to_numpy(candidates), dtype=int)
        niche_of = np.asarray(_to_numpy(niche_of), dtype=int)
        dist_to_niche = np.asarray(_to_numpy(dist_to_niche), dtype=float)
        niche_count = np.asarray(_to_numpy(niche_count), dtype=int)
        F = np.asarray(_to_numpy(F), dtype=float)

        if len(candidates) <= n_pick:
            return candidates

        progress = self._search_progress()
        scarcity = max(0.0, self.coverage_target - self.last_coverage)
        if self.search_mode == "diversity":
            rarity_w = float(np.clip(0.68 + 0.22 * scarcity, 0.55, 0.92))
        elif self.search_mode == "convergence":
            rarity_w = float(np.clip(0.42 + 0.20 * scarcity, 0.32, 0.78))
        else:
            rarity_w = float(np.clip(0.55 + 0.35 * scarcity, 0.40, 0.90))
        conv_w = 1.0 - rarity_w
        dist_penalty = (0.16 - 0.06 * progress) * (1.0 - 0.30 * scarcity)
        if self.search_mode == "convergence":
            dist_penalty *= 1.15
        elif self.search_mode == "diversity":
            dist_penalty *= 0.80

        dist = dist_to_niche[candidates]
        dist = self._normalize01(dist)
        F_sel = F[candidates]
        f_min = np.min(F_sel, axis=0)
        f_max = np.max(F_sel, axis=0)
        f_span = np.maximum(f_max - f_min, 1e-12)
        F_norm = (F_sel - f_min) / f_span
        # [FIX] Use Euclidean norm instead of Sum for local niching convergence
        conv = 1.0 - self._normalize01(np.linalg.norm(F_norm, axis=1))

        # [NOVO] PBI-score (θ-dominância) — substitui distância angular pura.
        # Para cada candidato, calcula PBI em relação ao vetor de referência
        # do seu nicho usando θ corrente (adaptativo ao progresso).
        # Scores menores = melhor qualidade no subproblema. Normaliza para [0,1].
        # Ref: Yuan et al. 2015 (θ-NSGA-III); caps-NSGA-III (Symmetry 2024).
        cand_niches = niche_of[candidates]
        ref = self.ref_dirs_np[cand_niches]
        ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        d1 = np.sum(F_norm * ref, axis=1)
        diff = F_norm - d1[:, None] * ref
        d2 = np.linalg.norm(diff, axis=1)
        pbi_scores = d1 + float(self._theta_current) * d2
        pbi_quality = 1.0 - self._normalize01(pbi_scores)  # maior = melhor

        f_unit = F_norm / np.maximum(np.linalg.norm(F_norm, axis=1, keepdims=True), 1e-12)
        ref = self.ref_dirs_np[cand_niches]
        ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        cosv = np.sum(f_unit * ref, axis=1)
        cosv = np.clip(cosv, -1.0, 1.0)
        angle = np.arccos(cosv) / np.pi
        angle_score = 1.0 - self._normalize01(angle)
        angle_w = float(np.clip(0.10 + self.angle_adapt_gain * (0.60 + 0.40 * scarcity), 0.05, 0.45))

        available = np.ones(len(candidates), dtype=bool)
        chosen_local = []

        while len(chosen_local) < n_pick and np.any(available):
            avail_idx = np.where(available)[0]
            avail_niches = niche_of[candidates[avail_idx]]
            occupancy = niche_count[avail_niches]
            min_occ = np.min(occupancy)

            sparse = avail_idx[occupancy == min_occ]
            sparse_niches = niche_of[candidates[sparse]]
            sparse_rarity = 1.0 / (1.0 + niche_count[sparse_niches])

            # [ADJUST] Integrar PBI-score domina o score de niching.
            # O peso 'pbi_w' agora é alto (>0.85) para garantir que em DTLZ2/4 (côncavas)
            # a seleção não favoreça cantos (viés de sum(f)) mas sim o vetor de referência.
            if self.problem.n_obj <= 2:
                pbi_w = float(np.clip(0.30 + 0.20 * progress + 0.10 * scarcity, 0.25, 0.60))
            elif self.problem.n_obj == 3:
                pbi_w = float(np.clip(0.55 + 0.25 * progress, 0.45, 0.85))
            else:
                pbi_w = 0.92 + 0.08 * progress
                pbi_w = float(np.clip(pbi_w, 0.85, 1.0))
            
            # Usar PBI quase puro para convergência local
            conv_component = (1.0 - pbi_w) * conv[sparse] + pbi_w * pbi_quality[sparse]

            score = (
                rarity_w * sparse_rarity
                + conv_w * conv_component
                + angle_w * angle_score[sparse]
                - dist_penalty * dist[sparse]
            )
            winner_local = sparse[int(np.argmax(score))]

            chosen_local.append(winner_local)
            niche_count[niche_of[candidates[winner_local]]] += 1
            available[winner_local] = False

        return candidates[np.asarray(chosen_local, dtype=int)]

    def _elite_refinement(
        self,
        survivors: np.ndarray,
        pop: Population,
        niche_of: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        survivors = np.asarray(_to_numpy(survivors), dtype=int)
        niche_of = np.asarray(_to_numpy(niche_of), dtype=int)
        F = np.asarray(_to_numpy(F), dtype=float)

        if self.elite_keep_frac <= 0.0 or len(survivors) == 0:
            return survivors

        # Keep refinement lightweight unless diversity/convergence signal is weak.
        if self.last_archive_improved and self._search_progress() < 0.70:
            return survivors

        all_idx = np.arange(len(pop), dtype=int)
        rem = np.setdiff1d(all_idx, survivors, assume_unique=False)
        if len(rem) == 0:
            return survivors

        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_span = np.maximum(f_max - f_min, 1e-12)
        conv = np.sum((F - f_min) / f_span, axis=1)  # lower is better

        chosen = survivors.tolist()
        chosen_set = set(chosen)
        scarcity = max(0.0, self.coverage_target - self.last_coverage)

        # Sparse-region enhancement: fill empty niches when possible.
        niche_count = np.bincount(niche_of[np.asarray(chosen, dtype=int)], minlength=len(self.ref_dirs))
        missing_niches = np.where(niche_count == 0)[0]
        for niche_id in missing_niches:
            cand = rem[niche_of[rem] == niche_id]
            if len(cand) == 0:
                continue
            best = int(cand[np.argmin(conv[cand])])
            worst_pos = int(np.argmax(conv[np.asarray(chosen, dtype=int)]))
            worst = int(chosen[worst_pos])
            accept = conv[best] < conv[worst]
            if (not accept) and scarcity > 0.0:
                # During diversity recovery, allow a small convergence tradeoff
                # to restore missing niches and prevent late collapse.
                conv_relax = 1.0 + 0.18 * scarcity
                accept = conv[best] <= conv[worst] * conv_relax
            if best not in chosen_set and accept:
                chosen_set.remove(worst)
                chosen_set.add(best)
                chosen[worst_pos] = best
                niche_count[niche_of[worst]] = max(0, niche_count[niche_of[worst]] - 1)
                niche_count[niche_id] += 1

        # Convergence-only elite retention (recent 2024 RVEA-2DCES-like idea).
        if self.search_mode == "diversity":
            return np.asarray(chosen, dtype=int)
        weak_diversity = (
            self.last_coverage < self.coverage_target
            or self.last_entropy < self.entropy_diversity_threshold
            or self.last_niche_cv > self.niche_cv_diversity_threshold
        )
        if weak_diversity:
            return np.asarray(chosen, dtype=int)
        rem2 = np.asarray([i for i in rem if i not in chosen_set], dtype=int)
        if len(rem2) == 0:
            return np.asarray(chosen, dtype=int)

        progress = self._search_progress()
        elite_frac = self.elite_keep_frac * (0.35 + 0.65 * progress) * (1.0 - 0.60 * scarcity)
        n_elite = max(1, int(round(elite_frac * len(chosen))))
        elite = rem2[np.argsort(conv[rem2])[:n_elite]]
        worst_order = np.asarray(chosen, dtype=int)[np.argsort(conv[np.asarray(chosen, dtype=int)])[::-1]]
        for e, w in zip(elite, worst_order):
            if conv[e] < conv[w] and e not in chosen_set:
                pos = chosen.index(int(w))
                chosen_set.remove(int(w))
                chosen_set.add(int(e))
                chosen[pos] = int(e)

        return np.asarray(chosen, dtype=int)

    # ------------------------------------------------------------------
    # [NOVO] Two-archive: arquivo de convergência por ASF por nicho
    # ------------------------------------------------------------------
    def _build_conv_archive(self, pop: Population) -> Population:
        """
        Arquivo de convergência: mantém conv_archive_frac * pop_size indivíduos
        com menor ASF (Achievement Scalarizing Function) de Tchebycheff em
        relação ao vetor de referência mais próximo.
        Complementa o nd_archive com pressão explícita de convergência.
        Ref: Two-archive (Wang et al. 2015); MaOEA-MS (Liu et al. 2022).
        """
        progress = self._search_progress()
        scarcity = max(0.0, self.coverage_target - self.last_coverage)
        mode_scale = 0.55 if self.search_mode == "diversity" else (0.75 if self.search_mode == "balanced" else 1.0)
        keep_scale = (0.55 + 0.45 * progress) * (1.0 - 0.45 * scarcity) * mode_scale
        keep_scale = float(np.clip(keep_scale, 0.30, 1.00))
        n_keep = max(1, int(round(self.conv_archive_frac * self.pop_size * keep_scale)))
        F = np.asarray(_to_numpy(pop.get("F", to_numpy=True)), dtype=float)
        n = len(F)
        if n == 0:
            return pop

        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        span = np.maximum(f_max - f_min, 1e-12)
        F_norm = (F - f_min) / span

        niche, _ = self._associate_niches(F)
        niche = np.asarray(_to_numpy(niche), dtype=int)
        ref_dirs_np = self.ref_dirs_np
        # Para cada nicho, manter o indivíduo com menor ASF de Tchebycheff
        best_per_niche: dict = {}
        for i in range(n):
            ni = int(niche[i])
            w = np.maximum(ref_dirs_np[ni], 1e-8)
            asf = float(np.max(F_norm[i] / w))
            if ni not in best_per_niche or asf < best_per_niche[ni][0]:
                best_per_niche[ni] = (asf, i)

        chosen = [v[1] for v in best_per_niche.values()]
        if self.search_mode == "diversity" and scarcity > 0.0:
            # During diversity recovery, avoid over-expanding convergence archive.
            n_keep = min(n_keep, max(1, len(chosen)))
        if len(chosen) < n_keep:
            # Preencher com melhores ASF globais
            asf_global = np.array([
                float(np.max(F_norm[i] / np.maximum(ref_dirs_np[int(niche[i])], 1e-8)))
                for i in range(n)
            ])
            order = np.argsort(asf_global)
            chosen_set = set(chosen)
            for idx in order:
                if len(chosen) >= n_keep:
                    break
                if int(idx) not in chosen_set:
                    chosen.append(int(idx))
                    chosen_set.add(int(idx))
        elif len(chosen) > n_keep:
            # [P6] Poda niche-diversa: mantem os n_keep nichos cujos
            # representantes tem menor ASF (1 por nicho), garantindo
            # que conv_archive cobre a frente inteira, nao so um canto.
            asf_vals_map = {idx: asf for asf, idx in best_per_niche.values()}
            chosen_asf = [(idx, float(asf_vals_map[idx]))
                          for idx in chosen]
            chosen_asf.sort(key=lambda x: x[1])
            chosen = [idx for idx, _ in chosen_asf[:n_keep]]

        return pop[np.asarray(chosen, dtype=int)]

    def _associate_niches(self, F: np.ndarray, nd: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Associate each solution to its closest reference direction.

        Uses _IdealNadirTracker (EMA nadir, running ideal) for normalization
        and _associate_to_niches_angular (angle-based) for association.
        Both replace the legacy NSGA-III hyperplane machinery entirely.
        """
        if self.use_gpu and cp is not None and isinstance(F, cp.ndarray):
            F_dev = F
            F_np = None
        else:
            F_np = np.asarray(_to_numpy(F), dtype=float)
            F_dev = _to_device(F_np, self.use_gpu)

        if nd is None:
            nd_src = F_np if F_np is not None else F_dev
            nd = self._nds_do(nd_src, only_non_dominated_front=True)
        nd = np.asarray(_to_numpy(nd), dtype=int)
        _prog = self._search_progress() if hasattr(
            self, "_search_progress") else 0.0
        self.hyp_norm.update(F_dev, nds=nd, progress=_prog)

        ideal = self.hyp_norm.ideal_point.copy()
        nadir = self.hyp_norm.nadir_point.copy()

        # Safety guards
        if not self.xp.all(self.xp.isfinite(ideal)):
            ideal = self.xp.min(F_dev, axis=0)
        if not self.xp.all(self.xp.isfinite(nadir)):
            nadir = self.xp.max(F_dev, axis=0)
        bad = (nadir - ideal) <= 1e-12
        if self.xp.any(bad):
            nadir[bad] = ideal[bad] + 1.0

        niche, d_perp = _associate_to_niches_angular(F_dev, self.ref_dirs_xp, ideal, nadir)
        return niche.astype(int), d_perp.astype(float)

    def _search_progress(self) -> float:
        n_eval = float(getattr(self.evaluator, "n_eval", 0))
        max_eval = self._get_max_evals()
        if max_eval is not None and max_eval > 0:
            return float(np.clip(n_eval / max_eval, 0.0, 1.0))
        
        n_gen = float(max(getattr(self, "n_gen", 1) - 1, 0))
        max_gen = self._get_max_gens()
        if max_gen is not None and max_gen > 0:
            return float(np.clip(n_gen / float(max_gen), 0.0, 1.0))
            
        return float(np.clip(n_gen / 120.0, 0.0, 1.0))

    def _get_max_gens(self) -> Optional[float]:
        term = getattr(self, "termination", None)
        if term is None:
            return None

        direct = getattr(term, "n_max_gen", None)
        if direct is not None:
            try:
                return float(direct)
            except (TypeError, ValueError):
                return None

        criteria = getattr(term, "criteria", None)
        if criteria is not None:
            items = criteria if isinstance(criteria, (list, tuple)) else [criteria]
            for item in items:
                val = getattr(item, "n_max_gen", None)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
        return None

    def _get_max_evals(self) -> Optional[float]:
        term = getattr(self, "termination", None)
        if term is None:
            return None

        direct = getattr(term, "n_max_evals", None)
        if direct is not None:
            try:
                return float(direct)
            except (TypeError, ValueError):
                return None

        criteria = getattr(term, "criteria", None)
        if criteria is not None:
            items = criteria if isinstance(criteria, (list, tuple)) else [criteria]
            for item in items:
                val = getattr(item, "n_max_evals", None)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
        return None

    def _phase_target_ratio(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        r0, r1, r2 = self.phase_ratio_targets
        if p < 0.33:
            t = p / 0.33
            return (1.0 - t) * r0 + t * r1
        if p < 0.75:
            t = (p - 0.33) / 0.42
            return (1.0 - t) * r1 + t * r2
        return r2

    def _update_archive_improvement(self) -> bool:
        if self.nd_archive is None or len(self.nd_archive) == 0:
            self.prev_archive_ideal = None
            self.prev_archive_size = 0
            self.prev_archive_score = None
            return True

        F = _to_device(self.nd_archive.get("F", to_numpy=True), self.use_gpu)
        curr_ideal = self.xp.min(F, axis=0)
        curr_size = len(F)
        score = float(self.xp.mean(self.xp.sum(F, axis=1)))

        improved_ideal = False
        improved_size = curr_size > self.prev_archive_size

        if self.prev_archive_ideal is not None:
            prev_ideal = self.xp.asarray(_to_device(self.prev_archive_ideal, self.use_gpu), dtype=float)
            scale = self.xp.maximum(self.xp.abs(prev_ideal), 1.0)
            improved_ideal = bool(
                self.xp.any((prev_ideal - curr_ideal) > (self.stagnation_tol * scale))
            )

        self.prev_archive_ideal = _to_numpy(curr_ideal)
        self.prev_archive_size = curr_size

        prev = self.prev_archive_score
        self.prev_archive_score = score
        if prev is None:
            return True

        scale = max(abs(prev), 1.0)
        improved_score = (prev - score) > (self.stagnation_tol * scale)
        improved = improved_ideal or improved_size or improved_score
        return bool(improved)

    def _adapt_epsilon(self, progress: float):
        scarcity = max(0.0, self.coverage_target - self.last_coverage)
        fail_signal = max(0.0, self.target_success - self.last_rate_sde)
        success_signal = max(0.0, self.last_rate_sde - self.target_success)
        sbx_gap = max(0.0, self.last_rate_sbx - self.last_rate_sde)
        stagnation = float(min(2.0, self.stagnation_streak))
        xp = self.xp

        csa_term = 0.0
        if self.last_total_sde > 0 and self.last_sde_z_mean is not None and self.eps_path is not None:
            eps_path = xp.asarray(self.eps_path, dtype=float)
            z_mean = xp.asarray(_to_device(self.last_sde_z_mean, self.use_gpu), dtype=float)
            self.eps_path = (
                (1.0 - self.c_sigma) * eps_path
                + xp.sqrt(self.c_sigma * (2.0 - self.c_sigma)) * z_mean
            )
            p_norm = float(xp.linalg.norm(self.eps_path))
            csa_term = p_norm / max(float(self.eps_path_target), 1e-12) - 1.0
        elif self.eps_path is not None:
            self.eps_path = (1.0 - 0.5 * self.c_sigma) * xp.asarray(self.eps_path, dtype=float)

        signal = (
            0.85 * scarcity
            + 0.45 * fail_signal
            + 0.20 * stagnation
            - 0.30 * success_signal
            - 0.35 * sbx_gap
            + self.epsilon_csa_gain * csa_term
        )
        mult = float(self.xp.exp(self.epsilon_adapt_lr * signal))
        eps_new = self.epsilon_state * mult

        base_schedule = self.epsilon_base * (1.10 - 0.55 * np.clip(progress, 0.0, 1.0))
        eps_new = 0.72 * eps_new + 0.28 * base_schedule
        eps_min_dyn = max(1e-14, self.epsilon_base * (0.12 + 0.10 * (1.0 - progress)))
        eps_max_dyn = max(
            eps_min_dyn * 1.05,
            self.epsilon_base * ((2.8 - 1.2 * progress) * (1.0 + 0.5 * scarcity)),
        )
        self.epsilon_state = float(np.clip(eps_new, eps_min_dyn, eps_max_dyn))

    def _record_telemetry(self, progress: float):
        sigma_mean = float(np.mean(self.pop.get("sigma_i"))) if len(self.pop) > 0 else 0.0
        row = {
            "gen": int(getattr(self, "n_gen", 0)),
            "n_eval": int(getattr(self.evaluator, "n_eval", 0)),
            "progress": float(progress),
            "coverage": float(self.last_coverage),
            "ratio_sde": float(self.ratio_sde),
            "epsilon_state": float(self.epsilon_state),
            "success_sde": float(self.last_rate_sde),
            "success_sbx": float(self.last_rate_sbx),
            "archive_improved": bool(self.last_archive_improved),
            "stagnation_streak": int(self.stagnation_streak),
            "search_mode": str(self.search_mode),
            "entropy": float(self.last_entropy),
            "niche_cv": float(self.last_niche_cv),
            "sigma_mean": sigma_mean,
            "sde_step_norm": float(self.last_sde_step_norm),
            "c_path": float(self.c_path),
            "c_cov": float(self.c_cov),
            "c_sigma": float(self.c_sigma),
            "sbx_eta": float(self.sbx.eta),
            "q_state": int(self.last_q_state),
            "rvx_mu_f": float(self.rvx_mu_f),
            "rvx_mu_cr": float(self.rvx_mu_cr),
            # [NOVO] campos adicionais da revisão
            "theta_pbi_current": float(self._theta_current),
            "conv_archive_size": int(len(self.conv_archive)) if self.conv_archive is not None else 0,
            "low_obj_mode": bool(self.low_obj_mode),
            "prob_neighbor_mating": float(self.prob_neighbor_mating),
            "archive_injection_rate": float(self.archive_injection_rate),
        }
        self.telemetry.append(row)
        if len(self.telemetry) > self.telemetry_limit:
            self.telemetry = self.telemetry[-self.telemetry_limit :]

    def get_telemetry(self):
        """Return a copy of generation telemetry for tuning/mapping."""
        return list(self.telemetry)

    def _population_metrics(self, F: np.ndarray, niche: Optional[np.ndarray] = None) -> Tuple[float, float]:
        F = _to_device(F, self.use_gpu)
        f_min = self.xp.min(F, axis=0)
        f_max = self.xp.max(F, axis=0)
        f_span = self.xp.maximum(f_max - f_min, 1e-12)
        F_norm = (F - f_min) / f_span

        # [ADJUST] Use Euclidean norm instead of Sum for convergence metric.
        # Sum(f) biases corner solutions on concave fronts (DTLZ2).
        # Euclidean distance to ideal point (0,0,...) is more geometry-neutral.
        conv = float(self.xp.mean(self.xp.linalg.norm(F_norm, axis=1)))

        if niche is None:
            niche, _ = self._associate_niches(F)
        else:
            niche = self.xp.asarray(niche, dtype=int)
        niche_count = self.xp.bincount(niche, minlength=len(self.ref_dirs))
        coverage = float(self.xp.count_nonzero(niche_count > 0) / max(len(self.ref_dirs), 1))
        # Sparse-region consistency proxy: lower std(count) with high coverage is better.
        if len(niche_count) > 0:
            dens_std = float(self.xp.std(niche_count / max(float(self.xp.sum(niche_count)), 1.0)))
        else:
            dens_std = 0.0
        div = float(coverage - 0.25 * dens_std)
        return conv, div

    def _policy_state(self, conv_metric: float, div_metric: float) -> int:
        if self.last_conv_metric is None or self.last_div_metric is None:
            return 1
        conv_delta = self.last_conv_metric - conv_metric
        div_delta = div_metric - self.last_div_metric
        if conv_delta > 5e-4 and div_delta >= -2e-4:
            return 2
        if conv_delta < -5e-4 or div_delta < -5e-4:
            return 0
        return 1

    def _policy_reward(self, conv_metric: float, div_metric: float) -> float:
        if self.last_conv_metric is None or self.last_div_metric is None:
            return 0.0
        conv_gain = self.last_conv_metric - conv_metric
        div_gain = div_metric - self.last_div_metric
        improve_bonus = 0.15 if self.last_archive_improved else -0.10
        return float(1.2 * conv_gain + 0.8 * div_gain + improve_bonus)

    def _select_policy_action(self, state: int) -> int:
        if self.random_state.random() < self.qlearn_eps:
            return int(self.random_state.integers(0, 3))
        return int(self.xp.argmax(self.q_table[state]))

    def _update_q_policy(self, reward: float, next_state: int):
        s = int(self.last_q_state)
        a = int(self.sbx_mode)
        td_target = reward + self.qlearn_gamma * float(self.xp.max(self.q_table[next_state]))
        self.q_table[s, a] = (1.0 - self.qlearn_alpha) * self.q_table[s, a] + self.qlearn_alpha * td_target

    def _apply_policy_action(self, action: int):
        mode = int(np.clip(action, 0, len(self.sbx_eta_candidates) - 1))
        if self.search_mode == "diversity":
            mode = min(mode, 1)
        elif self.search_mode == "convergence":
            mode = max(mode, 1)
        if self.last_archive_improved and self.last_q_state == 2:
            mode = 1
        self.sbx_mode = mode
        eta = float(self.sbx_eta_candidates[self.sbx_mode])
        self.sbx.eta = eta

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        """Scale values to [0,1]. Supports GPU."""
        xp = _get_xp(x)
        x = xp.asarray(x, dtype=float)
        lo = xp.min(x)
        hi = xp.max(x)
        if hi - lo <= 1e-14:
            return xp.zeros_like(x)
        return (x - lo) / (hi - lo)

    @staticmethod
    def _niche_diversity_metrics(niche_count: np.ndarray) -> Tuple[float, float]:
        """Diversity metrics (entropy, CV). Supports GPU."""
        xp = _get_xp(niche_count)
        counts = xp.asarray(niche_count, dtype=float)
        positive = counts[counts > 0]
        if len(positive) == 0:
            return 0.0, 1.0
        prob = positive / xp.sum(positive)
        ent = -xp.sum(prob * xp.log(xp.maximum(prob, 1e-16)))
        ent_norm = float(ent / xp.log(len(positive))) if len(positive) > 1 else 0.0
        cv = float(xp.std(positive) / max(xp.mean(positive), 1e-12))
        return ent_norm, cv

    @staticmethod
    def _stabilize_cov(C: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix C is symmetric and positive-definite. Supports GPU."""
        xp = _get_xp(C)
        C = 0.5 * (C + C.T)
        n = C.shape[0]
        diag = xp.diag(C).copy()
        diag[diag < 1e-12] = 1e-12
        xp.fill_diagonal(C, diag)

        # Cheap PSD safeguard using Gershgorin lower bound.
        row_abs = xp.sum(xp.abs(C), axis=1) - xp.abs(diag)
        lower_bound = float(xp.min(diag - row_abs))
        if lower_bound < 1e-12:
            C = C + xp.eye(n, dtype=float) * (1e-12 - lower_bound)
        return C

    @staticmethod
    def _cholesky_with_jitter(C: np.ndarray, base_jitter: float = 1e-12, tries: int = 4):
        """Perform Cholesky with increasing jitter. Supports GPU."""
        xp = _get_xp(C)
        n = C.shape[0]
        I = xp.eye(n, dtype=float)
        M = C
        for k in range(max(1, tries)):
            try:
                return M, xp.linalg.cholesky(M)
            except Exception:
                M = 0.5 * (M + M.T) + I * (base_jitter * (10.0 ** k))
        return I, I

    def _update_archive(self, pool: Population) -> Population:
        if pool.has("rank"):
            rank = np.asarray(_to_numpy(pool.get("rank")), dtype=int)
            nd = np.where(rank == 0)[0]
        else:
            nd = self._nds_do(pool.get("F"), only_non_dominated_front=True)
            nd = np.asarray(_to_numpy(nd), dtype=int)
        return pool[np.asarray(nd, dtype=int)]

    def _set_optimum(self):
        self.opt = self.nd_archive if self.nd_archive is not None else self.pop

    def _configure_cma_hyperparams(self, n_var: int, pop_size: int):
        n = float(max(n_var, 2))
        lam = int(max(pop_size, 4))
        mu = max(2, lam // 2)
        idx = np.arange(1, mu + 1, dtype=float)
        w = np.log(mu + 0.5) - np.log(_to_numpy(idx))
        w = np.maximum(w, 0.0)
        if np.sum(w) <= 0.0:
            w = self.xp.full(mu, 1.0 / mu, dtype=float)
        else:
            w = w / np.sum(w)
        mu_eff = float(1.0 / np.sum(w * w))

        c_path = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c1 = 2.0 / (((n + 1.3) ** 2.0) + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (((n + 2.0) ** 2.0) + mu_eff))
        c_cov = c1 + cmu
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma

        self.c_path = float(np.clip(c_path, 0.08, 0.55))
        self.c_sigma = float(np.clip(c_sigma, 0.05, 0.40))
        self.c_cov = float(np.clip(c_cov, 0.03, 0.35))
        self.sigma_damp = float(np.clip(d_sigma, 1.0, 6.0))

    def _auto_calibrate_runtime_controls(self, n_var: int, n_obj: int):
        """Automatic per-run calibration of control parameters for MaOP regimes."""
        dim = float(max(n_var, 2))
        m = float(max(n_obj, 2))
        many_level = max(0.0, m - 3.0)
        dim_log = float(np.log1p(dim))
        pop_term = float(np.log1p(max(self.pop_size, 4)) - np.log(20.0))

        self.low_obj_mode = bool(n_obj <= 3)

        if self.low_obj_mode:
            # Dedicated low-objective regime (m <= 3): prioritize global evolution
            # and keep local Jacobian branch as a sparse helper.
            low = 0.0
            high = float(np.clip(0.06 + 0.01 * pop_term, 0.03, 0.10))
            if high <= low + 1e-3:
                high = low + 1e-3
            self.ratio_bounds = (low, high)

            ratio0 = min(self._user_ratio_sde, 0.02 + 0.01 * float(dim > 20))
            self.ratio_sde = float(np.clip(ratio0, self.ratio_bounds[0], self.ratio_bounds[1]))

            t0 = self.ratio_bounds[0]
            t1 = min(self.ratio_bounds[1], max(t0, 0.01))
            t2 = min(self.ratio_bounds[1], max(t1, 0.03))
            self.phase_ratio_targets = (t0, t1, t2)

            cov = max(self._user_coverage_target, 0.90)
            if self.pop_size <= 120:
                cov += 0.02
            self.coverage_target = float(np.clip(cov, 0.88, 0.98))
            self.sparse_quantile = float(np.clip(0.85 * self._user_sparse_quantile, 0.15, 0.35))

            self.target_success = float(np.clip(self._user_target_success - 0.03, 0.10, 0.24))
            self.ucb_exploration = float(np.clip(0.65 * self._user_ucb_exploration, 0.01, 0.12))
            self.qlearn_eps = float(np.clip(0.35 * self._user_qlearn_eps, 0.0, 0.08))
            self.mirror_rate = float(np.clip(self._user_mirror_rate + 0.12, 0.55, 0.95))

            self.rvx_share_base = float(np.clip(0.45 * self._user_rvx_share_base, 0.05, 0.30))
            self.drift_scale = float(np.clip(0.60 * self._user_drift_scale, 0.05, 0.16))
            self.de_fallback_scale = float(np.clip(0.65 * self._user_de_fallback_scale, 0.03, 0.20))

            self.epsilon_adapt_lr = float(np.clip(0.60 * self._user_epsilon_adapt_lr, 0.02, 0.18))
            self.epsilon_csa_gain = float(np.clip(0.55 * self._user_epsilon_csa_gain, 0.0, 0.35))

            self.entropy_diversity_threshold = float(np.clip(0.78 + 0.02 * float(self.pop_size < 100), 0.72, 0.86))
            self.niche_cv_diversity_threshold = float(np.clip(1.05 + 0.08 * float(self.pop_size < 100), 0.95, 1.30))

            self.use_sdr = False
            self.use_two_archive = bool(self._user_use_two_archive and n_obj >= 3)
            if n_obj <= 2:
                self.use_two_archive = False

            self.prob_neighbor_mating = float(np.clip(0.45 * self._user_prob_neighbor_mating, 0.15, 0.60))
            self.archive_injection_rate = float(np.clip(0.50 * self._user_archive_injection_rate, 0.0, 0.12))

            if self.theta_adapt:
                theta_auto = 1.8 if n_obj <= 2 else 2.4
                self.theta_pbi = float(np.clip(theta_auto, 1.0, 4.0))
            else:
                self.theta_pbi = float(np.clip(self._user_theta_pbi, 1.0, 4.0))
            self._theta_current = float(self.theta_pbi)
            return

        self.low_obj_mode = False

        # Local-search bounds: reduce aggressiveness as objective count grows.
        low = float(np.clip(self._user_ratio_bounds[0] + 0.002 * many_level, 0.0, 0.20))
        high = float(
            np.clip(
                self._user_ratio_bounds[1] - 0.015 * many_level + 0.008 * pop_term,
                0.12,
                0.45,
            )
        )
        if high <= low + 1e-3:
            high = low + 1e-3
        self.ratio_bounds = (low, high)

        ratio0 = self._user_ratio_sde * (1.0 - 0.08 * many_level) + 0.01 * pop_term
        self.ratio_sde = float(np.clip(ratio0, self.ratio_bounds[0], self.ratio_bounds[1]))

        # Phase targets consistent with calibrated bounds.
        r0, r1, r2 = self._user_phase_ratio_targets
        shrink = float(np.clip(1.0 - 0.06 * many_level, 0.70, 1.00))
        boost = float(np.clip(1.0 + 0.05 * many_level, 1.00, 1.25))
        t0 = float(np.clip(r0 * shrink, self.ratio_bounds[0], self.ratio_bounds[1]))
        t1 = float(np.clip(r1 * shrink, self.ratio_bounds[0], self.ratio_bounds[1]))
        t2 = float(np.clip(r2 * boost, self.ratio_bounds[0], self.ratio_bounds[1]))
        t1 = max(t0, t1)
        t2 = max(t1, t2)
        self.phase_ratio_targets = (t0, t1, t2)

        # Diversity pressure in many-objective settings.
        cov = self._user_coverage_target + 0.03 * min(many_level, 4.0)
        if self.pop_size <= 120:
            cov -= 0.05
        self.coverage_target = float(np.clip(cov, 0.50, 0.78))
        self.sparse_quantile = float(np.clip(self._user_sparse_quantile + 0.02 * min(many_level, 4.0), 0.20, 0.45))

        # Operator and exploration calibration.
        self.target_success = float(
            np.clip(
                self._user_target_success - 0.01 * many_level + 0.006 * (dim_log - 2.0),
                0.14,
                0.28,
            )
        )
        self.ucb_exploration = float(np.clip(self._user_ucb_exploration + 0.01 * many_level, 0.04, 0.20))
        self.qlearn_eps = float(np.clip(self._user_qlearn_eps + 0.02 * float(many_level > 2.0), 0.01, 0.25))
        self.mirror_rate = float(np.clip(self._user_mirror_rate - 0.08 * many_level, 0.45, 0.90))

        # Drift/global branch balance.
        self.rvx_share_base = float(
            np.clip(
                self._user_rvx_share_base + 0.03 * many_level - 0.05 * float(self.pop_size < 80),
                0.20,
                0.60,
            )
        )
        self.drift_scale = float(
            np.clip(
                self._user_drift_scale - 0.02 * many_level + 0.01 * np.tanh((dim - 20.0) / 30.0),
                0.10,
                0.26,
            )
        )
        self.de_fallback_scale = float(np.clip(self._user_de_fallback_scale + 0.02 * many_level, 0.06, 0.30))

        # Epsilon controller calibration.
        self.epsilon_adapt_lr = float(
            np.clip(self._user_epsilon_adapt_lr * (0.90 + 0.04 * many_level), 0.05, 0.35)
        )
        self.epsilon_csa_gain = float(np.clip(self._user_epsilon_csa_gain + 0.04 * many_level, 0.0, 0.60))

        # Mode thresholds.
        self.entropy_diversity_threshold = float(
            np.clip(0.84 + 0.02 * many_level - 0.01 * pop_term, 0.78, 0.90)
        )
        self.niche_cv_diversity_threshold = float(
            np.clip(1.35 - 0.05 * many_level + 0.04 * float(self.pop_size < 100), 1.05, 1.45)
        )

        self.use_sdr = bool(self._user_use_sdr)
        self.use_two_archive = bool(self._user_use_two_archive)
        self.prob_neighbor_mating = float(np.clip(self._user_prob_neighbor_mating, 0.55, 0.90))
        self.archive_injection_rate = float(
            np.clip(0.35 * self._user_archive_injection_rate + 0.005 * many_level, 0.02, 0.08)
        )

        # [Dr. Evolve] Auto-tune theta_pbi for dimensionality.
        # Logic: High-dimensional spaces need stronger angular pressure (d2) to
        # enforce niching, otherwise solutions drift between rays.
        # Base 5.0 + 1.2 per extra objective > 3.
        if self.theta_adapt:
            theta_auto = 5.0 + 1.2 * many_level
            self.theta_pbi = float(np.clip(theta_auto, 5.0, 20.0))
            self._theta_current = float(self.theta_pbi)
