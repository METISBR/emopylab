import math
from typing import Any

import numpy as np

from metrics.indicators import HV as _NativeHV

try:
    from util.nds.non_dominated_sorting import NonDominatedSorting

    _NDS = NonDominatedSorting()
except Exception:  # noqa: BLE001
    _NDS = None


# References copied from the original MATLAB metric implementations when available.
METRIC_REFERENCES: dict[str, str] = {
    "CPF": (
        "Y. Tian, R. Cheng, X. Zhang, M. Li, and Y. Jin. Diversity assessment of "
        "multi-objective evolutionary algorithms: Performance metric and benchmark "
        "problems. IEEE Computational Intelligence Magazine, 2019, 14(3): 61-74."
    ),
    "DeltaP": (
        "O. Schutze, X. Esquivel, A. Lara, and C. A. Coello Coello. Using the averaged "
        "Hausdorff distance as a performance measure in evolutionary multiobjective "
        "optimization. IEEE Transactions on Evolutionary Computation, 2012, 16(4): 504-522."
    ),
    "DM": (
        "K. Deb and S. Jain. Running performance metrics for evolutionary "
        "multi-objective optimization. KanGAL Report 2002004, 2002."
    ),
    "GD": (
        "D. A. Van Veldhuizen. Multiobjective evolutionary algorithms: Classifications, "
        "analyses, and new innovations. Ph.D. thesis, 1999."
    ),
    "HV": (
        "E. Zitzler and L. Thiele. Multiobjective evolutionary algorithms: A comparative "
        "case study and the strength Pareto approach. IEEE TEC, 1999, 3(4): 257-271."
    ),
    "IGD": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
    "IGDp": (
        "H. Ishibuchi, H. Masuda, Y. Tanigaki, and Y. Nojima. Modified distance calculation "
        "in generational distance and inverted generational distance. EMO 2015, 110-125."
    ),
    "IGDX": (
        "A. Zhou, Q. Zhang, and Y. Jin. Approximating the set of Pareto-optimal solutions "
        "in both the decision and objective spaces by an estimation of distribution "
        "algorithm. IEEE TEC, 2009, 13(5): 1167-1189."
    ),
    "PD": (
        "H. Wang, Y. Jin, and X. Yao. Diversity assessment in many-objective optimization. "
        "IEEE Transactions on Cybernetics, 2017, 47(6): 1510-1522."
    ),
    "Spacing": (
        "J. R. Schott. Fault tolerant design using single and multicriteria genetic "
        "algorithm optimization. Master's thesis, MIT, 1995."
    ),
    "Spread": (
        "Y. Wang, L. Wu, and X. Yuan. Multi-objective self-adaptive differential evolution "
        "with elitist archive and crowding entropy-based diversity measure. Soft "
        "Computing, 2010, 14(3): 193-209."
    ),
    "Mean_HV": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
    "Mean_IGD": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
    "Worst_HV": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
    "Worst_IGD": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
    "Task1_HV": (
        "E. Zitzler and L. Thiele. Multiobjective evolutionary algorithms: A comparative "
        "case study and the strength Pareto approach. IEEE TEC, 1999, 3(4): 257-271."
    ),
    "Task1_IGD": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
    "Task2_HV": (
        "E. Zitzler and L. Thiele. Multiobjective evolutionary algorithms: A comparative "
        "case study and the strength Pareto approach. IEEE TEC, 1999, 3(4): 257-271."
    ),
    "Task2_IGD": (
        "C. A. Coello Coello and N. C. Cortes. Solving multiobjective optimization "
        "problems using an artificial immune system. Genetic Programming and Evolvable "
        "Machines, 2005, 6(2): 163-190."
    ),
}


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _pairwise_euclidean(a: Any, b: Any) -> np.ndarray:
    aa = _as_2d(a)
    bb = _as_2d(b)
    if aa.size == 0 or bb.size == 0:
        return np.empty((aa.shape[0], bb.shape[0]), dtype=float)
    try:
        from scipy.spatial.distance import cdist
        return np.asarray(cdist(aa, bb, metric='euclidean'), dtype=float)
    except ImportError:
        diff = aa[:, None, :] - bb[None, :, :]
        return np.linalg.norm(diff, axis=2)


def _pairwise_cityblock(a: Any, b: Any) -> np.ndarray:
    aa = _as_2d(a)
    bb = _as_2d(b)
    if aa.size == 0 or bb.size == 0:
        return np.empty((aa.shape[0], bb.shape[0]), dtype=float)
    diff = np.abs(aa[:, None, :] - bb[None, :, :])
    return np.sum(diff, axis=2)


def _pairwise_minkowski(a: Any, b: Any, p: float) -> np.ndarray:
    aa = _as_2d(a)
    bb = _as_2d(b)
    if aa.size == 0 or bb.size == 0:
        return np.empty((aa.shape[0], bb.shape[0]), dtype=float)
    diff = np.abs(aa[:, None, :] - bb[None, :, :])
    return np.sum(np.power(diff, p), axis=2) ** (1.0 / p)


def _igdp_p_from_context(context: dict[str, Any], default: float = 2.0) -> float:
    keys = ("igdp_p", "igdp_p_norm", "igdp_pnorm", "p_norm", "pnorm", "p")
    if not isinstance(context, dict):
        return float(default)
    for key in keys:
        if key not in context:
            continue
        raw = context.get(key)
        try:
            p = float(raw)
        except Exception:
            try:
                p = float(str(raw).strip())
            except Exception:
                continue
        if math.isfinite(p) and p > 0.0:
            return float(p)
    return float(default)


def _non_dominated_front(F: Any) -> np.ndarray:
    arr = _as_2d(F)
    if arr.size == 0:
        return arr.reshape(0, arr.shape[1] if arr.ndim == 2 else 0)
    if _NDS is None:
        return arr
    try:
        idx = _NDS.do(arr, only_non_dominated_front=True)
        return np.asarray(arr[idx], dtype=float)
    except Exception:  # noqa: BLE001
        return arr


def _get_front(front: Any) -> np.ndarray:
    arr = _as_2d(front)
    if arr.size == 0:
        return np.empty((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=float)
    return arr


def _get_current_F(context: dict[str, Any], front: Any) -> np.ndarray:
    values = context.get("current_population_F")
    if values is None:
        return _get_front(front)
    arr = _as_2d(values)
    if arr.size == 0:
        return np.empty((0, arr.shape[1]), dtype=float)
    return arr


def _get_current_X(context: dict[str, Any]) -> np.ndarray | None:
    values = context.get("current_population_X")
    if values is None:
        return None
    arr = _as_2d(values)
    if arr.size == 0:
        return np.empty((0, arr.shape[1]), dtype=float)
    return arr


def _squeeze_bool_column(values: Any, n_rows: int | None = None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if n_rows is not None and arr.shape[0] != n_rows:
        return None
    return np.asarray(np.all(arr.astype(bool), axis=1), dtype=bool)


def _infer_feasible_mask(context: dict[str, Any], n_rows: int) -> np.ndarray | None:
    feasible = _squeeze_bool_column(context.get("current_population_feasible"), n_rows)
    if feasible is not None:
        return feasible

    cv = context.get("current_population_CV")
    if cv is not None:
        arr = np.asarray(cv, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] == n_rows:
            return np.all(arr <= 0.0, axis=1)

    g = context.get("current_population_G")
    h = context.get("current_population_H")
    g_ok = None
    h_ok = None
    if g is not None:
        gg = np.asarray(g, dtype=float)
        if gg.ndim == 0:
            gg = gg.reshape(1, 1)
        elif gg.ndim == 1:
            gg = gg.reshape(-1, 1)
        if gg.shape[0] == n_rows:
            g_ok = np.all(gg <= 0.0, axis=1)
    if h is not None:
        hh = np.asarray(h, dtype=float)
        if hh.ndim == 0:
            hh = hh.reshape(1, 1)
        elif hh.ndim == 1:
            hh = hh.reshape(-1, 1)
        if hh.shape[0] == n_rows:
            h_ok = np.all(np.abs(hh) <= 1e-6, axis=1)

    if g_ok is None and h_ok is None:
        return None
    if g_ok is None:
        return h_ok
    if h_ok is None:
        return g_ok
    return np.asarray(g_ok & h_ok, dtype=bool)


def _resolve_problem_reference(
    context: dict[str, Any],
    *,
    kind: str,
) -> np.ndarray | None:
    problem = context.get("problem")
    if problem is None:
        return None

    ref_dirs = context.get("ref_dirs")
    n_obj = int(context.get("n_obj", 2) or 2)
    target = 200
    try:
        if ref_dirs is not None:
            ref_dirs_arr = np.asarray(ref_dirs, dtype=float)
            if ref_dirs_arr.ndim == 2 and ref_dirs_arr.size > 0:
                target = max(target, int(ref_dirs_arr.shape[0]))
    except Exception:  # noqa: BLE001
        ref_dirs = None

    method_candidates: list[str] = []
    if kind == "front":
        method_candidates = ["pareto_front", "_calc_pareto_front"]
    elif kind == "set":
        method_candidates = ["pareto_set", "_calc_pareto_set"]
    else:
        return None

    call_kwargs: list[dict[str, Any]] = [{}]
    if ref_dirs is not None:
        call_kwargs.append({"ref_dirs": ref_dirs})
    call_kwargs.append({"n_pareto_points": target})

    for method_name in method_candidates:
        fn = getattr(problem, method_name, None)
        if not callable(fn):
            continue
        for kwargs in call_kwargs:
            try:
                values = fn(**kwargs)
            except Exception:  # noqa: BLE001
                continue
            if values is None:
                continue
            try:
                arr = _as_2d(values)
            except Exception:  # noqa: BLE001
                continue
            if arr.size == 0:
                continue
            if arr.shape[1] != n_obj:
                # Some pareto set methods return decision vectors; only accept dimension match for the requested kind.
                if kind == "front":
                    continue
            return arr

    return None


def _get_reference_front(context: dict[str, Any]) -> np.ndarray | None:
    pf = context.get("pareto_front")
    if pf is None:
        pf = context.get("ref_pf")
    if pf is None:
        pf = _resolve_problem_reference(context, kind="front")
        if pf is not None:
            context["pareto_front"] = pf
            context["ref_pf"] = pf
    if pf is None:
        return None
    arr = _as_2d(pf)
    return arr if arr.size > 0 else None


def _get_reference_set(context: dict[str, Any]) -> np.ndarray | None:
    ps = context.get("pareto_set")
    if ps is None:
        ps = context.get("ref_ps")
    if ps is None:
        ps = _resolve_problem_reference(context, kind="set")
        if ps is not None:
            context["pareto_set"] = ps
            context["ref_ps"] = ps
    if ps is None:
        return None
    arr = _as_2d(ps)
    return arr if arr.size > 0 else None


def _safe_divisor(den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    return np.where(np.abs(den) <= 1e-12, 1.0, den)


def _community_hv(pop_obj: np.ndarray, optimum: np.ndarray, context: dict[str, Any]) -> float:
    if pop_obj.size == 0:
        return 0.0
    pop_obj = _as_2d(pop_obj)
    optimum = _as_2d(optimum)
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")

    n, m = pop_obj.shape
    fmin = np.minimum(np.min(pop_obj, axis=0), np.zeros(m, dtype=float))
    fmax = np.max(optimum, axis=0)
    den = _safe_divisor((fmax - fmin) * 1.1)
    norm_pop = (pop_obj - fmin) / den
    norm_pop = norm_pop[~np.any(norm_pop > 1.0, axis=1)]
    ref_point = np.ones(m, dtype=float)

    if norm_pop.size == 0:
        return 0.0

    norm_pop = _non_dominated_front(norm_pop)

    if m < 4:
        if _NativeHV is None:
            return float("nan")
        try:
            hv = _NativeHV(ref_point=ref_point)
            return float(hv(norm_pop))
        except Exception:  # noqa: BLE001
            return float("nan")

    # Fast Monte Carlo Hypervolume with Dynamic Sample Pruning (PlatEMO standard)
    try:
        from metrics.hv_fast_mc import HV_fast_MC
        return float(HV_fast_MC(pop_obj, optimum, sample_num=int(context.get("hv_mc_samples", 10_000))))
    except Exception:
        pass

    sample_num = max(1, int(context.get("hv_mc_samples", 10_000)))
    max_value = ref_point
    min_value = np.min(norm_pop, axis=0)
    if np.any(max_value < min_value):
        return 0.0

    chunk_size = max(1, int(context.get("hv_mc_chunk_size", min(sample_num, 100_000))))
    rng = np.random.default_rng(1)
    dominated_count = 0
    remaining = sample_num
    while remaining > 0:
        current_size = min(chunk_size, remaining)
        samples = rng.uniform(low=min_value, high=max_value, size=(current_size, m))
        dominated = np.any(np.all(norm_pop[:, None, :] <= samples[None, :, :], axis=2), axis=0)
        dominated_count += int(np.count_nonzero(dominated))
        remaining -= current_size
    return float(np.prod(max_value - min_value) * (dominated_count / sample_num))


def _subset_task_population(context: dict[str, Any], task_id: int) -> np.ndarray:
    X = _get_current_X(context)
    F = context.get("current_population_F")
    if X is None or F is None:
        return np.empty((0, 0), dtype=float)

    F_arr = _as_2d(F)
    X_arr = _as_2d(X)
    if F_arr.shape[0] != X_arr.shape[0] or F_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        return np.empty((0, F_arr.shape[1] if F_arr.ndim == 2 else 0), dtype=float)

    task_col = X_arr[:, -1]
    mask = np.isfinite(task_col) & np.isclose(task_col, float(task_id))
    if not np.any(mask):
        return np.empty((0, F_arr.shape[1]), dtype=float)
    return _non_dominated_front(F_arr[mask])


def _igd_value(pop_obj: np.ndarray, optimum: np.ndarray) -> float:
    if pop_obj.size == 0 or optimum.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    dist = _pairwise_euclidean(optimum, pop_obj)
    return float(np.mean(np.min(dist, axis=1)))


def _gd_value(pop_obj: np.ndarray, optimum: np.ndarray) -> float:
    if pop_obj.size == 0 or optimum.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    dist = np.min(_pairwise_euclidean(pop_obj, optimum), axis=1)
    return float(np.linalg.norm(dist) / max(1, dist.size))


def _deltap_value(pop_obj: np.ndarray, optimum: np.ndarray) -> float:
    if pop_obj.size == 0 or optimum.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    dist = _pairwise_euclidean(optimum, pop_obj)
    return float(max(np.mean(np.min(dist, axis=1)), np.mean(np.min(dist, axis=0))))


def _igdp_value(pop_obj: np.ndarray, optimum: np.ndarray, p: float = 2.0) -> float:
    if pop_obj.size == 0 or optimum.size == 0:
        return float("nan")
    pop_obj = _as_2d(pop_obj)
    optimum = _as_2d(optimum)
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    try:
        p = float(p)
    except Exception:
        p = 2.0
    if not math.isfinite(p) or p <= 0.0:
        p = 2.0
    diff = pop_obj[None, :, :] - optimum[:, None, :]
    diff = np.maximum(diff, 0.0)
    if abs(p - 2.0) <= 1e-12:
        delta = np.sqrt(np.sum(diff * diff, axis=2))
    elif abs(p - 1.0) <= 1e-12:
        delta = np.sum(diff, axis=2)
    else:
        delta = np.sum(np.power(diff, p), axis=2) ** (1.0 / p)
    return float(np.mean(np.min(delta, axis=1)))


def _robust_metric_samples(context: dict[str, Any]) -> list[np.ndarray]:
    problem = context.get("problem")
    X = _get_current_X(context)
    if problem is None or X is None or X.size == 0:
        return []
    perturb = getattr(problem, "Perturb", None)
    if not callable(perturb):
        return []

    try:
        perturbed = perturb(X)
    except Exception:  # noqa: BLE001
        return []

    n = X.shape[0]
    if n <= 0:
        return []

    # Case 1: perturb returns decision vectors -> evaluate them with problem.evaluate.
    if isinstance(perturbed, np.ndarray):
        arr = _as_2d(perturbed)
        if arr.shape[0] % n != 0:
            return []
        evaluate = getattr(problem, "evaluate", None)
        if not callable(evaluate):
            return []
        try:
            F = evaluate(arr, return_values_of=["F"])
            if isinstance(F, (tuple, list)):
                F = F[0]
            if isinstance(F, dict):
                F = F.get("F")
            F = _as_2d(F)
        except Exception:  # noqa: BLE001
            return []
        if F.shape[0] % n != 0:
            return []
        return [F[i : i + n] for i in range(0, F.shape[0], n)]

    # Case 2: perturb returns an object/list with objective data already evaluated.
    try:
        if hasattr(perturbed, "get"):
            F = perturbed.get("F")
            if F is not None:
                F = _as_2d(F)
                if F.shape[0] % n == 0:
                    return [F[i : i + n] for i in range(0, F.shape[0], n)]
    except Exception:  # noqa: BLE001
        pass

    return []


def _cpf_map(x: np.ndarray, pf: np.ndarray) -> np.ndarray:
    x = _as_2d(x).astype(float)
    pf = _as_2d(pf).astype(float)
    n, m = x.shape
    x = x - np.repeat(((np.sum(x, axis=1) - 1.0) / m)[:, None], m, axis=1)
    pf = pf - np.repeat(((np.sum(pf, axis=1) - 1.0) / m)[:, None], m, axis=1)
    x = x - np.repeat(np.min(pf, axis=0)[None, :], n, axis=0)
    x = x / _safe_divisor(np.repeat(np.sum(x, axis=1)[:, None], m, axis=1))
    x = np.maximum(1e-6, x)
    y = np.zeros((n, m - 1), dtype=float)
    for i in range(n):
        c = np.ones(m, dtype=float)
        nonzero = np.flatnonzero(x[i, :] != 0)
        if nonzero.size == 0:
            continue
        k = int(nonzero[0])
        for j in range(k + 1, m):
            if x[i, k] == 0:
                continue
            prod_tail = np.prod(c[(m - j + 1) : (m - k)]) if (m - j + 1) < (m - k) else 1.0
            temp = (x[i, j] / x[i, k]) * prod_tail
            c[m - j - 1] = 1.0 / (temp + 1.0)
        y[i, :] = c[: m - 1]
    powers = np.arange(m - 1, 0, -1, dtype=float)
    return y**powers


def _cpf_coverage(P: np.ndarray, maxv: float) -> float:
    P = _as_2d(P)
    n, m = P.shape
    if n == 0:
        return 0.0
    L = np.zeros(n, dtype=float)
    for x in range(n):
        P1 = P.copy()
        P1[x, :] = np.inf
        L[x] = np.min(np.max(np.abs(P1 - P[x, :]), axis=1))
    if np.isfinite(maxv):
        L = np.minimum(L, float(maxv) ** (1.0 / m))
    lower = np.maximum(0.0, P - (L[:, None] / 2.0))
    upper = np.minimum(1.0, P + (L[:, None] / 2.0))
    return float(np.sum(np.prod(upper - lower, axis=1)))


def _dm_cal_grid(P: np.ndarray, fmax: np.ndarray, fmin: np.ndarray, div: int) -> np.ndarray:
    P = _as_2d(P)
    n, m = P.shape
    if m == 0 or div <= 0:
        return np.zeros((m, max(div, 0)), dtype=bool)
    d = _safe_divisor((fmax - fmin) / float(div))
    gloc = np.ceil((P - np.repeat(fmin[None, :], n, axis=0)) / np.repeat(d[None, :], n, axis=0)).astype(int)
    gloc = np.maximum(1, gloc)
    h = np.zeros((m, div), dtype=bool)
    for i in range(m):
        idx = gloc[:, i]
        idx = idx[(idx >= 1) & (idx <= div)]
        if idx.size > 0:
            h[i, np.unique(idx) - 1] = True
    return h


def _dm_cal_m(h: np.ndarray, H: np.ndarray) -> float:
    m_dim = H.shape[0]
    h2 = np.concatenate([np.ones((m_dim, 1), dtype=bool), h.astype(bool), np.ones((m_dim, 1), dtype=bool)], axis=1)
    H2 = np.concatenate([np.ones((m_dim, 1), dtype=bool), H.astype(bool), np.ones((m_dim, 1), dtype=bool)], axis=1)
    m_val = 0.0
    for i in range(m_dim):
        for j in range(1, h2.shape[1] - 1):
            if not H2[i, j]:
                continue
            hij = bool(h2[i, j])
            him1 = bool(h2[i, j - 1])
            hip1 = bool(h2[i, j + 1])
            if hij:
                if him1:
                    m_val += 1.0 if hip1 else 0.67
                else:
                    m_val += 0.67 if hip1 else 0.75
            else:
                if him1:
                    m_val += 0.75 if hip1 else 0.5
                else:
                    m_val += 0.5 if hip1 else 0.0
    return float(m_val)


def _task_metric_optimum(context: dict[str, Any]) -> np.ndarray | None:
    return _get_reference_front(context)


def _metric_CPF(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None or pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    if optimum.shape[0] > 1:
        fmin = np.min(optimum, axis=0)
        fmax = np.max(optimum, axis=0)
        den = _safe_divisor(fmax - fmin)
        pop_n = (pop_obj - fmin) / den
        opt_n = (optimum - fmin) / den
        dist = _pairwise_euclidean(pop_n, opt_n)
        close = np.argmin(dist, axis=1)
        mapped_pop = opt_n[close, :]
        vpf = _cpf_coverage(_cpf_map(opt_n, opt_n), float("inf"))
        if vpf <= 0:
            return float("nan")
        v = _cpf_coverage(_cpf_map(mapped_pop, opt_n), vpf / max(1, mapped_pop.shape[0]))
        return float(v / vpf)
    fmin = np.min(pop_obj, axis=0)
    fmax = np.max(pop_obj, axis=0)
    pop_n = (pop_obj - fmin) / _safe_divisor(fmax - fmin)
    return float(_cpf_coverage(_cpf_map(pop_n, pop_n), 1.0 / max(1, pop_n.shape[0])))


def _metric_DeltaP(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _deltap_value(pop_obj, optimum)


def _metric_DM(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None or pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    fmax = np.max(optimum, axis=0)
    fmin = np.min(optimum, axis=0)
    H = _dm_cal_grid(optimum[:, :-1], fmax[:-1], fmin[:-1], pop_obj.shape[0])
    h = H & _dm_cal_grid(pop_obj[:, :-1], fmax[:-1], fmin[:-1], pop_obj.shape[0])
    den = _dm_cal_m(H, H)
    if den == 0:
        return float("nan")
    return float(_dm_cal_m(h, H) / den)


def _metric_Feasible_rate(front: np.ndarray, context: dict[str, Any]) -> float:
    F = _get_current_F(context, front)
    mask = _infer_feasible_mask(context, F.shape[0])
    if mask is None:
        return float("nan")
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def _metric_GD(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _gd_value(pop_obj, optimum)


def _metric_HV(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _community_hv(pop_obj, optimum, context)


def _metric_IGD(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _igd_value(pop_obj, optimum)


def _metric_IGDp(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _igdp_value(pop_obj, optimum, p=_igdp_p_from_context(context, default=2.0))


def _metric_IGDX(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_dec = _get_current_X(context)
    pos = _get_reference_set(context)
    if pop_dec is None or pos is None or pop_dec.size == 0:
        return float("nan")
    if pop_dec.shape[1] != pos.shape[1]:
        return float("nan")
    return float(np.mean(np.min(_pairwise_euclidean(pos, pop_dec), axis=1)))


def _metric_Lower_level_Min_value(front: np.ndarray, context: dict[str, Any]) -> float:
    F = _get_current_F(context, front)
    if F.size == 0 or F.shape[1] < 2:
        return float("nan")
    feasible = _infer_feasible_mask(context, F.shape[0])
    if feasible is None:
        feasible = np.ones(F.shape[0], dtype=bool)
    if not np.any(feasible):
        return float("nan")
    return float(np.min(F[feasible, 1]))


def _metric_Upper_level_Min_value(front: np.ndarray, context: dict[str, Any]) -> float:
    F = _get_current_F(context, front)
    if F.size == 0 or F.shape[1] < 1:
        return float("nan")
    feasible = _infer_feasible_mask(context, F.shape[0])
    if feasible is None:
        feasible = np.ones(F.shape[0], dtype=bool)
    if not np.any(feasible):
        return float("nan")
    return float(np.min(F[feasible, 0]))


def _metric_Min_value(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    if pop_obj.size == 0:
        return float("nan")
    return float(np.min(pop_obj[:, 0]))


def _metric_PD(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_all = _get_current_F(context, front)
    if pop_all.size == 0:
        return float("nan")
    n = pop_all.shape[0]
    if n <= 1:
        return 0.0

    # For large fronts, use scipy MST instead of O(n³) custom Prim's
    if n > 500:
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree
            D_full = _pairwise_minkowski(pop_all, pop_all, p=0.1)
            mst = minimum_spanning_tree(D_full)
            return float(mst.sum())
        except ImportError:
            pass  # Fall through to original algorithm

    C = np.eye(n, dtype=bool)
    D = _pairwise_minkowski(pop_all, pop_all, p=0.1)
    np.fill_diagonal(D, np.inf)
    score = 0.0

    for _ in range(n - 1):
        while True:
            d = np.min(D, axis=1)
            J = np.argmin(D, axis=1)
            i = int(np.argmax(d))
            j = int(J[i])

            if D[j, i] != -np.inf:
                D[j, i] = np.inf
            if D[i, j] != -np.inf:
                D[i, j] = np.inf

            P = np.any(C[i, :], axis=0)
            # Scalar bool can happen for degenerate shapes; normalize to vector.
            if np.isscalar(P):
                P = C[i, :].copy()
            while not bool(np.asarray(P)[j]):
                newP = np.any(C[np.asarray(P, dtype=bool), :], axis=0)
                if np.array_equal(np.asarray(P, dtype=bool), np.asarray(newP, dtype=bool)):
                    break
                P = newP

            if not bool(np.asarray(P)[j]):
                break

        C[i, j] = True
        C[j, i] = True
        D[i, :] = -np.inf
        score += float(d[i])

    return float(score)


def _metric_Spacing(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    if pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[0] <= 1:
        return 0.0
    dist = _pairwise_cityblock(pop_obj, pop_obj)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    if nearest.size <= 1:
        return 0.0
    return float(np.std(nearest, ddof=1))


def _metric_Spread(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _get_front(front)
    optimum = _get_reference_front(context)
    if optimum is None or pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    if pop_obj.shape[0] <= 1:
        return float("nan")
    dis1 = _pairwise_euclidean(pop_obj, pop_obj)
    np.fill_diagonal(dis1, np.inf)
    E = np.argmax(optimum, axis=0)
    dis2 = _pairwise_euclidean(optimum[E, :], pop_obj)
    d1 = np.sum(np.min(dis2, axis=1))
    nearest = np.min(dis1, axis=1)
    d2 = np.mean(nearest)
    den = d1 + (pop_obj.shape[0] - pop_obj.shape[1]) * d2
    if abs(den) <= 1e-12:
        return float("nan")
    return float((d1 + np.sum(np.abs(nearest - d2))) / den)


def _metric_Task1_HV(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _task_metric_optimum(context)
    if optimum is None:
        return float("nan")
    pop_obj = _subset_task_population(context, task_id=1)
    if pop_obj.size == 0:
        return 0.0
    return _community_hv(pop_obj, optimum, context)


def _metric_Task1_IGD(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _task_metric_optimum(context)
    if optimum is None:
        return float("nan")
    pop_obj = _subset_task_population(context, task_id=1)
    return _igd_value(pop_obj, optimum)


def _metric_Task1_Min_value(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _subset_task_population(context, task_id=1)
    if pop_obj.size == 0:
        return float("nan")
    return float(np.min(pop_obj[:, 0]))


def _metric_Task2_HV(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _task_metric_optimum(context)
    if optimum is None:
        return float("nan")
    pop_obj = _subset_task_population(context, task_id=2)
    if pop_obj.size == 0:
        return 0.0
    return _community_hv(pop_obj, optimum, context)


def _metric_Task2_IGD(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _task_metric_optimum(context)
    if optimum is None:
        return float("nan")
    pop_obj = _subset_task_population(context, task_id=2)
    return _igd_value(pop_obj, optimum)


def _metric_Task2_Min_value(front: np.ndarray, context: dict[str, Any]) -> float:
    pop_obj = _subset_task_population(context, task_id=2)
    if pop_obj.size == 0:
        return float("nan")
    return float(np.min(pop_obj[:, 0]))


def _robust_optimum_front(context: dict[str, Any]) -> np.ndarray | None:
    problem = context.get("problem")
    optimum = getattr(problem, "optimum", None) if problem is not None else None
    if optimum is None:
        return _get_reference_front(context)
    try:
        arr = _as_2d(optimum)
        return arr if arr.size > 0 else _get_reference_front(context)
    except Exception:  # noqa: BLE001
        return _get_reference_front(context)


def _metric_Mean_HV(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [s for s in (_community_hv(_non_dominated_front(F), optimum, context) for F in samples) if not math.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _metric_Mean_IGD(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [s for s in (_igd_value(_non_dominated_front(F), optimum) for F in samples) if not math.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _metric_Worst_HV(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [s for s in (_community_hv(_non_dominated_front(F), optimum, context) for F in samples) if not math.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.min(scores))


def _metric_Worst_IGD(front: np.ndarray, context: dict[str, Any]) -> float:
    optimum = _robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [s for s in (_igd_value(_non_dominated_front(F), optimum) for F in samples) if not math.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.max(scores))



def _metric_R2(front: np.ndarray, context: dict[str, Any]) -> float:
    """R2 Performance Indicator (Hansen & Jaszkiewicz, 1998)."""
    from metrics.evaluator import r2_indicator
    F = _as_2d(front)
    if F.size == 0:
        return float("nan")
    optimum = _robust_optimum_front(context)
    W = context.get("ref_dirs") or context.get("weights")
    p_part = context.get("r2_partitions")
    return float(r2_indicator(F, PF=optimum, W=W, n_partitions=p_part))

def _metric_IQHV(front: np.ndarray, context: dict[str, Any]) -> float:
    """Improved Quick Hypervolume (IQHV) metric wrapper."""
    from metrics.iqhv import iqhv
    F = _as_2d(front)
    if F.size == 0:
        return float("nan")
    optimum = _robust_optimum_front(context)
    ref = context.get("ref_point")
    if ref is None:
        ref = np.ones(F.shape[1]) if optimum is not None else np.max(F, axis=0) * 1.1
    return float(iqhv(F, ref))


def _metric_HBDA(front: np.ndarray, context: dict[str, Any]) -> float:
    """Hypervolume Box Decomposition Algorithm (HBDA) metric wrapper."""
    from metrics.hbda import hbda
    F = _as_2d(front)
    if F.size == 0:
        return float("nan")
    optimum = _robust_optimum_front(context)
    ref = context.get("ref_point")
    if ref is None:
        ref = np.ones(F.shape[1]) if optimum is not None else np.max(F, axis=0) * 1.1
    return float(hbda(F, ref))


def _metric_QEHVC(front: np.ndarray, context: dict[str, Any]) -> float:
    """Quick Extreme Hypervolume Contribution (QEHVC) sum wrapper."""
    from metrics.qehvc import qehvc
    F = _as_2d(front)
    if F.size == 0:
        return float("nan")
    optimum = _robust_optimum_front(context)
    ref = context.get("ref_point")
    if ref is None:
        ref = np.ones(F.shape[1]) if optimum is not None else np.max(F, axis=0) * 1.1
    return float(np.sum(qehvc(F, ref)))

METRICS = {
    "IQHV": _metric_IQHV,
    "HBDA": _metric_HBDA,
    "QEHVC": _metric_QEHVC,
    "CPF": _metric_CPF,
    "DeltaP": _metric_DeltaP,
    "DM": _metric_DM,
    "Feasible_rate": _metric_Feasible_rate,
    "GD": _metric_GD,
    "HV": _metric_HV,
    "IGD": _metric_IGD,
    "IGDp": _metric_IGDp,
    "IGDX": _metric_IGDX,
    "Lower_level_Min_value": _metric_Lower_level_Min_value,
    "Mean_HV": _metric_Mean_HV,
    "Mean_IGD": _metric_Mean_IGD,
    "Min_value": _metric_Min_value,
    "PD": _metric_PD,
    "Spacing": _metric_Spacing,
    "Spread": _metric_Spread,
    "Task1_HV": _metric_Task1_HV,
    "Task1_IGD": _metric_Task1_IGD,
    "Task1_Min_value": _metric_Task1_Min_value,
    "Task2_HV": _metric_Task2_HV,
    "Task2_IGD": _metric_Task2_IGD,
    "Task2_Min_value": _metric_Task2_Min_value,
    "Upper_level_Min_value": _metric_Upper_level_Min_value,
    "Worst_HV": _metric_Worst_HV,
    "R2": _metric_R2,
    "Worst_IGD": _metric_Worst_IGD,
}


def get_metrics() -> dict[str, Any]:
    return dict(METRICS)
