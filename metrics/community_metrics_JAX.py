from typing import Any

import numpy as np

from . import community_metrics as _cpu

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # noqa: BLE001
    jax = None
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


METRIC_REFERENCES = {f"{k}_JAX": v for k, v in _cpu.METRIC_REFERENCES.items()}


def _pairwise_euclidean_jax(a: Any, b: Any):
    aa = jnp.atleast_2d(jnp.asarray(a, dtype=jnp.float32))
    bb = jnp.atleast_2d(jnp.asarray(b, dtype=jnp.float32))
    if aa.size == 0 or bb.size == 0:
        return jnp.empty((aa.shape[0], bb.shape[0]), dtype=jnp.float32)
    diff = aa[:, None, :] - bb[None, :, :]
    return jnp.linalg.norm(diff, axis=2)


def _pairwise_cityblock_jax(a: Any, b: Any):
    aa = jnp.atleast_2d(jnp.asarray(a, dtype=jnp.float32))
    bb = jnp.atleast_2d(jnp.asarray(b, dtype=jnp.float32))
    if aa.size == 0 or bb.size == 0:
        return jnp.empty((aa.shape[0], bb.shape[0]), dtype=jnp.float32)
    return jnp.sum(jnp.abs(aa[:, None, :] - bb[None, :, :]), axis=2)


def _metric_DeltaP_JAX(front, context):
    if not _HAS_JAX:
        pop = _cpu._get_front(front)
        opt = _cpu._get_reference_front(context)
        if opt is None:
            return float("nan")
        return _cpu._deltap_value(pop, opt)
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None:
        return float("nan")
    if pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    d = _pairwise_euclidean_jax(opt, pop)
    v = jnp.maximum(jnp.mean(jnp.min(d, axis=1)), jnp.mean(jnp.min(d, axis=0)))
    return float(v)


def _metric_GD_JAX(front, context):
    if not _HAS_JAX:
        pop = _cpu._get_front(front)
        opt = _cpu._get_reference_front(context)
        if opt is None:
            return float("nan")
        return _cpu._gd_value(pop, opt)
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    d = _pairwise_euclidean_jax(pop, opt)
    nearest = jnp.min(d, axis=1)
    return float(jnp.linalg.norm(nearest) / max(1, int(nearest.shape[0])))


def _metric_IGD_JAX(front, context):
    if not _HAS_JAX:
        pop = _cpu._get_front(front)
        opt = _cpu._get_reference_front(context)
        if opt is None:
            return float("nan")
        return _cpu._igd_value(pop, opt)
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    d = _pairwise_euclidean_jax(opt, pop)
    return float(jnp.mean(jnp.min(d, axis=1)))


def _metric_IGDp_JAX(front, context):
    if not _HAS_JAX:
        pop = _cpu._get_front(front)
        opt = _cpu._get_reference_front(context)
        if opt is None:
            return float("nan")
        return _cpu._igdp_value(pop, opt, p=_cpu._igdp_p_from_context(context, default=2.0))
    pop = _cpu._get_front(front)
    opt = _cpu._get_reference_front(context)
    if opt is None or pop.size == 0 or pop.shape[1] != opt.shape[1]:
        return float("nan")
    p = _cpu._igdp_p_from_context(context, default=2.0)
    pop_j = jnp.atleast_2d(jnp.asarray(pop, dtype=jnp.float32))
    opt_j = jnp.atleast_2d(jnp.asarray(opt, dtype=jnp.float32))
    diff = jnp.maximum(pop_j[None, :, :] - opt_j[:, None, :], 0.0)
    if abs(float(p) - 2.0) <= 1e-12:
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=2))
    elif abs(float(p) - 1.0) <= 1e-12:
        dist = jnp.sum(diff, axis=2)
    else:
        p32 = jnp.asarray(float(p), dtype=jnp.float32)
        dist = jnp.sum(jnp.power(diff, p32), axis=2) ** (1.0 / p32)
    return float(jnp.mean(jnp.min(dist, axis=1)))


def _metric_IGDX_JAX(front, context):
    if not _HAS_JAX:
        pop_dec = _cpu._get_current_X(context)
        pos = _cpu._get_reference_set(context)
        if pop_dec is None or pos is None or pop_dec.size == 0:
            return float("nan")
        if pop_dec.shape[1] != pos.shape[1]:
            return float("nan")
        return float(np.mean(np.min(_cpu._pairwise_euclidean(pos, pop_dec), axis=1)))
    pop_dec = _cpu._get_current_X(context)
    pos = _cpu._get_reference_set(context)
    if pop_dec is None or pos is None or pop_dec.size == 0:
        return float("nan")
    if pop_dec.shape[1] != pos.shape[1]:
        return float("nan")
    d = _pairwise_euclidean_jax(pos, pop_dec)
    return float(jnp.mean(jnp.min(d, axis=1)))


def _metric_Spacing_JAX(front, context):
    if not _HAS_JAX:
        pop = _cpu._get_front(front)
        if pop.size == 0:
            return float("nan")
        if pop.shape[0] <= 1:
            return 0.0
        d = _cpu._pairwise_cityblock(pop, pop)
        np.fill_diagonal(d, np.inf)
        nearest = np.min(d, axis=1)
        if nearest.size <= 1:
            return 0.0
        return float(np.std(nearest, ddof=1))
    pop = _cpu._get_front(front)
    if pop.size == 0:
        return float("nan")
    if pop.shape[0] <= 1:
        return 0.0
    d = _pairwise_cityblock_jax(pop, pop)
    n = d.shape[0]
    d = d + jnp.eye(n, dtype=d.dtype) * jnp.inf
    nearest = jnp.min(d, axis=1)
    if int(nearest.shape[0]) <= 1:
        return 0.0
    return float(jnp.std(nearest, ddof=1))


def _metric_Feasible_rate_JAX(front, context):
    if not _HAS_JAX:
        F = _cpu._get_current_F(context, front)
        mask = _cpu._infer_feasible_mask(context, int(F.shape[0]))
        if mask is None or mask.size == 0:
            return float("nan")
        return float(np.mean(mask.astype(float)))
    F = _cpu._get_current_F(context, front)
    mask = _cpu._infer_feasible_mask(context, int(F.shape[0]))
    if mask is None or mask.size == 0:
        return float("nan")
    return float(jnp.mean(jnp.asarray(mask, dtype=jnp.float32)))


def _metric_Min_value_JAX(front, context):
    if not _HAS_JAX:
        pop = _cpu._get_front(front)
        if pop.size == 0:
            return float("nan")
        return float(np.min(pop[:, 0]))
    pop = _cpu._get_front(front)
    if pop.size == 0:
        return float("nan")
    return float(jnp.min(jnp.asarray(pop[:, 0], dtype=jnp.float32)))


def _metric_Upper_level_Min_value_JAX(front, context):
    if not _HAS_JAX:
        F = _cpu._get_current_F(context, front)
        if F.size == 0 or F.shape[1] < 1:
            return float("nan")
        feas = _cpu._infer_feasible_mask(context, int(F.shape[0]))
        if feas is None:
            feas = np.ones(F.shape[0], dtype=bool)
        if not np.any(feas):
            return float("nan")
        return float(np.min(F[feas, 0]))
    F = _cpu._get_current_F(context, front)
    if F.size == 0 or F.shape[1] < 1:
        return float("nan")
    feas = _cpu._infer_feasible_mask(context, int(F.shape[0]))
    if feas is None:
        feas = np.ones(F.shape[0], dtype=bool)
    if not np.any(feas):
        return float("nan")
    vals = jnp.asarray(F[feas, 0], dtype=jnp.float32)
    return float(jnp.min(vals))


def _metric_Lower_level_Min_value_JAX(front, context):
    if not _HAS_JAX:
        F = _cpu._get_current_F(context, front)
        if F.size == 0 or F.shape[1] < 2:
            return float("nan")
        feas = _cpu._infer_feasible_mask(context, int(F.shape[0]))
        if feas is None:
            feas = np.ones(F.shape[0], dtype=bool)
        if not np.any(feas):
            return float("nan")
        return float(np.min(F[feas, 1]))
    F = _cpu._get_current_F(context, front)
    if F.size == 0 or F.shape[1] < 2:
        return float("nan")
    feas = _cpu._infer_feasible_mask(context, int(F.shape[0]))
    if feas is None:
        feas = np.ones(F.shape[0], dtype=bool)
    if not np.any(feas):
        return float("nan")
    vals = jnp.asarray(F[feas, 1], dtype=jnp.float32)
    return float(jnp.min(vals))


def _metric_HV_JAX(front, context):
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    return _cpu._community_hv(pop_obj, optimum, context)


def _metric_CPF_JAX(front, context):
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None or pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    if optimum.shape[0] > 1:
        fmin = np.min(optimum, axis=0)
        fmax = np.max(optimum, axis=0)
        den = _cpu._safe_divisor(fmax - fmin)
        pop_n = (pop_obj - fmin) / den
        opt_n = (optimum - fmin) / den
        dist = _cpu._pairwise_euclidean(pop_n, opt_n)
        close = np.argmin(dist, axis=1)
        mapped_pop = opt_n[close, :]
        vpf = _cpu._cpf_coverage(_cpu._cpf_map(opt_n, opt_n), float("inf"))
        if vpf <= 0:
            return float("nan")
        v = _cpu._cpf_coverage(_cpu._cpf_map(mapped_pop, opt_n), vpf / max(1, mapped_pop.shape[0]))
        return float(v / vpf)
    fmin = np.min(pop_obj, axis=0)
    fmax = np.max(pop_obj, axis=0)
    pop_n = (pop_obj - fmin) / _cpu._safe_divisor(fmax - fmin)
    return float(_cpu._cpf_coverage(_cpu._cpf_map(pop_n, pop_n), 1.0 / max(1, pop_n.shape[0])))


def _metric_DM_JAX(front, context):
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None or pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    fmax = np.max(optimum, axis=0)
    fmin = np.min(optimum, axis=0)
    H = _cpu._dm_cal_grid(optimum[:, :-1], fmax[:-1], fmin[:-1], pop_obj.shape[0])
    h = H & _cpu._dm_cal_grid(pop_obj[:, :-1], fmax[:-1], fmin[:-1], pop_obj.shape[0])
    den = _cpu._dm_cal_m(H, H)
    if den == 0:
        return float("nan")
    return float(_cpu._dm_cal_m(h, H) / den)


def _metric_PD_JAX(front, context):
    pop_all = _cpu._get_current_F(context, front)
    if pop_all.size == 0:
        return float("nan")
    n = pop_all.shape[0]
    if n <= 1:
        return 0.0
    C = np.eye(n, dtype=bool)
    D = _cpu._pairwise_minkowski(pop_all, pop_all, p=0.1)
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


def _metric_Spread_JAX(front, context):
    pop_obj = _cpu._get_front(front)
    optimum = _cpu._get_reference_front(context)
    if optimum is None or pop_obj.size == 0:
        return float("nan")
    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")
    if pop_obj.shape[0] <= 1:
        return float("nan")
    dis1 = _cpu._pairwise_euclidean(pop_obj, pop_obj)
    np.fill_diagonal(dis1, np.inf)
    E = np.argmax(optimum, axis=0)
    dis2 = _cpu._pairwise_euclidean(optimum[E, :], pop_obj)
    d1 = np.sum(np.min(dis2, axis=1))
    nearest = np.min(dis1, axis=1)
    d2 = np.mean(nearest)
    den = d1 + (pop_obj.shape[0] - pop_obj.shape[1]) * d2
    if abs(den) <= 1e-12:
        return float("nan")
    return float((d1 + np.sum(np.abs(nearest - d2))) / den)


def _metric_Task1_HV_JAX(front, context):
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    pop_obj = _cpu._subset_task_population(context, task_id=1)
    if pop_obj.size == 0:
        return 0.0
    return _cpu._community_hv(pop_obj, optimum, context)


def _metric_Task1_IGD_JAX(front, context):
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    pop_obj = _cpu._subset_task_population(context, task_id=1)
    return _cpu._igd_value(pop_obj, optimum)


def _metric_Task1_Min_value_JAX(front, context):
    pop_obj = _cpu._subset_task_population(context, task_id=1)
    if pop_obj.size == 0:
        return float("nan")
    return float(np.min(pop_obj[:, 0]))


def _metric_Task2_HV_JAX(front, context):
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    pop_obj = _cpu._subset_task_population(context, task_id=2)
    if pop_obj.size == 0:
        return 0.0
    return _cpu._community_hv(pop_obj, optimum, context)


def _metric_Task2_IGD_JAX(front, context):
    optimum = _cpu._get_reference_front(context)
    if optimum is None:
        return float("nan")
    pop_obj = _cpu._subset_task_population(context, task_id=2)
    return _cpu._igd_value(pop_obj, optimum)


def _metric_Task2_Min_value_JAX(front, context):
    pop_obj = _cpu._subset_task_population(context, task_id=2)
    if pop_obj.size == 0:
        return float("nan")
    return float(np.min(pop_obj[:, 0]))


def _metric_Mean_HV_JAX(front, context):
    optimum = _cpu._robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _cpu._robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [
        s
        for s in (_cpu._community_hv(_cpu._non_dominated_front(F), optimum, context) for F in samples)
        if not np.isnan(s)
    ]
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _metric_Mean_IGD_JAX(front, context):
    optimum = _cpu._robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _cpu._robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [s for s in (_cpu._igd_value(_cpu._non_dominated_front(F), optimum) for F in samples) if not np.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _metric_Worst_HV_JAX(front, context):
    optimum = _cpu._robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _cpu._robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [
        s
        for s in (_cpu._community_hv(_cpu._non_dominated_front(F), optimum, context) for F in samples)
        if not np.isnan(s)
    ]
    if not scores:
        return float("nan")
    return float(np.min(scores))


def _metric_Worst_IGD_JAX(front, context):
    optimum = _cpu._robust_optimum_front(context)
    if optimum is None:
        return float("nan")
    samples = _cpu._robust_metric_samples(context)
    if not samples:
        return float("nan")
    scores = [s for s in (_cpu._igd_value(_cpu._non_dominated_front(F), optimum) for F in samples) if not np.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.max(scores))



def _metric_R2_JAX(front, context):
    """R2 Indicator computed via JAX tensor operations with CPU fallback."""
    F = _as_2d(front)
    if F.size == 0:
        return float("nan")
    if not np.all(np.isfinite(F)):
        raise ValueError("R2 indicator input front contains non-finite values (NaN or Inf)")
    if not _HAS_JAX:
        from metrics.community_metrics import _metric_R2
        return _metric_R2(front, context)

    optimum = _cpu._robust_optimum_front(context)
    PF = _as_2d(optimum) if optimum is not None else None
    if PF is not None:
        if PF.size == 0:
            return float("nan")
        if not np.all(np.isfinite(PF)):
            raise ValueError("R2 indicator reference front contains non-finite values (NaN or Inf)")
        if PF.shape[1] != F.shape[1]:
            raise ValueError(f"Reference front dimension {PF.shape[1]} does not match front dimension {F.shape[1]}")

    N, M = F.shape
    if M < 2:
        raise ValueError(f"R2 indicator requires at least 2 objectives, got {M}")

    W = context.get("ref_dirs") or context.get("weights")
    if W is None:
        from core.tensor.ref_dirs import get_reference_directions
        from metrics.indicators import _adaptive_r2_partitions
        p_part = context.get("r2_partitions") or _adaptive_r2_partitions(M)
        W = get_reference_directions("das-dennis", n_obj=M, n_partitions=p_part)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.shape[1] != M:
        raise ValueError(f"Weight vectors dimension {W.shape} does not match front objectives {M}")
    if np.any(np.sum(W, axis=1) <= 1e-12):
        raise ValueError("Weight vectors must not contain all-zero rows")
    if W.shape[0] > 2000:
        sub_idx = np.random.default_rng(0).choice(W.shape[0], size=2000, replace=False)
        W = W[np.sort(sub_idx)]

    if PF is not None and PF.size > 0:
        z_min = np.min(PF, axis=0)
        z_max = np.max(PF, axis=0)
    else:
        z_min = np.min(F, axis=0)
        z_max = np.max(F, axis=0)

    den = z_max - z_min
    den = np.where(np.abs(den) <= 1e-12, 1.0, den)

    F_norm_j = jnp.asarray((F - z_min) / den, dtype=jnp.float32)
    W_j = jnp.asarray(W, dtype=jnp.float32)
    z_ideal_j = jnp.zeros(M, dtype=jnp.float32)

    # JAX 3D tensor broadcasting
    diff = jnp.abs(F_norm_j[:, None, :] - z_ideal_j[None, None, :])
    utility = W_j[None, :, :] * diff
    chebyshev = jnp.max(utility, axis=2)
    min_over_pop = jnp.min(chebyshev, axis=0)
    return float(jnp.mean(min_over_pop))

def _metric_IQHV_JAX(front, context):
    """Fallback wrapper delegating IQHV to verified CPU implementation."""
    return _cpu._metric_IQHV(front, context)

def _metric_HBDA_JAX(front, context):
    """Fallback wrapper delegating HBDA to verified CPU implementation."""
    return _cpu._metric_HBDA(front, context)

def _metric_QEHVC_JAX(front, context):
    """Fallback wrapper delegating QEHVC to verified CPU implementation."""
    return _cpu._metric_QEHVC(front, context)

METRICS = {
    "CPF_JAX": _metric_CPF_JAX,
    "DeltaP_JAX": _metric_DeltaP_JAX,
    "DM_JAX": _metric_DM_JAX,
    "Feasible_rate_JAX": _metric_Feasible_rate_JAX,
    "GD_JAX": _metric_GD_JAX,
    "HV_JAX": _metric_HV_JAX,
    "IGD_JAX": _metric_IGD_JAX,
    "IGDp_JAX": _metric_IGDp_JAX,
    "IGDX_JAX": _metric_IGDX_JAX,
    "Lower_level_Min_value_JAX": _metric_Lower_level_Min_value_JAX,
    "Mean_HV_JAX": _metric_Mean_HV_JAX,
    "Mean_IGD_JAX": _metric_Mean_IGD_JAX,
    "Min_value_JAX": _metric_Min_value_JAX,
    "PD_JAX": _metric_PD_JAX,
    "Spacing_JAX": _metric_Spacing_JAX,
    "Spread_JAX": _metric_Spread_JAX,
    "Task1_HV_JAX": _metric_Task1_HV_JAX,
    "Task1_IGD_JAX": _metric_Task1_IGD_JAX,
    "Task1_Min_value_JAX": _metric_Task1_Min_value_JAX,
    "Task2_HV_JAX": _metric_Task2_HV_JAX,
    "Task2_IGD_JAX": _metric_Task2_IGD_JAX,
    "Task2_Min_value_JAX": _metric_Task2_Min_value_JAX,
    "Upper_level_Min_value_JAX": _metric_Upper_level_Min_value_JAX,
    "Worst_HV_JAX": _metric_Worst_HV_JAX,
    "Worst_IGD_JAX": _metric_Worst_IGD_JAX,
    "R2_JAX": _metric_R2_JAX,
    "IQHV_JAX": _metric_IQHV_JAX,
    "HBDA_JAX": _metric_HBDA_JAX,
    "QEHVC_JAX": _metric_QEHVC_JAX,
}


def evaluate_batch_igd_plus_jax(
    fronts_tensor: np.ndarray,
    ref_front: np.ndarray,
    p: float = 2.0,
) -> np.ndarray:
    """Computes IGD+ across a 3D batch of approximation fronts [R, N, M] against a single reference front [K, M].

    Utilizes JAX JIT and vmap for high-throughput evaluation of experimental campaigns in a single GPU call.
    """
    if not _HAS_JAX:
        # Fallback to iterated CPU calculation
        res = []
        for i in range(fronts_tensor.shape[0]):
            res.append(_metric_IGDp(fronts_tensor[i], {"pareto_front": ref_front, "p": p}))
        return np.array(res, dtype=float)

    F_batch = jnp.asarray(fronts_tensor, dtype=jnp.float32)
    opt_j = jnp.asarray(ref_front, dtype=jnp.float32)

    def _single_igd_plus(pop_single: Any, opt: Any) -> Any:
        diff = jnp.maximum(pop_single[None, :, :] - opt[:, None, :], 0.0)
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=2))
        return jnp.mean(jnp.min(dist, axis=1))

    vmapped_fn = jax.vmap(_single_igd_plus, in_axes=(0, None))
    return np.asarray(jax.jit(vmapped_fn)(F_batch, opt_j), dtype=float)


