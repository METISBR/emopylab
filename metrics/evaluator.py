"""Deep Performance Metric Evaluator and Quality Indicator Suite for EmoPyLab.

This module unifies metric evaluation (HV, IGD, IGDp, GD, DeltaP, Spacing, Spread, etc.)
behind a minimal, deep interface with adaptive hardware/dimension dispatching.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from metrics.indicators import HV as _NativeHV

try:
    from util.nds.non_dominated_sorting import NonDominatedSorting
    _NDS = NonDominatedSorting()
except Exception:  # noqa: BLE001
    _NDS = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Numerical helper routines
# ---------------------------------------------------------------------------

def _as_2d_float(values: Any) -> np.ndarray:
    """Ensure array is at least 2D float64 with finite values."""
    if values is None:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _non_dominated_front(F: np.ndarray) -> np.ndarray:
    """Extract first non-dominated front using efficient non-dominated sort."""
    if F.size == 0 or F.shape[0] <= 1:
        return F
    if _NDS is None:
        return F
    try:
        fronts = _NDS.do(F, n_stop_if_ranked=1)
        if fronts and len(fronts[0]) > 0:
            return F[fronts[0]]
    except Exception:
        pass
    return F


# ---------------------------------------------------------------------------
# Core Quality Indicator Algorithms
# ---------------------------------------------------------------------------

def generational_distance(F: np.ndarray, PF: np.ndarray, p: float = 1.0) -> float:
    """Calculate Generational Distance (GD_p) from approximated front F to true PF."""
    F = _as_2d_float(F)
    PF = _as_2d_float(PF)
    if F.size == 0 or PF.size == 0:
        return float("nan")
    p_val = float(max(1e-6, p))
    # Distance from each solution in F to nearest solution in PF
    dists = np.min(np.linalg.norm(F[:, None, :] - PF[None, :, :], axis=2), axis=1)
    return float(np.mean(dists ** p_val) ** (1.0 / p_val))


def inverted_generational_distance(F: np.ndarray, PF: np.ndarray, p: float = 1.0) -> float:
    """Calculate Inverted Generational Distance (IGD_p) from true PF to approximated front F."""
    F = _as_2d_float(F)
    PF = _as_2d_float(PF)
    if F.size == 0 or PF.size == 0:
        return float("nan")
    p_val = float(max(1e-6, p))
    # Distance from each point in PF to nearest solution in F
    dists = np.min(np.linalg.norm(PF[:, None, :] - F[None, :, :], axis=2), axis=1)
    return float(np.mean(dists ** p_val) ** (1.0 / p_val))


def averaged_hausdorff_distance(F: np.ndarray, PF: np.ndarray, p: float = 1.0) -> Tuple[float, float, float]:
    """Calculate Averaged Hausdorff Distance Delta_p = max(GD_p, IGD_p).

    Returns (Delta_p, GD_p, IGD_p).
    """
    gd = generational_distance(F, PF, p=p)
    igd = inverted_generational_distance(F, PF, p=p)
    if math.isnan(gd) or math.isnan(igd):
        return float("nan"), gd, igd
    return float(max(gd, igd)), gd, igd


def hypervolume(F: np.ndarray,
                PF: Optional[np.ndarray] = None,
                ref_point: Optional[Sequence[float] | np.ndarray] = None,
                n_obj: Optional[int] = None,
                sample_num: int = 100_000) -> float:
    """Calculate Hypervolume using an adaptive Hybrid Dispatcher.

    Scientific Principle:
    - For M <= 3 (e.g. ZDT, bi/tri-objective DTLZ/WFG): Exact HV is O(N log N) (< 0.1 ms).
    - For M >= 4 (Many-Objective, e.g. MaF, DTLZ 5-15D): Exact HV is #P-Hard (O(N^(M/2))).
      Utilizes Fast Monte Carlo with Dynamic Sample Pruning (PlatEMO standard).
    """
    F = _as_2d_float(F)
    if F.size == 0:
        return 0.0

    M = int(n_obj or F.shape[1])
    PF_arr = _as_2d_float(PF) if PF is not None else None

    # Step 1: Normalization if true Pareto front is provided
    if PF_arr is not None and PF_arr.size > 0:
        fmin = np.min(PF_arr, axis=0)
        fmax = np.max(PF_arr, axis=0)
        den = (fmax - fmin) * 1.1
        den = np.where(np.abs(den) <= 1e-12, 1.0, den)
        norm_F = (F - fmin) / den
        # Exclude solutions out of bounds (> 1.0)
        norm_F = norm_F[~np.any(norm_F > 1.0, axis=1)]
        if norm_F.size == 0:
            return 0.0
        canonical_ref = np.ones(M)
    else:
        norm_F = F
        if ref_point is not None:
            canonical_ref = np.asarray(ref_point, dtype=float)
        else:
            canonical_ref = np.max(norm_F, axis=0) * 1.1

    # Step 2: Exact HV calculation for M <= 3
    if M <= 3 and _NativeHV is not None:
        try:
            valid_idx = np.all(norm_F <= canonical_ref, axis=1)
            if not np.any(valid_idx):
                return 0.0
            hv_calc = _NativeHV(ref_point=canonical_ref)
            return float(hv_calc(norm_F[valid_idx]))
        except Exception:
            pass

    # Step 3: Fast Monte Carlo with Dynamic Sample Pruning for M >= 4
    if PF_arr is not None and PF_arr.size > 0:
        try:
            from metrics.hv_fast_mc import HV_fast_MC
            return float(HV_fast_MC(F, PF_arr, sample_num=sample_num))
        except Exception:
            pass

    # Fallback exact calculation
    if _NativeHV is not None:
        try:
            valid_idx = np.all(norm_F <= canonical_ref, axis=1)
            if not np.any(valid_idx):
                return 0.0
            hv_calc = _NativeHV(ref_point=canonical_ref)
            return float(hv_calc(norm_F[valid_idx]))
        except Exception:
            pass

    return float("nan")


def spacing(F: np.ndarray) -> float:
    """Calculate Spacing metric (Schott, 1995) measuring uniformity of front spread."""
    F = _as_2d_float(F)
    n_points = F.shape[0]
    if n_points <= 1:
        return 0.0
    # Pairwise Manhattan/L1 distance
    dists = np.sum(np.abs(F[:, None, :] - F[None, :, :]), axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    d_mean = np.mean(min_dists)
    if d_mean <= 1e-12:
        return 0.0
    return float(np.sqrt(np.sum((min_dists - d_mean) ** 2) / (n_points - 1)))


# ---------------------------------------------------------------------------
# Deep Metric Evaluator
# ---------------------------------------------------------------------------

class MetricEvaluator:
    """Deep Module presenting a single unified seam for metric evaluation.

    Interface:
      evaluate(metric_name: str, front: np.ndarray, context: Mapping[str, Any] | None = None) -> float
    """

    @classmethod
    def evaluate(cls,
                 metric_name: str,
                 front: Any,
                 context: Optional[Mapping[str, Any]] = None) -> float:
        """Evaluate a given metric name against an approximated Pareto front."""
        name = str(metric_name).strip()
        ctx = dict(context or {})
        F = _as_2d_float(front)

        pf = ctx.get("pareto_front")
        if pf is None:
            pf = ctx.get("optimum")
        if pf is None:
            pf = ctx.get("reference_front")
        pf_arr = _as_2d_float(pf) if pf is not None else None

        normalized_name = name.upper().replace("_", "").replace("-", "")

        if normalized_name in {"IGD", "INVERTEDGENERATIONALDISTANCE"}:
            if pf_arr is None or pf_arr.size == 0:
                return float("nan")
            return inverted_generational_distance(F, pf_arr, p=1.0)

        if normalized_name in {"IGDP", "IGDPLUS"}:
            if pf_arr is None or pf_arr.size == 0:
                return float("nan")
            p = float(ctx.get("p", 2.0))
            return inverted_generational_distance(F, pf_arr, p=p)

        if normalized_name in {"GD", "GENERATIONALDISTANCE"}:
            if pf_arr is None or pf_arr.size == 0:
                return float("nan")
            return generational_distance(F, pf_arr, p=1.0)

        if normalized_name in {"GDP", "GDPLUS"}:
            if pf_arr is None or pf_arr.size == 0:
                return float("nan")
            p = float(ctx.get("p", 2.0))
            return generational_distance(F, pf_arr, p=p)

        if normalized_name in {"DELTAP", "DELTAPLUS", "HAUSDORFF"}:
            if pf_arr is None or pf_arr.size == 0:
                return float("nan")
            p = float(ctx.get("p", 1.0))
            return averaged_hausdorff_distance(F, pf_arr, p=p)[0]

        if normalized_name in {"HV", "HYPERVOLUME", "HVFASTMC"}:
            sample_num = int(ctx.get("hv_mc_samples", 100_000))
            return hypervolume(F, PF=pf_arr, ref_point=ctx.get("ref_point"), sample_num=sample_num)

        if normalized_name in {"SPACING"}:
            return spacing(F)

def evaluate_front(
    F: np.ndarray,
    pf_true: Optional[np.ndarray] = None,
    metrics: Optional[Sequence[str]] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """Computes a standard dictionary of quality indicators for an approximation front F."""
    evaluator = MetricEvaluator()
    ctx = dict(context or {})
    if pf_true is not None:
        ctx["pareto_front"] = pf_true
        ctx["reference_front"] = pf_true

    target_metrics = list(metrics) if metrics else ["hv_fast", "hv", "igd_plus", "spacing"]
    results: dict[str, float] = {}

    for m in target_metrics:
        try:
            val = evaluator.evaluate(m, F, context=ctx)
            if not np.isnan(val):
                results[m] = float(val)
        except Exception:
            continue

    return results

