"""Unit tests and mathematical parity validation for the R2 Performance Indicator."""

import math
import numpy as np
import pytest

from metrics.indicators import R2, _adaptive_r2_partitions
from metrics.evaluator import r2_indicator, MetricEvaluator, evaluate_front
from metrics.community_metrics import METRICS as COMMUNITY_METRICS


def _ground_truth_r2_double_loop(F: np.ndarray, PF: np.ndarray, W: np.ndarray) -> float:
    """Naive unvectorized mathematical reference implementation of R2."""
    N, M = F.shape
    NW, _ = W.shape
    z_min = np.min(PF, axis=0)
    z_max = np.max(PF, axis=0)
    den = z_max - z_min
    den = np.where(np.abs(den) <= 1e-12, 1.0, den)
    F_norm = (F - z_min) / den

    # For each weight vector w, find min over all solutions a of max_m (w_m * a_m)
    r2_sum = 0.0
    for i in range(NW):
        w = W[i]
        min_u = float("inf")
        for j in range(N):
            a = F_norm[j]
            # Tchebycheff utility against ideal point at 0
            u = max(w[m] * abs(a[m]) for m in range(M))
            if u < min_u:
                min_u = u
        r2_sum += min_u

    return float(r2_sum / NW)


def test_r2_exact_mathematical_parity():
    """Verify that vectorized 3D tensor broadcasting matches naive scalar double-loop to machine epsilon."""
    rng = np.random.default_rng(42)
    N, M, K = 35, 3, 20
    F = rng.uniform(0.1, 10.0, size=(N, M))
    PF = rng.uniform(0.0, 8.0, size=(100, M))
    W = rng.uniform(0.05, 1.0, size=(K, M))
    W /= np.sum(W, axis=1, keepdims=True)

    expected = _ground_truth_r2_double_loop(F, PF, W)
    actual = R2(pf=PF, ref_dirs=W).do(F)

    assert isinstance(actual, float)
    assert not math.isnan(actual)
    assert abs(actual - expected) < 1e-12, f"Expected {expected}, got {actual}"


def test_r2_boundary_and_empty_conditions():
    """Verify strict boundary and empty edge cases."""
    rng = np.random.default_rng(123)
    PF = rng.random((50, 3))

    # Empty front yields NaN
    assert math.isnan(R2(pf=PF).do(np.empty((0, 3))))
    assert math.isnan(r2_indicator(np.empty((0, 3)), PF=PF))
    assert math.isnan(MetricEvaluator.evaluate("R2", np.empty((0, 3)), context={"pareto_front": PF}))

    # Empty PF yields NaN
    F = rng.random((20, 3))
    assert math.isnan(R2(pf=np.empty((0, 3))).do(F))

    # Single solution (1D array) auto-reshaped to 2D
    single = np.array([1.5, 2.5, 3.5])
    val_single = R2(pf=PF).do(single)
    assert isinstance(val_single, float)
    assert not math.isnan(val_single)


def test_r2_unary_without_reference_front():
    """Verify that unary R2 operates properly when no reference front is supplied."""
    rng = np.random.default_rng(7)
    F = rng.random((30, 2))
    val = R2().do(F)
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert val >= 0.0


def test_r2_weak_pareto_monotonicity():
    """Adding a strictly dominated point should not worsen the R2 score."""
    rng = np.random.default_rng(999)
    PF = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    score_orig = R2(pf=PF).do(F)

    # Add a strictly dominated point [2.0, 2.0]
    F_worse = np.vstack([F, [2.0, 2.0]])
    score_worse = R2(pf=PF).do(F_worse)

    # R2 minimizes; dominated points do not decrease any min_u, so score remains invariant
    assert abs(score_orig - score_worse) < 1e-12


def test_r2_evaluator_dispatch_and_catalog():
    """Verify integration across MetricEvaluator, evaluate_front, and community registry."""
    rng = np.random.default_rng(88)
    F = rng.random((25, 3))
    PF = rng.random((60, 3))

    # MetricEvaluator dispatch
    eval_score = MetricEvaluator.evaluate("R2", F, context={"pareto_front": PF})
    assert isinstance(eval_score, float)
    assert not math.isnan(eval_score)

    # evaluate_front runner
    front_dict = evaluate_front(F, pf_true=PF, metrics=["r2", "spacing"])
    assert "r2" in front_dict
    assert isinstance(front_dict["r2"], float)
    assert abs(front_dict["r2"] - eval_score) < 1e-12

    # Community metrics registry
    assert "R2" in COMMUNITY_METRICS
    comm_score = COMMUNITY_METRICS["R2"](F, {"pareto_front": PF})
    assert abs(comm_score - eval_score) < 1e-12


def test_r2_senior_safe_exception_guards():
    """Verify strict validation and exception behavior on invalid inputs."""
    rng = np.random.default_rng(55)
    PF = rng.random((20, 3))
    F = rng.random((15, 3))

    # Non-finite values in F
    F_nan = F.copy()
    F_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        R2(pf=PF).do(F_nan)

    F_inf = F.copy()
    F_inf[1, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite values"):
        R2(pf=PF).do(F_inf)

    # Non-finite values in PF
    PF_nan = PF.copy()
    PF_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        R2(pf=PF_nan).do(F)

    # Dimension mismatch between F and PF
    PF_2d = rng.random((20, 2))
    with pytest.raises(ValueError, match="does not match"):
        R2(pf=PF_2d).do(F)

    # Invalid M < 2
    F_1d = rng.random((15, 1))
    with pytest.raises(ValueError, match="at least 2 objectives"):
        R2().do(F_1d)

    # All-zero weight row
    W_zero = np.zeros((5, 3))
    with pytest.raises(ValueError, match="all-zero rows"):
        R2(pf=PF, ref_dirs=W_zero).do(F)


def test_adaptive_partitions():
    """Verify adaptive partition rules for dimension scaling."""
    assert _adaptive_r2_partitions(2) == 99
    assert _adaptive_r2_partitions(3) == 13
    assert _adaptive_r2_partitions(4) == 7
    assert _adaptive_r2_partitions(5) == 5
    assert _adaptive_r2_partitions(8) == 3
    assert _adaptive_r2_partitions(10) == 3
    assert _adaptive_r2_partitions(15) == 3
