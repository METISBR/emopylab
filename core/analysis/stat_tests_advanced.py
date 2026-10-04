"""Advanced Statistical Testing and Multi-metric Calculations for EmoPyLab Benchmarks.

Provides:
- Symmetric Averaged Hausdorff Distance (Delta_p = max(GD_p, IGD_p))
- Generational Distance (GD_p) and Inverted Generational Distance (IGD_p)
- Normalized Hypervolume (HV)
- Non-parametric Vargha-Delaney A12 effect size
- Holm-Bonferroni step-down multiple comparison correction
- Friedman ranking test and average rank calculation
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def _rankdata_np(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    ranks = np.empty(len(a), dtype=float)
    u, counts = np.unique(a, return_counts=True)
    curr = 1.0
    for val, count in zip(u, counts):
        rank_val = curr + (count - 1.0) / 2.0
        ranks[a == val] = rank_val
        curr += count
    return ranks

def _mannwhitneyu_np(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> tuple[float, float]:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = _rankdata_np(combined)
    r1 = np.sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1), float("nan")

def _friedmanchisquare_np(*args: Any) -> tuple[float, float]:
    data = np.column_stack(args)
    n, k = data.shape
    ranks = np.array([_rankdata_np(row) for row in data])
    r_j = np.sum(ranks, axis=0)
    q = (12.0 / (n * k * (k + 1.0))) * np.sum(r_j ** 2) - 3.0 * n * (k + 1.0)
    return float(q), float("nan")

try:
    from scipy import stats
except Exception:
    class _StatsFallback:
        rankdata = staticmethod(_rankdata_np)
        mannwhitneyu = staticmethod(_mannwhitneyu_np)
        friedmanchisquare = staticmethod(_friedmanchisquare_np)
    stats = _StatsFallback()  # type: ignore[assignment]
from metrics.evaluator import (
    averaged_hausdorff_distance as _calc_delta_p,
    generational_distance as _calc_gd_p,
    hypervolume as _calc_hv,
    inverted_generational_distance as _calc_igd_p,
)


def calc_gd_p(F: np.ndarray, PF: np.ndarray, p: float = 1.0) -> float:
    """Calculate Generational Distance (GD_p)."""
    return _calc_gd_p(F, PF, p=p)


def calc_igd_p(F: np.ndarray, PF: np.ndarray, p: float = 1.0) -> float:
    """Calculate Inverted Generational Distance (IGD_p)."""
    return _calc_igd_p(F, PF, p=p)


def calc_delta_p(F: np.ndarray, PF: np.ndarray, p: float = 1.0) -> float:
    """Calculate Averaged Hausdorff Distance (Delta_p = max(GD_p, IGD_p))."""
    res = _calc_delta_p(F, PF, p=p)
    return float(res[0] if isinstance(res, (tuple, list)) else res)


def calc_normalized_hypervolume(F: np.ndarray, PF: Optional[np.ndarray] = None, n_obj: Optional[int] = None, sample_num: int = 100_000) -> float:
    """Calculate Hypervolume using an adaptive Hybrid Dispatcher."""
    return _calc_hv(F, PF=PF, n_obj=n_obj, sample_num=sample_num)


def vargha_delaney_a12(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, str]:
    """Calculate the non-parametric Vargha-Delaney A12 effect size statistic.

    A12 > 0.5 means sample1 has higher values (or lower in minimization contexts).
    Magnitude tiers:
      - Negligible: 0.50 <= A12 < 0.56
      - Small:      0.56 <= A12 < 0.64
      - Medium:     0.64 <= A12 < 0.71
      - Large:      0.71 <= A12 <= 1.00
    """
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]
    n1 = len(s1)
    n2 = len(s2)
    if n1 == 0 or n2 == 0:
        return 0.5, "negligible"

    # Compute Mann-Whitney U
    u_stat, _ = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    a12 = float(u_stat / (n1 * n2))

    # Symmetric magnitude on max(A12, 1 - A12). Rounding to 12 decimals removes the
    # floating-point asymmetry that labelled 0.71 "medium" but 0.29 "large"
    # (revision 2026-09; see tests/test_stat_tests_advanced.py).
    a_sym = round(max(a12, 1.0 - a12), 12)
    if a_sym < 0.56:
        mag = "negligible"
    elif a_sym < 0.64:
        mag = "small"
    elif a_sym < 0.71:
        mag = "medium"
    else:
        mag = "large"
    return a12, mag


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply the step-down Holm-Bonferroni multiple-comparison correction.

    Returns list of (adjusted_p_value, is_significant) tuples preserving original order.
    """
    if len(p_values) == 0:
        return []

    # Non-finite p-values (tests that could not be computed) are not members of
    # the family: they are returned as (nan, False) and do not count in m
    # (revision 2026-09; previously one NaN poisoned the whole family).
    finite = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    m = len(finite)
    if m == 0:
        return [(float("nan"), False) for _ in p_values]

    indexed_p = sorted(finite, key=lambda x: x[1])
    adjusted_indexed = []

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        k = m - rank
        adjusted = min(1.0, p_val * k)
        running_max = max(running_max, adjusted)
        is_sig = running_max < alpha
        adjusted_indexed.append((orig_idx, running_max, is_sig))

    # Restore original index order (non-finite inputs -> (nan, False))
    out: List[Tuple[float, bool]] = [(float("nan"), False) for _ in p_values]
    for orig_idx, adj, sig in adjusted_indexed:
        out[orig_idx] = (adj, bool(sig))
    return out


def hodges_lehmann_shift(
    sample1: np.ndarray, sample2: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Hodges-Lehmann location shift (sample1 - sample2) with a distribution-free CI.

    Estimate: median of all pairwise differences x_i - y_j. Confidence interval:
    Moses' order-statistic interval on the sorted pairwise differences, with the
    Mann-Whitney critical value from the normal approximation
    (Hollander, Wolfe & Chicken, Nonparametric Statistical Methods, Sec. 4.3).
    Complements the Wilcoxon rank-sum (Mann-Whitney U) test and the A12 effect size
    by giving a confidence interval in the units of the metric.

    Returns (estimate, lower, upper); (nan, nan, nan) if a sample is empty.
    """
    x = np.asarray(sample1, dtype=float)
    y = np.asarray(sample2, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = np.sort((x[:, None] - y[None, :]).ravel())
    estimate = float(np.median(diffs))
    n_pairs = diffs.size
    z = float(NormalDist().inv_cdf(0.5 + confidence / 2.0))  # stdlib: works without scipy
    k = int(np.floor(n1 * n2 / 2.0 - z * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)))
    k = max(0, min(k, (n_pairs - 1) // 2))
    lower = float(diffs[k])
    upper = float(diffs[n_pairs - 1 - k])
    return estimate, lower, upper


def friedman_ranking_test(data_matrix: np.ndarray, algorithm_names: List[str]) -> Dict[str, Any]:
    """Perform the non-parametric Friedman test and compute average rankings.

    data_matrix: 2D array of shape (n_datasets, n_algorithms), lower is better.
    """
    data = np.asarray(data_matrix, dtype=float)
    n_datasets, n_algs = data.shape

    # Compute ranks per row (problem instance) - rank 1 is best (lowest value)
    ranks = np.zeros_like(data)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(data[i], method="average")

    avg_ranks = np.mean(ranks, axis=0)
    stat, p_value = stats.friedmanchisquare(*[data[:, j] for j in range(n_algs)])

    ranking_dict = {alg: float(avg_ranks[j]) for j, alg in enumerate(algorithm_names)}
    sorted_ranking = sorted(ranking_dict.items(), key=lambda x: x[1])

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_significant": bool(p_value < 0.05),
        "average_ranks": ranking_dict,
        "sorted_rankings": sorted_ranking,
    }
