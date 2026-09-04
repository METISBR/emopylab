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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

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

    dev = abs(a12 - 0.5)
    if dev < 0.06:
        mag = "negligible"
    elif dev < 0.14:
        mag = "small"
    elif dev < 0.21:
        mag = "medium"
    else:
        mag = "large"
    return a12, mag


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply the step-down Holm-Bonferroni multiple-comparison correction.

    Returns list of (adjusted_p_value, is_significant) tuples preserving original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_indexed = []

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        k = m - rank
        adjusted = min(1.0, p_val * k)
        running_max = max(running_max, adjusted)
        is_sig = running_max < alpha
        adjusted_indexed.append((orig_idx, running_max, is_sig))

    # Restore original index order
    adjusted_indexed.sort(key=lambda x: x[0])
    return [(adj, sig) for _, adj, sig in adjusted_indexed]


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
