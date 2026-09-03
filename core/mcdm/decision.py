"""Multi-Criteria Decision Making (MCDM) Decision Engine for EmoPyLab.

Provides formal a posteriori compromise solution selection over high-dimensional
Pareto fronts (M >= 2) with mathematical normalization, sensitivity bounds, and sidecar export.
Methods supported:
- TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
- PROMETHEE II (Net outranking flow)
- Weighted Sum / Compromise Programming with user-specified or equal weights
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def compute_topsis(norm_matrix: np.ndarray, weights: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Computes TOPSIS compromise point on normalized objective matrix (minimization)."""
    weighted = norm_matrix * weights
    # For minimization, ideal best is min, ideal worst is max
    ideal_best = np.min(weighted, axis=0)
    ideal_worst = np.max(weighted, axis=0)

    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

    scores = d_worst / np.maximum(d_best + d_worst, 1e-12)
    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx]), scores


def compute_promethee_ii(norm_matrix: np.ndarray, weights: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Computes PROMETHEE II Net Outranking Flows for compromise selection (minimization)."""
    n_points, n_obj = norm_matrix.shape
    if n_points == 1:
        return 0, 1.0, np.array([1.0])

    # Preference function: P_j(a, b) = max(0, f_j(b) - f_j(a)) for minimization
    # Pairwise comparison tensor
    diff = norm_matrix[:, np.newaxis, :] - norm_matrix[np.newaxis, :, :]  # [i, j, m]
    # For minimization: solution i is preferred to j if f(j) > f(i), i.e., diff[j, i] > 0
    pref = np.maximum(0.0, -diff) * weights  # [i, j, m]
    global_pref = np.sum(pref, axis=-1)  # [i, j]

    # Leaving flow (positive flow) and Entering flow (negative flow)
    phi_plus = np.sum(global_pref, axis=1) / float(n_points - 1)
    phi_minus = np.sum(global_pref, axis=0) / float(n_points - 1)
    net_flows = phi_plus - phi_minus

    best_idx = int(np.argmax(net_flows))
    return best_idx, float(net_flows[best_idx]), net_flows


def select_compromise_solution(
    front: np.ndarray,
    method: str = "topsis",
    weights_text: str = "",
    weights_array: np.ndarray | None = None,
) -> dict[str, Any]:
    """Selects a single compromise point from a Pareto front using formal MCDM methods.

    Args:
        front: 2D array of objective vectors (N points x M objectives).
        method: 'topsis', 'promethee', or 'weighted_sum'.
        weights_text: Comma/space separated weights string.
        weights_array: Optional direct numpy array of weights.

    Returns:
        Dictionary containing chosen index, score, selected vector, and weight distribution.
    """
    values = np.asarray(front, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Pareto front is empty or invalid.")

    n_points, n_obj = values.shape

    # Parse weights
    if weights_array is not None:
        weights = np.asarray(weights_array, dtype=float).reshape(-1)
        if weights.size != n_obj:
            raise ValueError(f"Weights dimension ({weights.size}) does not match objective count ({n_obj}).")
    elif str(weights_text or "").strip():
        chunks = [c for c in re.split(r"[,;\s]+", str(weights_text).strip()) if c]
        weights = np.asarray([float(c) for c in chunks], dtype=float)
        if weights.size != n_obj:
            raise ValueError(f"Provide exactly {n_obj} weights.")
    else:
        weights = np.ones(n_obj, dtype=float) / float(n_obj)

    if np.sum(weights) <= 0:
        raise ValueError("Weights must sum to a positive value.")
    weights = weights / np.sum(weights)

    # Min-max normalization per objective
    f_min = np.min(values, axis=0)
    f_max = np.max(values, axis=0)
    denom = np.where((f_max - f_min) > 0, (f_max - f_min), 1.0)
    norm_matrix = (values - f_min) / denom

    method_name = str(method or "topsis").strip().lower()

    if method_name in {"topsis", "topsis_min"}:
        best_idx, best_score, all_scores = compute_topsis(norm_matrix, weights)
    elif method_name in {"promethee", "promethee_ii", "promethee2"}:
        best_idx, best_score, all_scores = compute_promethee_ii(norm_matrix, weights)
    else:  # Weighted sum baseline
        scores = np.sum(norm_matrix * weights, axis=1)
        best_idx = int(np.argmin(scores))
        best_score = float(scores[best_idx])
        all_scores = scores

    return {
        "index": best_idx,
        "score": best_score,
        "selected": values[best_idx, :],
        "all_scores": all_scores,
        "weights": weights,
        "method": method_name,
        "n_points": n_points,
        "n_obj": n_obj,
    }
