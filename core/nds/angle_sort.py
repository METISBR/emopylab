"""EmoPyLab Angle-Penalized Distance (APD) and Strengthened Dominance (SDR) Sorting.

Provides high-dimensional many-objective selection mechanisms:
1. Angle-Penalized Distance (APD - RVEA, Cheng et al., 2016)
2. Strengthened Dominance Relation (SDR - Tian et al., 2018)
3. Reference Vector Association via Tensorized Cosine Similarity
"""

from __future__ import annotations

import numpy as np

from core.tensor.backend import get_array_module, to_device, to_numpy


def associate_to_reference_vectors(
    F_matrix: np.ndarray,
    ref_dirs: np.ndarray,
    ideal_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associates each candidate solution to its closest reference vector by acute angle.

    Args:
        F_matrix: Objective vectors [N, M].
        ref_dirs: Unit reference directions [K, M].
        ideal_point: Optional ideal reference point [M].

    Returns:
        Tuple (associations, cos_angles, distances):
            - associations: Index of closest reference direction for each solution [N].
            - cos_angles: Cosine of angle to associated vector [N].
            - distances: Euclidean norm ||F'(x)|| from ideal point [N].
    """
    F = np.asarray(F_matrix, dtype=np.float64)
    V = np.asarray(ref_dirs, dtype=np.float64)
    N, M = F.shape
    K = V.shape[0]

    if ideal_point is None:
        z_min = np.min(F, axis=0)
    else:
        z_min = np.asarray(ideal_point, dtype=np.float64)

    # Translated objectives
    F_trans = F - z_min
    distances = np.linalg.norm(F_trans, axis=1)

    # Normalize vectors
    norm_F = np.where(distances[:, None] > 1e-12, F_trans / distances[:, None], 0.0)
    norm_V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    # Cosine similarity matrix [N, K]
    cosine_matrix = np.dot(norm_F, norm_V.T)
    cosine_matrix = np.clip(cosine_matrix, -1.0, 1.0)

    # Closest reference vector has maximum cosine (minimum angle)
    associations = np.argmax(cosine_matrix, axis=1)
    cos_angles = np.max(cosine_matrix, axis=1)

    return associations, cos_angles, distances


def calculate_apd(
    F_matrix: np.ndarray,
    ref_dirs: np.ndarray,
    current_gen: int,
    max_gen: int,
    alpha: float = 2.0,
    ideal_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates Angle-Penalized Distance (APD) for RVEA selection on device/host.

    APD(x, v) = (1 + P(t) * (theta / gamma_v)) * ||F'(x)||
    where P(t) = M * (t / t_max)^alpha
    """
    F = np.asarray(F_matrix, dtype=np.float64)
    V = np.asarray(ref_dirs, dtype=np.float64)
    N, M = F.shape
    K = V.shape[0]

    associations, cos_angles, distances = associate_to_reference_vectors(F, V, ideal_point=ideal_point)

    # Acute angles in radians
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    # Calculate smallest angle among reference vectors (gamma)
    norm_V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    v_cosine = np.dot(norm_V, norm_V.T)
    np.fill_diagonal(v_cosine, -1.0)
    gamma = np.arccos(np.clip(np.max(v_cosine, axis=1), -1.0, 1.0))
    gamma = np.maximum(gamma, 1e-6)

    # Progression penalty P(t)
    t = float(max(1, current_gen))
    t_max = float(max(1, max_gen))
    penalty_factor = float(M) * ((t / t_max) ** alpha)

    # Assigned gamma per solution
    gamma_assigned = gamma[associations]
    apd_scores = (1.0 + penalty_factor * (angles / gamma_assigned)) * distances

    return apd_scores, associations


def strengthened_dominance_relation(
    F_matrix: np.ndarray,
    ref_dirs: np.ndarray,
    ideal_point: np.ndarray | None = None,
) -> np.ndarray:
    """Computes Strengthened Dominance Relation (SDR) matrix [N, N] for Many-Objective regimes.

    Solution i SDR-dominates j if:
    1. i is associated to same niche as j and ||F'(i)|| < ||F'(j)||, OR
    2. Pareto dominance holds: i dominates j in all objectives.
    """
    F = np.asarray(F_matrix, dtype=np.float64)
    N, M = F.shape

    associations, _, distances = associate_to_reference_vectors(F, ref_dirs, ideal_point=ideal_point)

    # 1. Standard Pareto dominance
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    pareto_dom = np.all(le, axis=2) & np.any(lt, axis=2)

    # 2. Niche convergence dominance
    same_niche = associations[:, None] == associations[None, :]
    closer_distance = distances[:, None] < (distances[None, :] - 1e-8)
    niche_dom = same_niche & closer_distance

    # Combined SDR dominance matrix
    sdr_matrix = pareto_dom | niche_dom
    return sdr_matrix
