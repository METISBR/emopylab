"""Utility functions loader for EmoPyLab."""

import numpy as np


def calc_perpendicular_distance(N, ref_dirs):
    """Calculate perpendicular distance of points N to reference directions ref_dirs."""
    N = np.asarray(N, dtype=float)
    ref_dirs = np.asarray(ref_dirs, dtype=float)
    u = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    proj = np.dot(N, u.T)
    norm_N_sq = np.sum(N**2, axis=1, keepdims=True)
    dist_sq = np.maximum(0.0, norm_N_sq - proj**2)
    return np.sqrt(dist_sq)


def load_function(name):
    if name in ("calc_mnn", "calc_mnn_fast"):
        from operators.survival.rank_and_crowding.metrics import calc_mnn_fast
        return calc_mnn_fast
    elif name in ("calc_2nn", "calc_2nn_fast"):
        from operators.survival.rank_and_crowding.metrics import calc_2nn_fast
        return calc_2nn_fast
    elif name in ("calc_pcd", "calc_crowding_distance"):
        from operators.survival.rank_and_crowding.metrics import calc_crowding_distance
        return calc_crowding_distance
    elif name in ("calc_perpendicular_distance", "perpendicular_distance"):
        return calc_perpendicular_distance
    else:
        from operators.survival.rank_and_crowding.metrics import calc_crowding_distance
        return calc_crowding_distance
