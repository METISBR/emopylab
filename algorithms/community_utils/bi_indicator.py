"""Pareto-based bi-indicator infill criterion (convergence vs. diversity indicators) and per-objective Kriging training."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import decs, nd_sort, objs
from algorithms.community_utils.dace import DaceModel

__all__ = ["train_models", "bi_indicator_select"]


def train_models(A, theta, regr="regpoly0"):
    """Fit one Kriging model per objective on the distinct (1e-6 rounded) archive rows; ``theta`` (M x D) is updated in place."""
    X, Y = decs(A), objs(A)
    distinct = np.unique(np.round(X * 1e6) / 1e6, axis=0, return_index=True)[1]
    X, Y = X[distinct], Y[distinct]
    D = X.shape[1]
    models = []
    for i in range(Y.shape[1]):
        dm = DaceModel(X, Y[:, i], regr, theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
        models.append(dm)
        theta[i] = dm.theta
    return models


def bi_indicator_select(Dec, Obj, DA_obj):
    """Unique decision vectors of the first front of (diversity = -distance to nearest evaluated, convergence = distance to ideal)."""
    allo = np.vstack([Obj, DA_obj])
    lo, hi = allo.min(axis=0), allo.max(axis=0)
    with np.errstate(all="ignore"):
        da_nor = (DA_obj - lo) / (hi - lo)
        pre = (Obj - lo) / (hi - lo)
    zmin = np.vstack([da_nor, pre]).min(axis=0)
    dist = np.sqrt(((pre[:, None, :] - da_nor[None, :, :]) ** 2).sum(axis=2))
    di = -dist.min(axis=1)
    ci = np.sqrt(((pre - zmin) ** 2).sum(axis=1))
    front, _ = nd_sort(np.column_stack([di, ci]), None, 1)
    return np.unique(Dec[front == 1], axis=0)
