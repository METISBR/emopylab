"""Determinantal point process sampling (k-DPP by the eigen-decomposition algorithm; no extra packages)."""

from __future__ import annotations

import numpy as np

__all__ = ["decompose_kernel", "sample_dpp"]


def decompose_kernel(M):
    """Eigenvalues (ascending) and eigenvectors of the symmetric kernel ``M``."""
    M = np.asarray(M, dtype=float)
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    return V, w


def _orth(V):
    """Orthonormal basis of the column space (SVD, numerical rank as in the usual ``orth``)."""
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    tol = max(V.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
    return U[:, : int(np.sum(s > tol))]


def sample_dpp(V, w, k):
    """Indices (0-based, ascending) of a ``k``-element sample: the eigenvectors of the ``k`` largest eigenvalues are
    reduced one dimension at a time, each time picking the item with the largest remaining mass."""
    n = len(w)
    idx = np.argsort(w, kind="stable")[n - k:]
    k = len(idx)
    V = V[:, idx]
    Y = np.zeros(k, dtype=int)
    for i in range(k - 1, -1, -1):
        if V.shape[1] == 1 and i != 0:
            v = np.abs(V[:, 0])
            top = np.argsort(v, kind="stable")[n - (i + 1):]
            Y[: i + 1] = top
            break
        P = np.sum(V ** 2, axis=1)
        P = P / P.sum()
        Y[i] = int(np.argmax(P))
        nz = np.where(V[Y[i]] != 0)[0]
        j = int(nz[0]) if len(nz) else int(np.argmax(np.abs(V[Y[i]])))
        Vj = V[:, j]
        V = np.delete(V, j, axis=1)
        V = V - Vj[:, None] * (V[Y[i]] / Vj[Y[i]])[None, :]
        V = _orth(V)
    return np.sort(Y)
