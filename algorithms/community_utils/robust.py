"""Helpers shared by the robust multi-objective algorithms (uncharged noise sampling of a robust problem)."""

from __future__ import annotations

import numpy as np

__all__ = ["perturbed_objs", "cosine_dist"]


def perturbed_objs(problem, X, n_perturb=None, rng=None) -> np.ndarray:
    """Objective values of ``n_perturb`` disturbed copies of every row of ``X`` -> array ``(n_perturb, n, M)``.

    Disturbed evaluations estimate robustness and are not charged to the evaluation budget."""
    if not hasattr(problem, "perturb"):
        raise ValueError(f"{type(problem).__name__} is not a robust problem (no `perturb` method)")
    F, _ = problem.perturb(np.atleast_2d(np.asarray(X, dtype=float)), n_perturb, rng=rng)
    return np.asarray(F, dtype=float)


def cosine_dist(A, B) -> np.ndarray:
    """Pairwise ``1 - cos`` distance; rows with zero norm give NaN (as in the reference distance routine)."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        return 1.0 - (A @ B.T) / (na[:, None] * nb[None, :])
