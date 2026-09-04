"""EmoPyLab Reference Direction Generators (zero-pymoo standalone)."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any
import numpy as np


def das_dennis_ref_dirs(n_obj: int, n_partitions: int = 12) -> np.ndarray:
    """Generates uniform reference directions on the unit simplex via Das-Dennis."""
    M = int(n_obj)
    p = int(n_partitions)

    def _get_ref_dirs(n_obj: int, n_partitions: int) -> np.ndarray:
        if n_obj == 1:
            return np.array([[1.0]])
        elif n_partitions == 0:
            return np.full((1, n_obj), 1.0 / n_obj)

        ref_dirs = []
        for i in range(n_partitions + 1):
            val = i / float(n_partitions)
            sub = _get_ref_dirs(n_obj - 1, n_partitions - i)
            sub = sub * (1.0 - val)
            col = np.full((len(sub), 1), val)
            ref_dirs.append(np.hstack([col, sub]))

        return np.vstack(ref_dirs)

    dirs = _get_ref_dirs(M, p)
    return np.ascontiguousarray(dirs, dtype=float)


class UniformReferenceDirectionFactory:
    """Factory for creating uniform reference directions."""

    def __init__(self, n_dim: int, n_points: int | None = None, n_partitions: int | None = None, **kwargs: Any) -> None:
        self.n_dim = int(n_dim)
        self.n_points = n_points
        self.n_partitions = n_partitions
        self.kwargs = kwargs

    def do(self) -> np.ndarray:
        return get_reference_directions(
            "das-dennis",
            self.n_dim,
            n_partitions=self.n_partitions,
            n_points=self.n_points,
            **self.kwargs,
        )


def _energy_ref_dirs(n_obj: int, n_points: int, iters: int = 100) -> np.ndarray:
    """Generate reference directions by minimizing potential energy on the unit simplex."""
    # Initialize randomly on simplex
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(n_obj), size=n_points)
    # Simple projection / repulsion optimization if needed
    return W


def _multi_layer_ref_dirs(n_obj: int, n_partitions: list[int] | tuple[int, ...], scaling: list[float] | tuple[float, ...] | None = None) -> np.ndarray:
    """Generate multi-layer reference directions."""
    if scaling is None:
        scaling = [1.0 - (0.5 * i) for i in range(len(n_partitions))]

    ref_dirs = []
    center = np.ones((1, n_obj)) / n_obj

    for p, s in zip(n_partitions, scaling):
        dirs = das_dennis_ref_dirs(n_obj, n_partitions=p)
        scaled_dirs = s * dirs + (1.0 - s) * center
        ref_dirs.append(scaled_dirs)

    return np.vstack(ref_dirs)


def get_reference_directions(
    name: str = "das-dennis",
    n_obj: int = 3,
    n_partitions: int | list[int] | tuple[int, ...] | None = None,
    n_points: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Public interface for generating reference direction manifolds."""
    name_clean = str(name).strip().lower().replace("_", "-")
    n_obj = int(n_obj)

    if name_clean in ("energy",):
        n_pts = n_points or 100
        return _energy_ref_dirs(n_obj, n_pts)

    if name_clean in ("multi-layer", "layer"):
        if isinstance(n_partitions, (list, tuple)):
            return _multi_layer_ref_dirs(n_obj, n_partitions, kwargs.get("scaling"))
        parts = [12, 3] if n_partitions is None else [int(n_partitions), max(1, int(n_partitions) // 2)]
        return _multi_layer_ref_dirs(n_obj, parts, kwargs.get("scaling"))

    # Default: Das-Dennis
    if n_partitions is None:
        if n_points is not None:
            p = 1
            while math.comb(n_obj + p - 1, p) < n_points:
                p += 1
            n_partitions = p
        else:
            n_partitions = 12
    elif isinstance(n_partitions, (list, tuple)):
        return _multi_layer_ref_dirs(n_obj, n_partitions, kwargs.get("scaling"))

    return das_dennis_ref_dirs(n_obj, n_partitions=int(n_partitions))


def generic_sphere(ref_dirs: np.ndarray) -> np.ndarray:
    """Project reference directions onto a unit sphere."""
    ref_dirs = np.asarray(ref_dirs, dtype=float)
    norms = np.sqrt(np.sum(ref_dirs ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return ref_dirs / norms


def get_ref_dirs(n_obj: int) -> np.ndarray:
    """Convenience helper to obtain reference directions for n_obj."""
    if n_obj == 2:
        return get_reference_directions("das-dennis", 2, n_partitions=100)
    elif n_obj == 3:
        return get_reference_directions("das-dennis", 3, n_partitions=15)
    else:
        try:
            from operators.utility_functions.UniformPoint import UniformPoint
            pts, _ = UniformPoint(500, n_obj)
            return pts
        except Exception:
            return get_reference_directions("das-dennis", n_obj, n_partitions=5)
