"""EmoPyLab Reference Direction Generators in Pure Tensors.

Provides Das-Dennis Simplex Lattice and Multi-layer Reference Direction
generation without external dependencies.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.tensor.backend import to_device


def das_dennis_ref_dirs(n_obj: int, n_partitions: int = 12) -> np.ndarray:
    """Generates uniform reference directions on the unit simplex via Das-Dennis."""
    M = int(n_obj)
    p = int(n_partitions)

    def _get_ref_dirs(n_obj, n_partitions):
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
    return np.ascontiguousarray(dirs, dtype=np.float32)


def get_reference_directions(
    name: str = "das-dennis",
    n_obj: int = 3,
    n_partitions: int = 12,
    n_points: int | None = None,
) -> np.ndarray:
    """Public interface for generating reference direction manifolds."""
    if n_points is not None and n_partitions is None:
        # Approximate partitions from target points
        p = 1
        while math.comb(n_obj + p - 1, p) < n_points:
            p += 1
        n_partitions = p

    return das_dennis_ref_dirs(n_obj, n_partitions=n_partitions or 12)
