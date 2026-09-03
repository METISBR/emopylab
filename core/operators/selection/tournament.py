"""EmoPyLab Vectorized Tournament and Environmental Selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.tensor.backend import to_device, to_numpy


def binary_tournament_selection(
    ranks: np.ndarray,
    crowdings: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> np.ndarray:
    """Executes vectorized binary tournament based on Pareto rank and crowding distance."""
    n_pop = len(ranks)
    rng = np.random.default_rng(seed)

    i1 = rng.integers(0, n_pop, size=n_select)
    i2 = rng.integers(0, n_pop, size=n_select)

    r1, r2 = ranks[i1], ranks[i2]
    c1, c2 = crowdings[i1], crowdings[i2]

    # Winner: lower rank wins; if rank equal, higher crowding wins
    win1 = (r1 < r2) | ((r1 == r2) & (c1 > c2))
    selected = np.where(win1, i1, i2)
    return np.ascontiguousarray(selected, dtype=np.int64)
