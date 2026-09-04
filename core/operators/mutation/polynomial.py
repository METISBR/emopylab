"""EmoPyLab Vectorized Polynomial Mutation Operator in Pure Tensors."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.tensor.backend import clip_bounds, get_array_module, to_device


def polynomial_mutation_tensor(
    X: Any,
    xl: Any,
    xu: Any,
    eta: float = 20.0,
    prob_var: float | None = None,
    seed: int = 42,
) -> Any:
    """Computes Polynomial Mutation (PM) in a single vectorized GPU tensor operation."""
    xp = get_array_module()
    N, D = X.shape

    if prob_var is None:
        prob_var = 1.0 / float(D)

    rng = np.random.default_rng(seed)
    mutate_mask = to_device(rng.random((N, D)) < prob_var, dtype=bool)
    u = to_device(rng.random((N, D)), dtype=np.float32)

    delta_1 = (X - xl) / (xu - xl + 1e-8)
    delta_2 = (xu - X) / (xu - xl + 1e-8)

    val = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - delta_1) ** (eta + 1.0))
    delta_q_left = (val ** (1.0 / (eta + 1.0))) - 1.0

    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - delta_2) ** (eta + 1.0))
    delta_q_right = 1.0 - (val ** (1.0 / (eta + 1.0)))

    delta_q = xp.where(u <= 0.5, delta_q_left, delta_q_right)

    X_mut = X + delta_q * (xu - xl)
    X_out = xp.where(mutate_mask, X_mut, X)
    return clip_bounds(X_out, xl, xu)
