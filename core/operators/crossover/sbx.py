"""EmoPyLab Vectorized SBX Crossover Operator in Pure Tensors."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.tensor.backend import clip_bounds, get_array_module, to_device


def sbx_crossover_tensor(
    parent1: Any,
    parent2: Any,
    xl: Any,
    xu: Any,
    eta: float = 15.0,
    prob: float = 0.9,
    prob_var: float = 0.5,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Computes Simulated Binary Crossover (SBX) in a single vectorized GPU tensor operation."""
    xp = get_array_module()
    N, D = parent1.shape
    is_torch = "torch" in type(parent1).__module__
    is_mlx = "mlx" in type(parent1).__module__

    if is_torch:
        import torch
        dev = parent1.device
        do_cross = torch.rand(N, device=dev) < prob
        do_cross_var = torch.rand((N, D), device=dev) < prob_var
        u = torch.rand((N, D), device=dev, dtype=torch.float32)
        xp = torch
    elif is_mlx:
        import mlx.core as mx
        do_cross = mx.random.uniform(0.0, 1.0, shape=(N,)) < prob
        do_cross_var = mx.random.uniform(0.0, 1.0, shape=(N, D)) < prob_var
        u = mx.random.uniform(0.0, 1.0, shape=(N, D))
        xp = mx
    else:
        xp = np
        rng = np.random.default_rng(seed)
        do_cross = to_device(rng.random(N) < prob, dtype=bool)
        do_cross_var = to_device(rng.random((N, D)) < prob_var, dtype=bool)
        u = to_device(rng.random((N, D)), dtype=np.float32)
    # Beta distribution calculation
    beta = xp.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u + 1e-8))) ** (1.0 / (eta + 1.0)),
    )

    c1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
    c2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

    # Apply masks
    mask = do_cross[:, None] & do_cross_var
    offspring1 = xp.where(mask, c1, parent1)
    offspring2 = xp.where(mask, c2, parent2)

    offspring1 = clip_bounds(offspring1, xl, xu)
    offspring2 = clip_bounds(offspring2, xl, xu)
    return offspring1, offspring2
