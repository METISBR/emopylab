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
    at_least_once: bool = False,
) -> Any:
    """Vectorized Polynomial Mutation for native tensor solvers (not a drop-in mut_pm)."""
    xp = get_array_module()
    N, D = X.shape

    if prob_var is None:
        prob_var = 1.0 / float(D)

    is_torch = "torch" in type(X).__module__
    is_mlx = "mlx" in type(X).__module__

    if is_torch:
        import torch
        dev = X.device
        mutate_mask = torch.rand((N, D), device=dev) < prob_var
        u = torch.rand((N, D), device=dev, dtype=torch.float32)
        xp = torch
    elif is_mlx:
        import mlx.core as mx
        mutate_mask = mx.random.uniform(0.0, 1.0, shape=(N, D)) < prob_var
        u = mx.random.uniform(0.0, 1.0, shape=(N, D))
        xp = mx
    else:
        xp = np
        rng = np.random.default_rng(seed)
        mutate_mask = to_device(rng.random((N, D)) < prob_var, dtype=bool)
        u = to_device(rng.random((N, D)), dtype=np.float32)
    if is_torch:
        import torch
        delta_1 = torch.clamp((X - xl) / (xu - xl + 1e-8), 0.0, 1.0)
        delta_2 = torch.clamp((xu - X) / (xu - xl + 1e-8), 0.0, 1.0)
    elif is_mlx:
        import mlx.core as mx
        xl_m = xl if "mlx" in type(xl).__module__ else mx.array(xl)
        xu_m = xu if "mlx" in type(xu).__module__ else mx.array(xu)
        X_m = X if "mlx" in type(X).__module__ else mx.array(X)
        delta_1 = mx.clip((X_m - xl_m) / (xu_m - xl_m + 1e-8), 0.0, 1.0)
        delta_2 = mx.clip((xu_m - X_m) / (xu_m - xl_m + 1e-8), 0.0, 1.0)
    else:
        delta_1 = np.clip((X - xl) / (xu - xl + 1e-8), 0.0, 1.0)
        delta_2 = np.clip((xu - X) / (xu - xl + 1e-8), 0.0, 1.0)
    val = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - delta_1) ** (eta + 1.0))
    delta_q_left = (val ** (1.0 / (eta + 1.0))) - 1.0

    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - delta_2) ** (eta + 1.0))
    delta_q_right = 1.0 - (val ** (1.0 / (eta + 1.0)))

    delta_q = xp.where(u <= 0.5, delta_q_left, delta_q_right)

    if at_least_once:
        if is_torch:
            empty = ~mutate_mask.any(dim=1)
            mutate_mask[empty, 0] = True
        elif is_mlx:
            import mlx.core as mx  # noqa: F811
            empty = np.asarray((~mutate_mask.any(axis=1)))
            if bool(np.any(empty)):
                mutate_mask = mx.array(np.asarray(mutate_mask))
                mutate_mask[empty, 0] = True
        else:
            empty = ~mutate_mask.any(axis=1)
            mutate_mask[empty, 0] = True
    X_mut = X + delta_q * (xu - xl)
    X_out = xp.where(mutate_mask, X_mut, X)
    return clip_bounds(X_out, xl, xu)
