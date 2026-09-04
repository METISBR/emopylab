"""EmoPyLab Initial Population Samplers (LHS / Uniform)."""

from __future__ import annotations

import numpy as np

from core.tensor.backend import to_device


def random_sampling(n_samples: int, n_var: int, xl: np.ndarray, xu: np.ndarray, seed: int = 42) -> Any:
    """Generates uniform random initial population matrix on device."""
    rng = np.random.default_rng(seed)
    rand_matrix = rng.random((n_samples, n_var), dtype=np.float32)
    X = xl + rand_matrix * (xu - xl)
    return to_device(X)


def latin_hypercube_sampling(n_samples: int, n_var: int, xl: np.ndarray, xu: np.ndarray, seed: int = 42) -> Any:
    """Generates stratified Latin Hypercube Sample (LHS) matrix on device."""
    rng = np.random.default_rng(seed)
    N = int(n_samples)
    D = int(n_var)

    result = np.empty((N, D), dtype=np.float32)
    for d in range(D):
        perm = rng.permutation(N)
        rand = rng.random(N, dtype=np.float32)
        result[:, d] = (perm + rand) / float(N)

    X = xl + result * (xu - xl)
    return to_device(X)
