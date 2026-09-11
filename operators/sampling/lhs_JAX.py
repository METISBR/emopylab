try:
    import jax.numpy as jnp
    _HAS_JAX = True
except Exception:
    jnp = None
    _HAS_JAX = False

import numpy as np
from core.sampling import Sampling
from operators.sampling.lhs import LatinHypercubeSampling as _LatinHypercubeSamplingCPU


class LatinHypercubeSampling(_LatinHypercubeSamplingCPU):
    """JAX-compatible Latin Hypercube Sampling operator."""

    def sample_array(self, problem, n_samples, *, random_state=None):
        X = super().sample_array(problem, n_samples, random_state=random_state)
        if _HAS_JAX and jnp is not None:
            return jnp.asarray(X)
        return X

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        return self.sample_array(problem, n_samples, random_state=random_state)


LatinHypercubeSampling_JAX = LatinHypercubeSampling
LHS_JAX = LatinHypercubeSampling
