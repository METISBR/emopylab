from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.sampling import Sampling
from util import default_random_state

try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except Exception:
    qmc = None
    _HAS_SCIPY_QMC = False


def _latin_hypercube_numpy(
    n_samples: int,
    n_var: int,
    scramble: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact stratified Latin Hypercube Sampling in pure NumPy (McKay et al., 1979).
    Partitions each dimension into n_samples equiprobable strata.
    """
    if n_samples <= 0 or n_var <= 0:
        return np.empty((max(0, n_samples), max(0, n_var)), dtype=float)

    X = np.empty((n_samples, n_var), dtype=float)
    for j in range(n_var):
        perm = rng.permutation(n_samples)
        if scramble:
            offsets = rng.uniform(0.0, 1.0, size=n_samples)
        else:
            offsets = 0.5
        X[:, j] = (perm + offsets) / float(n_samples)
    return X


class LatinHypercubeSampling(Sampling):
    """
    Latin Hypercube Sampling (LHS) operator for continuous decision variables.
    Provides stratified space-filling initial populations with reduced discrepancy
    and lower variance than pseudo-random uniform sampling.
    """

    def __init__(
        self,
        scramble: bool = True,
        optimization: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scramble = bool(scramble)
        self.optimization = optimization
        self.seed = seed

    def sample_array(
        self,
        problem: Any,
        n_samples: int,
        *,
        random_state: Any = None,
    ) -> np.ndarray:
        """
        Generate a stratified Latin Hypercube decision matrix bounded by [xl, xu].
        Compatible with both population-based and hardware-accelerated algorithms.
        """
        n_samples = int(n_samples)
        n_var = int(problem.n_var)

        if n_samples <= 0 or n_var <= 0:
            return np.empty((max(0, n_samples), max(0, n_var)), dtype=float)

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif isinstance(random_state, (int, np.integer)):
            rng = np.random.default_rng(int(random_state))
        elif hasattr(random_state, "randint"):
            rng = np.random.default_rng(int(random_state.randint(0, 2**31 - 1)))
        elif self.seed is not None:
            rng = np.random.default_rng(int(self.seed))
        else:
            rng = np.random.default_rng()

        X = None
        if _HAS_SCIPY_QMC:
            try:
                sampler = qmc.LatinHypercube(
                    d=n_var,
                    scramble=self.scramble,
                    optimization=self.optimization,
                    seed=rng,
                )
                X = sampler.random(n=n_samples)
            except Exception:
                X = None

        if X is None:
            X = _latin_hypercube_numpy(n_samples, n_var, self.scramble, rng)

        if problem.has_bounds():
            xl, xu = problem.bounds()
            xl = np.asarray(xl, dtype=float)
            xu = np.asarray(xu, dtype=float)
            assert np.all(xu >= xl), "Upper bounds must be greater than or equal to lower bounds."
            X = xl + (xu - xl) * X

        return X

    def _do(
        self,
        problem: Any,
        n_samples: int,
        *args: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.sample_array(problem, n_samples, random_state=random_state)


LHS = LatinHypercubeSampling
LatinHypercube = LatinHypercubeSampling
