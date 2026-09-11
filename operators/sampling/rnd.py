import numpy as np

from core.sampling import Sampling
from util import default_random_state


@default_random_state
def random(problem, n_samples=1, random_state=None):
    X = random_state.random((n_samples, problem.n_var))

    if problem.has_bounds():
        xl, xu = problem.bounds()
        assert np.all(xu >= xl)
        X = xl + (xu - xl) * X

    return X


class FloatRandomSampling(Sampling):

    def sample_array(self, problem, n_samples, *, random_state) -> np.ndarray:
        """Generate a uniform floating-point decision matrix."""
        X = random_state.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        return X

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        return self.sample_array(problem, n_samples, random_state=random_state)


class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        val = random_state.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.column_stack([random_state.integers(xl[k], xu[k] + 1, size=n_samples) for k in range(n)])


class PermutationRandomSampling(Sampling):

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = random_state.permutation(problem.n_var)
        return X
