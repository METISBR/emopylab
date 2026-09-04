"""Stochastic Steepest Weights (SSW) for vector optimization.

This module implements the drift--diffusion method introduced by Schaeffler,
Schultz, and Weinzierl (JOTA, 2002).  The default update is the projected
Euler--Maruyama step

    x_next = projection(x - step_size * q(x)
                        + epsilon * sqrt(step_size) * normal_noise).

The common descent vector ``q(x)`` is the minimum-norm element of the convex
hull of the objective gradients.  Jacobians can be supplied by the caller,
computed with optional JAX automatic differentiation, or approximated with
centered finite differences.  Independent trajectories remain in the working
population; a separate bounded non-dominated archive is truncated by crowding
distance to preserve the final front coverage.

References
----------
S. Schaeffler, R. Schultz, and K. Weinzierl (2002), "Stochastic Method
for the Solution of Unconstrained Vector Optimization Problems," Journal of
Optimization Theory and Applications, 114(1), 209--222.

J. Blank and K. Deb (2020), "pymoo: Multi-Objective Optimization in Python,"
IEEE Access, 8, 89497--89509.
"""

from util.array_backend import xp as np

from core.algorithm import Algorithm
from core.population import Population
from operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from util.display.multi import MultiObjectiveOutput
from util.optimum import filter_optimum


ALGORITHM_FLAGS = {
    "SSW": {"multi", "many"},
}


def _force_cpu_backend(algorithm, reason: str) -> None:
    """Keep the current Jacobian and QP implementation on the CPU backend."""
    if not bool(getattr(algorithm, "use_gpu", False)):
        return
    algorithm.use_gpu = False
    algorithm.array_backend_effective = "numpy"
    state = dict(getattr(algorithm, "backend_state", {}) or {})
    state["forced_cpu_reason"] = str(reason)
    algorithm.backend_state = state


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Return the Euclidean projection of ``v`` onto the probability simplex."""
    x = np.asarray(v, dtype=float)
    n = x.size
    if n == 1:
        return np.array([1.0], dtype=float)

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / idx > 0.0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=float)

    rho = int(np.where(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(x - theta, 0.0)
    total = float(np.sum(w))
    if total <= 1e-16:
        return np.full(n, 1.0 / n, dtype=float)
    return w / total


def _solve_simplex_qp(
    gram: np.ndarray,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve ``min alpha.T @ gram @ alpha`` over the probability simplex."""
    m = gram.shape[0]
    if m == 1:
        return np.array([1.0], dtype=float)

    alpha = np.full(m, 1.0 / m, dtype=float)
    lipschitz = float(np.linalg.norm(gram, ord=np.inf))
    step = 1.0 / max(lipschitz, 1e-12)

    for _ in range(max_iter):
        previous = alpha.copy()
        alpha = _project_to_simplex(alpha - step * (gram @ alpha))
        if float(np.linalg.norm(alpha - previous)) <= tol:
            break
    return alpha


def _project_to_simplex_batch(v: np.ndarray) -> np.ndarray:
    """Project each row of a matrix onto the probability simplex."""
    values = np.asarray(v, dtype=float)
    if values.shape[1] == 1:
        return np.ones_like(values)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=float)
    positive = ordered - cumulative / indices[np.newaxis, :] > 0.0
    rho = np.maximum(np.sum(positive, axis=1) - 1, 0).astype(int)
    rows = np.arange(values.shape[0])
    theta = cumulative[rows, rho] / (rho + 1.0)
    projected = np.maximum(values - theta[:, np.newaxis], 0.0)
    total = np.sum(projected, axis=1, keepdims=True)
    return projected / np.maximum(total, 1e-16)


def _compute_q_batch(
    jacobians: np.ndarray,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> np.ndarray:
    """Vectorized minimum-norm convex combinations for one population."""
    batch = np.asarray(jacobians, dtype=float)
    gram = batch @ np.swapaxes(batch, 1, 2)
    n_points, n_obj, _ = batch.shape
    alpha = np.full((n_points, n_obj), 1.0 / n_obj, dtype=float)
    lipschitz = np.max(np.sum(np.abs(gram), axis=2), axis=1)
    step = 1.0 / np.maximum(lipschitz, 1e-12)

    for _ in range(max_iter):
        previous = alpha
        gradient = np.einsum("bij,bj->bi", gram, alpha)
        alpha = _project_to_simplex_batch(alpha - step[:, np.newaxis] * gradient)
        if float(np.max(np.linalg.norm(alpha - previous, axis=1))) <= tol:
            break
    return np.einsum("bmn,bm->bn", batch, alpha)


def _compute_q(jacobian: np.ndarray) -> np.ndarray:
    """Return the unique minimum-norm common descent vector.

    The optimal weight vector need not be unique when objective gradients are
    affinely dependent.  Its image ``jacobian.T @ alpha``, however, is the
    Euclidean projection of the origin onto the convex hull of the gradients
    and is therefore unique.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    gram = jacobian @ jacobian.T
    alpha_star = _solve_simplex_qp(gram)
    return jacobian.T @ alpha_star


class SSW(Algorithm):
    """Projected Stochastic Steepest Weights optimizer.

    Parameters
    ----------
    n_points
        Number of independent stochastic trajectories.
    step_size
        Euler--Maruyama step size.
    epsilon
        Diffusion intensity.
    jac
        Optional analytic or externally generated Jacobian.  A vectorized
        callable may accept ``X`` with shape ``(N, n)`` and return
        ``(N, m, n)``; a pointwise callable may accept ``x`` and return
        ``(m, n)``.
    jacobian_mode
        ``"auto"`` selects ``jac`` when supplied, then ``autodiff_func`` when
        supplied, and otherwise centered finite differences.  Explicit values
        are ``"provided"``, ``"autodiff"``, and ``"finite_difference"``.
    autodiff_func
        JAX-compatible pointwise vector objective used only in automatic-
        differentiation mode.
    archive_size
        Maximum size of the external non-dominated archive.  Defaults to
        ``n_points``.
    adaptive_step
        If true, use the legacy two-half-step error controller.  The default
        false value executes the Euler--Maruyama iteration stated in the paper.
    strict_evaluation_budget
        Stop before a generation whose objective and finite-difference calls
        would exceed an ``n_eval`` termination budget.  The default is true.
    """

    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    _JACOBIAN_MODES = {
        "auto",
        "provided",
        "autodiff",
        "finite_difference",
    }

    def __init__(
        self,
        n_points: int = 100,
        step_size: float = 0.01,
        epsilon: float = 0.15,
        delta: float = 0.1,
        jac=None,
        jacobian_mode: str = "auto",
        autodiff_func=None,
        archive_size: int | None = None,
        adaptive_step: bool = False,
        max_halvings: int = 3,
        finite_difference_step: float = 1e-7,
        strict_evaluation_budget: bool = True,
        output=MultiObjectiveOutput(),
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(
            output=output,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )
        _force_cpu_backend(
            self,
            "SSW Jacobian and simplex-QP paths currently use the CPU backend.",
        )

        if int(n_points) < 1:
            raise ValueError("n_points must be positive")
        if float(step_size) <= 0.0:
            raise ValueError("step_size must be positive")
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be non-negative")
        if archive_size is not None and int(archive_size) < 1:
            raise ValueError("archive_size must be positive")
        if float(finite_difference_step) <= 0.0:
            raise ValueError("finite_difference_step must be positive")

        mode = str(jacobian_mode).strip().lower()
        if mode not in self._JACOBIAN_MODES:
            allowed = ", ".join(sorted(self._JACOBIAN_MODES))
            raise ValueError(f"jacobian_mode must be one of: {allowed}")
        if mode == "provided" and jac is None:
            raise ValueError("jacobian_mode='provided' requires jac")
        if mode == "autodiff" and autodiff_func is None:
            raise ValueError("jacobian_mode='autodiff' requires autodiff_func")

        self.n_points = int(n_points)
        self.step_size = float(step_size)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.jac = jac
        self.jacobian_mode = mode
        self.autodiff_func = autodiff_func
        self.archive_size = int(archive_size or n_points)
        self.adaptive_step = bool(adaptive_step)
        self.max_halvings = max(0, int(max_halvings))
        self.finite_difference_step = float(finite_difference_step)
        self.strict_evaluation_budget = bool(strict_evaluation_budget)

        self.sigma = np.full(self.n_points, self.step_size, dtype=float)
        # ``Algorithm.archive`` is reserved by EmoPyLab for Archive objects with
        # an ``add`` method.  SSW keeps its Population-based front archive in a
        # separate attribute so the framework does not mutate it implicitly.
        self.ssw_archive = None
        self._jax_batch_jacobian = None
        self.jacobian_mode_effective = None

    def _setup(self, problem, **kwargs):
        """Validate problem metadata before initialization."""
        if int(problem.n_var) < 1 or int(problem.n_obj) < 1:
            raise ValueError("SSW requires at least one variable and objective")
        self.jacobian_mode_effective = self._resolve_jacobian_mode()
    def _initialize_infill(self):
        """Sample the initial trajectories uniformly inside the box bounds."""
        if not self.problem.has_bounds():
            raise ValueError("The current SSW implementation requires box bounds")
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        random = self.random_state.random((self.n_points, self.problem.n_var))
        return Population.new("X", lower + random * (upper - lower))

    def _initialize_advance(self, infills=None, **kwargs):
        """Store the evaluated trajectories and initialize the archive."""
        self.pop = infills
        self.ssw_archive = self._truncate_archive(
            filter_optimum(infills, least_infeasible=True)
        )

    def _resolve_jacobian_mode(self) -> str:
        if self.jacobian_mode != "auto":
            return self.jacobian_mode
        if self.jac is not None:
            return "provided"
        if self.autodiff_func is not None:
            return "autodiff"
        return "finite_difference"

    def _provided_jacobian(self, X: np.ndarray) -> np.ndarray:
        try:
            batch = self.jac(X)
            batch = np.asarray(batch, dtype=float)
            if batch.shape == (len(X), self.problem.n_obj, self.problem.n_var):
                return batch
        except (TypeError, ValueError):
            pass
        batch = np.asarray([self.jac(x) for x in X], dtype=float)
        expected = (len(X), self.problem.n_obj, self.problem.n_var)
        if batch.shape != expected:
            raise ValueError(f"jac returned {batch.shape}; expected {expected}")
        return batch

    def _autodiff_jacobian(self, X: np.ndarray) -> np.ndarray:
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                "JAX is required for jacobian_mode='autodiff'; install the "
                "optional JAX profile or provide jac explicitly"
            ) from exc

        if self._jax_batch_jacobian is None:
            # Forward mode needs one tangent sweep per variable, whereas reverse
            # mode needs one pullback per objective.  Choose the cheaper side of
            # the full-Jacobian calculation from the problem dimensions.
            transform = (
                jax.jacfwd
                if int(self.problem.n_var) <= int(self.problem.n_obj)
                else jax.jacrev
            )
            pointwise = transform(self.autodiff_func)
            self._jax_batch_jacobian = jax.jit(jax.vmap(pointwise))

        batch = self._jax_batch_jacobian(jnp.asarray(X))
        return np.asarray(batch, dtype=float)

    def _finite_difference_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Approximate Jacobians without evaluating outside the box.

        Centered differences are used in the interior.  At a bound, clipping
        automatically produces the corresponding one-sided difference.  This
        keeps every charged function evaluation feasible and makes the
        derivative path consistent with the projected numerical dynamics.
        """
        n_points, n_var = X.shape
        n_obj = int(self.problem.n_obj)
        h = self.finite_difference_step

        replicated = np.tile(X[:, np.newaxis, :], (1, n_var, 1))
        perturbation = np.eye(n_var) * h
        plus_batch = replicated + perturbation[np.newaxis, :, :]
        minus_batch = replicated - perturbation[np.newaxis, :, :]
        plus_batch = np.clip(plus_batch, self.problem.xl, self.problem.xu)
        minus_batch = np.clip(minus_batch, self.problem.xl, self.problem.xu)
        plus = plus_batch.reshape(-1, n_var)
        minus = minus_batch.reshape(-1, n_var)

        pop_plus = Population.new("X", plus)
        pop_minus = Population.new("X", minus)
        self.evaluator.eval(self.problem, pop_plus)
        self.evaluator.eval(self.problem, pop_minus)

        f_plus = pop_plus.get("F").reshape(n_points, n_var, n_obj)
        f_minus = pop_minus.get("F").reshape(n_points, n_var, n_obj)
        coordinate = np.arange(n_var)
        denominator = (
            plus_batch[:, coordinate, coordinate]
            - minus_batch[:, coordinate, coordinate]
        )
        if np.any(denominator <= 0.0):
            raise ValueError("finite differences require non-degenerate bounds")
        return ((f_plus - f_minus) / denominator[:, :, np.newaxis]).transpose(
            0, 2, 1
        )

    def _compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute a batch of Jacobians with the selected derivative source."""
        mode = self._resolve_jacobian_mode()
        self.jacobian_mode_effective = mode
        if mode == "provided":
            return self._provided_jacobian(X)
        if mode == "autodiff":
            return self._autodiff_jacobian(X)
        return self._finite_difference_jacobian(X)

    @staticmethod
    def _descent_batch(jacobians: np.ndarray) -> np.ndarray:
        return _compute_q_batch(jacobians)

    def _project_bounds(self, X: np.ndarray) -> np.ndarray:
        if not self.problem.has_bounds():
            return X
        return np.clip(X, self.problem.xl, self.problem.xu)

    def _euler_maruyama_infill(self, X: np.ndarray) -> np.ndarray:
        jacobians = self._compute_jacobian(X)
        descent = self._descent_batch(jacobians)
        noise = self.random_state.standard_normal(X.shape)
        next_x = (
            X
            - self.step_size * descent
            + self.epsilon * np.sqrt(self.step_size) * noise
        )
        return self._project_bounds(next_x)

    def _adaptive_infill(self, X: np.ndarray) -> np.ndarray:
        """Execute the optional legacy two-half-step error controller."""
        n_points, n_var = X.shape
        done = np.zeros(n_points, dtype=bool)
        next_x = np.empty_like(X)
        descent = self._descent_batch(self._compute_jacobian(X))
        halvings = np.zeros(n_points, dtype=int)

        while not np.all(done):
            active = np.where(~done)[0]
            x_active = X[active]
            q_active = descent[active]
            sigma = self.sigma[active, np.newaxis]
            half_noise_scale = np.sqrt(sigma / 2.0)

            noise_1 = self.random_state.standard_normal((len(active), n_var))
            noise_2 = self.random_state.standard_normal((len(active), n_var))
            full = (
                x_active
                - sigma * q_active
                + self.epsilon * (noise_1 + noise_2) * half_noise_scale
            )
            midpoint = (
                x_active
                - 0.5 * sigma * q_active
                + self.epsilon * noise_1 * half_noise_scale
            )
            q_midpoint = self._descent_batch(self._compute_jacobian(midpoint))
            two_half_steps = (
                midpoint
                - 0.5 * sigma * q_midpoint
                + self.epsilon * noise_2 * half_noise_scale
            )

            accept = np.linalg.norm(full - two_half_steps, axis=1) < self.delta
            accepted = active[accept]
            next_x[accepted] = two_half_steps[accept]
            done[accepted] = True

            rejected = active[~accept]
            if len(rejected) > 0:
                self.sigma[rejected] /= 2.0
                halvings[rejected] += 1
                forced_mask = halvings[rejected] >= self.max_halvings
                forced = rejected[forced_mask]
                if len(forced) > 0:
                    next_x[forced] = two_half_steps[~accept][forced_mask]
                    done[forced] = True

        return self._project_bounds(next_x)

    def _infill(self):
        X = np.asarray(self.pop.get("X"), dtype=float)
        if self.strict_evaluation_budget and not self._generation_fits_budget(len(X)):
            self.termination.terminate()
            return None
        next_x = (
            self._adaptive_infill(X)
            if self.adaptive_step
            else self._euler_maruyama_infill(X)
        )
        return Population.new("X", next_x)

    def _generation_fits_budget(self, population_size: int) -> bool:
        """Check the full next-generation cost against an FE termination."""
        maximum = 0
        termination = getattr(self, "termination", None)
        for attr in ("n_max_evals", "n_max_eval", "max_evals"):
            value = getattr(termination, attr, None)
            if value is not None:
                try:
                    maximum = int(value)
                except (TypeError, ValueError):
                    maximum = 0
                break
        if maximum <= 0:
            return True

        mode = self._resolve_jacobian_mode()
        derivative_batches = 0
        if mode == "finite_difference":
            derivative_batches = 2 * int(self.problem.n_var)
            if self.adaptive_step:
                derivative_batches *= 1 + self.max_halvings
        required = int(population_size) * (1 + derivative_batches)
        used = int(getattr(self.evaluator, "n_eval", 0) or 0)
        return used + required <= maximum

    def _truncate_archive(self, archive):
        if archive is None or len(archive) == 0:
            return archive

        objectives = np.asarray(archive.get("F"), dtype=float)
        _, unique_indices = np.unique(objectives, axis=0, return_index=True)
        archive = archive[np.sort(unique_indices)]
        if len(archive) <= self.archive_size:
            return archive

        crowding = calc_crowding_distance(np.asarray(archive.get("F"), dtype=float))
        order = np.argsort(-crowding, kind="mergesort")
        return archive[order[: self.archive_size]]

    def _advance(self, infills=None, **kwargs):
        """Advance trajectories and update the separate coverage archive."""
        if infills is None:
            return
        self.pop = infills
        candidates = (
            infills
            if self.ssw_archive is None
            else Population.merge(self.ssw_archive, infills)
        )
        non_dominated = filter_optimum(candidates, least_infeasible=True)
        self.ssw_archive = self._truncate_archive(non_dominated)

    def _set_optimum(self):
        self.opt = (
            self.ssw_archive
            if self.ssw_archive is not None and len(self.ssw_archive) > 0
            else filter_optimum(self.pop, least_infeasible=True)
        )
