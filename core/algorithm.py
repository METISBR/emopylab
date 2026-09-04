"""
Standalone implementation of Algorithm, LoopwiseAlgorithm, and MetaAlgorithm (EmoPyLab 2026).
"""

from __future__ import annotations

import copy
import time
from typing import Any, Optional, Union
import numpy as np

from core.callback import Callback
from core.evaluator import Evaluator
from core.population import Population
from core.result import Result, Meta

__all__ = [
    "Algorithm",
    "LoopwiseAlgorithm",
    "MetaAlgorithm",
    "default_termination",
]


class _SimpleDisplay:
    """Fallback display implementation."""

    def __init__(self, output=None, verbose: bool = False, progress: bool = False) -> None:
        self.output = output
        self.verbose = verbose
        self.progress = progress

    def __call__(self, algorithm: Any) -> None:
        pass

    def finalize(self) -> None:
        pass


def _default_filter_optimum(pop: Population, least_infeasible: bool = False):
    """Local fallback filter_optimum without circular dependencies."""
    try:
        from util.optimum import filter_optimum
        return filter_optimum(pop, least_infeasible=least_infeasible)
    except Exception:
        if pop is None or len(pop) == 0:
            return None

        # Filter feasible
        feas = [ind for ind in pop if ind.feas]
        if len(feas) > 0:
            F = np.array([ind.F for ind in feas])
            if F.shape[1] == 1:
                best_idx = int(np.argmin(F[:, 0]))
                return Population.create(feas[best_idx])
            else:
                # Pareto non-dominated
                try:
                    from util.nds.non_dominated_sorting import NonDominatedSorting
                    I = NonDominatedSorting().do(F, only_non_dominated_front=True)
                    return Population.create([feas[i] for i in I])
                except Exception:
                    return Population.create(feas)
        else:
            if least_infeasible:
                cvs = [ind.cv if ind.cv is not None else 0.0 for ind in pop]
                best_idx = int(np.argmin(cvs))
                return Population.create(pop[best_idx])
            return None


def default_termination(problem: Any):
    """Default termination criteria."""
    try:
        from termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
        if problem.n_obj > 1:
            return DefaultMultiObjectiveTermination()
        else:
            return DefaultSingleObjectiveTermination()
    except Exception:
        try:
            from core.termination import default_termination as term_default
            return term_default(problem)
        except Exception:
            class _FallbackTermination:
                def __init__(self, n_max_gen: int = 100) -> None:
                    self.n_max_gen = n_max_gen
                    self.perc = 0.0
                    self.force_termination = False

                def update(self, algorithm: Any) -> float:
                    if self.force_termination:
                        self.perc = 1.0
                    else:
                        n_gen = getattr(algorithm, "n_gen", 1) or 1
                        self.perc = min(1.0, float(n_gen) / float(self.n_max_gen))
                    return self.perc

                def has_terminated(self) -> bool:
                    return self.perc >= 1.0

                def terminate(self) -> None:
                    self.force_termination = True

            return _FallbackTermination()


def _resolve_termination(term: Any, problem: Any):
    if term is None:
        return default_termination(problem)
    if isinstance(term, tuple) or isinstance(term, str) or isinstance(term, (int, np.integer)):
        try:
            from core.termination import get_termination
            return get_termination(term)
        except Exception:
            pass
    try:
        from util.misc import termination_from_tuple
        return termination_from_tuple(term)
    except Exception:
        return term

class Algorithm:
    """Base class for population-based optimization algorithms in EmoPyLab."""

    def __init__(
        self,
        *args: Any,
        termination: Optional[Any] = None,
        output: Optional[Any] = None,
        display: Optional[Any] = None,
        callback: Optional[Any] = None,
        archive: Optional[Any] = None,
        return_least_infeasible: bool = False,
        save_history: bool = False,
        verbose: bool = False,
        seed: Optional[int] = None,
        evaluator: Optional[Any] = None,
        use_gpu: bool = False,
        array_backend: str = "auto",
        gpu_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Hardware-backend configuration
        self.use_gpu_requested = bool(use_gpu)
        self.array_backend_requested = str(array_backend).strip().lower() or "auto"
        self.gpu_dtype = str(gpu_dtype).strip().lower() or "float32"

        try:
            from core.execution.backend_runtime import (
                apple_silicon_available,
                detect_mlx_runtime,
            )

            req = self.array_backend_requested
            if req == "auto":
                req = (
                    "mlx"
                    if apple_silicon_available()
                    else ("jax" if self.use_gpu_requested else "numpy")
                )

            self.array_backend_effective = req
            if req == "mlx":
                info = detect_mlx_runtime()
                if not info.get("mlx_ok"):
                    self.array_backend_effective = "numpy"
        except Exception:
            self.array_backend_effective = (
                "jax" if self.use_gpu_requested else "numpy"
            )

        self.array_backend = self.array_backend_effective
        self.use_gpu = self.array_backend_effective in {"jax", "mlx"}
        self.backend_state = {
            "requested_backend": self.array_backend_requested,
            "effective_backend": self.array_backend_effective,
            "use_gpu": self.use_gpu,
            "gpu_dtype": self.gpu_dtype,
        }

        # Evolutionary state
        self.problem = None
        self.termination = termination
        self.output = output
        self.display = display
        self.callback = callback if callback is not None else Callback()
        self.archive = archive
        self.return_least_infeasible = return_least_infeasible
        self.save_history = save_history
        self.verbose = verbose
        self.seed = seed
        self.random_state = None
        self.evaluator = evaluator if evaluator is not None else Evaluator()
        self.history: list = []
        self.pop: Optional[Population] = None
        self.off: Optional[Population] = None
        self.opt: Optional[Population] = None
        self.n_iter: Optional[int] = None
        self.data: dict = {}
        self.is_initialized: bool = False
        self.start_time: Optional[float] = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_array_module(self) -> Any:
        if self.array_backend_effective == "mlx":
            try:
                import mlx.core as mx
                return mx
            except ImportError:
                pass
        try:
            from util import array_backend as _array_backend
            return _array_backend.xp
        except Exception:
            return np

    def setup(self, problem: Any, verbose: bool = False, progress: bool = False, **kwargs: Any) -> Algorithm:
        self.problem = problem

        if self.output is not None:
            self.output = copy.deepcopy(self.output)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.random_state = np.random.default_rng(self.seed)
        self.termination = _resolve_termination(self.termination, problem)

        if self.display is None:
            try:
                from util.display.display import Display
                self.display = Display(self.output, verbose=verbose, progress=progress)
            except Exception:
                try:
                    from util.display.multi import MultiObjectiveDisplay
                    self.display = MultiObjectiveDisplay()
                except Exception:
                    self.display = _SimpleDisplay(self.output, verbose=verbose, progress=progress)

        self._setup(problem, **kwargs)
        return self

    def run(self) -> Result:
        while self.has_next():
            self.next()
        return self.result()

    def has_next(self) -> bool:
        if self.termination is None:
            return False
        try:
            return not self.termination.has_terminated(self)
        except TypeError:
            return not self.termination.has_terminated()

    def finalize(self) -> Any:
        if self.display is not None and hasattr(self.display, "finalize"):
            self.display.finalize()
        return self._finalize()

    def next(self) -> None:
        infills = self.infill()
        if infills is not None:
            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)
        else:
            self.advance()

    def _initialize(self) -> None:
        self.start_time = time.time()
        self.n_iter = 1
        self.pop = Population.empty()
        self.opt = None

    def infill(self) -> Optional[Population]:
        if self.problem is None:
            raise ValueError("Please call `setup(problem)` before calling next().")

        if not self.is_initialized:
            self._initialize()
            infills = self._initialize_infill()
        else:
            infills = self._infill()

        if infills is not None and len(infills) > 0:
            # Check if evaluation budget requires infill truncating
            if hasattr(self, "termination") and self.termination is not None:
                max_evals = getattr(self.termination, "n_max_evals", getattr(self.termination, "n_max_eval", None))
                if max_evals is not None:
                    curr_evals = getattr(self.evaluator, "n_eval", 0) if self.evaluator is not None else getattr(self, "n_evals", 0)
                    remaining = int(max_evals) - int(curr_evals)
                    if remaining > 0 and len(infills) > remaining:
                        infills = infills[:remaining]
        return infills

    def advance(self, infills: Optional[Population] = None, **kwargs: Any) -> Any:
        self.off = infills

        if not self.is_initialized:
            self.n_iter = 1
            self.pop = infills
            self._initialize_advance(infills=infills, **kwargs)
            self.is_initialized = True
            self._post_advance()
        else:
            val = self._advance(infills=infills, **kwargs)
            if val is None or val:
                self._post_advance()

        is_term = False
        if self.termination is not None:
            try:
                is_term = self.termination.has_terminated(self)
            except TypeError:
                is_term = self.termination.has_terminated()
        if is_term:
            self.finalize()
            ret = self.result()
        else:
            ret = self.opt

        if self.archive is not None and infills is not None:
            if hasattr(self.archive, "add"):
                self.archive = self.archive.add(infills)

        return ret

    def result(self) -> Result:
        res = Result()
        res.start_time = self.start_time
        res.end_time = time.time()
        res.exec_time = (
            (res.end_time - res.start_time)
            if (res.start_time is not None and res.end_time is not None)
            else None
        )

        res.pop = self.pop
        res.archive = self.archive
        res.data = self.data
        res.algorithm = self

        opt = self.opt
        if opt is None or len(opt) == 0:
            opt = None
        elif not np.any(opt.get("FEAS")):
            if self.return_least_infeasible:
                opt = _default_filter_optimum(opt, least_infeasible=True)
            else:
                opt = None
        res.opt = opt

        if res.opt is None:
            X, F, CV, G, H = None, None, None, None, None
        else:
            X, F, CV, G, H = self.opt.get("X", "F", "CV", "G", "H")
            if self.problem is not None and getattr(self.problem, "n_obj", 1) == 1 and len(X) == 1:
                X, F, CV, G, H = X[0], F[0], CV[0], G[0], H[0]

        res.X, res.F, res.CV, res.G, res.H = X, F, CV, G, H
        res.problem = self.problem
        res.history = self.history
        return res

    def ask(self) -> Optional[Population]:
        return self.infill()

    def tell(self, *args: Any, **kwargs: Any) -> Any:
        return self.advance(*args, **kwargs)

    def _set_optimum(self) -> None:
        self.opt = _default_filter_optimum(self.pop, least_infeasible=True)

    def _post_advance(self) -> None:
        self._set_optimum()

        if self.termination is not None:
            self.termination.update(self)

        if self.display is not None and callable(self.display):
            self.display(self)

        if self.save_history:
            _hist, _callback, _display = self.history, self.callback, self.display
            self.history, self.callback, self.display = None, None, None
            obj = copy.deepcopy(self)
            self.history, self.callback, self.display = _hist, _callback, _display
            self.history.append(obj)

        if self.callback is not None:
            self.callback(self)

        if self.n_iter is not None:
            self.n_iter += 1

    # =========================================================================
    # Methods to be overwritten by subclasses
    # =========================================================================

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        pass

    def _initialize_infill(self) -> Optional[Population]:
        pass

    def _initialize_advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        pass

    def _infill(self) -> Optional[Population]:
        pass

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> Any:
        pass

    def _finalize(self) -> Any:
        pass

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_gen(self) -> int:
        return self.n_iter if self.n_iter is not None else 0

    @n_gen.setter
    def n_gen(self, value: Optional[int]) -> None:
        self.n_iter = value

    @property
    def n_evals(self) -> int:
        return getattr(self.evaluator, "n_eval", 0) if self.evaluator is not None else 0


class LoopwiseAlgorithm(Algorithm):
    """Algorithm executing generation generator pattern."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.generator = None
        self.state = None

    def _next(self) -> Any:
        pass

    def _infill(self) -> Any:
        if self.state is None:
            self._advance()
        return self.state

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> bool:
        if self.generator is None:
            self.generator = self._next()
        try:
            self.state = self.generator.send(infills)
        except StopIteration:
            self.generator = None
            self.state = None
            return True

        return False


class MetaAlgorithm(Meta):
    """Transparent proxy algorithm wrapper."""

    def __init__(self, algorithm: Any, copy_obj: bool = True, **kwargs: Any) -> None:
        if isinstance(algorithm, Meta):
            copy_obj = False
        super().__init__(algorithm, copy_obj=copy_obj)
        for key, value in kwargs.items():
            setattr(self, key, value)
