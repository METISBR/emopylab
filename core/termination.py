"""EmoPyLab Termination Criteria (zero-pymoo standalone)."""

from __future__ import annotations

import time
from typing import Any
import numpy as np


class Termination:
    """Base class for all termination criteria."""

    def __init__(self) -> None:
        self.force_termination = False
        self.perc: float = 0.0

    def do_continue(self, *args: Any, **kwargs: Any) -> bool:
        """Return True if algorithm should continue, False if it should terminate."""
        return not self.has_terminated(*args, **kwargs)

    def terminate(self) -> None:
        """Force the algorithm to terminate on the next check."""
        self.force_termination = True
        self.perc = 1.0

    def has_terminated(self, *args: Any, **kwargs: Any) -> bool:
        """Check if termination condition has been reached.

        Accepts an optional algorithm argument (ignored for progress
        recomputation) so it can be called both standalone and with an
        algorithm instance.
        """
        if self.force_termination:
            return True
        return self.perc >= 1.0

    def update(self, algorithm: Any = None) -> float:
        """Update termination progress and return fraction complete (0.0 to 1.0)."""
        if self.force_termination:
            self.perc = 1.0
            return 1.0
        if algorithm is not None:
            self.perc = float(self._update(algorithm))
        return self.perc

    def _update(self, algorithm: Any) -> float:
        return self.perc


class MaximumGenerationTermination(Termination):
    """Terminates when maximum generations are reached."""

    def __init__(self, n_max_gen: int = 100) -> None:
        super().__init__()
        self.n_max_gen = int(n_max_gen)

    def _update(self, algorithm: Any) -> float:
        if self.n_max_gen is None or self.n_max_gen <= 0:
            return 0.0
        n_gen = getattr(algorithm, "n_gen", getattr(algorithm, "n_iter", 0)) or 0
        return float(n_gen) / float(self.n_max_gen)


class MaximumFunctionEvaluationTermination(Termination):
    """Terminates when maximum function evaluations are reached."""

    def __init__(self, n_max_evals: int = 10000) -> None:
        super().__init__()
        self.n_max_evals = int(n_max_evals)
        self.n_max_eval = self.n_max_evals
        self.max_evals = self.n_max_evals

    def _update(self, algorithm: Any) -> float:
        if self.n_max_evals is None or self.n_max_evals <= 0:
            return 0.0
        if hasattr(algorithm, "evaluator") and getattr(algorithm.evaluator, "n_eval", None) is not None:
            n_eval = algorithm.evaluator.n_eval
        else:
            n_eval = getattr(algorithm, "n_evals", getattr(algorithm, "evals", 0))
        n_eval = n_eval or 0
        return float(n_eval) / float(self.n_max_evals)


class MaximumTimeTermination(Termination):
    """Terminates when maximum wall-clock time has passed."""

    def __init__(self, max_time: float = 3600.0) -> None:
        super().__init__()
        self.max_time = float(max_time)
        self.start_time: float | None = None

    def _update(self, algorithm: Any) -> float:
        if self.start_time is None:
            self.start_time = time.time()
            return 0.0
        elapsed = time.time() - self.start_time
        return float(elapsed) / float(max(1e-6, self.max_time))

class DefaultMultiObjectiveTermination(MaximumGenerationTermination):
    """Default termination for multi-objective optimization."""
    pass


class DefaultSingleObjectiveTermination(MaximumGenerationTermination):
    """Default termination for single-objective optimization."""
    pass


def get_termination(type_or_tuple: Any, *args: Any, **kwargs: Any) -> Termination:
    """Factory function for termination criteria.

    Examples:
        get_termination("n_gen", 100)
        get_termination("n_eval", 25000)
        get_termination("time", 60.0)
        get_termination(("n_gen", 50))
    """
    if isinstance(type_or_tuple, Termination):
        return type_or_tuple

    if isinstance(type_or_tuple, tuple):
        name = type_or_tuple[0]
        t_args = type_or_tuple[1:] + args
    else:
        name = type_or_tuple
        t_args = args

    name_str = str(name).strip().lower().replace("-", "_")

    if name_str in ("n_gen", "n_max_gen", "gen", "max_gen", "generations"):
        val = t_args[0] if len(t_args) > 0 else kwargs.get("n_max_gen", 100)
        return MaximumGenerationTermination(n_max_gen=int(val))

    if name_str in ("n_eval", "n_evals", "n_max_evals", "eval", "evals", "max_evals", "fe", "max_fe"):
        val = t_args[0] if len(t_args) > 0 else kwargs.get("n_max_evals", kwargs.get("max_evals", 10000))
        return MaximumFunctionEvaluationTermination(n_max_evals=int(val))

    if name_str in ("time", "max_time", "runtime"):
        val = t_args[0] if len(t_args) > 0 else kwargs.get("max_time", 3600.0)
        return MaximumTimeTermination(max_time=float(val))

    if name_str in ("default_single", "single"):
        val = t_args[0] if len(t_args) > 0 else kwargs.get("n_max_gen", 100)
        return DefaultSingleObjectiveTermination(n_max_gen=int(val))

    if name_str in ("default_multi", "multi"):
        val = t_args[0] if len(t_args) > 0 else kwargs.get("n_max_gen", 100)
        return DefaultMultiObjectiveTermination(n_max_gen=int(val))

    if isinstance(name, (int, np.integer)):
        return MaximumGenerationTermination(n_max_gen=int(name))

    # Fallback to generations if string is numeric
    try:
        val = int(name_str)
        return MaximumGenerationTermination(n_max_gen=val)
    except ValueError:
        pass

    return MaximumGenerationTermination(n_max_gen=100)
