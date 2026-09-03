"""EmoPyLab Multi-Objective and Single-Objective Display / Output (zero-pymoo standalone)."""

from __future__ import annotations

import time
from typing import Any
import numpy as np


class Display:
    """Base Display / Output class for evolutionary algorithms."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def update(self, algorithm: Any) -> None:
        """Called each generation to print or format output."""
        pass

    def __call__(self, algorithm: Any) -> None:
        self.update(algorithm)


class SingleObjectiveDisplay(Display):
    """Console display for Single-Objective Optimization algorithms."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def update(self, algorithm: Any) -> None:
        if getattr(algorithm, "verbose", False):
            n_gen = getattr(algorithm, "n_gen", 0)
            n_evals = getattr(algorithm.evaluator, "n_eval", getattr(algorithm, "n_evals", 0))
            opt = getattr(algorithm, "opt", None)
            f_min = None
            if opt is not None and len(opt) > 0:
                F = opt.get("F") if hasattr(opt, "get") else getattr(opt[0], "F", None)
                if F is not None:
                    f_min = np.min(F)
            print(f"Gen: {n_gen:4d} | Evals: {n_evals:7d} | Best f: {f_min}")


class MultiObjectiveDisplay(Display):
    """Console display for Multi-Objective Optimization algorithms."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def update(self, algorithm: Any) -> None:
        if getattr(algorithm, "verbose", False):
            n_gen = getattr(algorithm, "n_gen", 0)
            n_evals = getattr(algorithm.evaluator, "n_eval", getattr(algorithm, "n_evals", 0)) if hasattr(algorithm, "evaluator") else getattr(algorithm, "n_evals", 0)
            n_nds = len(getattr(algorithm, "opt", []))
            print(f"Gen: {n_gen:4d} | Evals: {n_evals:7d} | NDS count: {n_nds}")


# Alias for backward compatibility
MultiObjectiveOutput = MultiObjectiveDisplay
SingleObjectiveOutput = SingleObjectiveDisplay
Output = Display
