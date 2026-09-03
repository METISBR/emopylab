"""EmoPyLab Termination Package (zero-pymoo standalone)."""

from __future__ import annotations

from core.termination import (
    DefaultMultiObjectiveTermination,
    DefaultSingleObjectiveTermination,
    MaximumFunctionEvaluationTermination,
    MaximumGenerationTermination,
    MaximumTimeTermination,
    Termination,
    get_termination,
)

__all__ = [
    "Termination",
    "MaximumGenerationTermination",
    "MaximumFunctionEvaluationTermination",
    "MaximumTimeTermination",
    "DefaultSingleObjectiveTermination",
    "DefaultMultiObjectiveTermination",
    "get_termination",
]
