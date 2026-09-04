# Author: Professor Thiago Santos at UFOP, Brazil
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class AlgorithmSpec:
    id: str
    name: str
    source: str
    module: str
    factory: Callable[[dict[str, Any]], Any]
    flags: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f"{self.name} [{self.source}]"

    @property
    def is_factory_callable(self) -> bool:
        """Best-effort flag: local create_algorithm wrappers are usually callables, not classes."""
        try:
            return bool(callable(self.factory) and not inspect.isclass(self.factory))
        except Exception:  # noqa: BLE001
            return True


@dataclass
class ProblemSpec:
    id: str
    name: str
    source: str
    module: str
    factory: Callable[[dict[str, Any]], Any]
    default_n_var: int = 30
    default_n_obj: int = 2

    @property
    def label(self) -> str:
        return f"{self.name} [{self.source}]"


@dataclass
class MetricSpec:
    id: str
    name: str
    source: str
    module: str
    factory: Callable[[dict[str, Any]], Callable[[np.ndarray], float]]

    @property
    def label(self) -> str:
        return f"{self.name} [{self.source}]"


@dataclass
class OperatorSpec:
    id: str
    name: str
    source: str
    category: str
    module: str
    class_name: str

    @property
    def label(self) -> str:
        return f"{self.name} [{self.source}]"
