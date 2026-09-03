"""
Standalone implementation of Callback and CallbackCollection (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, List

__all__ = [
    "Callback",
    "CallbackCollection",
]


class Callback:
    """Base class for algorithm progress/iteration callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.data: dict = {}
        self.is_initialized: bool = False

    def initialize(self, algorithm: Any) -> None:
        pass

    def notify(self, algorithm: Any) -> None:
        pass

    def update(self, algorithm: Any) -> Any:
        return self._update(algorithm)

    def _update(self, algorithm: Any) -> Any:
        pass

    def __call__(self, algorithm: Any) -> None:
        if not self.is_initialized:
            self.initialize(algorithm)
            self.is_initialized = True

        self.notify(algorithm)
        self.update(algorithm)


class CallbackCollection(Callback):
    """Collection of callbacks executed in sequence."""

    def __init__(self, *args: Callback) -> None:
        super().__init__()
        self.callbacks = list(args)

    def update(self, algorithm: Any) -> None:
        for callback in self.callbacks:
            callback.update(algorithm)

    def notify(self, algorithm: Any) -> None:
        for callback in self.callbacks:
            callback.notify(algorithm)

    def initialize(self, algorithm: Any) -> None:
        for callback in self.callbacks:
            callback.initialize(algorithm)
