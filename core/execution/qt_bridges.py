# Author: Professor Thiago Santos at UFOP, Brazil
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, Signal, Slot


class LLMTaskBridge(QObject):
    """Run a callable in a QThread and emit result/error back to the UI thread."""

    result_ready = Signal(object)
    partial_update = Signal(object)
    error = Signal(str)
    done = Signal()

    def __init__(self, fn: Callable[..., Any], *, use_partial_callback: bool = False) -> None:
        super().__init__()
        self._fn = fn
        self._use_partial_callback = bool(use_partial_callback)

    @Slot()
    def run(self) -> None:
        try:
            if self._use_partial_callback:
                result = self._fn(self.partial_update.emit)
            else:
                result = self._fn()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        else:
            self.result_ready.emit(result)
        finally:
            self.done.emit()
