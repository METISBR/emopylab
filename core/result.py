"""
Standalone Result and Meta representations (EmoPyLab 2026).
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional


class Result:
    """The resulting object of an optimization run."""

    def __init__(self) -> None:
        super().__init__()
        self.opt = None
        self.success: Optional[bool] = None
        self.message: Optional[str] = None
        self.problem = None
        self.archive = None
        self.pf = None
        self.algorithm = None
        self.pop = None
        self.X = None
        self.F = None
        self.CV = None
        self.G = None
        self.H = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.exec_time: Optional[float] = None
        self.history: list = []
        self.data: Any = None

    @property
    def cv(self):
        return self.CV[0] if self.CV is not None and len(self.CV) > 0 else None

    @property
    def f(self):
        return self.F[0] if self.F is not None and len(self.F) > 0 else None

    @property
    def feas(self) -> bool:
        cv_val = self.cv
        return cv_val is None or cv_val <= 0


class Meta:
    """Delegation proxy base class."""

    def __init__(self, obj: Any, copy_obj: bool = True, clazz: Optional[type] = None) -> None:
        if clazz is None:
            clazz = self.__class__

        wrapped = obj
        if copy_obj:
            wrapped = copy.deepcopy(wrapped)

        self.__class__ = type(
            clazz.__name__,
            tuple([clazz] + wrapped.__class__.mro()),
            {},
        )
        self.__dict__ = wrapped.__dict__
        self.__object__ = obj
        self.__super__ = wrapped
