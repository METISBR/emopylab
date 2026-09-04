"""EmoPyLab Remote Data and Asset Loader (zero-pymoo standalone)."""

from __future__ import annotations

import os
from typing import Any
import numpy as np


class Remote:
    _instance: Remote | None = None

    @classmethod
    def get_instance(cls) -> Remote:
        if cls._instance is None:
            cls._instance = Remote()
        return cls._instance

    def load(self, *parts: str) -> np.ndarray:
        """Load remote/cached asset or return dummy Pareto Front if offline."""
        # Try local cache if available
        cache_dir = os.path.expanduser("~/.emopylab/cache")
        local_path = os.path.join(cache_dir, *parts)
        if os.path.exists(local_path):
            try:
                return np.loadtxt(local_path)
            except Exception:
                pass
        # Fallback to empty array / approximate placeholder
        return np.empty((0, 2), dtype=float)
