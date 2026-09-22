# -*- coding: utf-8 -*-
# emopylab 2026
"""RVEA-a (RVEA with Reference Vector Adaptation).

Reference:
R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff. A reference vector guided
evolutionary algorithm for many-objective optimization. IEEE Transactions
on Evolutionary Computation, 2016, 20(5): 773-791.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from algorithms.rvea.rvea import RVEA, RVEASurvival, _cosine_similarity_matrix
from core.population import Population


ALGORITHM_FLAGS = {
    "RVEAa": {"binary", "constrained", "integer", "label", "many", "multi", "permutation", "real"},
    "RVEA-a": {"binary", "constrained", "integer", "label", "many", "multi", "permutation", "real"},
}


class RVEAa(RVEA):
    """RVEA with Reference Vector Adaptation (RVEA-a) for irregular/degenerate fronts."""

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        pop_size: int | None = None,
        alpha: float = 2.0,
        fr: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(ref_dirs=ref_dirs, pop_size=pop_size, alpha=alpha, **kwargs)
        self.fr = float(fr)
        self.V0: Optional[np.ndarray] = None
        self._last_adapt_tau: float = 0.0

    def _setup(self, problem: Any, **kwargs: Any) -> None:
        super()._setup(problem, **kwargs)
        self.V0 = np.copy(self.ref_dirs)
        self._last_adapt_tau = 0.0

    def _advance(self, infills: Optional[Population] = None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return

        F_off = np.asarray(infills.get("F"), dtype=float)
        self.z_min = np.minimum(self.z_min, np.min(F_off, axis=0))

        merged = Population.merge(self.pop, infills)

        # Calculate search progress tau in [0, 1]
        n_iter = self.n_iter or 1
        n_max = getattr(self.termination, "n_max_gen", getattr(self.termination, "n_max_evals", 30000))
        curr_eval = float(self.n_evals)
        tau = min(1.0, float(curr_eval) / float(max(1, n_max)))

        # Reference vector adaptation strategy (Cheng et al. 2016)
        if (tau - self._last_adapt_tau) >= self.fr and self.V0 is not None:
            F_all = np.asarray(merged.get("F"), dtype=float)
            z_max = np.max(F_all, axis=0)
            z_min = np.min(F_all, axis=0)
            scale = np.maximum(z_max - z_min, 1e-6)

            # V_t = V_0 * (z_max - z_min)
            V_adapted = self.V0 * scale[None, :]
            norm_v = np.linalg.norm(V_adapted, axis=1, keepdims=True)
            V_adapted = V_adapted / np.maximum(norm_v, 1e-12)

            self.ref_dirs = V_adapted
            self.survival = RVEASurvival(self.ref_dirs, alpha=self.alpha)
            self._last_adapt_tau = tau

        self.pop = self.survival.do(
            self.problem,
            merged,
            n_survive=self.pop_size,
            theta=tau,
            z_min=self.z_min,
        )
        self._set_optimum()


RVEA_a = RVEAa

__all__ = ["RVEAa", "RVEA_a", "ALGORITHM_FLAGS"]
