"""EmoPyLab Columnar Structure-of-Arrays (SoA) Population.

Pre-allocates contiguous memory blocks for decision variables X, objectives F,
constraints G, constraint violations CV, and Pareto ranks, eliminating Python
per-individual object allocations.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.tensor.backend import get_array_module, index_tensor, to_device, to_numpy, vstack, zeros

class TensorPopulation:
    """Contiguous columnar population matrix on device/host."""

    def __init__(
        self,
        n_individuals: int,
        n_var: int,
        n_obj: int,
        n_constr: int = 0,
        X: Optional[Any] = None,
        F: Optional[Any] = None,
        G: Optional[Any] = None,
    ) -> None:
        self.n_individuals = int(n_individuals)
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.n_constr = int(n_constr)

        self.X = to_device(X) if X is not None else zeros((self.n_individuals, self.n_var))
        self.F = to_device(F) if F is not None else zeros((self.n_individuals, self.n_obj))
        self.G = to_device(G) if G is not None else (zeros((self.n_individuals, self.n_constr)) if self.n_constr > 0 else None)

        self.CV = zeros((self.n_individuals, 1))
        self.rank = np.zeros(self.n_individuals, dtype=np.int32)
        self.crowding = np.zeros(self.n_individuals, dtype=np.float32)

    def __len__(self) -> int:
        return self.n_individuals

    def update_constraint_violations(self) -> None:
        """Computes aggregate CV = sum(max(0, G)) on active device."""
        if self.G is not None and self.n_constr > 0:
            xp = get_array_module()
            g_pos = xp.maximum(0.0, self.G)
            self.CV = xp.sum(g_pos, axis=1, keepdims=True)
        else:
            self.CV = zeros((self.n_individuals, 1))

    def slice(self, indices: Any) -> TensorPopulation:
        """Returns a new TensorPopulation containing only the specified index slice."""
        idx_cpu = np.asarray(to_numpy(indices), dtype=np.int64).reshape(-1)
        sub_pop = TensorPopulation(
            n_individuals=len(idx_cpu),
            n_var=self.n_var,
            n_obj=self.n_obj,
            n_constr=self.n_constr,
            X=index_tensor(self.X, idx_cpu) if self.X is not None else None,
            F=index_tensor(self.F, idx_cpu) if self.F is not None else None,
            G=index_tensor(self.G, idx_cpu) if self.G is not None else None,
        )
        sub_pop.CV = index_tensor(self.CV, idx_cpu) if self.CV is not None else None
        sub_pop.rank = self.rank[idx_cpu]
        sub_pop.crowding = self.crowding[idx_cpu]
        return sub_pop

    @classmethod
    def merge(cls, pop1: TensorPopulation, pop2: TensorPopulation) -> TensorPopulation:
        """Concatenates two populations into a unified 2N candidate pool."""
        xp = get_array_module()
        n_total = len(pop1) + len(pop2)
        
        merged_X = vstack([pop1.X, pop2.X])
        merged_F = vstack([pop1.F, pop2.F])
        merged_G = vstack([pop1.G, pop2.G]) if pop1.G is not None and pop2.G is not None else None

        merged = cls(
            n_individuals=n_total,
            n_var=pop1.n_var,
            n_obj=pop1.n_obj,
            n_constr=pop1.n_constr,
            X=merged_X,
            F=merged_F,
            G=merged_G,
        )
        merged.update_constraint_violations()
        return merged
