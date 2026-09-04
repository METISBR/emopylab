"""EmoPyLab Bitwise Boolean Matrix Non-Dominated Sorting for GPU/SIMD."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.tensor.backend import get_array_module, to_device, to_numpy


def boolean_matrix_nds(F_tensor: Any) -> list[np.ndarray]:
    """Computes Pareto dominance fronts using vectorized broadcast operations in O(N^2).

    D_{i,j} = (and_{m=1}^M f_m(x_i) <= f_m(x_j)) and (or_{m=1}^M f_m(x_i) < f_m(x_j))
    """
    xp = get_array_module()
    F = F_tensor
    N = F.shape[0]

    # Pairwise comparison tensor [N, N, M]
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]

    # D[i, j] is True if solution i dominates solution j
    dom_matrix = xp.all(le, axis=2) & xp.any(lt, axis=2)
    dom_np = to_numpy(dom_matrix)

    domination_counts = np.sum(dom_np, axis=0)  # number of solutions dominating j
    dominated_sets = [np.where(dom_np[i, :])[0] for i in range(N)]

    fronts = []
    current_front = np.where(domination_counts == 0)[0]

    while len(current_front) > 0:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = np.array(next_front, dtype=np.int64)

    return fronts
