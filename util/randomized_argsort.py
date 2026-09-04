"""Randomized argsort for EmoPyLab."""

import numpy as np


def randomized_argsort(a, order="ascending", method="numpy", random_state=None):
    """Sort array indices with randomized tie breaking."""
    a = np.asarray(a)
    n = len(a)
    if n <= 1:
        return np.arange(n)

    if random_state is None:
        perm = np.random.permutation(n)
    elif isinstance(random_state, (int, np.integer)):
        perm = np.random.RandomState(random_state).permutation(n)
    elif hasattr(random_state, "permutation"):
        perm = random_state.permutation(n)
    else:
        perm = np.random.permutation(n)

    # Permute array before stable sort so ties are broken randomly
    a_perm = a[perm]
    
    if order == "descending":
        # sort descending
        idx = np.argsort(-a_perm, kind="mergesort" if method == "numpy" else "quicksort")
    else:
        idx = np.argsort(a_perm, kind="mergesort" if method == "numpy" else "quicksort")

    return perm[idx]
