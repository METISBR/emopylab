"""EmoPyLab Miscellaneous Utilities (zero-pymoo standalone)."""

from __future__ import annotations

from functools import wraps
import itertools
from typing import Any, Callable
import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist


def default_random_state(func: Callable) -> Callable:
    """Decorator to inject a default random state (seed or np.random.RandomState/Generator) if not provided."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        seed = kwargs.get("seed", None)
        if seed is None and len(args) > 0 and hasattr(args[0], "seed"):
            seed = getattr(args[0], "seed")
        return func(*args, **kwargs)
    return wrapper


def at_least_2d_array(x: Any, return_if_reshaped: bool = False) -> np.ndarray | tuple[np.ndarray, bool]:
    """Ensure array is at least 2-dimensional."""
    arr = np.asarray(x)
    only_1d = arr.ndim <= 1
    if arr.ndim == 0:
        res = arr.reshape(1, 1)
    elif arr.ndim == 1:
        res = arr.reshape(1, -1)
    else:
        res = arr
    if return_if_reshaped:
        return res, only_1d
    return res


def cdist(XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean", **kwargs: Any) -> np.ndarray:
    """Compute distance between each pair of the two collections of inputs."""
    XA_arr = np.asarray(XA, dtype=float)
    XB_arr = np.asarray(XB, dtype=float)
    if XA_arr.ndim == 1:
        XA_arr = XA_arr.reshape(1, -1)
    if XB_arr.ndim == 1:
        XB_arr = XB_arr.reshape(1, -1)
    return scipy_cdist(XA_arr, XB_arr, metric=metric, **kwargs)


def crossover_mask(n_parents: int, n_matings: int, n_var: int) -> np.ndarray:
    """Generate boolean crossover masks for parent selection per variable."""
    return np.random.choice([True, False], size=(n_matings, n_var))


def row_at_least_once_true(M: np.ndarray) -> np.ndarray:
    """Ensure every row of a boolean matrix has at least one True entry."""
    M = np.asarray(M, dtype=bool).copy()
    if M.ndim == 1:
        M = M.reshape(1, -1)
    no_true = ~np.any(M, axis=1)
    if np.any(no_true):
        idx = np.where(no_true)[0]
        rand_cols = np.random.randint(0, M.shape[1], size=len(idx))
        M[idx, rand_cols] = True
    return M


def find_duplicates(X: np.ndarray, epsilon: float = 1e-16) -> np.ndarray:
    """Find duplicate rows in a 2D array and return boolean mask of duplicates (2nd+ occurrences)."""
    X = np.asarray(X)
    if X.ndim != 2 or len(X) <= 1:
        return np.zeros(len(X), dtype=bool)

    N = len(X)
    is_duplicate = np.zeros(N, dtype=bool)

    # If exact equality or floating point with epsilon
    if epsilon == 0:
        _, idx = np.unique(X, axis=0, return_index=True)
        unique_mask = np.zeros(N, dtype=bool)
        unique_mask[idx] = True
        return ~unique_mask
    else:
        # Distance-based duplicate check
        D = cdist(X, X)
        for i in range(N):
            if is_duplicate[i]:
                continue
            dups = np.where(D[i, (i + 1):] <= epsilon)[0] + (i + 1)
            is_duplicate[dups] = True
        return is_duplicate


def get_duplicates(X: np.ndarray, epsilon: float = 1e-16) -> np.ndarray:
    """Alias for find_duplicates."""
    return find_duplicates(X, epsilon=epsilon)


def has_feasible(pop: Any) -> bool:
    """Check whether a population contains at least one feasible individual."""
    if pop is None or len(pop) == 0:
        return False

    if hasattr(pop, "get"):
        CV = pop.get("CV")
        if CV is not None:
            return bool(np.any(np.asarray(CV) <= 0.0))
        G = pop.get("G")
        if G is not None:
            G_arr = np.asarray(G, dtype=float)
            if G_arr.ndim == 1:
                G_arr = G_arr[:, None]
            cv = np.sum(np.maximum(0.0, G_arr), axis=1)
            return bool(np.any(cv <= 0.0))
        feasible = pop.get("feasible")
        if feasible is not None:
            return bool(np.any(np.asarray(feasible, dtype=bool)))
    return True


def powerset(iterable: Any) -> itertools.chain:
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def calc_perpendicular_distance(N: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """Calculate perpendicular distance of points N to reference directions ref_dirs."""
    N = np.asarray(N, dtype=float)
    ref_dirs = np.asarray(ref_dirs, dtype=float)
    u = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    proj = np.dot(N, u.T)
    norm_N_sq = np.sum(N**2, axis=1, keepdims=True)
    dist_sq = np.maximum(0.0, norm_N_sq - proj**2)
    return np.sqrt(dist_sq)


def random_permutations(n_perms: int, length: int, random_state: Any = None) -> np.ndarray:
    """Generate n_perms random permutations of range(length) concatenated as 1D array."""
    if length <= 0 or n_perms <= 0:
        return np.empty(0, dtype=int)
    
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.RandomState(random_state)
    elif hasattr(random_state, "permutation"):
        rng = random_state
    else:
        rng = np.random

    perms = [rng.permutation(length) for _ in range(n_perms)]
    return np.concatenate(perms)
