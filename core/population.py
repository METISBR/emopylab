"""
Standalone implementation of Population (EmoPyLab 2026).
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional, Union

from core.individual import Individual, calc_cv as ind_calc_cv

__all__ = [
    "Population",
    "pop_from_array_or_individual",
    "merge",
    "interleaving_args",
    "calc_cv",
]


class Population(np.ndarray):
    """Array of Individuals representing a population in evolutionary algorithms."""

    def __new__(cls, individuals=None):
        if individuals is None:
            individuals = []
        elif isinstance(individuals, Individual):
            individuals = [individuals]
        return np.array(individuals, dtype=object).view(cls)

    def has(self, key: str) -> bool:
        return all([ind.has(key) for ind in self])

    def collect(self, func, to_numpy: bool = True):
        val = [func(self[i]) for i in range(len(self))]
        if to_numpy:
            val = np.array(val)
        return val

    def apply(self, func):
        self.collect(func, to_numpy=False)
        return self

    def set(self, *args, **kwargs):
        if self.size == 0:
            return self

        kwargs = interleaving_args(*args, kwargs=kwargs)

        for key, values in kwargs.items():
            is_iterable = hasattr(values, "__len__") and not isinstance(values, str)

            if is_iterable and len(values) != len(self):
                raise ValueError(
                    f"Population Set Attribute Error: Number of values ({len(values)}) "
                    f"and population size ({len(self)}) do not match!"
                )

            for i in range(len(self)):
                val = values[i] if is_iterable else values

                if isinstance(val, np.ndarray) and not val.flags["OWNDATA"]:
                    val = val.copy()

                if isinstance(val, np.ndarray) and key in ("F", "X", "G", "H", "_F", "_X", "_G", "_H"):
                    if val.ndim > 1:
                        val = np.squeeze(val)
                    if val.ndim == 0:
                        val = np.array([val])

                self[i].set(key, val)

        return self

    def get(self, *args, to_numpy: bool = True, **kwargs):
        val = {c: [] for c in args}

        for i in range(len(self)):
            for c in args:
                val[c].append(self[i].get(c, **kwargs))

        res = []
        for c in args:
            e = val[c]
            if to_numpy:
                # Safely construct homogeneous numpy array even if elements differ in shape (e.g. None or empty)
                try:
                    arr = np.array(e)
                except ValueError:
                    # Handle inhomogeneous list
                    clean_e = [x if x is not None else np.array([]) for x in e]
                    try:
                        arr = np.array(clean_e, dtype=object)
                    except Exception:
                        arr = np.array(clean_e)
                
                # Special handling for standard tensor/array attributes to ensure clean 2D arrays
                if c in ("F", "X", "G", "H", "_F", "_X", "_G", "_H", "CV", "_CV"):
                    if arr.dtype == object and len(arr) > 0 and isinstance(arr[0], np.ndarray):
                        try:
                            arr = np.vstack([np.atleast_1d(x) for x in arr])
                        except Exception:
                            pass
                    if arr.ndim == 3 and arr.shape[1] == 1:
                        arr = np.squeeze(arr, axis=1)
                    elif arr.ndim == 3 and arr.shape[2] == 1:
                        arr = np.squeeze(arr, axis=2)
                    elif arr.ndim == 1 and len(self) > 0 and arr.dtype != object:
                        arr = np.atleast_2d(arr).T if c in ("CV", "_CV") else np.atleast_2d(arr)
                res.append(arr)
            else:
                res.append(e)

        if len(args) == 1:
            return res[0]
        else:
            return tuple(res)

    def filter(self, mask):
        return self[mask]

    def extract(self, *keys, to_numpy: bool = True):
        return self.get(*keys, to_numpy=to_numpy)

    def copy(self, deep: bool = True):
        if deep:
            import copy
            copied_individuals = [ind.copy(deep=True) for ind in self]
            return Population(copied_individuals)
        else:
            return Population([ind for ind in self])

    @classmethod
    def merge(cls, a, b, *args):
        m = merge(a, b)
        others = list(args)
        while len(others) > 0:
            m = merge(m, others.pop(0))
        return m

    @classmethod
    def create(cls, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            return Population(args[0])
        return Population(list(args))

    @classmethod
    def empty(cls, size: int = 0):
        individuals = [Individual() for _ in range(size)]
        return Population(individuals)

    @classmethod
    def new(cls, *args, **kwargs):
        kwargs = interleaving_args(*args, kwargs=kwargs)

        if len(kwargs) > 0:
            sizes = np.unique(np.array([len(v) for _, v in kwargs.items()]))
            if len(sizes) == 1:
                size = int(sizes[0])
            else:
                raise ValueError(f"Population.new inputs must have same sizes, got {sizes}")
        else:
            size = 0

        pop = Population.empty(size)
        pop.set(**kwargs)
        return pop


def pop_from_array_or_individual(array, pop=None):
    if pop is None:
        pop = Population.empty()

    if array is None:
        return None
    if isinstance(array, Population):
        return array
    if hasattr(array, "__len__") and hasattr(array, "get") and not isinstance(array, (dict, str)):
        # Population-like object from external/pymoo module
        try:
            return Population([ind for ind in array])
        except Exception:
            pass
    if isinstance(array, np.ndarray):
        if array.dtype == object and len(array) > 0 and hasattr(array[0], "X"):
            return Population(list(array))
        arr2d = np.atleast_2d(array)
        return pop.new("X", arr2d)
    elif hasattr(array, "X") and not hasattr(array, "__len__"):
        new_pop = Population.empty(1)
        new_pop[0] = array
        return new_pop
    elif isinstance(array, (list, tuple)):
        if len(array) > 0 and hasattr(array[0], "X"):
            return Population(list(array))
        else:
            arr2d = np.atleast_2d(np.array(array))
            return pop.new("X", arr2d)
    else:
        return None

def merge(a, b):
    if a is None:
        return b
    elif b is None:
        return a

    a = pop_from_array_or_individual(a)
    b = pop_from_array_or_individual(b)

    if a is None or len(a) == 0:
        return b
    elif b is None or len(b) == 0:
        return a
    else:
        return np.concatenate([a, b]).view(Population)


def interleaving_args(*args, kwargs=None):
    if len(args) % 2 != 0:
        raise ValueError(f"Even number of arguments required but {len(args)} provided.")

    if kwargs is None:
        kwargs = {}

    for i in range(len(args) // 2):
        key, values = args[i * 2], args[i * 2 + 1]
        kwargs[key] = values
    return kwargs


def calc_cv(pop, config=None):
    if config is None:
        config = Individual.default_config()

    G, H = pop.get("G", "H")
    CV = np.array([ind_calc_cv(g, h, config) for g, h in zip(G, H)])
    return CV
