"""
Standalone implementation of Problem and ElementwiseProblem (EmoPyLab 2026).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def _cache_decorator(func: Callable) -> Callable:
    """Simple cache decorator remembering the result of the first evaluation."""
    func_name = func.__name__

    def wrapper(self, *args, use_cache: bool = True, set_cache: bool = True, **kwargs):
        if not hasattr(self, "cache"):
            setattr(self, "cache", {})

        cache_dict = getattr(self, "cache")
        if use_cache and func_name in cache_dict:
            return cache_dict[func_name]
        else:
            obj = func(self, *args, **kwargs)
            if set_cache:
                cache_dict[func_name] = obj
            return obj

    return wrapper


Cache = _cache_decorator


def at_least_2d_array(x, extend_as: str = "row", return_if_reshaped: bool = False):
    """Ensure array is at least 2D."""
    if x is None:
        return (x, False) if return_if_reshaped else x
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    has_been_reshaped = False
    if x.ndim == 1:
        if extend_as.startswith("r"):
            x = x[None, :]
        elif extend_as.startswith("c"):
            x = x[:, None]
        else:
            raise ValueError("The option `extend_as` should be either `row` or `column`.")
        has_been_reshaped = True

    if return_if_reshaped:
        return x, has_been_reshaped
    else:
        return x


class LoopedElementwiseEvaluation:
    """Default sequential evaluation for elementwise problems."""

    def __call__(self, f: Callable, X: Union[np.ndarray, list]) -> list:
        return [f(x) for x in X]


class ElementwiseEvaluationFunction:
    """Wrapper function for single individual evaluation."""

    def __init__(self, problem: Problem, args: tuple, kwargs: dict) -> None:
        self.problem = problem
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x: np.ndarray) -> dict:
        out: dict = {}
        self.problem._evaluate(x, out, *self.args, **self.kwargs)
        return out


def default_shape(problem: Problem, n: int) -> dict:
    n_var = problem.n_var
    return dict(
        F=(n, problem.n_obj),
        G=(n, problem.n_ieq_constr),
        H=(n, problem.n_eq_constr),
        dF=(n, problem.n_obj, n_var),
        dG=(n, problem.n_ieq_constr, n_var),
        dH=(n, problem.n_eq_constr, n_var),
    )


class Problem:
    """Base class for all optimization problems."""

    def __init__(
        self,
        n_var: int = -1,
        n_obj: int = 1,
        n_ieq_constr: int = 0,
        n_eq_constr: int = 0,
        xl: Optional[Union[np.ndarray, float, int, list]] = None,
        xu: Optional[Union[np.ndarray, float, int, list]] = None,
        vtype: Optional[type] = None,
        vars: Optional[dict] = None,
        elementwise: bool = False,
        elementwise_func: type = ElementwiseEvaluationFunction,
        elementwise_runner: Any = None,
        requires_kwargs: bool = False,
        replace_nan_values_by: Optional[float] = None,
        exclude_from_serialization: Optional[list] = None,
        callback: Optional[Callable] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = (
            n_ieq_constr
            if "n_constr" not in kwargs
            else max(n_ieq_constr, kwargs.get("n_constr", 0))
        )
        self.n_eq_constr = n_eq_constr
        self.data = dict(**kwargs)
        self.xl = xl
        self.xu = xu
        self.callback = callback
        self.vtype = vtype
        self.elementwise = elementwise
        self.elementwise_func = elementwise_func
        self.elementwise_runner = (
            elementwise_runner
            if elementwise_runner is not None
            else LoopedElementwiseEvaluation()
        )
        self.requires_kwargs = requires_kwargs
        self.strict = strict
        self.replace_nan_values_by = replace_nan_values_by
        self.exclude_from_serialization = exclude_from_serialization

        if vars is not None:
            self.vars = vars
            self.n_var = len(vars)
            if self.xl is None:
                self.xl = {
                    name: var.lb if hasattr(var, "lb") else None
                    for name, var in vars.items()
                }
            if self.xu is None:
                self.xu = {
                    name: var.ub if hasattr(var, "ub") else None
                    for name, var in vars.items()
                }

        if self.n_var > 0:
            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(self.n_var) * self.xl
                self.xl = self.xl.astype(float)

            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(self.n_var) * self.xu
                self.xu = self.xu.astype(float)

    @property
    def n_constr(self) -> int:
        return self.n_ieq_constr + self.n_eq_constr

    def has_bounds(self) -> bool:
        return self.xl is not None and self.xu is not None

    def has_constraints(self) -> bool:
        return self.n_constr > 0

    def bounds(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.xl, self.xu

    def name(self) -> str:
        return self.__class__.__name__

    def evaluate(
        self,
        X: Union[np.ndarray, list],
        *args: Any,
        return_values_of: Optional[List[str]] = None,
        return_as_dictionary: bool = False,
        **kwargs: Any,
    ) -> Union[dict, np.ndarray, tuple]:
        if not self.requires_kwargs:
            kwargs = dict()

        if return_values_of is None:
            return_values_of = ["F"]
            if self.n_ieq_constr > 0:
                return_values_of.append("G")
            if self.n_eq_constr > 0:
                return_values_of.append("H")

        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = at_least_2d_array(
                X, extend_as="row", return_if_reshaped=True
            )
            if self.n_var > 0 and X.shape[1] != self.n_var:
                raise ValueError(
                    f"Input dimension {X.shape[1]} does not equal n_var {self.n_var}!"
                )
        else:
            only_single_value = not (isinstance(X, (list, tuple, np.ndarray)))

        _out = self.do(X, return_values_of, *args, **kwargs)

        out = {}
        for k, v in _out.items():
            if v is not None:
                v = np.asarray(v)
                if only_single_value and v.ndim > 0:
                    v = v[0]
                if self.replace_nan_values_by is not None:
                    v[np.isnan(v)] = self.replace_nan_values_by
                try:
                    out[k] = v.astype(np.float64)
                except Exception:
                    out[k] = v
            else:
                out[k] = v

        if self.callback is not None:
            self.callback(X, out)

        if return_as_dictionary:
            return out

        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            return tuple([out[e] for e in return_values_of])

    def do(
        self,
        X: Union[np.ndarray, list],
        return_values_of: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        out: dict = {name: None for name in return_values_of}

        if self.elementwise:
            self._evaluate_elementwise(X, out, *args, **kwargs)
        else:
            self._evaluate_vectorized(X, out, *args, **kwargs)

        return self._format_dict(out, len(X), return_values_of)

    def _evaluate_vectorized(
        self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any
    ) -> None:
        self._evaluate(X, out, *args, **kwargs)

    def _evaluate_elementwise(
        self, X: Union[np.ndarray, list], out: dict, *args: Any, **kwargs: Any
    ) -> None:
        f = self.elementwise_func(self, args, kwargs)
        elems = self.elementwise_runner(f, X)

        for elem in elems:
            for k, v in elem.items():
                if out.get(k, None) is None:
                    out[k] = []
                out[k].append(v)

        for k in out:
            if out[k] is not None:
                out[k] = np.asarray(out[k])

    def _format_dict(self, out: dict, N: int, return_values_of: List[str]) -> dict:
        shape = default_shape(self, N)
        ret = {}

        for name, v in out.items():
            if v is not None:
                v = np.asarray(v)
                if name in shape:
                    if isinstance(v, list):
                        v = np.column_stack(v)
                    try:
                        v = v.reshape(shape[name])
                    except Exception as e:
                        raise ValueError(
                            f"Problem Error: {name} expected shape {shape[name]} but got {v.shape}"
                        ) from e
                ret[name] = v

        for name in return_values_of:
            if name not in ret:
                s = shape.get(name, N)
                ret[name] = np.full(s, np.inf)

        return ret

    @Cache
    def nadir_point(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        pf = self.pareto_front(*args, **kwargs)
        if pf is not None:
            return np.max(pf, axis=0)
        return None

    @Cache
    def ideal_point(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        pf = self.pareto_front(*args, **kwargs)
        if pf is not None:
            return np.min(pf, axis=0)
        return None

    @Cache
    def pareto_front(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        pf = self._calc_pareto_front(*args, **kwargs)
        pf = at_least_2d_array(pf, extend_as="r")
        if pf is not None and pf.shape[1] == 2:
            pf = pf[np.argsort(pf[:, 0])]
        return pf

    @Cache
    def pareto_set(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        ps = self._calc_pareto_set(*args, **kwargs)
        return at_least_2d_array(ps, extend_as="r")

    @abstractmethod
    def _evaluate(self, x: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        pass

    def _calc_pareto_front(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def _calc_pareto_set(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def __str__(self) -> str:
        s = f"# name: {self.name()}\n"
        s += f"# n_var: {self.n_var}\n"
        s += f"# n_obj: {self.n_obj}\n"
        s += f"# n_ieq_constr: {self.n_ieq_constr}\n"
        s += f"# n_eq_constr: {self.n_eq_constr}\n"
        return s

    def __getstate__(self) -> dict:
        if self.exclude_from_serialization is not None:
            state = self.__dict__.copy()
            for key in self.exclude_from_serialization:
                state[key] = None
            return state
        return self.__dict__


class ElementwiseProblem(Problem):
    """Problem evaluated element-by-element."""

    def __init__(self, elementwise: bool = True, **kwargs: Any) -> None:
        super().__init__(elementwise=elementwise, **kwargs)
