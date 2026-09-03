"""Decoupled Execution Substrate and Optimization Runner for EmoPyLab.

Provides pure Python / headless optimization execution completely detached from
PySide6, allowing CLI runs, batch scripting, and GUI workers to share identical
reproducibility and metric evaluation logic.
"""

from __future__ import annotations

import importlib
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from core.execution.reproducibility import (
    SEED_MODE_FIXED,
    SEED_MODE_RANDOM,
    SEED_MODE_SEQUENCE,
    plan_run_seeds,
)
from metrics.evaluator import evaluate_front


@dataclass
class OptimizationResult:
    """Standardized result container for a single optimization run."""
    algorithm_name: str
    problem_name: str
    seed: int
    n_gen: int
    pop_size: int
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    F: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    G: Optional[np.ndarray] = None
    cv: Optional[np.ndarray] = None
    feasible: Optional[np.ndarray] = None
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    sidecar_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "problem": self.problem_name,
            "seed": self.seed,
            "n_gen": self.n_gen,
            "pop_size": self.pop_size,
            "n_solutions": len(self.F),
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "success": self.success,
            "error": self.error_message,
        }


def _resolve_problem_instance(problem_name: str, n_var: Optional[int] = None, n_obj: Optional[int] = None, **kwargs: Any) -> Any:
    """Dynamically resolve and instantiate a benchmark problem."""
    name_clean = problem_name.upper().replace("_", "").replace("-", "")

    # 1. Direct standard problem lookup
    try:
        from problems.multi import zdt
        for attr in dir(zdt):
            if attr.upper().replace("_", "").replace("-", "") == name_clean:
                cls = getattr(zdt, attr)
                if inspect.isclass(cls):
                    inst_kwargs = dict(kwargs)
                    if n_var is not None:
                        inst_kwargs["n_var"] = n_var
                    if n_obj is not None:
                        inst_kwargs["n_obj"] = n_obj
                    return cls(**inst_kwargs)
    except Exception:
        pass

    try:
        from problems.many import dtlz
        for attr in dir(dtlz):
            if attr.upper().replace("_", "").replace("-", "") == name_clean:
                cls = getattr(dtlz, attr)
                if inspect.isclass(cls):
                    inst_kwargs = dict(kwargs)
                    if n_var is not None:
                        inst_kwargs["n_var"] = n_var
                    if n_obj is not None:
                        inst_kwargs["n_obj"] = n_obj
                    return cls(**inst_kwargs)
    except Exception:
        pass

    # 2. Try local EmoPyLab problem catalog (many, multi, single)
    root_dir = Path(__file__).resolve().parent.parent.parent
    prob_dir = root_dir / "problems"
    for py_path in prob_dir.rglob("*.py"):
        if py_path.name.startswith("_"):
            continue
        try:
            rel_parts = py_path.relative_to(root_dir).with_suffix("").parts
            mod_name = ".".join(rel_parts)
            mod = importlib.import_module(mod_name)
            for attr_name in dir(mod):
                if attr_name.upper().replace("_", "").replace("-", "") == name_clean:
                    cls = getattr(mod, attr_name)
                    if inspect.isclass(cls):
                        inst_kwargs = dict(kwargs)
                        if n_var is not None:
                            inst_kwargs["n_var"] = n_var
                        if n_obj is not None:
                            inst_kwargs["n_obj"] = n_obj
                        return cls(**inst_kwargs)
        except Exception:
            continue

    raise ValueError(f"Problem '{problem_name}' could not be resolved in EmoPyLab catalog.")


def _resolve_algorithm_instance(algorithm_name: str, pop_size: int = 100, **kwargs: Any) -> Any:
    """Dynamically resolve and instantiate an algorithm from the EmoPyLab catalog."""
    name_clean = algorithm_name.upper().replace("_", "").replace("-", "")

    # 1. Try algorithms.moo.* or algorithms.native.* first
    if name_clean in ("NSGA2", "NSGAII"):
        try:
            from algorithms.moo.nsga2 import NSGA2
            return NSGA2(pop_size=pop_size, **kwargs)
        except Exception:
            try:
                from algorithms.native.nsga2 import NativeNSGA2
                sig = inspect.signature(NativeNSGA2.__init__)
                valid_kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return NativeNSGA2(pop_size=pop_size, **valid_kw)
            except Exception:
                pass

    elif name_clean in ("NSGA3", "NSGAIII"):
        try:
            from algorithms.moo.nsga3 import NSGA3
            from util.ref_dirs import get_reference_directions
            n_obj = kwargs.get("n_obj", 3)
            ref_dirs = kwargs.get("ref_dirs")
            if ref_dirs is None:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
        except Exception:
            try:
                from algorithms.native.nsga3 import NativeNSGA3
                sig = inspect.signature(NativeNSGA3.__init__)
                valid_kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return NativeNSGA3(pop_size=pop_size, **valid_kw)
            except Exception:
                pass

    elif name_clean in ("MOEAD", "MOEA/D"):
        try:
            from algorithms.moo.moead import MOEAD
            from util.ref_dirs import get_reference_directions
            n_obj = kwargs.get("n_obj", 3)
            ref_dirs = kwargs.get("ref_dirs")
            if ref_dirs is None:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            return MOEAD(ref_dirs=ref_dirs, n_neighbors=15)
        except Exception:
            try:
                from algorithms.native.moead import NativeMOEAD
                sig = inspect.signature(NativeMOEAD.__init__)
                valid_kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return NativeMOEAD(n_neighbors=15, **valid_kw)
            except Exception:
                pass

    elif name_clean in ("RVEA",):
        try:
            from algorithms.moo.rvea import RVEA
            return RVEA(pop_size=pop_size, **kwargs)
        except Exception:
            pass

    elif name_clean in ("AGEMOEA2", "AGE2", "AGEII"):
        try:
            from algorithms.moo.age2 import AGEMOEA2
            return AGEMOEA2(pop_size=pop_size, **kwargs)
        except Exception:
            try:
                from algorithms.age_ii.age_ii import AGEII
                return AGEII(pop_size=pop_size, **kwargs)
            except Exception:
                pass
    # 2. Try exact match or normalized match in algorithms/ directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    algo_dir = root_dir / "algorithms"
    for sub in algo_dir.iterdir():
        if sub.is_dir():
            sub_clean = sub.name.upper().replace("_", "").replace("-", "")
            if sub_clean == name_clean:
                main_file = sub / f"{sub.name}.py"
                if main_file.exists():
                    mod_name = f"algorithms.{sub.name}.{sub.name}"
                    mod = importlib.import_module(mod_name)
                    for attr in dir(mod):
                        cls = getattr(mod, attr)
                        if inspect.isclass(cls) and cls.__name__.upper().replace("_", "").replace("-", "") == name_clean:
                            try:
                                sig = inspect.signature(cls.__init__)
                                inst_kwargs: dict[str, Any] = {}
                                if "pop_size" in sig.parameters:
                                    inst_kwargs["pop_size"] = pop_size
                                inst_kwargs.update(kwargs)
                                return cls(**inst_kwargs)
                            except Exception:
                                return cls()

    raise ValueError(f"Algorithm '{algorithm_name}' could not be resolved.")



def run_single_optimization(
    algorithm_name: str,
    problem_name: str,
    pop_size: int = 100,
    n_gen: int = 250,
    seed: int = 42,
    n_var: Optional[int] = None,
    n_obj: Optional[int] = None,
    custom_algo_params: Optional[dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, dict[str, float]], None]] = None,
) -> OptimizationResult:
    """Execute a single deterministic multi-objective optimization run.

    Args:
        algorithm_name: Identifier for the solver.
        problem_name: Benchmark problem identifier.
        pop_size: Population size.
        n_gen: Maximum generations.
        seed: Random seed for exact determinism.
        n_var: Optional number of variables.
        n_obj: Optional number of objectives.
        custom_algo_params: Hyperparameter dictionary.
        progress_callback: Optional callable(gen, max_gen, current_metrics).

    Returns:
        OptimizationResult dataclass containing non-dominated solutions and metrics.
    """
    t_start = time.perf_counter()
    np.random.seed(seed)

    try:
        problem = _resolve_problem_instance(problem_name, n_var=n_var, n_obj=n_obj)
        algo_kwargs = custom_algo_params or {}
        if n_obj is not None:
            algo_kwargs["n_obj"] = n_obj
        algorithm = _resolve_algorithm_instance(algorithm_name, pop_size=pop_size, **algo_kwargs)
        
        from core.optimize import minimize
        from core.callback import Callback

        cb = None
        if progress_callback is not None:
            class GenerationReporter(Callback):
                def __init__(self, callback_fn: Callable[..., None], total_gen: int) -> None:
                    super().__init__()
                    self.callback_fn = callback_fn
                    self.total_gen = total_gen

                def notify(self, opt_algo: Any) -> None:
                    try:
                        cur_gen = opt_algo.n_gen
                        pop = opt_algo.pop
                        F_cur = pop.get("F") if pop is not None else np.empty((0, 0))
                        self.callback_fn(cur_gen, self.total_gen, {"n_solutions": float(len(F_cur))})
                    except Exception:
                        pass
            cb = GenerationReporter(progress_callback, n_gen)

        minimize_kwargs: dict[str, Any] = {
            "seed": seed,
            "verbose": False,
        }
        if cb is not None:
            minimize_kwargs["callback"] = cb

        if hasattr(algorithm, "solve") and not hasattr(algorithm, "setup"):
            res_solve = algorithm.solve(problem, n_gen=n_gen, seed=seed)
            if isinstance(res_solve, OptimizationResult):
                return res_solve
            F_res = res_solve.F if hasattr(res_solve, "F") and res_solve.F is not None else np.empty((0, problem.n_obj))
            X_res = res_solve.X if hasattr(res_solve, "X") and res_solve.X is not None else np.empty((0, problem.n_var))
        else:
            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                **minimize_kwargs,
            )
            F_res = res.F if res.F is not None else np.empty((0, problem.n_obj))
            X_res = res.X if res.X is not None else np.empty((0, problem.n_var))
        t_elapsed = time.perf_counter() - t_start
        # Evaluate quality indicators
        pf_true = None
        try:
            if hasattr(problem, "pareto_front"):
                attr = getattr(problem, "pareto_front")
                if callable(attr):
                    pf_true = attr()
                elif isinstance(attr, np.ndarray):
                    pf_true = attr
        except Exception:
            pf_true = None
        metrics_computed = evaluate_front(F_res, pf_true=pf_true) if F_res.ndim == 2 and len(F_res) > 0 else {}

        return OptimizationResult(
            algorithm_name=algorithm_name,
            problem_name=problem_name,
            seed=seed,
            n_gen=n_gen,
            pop_size=pop_size,
            X=X_res,
            F=F_res,
            metrics=metrics_computed,
            runtime_seconds=t_elapsed,
            success=True,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        t_elapsed = time.perf_counter() - t_start
        return OptimizationResult(
            algorithm_name=algorithm_name,
            problem_name=problem_name,
            seed=seed,
            n_gen=n_gen,
            pop_size=pop_size,
            runtime_seconds=t_elapsed,
            success=False,
            error_message=str(exc),
        )


instantiate_problem = _resolve_problem_instance
instantiate_algorithm = _resolve_algorithm_instance


def run_optimization(config: dict[str, Any]) -> OptimizationResult:
    """Convenience alias accepting a configuration dictionary."""
    return run_single_optimization(
        algorithm_name=config.get("algorithm", "NSGA2"),
        problem_name=config.get("problem", "ZDT1"),
        pop_size=config.get("pop_size", 100),
        n_gen=config.get("n_gen", 250),
        seed=config.get("seed", 42),
        n_var=config.get("n_var"),
        n_obj=config.get("n_obj"),
        custom_algo_params=config.get("algo_params"),
    )


def evaluate_optimization_result(result: OptimizationResult, pf_true: Optional[np.ndarray] = None) -> dict[str, float]:
    """Computes quality metrics on a finalized OptimizationResult."""
    if not result.success or len(result.F) == 0:
        return {}
    return evaluate_front(result.F, pf_true=pf_true)

