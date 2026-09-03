"""EmoPyLab Optimization Engine and minimize() function (zero-pymoo standalone)."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable
import numpy as np

from core.termination import Termination, get_termination
from util.display.multi import Display, MultiObjectiveDisplay, SingleObjectiveDisplay


class Result:
    """Result object returned by minimize()."""

    def __init__(self) -> None:
        self.algorithm: Any = None
        self.problem: Any = None
        self.pop: Any = None
        self.opt: Any = None
        self.X: np.ndarray = np.empty((0, 0))
        self.F: np.ndarray = np.empty((0, 0))
        self.G: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.CV: np.ndarray | None = None
        self.feasible: np.ndarray | None = None
        self.exec_time: float = 0.0
        self.history: list[Any] | None = None


def minimize(
    problem: Any,
    algorithm: Any,
    termination: Any = None,
    seed: int | None = None,
    verbose: bool = False,
    display: Display | None = None,
    callback: Callable[[Any], None] | None = None,
    save_history: bool = False,
    copy_algorithm: bool = False,
    copy_termination: bool = False,
    **kwargs: Any,
) -> Result:
    """Execute optimization run on a problem using an evolutionary algorithm.

    Args:
        problem: Problem instance to optimize.
        algorithm: Evolutionary Algorithm instance (e.g. NSGA2, NSGA3, MOEAD, etc.).
        termination: Termination condition (n_gen, n_eval, or Termination instance).
        seed: Random seed for reproducibility.
        verbose: If True, prints generation statistics.
        display: Display callback for formatting/printing.
        callback: Custom callback function invoked after every generation.
        save_history: If True, stores deep copies of the algorithm/population per generation.
        copy_algorithm: If True, operates on a deepcopy of the algorithm.
        copy_termination: If True, copies termination.

    Returns:
        Result instance containing optimal solutions, objective values, and runtime.
    """
    t_start = time.perf_counter()

    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # Set seed
    if seed is not None:
        np.random.seed(seed)
        if hasattr(algorithm, "seed"):
            algorithm.seed = seed

    # Resolve termination
    if termination is None:
        n_obj = getattr(problem, "n_obj", 1)
        if n_obj > 1:
            termination = get_termination("default_multi")
        else:
            termination = get_termination("default_single")
    else:
        termination = get_termination(termination)

    if copy_termination:
        termination = copy.deepcopy(termination)

    algorithm.termination = termination
    algorithm.verbose = verbose

    # Setup display
    if display is None and verbose:
        n_obj = getattr(problem, "n_obj", 1)
        if n_obj > 1:
            display = MultiObjectiveDisplay()
        else:
            display = SingleObjectiveDisplay()
    algorithm.display = display

    # Setup callback
    if callback is not None:
        algorithm.callback = callback

    # Setup history
    history = [] if save_history else None

    # Algorithm setup & run loop
    if hasattr(algorithm, "setup"):
        algorithm.setup(problem, termination=termination, seed=seed, verbose=verbose, **kwargs)
    elif hasattr(algorithm, "initialize"):
        algorithm.initialize(problem)

    # If algorithm is a full Algorithm subclass with run(), use algorithm.run()
    if hasattr(algorithm, "run"):
        res = algorithm.run()
        if res is not None and isinstance(res, Result):
            if save_history and history is not None:
                res.history = history
            return res
    else:
        # Standard step-by-step evolutionary loop
        while getattr(algorithm, "has_next", lambda: True)():
            if termination.has_terminated(algorithm):
                break

            algorithm.next()

            if display is not None:
                display(algorithm)

            if callback is not None:
                if hasattr(callback, "notify"):
                    callback.notify(algorithm)
                elif callable(callback):
                    callback(algorithm)

            if save_history:
                history.append(copy.deepcopy(algorithm))

    exec_time = time.perf_counter() - t_start
    # Build Result container
    res = Result()
    res.algorithm = algorithm
    res.problem = problem
    res.exec_time = exec_time
    res.history = history

    # Extract population and optimal solutions
    opt = getattr(algorithm, "opt", None)
    pop = getattr(algorithm, "pop", None)

    if opt is None and pop is not None:
        from util.optimum import filter_optimum
        opt = filter_optimum(pop)

    res.pop = pop
    res.opt = opt

    if opt is not None and len(opt) > 0:
        if hasattr(opt, "get"):
            res.X = opt.get("X")
            res.F = opt.get("F")
            res.G = opt.get("G")
            res.H = opt.get("H")
            res.CV = opt.get("CV")
            res.feasible = opt.get("feasible")
        elif isinstance(opt, np.ndarray):
            res.F = opt
        else:
            # list of individuals
            res.X = np.array([ind.X for ind in opt if hasattr(ind, "X")])
            res.F = np.array([ind.F for ind in opt if hasattr(ind, "F")])
            if any(hasattr(ind, "G") for ind in opt):
                res.G = np.array([getattr(ind, "G", None) for ind in opt])
            if any(hasattr(ind, "CV") for ind in opt):
                res.CV = np.array([getattr(ind, "CV", None) for ind in opt])

    return res
