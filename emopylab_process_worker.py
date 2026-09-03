"""
Módulo worker para execução em ProcessPoolExecutor.
NÃO importa PySide6/Qt — é executado em subprocessos isolados.

Padrão arquitetural: Worker Process + Future.result() Bridge
- Cada trial roda em um processo isolado (RNG independente)
- Retorna dict serializável com resultados brutos
- O processo principal (QThread) computa métricas e emite sinais Qt
"""
from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


_ALLOWED_ALGO_PREFIXES = ("algorithms.", "core.", "pymoo.")
_ALLOWED_PROBLEM_PREFIXES = ("problems.", "core.", "pymoo.")


# ---------------------------------------------------------------------------
# Utilitários de instanciação (replicados de EmoPyLab sem dependência Qt)
# ---------------------------------------------------------------------------

def _is_allowed_worker_module(module_name: str, allowed_prefixes: tuple[str, ...]) -> bool:
    module = str(module_name or "").strip()
    if not module or module.startswith(".") or "/" in module or "\\" in module:
        return False
    return any(module.startswith(prefix) for prefix in allowed_prefixes)


def _positive_int(value: Any, default: int = 1, minimum: int = 0) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, v)


def _worker_build_reference_dirs(n_obj: int, target: int) -> np.ndarray:
    """Gera reference directions via das-dennis (idêntico ao EmoPyLab)."""
    target = max(2, int(target))
    n_obj = max(2, int(n_obj))
    try:
        from util.ref_dirs import get_reference_directions
    except Exception:
        rng = np.random.default_rng(42)
        dirs = rng.random((target, n_obj))
        dirs /= np.sum(dirs, axis=1, keepdims=True)
        return dirs

    last_dirs = None
    for partitions in range(1, 120):
        try:
            dirs = np.asarray(
                get_reference_directions("das-dennis", n_obj, n_partitions=partitions),
                dtype=float,
            )
        except Exception:
            break
        last_dirs = dirs
        if dirs.shape[0] >= target:
            return dirs

    if last_dirs is not None and last_dirs.size:
        return last_dirs

    rng = np.random.default_rng(42)
    dirs = rng.random((target, n_obj))
    dirs /= np.sum(dirs, axis=1, keepdims=True)
    return dirs


def _worker_instantiate_algorithm(cls: Any, config: dict[str, Any]) -> Any:
    """Instancia algoritmo a partir de config (idêntico ao EmoPyLab.instantiate_algorithm_class)."""
    pop_size = int(config.get("pop_size", 100))
    n_obj = int(config.get("n_obj", 2))
    n_inds = int(config.get("n_inds", pop_size))

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}

    prepared = {
        "pop_size": pop_size,
        "n_inds": n_inds,
        "ref_dirs": _worker_build_reference_dirs(n_obj=n_obj, target=max(12, pop_size)),
        "n_neighbors": max(2, min(20, pop_size // 4)),
        "prob_neighbor_mating": 0.9,
        "weights": np.ones(max(1, n_obj), dtype=float) / max(1, n_obj),
        "ref_points": np.vstack(
            [np.full(max(1, n_obj), 0.2, dtype=float), np.full(max(1, n_obj), 0.8, dtype=float)]
        ),
    }

    # Converte campos numpy-like de list → ndarray no config
    # (serializados via .tolist() para pickle/IPC)
    _numpy_fields = ("ref_dirs", "weights", "ref_points")
    for nf in _numpy_fields:
        if nf in config and config[nf] is not None and isinstance(config[nf], list):
            config[nf] = np.asarray(config[nf], dtype=float)

    # Se ref_dirs veio no config, usa ele (não recalcula)
    if "ref_dirs" in config and config["ref_dirs"] is not None:
        prepared["ref_dirs"] = config["ref_dirs"]

    operator_params = ["crossover", "mutation", "selection", "repair", "sampling"]

    for name, param in list(sig.parameters.items())[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in kwargs:
            continue
        if name in operator_params:
            continue
        if name in config:
            kwargs[name] = config[name]
        elif name in prepared:
            kwargs[name] = prepared[name]
        elif param.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(f"{cls.__name__} requires unsupported init parameter: '{name}'")

    return cls(**kwargs)


def _extract_front(F: Any) -> np.ndarray:
    """Extrai fronteira não-dominada de F."""
    if F is None:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(F, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0:
        return arr
    try:
        from util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting()
        idx = nds.do(arr, only_non_dominated_front=True)
        return np.asarray(arr[idx], dtype=float)
    except Exception:
        return arr

def _extract_pop_data(pop: Any) -> dict[str, Any]:
    """Extrai dados serializáveis de uma Population."""
    data: dict[str, Any] = {}
    if pop is None:
        return data
    for field in ("F", "X", "G", "H", "CV", "feasible"):
        try:
            val = pop.get(field)
            if val is not None:
                data[field] = np.asarray(val).tolist()
            else:
                data[field] = None
        except Exception:
            data[field] = None
    return data


# ---------------------------------------------------------------------------
# Função worker principal (top-level, serializável, sem Qt)
# ---------------------------------------------------------------------------

def run_trial_in_process(
    *,
    algo_module: str,
    algo_class_name: str,
    algo_spec_id: str,
    algo_spec_name: str,
    problem_source: str,
    problem_name: str,
    problem_module: str,
    problem_class_name: str,
    problem_kwargs: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
    max_fe: int,
    run_index: int,
    n_runs: int,
    project_root: str,
) -> dict[str, Any]:
    """
    Executa um trial completo em subprocesso isolado.

    Retorna dict serializável com resultados brutos.
    Métricas são calculadas no processo principal (QThread).
    """
    # Validate project_root before inserting into sys.path
    _project_path = Path(project_root).resolve()
    if not _project_path.is_dir():
        raise RuntimeError(
            f"project_root is not a valid directory: {project_root}"
        )
    _root_str = str(_project_path)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from optimize import minimize
    from termination import get_termination

    if not _is_allowed_worker_module(problem_module, _ALLOWED_PROBLEM_PREFIXES):
        raise RuntimeError(
            f"Módulo de problema não permitido: '{problem_module}'. "
            f"Prefixos aceitos: {_ALLOWED_PROBLEM_PREFIXES}"
        )
    if not _is_allowed_worker_module(algo_module, _ALLOWED_ALGO_PREFIXES):
        raise RuntimeError(
            f"Módulo de algoritmo não permitido: '{algo_module}'. "
            f"Prefixos aceitos: {_ALLOWED_ALGO_PREFIXES}"
        )

    # ---- Criar problema ----
    problem = None
    backend_code = "cpu"
    backend_label = "CPU"
    if problem_source in ("emopylab", "builtin", "local", "pymoo"):
        try:
            from problems import get_problem
            problem = get_problem(problem_name, **problem_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Falha ao criar problema '{problem_name}': {exc}") from exc
    else:
        try:
            mod = importlib.import_module(problem_module)
            cls = getattr(mod, problem_class_name)
            if hasattr(cls, "__call__") and not inspect.isclass(cls):
                problem = cls(runtime_cfg)
            else:
                # Instanciar problema local
                problem_sig = inspect.signature(cls.__init__)
                problem_kwargs_filtered = {}
                for pname, pparam in list(problem_sig.parameters.items())[1:]:
                    if pparam.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    if pname in runtime_cfg:
                        problem_kwargs_filtered[pname] = runtime_cfg[pname]
                    elif pname in problem_kwargs:
                        problem_kwargs_filtered[pname] = problem_kwargs[pname]
                problem = cls(**problem_kwargs_filtered)
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao criar problema local '{problem_module}.{problem_class_name}': {exc}"
            ) from exc

    if problem is None:
        raise RuntimeError(f"Problema não pôde ser criado: {problem_name}")

    # ---- Criar algoritmo ----
    try:
        algo_mod = importlib.import_module(algo_module)
        algo_cls = getattr(algo_mod, algo_class_name)
        algorithm = _worker_instantiate_algorithm(algo_cls, runtime_cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao criar algoritmo '{algo_module}.{algo_class_name}': {exc}"
        ) from exc

    # ---- Executar minimize ----
    termination = get_termination("n_eval", max_fe)
    start = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=seed,
        verbose=False,
        save_history=False,
        copy_algorithm=False,
    )
    elapsed = time.perf_counter() - start

    # ---- Extrair resultados serializáveis ----
    final_F = result.F
    final_front = _extract_front(final_F)

    final_pop = getattr(result, "pop", None)
    if final_pop is None:
        final_pop = getattr(getattr(result, "algorithm", None), "pop", None)

    pop_data = _extract_pop_data(final_pop)

    # Se final_front vazio, tenta do pop
    if final_front.size == 0 and pop_data.get("F") is not None:
        final_front = _extract_front(np.array(pop_data["F"]))

    evaluations = -1
    try:
        evaluations = int(result.algorithm.evaluator.n_eval)
    except Exception:
        pass

    backend_effective = str(getattr(algorithm, "array_backend_effective", "numpy")).strip().lower()
    actual_gpu = bool(getattr(algorithm, "use_gpu", False))
    if backend_effective == "mlx":
        backend_code = "mlx"
        backend_label = "MLX"
    elif actual_gpu or backend_effective == "jax":
        backend_code = "gpu"
        backend_label = "GPU"

    return {
        "algo_spec_id": algo_spec_id,
        "algo_spec_name": algo_spec_name,
        "run_index": run_index,
        "seed": seed,
        "time_s": float(elapsed),
        "evaluations": evaluations,
        "backend_code": backend_code,
        "backend_label": backend_label,
        "final_front": final_front.tolist() if final_front is not None else [],
        "final_population": pop_data,
        "final_F": final_F.tolist() if final_F is not None else None,
    }
