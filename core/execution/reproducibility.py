from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Sequence

import numpy as np

EXPERIMENT_MANIFEST_VERSION = 1
SEED_MODE_RANDOM = "random"
SEED_MODE_FIXED = "fixed"
SEED_MODE_SEQUENCE = "sequence"


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _random_seed() -> int:
    return int(np.random.default_rng().integers(1, 2_147_483_647))


def _normalize_seed_value(value: Any, default: int = 1) -> int:
    seed = _positive_int(value, default, minimum=1)
    return int(max(1, min(seed, 2_147_483_647)))


def _canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitize_config_for_manifest(config: dict[str, Any]) -> dict[str, Any]:
    safe = dict(config)
    safe.pop("__single_problem_mode__", None)
    safe.pop("__suppress_progress__", None)
    return safe


def build_seed_plan(config: dict[str, Any], total_slots: int) -> dict[str, Any]:
    slots = max(1, int(total_slots))
    mode_raw = str(config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower()
    if mode_raw not in {SEED_MODE_RANDOM, SEED_MODE_FIXED, SEED_MODE_SEQUENCE}:
        mode_raw = SEED_MODE_RANDOM

    base_seed = _normalize_seed_value(config.get("seed_base", config.get("seed", config.get("seed_value", 1))), default=1)
    step = _positive_int(config.get("seed_step", 1), 1, minimum=1)
    sequence_raw = config.get("seed_sequence", [])

    sequence: list[int] = []
    if isinstance(sequence_raw, str):
        chunks = re.split(r"[,;\s]+", sequence_raw.strip())
        sequence = [_normalize_seed_value(chunk, default=1) for chunk in chunks if chunk.strip()]
    elif isinstance(sequence_raw, (list, tuple)):
        for item in sequence_raw:
            try:
                sequence.append(_normalize_seed_value(item, default=1))
            except Exception:  # noqa: BLE001
                continue

    if mode_raw == SEED_MODE_FIXED:
        seeds = [base_seed for _ in range(slots)]
    elif mode_raw == SEED_MODE_SEQUENCE:
        if sequence:
            seeds = [sequence[idx % len(sequence)] for idx in range(slots)]
        else:
            seeds = [_normalize_seed_value(base_seed + idx * step, default=1) for idx in range(slots)]
    else:
        seeds = [_random_seed() for _ in range(slots)]

    return {
        "mode": mode_raw,
        "deterministic": mode_raw in {SEED_MODE_FIXED, SEED_MODE_SEQUENCE},
        "base_seed": base_seed,
        "step": int(step),
        "provided_sequence": sequence,
        "seeds": seeds,
    }


def plan_run_seeds(
    n_runs: int,
    seed_mode: str = SEED_MODE_RANDOM,
    seed_base: int = 1,
    seed_step: int = 1,
    seed_sequence: Sequence[int] | None = None,
) -> list[int]:
    """Generates an explicit list of integer seeds for n_runs given the seed mode."""
    cfg = {
        "seed_mode": seed_mode,
        "seed_base": seed_base,
        "seed_step": seed_step,
        "seed_sequence": seed_sequence or [],
    }
    plan = build_seed_plan(cfg, total_slots=n_runs)
    return plan["seeds"]


def build_execution_manifest(
    *,
    config: dict[str, Any],
    seed_plan: dict[str, Any],
    selected_problem_ids: list[str],
    selected_algorithm_ids: list[str],
    selected_metric_ids: list[str],
    execution_backend: str = "cpu",
    execution_backend_label: str = "CPU (NumPy)",
) -> dict[str, Any]:
    if execution_backend == "mlx":
        try:
            from core.execution.backend_runtime import detect_mlx_runtime
            info = detect_mlx_runtime()
            if info.get("mlx_ok"):
                device_name = info.get("device") or "Apple Silicon"
                if device_name not in execution_backend_label:
                    execution_backend_label = f"{execution_backend_label} ({device_name})"
        except Exception:
            pass

    manifest = {
        "manifest_version": EXPERIMENT_MANIFEST_VERSION,
        "timestamp_iso": datetime.now().astimezone().isoformat(),
        "config": _sanitize_config_for_manifest(config),
        "selection": {
            "problem_ids": list(selected_problem_ids),
            "algorithm_ids": list(selected_algorithm_ids),
            "metric_ids": list(selected_metric_ids),
        },
        "seed_plan": {
            "mode": seed_plan.get("mode", SEED_MODE_RANDOM),
            "deterministic": bool(seed_plan.get("deterministic", False)),
            "base_seed": int(seed_plan.get("base_seed", 1)),
            "step": int(seed_plan.get("step", 1)),
            "provided_sequence": list(seed_plan.get("provided_sequence", [])),
            "seeds": list(seed_plan.get("seeds", [])),
        },
        "execution_backend": execution_backend,
        "execution_backend_label": execution_backend_label,
    }
    manifest_json = _canonical_json_dumps(manifest)
    sha = _sha256_text(manifest_json)
    manifest["manifest_sha256"] = sha
    manifest["config_hash"] = sha
    return manifest


def create_run_sidecar_metadata(
    *,
    algorithm_name: str,
    problem_name: str,
    run_index: int,
    seed: int,
    n_eval: int,
    n_gen: int,
    pop_size: int,
    runtime_seconds: float,
    metrics: dict[str, float] | None = None,
    backend: str = "cpu",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generates structured sidecar metadata for a single completed run."""
    sidecar = {
        "algorithm_name": str(algorithm_name),
        "problem_name": str(problem_name),
        "run_index": int(run_index),
        "seed": int(seed),
        "n_eval": int(n_eval),
        "n_gen": int(n_gen),
        "pop_size": int(pop_size),
        "runtime_seconds": float(runtime_seconds),
        "backend": str(backend),
        "metrics": dict(metrics or {}),
        "timestamp_iso": datetime.now().astimezone().isoformat(),
    }
    if extra:
        sidecar["extra"] = extra
    return sidecar


def generate_experiment_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Generates an immutable execution manifest with SHA-256 hash for a given configuration."""
    seed_plan = build_seed_plan(config, total_slots=config.get("n_runs", 1))
    return build_execution_manifest(
        config=config,
        seed_plan=seed_plan,
        selected_problem_ids=config.get("problems", []),
        selected_algorithm_ids=config.get("algorithms", []),
        selected_metric_ids=config.get("metrics", []),
        execution_backend=config.get("backend", "cpu"),
        execution_backend_label=config.get("backend_label", "CPU (NumPy)"),
    )
