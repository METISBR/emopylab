"""Headless CLI entry point for EmoPyLab.

Allows researchers to run deterministic EMO/MaOP benchmark campaigns in parallel,
compute quality indicators, and export statistical tables directly
from the terminal or HPC batch environments without initializing PySide6.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure root directory is on sys.path for standalone invocations
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.engine.runner import run_single_optimization
from core.execution.reproducibility import (
    SEED_MODE_FIXED,
    SEED_MODE_RANDOM,
    SEED_MODE_SEQUENCE,
    create_run_sidecar_metadata,
    generate_experiment_manifest,
    plan_run_seeds,
)
from version import __version__


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="emopylab",
        description=f"EmoPyLab v{__version__}: Unified Scientific Laboratory for EMO/MaOP Research",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run (Headless single/multi-run optimization)
    run_parser = subparsers.add_parser("run", help="Execute an optimization run or benchmark campaign headlessly")
    run_parser.add_argument("--algo", type=str, required=True, help="Algorithm name (e.g. NSGA2, GCS_MaOEA, LARC_NSGA3)")
    run_parser.add_argument("--problem", type=str, required=True, help="Problem name (e.g. ZDT1, DTLZ2, WFG4, MaF1)")
    run_parser.add_argument("--n-var", type=int, default=None, help="Number of decision variables")
    run_parser.add_argument("--n-obj", type=int, default=None, help="Number of objectives")
    run_parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    run_parser.add_argument("--n-gen", type=int, default=250, help="Number of generations")
    run_parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    run_parser.add_argument("--seed-mode", choices=["random", "fixed", "sequence"], default="sequence", help="Seed sequence planning mode")
    run_parser.add_argument("--n-runs", type=int, default=1, help="Number of independent runs")
    run_parser.add_argument("--n-workers", type=int, default=1, help="Number of parallel worker processes (default: 1)")
    run_parser.add_argument("--out-dir", type=str, default="results", help="Output directory for sidecar JSON and Pareto fronts")

    # Command: list (Inspect available algorithms and problems)
    list_parser = subparsers.add_parser("list", help="List registered catalog elements")
    list_parser.add_argument("--category", choices=["algorithms", "problems", "metrics", "taxonomy"], default="algorithms")

    # Command: stats (Statistical tests and LaTeX generation)
    stats_parser = subparsers.add_parser("stats", help="Compute statistical tests from experimental runs")
    stats_parser.add_argument("--results-dir", type=str, required=True, help="Directory containing run results")
    stats_parser.add_argument("--export-latex", action="store_true", help="Generate publication-ready LaTeX tables")

    return parser.parse_args(args)


def list_catalog(category: str) -> None:
    root = ROOT_DIR
    if category == "algorithms":
        algo_dir = root / "algorithms"
        algos = sorted([d.name for d in algo_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))])
        print(f"EmoPyLab Registered Algorithms ({len(algos)} total):")
        for i, a in enumerate(algos, 1):
            print(f"  [{i:03d}] {a}")
    elif category == "problems":
        prob_dir = root / "problems"
        probs = sorted([p.stem for p in prob_dir.rglob("*.py") if not p.name.startswith((".", "_"))])
        print(f"EmoPyLab Registered Benchmark Problems ({len(probs)} total):")
        for i, p in enumerate(probs, 1):
            print(f"  [{i:03d}] {p}")
    elif category == "metrics":
        print("EmoPyLab Quality Indicators:")
        print("  - Hypervolume (HV, Exact & Fast Monte-Carlo via JAX/MLX/CPU)")
        print("  - Generational Distance (GD, GD+)")
        print("  - Inverted Generational Distance (IGD, IGD+)")
        print("  - Averaged Hausdorff Distance (Delta_p)")
        print("  - Spacing & Spread")
    elif category == "taxonomy":
        from core.registry.taxonomy import print_taxonomy_summary
        print_taxonomy_summary()


def _worker_task(task_payload: dict[str, Any]) -> dict[str, Any]:
    """Independent worker task for parallel campaign execution."""
    algo = task_payload["algo"]
    problem = task_payload["problem"]
    pop_size = task_payload["pop_size"]
    n_gen = task_payload["n_gen"]
    run_seed = task_payload["run_seed"]
    run_idx = task_payload["run_idx"]
    n_var = task_payload["n_var"]
    n_obj = task_payload["n_obj"]
    out_dir = task_payload["out_dir"]

    out_path = Path(out_dir)
    t0 = time.perf_counter()
    opt_res = run_single_optimization(
        algorithm_name=algo,
        problem_name=problem,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=run_seed,
        n_var=n_var,
        n_obj=n_obj,
    )
    elapsed = time.perf_counter() - t0

    if opt_res.success:
        # Save Pareto front CSV
        pf_file = out_path / f"front_{algo}_{problem}_run{run_idx}_seed{run_seed}.csv"
        np.savetxt(pf_file, opt_res.F, delimiter=",", header=",".join([f"f{i+1}" for i in range(opt_res.F.shape[1])]), comments="")

        # Save sidecar metadata JSON
        sidecar = create_run_sidecar_metadata(
            algorithm_name=algo,
            problem_name=problem,
            run_index=run_idx,
            seed=run_seed,
            n_eval=pop_size * n_gen,
            n_gen=n_gen,
            pop_size=pop_size,
            runtime_seconds=elapsed,
            metrics=opt_res.metrics,
        )
        sidecar_file = out_path / f"meta_{algo}_{problem}_run{run_idx}_seed{run_seed}.json"
        with open(sidecar_file, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

        return {"success": True, "sidecar": sidecar, "run_idx": run_idx, "seed": run_seed, "elapsed": elapsed, "metrics": opt_res.metrics}
    else:
        return {"success": False, "error": opt_res.error_message, "run_idx": run_idx, "seed": run_seed}


def execute_campaign(
    algo: str,
    problem: str,
    n_runs: int,
    pop_size: int,
    n_gen: int,
    seed: int,
    seed_mode: str,
    n_var: int | None,
    n_obj: int | None,
    out_dir: str,
    n_workers: int = 1,
) -> list[dict[str, Any]]:
    """Runs a batch campaign (sequentially or in parallel) and saves sidecars + CSV/JSON summaries."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seeds = plan_run_seeds(n_runs=n_runs, seed_mode=seed_mode, seed_base=seed)
    manifest = generate_experiment_manifest({
        "algorithms": [algo],
        "problems": [problem],
        "n_runs": n_runs,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "seed_base": seed,
        "seed_mode": seed_mode,
        "n_var": n_var,
        "n_obj": n_obj,
        "n_workers": n_workers,
    })

    # Save immutable manifest
    manifest_file = out_path / f"manifest_{algo}_{problem}.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    results_summary = []
    effective_workers = max(1, min(n_workers, os.cpu_count() or 1, n_runs))
    print(f"[*] Starting campaign: {algo} on {problem} ({n_runs} runs, seed_mode={seed_mode}, workers={effective_workers})")

    tasks = [
        {
            "algo": algo,
            "problem": problem,
            "pop_size": pop_size,
            "n_gen": n_gen,
            "run_seed": r_seed,
            "run_idx": r_idx,
            "n_var": n_var,
            "n_obj": n_obj,
            "out_dir": out_dir,
        }
        for r_idx, r_seed in enumerate(seeds, start=1)
    ]

    t_start_all = time.perf_counter()

    if effective_workers == 1:
        for t in tasks:
            res = _worker_task(t)
            if res["success"]:
                results_summary.append(res["sidecar"])
                metric_str = ", ".join([f"{k}={v:.4e}" for k, v in res["metrics"].items()])
                print(f"  [Run {res['run_idx']:02d}/{n_runs:02d}] Seed {res['seed']} | {res['elapsed']:.2f}s | {metric_str}")
            else:
                print(f"  [Run {res['run_idx']:02d}/{n_runs:02d}] FAILED: {res['error']}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {executor.submit(_worker_task, t): t["run_idx"] for t in tasks}
            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                if res["success"]:
                    results_summary.append(res["sidecar"])
                    metric_str = ", ".join([f"{k}={v:.4e}" for k, v in res["metrics"].items()])
                    print(f"  [Run {res['run_idx']:02d}/{n_runs:02d}] Seed {res['seed']} | {res['elapsed']:.2f}s | {metric_str}")
                else:
                    print(f"  [Run {res['run_idx']:02d}/{n_runs:02d}] FAILED: {res['error']}")

    total_time = time.perf_counter() - t_start_all
    print(f"[+] Campaign completed in {total_time:.2f}s. Results archived to: {out_path.resolve()}")
    return results_summary


def main(args_list: list[str] | None = None) -> int:
    args = parse_args(args_list)
    if not args.command:
        print(f"EmoPyLab v{__version__} - Scientific Optimization Framework.")
        print("Use 'emopylab --help' or 'emopylab <command> --help' for CLI options.")
        print("To launch the GUI, run: emopylab-gui (or python EmoPyLab.py)")
        return 0

    if args.command == "list":
        list_catalog(args.category)
        return 0

    if args.command == "run":
        execute_campaign(
            algo=args.algo,
            problem=args.problem,
            n_runs=args.n_runs,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            seed=args.seed,
            seed_mode=args.seed_mode,
            n_var=args.n_var,
            n_obj=args.n_obj,
            out_dir=args.out_dir,
            n_workers=args.n_workers,
        )
        return 0

    if args.command == "stats":
        from core.analysis.stat_tests import analyze_directory_and_export_latex
        analyze_directory_and_export_latex(args.results_dir, export_latex=args.export_latex)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
