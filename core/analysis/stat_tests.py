from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from scipy.stats import friedmanchisquare as _scipy_friedmanchisquare
    from scipy.stats import wilcoxon as _scipy_wilcoxon

    SCIPY_STATS_AVAILABLE = True
    SCIPY_STATS_ERROR = ""
except Exception as exc:  # noqa: BLE001
    _scipy_friedmanchisquare = None
    _scipy_wilcoxon = None
    SCIPY_STATS_AVAILABLE = False
    SCIPY_STATS_ERROR = str(exc)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        return np.asarray([], dtype=float)
    return data[np.isfinite(data)]


def _mean_std(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    data = _finite_array(values)
    if data.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(data)), float(np.std(data, ddof=0))


def _rank_average(values: Sequence[float] | np.ndarray, *, descending: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=float).reshape(-1)
    if descending:
        data = -data
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(data.size, dtype=float)
    i = 0
    while i < data.size:
        j = i
        left = float(data[order[i]])
        while j + 1 < data.size and math.isclose(
            float(data[order[j + 1]]),
            left,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _rank_biserial(oriented_diff: np.ndarray) -> float:
    diff = np.asarray(oriented_diff, dtype=float).reshape(-1)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    non_zero = ~np.isclose(diff, 0.0, rtol=1e-12, atol=1e-12)
    diff = diff[non_zero]
    if diff.size == 0:
        return 0.0
    ranks = _rank_average(np.abs(diff))
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    total = w_plus + w_minus
    if not math.isfinite(total) or total <= 0.0:
        return float("nan")
    return (w_plus - w_minus) / total


def _format_float(value: float, *, fmt: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return format(float(value), fmt)


def run_wilcoxon(
    values_algo: Sequence[float],
    values_ref: Sequence[float],
    *,
    alpha: float = 0.05,
    higher_better: bool = False,
    min_samples: int = 5,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": bool(SCIPY_STATS_AVAILABLE),
        "n": 0,
        "mean": float("nan"),
        "std": float("nan"),
        "ref_mean": float("nan"),
        "ref_std": float("nan"),
        "p_value": float("nan"),
        "effect": float("nan"),
        "decision": "?",
        "error": "",
    }

    algo = np.asarray(values_algo, dtype=float).reshape(-1)
    ref = np.asarray(values_ref, dtype=float).reshape(-1)
    if algo.size == 0 or ref.size == 0:
        result["error"] = "No data available for Wilcoxon."
        return result

    n = min(algo.size, ref.size)
    algo = algo[:n]
    ref = ref[:n]
    valid = np.isfinite(algo) & np.isfinite(ref)
    algo = algo[valid]
    ref = ref[valid]
    result["n"] = int(algo.size)
    result["mean"], result["std"] = _mean_std(algo)
    result["ref_mean"], result["ref_std"] = _mean_std(ref)

    min_required = max(2, int(min_samples))
    if algo.size < min_required:
        result["error"] = f"Insufficient paired samples (n={algo.size}, required>={min_required})."
        return result

    oriented_diff = (algo - ref) if higher_better else (ref - algo)
    result["effect"] = _rank_biserial(oriented_diff)

    if np.all(np.isclose(oriented_diff, 0.0, rtol=1e-12, atol=1e-12)):
        result["p_value"] = 1.0
        result["decision"] = "="
        return result

    if not SCIPY_STATS_AVAILABLE or _scipy_wilcoxon is None:
        result["error"] = "scipy is not available."
        return result

    try:
        _stat, p_raw = _scipy_wilcoxon(
            algo,
            ref,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
        )
        p_value = _float_or_nan(p_raw)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result

    result["p_value"] = p_value
    if not math.isfinite(p_value):
        result["error"] = "Wilcoxon returned non-finite p-value."
        return result

    if p_value >= float(alpha):
        result["decision"] = "="
        return result

    diff_mean = float(np.mean(oriented_diff))
    if diff_mean > 0:
        result["decision"] = "+"
    elif diff_mean < 0:
        result["decision"] = "-"
    else:
        result["decision"] = "="
    return result


def run_friedman(
    blocks: Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    higher_better: bool = False,
    min_blocks: int = 5,
) -> dict[str, Any]:
    data = np.asarray(blocks, dtype=float)
    if data.ndim != 2:
        data = np.asarray([], dtype=float).reshape(0, 0)

    if data.size > 0:
        valid_rows = np.all(np.isfinite(data), axis=1)
        data = data[valid_rows]

    n_blocks = int(data.shape[0]) if data.ndim == 2 else 0
    n_algorithms = int(data.shape[1]) if data.ndim == 2 else 0

    result: dict[str, Any] = {
        "available": bool(SCIPY_STATS_AVAILABLE),
        "n_blocks": n_blocks,
        "n_algorithms": n_algorithms,
        "p_value": float("nan"),
        "kendall_w": float("nan"),
        "avg_ranks": [],
        "decisions": [],
        "error": "",
    }

    if n_algorithms < 3:
        result["error"] = "Friedman requires at least 3 algorithms."
        return result

    min_required = max(2, int(min_blocks))
    if n_blocks < min_required:
        result["error"] = f"Insufficient aligned blocks (n={n_blocks}, required>={min_required})."
        return result

    rank_sum: np.ndarray = np.zeros(n_algorithms, dtype=float)
    for row in data:
        rank_sum += _rank_average(row, descending=higher_better)
    avg_ranks = rank_sum / float(n_blocks)
    result["avg_ranks"] = [float(v) for v in avg_ranks]

    centered = rank_sum - (n_blocks * (n_algorithms + 1) / 2.0)
    denom = (n_blocks**2) * (n_algorithms**3 - n_algorithms)
    if denom > 0:
        result["kendall_w"] = float(12.0 * np.sum(centered * centered) / denom)

    if not SCIPY_STATS_AVAILABLE or _scipy_friedmanchisquare is None:
        result["error"] = "scipy is not available."
        result["decisions"] = ["="] * n_algorithms
        return result

    try:
        cols = [np.asarray(data[:, i], dtype=float) for i in range(n_algorithms)]
        _stat, p_raw = _scipy_friedmanchisquare(*cols)
        p_value = _float_or_nan(p_raw)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        result["decisions"] = ["="] * n_algorithms
        return result

    result["p_value"] = p_value
    if not math.isfinite(p_value) or p_value >= float(alpha):
        result["decisions"] = ["="] * n_algorithms
        return result

    best_rank = float(np.min(avg_ranks))
    worst_rank = float(np.max(avg_ranks))
    decisions: list[str] = []
    for rank in avg_ranks:
        if math.isclose(float(rank), best_rank, rel_tol=1e-12, abs_tol=1e-12):
            decisions.append("+")
        elif math.isclose(float(rank), worst_rank, rel_tol=1e-12, abs_tol=1e-12):
            decisions.append("-")
        else:
            decisions.append("=")
    result["decisions"] = decisions
    return result


def summarize_stat_results(
    *,
    method: str,
    metric_name: str,
    alpha: float,
    backend_label: str,
    rows: Sequence[dict[str, Any]],
    higher_better: bool,
    reference_algorithm: str = "",
    note: str = "",
    global_p_value: float | None = None,
    global_effect: float | None = None,
) -> dict[str, Any]:
    method_norm = str(method).strip().lower() or "none"
    if method_norm == "wilcoxon":
        method_label = "Wilcoxon vs reference"
    elif method_norm == "friedman":
        method_label = "Friedman rank global"
    else:
        method_label = "Detailed trials"

    direction_text = "higher-better" if higher_better else "lower-better"
    header_parts = [
        "EmoPyLab | Statistical Validation",
        f"Method={method_label}",
        f"Metric={metric_name}",
        f"alpha={float(alpha):.3g}",
        f"Backend={backend_label}",
        f"Direction={direction_text}",
    ]
    if reference_algorithm:
        header_parts.append(f"Reference={reference_algorithm}")
    if global_p_value is not None and math.isfinite(float(global_p_value)):
        header_parts.append(f"global p={float(global_p_value):.4g}")
    if global_effect is not None and math.isfinite(float(global_effect)):
        header_parts.append(f"global effect={float(global_effect):.4g}")
    if note:
        header_parts.append(note)
    header_text = " | ".join(header_parts)

    normalized_rows: list[dict[str, Any]] = []
    for raw in rows:
        mean = _float_or_nan(raw.get("mean"))
        std = _float_or_nan(raw.get("std"))
        p_value = _float_or_nan(raw.get("p_value"))
        effect = _float_or_nan(raw.get("effect"))
        normalized_rows.append(
            {
                "algorithm": str(raw.get("algorithm", "")),
                "n": int(raw.get("n", 0)),
                "mean": mean,
                "std": std,
                "p_value": p_value,
                "effect": effect,
                "decision": str(raw.get("decision", "?")),
                "mean_std_text": "--" if not math.isfinite(mean) else f"{mean:.4g} +- {std:.2g}",
                "p_value_text": _format_float(p_value, fmt=".4g"),
                "effect_text": _format_float(effect, fmt=".4g"),
            }
        )

def export_latex_statistical_table(
    table_data: dict[str, Any],
    caption: str = "Statistical comparison of algorithm performance.",
    label: str = "tab:stat_comparison",
) -> str:
    """Converts statistical test outcomes into publication-ready IEEE/Elsevier LaTeX tables."""
    rows = table_data.get("rows", [])
    metric = table_data.get("metric", "Indicator")
    method = table_data.get("method", "Statistical Test")

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{Sample Size ($N$)} & \textbf{Mean $\pm$ Std. Dev.} & \textbf{$p$-value} & \textbf{Effect Size} & \textbf{Decision} \\",
        r"\midrule",
    ]

    for row in rows:
        algo = row.get("algorithm", "").replace("_", r"\_")
        n = row.get("n", "--")
        mean_std = row.get("mean_std_text", "--").replace("+-", r"\pm")
        p_val = row.get("p_value_text", "--")
        effect = row.get("effect_text", "--")
        decision = row.get("decision", "=")

        dec_symbol = decision
        if decision == "+":
            dec_symbol = r"\textbf{+}"
        elif decision == "-":
            dec_symbol = r"\textbf{--}"
        elif decision == "=":
            dec_symbol = r"$\approx$"

        latex_lines.append(f"{algo} & {n} & {mean_std} & {p_val} & {effect} & {dec_symbol} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(latex_lines)


def analyze_directory_and_export_latex(results_dir: str, export_latex: bool = True) -> str:
    """Scans a directory of sidecar JSON metadata files and computes summary stats and LaTeX table."""
    import json
    from pathlib import Path

    dir_path = Path(results_dir)
    if not dir_path.exists():
        print(f"[!] Directory '{results_dir}' does not exist.")
        return ""

    meta_files = list(dir_path.glob("meta_*.json"))
    if not meta_files:
        print(f"[!] No sidecar metadata files (meta_*.json) found in {results_dir}.")
        return ""

    # Group by algorithm and metric
    data_by_algo: dict[str, dict[str, list[float]]] = {}
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
                algo = d.get("algorithm_name", "Unknown")
                metrics = d.get("metrics", {})
                if algo not in data_by_algo:
                    data_by_algo[algo] = {}
                for m_name, m_val in metrics.items():
                    if m_name not in data_by_algo[algo]:
                        data_by_algo[algo][m_name] = []
                    data_by_algo[algo][m_name].append(float(m_val))
        except Exception:
            continue

    # Build summary rows for the primary metric
    primary_metric = "igd_plus"
    rows = []
    for algo, m_dict in sorted(data_by_algo.items()):
        vals = m_dict.get(primary_metric, [])
        if vals:
            mean, std = _mean_std(vals)
            rows.append({
                "algorithm": algo,
                "n": len(vals),
                "mean": mean,
                "std": std,
                "p_value": float("nan"),
                "effect": float("nan"),
                "decision": "=",
                "mean_std_text": f"{mean:.4e} +- {std:.2e}",
                "p_value_text": "--",
                "effect_text": "--",
            })

    table_data = {
        "method": "Wilcoxon & Descriptive Synthesis",
        "metric": primary_metric,
        "rows": rows,
    }

    return {
        "method": method_norm,
        "metric_name": metric_name,
        "alpha": float(alpha),
        "backend_label": backend_label,
        "higher_better": bool(higher_better),
        "reference_algorithm": reference_algorithm,
        "columns": ["Algorithm", "N", "Mean+-Std", "p-value", "Effect", "Decision"],
        "rows": normalized_rows,
        "header_text": header_text,
        "note": note,
        "global_p_value": _float_or_nan(global_p_value) if global_p_value is not None else float("nan"),
        "global_effect": _float_or_nan(global_effect) if global_effect is not None else float("nan"),
    }


def export_latex_statistical_table(
    table_data: dict[str, Any],
    caption: str = "Statistical comparison of algorithm performance.",
    label: str = "tab:stat_comparison",
) -> str:
    """Converts statistical test outcomes into publication-ready IEEE/Elsevier LaTeX tables."""
    rows = table_data.get("rows", [])
    metric = table_data.get("metric", "Indicator")
    method = table_data.get("method", "Statistical Test")

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{Sample Size ($N$)} & \textbf{Mean $\pm$ Std. Dev.} & \textbf{$p$-value} & \textbf{Effect Size} & \textbf{Decision} \\",
        r"\midrule",
    ]

    for row in rows:
        algo = row.get("algorithm", "").replace("_", r"\_")
        n = row.get("n", "--")
        mean_std = row.get("mean_std_text", "--").replace("+-", r"\pm")
        p_val = row.get("p_value_text", "--")
        effect = row.get("effect_text", "--")
        decision = row.get("decision", "=")

        dec_symbol = decision
        if decision == "+":
            dec_symbol = r"\textbf{+}"
        elif decision == "-":
            dec_symbol = r"\textbf{--}"
        elif decision == "=":
            dec_symbol = r"$\approx$"

        latex_lines.append(f"{algo} & {n} & {mean_std} & {p_val} & {effect} & {dec_symbol} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(latex_lines)


def analyze_directory_and_export_latex(results_dir: str, export_latex: bool = True) -> str:
    """Scans a directory of sidecar JSON metadata files and computes summary stats and LaTeX table."""
    import json
    from pathlib import Path

    dir_path = Path(results_dir)
    if not dir_path.exists():
        print(f"[!] Directory '{results_dir}' does not exist.")
        return ""

    meta_files = list(dir_path.glob("meta_*.json"))
    if not meta_files:
        print(f"[!] No sidecar metadata files (meta_*.json) found in {results_dir}.")
        return ""

    # Group by algorithm and metric
    data_by_algo: dict[str, dict[str, list[float]]] = {}
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
                algo = d.get("algorithm_name", "Unknown")
                metrics = d.get("metrics", {})
                if algo not in data_by_algo:
                    data_by_algo[algo] = {}
                for m_name, m_val in metrics.items():
                    if m_name not in data_by_algo[algo]:
                        data_by_algo[algo][m_name] = []
                    data_by_algo[algo][m_name].append(float(m_val))
        except Exception:
            continue

    # Build summary rows for the primary metric
    primary_metric = "igd_plus"
    rows = []
    for algo, m_dict in sorted(data_by_algo.items()):
        vals = m_dict.get(primary_metric, [])
        if vals:
            mean, std = _mean_std(vals)
            rows.append({
                "algorithm": algo,
                "n": len(vals),
                "mean": mean,
                "std": std,
                "p_value": float("nan"),
                "effect": float("nan"),
                "decision": "=",
                "mean_std_text": f"{mean:.4e} +- {std:.2e}",
                "p_value_text": "--",
                "effect_text": "--",
            })

    table_data = {
        "method": "Wilcoxon & Descriptive Synthesis",
        "metric": primary_metric,
        "rows": rows,
    }

    latex_table = export_latex_statistical_table(table_data)
    print(latex_table)
    return latex_table

