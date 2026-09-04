from __future__ import annotations

import ast
import builtins
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast
from core.llm.local_llm import LocalLLMClient


class LLMFormulationService:
    """LLM-assisted formulation helper for EmoPyLab artifact generation and validation."""

    TEMPLATE_PROVIDER = "template"
    LOCAL_PROVIDER = "local_qwen"
    DEFAULT_PROVIDER = LOCAL_PROVIDER

    DEFAULT_LOCAL_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    METRIC_MODE_CANONICAL_WRAPPER = "canonical_wrapper"
    METRIC_MODE_GITHUB_CONVERTED = "github_converted"

    _BLOCKED_AST_NODES = (
        ast.With,
        ast.AsyncWith,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
        ast.Lambda,
    )
    _ALLOWED_IMPORT_ROOTS = {"numpy", "core", "jax", "typing", "math", "metrics", "problems"}
    _PROMPT_SIGNATURE_MARKER = "emopylab prompt engineering"
    _BLOCKED_CALL_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
    _BLOCKED_ATTR_CALLS = {
        ("os", "system"),
        ("os", "popen"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "check_output"),
        ("requests", "get"),
        ("requests", "post"),
    }
    _SAFE_BUILTIN_NAMES = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "float",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "object",
        "pow",
        "range",
        "round",
        "set",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "ValueError",
        "zip",
    }

    @staticmethod
    def _slugify_problem_name(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
        if not cleaned:
            cleaned = "GeneratedProblem"
        if cleaned[0].isdigit():
            cleaned = f"P_{cleaned}"
        return cleaned

    @classmethod
    def _problem_symbol_upper(cls, name: str, *, default: str = "GENERATEDPROBLEM") -> str:
        raw = cls._slugify_problem_name(name or default)
        return raw.upper()

    @staticmethod
    def _slugify_module_name(name: str, *, default: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "")).strip("_")
        if not cleaned:
            cleaned = default
        if cleaned[0].isdigit():
            cleaned = f"m_{cleaned}"
        return cleaned.lower()

    @staticmethod
    def _camelize(name: str, *, default: str) -> str:
        raw = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "")).strip("_")
        if not raw:
            raw = default
        parts = [p for p in raw.split("_") if p]
        if not parts:
            parts = [default]
        out = "".join(p[:1].upper() + p[1:] for p in parts)
        if out and out[0].isdigit():
            out = f"P{out}"
        return out

    @staticmethod
    def _coerce_positive_int_or_default(value: Any, *, default: int) -> int:
        try:
            if value is None:
                raise ValueError("none")
            parsed = int(value)
            if parsed <= 0:
                raise ValueError("non-positive")
            return parsed
        except Exception:
            return int(max(1, int(default)))

    @classmethod
    def _normalize_problem_ui_defaults(cls, n_var: Any, n_obj: Any) -> tuple[int, int]:
        """Use caller-provided defaults when present; fall back to 30/2 only for missing/invalid values."""
        return (
            cls._coerce_positive_int_or_default(n_var, default=30),
            cls._coerce_positive_int_or_default(n_obj, default=2),
        )

    @classmethod
    def _normalize_metric_generation_mode(cls, mode: Any) -> str:
        _ = mode
        # Metric generation is fixed to GitHub-converted mode.
        return cls.METRIC_MODE_GITHUB_CONVERTED

    @staticmethod
    def _coerce_optional_positive_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            parsed = int(value)
            if parsed <= 0:
                return None
            return parsed
        except Exception:
            return None

    @staticmethod
    def _prompt_has_explicit_n_var(prompt: str) -> bool:
        text = str(prompt or "")
        if not text:
            return False
        patterns = (
            r"\bn_var\s*=\s*\d+\b",
            r"\b\d+\s+decision\s+variables?\b",
            r"\buse\s+\d+\s+decision\s+variables?\b",
            r"\bwith\s+\d+\s+variables?\b",
            r"\bdimension(?:s)?\s*[:=]?\s*\d+\b",
        )
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _prompt_has_explicit_n_obj_exact(prompt: str) -> bool:
        text = str(prompt or "")
        if not text:
            return False
        patterns = (
            r"\bn_obj\s*=\s*\d+\b",
            r"\bm\s*=\s*\d+\b",
            r"\b\d+\s+objectives?\b",
            r"\bwith\s+\d+\s+objectives?\b",
        )
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    @classmethod
    def _apply_spec_suggested_problem_defaults(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        spec_report: dict[str, Any] | None,
        n_var_default: int,
        n_obj_default: int,
    ) -> tuple[int, int]:
        if str(artifact_type or "").strip().lower() != "problem":
            return int(n_var_default), int(n_obj_default)
        spec = spec_report if isinstance(spec_report, dict) else {}
        suggested_n_var = cls._coerce_optional_positive_int(spec.get("suggested_n_var_default"))
        suggested_n_obj = cls._coerce_optional_positive_int(spec.get("suggested_n_obj_default"))

        out_n_var = int(n_var_default)
        out_n_obj = int(n_obj_default)

        if suggested_n_var is not None and (not cls._prompt_has_explicit_n_var(prompt)):
            out_n_var = int(suggested_n_var)
        if suggested_n_obj is not None and (not cls._prompt_has_explicit_n_obj_exact(prompt)):
            out_n_obj = int(suggested_n_obj)
        return out_n_var, out_n_obj

    @staticmethod
    def _ensure_numpy_import(code: str) -> str:
        text = str(code or "")
        if re.search(r"^\s*import\s+numpy\s+as\s+np\s*$", text, flags=re.MULTILINE):
            return text
        return "import numpy as np\n" + text

    @staticmethod
    def _force_problem_class_name(code: str, expected_class_name: str) -> str:
        text = str(code or "")
        expected = str(expected_class_name or "").strip()
        if not text or not expected:
            return text
        return re.sub(
            r"(?m)^(\s*class\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*\()",
            lambda m: f"{m.group(1)}{expected}{m.group(3)}",
            text,
            count=1,
        )

    @staticmethod
    def _normalize_problem_init_keywords(code: str) -> str:
        text = str(code or "")
        # Older/generated snippets may use n_constr; pymoo current API expects n_ieq_constr.
        text = re.sub(r"(?<![A-Za-z0-9_])n_constr\s*=", "n_ieq_constr=", text)
        return text

    @classmethod
    def _prompt_engineering_signature_comment(cls) -> str:
        year = int(datetime.now().year)
        return f"# Made by EmoPyLab {year}."

    @classmethod
    def _apply_prompt_signature(cls, code: str) -> str:
        text = str(code or "").replace("\r\n", "\n").strip()
        if not text:
            return text
        if cls._PROMPT_SIGNATURE_MARKER in text.lower():
            return text
        sig = cls._prompt_engineering_signature_comment()
        # Keep any existing reference/comment block from conversions; prepend only one signature line.
        return f"{sig}\n{text}"

    @classmethod
    def _inject_method_before_evaluate(cls, code: str, method_src: str) -> str:
        text = str(code or "")
        if "_calc_pareto_front" in text:
            return text
        marker = re.search(r"(?m)^(\s*)def\s+_evaluate\s*\(", text)
        if not marker:
            return text
        indent = marker.group(1)
        method = str(method_src).rstrip() + "\n\n"
        # Ensure method indentation matches class body indentation.
        if not method.startswith(indent):
            method = "\n".join((indent + line if line.strip() else line) for line in method.splitlines()) + "\n\n"
        pos = marker.start()
        return text[:pos] + method + text[pos:]
    @staticmethod
    def _generic_pf_method_source() -> str:
        return (
            "def _calc_pareto_front(self, n_pareto_points=200):\n"
            "    # Approximate PF fallback inserted by EmoPyLab for metric compatibility.\n"
            "    # Replace with a true/reference Pareto front whenever available.\n"
            "    n_pf = int(max(20, n_pareto_points))\n"
            "    n_samples = int(max(2000, n_pf * 40))\n"
            "    rng = np.random.default_rng(1)\n"
            "    xl = np.asarray(getattr(self, 'xl', 0.0), dtype=float)\n"
            "    xu = np.asarray(getattr(self, 'xu', 1.0), dtype=float)\n"
            "    if xl.ndim == 0:\n"
            "        xl = np.full(int(self.n_var), float(xl))\n"
            "    if xu.ndim == 0:\n"
            "        xu = np.full(int(self.n_var), float(xu))\n"
            "    X = rng.uniform(xl, xu, size=(n_samples, int(self.n_var)))\n"
            "    out = {}\n"
            "    self._evaluate(X, out)\n"
            "    F = np.asarray(out.get('F', []), dtype=float)\n"
            "    if F.ndim != 2 or F.shape[0] == 0:\n"
            "        return np.empty((0, int(getattr(self, 'n_obj', 0))), dtype=float)\n"
            "    if 'G' in out and out['G'] is not None:\n"
            "        G = np.asarray(out['G'], dtype=float)\n"
            "        if G.ndim == 1:\n"
            "            G = G[:, None]\n"
            "        if G.shape[0] == F.shape[0]:\n"
            "            feasible = np.all(G <= 0.0, axis=1)\n"
            "            if np.any(feasible):\n"
            "                F = F[feasible]\n"
            "    if F.shape[0] == 0:\n"
            "        return np.empty((0, int(getattr(self, 'n_obj', 0))), dtype=float)\n"
            "    keep = np.ones(F.shape[0], dtype=bool)\n"
            "    for i in range(F.shape[0]):\n"
            "        if not keep[i]:\n"
            "            continue\n"
            "        fi = F[i]\n"
            "        for j in range(F.shape[0]):\n"
            "            if i == j or not keep[j]:\n"
            "                continue\n"
            "            fj = F[j]\n"
            "            if np.all(fj <= fi) and np.any(fj < fi):\n"
            "                keep[i] = False\n"
            "                break\n"
            "    PF = F[keep]\n"
            "    if PF.shape[0] <= n_pf:\n"
            "        return PF\n"
            "    order = np.argsort(PF[:, 0], kind='mergesort')\n"
            "    PF = PF[order]\n"
            "    idx = np.linspace(0, PF.shape[0] - 1, n_pf).astype(int)\n"
            "    return PF[idx]"
        )

    @staticmethod
    def _looks_like_native_pymoo_problem_wrapper(code: str) -> bool:
        text = str(code or "")
        tl = text.lower()
        if not tl.strip():
            return False
        # Detect thin wrappers around local canonical benchmark modules or pymoo benchmark modules.
        wrapper_import_hints = (
            "from problems.multi.zdt import",
            "from problems.many.dtlz import",
            "from problems.many.zcat import",
            "from pymoo.problems",
        )
        if any(hint in tl for hint in wrapper_import_hints):
            if "_canonicalproblem" in tl or re.search(r"class\s+\w+\s*\(\s*\w+\s*\)\s*:", text):
                return True
        if "_canonicalproblem" in tl:
            return True
        return False

    @staticmethod
    def _looks_like_placeholder_benchmark_fallback_text(code: str) -> bool:
        tl = str(code or "").lower()
        if not tl.strip():
            return False
        placeholder_markers = (
            "placeholder semantics",
            "could not be reliably recovered",
            "keeps the existing placeholder semantics",
            "fallback semantics",
        )
        if not any(tok in tl for tok in placeholder_markers):
            return False
        # Keep the rule narrow to benchmark/problem-generation scenarios.
        benchmark_markers = ("zcat", "zdt", "dtlz", "wfg", "cec", "uf", "cf", "maf", "lz")
        return any(tok in tl for tok in benchmark_markers)

    @staticmethod
    def _emit_stream_event(
        cb: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any] | None,
    ) -> None:
        if cb is None or not isinstance(payload, dict):
            return
        try:
            cb(payload)
        except Exception:
            pass

    @classmethod
    def _augment_problem_code_for_metrics(
        cls,
        code: str,
        *,
        base_name: str,
        class_name: str,
        n_obj: int | None = None,
    ) -> str:
        text = cls._force_problem_class_name(code, class_name)
        text = cls._normalize_problem_init_keywords(text)
        # Generic fallback PF approximation for PF-based metrics (DeltaP, GD/IGD variants, HV, etc.)
        try:
            nobj_val = int(n_obj) if n_obj is not None else None
        except Exception:
            nobj_val = None
        if (nobj_val is None or nobj_val >= 2) and "_calc_pareto_front" not in text:
            text = cls._ensure_numpy_import(text)
            text = cls._inject_method_before_evaluate(text, cls._generic_pf_method_source())
        return text

    @classmethod
    def _build_problem_template(
        cls,
        class_name: str,
        n_var: int,
        n_obj: int,
        has_constraints: bool,
    ) -> str:
        n_ieq = 1 if has_constraints else 0
        objective_lines: list[str] = []
        for idx in range(max(1, int(n_obj))):
            shift = 0.15 + 0.7 * (idx / max(1, int(n_obj) - 1)) if int(n_obj) > 1 else 0.5
            objective_lines.append(
                f"        f{idx+1} = np.sum((X - {shift:.4f}) ** 2, axis=1) + {0.05*idx:.4f} * np.sum(X, axis=1)"
            )
        stack_expr = ", ".join(f"f{i+1}" for i in range(max(1, int(n_obj))))
        constr = (
            "        g1 = np.sum(X, axis=1) - (0.5 * self.n_var)\n"
            "        out['G'] = g1[:, None]\n"
            if has_constraints
            else ""
        )
        return (
            "import numpy as np\n"
            "from core.problem import Problem\n\n\n"
            f"class {class_name}(Problem):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__(\n"
            f"            n_var={int(n_var)},\n"
            f"            n_obj={int(n_obj)},\n"
            f"            n_ieq_constr={int(n_ieq)},\n"
            "            xl=0.0,\n"
            "            xu=1.0,\n"
            "        )\n\n"
            "    def _evaluate(self, X, out, *args, **kwargs):\n"
            "        X = np.asarray(X, dtype=float)\n"
            + "\n".join(objective_lines)
            + "\n"
            f"        out['F'] = np.column_stack([{stack_expr}])\n"
            + constr
        )

    @classmethod
    def _build_problem_jax_template(
        cls,
        class_name: str,
        n_var: int,
        n_obj: int,
        has_constraints: bool,
    ) -> str:
        n_ieq = 1 if has_constraints else 0
        objective_lines: list[str] = []
        for idx in range(max(1, int(n_obj))):
            shift = 0.15 + 0.7 * (idx / max(1, int(n_obj) - 1)) if int(n_obj) > 1 else 0.5
            objective_lines.append(
                f"        f{idx+1} = jnp.sum((Xj - {shift:.4f}) ** 2, axis=1) + {0.05*idx:.4f} * jnp.sum(Xj, axis=1)"
            )
        stack_expr = ", ".join(f"f{i+1}" for i in range(max(1, int(n_obj))))
        constr = (
            "        g1 = jnp.sum(Xj, axis=1) - (0.5 * self.n_var)\n"
            "        out['G'] = np.asarray(g1[:, None], dtype=float)\n"
            if has_constraints
            else ""
        )
        return (
            "import numpy as np\n"
            "from core.problem import Problem\n\n"
            "try:\n"
            "    import jax.numpy as jnp\n"
            "except Exception:  # noqa: BLE001\n"
            "    import numpy as jnp  # type: ignore\n\n\n"
            f"class {class_name}(Problem):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__(\n"
            f"            n_var={int(n_var)},\n"
            f"            n_obj={int(n_obj)},\n"
            f"            n_ieq_constr={int(n_ieq)},\n"
            "            xl=0.0,\n"
            "            xu=1.0,\n"
            "        )\n\n"
            "    def _evaluate(self, X, out, *args, **kwargs):\n"
            "        Xj = jnp.asarray(X, dtype=jnp.float32)\n"
            + "\n".join(objective_lines)
            + "\n"
            f"        out['F'] = np.asarray(jnp.column_stack([{stack_expr}]), dtype=float)\n"
            + constr
        )

    @staticmethod
    def _build_metric_template(module_name: str) -> str:
        metric_name = module_name
        return (
            "import numpy as np\n\n\n"
            "def create_metric(context):\n"
            f"    \"\"\"Auto-generated metric '{metric_name}'.\"\"\"\n"
            "    def metric(front):\n"
            "        F = np.asarray(front, dtype=float)\n"
            "        if F.ndim == 1:\n"
            "            F = F.reshape(1, -1)\n"
            "        if F.size == 0:\n"
            "            return float('nan')\n"
            "        # Placeholder metric: average value of the first objective.\n"
            "        return float(np.mean(F[:, 0]))\n"
            "    return metric\n"
        )

    @staticmethod
    def _build_metric_jax_template(module_name: str) -> str:
        metric_name = module_name + "_JAX"
        return (
            "import numpy as np\n\n"
            "try:\n"
            "    import jax.numpy as jnp\n"
            "    _HAS_JAX = True\n"
            "except Exception:  # noqa: BLE001\n"
            "    import numpy as jnp  # type: ignore\n"
            "    _HAS_JAX = False\n\n\n"
            "def create_metric(context):\n"
            f"    \"\"\"Auto-generated JAX metric '{metric_name}'.\"\"\"\n"
            "    def metric(front):\n"
            "        F = np.asarray(front, dtype=float)\n"
            "        if F.ndim == 1:\n"
            "            F = F.reshape(1, -1)\n"
            "        if F.size == 0:\n"
            "            return float('nan')\n"
            "        if not _HAS_JAX:\n"
            "            return float(np.mean(F[:, 0]))\n"
            "        Fj = jnp.asarray(F, dtype=jnp.float32)\n"
            "        return float(jnp.mean(Fj[:, 0]))\n"
            "    return metric\n"
        )

    @staticmethod
    def _augment_metric_code_for_framework(code: str) -> str:
        text = str(code or "")
        if "def create_metric(context)" not in text:
            return text

        helper_name = "_emopylab_unwrap_metric_context"
        if helper_name not in text:
            helper = (
                "\n\n"
                f"def {helper_name}(context):\n"
                "    if not isinstance(context, dict):\n"
                "        return {}\n"
                "    cfg = context.get('config')\n"
                "    if isinstance(cfg, dict):\n"
                "        merged = dict(cfg)\n"
                "        for key, value in context.items():\n"
                "            if key == 'config':\n"
                "                continue\n"
                "            merged.setdefault(key, value)\n"
                "        return merged\n"
                "    return context\n"
            )
            # Insert helper after imports when possible.
            marker = re.search(r"(?m)^(from\\s+.+|import\\s+.+)$", text)
            if marker:
                last_import_end = 0
                for m in re.finditer(r"(?m)^(from\\s+.+|import\\s+.+)$", text):
                    last_import_end = m.end()
                text = text[:last_import_end] + helper + text[last_import_end:]
            else:
                text = helper.lstrip("\n") + "\n\n" + text

        if helper_name in text and f"context = {helper_name}(context)" not in text:
            text = re.sub(
                r"def\s+create_metric\(context\):\s*\n",
                "def create_metric(context):\n    context = _emopylab_unwrap_metric_context(context)\n",
                text,
                count=1,
            )
        return text

    @staticmethod
    def _looks_like_hv_montecarlo_request(prompt: str, base_name: str) -> bool:
        text = f"{str(prompt or '')} {str(base_name or '')}".lower()
        hv_hint = ("hypervolume" in text) or re.search(r"\bhv\b", text) is not None
        mc_hint = ("monte" in text) or ("montecarlo" in text) or ("monte-carlo" in text)
        return bool(hv_hint and mc_hint)

    @staticmethod
    def _looks_like_novel_metric_variant_request(prompt: str) -> bool:
        text = str(prompt or "").lower()
        return any(
            tok in text
            for tok in (
                "novel",
                "new metric",
                "custom metric",
                "variant",
                "inspired",
                "adapted",
                "like",
                "similar to",
                "p-norm",
                "pnorm",
                "p norm",
                "minkowski",
                "lp-norm",
                "l_p",
                "norm order",
                "order p",
                "parameter p",
                "weighted distance",
                "generalized distance",
            )
        )

    @staticmethod
    def _looks_like_parameterized_known_metric_request(prompt: str) -> bool:
        text = str(prompt or "").lower()
        # Parameterization cues for known indicators that should not collapse into canonical proxies.
        return any(
            tok in text
            for tok in (
                "p-norm",
                "pnorm",
                "p norm",
                "minkowski",
                "lp norm",
                "lp-norm",
                "l_p",
                "norm order",
                "order p",
                "parameter p",
                "weighted distance",
                "generalized distance",
                "distance exponent",
            )
        )

    @staticmethod
    def _looks_like_novel_problem_variant_request(prompt: str) -> bool:
        text = str(prompt or "").lower()
        return any(
            tok in text
            for tok in (
                "novel",
                "new problem",
                "custom problem",
                "variant",
                "inspired",
                "adapted",
                "like",
                "similar to",
            )
        )

    @classmethod
    def analyze_request_intent(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str = "",
    ) -> dict[str, Any]:
        """Classify whether the request fits plugin generation or asks for a benchmark survey/research task."""
        artifact = str(artifact_type or "problem").strip().lower()
        prompt_text = str(prompt or "")
        base_text = str(base_name or "")
        text = f"{base_text} {prompt_text}".lower()

        external_source_terms = (
            "gecco",
            "cec",
            "congress",
            "conference",
            "journal",
            "proceedings",
            "paper",
            "papers",
            "article",
            "literature",
            "survey",
            "review",
            "doi",
        )
        listing_terms = (
            "provide all",
            "list all",
            "all benchmark",
            "which benchmarks",
            "used in",
            "used by",
            "adopted in",
            "including the correct dimensions",
            "including dimensions",
        )
        benchmark_terms = (
            "benchmark",
            "benchmarks",
            "benchmark function",
            "benchmark functions",
            "test problem",
            "test suite",
            "suite",
            "multi-objective algorithms",
        )
        dimension_terms = (
            "dimension",
            "dimensions",
            "decision variables",
            "objectives",
            "n_var",
            "n_obj",
        )
        codegen_terms = (
            "generate",
            "implement",
            "write code",
            "python module",
            "plugin",
            "pymoo problem",
            "problem subclass",
            "create_metric",
            "cpu and jax",
            "jax variant",
        )

        external_hits = [tok for tok in external_source_terms if tok in text]
        listing_hits = [tok for tok in listing_terms if tok in text]
        benchmark_hits = [tok for tok in benchmark_terms if tok in text]
        dimension_hits = [tok for tok in dimension_terms if tok in text]
        codegen_hits = [tok for tok in codegen_terms if tok in text]
        known_family_tokens = sorted({m.group(1).upper() for m in re.finditer(r"\b(ZDT[1-6]|DTLZ[1-7]|WFG[1-9])\b", text, flags=re.IGNORECASE)})

        strong_survey_markers = bool(external_hits and (listing_hits or ("used in" in text)))
        benchmark_catalog_intent = bool(benchmark_hits and (listing_hits or dimension_hits))
        needs_external_sources = bool(external_hits or ("gecco" in text) or ("proceedings" in text))
        benchmark_survey = bool(
            artifact == "problem"
            and (strong_survey_markers or benchmark_catalog_intent)
            and (
                needs_external_sources
                or "used in" in text
                or "congress" in text
                or "conference" in text
            )
        )

        task_kind = "benchmark_survey" if benchmark_survey else "plugin_generation"
        fit_for_llm_agent = not benchmark_survey
        unsupported_for_generation = benchmark_survey

        reasons: list[str] = []
        if benchmark_survey:
            reasons.append("Prompt asks for benchmark list/survey instead of a single plugin implementation.")
            if needs_external_sources:
                reasons.append("Prompt references conference/papers/proceedings, requiring external sources.")
            if dimension_hits:
                reasons.append("Prompt requests dimensions across benchmarks/papers, which is a structured extraction task.")
        else:
            reasons.append("Prompt is compatible with single plugin generation workflow.")
            if codegen_hits:
                reasons.append("Code-generation cues detected in request.")

        return {
            "task_kind": task_kind,
            "artifact_type": artifact,
            "fit_for_llm_agent": bool(fit_for_llm_agent),
            "unsupported_for_generation": bool(unsupported_for_generation),
            "needs_external_sources": bool(needs_external_sources),
            "detected_known_families": known_family_tokens,
            "signals": {
                "external_source_terms": external_hits,
                "listing_terms": listing_hits,
                "benchmark_terms": benchmark_hits,
                "dimension_terms": dimension_hits,
                "codegen_terms": codegen_hits,
            },
            "reasons": reasons,
        }

    @classmethod
    def _apply_request_intent_flags_to_spec_report(
        cls,
        spec_report: dict[str, Any] | None,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
    ) -> dict[str, Any] | None:
        if not isinstance(spec_report, dict):
            return spec_report
        intent = cls.analyze_request_intent(prompt=prompt, artifact_type=artifact_type, base_name=base_name)
        payload = dict(spec_report)
        payload["task_kind"] = str(intent.get("task_kind", "plugin_generation"))
        payload["needs_external_sources"] = bool(intent.get("needs_external_sources", False))
        payload["fit_for_llm_agent"] = bool(intent.get("fit_for_llm_agent", True))
        payload["unsupported_for_generation"] = bool(intent.get("unsupported_for_generation", False))
        payload["intent_reasons"] = [str(x) for x in (intent.get("reasons", []) or [])]
        if intent.get("detected_known_families"):
            payload["detected_known_families"] = list(intent.get("detected_known_families") or [])
        return payload

    @classmethod
    def _infer_known_metric_kind_from_request(cls, prompt: str, base_name: str) -> str:
        probe = {"base_name": str(base_name or ""), "_prompt": str(prompt or ""), "cpu_code": ""}
        kind = cls._metric_kind_hint(probe)
        return kind if kind in {"hv", "gd", "igd", "igdp", "deltap"} else ""

    @staticmethod
    def _infer_known_problem_kind_from_request(prompt: str, base_name: str) -> str:
        text = f"{str(base_name or '')} {str(prompt or '')}"
        m = re.search(r"\b(ZDT([1-6])|DTLZ([1-7]))\b", text, flags=re.IGNORECASE)
        if not m:
            return ""
        token = str(m.group(1) or "").strip().lower()
        return token if token else ""

    @staticmethod
    def _build_canonical_metric_proxy_code(*, metric_kind: str, jax: bool) -> str:
        metric_kind = str(metric_kind).strip().lower()
        metric_map_cpu = {
            "hv": "_metric_HV",
            "gd": "_metric_GD",
            "igd": "_metric_IGD",
            "igdp": "_metric_IGDp",
            "deltap": "_metric_DeltaP",
        }
        metric_map_jax = {
            "hv": "_metric_HV_JAX",
            "gd": "_metric_GD_JAX",
            "igd": "_metric_IGD_JAX",
            "igdp": "_metric_IGDp_JAX",
            "deltap": "_metric_DeltaP_JAX",
        }
        fn_name = metric_map_jax.get(metric_kind) if jax else metric_map_cpu.get(metric_kind)
        if not fn_name:
            raise ValueError(f"Unsupported canonical metric kind: {metric_kind}")
        module_name = "metrics.community_metrics_JAX" if jax else "metrics.community_metrics"
        return (
            "import numpy as np\n"
            f"from {module_name} import {fn_name}\n\n"
            "def create_metric(context):\n"
            "    local_context = dict(context or {}) if isinstance(context, dict) else {}\n"
            "    def metric(front):\n"
            "        F = np.asarray(front, dtype=float)\n"
            "        if F.ndim == 1:\n"
            "            F = F.reshape(1, -1)\n"
            f"        return float({fn_name}(F, local_context))\n"
            "    return metric\n"
        )

    @staticmethod
    def _build_canonical_hv_montecarlo_metric_code(*, jax: bool) -> str:
        jax_import = (
            "try:\n"
            "    import jax.numpy as jnp  # noqa: F401\n"
            "except Exception:  # noqa: BLE001\n"
            "    jnp = None  # type: ignore[assignment]\n\n"
            if jax
            else ""
        )
        return (
            "import numpy as np\n"
            "from metrics.community_metrics import _get_front, _get_reference_front, _as_2d, _safe_divisor, _non_dominated_front\n\n"
            + jax_import
            + "def create_metric(context):\n"
            "    \"\"\"Monte Carlo Hypervolume (project-compatible, MaOP-focused, approximate for all m>=2).\"\"\"\n"
            "    local_context = dict(context or {}) if isinstance(context, dict) else {}\n"
            "    cfg = local_context.get('config') if isinstance(local_context, dict) else None\n"
            "    cfg_map = dict(cfg) if isinstance(cfg, dict) else {}\n"
            "    hv_samples = cfg_map.get('hv_samples', local_context.get('hv_samples', None))\n"
            "    hv_mc_samples = cfg_map.get('hv_mc_samples', local_context.get('hv_mc_samples', None))\n"
            "    if hv_mc_samples is None and hv_samples is not None:\n"
            "        try:\n"
            "            hv_mc_samples = int(hv_samples)\n"
            "        except Exception:  # noqa: BLE001\n"
            "            hv_mc_samples = None\n"
            "    if hv_mc_samples is None:\n"
            "        hv_mc_samples = 100000\n"
            "    if isinstance(cfg, dict):\n"
            "        cfg_map['hv_mc_samples'] = int(max(1, hv_mc_samples))\n"
            "        local_context['config'] = cfg_map\n"
            "    local_context['hv_mc_samples'] = int(max(1, hv_mc_samples))\n\n"
            "    def _hv_mc_only(pop_obj, optimum, ctx):\n"
            "        pop_obj = _as_2d(pop_obj)\n"
            "        optimum = _as_2d(optimum)\n"
            "        if pop_obj.size == 0:\n"
            "            return 0.0\n"
            "        if pop_obj.shape[1] != optimum.shape[1]:\n"
            "            return float('nan')\n"
            "        _, m = pop_obj.shape\n"
            "        if m < 2:\n"
            "            return float('nan')\n"
            "        fmin = np.minimum(np.min(pop_obj, axis=0), np.zeros(m, dtype=float))\n"
            "        fmax = np.max(optimum, axis=0)\n"
            "        den = _safe_divisor((fmax - fmin) * 1.1)\n"
            "        norm_pop = (pop_obj - fmin) / den\n"
            "        norm_pop = norm_pop[~np.any(norm_pop > 1.0, axis=1)]\n"
            "        if norm_pop.size == 0:\n"
            "            return 0.0\n"
            "        norm_pop = _non_dominated_front(norm_pop)\n"
            "        ref_point = np.ones(m, dtype=float)\n"
            "        min_value = np.min(norm_pop, axis=0)\n"
            "        max_value = ref_point\n"
            "        if np.any(max_value < min_value):\n"
            "            return 0.0\n"
            "        seed = int(ctx.get('seed', 1) or 1)\n"
            "        rng = np.random.default_rng(seed)\n"
            "        sample_num = int(max(1, ctx.get('hv_mc_samples', 100000)))\n"
            "        samples = rng.uniform(low=min_value, high=max_value, size=(sample_num, m))\n"
            "        dominated = np.zeros(sample_num, dtype=bool)\n"
            "        chunk = 2048\n"
            "        for i in range(0, sample_num, chunk):\n"
            "            s = samples[i:i+chunk]\n"
            "            dominated[i:i+len(s)] = np.any(np.all(norm_pop[None, :, :] <= s[:, None, :], axis=2), axis=1)\n"
            "        return float(np.prod(max_value - min_value) * np.mean(dominated))\n\n"
            "    def metric(front):\n"
            "        pop_obj = _get_front(front)\n"
            "        optimum = _get_reference_front(local_context)\n"
            "        if optimum is None:\n"
            "            return float('nan')\n"
            "        return float(_hv_mc_only(pop_obj, optimum, local_context))\n\n"
            "    return metric\n"
        )

    @classmethod
    def _build_canonical_problem_proxy_code(
        cls,
        *,
        problem_kind: str,
        jax: bool,
        base_name: str,
        n_var: int,
        n_obj: int,
    ) -> str:
        kind = str(problem_kind or "").strip().lower()
        if not kind:
            raise ValueError("problem_kind is required")
        class_name = cls._problem_symbol_upper(base_name or "GeneratedProblem", default="GENERATEDPROBLEM")
        if jax and not class_name.endswith("_JAX"):
            class_name = f"{class_name}_JAX"

        if kind.startswith("zdt"):
            canonical = kind.upper()
            module_name = "problems.multi.zdt"
            import_name = f"{canonical}_JAX" if jax else canonical
            return (
                "import numpy as np\n"
                f"from {module_name} import {import_name} as _CanonicalProblem\n\n"
                f"class {class_name}(_CanonicalProblem):\n"
                f"    \"\"\"Canonical {canonical} proxy generated by the EmoPyLab Llm Agent.\"\"\"\n"
                f"    def __init__(self, n_var={int(max(1, n_var))}, **kwargs):\n"
                f"        super().__init__(n_var=n_var, **kwargs)\n\n"
                "    def _evaluate(self, X, out, *args, **kwargs):\n"
                "        # Delegates vectorized out[\"F\"] computation to the canonical local implementation.\n"
                "        X = np.asarray(X)\n"
                "        return super()._evaluate(X, out, *args, **kwargs)\n\n"
                "    def _calc_pareto_front(self, *args, **kwargs):\n"
                "        return super()._calc_pareto_front(*args, **kwargs)\n"
            )

        if kind.startswith("dtlz"):
            canonical = kind.upper()
            module_name = "problems.many.dtlz"
            import_name = f"{canonical}_JAX" if jax else canonical
            return (
                "import numpy as np\n"
                f"from {module_name} import {import_name} as _CanonicalProblem\n\n"
                f"class {class_name}(_CanonicalProblem):\n"
                f"    \"\"\"Canonical {canonical} proxy generated by the EmoPyLab Llm Agent.\"\"\"\n"
                f"    def __init__(self, n_var={int(max(1, n_var))}, n_obj={int(max(1, n_obj))}, **kwargs):\n"
                f"        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)\n\n"
                "    def _evaluate(self, X, out, *args, **kwargs):\n"
                "        # Delegates vectorized out[\"F\"] computation to the canonical local implementation.\n"
                "        X = np.asarray(X)\n"
                "        return super()._evaluate(X, out, *args, **kwargs)\n\n"
                "    def _calc_pareto_front(self, *args, **kwargs):\n"
                "        return super()._calc_pareto_front(*args, **kwargs)\n"
            )

        raise ValueError(f"Unsupported canonical problem kind: {problem_kind}")

    @classmethod
    def _apply_known_metric_request_overrides(cls, bundle: dict[str, Any], *, prompt: str) -> None:
        if str(bundle.get("artifact_type", "")).strip().lower() != "metric":
            return
        intent = cls.analyze_request_intent(
            prompt=str(prompt or ""),
            artifact_type="metric",
            base_name=str(bundle.get("base_name", "")),
        )
        if bool(intent.get("unsupported_for_generation", False)):
            return
        base_name = str(bundle.get("base_name", ""))
        if cls._looks_like_hv_montecarlo_request(prompt, base_name):
            bundle["cpu_code"] = cls._build_canonical_hv_montecarlo_metric_code(jax=False)
            bundle["jax_code"] = cls._build_canonical_hv_montecarlo_metric_code(jax=True)
            return

        if cls._looks_like_novel_metric_variant_request(prompt):
            return

        kind = cls._infer_known_metric_kind_from_request(prompt, base_name)
        if kind and cls._looks_like_parameterized_known_metric_request(prompt):
            return
        if kind:
            bundle["cpu_code"] = cls._build_canonical_metric_proxy_code(metric_kind=kind, jax=False)
            bundle["jax_code"] = cls._build_canonical_metric_proxy_code(metric_kind=kind, jax=True)

    @classmethod
    def _apply_known_problem_request_overrides(cls, bundle: dict[str, Any], *, prompt: str) -> None:
        if str(bundle.get("artifact_type", "")).strip().lower() != "problem":
            return
        intent = cls.analyze_request_intent(
            prompt=str(prompt or ""),
            artifact_type="problem",
            base_name=str(bundle.get("base_name", "")),
        )
        if bool(intent.get("unsupported_for_generation", False)):
            return
        if cls._looks_like_novel_problem_variant_request(prompt):
            return
        base_name = str(bundle.get("base_name", ""))
        kind = cls._infer_known_problem_kind_from_request(prompt, base_name)
        if not kind:
            return
        n_var = int(max(1, int(bundle.get("n_var", 30) or 30)))
        n_obj = int(max(1, int(bundle.get("n_obj", 2) or 2)))
        bundle["cpu_code"] = cls._build_canonical_problem_proxy_code(
            problem_kind=kind,
            jax=False,
            base_name=base_name,
            n_var=n_var,
            n_obj=n_obj,
        )
        bundle["jax_code"] = cls._build_canonical_problem_proxy_code(
            problem_kind=kind,
            jax=True,
            base_name=base_name,
            n_var=n_var,
            n_obj=n_obj,
        )

    @classmethod
    def _build_local_spec_report(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
    ) -> dict[str, Any]:
        prompt_text = str(prompt or "").strip()
        text_l = prompt_text.lower()
        intent = cls.analyze_request_intent(prompt=prompt_text, artifact_type=artifact_type, base_name=base_name)
        limitations = [
            "Local template path uses heuristic defaults and cannot infer domain-specific formulas from natural language."
        ]
        if bool(intent.get("unsupported_for_generation", False)):
            limitations.append(
                "Prompt requests a benchmark survey/listing (conference/paper coverage + dimensions), which requires web/literature extraction instead of plugin code generation."
            )
        payload = {
            "mode": "local_template_spec",
            "artifact_type": str(artifact_type),
            "base_name": str(base_name),
            "summary": f"Local spec summary for {artifact_type} artifact '{base_name}' with n_var={n_var}, n_obj={n_obj}.",
            "assumptions": [
                "Use vectorized NumPy/JAX-compatible implementation.",
                "Preserve EmoPyLab interface contracts for problem/metric plugins.",
            ],
            "invariants": [
                "CPU and JAX variants should be numerically consistent under the same inputs.",
                "Generated code must pass AST, compile, and runtime smoke validation.",
            ],
            "limitations": limitations,
            "suggested_n_var_default": None,
            "suggested_n_obj_default": None,
            "dimension_defaults_source_note": "Local template path cannot infer benchmark defaults from web sources.",
            "novel_variant_hint": cls._looks_like_novel_metric_variant_request(prompt_text) if str(artifact_type).lower() == "metric" else False,
            "known_metric_hint": cls._infer_known_metric_kind_from_request(prompt_text, str(base_name)) if str(artifact_type).lower() == "metric" else "",
            "keywords": [tok for tok in ("constraint", "hypervolume", "igd", "gd", "delta", "dtlz", "zdt", "wfg") if tok in text_l],
        }
        if str(artifact_type).lower() == "problem":
            try:
                payload["assumptions"] = list(payload.get("assumptions", [])) + [
                    "Use core.problem.Problem with vectorized _evaluate(self, X, out, *args, **kwargs) and out['F'].",
                    "Use constraint API n_ieq_constr/n_eq_constr with out['G']/out['H'] when constraints are present.",
                ]
                payload["invariants"] = list(payload.get("invariants", [])) + [
                    "Return plugin modules only (no __main__ demo/test harness).",
                ]
            except Exception:
                pass
        return cast(dict[str, Any], cls._apply_request_intent_flags_to_spec_report(payload, prompt=prompt_text, artifact_type=artifact_type, base_name=base_name))

    @classmethod
    def _generate_local_spec_report(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        timeout_s: float = 30.0,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        n_var_default, n_obj_default = cls._normalize_problem_ui_defaults(n_var, n_obj)
        system_prompt = (
            "Act as a senior software engineer specialized in PlatEMO and optimization frameworks. "
            "Produce a compact specification JSON for code generation. Return strict JSON only."
        )
        user_prompt = (
            f"Create a specification for generating a {artifact_type} plugin for EmoPyLab.\n"
            f"Base name: {base_name}\n"
            f"EmoPyLab UI defaults/presets: n_var_default={n_var_default}, n_obj_default={n_obj_default}\n"
            f"Request:\n{str(prompt or '').strip()}\n\n"
            "Return JSON with keys: summary (string), assumptions (list[string]), invariants (list[string]), "
            "limitations (list[string]), known_family_hint (string), metric_kind_hint (string), ambiguity_notes (list[string]), "
            "suggested_n_var_default (integer|null), suggested_n_obj_default (integer|null), dimension_defaults_source_note (string), "
            "task_kind (string), needs_external_sources (bool), fit_for_llm_agent (bool), unsupported_for_generation (bool)."
        )
        cls._emit_stream_event(
            stream_event_cb,
            {
                "kind": "llm_stream",
                "phase": "spec_first",
                "event": "stage_start",
                "message": "Spec-first: querying local Qwen for refined generation spec.",
            },
        )
        payload = None
        try:
            client = LocalLLMClient(timeout=float(timeout_s))
            payload = client.json_call(prompt=user_prompt, system=system_prompt)
        except Exception:
            payload = None
        cls._emit_stream_event(
            stream_event_cb,
            {
                "kind": "llm_stream",
                "phase": "spec_first",
                "event": "stage_end",
                "message": "Spec-first completed.",
            },
        )
        if not isinstance(payload, dict):
            return cls._build_local_spec_report(
                prompt=prompt,
                artifact_type=artifact_type,
                base_name=base_name,
                n_var=n_var_default,
                n_obj=n_obj_default,
            )
        payload = dict(payload)
        payload["mode"] = "local_qwen_spec_first"
        payload["model"] = cls.DEFAULT_LOCAL_MODEL
        return cast(
            dict[str, Any],
            cls._apply_request_intent_flags_to_spec_report(
                payload,
                prompt=prompt,
                artifact_type=artifact_type,
                base_name=base_name,
            ),
        )

    @classmethod
    def generate_artifact_bundle(
        cls,
        prompt: str,
        *,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        provider: str = DEFAULT_PROVIDER,
        api_key: str | None = None,
        timeout_s: float = 90.0,
        spec_first: bool = False,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        _ = api_key
        artifact = str(artifact_type or "problem").strip().lower()
        if artifact not in {"problem", "metric"}:
            raise ValueError("artifact_type must be 'problem' or 'metric'.")

        provider_token = str(provider or cls.DEFAULT_PROVIDER).strip().lower()
        prompt_text = str(prompt or "")
        metric_mode = cls._normalize_metric_generation_mode(metric_generation_mode)
        n_var_default, n_obj_default = cls._normalize_problem_ui_defaults(n_var, n_obj)
        request_intent = cls.analyze_request_intent(prompt=prompt_text, artifact_type=artifact, base_name=base_name)
        spec_report: dict[str, Any] | None = None
        if bool(spec_first):
            try:
                if provider_token == cls.LOCAL_PROVIDER:
                    spec_report = cls._generate_local_spec_report(
                        prompt=prompt_text,
                        artifact_type=artifact,
                        base_name=base_name,
                        n_var=n_var_default,
                        n_obj=n_obj_default,
                        timeout_s=timeout_s,
                        metric_generation_mode=metric_mode,
                        stream_event_cb=stream_event_cb,
                    )
                else:
                    spec_report = cls._build_local_spec_report(
                        prompt=prompt_text,
                        artifact_type=artifact,
                        base_name=base_name,
                        n_var=n_var_default,
                        n_obj=n_obj_default,
                    )
            except Exception as exc:  # noqa: BLE001
                spec_report = {
                    "mode": "spec_first_failed",
                    "error": str(exc),
                }
            spec_report = cls._apply_request_intent_flags_to_spec_report(
                spec_report,
                prompt=prompt_text,
                artifact_type=artifact,
                base_name=base_name,
            )
        n_var_default, n_obj_default = cls._apply_spec_suggested_problem_defaults(
            prompt=prompt_text,
            artifact_type=artifact,
            spec_report=spec_report,
            n_var_default=n_var_default,
            n_obj_default=n_obj_default,
        )
        generation_prompt = prompt_text
        if isinstance(spec_report, dict) and spec_report and "summary" in spec_report:
            generation_prompt = (
                prompt_text
                + "\n\nGeneration spec (validated pre-step, use this as a constraint, not as prose output):\n"
                + json.dumps(spec_report, ensure_ascii=False)
            )
        if provider_token == cls.LOCAL_PROVIDER:
            bundle = cls._generate_artifact_bundle_local(
                prompt=generation_prompt,
                artifact_type=artifact,
                base_name=base_name,
                n_var=n_var_default,
                n_obj=n_obj_default,
                timeout_s=timeout_s,
                metric_generation_mode=metric_mode,
                stream_event_cb=stream_event_cb,
            )
        else:
            bundle = cls._generate_artifact_bundle_template(
                prompt=generation_prompt,
                artifact_type=artifact,
                base_name=base_name,
                n_var=n_var_default,
                n_obj=n_obj_default,
            )

        bundle.setdefault("artifact_type", artifact)
        bundle.setdefault("provider", provider_token)
        bundle.setdefault("n_var", int(n_var_default))
        bundle.setdefault("n_obj", int(n_obj_default))
        bundle["_prompt"] = prompt_text
        bundle["_request_intent"] = request_intent
        bundle["_spec_first"] = bool(spec_first)
        bundle["_metric_generation_mode"] = metric_mode
        if spec_report is not None:
            bundle["_spec_report"] = spec_report
        if artifact == "metric" and metric_mode == cls.METRIC_MODE_CANONICAL_WRAPPER:
            cls._apply_known_metric_request_overrides(bundle, prompt=prompt_text)
        cls._normalize_bundle_metadata(bundle)
        cls.validate_artifact_bundle_detailed(bundle)
        return bundle

    @classmethod
    def _normalize_bundle_metadata(cls, bundle: dict[str, Any]) -> None:
        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        n_var = int(max(1, int(bundle.get("n_var", 30))))
        n_obj = int(max(1, int(bundle.get("n_obj", 2))))
        raw_base = str(bundle.get("base_name", "")).strip()

        if artifact == "problem":
            base_class = cls._problem_symbol_upper(raw_base or "GeneratedProblem", default="GENERATEDPROBLEM")
            cpu_class = cls._problem_symbol_upper(str(bundle.get("cpu_symbol", base_class)), default=base_class)
            jax_class = cls._problem_symbol_upper(str(bundle.get("jax_symbol", f"{base_class}_JAX")), default=f"{base_class}_JAX")
            if not jax_class.endswith("_JAX"):
                jax_class = f"{jax_class}_JAX"
            base_module = cls._slugify_module_name(raw_base or base_class, default="generated_problem")
            bundle["base_name"] = base_class
            bundle["cpu_symbol"] = cpu_class
            bundle["jax_symbol"] = jax_class
            bundle["cpu_file"] = f"{base_module}.py"
            bundle["jax_file"] = f"{base_module}_JAX.py"
        else:
            base_module = cls._slugify_module_name(raw_base, default="generated_metric").upper()
            bundle["base_name"] = base_module
            bundle["cpu_symbol"] = "create_metric"
            bundle["jax_symbol"] = "create_metric"
            bundle["cpu_file"] = f"{base_module}.py"
            bundle["jax_file"] = f"{base_module}_JAX.py"

        bundle["n_var"] = n_var
        bundle["n_obj"] = n_obj
        cpu_code = str(bundle.get("cpu_code", "")).replace("\r\n", "\n").strip()
        jax_code = str(bundle.get("jax_code", "")).replace("\r\n", "\n").strip()
        if artifact == "problem":
            cpu_code = cls._augment_problem_code_for_metrics(
                cpu_code,
                base_name=str(bundle.get("base_name", "")),
                class_name=str(bundle.get("cpu_symbol", "")),
                n_obj=n_obj,
            )
            jax_code = cls._augment_problem_code_for_metrics(
                jax_code,
                base_name=str(bundle.get("base_name", "")),
                class_name=str(bundle.get("jax_symbol", "")),
                n_obj=n_obj,
            )
        else:
            cpu_code = cls._augment_metric_code_for_framework(cpu_code)
            jax_code = cls._augment_metric_code_for_framework(jax_code)
        cpu_code = cls._apply_prompt_signature(cpu_code)
        jax_code = cls._apply_prompt_signature(jax_code)
        bundle["cpu_code"] = cpu_code + "\n"
        bundle["jax_code"] = jax_code + "\n"

    @classmethod
    def _generate_artifact_bundle_template(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
    ) -> dict[str, Any]:
        text = str(prompt or "").lower()
        if artifact_type == "problem":
            has_constraints = any(tok in text for tok in ("constraint", "restri", "g(", "<=", ">="))
            base_class = cls._problem_symbol_upper(base_name or "GeneratedProblem", default="GENERATEDPROBLEM")
            return {
                "artifact_type": "problem",
                "base_name": base_class,
                "cpu_symbol": base_class,
                "jax_symbol": f"{base_class}_JAX",
                "cpu_code": cls._build_problem_template(base_class, max(1, int(n_var)), max(1, int(n_obj)), has_constraints),
                "jax_code": cls._build_problem_jax_template(f"{base_class}_JAX", max(1, int(n_var)), max(1, int(n_obj)), has_constraints),
            }

        base_module = cls._slugify_module_name(base_name, default="generated_metric").upper()
        return {
            "artifact_type": "metric",
            "base_name": base_module,
            "cpu_symbol": "create_metric",
            "jax_symbol": "create_metric",
            "cpu_code": cls._build_metric_template(base_module),
            "jax_code": cls._build_metric_jax_template(base_module),
        }

    @classmethod
    def _generate_artifact_bundle_local(
        cls,
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        timeout_s: float = 90.0,
        metric_generation_mode: str = METRIC_MODE_CANONICAL_WRAPPER,
        stream_event_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        n_var_default, n_obj_default = cls._normalize_problem_ui_defaults(n_var, n_obj)
        metric_mode = cls._normalize_metric_generation_mode(metric_generation_mode)
        raw_base = str(base_name or "").strip()
        if artifact_type == "problem":
            safe_base = cls._problem_symbol_upper(raw_base or "GeneratedProblem", default="GENERATEDPROBLEM")
            cpu_symbol = safe_base
            jax_symbol = f"{safe_base}_JAX"
            system_prompt = (
                "Act as a senior software engineer specialized in PlatEMO and optimization frameworks. "
                "You write safe, vectorized Python code for EmoPyLab Problem subclasses. "
                "Use EmoPyLab's current Problem/constraint APIs (`out['F']`, `n_ieq_constr`/`n_eq_constr`, `out['G']`/`out['H']`). "
                "Do not emit thin wrappers around local canonical benchmark classes. "
                "Do not include `if __name__ == '__main__':` or demo code. "
                "Return a strict JSON object with cpu_code and jax_code as strings. "
                "No prose, no markdown, no file I/O, no subprocess, no network code."
            )
            user_prompt = (
                f"Generate two Python modules for EmoPyLab problems based on this requirement:\n{str(prompt or '').strip()}\n\n"
                f"CPU class name: {cpu_symbol}. JAX class name: {jax_symbol}.\n"
                f"UI defaults: n_var={n_var_default}, n_obj={n_obj_default}.\n"
                "Both modules must import Problem from core.problem and subclass it.\n"
                "Both modules must use vectorized _evaluate(self, X, out, *args, **kwargs) setting out['F'].\n"
                "Return JSON only with keys cpu_code and jax_code."
            )
        else:
            safe_base = cls._slugify_module_name(raw_base, default="generated_metric")
            system_prompt = (
                "Act as a senior software engineer specialized in PlatEMO and optimization frameworks. "
                "You write safe Python code for EmoPyLab metric modules. "
                "Each module must expose create_metric(context) returning a callable metric(front)->float. "
                "Return a strict JSON object with cpu_code and jax_code as strings. "
                "No prose, no markdown, no file I/O, no subprocess, no network code."
            )
            user_prompt = (
                f"Generate two Python metric modules for EmoPyLab based on this requirement:\n{str(prompt or '').strip()}\n\n"
                f"Metric base name: {safe_base}\n"
                "Return JSON only with keys cpu_code and jax_code."
            )

        cls._emit_stream_event(
            stream_event_cb,
            {
                "kind": "llm_stream",
                "phase": "generation",
                "event": "stage_start",
                "message": f"Generating {artifact_type} code via local Qwen ({cls.DEFAULT_LOCAL_MODEL}).",
            },
        )
        payload = None
        try:
            client = LocalLLMClient(timeout=float(timeout_s))
            payload = client.json_call(prompt=user_prompt, system=system_prompt)
        except Exception:
            payload = None

        cls._emit_stream_event(
            stream_event_cb,
            {
                "kind": "llm_stream",
                "phase": "generation",
                "event": "stage_end",
                "message": "Local code generation finished.",
            },
        )

        if isinstance(payload, dict) and payload.get("cpu_code") and payload.get("jax_code"):
            cpu_code = str(payload.get("cpu_code", "")).strip()
            jax_code = str(payload.get("jax_code", "")).strip()
            if artifact_type == "problem":
                cpu_code = cls._augment_problem_code_for_metrics(
                    cpu_code,
                    base_name=safe_base,
                    class_name=cpu_symbol,
                    n_obj=n_obj_default,
                )
                jax_code = cls._augment_problem_code_for_metrics(
                    jax_code,
                    base_name=safe_base,
                    class_name=jax_symbol,
                    n_obj=n_obj_default,
                )
            return {
                "artifact_type": artifact_type,
                "base_name": safe_base,
                "cpu_code": cpu_code,
                "jax_code": jax_code,
                "provider": cls.LOCAL_PROVIDER,
                "model": cls.DEFAULT_LOCAL_MODEL,
                "_api_raw_text": json.dumps(payload, ensure_ascii=False),
                "_api_call_debug": {"provider": cls.LOCAL_PROVIDER, "model": cls.DEFAULT_LOCAL_MODEL},
            }

        # Fallback to deterministic template bundle on local LLM absence or invalid return
        template_bundle = cls._generate_artifact_bundle_template(
            prompt=prompt,
            artifact_type=artifact_type,
            base_name=base_name,
            n_var=n_var_default,
            n_obj=n_obj_default,
        )
        template_bundle["provider"] = cls.LOCAL_PROVIDER
        template_bundle["model"] = cls.DEFAULT_LOCAL_MODEL
        return template_bundle

    @classmethod
    def _validate_common_python_code(cls, code: str, *, require_class: bool, require_create_metric: bool) -> tuple[bool, list[str]]:
        issues: list[str] = []
        try:
            tree = ast.parse(code)
        except Exception as exc:  # noqa: BLE001
            return False, [f"Syntax error: {exc}"]

        for node in ast.walk(tree):
            if isinstance(node, cls._BLOCKED_AST_NODES):
                issues.append(f"Unsupported AST node: {type(node).__name__}")
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = str(alias.name or "").split(".")[0]
                    if root not in cls._ALLOWED_IMPORT_ROOTS:
                        issues.append(f"Unsupported import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = str(getattr(node, "module", "") or "")
                root = module.split(".")[0] if module else ""
                if root not in cls._ALLOWED_IMPORT_ROOTS:
                    issues.append(f"Unsupported import-from: {module or '<empty module>'}")
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    if fn.id in cls._BLOCKED_CALL_NAMES:
                        issues.append(f"Unsupported call: {fn.id}(...)")
                elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                    if (fn.value.id, fn.attr) in cls._BLOCKED_ATTR_CALLS:
                        issues.append(f"Unsupported call: {fn.value.id}.{fn.attr}(...)")

        class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        fn_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if require_class and not class_defs:
            issues.append("No class definition found.")
        if require_class:
            has_evaluate = False
            for cls_node in class_defs:
                for item in cls_node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_evaluate":
                        has_evaluate = True
                        break
            if not has_evaluate:
                issues.append("Class must implement _evaluate(self, X, out, *args, **kwargs).")

        if require_create_metric and not any(fn.name == "create_metric" for fn in fn_defs):
            issues.append("Module must expose create_metric(context).")

        try:
            compile(code, "<llm_artifact>", "exec")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Compile error: {exc}")

        return len(issues) == 0, issues

    @classmethod
    def _safe_generated_import(
        cls,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("Relative imports are not allowed in generated artifacts.")
        root = str(name or "").split(".", 1)[0]
        if root not in cls._ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"Unsupported import: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    @classmethod
    def _restricted_exec_namespace(cls) -> dict[str, Any]:
        safe_builtins = {
            name: getattr(builtins, name)
            for name in cls._SAFE_BUILTIN_NAMES
            if hasattr(builtins, name)
        }
        safe_builtins["__build_class__"] = builtins.__build_class__
        safe_builtins["__import__"] = cls._safe_generated_import
        return {
            "__builtins__": safe_builtins,
            "__name__": "__emopylab_llm_artifact__",
        }

    @classmethod
    def validate_problem_code(cls, code: str) -> tuple[bool, list[str]]:
        ok, issues = cls._validate_common_python_code(code, require_class=True, require_create_metric=False)
        vectorization_hints = ("axis=1", "column_stack", "np.asarray", "out['F']", 'out["F"]')
        if not any(tok in code for tok in vectorization_hints):
            issues.append("Vectorization hints not found (expected operations over population matrices).")
        if cls._looks_like_placeholder_benchmark_fallback_text(code):
            issues.append(
                "Placeholder benchmark fallback text detected (e.g., 'placeholder semantics' / 'could not be reliably recovered'). "
                "LLM Agent must recover and implement the requested benchmark semantics via web_search sources, not ship placeholder semantics."
            )
        if cls._looks_like_native_pymoo_problem_wrapper(code):
            issues.append(
                "Native/local benchmark wrapper pattern detected. LLM Agent problem generation must implement/adapt the requested problem semantics directly (no thin wrapper around pymoo or local canonical problem classes)."
            )
        return len(issues) == 0, issues

    @classmethod
    def validate_metric_code(cls, code: str) -> tuple[bool, list[str]]:
        ok, issues = cls._validate_common_python_code(code, require_class=False, require_create_metric=True)
        if "create_metric" in code and "front" not in code.lower():
            issues.append("Metric code should reference front values (expected 'front' usage).")
        tl = str(code or "").lower()
        if "placeholder metric" in tl or ("placeholder semantics" in tl and "metric" in tl):
            issues.append(
                "Placeholder metric fallback text detected. LLM Agent metric generation must produce complete metric semantics via GitHub-sourced implementations."
            )
        return len(issues) == 0, issues

    @staticmethod
    def _metric_kind_hint(bundle: dict[str, Any]) -> str:
        text_raw = " ".join(
            [
                str(bundle.get("base_name", "")),
                str(bundle.get("_prompt", "")),
                str(bundle.get("cpu_code", ""))[:4000],
            ]
        ).lower()
        # Normalize common separators so names like "HV_LLM_MONTECARLO" are recognized.
        text = re.sub(r"[_/\\\\-]+", " ", text_raw)
        if ("hypervolume" in text) or re.search(r"\bhv\b", text):
            mc_hint = (
                ("monte" in text)
                or ("montecarlo" in text)
                or ("monte carlo" in text)
                or re.search(r"\bhv\s*mc\b", text) is not None
                or re.search(r"\bmc\s*hv\b", text) is not None
            )
            return "hv_mc" if mc_hint else "hv"
        if "deltap" in text or "delta p" in text:
            return "deltap"
        if re.search(r"\bigd\+\b", text) or "igdp" in text:
            return "igdp"
        if re.search(r"\bigd\b", text):
            return "igd"
        if re.search(r"\bgd\b", text):
            return "gd"
        return ""

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            return float("nan")
        return out

    @classmethod
    def _exec_metric_factory(cls, code: str) -> tuple[Any, list[str]]:
        ok, issues = cls._validate_common_python_code(
            str(code),
            require_class=False,
            require_create_metric=True,
        )
        if not ok:
            return None, issues
        ns = cls._restricted_exec_namespace()
        try:
            exec(compile(str(code), "<llm_metric>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001
            return None, [f"Runtime exec failed: {exc}"]
        fn = ns.get("create_metric")
        if not callable(fn):
            issues.append("create_metric(context) not found after exec.")
            return None, issues
        return fn, issues

    @classmethod
    def _exec_problem_class(cls, code: str) -> tuple[type[Any] | None, list[str]]:
        ok, issues = cls._validate_common_python_code(
            str(code),
            require_class=True,
            require_create_metric=False,
        )
        if not ok:
            return None, issues
        ns = cls._restricted_exec_namespace()
        try:
            exec(compile(str(code), "<llm_problem>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001
            return None, [f"Runtime exec failed: {exc}"]
        imported_problem = ns.get("Problem")
        for name, obj in ns.items():
            try:
                if not isinstance(obj, type):
                    continue
                if obj is imported_problem or name == "Problem":
                    continue
                if hasattr(obj, "_evaluate"):
                    return obj, issues
            except Exception:
                continue
        return None, ["No Problem class found after exec."]

    @classmethod
    def _metric_oracle_value(cls, metric_kind: str, front: Any, context: dict[str, Any]) -> float:
        try:
            from metrics import community_metrics as cm
        except Exception:  # noqa: BLE001
            return float("nan")
        metric_map = {
            "hv": getattr(cm, "_metric_HV", None),
            "hv_mc": getattr(cm, "_metric_HV", None),
            "gd": getattr(cm, "_metric_GD", None),
            "igd": getattr(cm, "_metric_IGD", None),
            "igdp": getattr(cm, "_metric_IGDp", None),
            "deltap": getattr(cm, "_metric_DeltaP", None),
        }
        fn = metric_map.get(metric_kind)
        if not callable(fn):
            return float("nan")
        try:
            return cls._safe_float(fn(front, dict(context)))
        except Exception:  # noqa: BLE001
            return float("nan")

    @classmethod
    def _build_metric_validation_case(cls, n_obj: int) -> tuple[Any, dict[str, Any]]:
        import numpy as _np

        m = int(max(2, n_obj))
        rng = _np.random.default_rng(7)
        # Synthetic PF on simplex-like manifold in [0,1]^m.
        pf = rng.random((256, m))
        pf /= _np.clip(pf.sum(axis=1, keepdims=True), 1e-12, None)
        # Candidate front = perturbed subset, clipped.
        idx = rng.choice(_np.arange(len(pf)), size=96, replace=False)
        F = _np.clip(pf[idx] + 0.03 * rng.normal(size=(96, m)), 0.0, 1.2)
        context = {
            "pareto_front": pf,
            "ref_pf": pf,
            "n_obj": m,
            "hv_mc_samples": 30000,
            "seed": 11,
            "case_name": f"synthetic_m{m}",
        }
        return F, context

    @classmethod
    def _build_metric_validation_problem_case_zdt1(cls) -> tuple[Any, dict[str, Any]] | None:
        import numpy as _np

        try:
            from problems.multi.zdt import ZDT1 as _ZDT1
        except Exception:
            return None

        try:
            problem = _ZDT1(n_var=30)
            n_var = int(max(1, int(getattr(problem, "n_var", 30) or 30)))
            rng = _np.random.default_rng(23)
            X = rng.random((96, n_var), dtype=float)
            F = None
            evaluate = getattr(problem, "evaluate", None)
            if callable(evaluate):
                F = evaluate(X, return_values_of=["F"])
                if isinstance(F, (tuple, list)):
                    F = F[0]
            if F is None:
                out: dict[str, Any] = {}
                problem._evaluate(X, out)
                F = out.get("F")
            F_arr = _np.asarray(F, dtype=float)
            if F_arr.ndim == 1:
                F_arr = F_arr.reshape(1, -1)
            if F_arr.ndim != 2 or F_arr.shape[1] != 2 or F_arr.size == 0 or not _np.all(_np.isfinite(F_arr)):
                return None

            pf = None
            for fn_name in ("pareto_front", "_calc_pareto_front"):
                fn = getattr(problem, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    pf = fn(n_pareto_points=256)
                except TypeError:
                    try:
                        pf = fn()
                    except Exception:
                        pf = None
                except Exception:
                    pf = None
                if pf is not None:
                    break
            if pf is None:
                return None
            pf_arr = _np.asarray(pf, dtype=float)
            if pf_arr.ndim == 1:
                pf_arr = pf_arr.reshape(1, -1)
            if pf_arr.ndim != 2 or pf_arr.shape[1] != 2 or pf_arr.size == 0 or not _np.all(_np.isfinite(pf_arr)):
                return None

            nested_cfg = {
                "pareto_front": pf_arr,
                "ref_pf": pf_arr,
                "n_obj": 2,
                "hv_mc_samples": 8000,
                "seed": 29,
                "problem": problem,
            }
            context = {
                "pareto_front": pf_arr,
                "ref_pf": pf_arr,
                "n_obj": 2,
                "hv_mc_samples": 8000,
                "seed": 29,
                "problem": problem,
                "config": dict(nested_cfg),
                "case_name": "zdt1_real_context",
            }
            return F_arr, context
        except Exception:
            return None

    @classmethod
    def _build_metric_validation_cases(cls, metric_kind: str, n_obj: int) -> list[tuple[Any, dict[str, Any]]]:
        dims: list[int] = []
        target = int(max(2, n_obj))
        dims.append(target)
        if metric_kind in {"hv", "hv_mc", "gd", "igd", "igdp", "deltap"}:
            dims.extend([2, 3])
            if target > 3:
                dims.append(min(10, target))
            else:
                dims.append(5)
        uniq_dims: list[int] = []
        for d in dims:
            d = int(max(2, d))
            if d not in uniq_dims:
                uniq_dims.append(d)
        cases: list[tuple[Any, dict[str, Any]]] = []
        for d in uniq_dims[:4]:
            F, context = cls._build_metric_validation_case(d)
            # Keep validation fast while still meaningful across dimensions.
            context["hv_mc_samples"] = int(min(20000, max(4000, context.get("hv_mc_samples", 8000))))
            cases.append((F, context))
            # Also test wrapped EmoPyLab-style metric context (`context['config']`) to catch missing unwrapping.
            wrapped_cfg = {
                key: value
                for key, value in context.items()
                if key not in {"case_name", "config"}
            }
            wrapped_context = {
                "config": dict(wrapped_cfg),
                "n_obj": int(context.get("n_obj", d) or d),
                "case_name": f"wrapped_{context.get('case_name', f'synthetic_m{d}')}",
            }
            cases.append((F, wrapped_context))

        if metric_kind in {"hv", "hv_mc", "gd", "igd", "igdp", "deltap"}:
            zdt1_case = cls._build_metric_validation_problem_case_zdt1()
            if zdt1_case is not None:
                cases.append(zdt1_case)
        return cases

    @staticmethod
    def _problem_pf_mode_from_code(code: str) -> str:
        text = str(code or "")
        if "_calc_pareto_front" not in text:
            return "unavailable"
        if "n_samples = int(max(2000, n_pf * 40))" in text:
            return "approximate_generic"
        if "np.linspace(0.2807753191" in text or "regions = [" in text or "1.0 - np.sqrt(x)" in text:
            return "exact_family"
        return "custom_or_unknown"

    @classmethod
    def _probe_problem_pareto_front(cls, problem: Any, *, n_obj: int) -> tuple[str, dict[str, Any]]:
        import numpy as _np
        report: dict[str, Any] = {}
        fn = getattr(problem, "_calc_pareto_front", None)
        if not callable(fn):
            return "unavailable", report
        try:
            pf = fn(n_pareto_points=32)
        except TypeError:
            try:
                from util.ref_dirs import get_reference_directions as _get_ref_dirs
                dirs = _get_ref_dirs("das-dennis", int(n_obj), n_partitions=12)
                pf = fn(ref_dirs=dirs)
            except Exception as exc:  # noqa: BLE001
                report["pf_probe_error"] = str(exc)
                return "available_but_probe_failed", report
        except Exception as exc:  # noqa: BLE001
            report["pf_probe_error"] = str(exc)
            return "available_but_probe_failed", report

        try:
            pf_arr = _np.asarray(pf, dtype=float)
            report["pf_probe_shape"] = list(pf_arr.shape)
            report["pf_probe_finite"] = bool(_np.all(_np.isfinite(pf_arr))) if pf_arr.size else True
            if pf_arr.ndim == 2 and pf_arr.shape[1] == int(max(1, n_obj)):
                return "available", report
            return "available_shape_mismatch", report
        except Exception as exc:  # noqa: BLE001
            report["pf_probe_error"] = str(exc)
            return "available_but_probe_failed", report

    @classmethod
    def _validate_metric_bundle_runtime(cls, bundle: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
        import numpy as _np
        issues: list[str] = []
        report: dict[str, Any] = {"kind": "metric", "checks": [], "validation_cases": []}

        cpu_factory_fn, cpu_exec_issues = cls._exec_metric_factory(str(bundle.get("cpu_code", "")))
        jax_factory_fn, jax_exec_issues = cls._exec_metric_factory(str(bundle.get("jax_code", "")))
        issues.extend([f"CPU runtime: {x}" for x in cpu_exec_issues])
        issues.extend([f"JAX runtime: {x}" for x in jax_exec_issues])
        if not callable(cpu_factory_fn) or not callable(jax_factory_fn):
            return False, issues, report

        metric_kind = cls._metric_kind_hint(bundle)
        report["metric_kind_hint"] = metric_kind or "generic"

        try:
            # Smoke instantiate once on default dimension first.
            base_n_obj = int(max(2, int(bundle.get("n_obj", 2) or 2)))
            _, base_context = cls._build_metric_validation_case(base_n_obj)
            cpu_metric = cpu_factory_fn(dict(base_context))
            jax_metric = jax_factory_fn(dict(base_context))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Metric factory instantiation failed: {exc}")
            return False, issues, report

        if not callable(cpu_metric) or not callable(jax_metric):
            issues.append("create_metric(context) did not return callable metric(front).")
            return False, issues, report

        prompt_text = str(bundle.get("_prompt", "") or "")
        skip_oracle_for_variant = bool(metric_kind and cls._looks_like_novel_metric_variant_request(prompt_text))
        if skip_oracle_for_variant:
            report["oracle_skipped_reason"] = "variant_or_novel_metric_request"

        worst_cpu_jax_abs = 0.0
        worst_cpu_jax_rel = 0.0
        worst_oracle_abs = 0.0
        worst_oracle_rel = 0.0
        first_cpu_val = float("nan")
        first_jax_val = float("nan")
        first_oracle_val = float("nan")

        cases = cls._build_metric_validation_cases(metric_kind or "generic", int(max(2, int(bundle.get("n_obj", 2) or 2))))

        for case_idx, (F, context) in enumerate(cases):
            case_report: dict[str, Any] = {"idx": case_idx, "m": int(context.get("n_obj", 0))}
            if context.get("case_name") is not None:
                case_report["case_name"] = str(context.get("case_name"))
            try:
                cpu_metric_case = cpu_factory_fn(dict(context))
                jax_metric_case = jax_factory_fn(dict(context))
            except Exception as exc:  # noqa: BLE001
                issues.append(f"Metric factory instantiation failed on case m={case_report['m']}: {exc}")
                report["validation_cases"].append(case_report)
                continue

            try:
                cpu_val = cls._safe_float(cpu_metric_case(F))
            except Exception as exc:  # noqa: BLE001
                cpu_val = float("nan")
                issues.append(f"CPU metric(front) failed on case m={case_report['m']}: {exc}")
            try:
                jax_val = cls._safe_float(jax_metric_case(F))
            except Exception as exc:  # noqa: BLE001
                jax_val = float("nan")
                issues.append(f"JAX metric(front) failed on case m={case_report['m']}: {exc}")

            case_report["cpu_value"] = cpu_val
            case_report["jax_value"] = jax_val

            if case_idx == 0:
                first_cpu_val = cpu_val
                first_jax_val = jax_val

            if not math.isfinite(cpu_val):
                issues.append(f"CPU metric returned non-finite value in runtime check (m={case_report['m']}).")
            if not math.isfinite(jax_val):
                issues.append(f"JAX metric returned non-finite value in runtime check (m={case_report['m']}).")

            if math.isfinite(cpu_val) and math.isfinite(jax_val):
                abs_err = abs(cpu_val - jax_val)
                rel_err = abs_err / max(abs(cpu_val), abs(jax_val), 1e-12)
                case_report["cpu_jax_abs_err"] = abs_err
                case_report["cpu_jax_rel_err"] = rel_err
                worst_cpu_jax_abs = max(worst_cpu_jax_abs, abs_err)
                worst_cpu_jax_rel = max(worst_cpu_jax_rel, rel_err)
                rel_tol = 0.25 if metric_kind == "hv_mc" else 0.05
                abs_tol = 1e-6 if metric_kind != "hv_mc" else 5e-3
                if not (abs_err <= abs_tol or rel_err <= rel_tol):
                    issues.append(
                        f"CPU/JAX parity check failed on m={case_report['m']} (abs_err={abs_err:.6g}, rel_err={rel_err:.6g})."
                    )

            if metric_kind and not skip_oracle_for_variant:
                oracle = cls._metric_oracle_value(metric_kind, F, dict(context))
                case_report["oracle_value"] = oracle
                if case_idx == 0:
                    first_oracle_val = oracle
                if math.isfinite(oracle) and math.isfinite(cpu_val):
                    abs_err = abs(cpu_val - oracle)
                    rel_err = abs_err / max(abs(oracle), 1e-12)
                    case_report["oracle_abs_err"] = abs_err
                    case_report["oracle_rel_err"] = rel_err
                    worst_oracle_abs = max(worst_oracle_abs, abs_err)
                    worst_oracle_rel = max(worst_oracle_rel, rel_err)
                    if metric_kind == "hv_mc":
                        oracle_scale = abs(oracle)
                        hv_mc_ok = (rel_err <= 0.35) if oracle_scale > 1e-6 else (abs_err <= 5e-3)
                        if not hv_mc_ok:
                            issues.append(
                                f"Known-metric sanity check failed for HV Monte Carlo on m={case_report['m']} "
                                f"(abs_err={abs_err:.6g}, rel_err={rel_err:.6g})."
                            )
                    else:
                        if not (abs_err <= 1e-4 or rel_err <= 5e-2):
                            issues.append(
                                f"Known-metric compatibility check failed on m={case_report['m']} "
                                f"(abs_err={abs_err:.6g}, rel_err={rel_err:.6g})."
                            )

            # Metamorphic checks (record for all; enforce for known metrics only)
            try:
                perm = _np.random.default_rng(101 + case_idx).permutation(F.shape[0])
                F_perm = F[perm]
                cpu_perm_val = cls._safe_float(cpu_metric_case(F_perm))
                case_report["cpu_perm_value"] = cpu_perm_val
                if math.isfinite(cpu_val) and math.isfinite(cpu_perm_val):
                    perm_abs = abs(cpu_val - cpu_perm_val)
                    perm_rel = perm_abs / max(abs(cpu_val), abs(cpu_perm_val), 1e-12)
                    case_report["perm_abs_err"] = perm_abs
                    case_report["perm_rel_err"] = perm_rel
                    if metric_kind and perm_rel > 1e-6 and perm_abs > 1e-8:
                        issues.append(
                            f"Permutation invariance failed for known metric on m={case_report['m']} "
                            f"(abs_err={perm_abs:.6g}, rel_err={perm_rel:.6g})."
                        )
                cpu_repeat_val = cls._safe_float(cpu_metric_case(F))
                case_report["cpu_repeat_value"] = cpu_repeat_val
                if math.isfinite(cpu_val) and math.isfinite(cpu_repeat_val):
                    rep_abs = abs(cpu_val - cpu_repeat_val)
                    rep_rel = rep_abs / max(abs(cpu_val), abs(cpu_repeat_val), 1e-12)
                    case_report["repeat_abs_err"] = rep_abs
                    case_report["repeat_rel_err"] = rep_rel
                    if metric_kind and rep_rel > 1e-6 and rep_abs > 1e-8:
                        issues.append(
                            f"Determinism check failed for known metric on m={case_report['m']} "
                            f"(abs_err={rep_abs:.6g}, rel_err={rep_rel:.6g})."
                        )
            except Exception as exc:  # noqa: BLE001
                case_report["metamorphic_error"] = str(exc)
                if metric_kind:
                    issues.append(f"Metamorphic checks failed on m={case_report['m']}: {exc}")

            report["validation_cases"].append(case_report)

        report["cpu_value"] = first_cpu_val
        report["jax_value"] = first_jax_val
        if math.isfinite(worst_cpu_jax_abs) or math.isfinite(worst_cpu_jax_rel):
            report["cpu_jax_abs_err"] = worst_cpu_jax_abs
            report["cpu_jax_rel_err"] = worst_cpu_jax_rel
        if metric_kind and not skip_oracle_for_variant:
            report["oracle_value"] = first_oracle_val
            report["oracle_abs_err"] = worst_oracle_abs
            report["oracle_rel_err"] = worst_oracle_rel
        report["multi_scenario_dims"] = [int(c[1].get("n_obj", 0)) for c in cases]

        return len(issues) == 0, issues, report

    @classmethod
    def _validate_problem_bundle_runtime(cls, bundle: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
        import numpy as _np

        issues: list[str] = []
        report: dict[str, Any] = {"kind": "problem", "checks": []}
        report["pf_mode_from_code"] = cls._problem_pf_mode_from_code(str(bundle.get("cpu_code", "")))
        cpu_cls, cpu_exec_issues = cls._exec_problem_class(str(bundle.get("cpu_code", "")))
        jax_cls, jax_exec_issues = cls._exec_problem_class(str(bundle.get("jax_code", "")))
        issues.extend([f"CPU runtime: {x}" for x in cpu_exec_issues])
        issues.extend([f"JAX runtime: {x}" for x in jax_exec_issues])
        if cpu_cls is None or jax_cls is None:
            return False, issues, report

        try:
            cpu_problem = cpu_cls()
            jax_problem = jax_cls()
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Problem instantiation failed: {exc}")
            return False, issues, report

        n_var = int(max(1, int(getattr(cpu_problem, "n_var", bundle.get("n_var", 30)) or 30)))
        n_obj = int(max(1, int(getattr(cpu_problem, "n_obj", bundle.get("n_obj", 2)) or 2)))
        rng = _np.random.default_rng(17)
        xl = _np.asarray(getattr(cpu_problem, "xl", 0.0), dtype=float)
        xu = _np.asarray(getattr(cpu_problem, "xu", 1.0), dtype=float)
        if xl.ndim == 0:
            xl = _np.full(n_var, float(xl))
        else:
            xl = _np.ravel(xl).astype(float)
            if xl.size != n_var:
                xl = _np.resize(xl, n_var)
        if xu.ndim == 0:
            xu = _np.full(n_var, float(xu))
        else:
            xu = _np.ravel(xu).astype(float)
            if xu.size != n_var:
                xu = _np.resize(xu, n_var)

        # Some generated problems may expose placeholder/invalid bounds (nan/inf or reversed bounds).
        xl = _np.where(_np.isfinite(xl), xl, 0.0)
        xu = _np.where(_np.isfinite(xu), xu, 1.0)
        lo = _np.minimum(xl, xu)
        hi = _np.maximum(xl, xu)
        same = _np.isclose(lo, hi)
        hi = _np.where(same, lo + 1.0, hi)

        try:
            X = rng.uniform(lo, hi, size=(12, n_var))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Runtime input sampling failed with problem bounds ({exc}); using [0,1] fallback.")
            X = rng.uniform(0.0, 1.0, size=(12, n_var))

        out_cpu: dict[str, Any] = {}
        out_jax: dict[str, Any] = {}
        try:
            cpu_problem._evaluate(X, out_cpu)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"CPU _evaluate runtime failed: {exc}")
        try:
            jax_problem._evaluate(X, out_jax)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"JAX _evaluate runtime failed: {exc}")

        Fc = _np.asarray(out_cpu.get("F", []), dtype=float)
        Fj = _np.asarray(out_jax.get("F", []), dtype=float)
        report["cpu_F_shape"] = list(Fc.shape) if Fc.ndim == 2 else [*Fc.shape]
        report["jax_F_shape"] = list(Fj.shape) if Fj.ndim == 2 else [*Fj.shape]
        if Fc.ndim != 2 or Fc.shape != (X.shape[0], n_obj):
            issues.append(f"CPU _evaluate produced invalid F shape {Fc.shape}, expected ({X.shape[0]}, {n_obj}).")
        if Fj.ndim != 2 or Fj.shape != (X.shape[0], n_obj):
            issues.append(f"JAX _evaluate produced invalid F shape {Fj.shape}, expected ({X.shape[0]}, {n_obj}).")
        if Fc.size and not _np.all(_np.isfinite(Fc)):
            issues.append("CPU _evaluate produced non-finite F values.")
        if Fj.size and not _np.all(_np.isfinite(Fj)):
            issues.append("JAX _evaluate produced non-finite F values.")

        if Fc.shape == Fj.shape and Fc.size:
            abs_err = float(_np.max(_np.abs(Fc - Fj)))
            denom = float(max(_np.max(_np.abs(Fc)), _np.max(_np.abs(Fj)), 1e-12))
            rel_err = abs_err / denom
            report["cpu_jax_max_abs_err"] = abs_err
            report["cpu_jax_max_rel_err"] = rel_err
            if not (abs_err <= 1e-5 or rel_err <= 1e-3):
                issues.append(
                    f"Problem CPU/JAX parity check failed (max_abs_err={abs_err:.6g}, max_rel_err={rel_err:.6g})."
                )

        pf_probe_status, pf_probe_report = cls._probe_problem_pareto_front(cpu_problem, n_obj=n_obj)
        report["pf_probe_status"] = pf_probe_status
        report.update(pf_probe_report)
        if report.get("pf_mode_from_code") == "unavailable":
            report["pf_mode"] = "unavailable"
        elif report.get("pf_mode_from_code") == "approximate_generic":
            report["pf_mode"] = "approximate"
        elif report.get("pf_mode_from_code") == "exact_family":
            report["pf_mode"] = "exact"
        else:
            report["pf_mode"] = "custom"
        if pf_probe_status in {"available", "available_shape_mismatch", "available_but_probe_failed"}:
            # Keep explicit runtime observation separated from code inference.
            report["pf_probe_available"] = True
        else:
            report["pf_probe_available"] = False

        return len(issues) == 0, issues, report

    @classmethod
    def validate_artifact_bundle_detailed(cls, bundle: dict[str, Any]) -> dict[str, Any]:
        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        cpu_code = str(bundle.get("cpu_code", ""))
        jax_code = str(bundle.get("jax_code", ""))
        issues: list[str] = []
        report: dict[str, Any] = {"artifact_type": artifact, "ok": False, "issues": issues, "checks": []}
        if not cpu_code.strip():
            issues.append("CPU code is empty.")
        if not jax_code.strip():
            issues.append("JAX code is empty.")
        if issues:
            bundle["_validation_report"] = report
            return report

        if artifact == "metric":
            ok_cpu, cpu_issues = cls.validate_metric_code(cpu_code)
            ok_jax, jax_issues = cls.validate_metric_code(jax_code)
        else:
            ok_cpu, cpu_issues = cls.validate_problem_code(cpu_code)
            ok_jax, jax_issues = cls.validate_problem_code(jax_code)
        issues.extend([f"CPU: {msg}" for msg in cpu_issues])
        issues.extend([f"JAX: {msg}" for msg in jax_issues])

        runtime_ok = False
        runtime_report: dict[str, Any] = {}
        if artifact == "metric":
            runtime_ok, runtime_issues, runtime_report = cls._validate_metric_bundle_runtime(bundle)
        else:
            runtime_ok, runtime_issues, runtime_report = cls._validate_problem_bundle_runtime(bundle)
        issues.extend(runtime_issues)
        report["runtime"] = runtime_report
        report["ok"] = bool(ok_cpu and ok_jax and runtime_ok and not issues)
        bundle["_validation_report"] = report
        return report

    @classmethod
    def validate_artifact_bundle(cls, bundle: dict[str, Any]) -> tuple[bool, list[str]]:
        report = cls.validate_artifact_bundle_detailed(bundle)
        return bool(report.get("ok", False)), list(report.get("issues", []))

    @classmethod
    def _problem_target_dir(cls, base_dir: Path, n_obj: int) -> Path:
        n = int(max(1, int(n_obj)))
        bucket = "single" if n == 1 else ("multi" if n <= 3 else "many")
        return Path(base_dir) / "problems" / bucket

    @staticmethod
    def _safe_python_artifact_filename(value: Any, *, field_name: str) -> str:
        text = str(value or "").strip()
        path = Path(text)
        if not text or path.is_absolute() or path.name != text or path.suffix.lower() != ".py":
            raise ValueError(f"Unsafe {field_name}: {text!r}. Expected a simple .py file name.")
        if path.stem in {"", ".", ".."}:
            raise ValueError(f"Unsafe {field_name}: {text!r}. Expected a simple .py file name.")
        return text

    @classmethod
    def save_artifact_bundle(cls, *, base_dir: Path, bundle: dict[str, Any]) -> tuple[Path, Path]:
        for field_name in ("cpu_file", "jax_file"):
            if field_name in bundle:
                cls._safe_python_artifact_filename(bundle.get(field_name), field_name=field_name)
        cls._normalize_bundle_metadata(bundle)
        artifact = str(bundle.get("artifact_type", "problem")).strip().lower()
        cpu_file = cls._safe_python_artifact_filename(
            bundle.get("cpu_file", "generated.py"),
            field_name="cpu_file",
        )
        jax_file = cls._safe_python_artifact_filename(
            bundle.get("jax_file", "generated_JAX.py"),
            field_name="jax_file",
        )
        cpu_code = str(bundle.get("cpu_code", ""))
        jax_code = str(bundle.get("jax_code", ""))

        if artifact == "metric":
            out_dir = Path(base_dir) / "metrics"
        else:
            out_dir = cls._problem_target_dir(base_dir, int(bundle.get("n_obj", 2)))
        out_dir.mkdir(parents=True, exist_ok=True)

        cpu_path = out_dir / cpu_file
        jax_path = out_dir / jax_file
        cpu_path.write_text(cpu_code, encoding="utf-8")
        jax_path.write_text(jax_code, encoding="utf-8")
        return cpu_path, jax_path

    # Convenience wrappers.
    @classmethod
    def generate_problem_code(
        cls,
        prompt: str,
        *,
        class_name: str,
        n_var: int,
        n_obj: int,
        provider: str = DEFAULT_PROVIDER,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> str:
        _ = model
        bundle = cls.generate_artifact_bundle(
            prompt,
            artifact_type="problem",
            base_name=class_name,
            n_var=n_var,
            n_obj=n_obj,
            provider=provider,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        return str(bundle.get("cpu_code", ""))

    @classmethod
    def save_problem_code(cls, *, base_dir: Path, class_name: str, code: str) -> Path:
        bundle = cls._generate_artifact_bundle_template(
            prompt="",
            artifact_type="problem",
            base_name=class_name,
            n_var=30,
            n_obj=2,
        )
        bundle["cpu_code"] = str(code)
        cls._normalize_bundle_metadata(bundle)
        out_dir = cls._problem_target_dir(base_dir, int(bundle.get("n_obj", 2)))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(bundle.get("cpu_file", f"{class_name}.py"))
        out_path.write_text(str(code), encoding="utf-8")
        return out_path
