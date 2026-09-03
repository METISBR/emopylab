from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Sequence, TypeVar

from .backends import all_variant_suffixes, backend_suffix, normalize_backend


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Backend-token resolution                                                    #
# --------------------------------------------------------------------------- #
def _resolve_backend(prefer_jax: bool | None, backend: str | None) -> str:
    """Resolve the effective backend token from the (legacy) prefer_jax flag
    and/or the new explicit backend token. The explicit token wins."""
    if backend is not None:
        return normalize_backend(backend)
    if prefer_jax is True:
        return "jax"
    if prefer_jax is False:
        return "cpu"
    return "cpu"


# --------------------------------------------------------------------------- #
# Variant-identifier helpers (suffix-parametrized; JAX wrappers kept)         #
# --------------------------------------------------------------------------- #
def looks_like_variant_identifier(value: Any, suffix: str) -> bool:
    text = str(value or "").strip().lower()
    s = str(suffix or "").strip().lower()
    if not text or not s:
        return False
    return (
        text.endswith(f"_{s}")
        or f"_{s}." in text
        or f"_{s}_" in text
        or f".{s}." in text
        or f".{s}_" in text
    )


def looks_like_jax_identifier(value: Any) -> bool:
    return looks_like_variant_identifier(value, "jax")


def strip_variant_suffix(value: str, suffix: str) -> str:
    return re.sub(rf"_{re.escape(str(suffix))}$", "", str(value).strip(), flags=re.IGNORECASE)


def strip_jax_suffix(value: str) -> str:
    return strip_variant_suffix(value, "jax")


def _strip_any_variant_suffix(value: str) -> str:
    out = str(value).strip()
    for suffix in all_variant_suffixes():
        out = strip_variant_suffix(out, suffix)
    return out


def backend_prefers_jax(
    *,
    backend: str | None = None,
    array_backend: str | None = None,
    use_gpu: bool | None = None,
) -> bool:
    if use_gpu is True:
        return True
    if use_gpu is False:
        return False
    token = str(backend or array_backend or "").strip().lower()
    return token in {"gpu", "jax"}


def normalize_backend_token(value: str) -> str:
    """Reduce a spec name to a backend-agnostic comparison token by stripping
    any accelerator suffix and non-alphanumeric characters."""
    base = str(value).strip().lower()
    for suffix in all_variant_suffixes():
        s = suffix.lower()
        base = re.sub(rf"[\s\-_]+{re.escape(s)}$", "", base)
        base = strip_variant_suffix(base, s)
        base = re.sub(rf"\.{re.escape(s)}$", "", base)
    return re.sub(r"[^a-z0-9]+", "", base)


def is_variant_spec(
    name: Any,
    module: Any = None,
    spec_id: Any = None,
    *,
    suffix: str,
) -> bool:
    return (
        looks_like_variant_identifier(name, suffix)
        or looks_like_variant_identifier(module, suffix)
        or looks_like_variant_identifier(spec_id, suffix)
    )


def is_jax_spec(name: Any, module: Any = None, spec_id: Any = None) -> bool:
    return is_variant_spec(name, module, spec_id, suffix="JAX")


def _spec_backend_token(
    name: Any, module: Any = None, spec_id: Any = None
) -> str:
    """Return the backend a spec belongs to, by its identifier suffix."""
    for suffix in all_variant_suffixes():
        if is_variant_spec(name, module, spec_id, suffix=suffix):
            return normalize_backend(suffix)
    return "cpu"


# --------------------------------------------------------------------------- #
# Spec selection                                                              #
# --------------------------------------------------------------------------- #
def select_specs_for_backend(
    specs: Sequence[T],
    *,
    prefer_jax: bool | None = None,
    backend: str | None = None,
    name_getter: Callable[[T], Any],
    module_getter: Callable[[T], Any] | None = None,
    id_getter: Callable[[T], Any] | None = None,
) -> list[T]:
    backend = _resolve_backend(prefer_jax, backend)
    suffix = backend_suffix(backend)

    def _name(spec: T) -> str:
        return str(name_getter(spec)).strip().lower()

    def _spec_token(spec: T) -> str:
        return _spec_backend_token(
            name_getter(spec),
            module_getter(spec) if module_getter is not None else None,
            id_getter(spec) if id_getter is not None else None,
        )

    all_specs = list(specs)

    if suffix:  # accelerated backend (jax / mlx)
        s = suffix.lower()
        variant_specs = [spec for spec in all_specs if _spec_token(spec) == backend]
        if not variant_specs:
            return all_specs
        variant_names = {_name(spec) for spec in variant_specs}
        return [
            spec
            for spec in variant_specs
            if _name(spec).endswith(f"_{s}") or f"{_name(spec)}_{s}" not in variant_names
        ]

    # CPU backend: keep only non-variant specs.
    cpu_specs = [spec for spec in all_specs if _spec_token(spec) == "cpu"]
    return cpu_specs if cpu_specs else all_specs


def map_selected_ids_to_backend(
    selected_ids: Sequence[str],
    *,
    specs_by_id: dict[str, T],
    prefer_jax: bool | None = None,
    backend: str | None = None,
    name_getter: Callable[[T], Any],
    module_getter: Callable[[T], Any] | None = None,
    id_getter: Callable[[T], Any] | None = None,
) -> list[str]:
    backend = _resolve_backend(prefer_jax, backend)
    suffix = backend_suffix(backend).lower()

    def _spec_token(spec: T) -> str:
        sid = id_getter(spec) if id_getter is not None else None
        return _spec_backend_token(
            name_getter(spec),
            module_getter(spec) if module_getter is not None else None,
            sid,
        )

    def _name(spec: T) -> str:
        return str(name_getter(spec)).strip().lower()

    token_index: dict[str, list[str]] = {}
    for sid, spec in specs_by_id.items():
        token = normalize_backend_token(str(name_getter(spec)))
        if not token:
            continue
        token_index.setdefault(token, []).append(str(sid))

    resolved: list[str] = []
    seen: set[str] = set()

    for raw_id in selected_ids:
        sid = str(raw_id)
        selected_spec: T | None = specs_by_id.get(sid)
        if selected_spec is None:
            continue

        chosen_id = sid
        spec_token = _spec_token(selected_spec)
        token = normalize_backend_token(str(name_getter(selected_spec)))
        candidates = [specs_by_id[cid] for cid in token_index.get(token, []) if cid in specs_by_id]

        desired = [c for c in candidates if _spec_token(c) == backend]
        if desired:
            desired_ids = [
                str(id_getter(c)) if id_getter is not None else next(
                    (k for k, v in specs_by_id.items() if v is c),
                    "",
                )
                for c in desired
            ]
            desired_ids = [d for d in desired_ids if d]
            if desired_ids:
                # Prefer explicit suffixed naming for accelerated backends, and
                # non-suffixed naming for CPU.
                desired_ids.sort(
                    key=lambda did: (
                        0
                        if (
                            (suffix and _name(specs_by_id[did]).endswith(f"_{suffix}"))
                            or (not suffix and _spec_token(specs_by_id[did]) == "cpu")
                        )
                        else 1,
                        _name(specs_by_id[did]),
                    )
                )
                chosen_id = desired_ids[0]
        elif spec_token != backend:
            chosen_id = sid

        if chosen_id not in seen:
            seen.add(chosen_id)
            resolved.append(chosen_id)

    return resolved


# --------------------------------------------------------------------------- #
# Operator runtime candidate resolution                                       #
# --------------------------------------------------------------------------- #
def iter_operator_runtime_candidates(
    module_path: str,
    class_name: str,
    *,
    prefer_jax: bool | None = None,
    backend: str | None = None,
) -> list[tuple[str, str]]:
    backend = _resolve_backend(prefer_jax, backend)
    suffix = backend_suffix(backend)  # "" | "JAX" | "MLX"

    module_clean = str(module_path).strip()
    class_clean = str(class_name).strip()
    module_base = _strip_any_variant_suffix(module_clean)
    class_base = _strip_any_variant_suffix(class_clean)

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(mod: str, cls: str) -> None:
        mod_s = str(mod).strip()
        cls_s = str(cls).strip()
        if not mod_s or not cls_s:
            return
        key = (mod_s, cls_s)
        if key in seen:
            return
        seen.add(key)
        candidates.append(key)

    if suffix:  # accelerated backend (jax / mlx)
        _add(f"{module_base}_{suffix}", f"{class_base}_{suffix}")
        _add(f"{module_base}_{suffix}", class_base)
        _add(module_base, f"{class_base}_{suffix}")

        if module_base.startswith("pymoo.operators."):
            local_base = f"operators.{module_base[len('pymoo.operators.'):]}"
            _add(f"{local_base}_{suffix}", class_base)
            _add(f"{local_base}_{suffix}", f"{class_base}_{suffix}")
            _add(local_base, f"{class_base}_{suffix}")
            _add(local_base, class_base)
        _add(module_clean, class_clean)
    else:  # CPU backend
        _add(module_base, class_base)
        _add(module_clean, class_clean)
        _add(module_clean, class_base)
        _add(module_base, class_clean)

        if module_base.startswith("operators."):
            pymoo_base = f"pymoo.operators.{module_base[len('operators.'):]}"
            _add(pymoo_base, class_base)

    return candidates


def resolve_available_operator_target(
    module_path: str,
    class_name: str,
    *,
    prefer_jax: bool | None = None,
    backend: str | None = None,
) -> tuple[str, str] | None:
    for cand_module, cand_class in iter_operator_runtime_candidates(
        module_path,
        class_name,
        prefer_jax=prefer_jax,
        backend=backend,
    ):
        try:
            module = importlib.import_module(cand_module)
            _ = getattr(module, cand_class)
            return cand_module, cand_class
        except Exception:  # noqa: BLE001
            continue
    return None
