"""EmoPyLab Benchmark Problem Registry and Factory."""

from __future__ import annotations

import importlib
from typing import Any
from core.problem import Problem


_PROBLEM_REGISTRY: dict[str, type[Problem]] = {}


def register_problem(name: str, cls: type[Problem]) -> None:
    """Register a Problem class under a case-insensitive key."""
    _PROBLEM_REGISTRY[name.lower()] = cls


def get_problem(name: str, *args: Any, **kwargs: Any) -> Problem:
    """Instantiate a benchmark problem by name (e.g. 'zdt1', 'dtlz2', 'wfg1', 'maf1')."""
    key = name.lower()

    if key in _PROBLEM_REGISTRY:
        return _PROBLEM_REGISTRY[key](*args, **kwargs)

    # Auto-resolve standard suites
    # 1. ZDT
    if key.startswith("zdt"):
        from problems.multi.zdt import (
            ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6,
            ZDT1_JAX, ZDT2_JAX, ZDT3_JAX, ZDT4_JAX, ZDT5_JAX, ZDT6_JAX
        )
        mod_map = {
            "zdt1": ZDT1, "zdt2": ZDT2, "zdt3": ZDT3, "zdt4": ZDT4, "zdt5": ZDT5, "zdt6": ZDT6,
            "zdt1_jax": ZDT1_JAX, "zdt2_jax": ZDT2_JAX, "zdt3_jax": ZDT3_JAX,
            "zdt4_jax": ZDT4_JAX, "zdt5_jax": ZDT5_JAX, "zdt6_jax": ZDT6_JAX
        }
        if key in mod_map:
            return mod_map[key](*args, **kwargs)

    # 2. DTLZ
    if key.startswith("dtlz") or key.startswith("c1_dtlz") or key.startswith("c2_dtlz") or key.startswith("c3_dtlz") or key.startswith("cdtlz"):
        try:
            import problems.many.dtlz as dtlz_mod
            cls_name = name.upper()
            if hasattr(dtlz_mod, cls_name):
                return getattr(dtlz_mod, cls_name)(*args, **kwargs)
            # Try case insensitive match in module
            for attr in dir(dtlz_mod):
                if attr.lower() == key:
                    return getattr(dtlz_mod, attr)(*args, **kwargs)
        except Exception:
            pass

    # 3. WFG
    if key.startswith("wfg"):
        try:
            import problems.many.wfg as wfg_mod
            cls_name = name.upper()
            if hasattr(wfg_mod, cls_name):
                return getattr(wfg_mod, cls_name)(*args, **kwargs)
            for attr in dir(wfg_mod):
                if attr.lower() == key:
                    return getattr(wfg_mod, attr)(*args, **kwargs)
        except Exception:
            pass

    # 4. MaF
    if key.startswith("maf"):
        try:
            import problems.many.maf as maf_mod
            cls_name = name.upper()
            if hasattr(maf_mod, cls_name):
                return getattr(maf_mod, cls_name)(*args, **kwargs)
            for attr in dir(maf_mod):
                if attr.lower() == key:
                    return getattr(maf_mod, attr)(*args, **kwargs)
        except Exception:
            pass

    # 5. IMOP
    if key.startswith("imop"):
        try:
            import problems.multi.imop as imop_mod
            cls_name = name.upper()
            if hasattr(imop_mod, cls_name):
                return getattr(imop_mod, cls_name)(*args, **kwargs)
            for attr in dir(imop_mod):
                if attr.lower() == key:
                    return getattr(imop_mod, attr)(*args, **kwargs)
        except Exception:
            pass

    # 6. UF
    if key.startswith("uf"):
        try:
            import problems.multi.uf as uf_mod
            cls_name = name.upper()
            if hasattr(uf_mod, cls_name):
                return getattr(uf_mod, cls_name)(*args, **kwargs)
            for attr in dir(uf_mod):
                if attr.lower() == key:
                    return getattr(uf_mod, attr)(*args, **kwargs)
        except Exception:
            pass

    # 7. Single-objective BBOB / CEC
    if key.startswith("bbob") or key.startswith("f"):
        try:
            import problems.single.bbob as bbob_mod
            for attr in dir(bbob_mod):
                if attr.lower() == key:
                    return getattr(bbob_mod, attr)(*args, **kwargs)
        except Exception:
            pass

    # Generic search across all problem submodules
    for sub in ["problems.many", "problems.multi", "problems.single"]:
        try:
            sub_pkg = importlib.import_module(sub)
            if hasattr(sub_pkg, "__all__"):
                for m in sub_pkg.__all__:
                    try:
                        m_mod = importlib.import_module(f"{sub}.{m}")
                        for attr in dir(m_mod):
                            if attr.lower() == key:
                                target = getattr(m_mod, attr)
                                if isinstance(target, type) and issubclass(target, Problem):
                                    return target(*args, **kwargs)
                    except Exception:
                        pass
        except Exception:
            pass

    raise ValueError(f"Unknown problem '{name}'. Ensure problem module exists under problems/.")
