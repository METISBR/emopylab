"""
Standalone implementation of Parameters helpers (EmoPyLab 2026).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, Optional, Tuple

from core.variable import Variable

__all__ = [
    "get_data",
    "get_params",
    "get_params_bfs",
    "get_params_rec",
    "apply_to_params",
    "deactivate_params",
    "set_params",
    "flatten",
    "flatten_rec",
    "hierarchical",
]


def get_data(obj: Any) -> dict:
    if not isinstance(obj, dict):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return {}
    else:
        return obj


def get_params(obj: Any, flag: str = "default", only_active: bool = True) -> dict:
    return get_params_bfs(obj, flag, only_active)


def get_params_bfs(obj: Any, flag: str, only_active: bool) -> dict:
    ret: dict = {}
    q = [(None, obj)]
    visited = set()

    while len(q) > 0:
        prefix, current_obj = q.pop()

        if isinstance(current_obj, Variable):
            if current_obj not in visited and current_obj.flag == flag and (not only_active or current_obj.active):
                e = ret
                for name in prefix[:-1]:
                    if name not in e:
                        e[name] = {}
                    e = e[name]
                e[prefix[-1]] = current_obj
            visited.add(current_obj)
        else:
            data = get_data(current_obj)
            for key in data:
                new_prefix = [key] if prefix is None else prefix + [key]
                entry = (new_prefix, data[key])
                q.append(entry)

    return ret


def get_params_rec(obj: Any, visited: set, flag: str, only_active: bool) -> dict:
    data = get_data(obj)
    ret: dict = {}
    for k, v in data.items():
        if isinstance(v, Variable):
            if v not in visited and v.flag == flag and (not only_active or v.active):
                ret[k] = v
            visited.add(v)
        else:
            entry = get_params_rec(v, visited, flag, only_active)
            if entry is not None and len(entry) > 0:
                ret[k] = entry
    return ret


def apply_to_params(obj: Any, func_apply: Callable[[Variable], None]) -> None:
    for _, v in flatten_rec(get_params(obj)):
        func_apply(v)


def deactivate_params(obj: Any) -> None:
    def func(param: Variable) -> None:
        param.active = False

    apply_to_params(obj, func)


def set_params(obj: Any, params: dict, as_value: bool = True) -> None:
    data = get_data(obj)

    for k, v in params.items():
        if isinstance(v, dict):
            set_params(data[k], v, as_value=as_value)
        else:
            if as_value:
                if hasattr(data[k], "set"):
                    data[k].set(v)
                else:
                    data[k] = v
            else:
                data[k] = v


def flatten(params: dict) -> dict:
    return {k: v for k, v in flatten_rec(params)}


def flatten_rec(params: Any, prefix: Optional[str] = None) -> Generator[Tuple[str, Any], None, None]:
    if hasattr(params, "items"):
        for k, v in params.items():
            yield from flatten_rec(v, prefix=f"{prefix}.{k}" if prefix is not None else k)
    else:
        if prefix is not None:
            yield prefix, params


def hierarchical(data: dict) -> dict:
    ret: dict = {}
    groups: dict = {}

    for k, v in data.items():
        a = k.split(".")
        if len(a) > 1:
            prefix = a[0]
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][".".join(a[1:])] = v
        else:
            ret[k] = v

    for name, group in groups.items():
        ret[name] = hierarchical(group)

    return ret
