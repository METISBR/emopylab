from __future__ import annotations

import os


ENV_BACKEND_AWARE_LOADING = "EMOPYLAB_BACKEND_AWARE_LOADING"
ENV_BACKEND_AWARE_STAGE = "EMOPYLAB_BACKEND_AWARE_STAGE"

STAGE_OFF = "off"
STAGE_PROBLEMS = "problems"
STAGE_OPERATORS = "operators"
STAGE_METRICS = "metrics"
STAGE_ALL = "all"

VALID_STAGES = {
    STAGE_OFF,
    STAGE_PROBLEMS,
    STAGE_OPERATORS,
    STAGE_METRICS,
    STAGE_ALL,
}

_STAGE_ALIASES = {
    "0": STAGE_OFF,
    "false": STAGE_OFF,
    "off": STAGE_OFF,
    "none": STAGE_OFF,
    "disabled": STAGE_OFF,
    "problem": STAGE_PROBLEMS,
    "problems": STAGE_PROBLEMS,
    "operator": STAGE_OPERATORS,
    "operators": STAGE_OPERATORS,
    "metric": STAGE_METRICS,
    "metrics": STAGE_METRICS,
    "1": STAGE_ALL,
    "true": STAGE_ALL,
    "on": STAGE_ALL,
    "full": STAGE_ALL,
    "all": STAGE_ALL,
    "enabled": STAGE_ALL,
}

_DOMAIN_ORDER = {
    STAGE_PROBLEMS: 1,
    STAGE_OPERATORS: 2,
    STAGE_METRICS: 3,
}


def _is_truthy_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if not token:
        return bool(default)
    if token in {"1", "true", "yes", "on", "enabled"}:
        return True
    if token in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def backend_aware_loading_enabled() -> bool:
    raw = os.getenv(ENV_BACKEND_AWARE_LOADING)
    return _is_truthy_env(raw, default=True)

def rollout_stage() -> str:
    if not backend_aware_loading_enabled():
        return STAGE_OFF

    raw = os.getenv(ENV_BACKEND_AWARE_STAGE)
    token = str(raw or STAGE_ALL).strip().lower()
    stage = _STAGE_ALIASES.get(token, STAGE_ALL)
    if stage not in VALID_STAGES:
        return STAGE_ALL
    return stage



def rollout_allows_domain(domain: str) -> bool:
    domain_token = str(domain or "").strip().lower()
    if domain_token not in _DOMAIN_ORDER:
        return False

    stage = rollout_stage()
    if stage == STAGE_OFF:
        return False
    if stage == STAGE_ALL:
        return True

    stage_rank = _DOMAIN_ORDER.get(stage, 0)
    domain_rank = _DOMAIN_ORDER.get(domain_token, 999)
    return stage_rank >= domain_rank

