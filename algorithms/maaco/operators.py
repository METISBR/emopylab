"""Operator library for MaACO v2 — four distinct variation operators
that can be chosen dynamically by the contextual bandit or the LLM.

Operators
---------
1. ``llm_sbx`` — Simulated Binary Crossover (SBX) with dynamic
   distribution index ``eta_c`` and per-variable crossover mask.
   Recombination preserves Pareto-aligned building blocks.
2. ``llm_de`` — Differential Evolution (DE/rand/1/bin) with dynamic
   scaling factor ``F`` and crossover probability ``CR``.
3. ``llm_perturb`` — Polynomial Mutation (PM) with dynamic index
   ``eta_m`` and mutation probability ``prob_m``.
4. ``acor_mixture`` — Continuous ACO Gaussian-mixture kernel sampling
   (the classical ACOR operator from Socha & Dorigo 2008), kept as a
   robust baseline exploration arm.

Every operator exposes a uniform interface:
``(archive_X, archive_F, problem, rng, params) -> np.ndarray (pop_size, n_var)``
so they can be called interchangeably.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public operator registry
# ---------------------------------------------------------------------------

OPERATOR_NAMES = ("llm_sbx", "llm_de", "llm_perturb", "acor_mixture")


# ---------------------------------------------------------------------------
# 1. Simulated Binary Crossover (SBX)
# ---------------------------------------------------------------------------

def llm_sbx(X_pop: np.ndarray,
            F_pop: np.ndarray,
            problem: Any,
            rng: np.random.Generator,
            pop_size: int,
            params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Simulated Binary Crossover (Deb & Agrawal 1995) + Polynomial Mutation."""
    params = params or {}
    eta_c = float(params.get("eta_c", 20.0))
    prob_c = float(params.get("prob_c", 1.0))
    eta_m = float(params.get("eta_m", 20.0))
    prob_m = float(params.get("prob_m", 1.0 / max(1, problem.n_var)))

    n_pop, n_var = X_pop.shape
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    if n_pop < 2:
        return rng.uniform(xl, xu, size=(pop_size, n_var))

    # Binary tournament selection: prefer lower index (solutions are
    # already sorted by NSGA-II rank + crowding in MaACO._infill)
    def _pick_parent() -> int:
        i1, i2 = rng.integers(0, n_pop, size=2)
        return min(i1, i2)

    offspring = np.empty((pop_size, n_var), dtype=float)
    idx = 0
    while idx < pop_size:
        idx1, idx2 = _pick_parent(), _pick_parent()
        while idx2 == idx1 and n_pop > 1:
            idx2 = _pick_parent()
        p1 = X_pop[idx1]
        p2 = X_pop[idx2]
        c1, c2 = p1.copy(), p2.copy()

        if rng.random() <= prob_c:
            for j in range(n_var):
                if rng.random() <= 0.5 and abs(p1[j] - p2[j]) > 1e-14:
                    y1, y2 = min(p1[j], p2[j]), max(p1[j], p2[j])
                    yl, yu = xl[j], xu[j]
                    rand = rng.random()
                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta_c + 1.0)
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                    c1[j] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                    beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta_c + 1.0)
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                    c2[j] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                    c1[j] = np.clip(c1[j], yl, yu)
                    c2[j] = np.clip(c2[j], yl, yu)

        # Polynomial mutation on both children
        for child in (c1, c2):
            for j in range(n_var):
                if rng.random() <= prob_m:
                    y = child[j]
                    yl, yu = xl[j], xu[j]
                    if abs(yu - yl) > 1e-14:
                        delta1 = (y - yl) / (yu - yl)
                        delta2 = (yu - y) / (yu - yl)
                        rand = rng.random()
                        mut_pow = 1.0 / (eta_m + 1.0)
                        if rand <= 0.5:
                            xy = 1.0 - delta1
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                            delta_q = val ** mut_pow - 1.0
                        else:
                            xy = 1.0 - delta2
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                            delta_q = 1.0 - val ** mut_pow
                        child[j] = np.clip(y + delta_q * (yu - yl), yl, yu)

        offspring[idx] = c1
        idx += 1
        if idx < pop_size:
            offspring[idx] = c2
            idx += 1

    return np.clip(offspring, xl, xu)


# ---------------------------------------------------------------------------
# 2. Differential Evolution (DE/rand/1/bin)
# ---------------------------------------------------------------------------

def llm_de(X_pop: np.ndarray,
           F_pop: np.ndarray,
           problem: Any,
           rng: np.random.Generator,
           pop_size: int,
           params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Differential Evolution mutation (Storn & Price 1997)."""
    params = params or {}
    F_scale = float(params.get("F", 0.5))
    CR = float(params.get("CR", 0.9))

    n_pop, n_var = X_pop.shape
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    if n_pop < 4:
        return rng.uniform(xl, xu, size=(pop_size, n_var))

    offspring = np.empty((pop_size, n_var), dtype=float)
    for i in range(pop_size):
        # Pick 3 distinct random donors
        candidates = rng.choice(n_pop, size=3, replace=False)
        r1, r2, r3 = candidates[0], candidates[1], candidates[2]
        donor = X_pop[r1] + F_scale * (X_pop[r2] - X_pop[r3])

        # Binomial crossover
        target = X_pop[i % n_pop]
        cross_points = rng.random(n_var) < CR
        if not np.any(cross_points):
            cross_points[rng.integers(0, n_var)] = True
        offspring[i] = np.where(cross_points, donor, target)

    return np.clip(offspring, xl, xu)


# ---------------------------------------------------------------------------
# 3. Polynomial Mutation (PM)
# ---------------------------------------------------------------------------

def llm_perturb(X_pop: np.ndarray,
                F_pop: np.ndarray,
                problem: Any,
                rng: np.random.Generator,
                pop_size: int,
                params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Polynomial Mutation (Deb & Goyal 1996)."""
    params = params or {}
    eta_m = float(params.get("eta_m", 20.0))
    prob_m = float(params.get("prob_m", 1.0 / max(1, problem.n_var)))

    n_pop, n_var = X_pop.shape
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    if n_pop == 0:
        return rng.uniform(xl, xu, size=(pop_size, n_var))

    # Base solutions drawn randomly from archive
    idx = rng.integers(0, n_pop, size=pop_size)
    offspring = X_pop[idx].copy()

    for i in range(pop_size):
        for j in range(n_var):
            if rng.random() <= prob_m:
                y = offspring[i, j]
                yl, yu = xl[j], xu[j]
                if abs(yu - yl) < 1e-14:
                    continue
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rand = rng.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                y = y + delta_q * (yu - yl)
                offspring[i, j] = np.clip(y, yl, yu)

    return np.clip(offspring, xl, xu)


# ---------------------------------------------------------------------------
# 4. Continuous Ant Colony (ACOR) Gaussian Mixture
# ---------------------------------------------------------------------------

def acor_mixture(X_pop: np.ndarray,
                 F_pop: np.ndarray,
                 problem: Any,
                 rng: np.random.Generator,
                 pop_size: int,
                 params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """ACOR continuous ACO kernel sampling (Socha & Dorigo 2008)."""
    params = params or {}
    q = float(params.get("q", 1.0))
    xi = float(params.get("xi", 0.85))

    n_pop, n_var = X_pop.shape
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    if n_pop == 0:
        return rng.uniform(xl, xu, size=(pop_size, n_var))

    # Equal weighting across the archive (preserves diversity)
    k = n_pop
    r_pos = np.arange(k, dtype=float)
    weights = np.exp(-(r_pos ** 2) / (2.0 * (max(q, 0.1) * k) ** 2))
    weights /= weights.sum() + 1e-12

    sigma = np.zeros((k, n_var), dtype=float)
    for l in range(k):
        diffs = np.abs(X_pop - X_pop[l])
        sigma[l] = xi * np.sum(diffs * weights[:, None], axis=0) / max(k - 1, 1)
    sigma = np.maximum(sigma, 1e-10 * (xu - xl))

    chosen = rng.choice(k, size=pop_size, p=weights, replace=True)
    noise = rng.normal(0.0, 1.0, size=(pop_size, n_var)) * sigma[chosen]
    return np.clip(X_pop[chosen] + noise, xl, xu)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

OPERATOR_FUNCS = {
    "llm_sbx": llm_sbx,
    "llm_de": llm_de,
    "llm_perturb": llm_perturb,
    "acor_mixture": acor_mixture,
}


def apply_operator(op_name: str,
                   X_pop: np.ndarray,
                   F_pop: np.ndarray,
                   problem: Any,
                   rng: np.random.Generator,
                   pop_size: int,
                   params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Dispatch to the chosen operator with bounds clipping guaranteed."""
    func = OPERATOR_FUNCS.get(op_name, llm_sbx)
    return func(X_pop, F_pop, problem, rng, pop_size, params=params)
