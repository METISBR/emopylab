# -*- coding: utf-8 -*-
"""
KKT-based convergence indicators (H_old and H_adap) for EmoPyLab.

Faithful and rigorous implementation of:
1. Santos & Xavier (2018): Entropy-inspired convergence indicator with fixed saturation (H_old).
2. Santos & Xavier (2026): Adaptive KKT convergence indicator with quantile winsorization (H_adap).

Both indicators measure the violation of first-order Karush-Kuhn-Tucker (KKT) Pareto
stationarity conditions on the decision space R^n without requiring an external reference set.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize


def compute_kkt_residuals(
    pop_dec: np.ndarray,
    problem,
    h: float = 1e-6,
) -> np.ndarray:
    """Compute Karush-Kuhn-Tucker (KKT) stationarity residuals s(x) = ||q(x)||^2.

    For each candidate solution x in R^n:
      1. Approximates the Jacobian matrix G(x) in R^{n x m} via central finite
         differences along each coordinate direction of the decision space:
             G_{k, j} = [f_j(x + h_k e_k) - f_j(x - h_k e_k)] / (2 h_k)
         requiring 2n function evaluations per individual.
      2. Solves the convex quadratic program over the unit simplex:
             min_{alpha >= 0, sum(alpha) = 1} ||G(x) @ alpha||^2
      3. Calculates the stationarity residual:
             s(x) = ||G(x) @ alpha*||^2

    Parameters
    ----------
    pop_dec : np.ndarray
        Population decision vectors of shape (N, n).
    problem : Problem
        EmoPyLab Problem instance with attributes `n_obj`, `xl`, `xu`, and method `evaluate`.
    h : float, optional
        Relative finite difference step size (default is 1e-6).

    Returns
    -------
    np.ndarray
        Vector of shape (N,) containing non-negative stationarity residuals s(x_i).
    """
    pop_dec = np.asarray(pop_dec, dtype=float)
    if pop_dec.ndim == 1:
        pop_dec = pop_dec[None, :]

    N, n = pop_dec.shape
    m = int(problem.n_obj)
    residuals = np.full(N, np.nan, dtype=float)

    xl = np.asarray(problem.xl, dtype=float).ravel()
    xu = np.asarray(problem.xu, dtype=float).ravel()
    bounds_range = xu - xl
    bounds_range[~np.isfinite(bounds_range) | (bounds_range <= 0.0)] = 1.0
    hvec = h * bounds_range

    for i in range(N):
        x = pop_dec[i].copy()
        G = np.zeros((n, m), dtype=float)
        valid_jacobian = True

        for k in range(n):
            step = hvec[k]
            xp = x.copy()
            xm = x.copy()
            xp[k] = min(xu[k], x[k] + step)
            xm[k] = max(xl[k], x[k] - step)

            denom = xp[k] - xm[k]
            if denom == 0.0:
                G[k, :] = 0.0
                continue

            try:
                # EmoPyLab evaluate expects 2D array (1, n) and returns (1, m)
                fp = np.asarray(problem.evaluate(xp[None, :]), dtype=float).reshape(-1)
                fm = np.asarray(problem.evaluate(xm[None, :]), dtype=float).reshape(-1)

                if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fm))):
                    valid_jacobian = False
                    break

                G[k, :] = (fp - fm) / denom
            except Exception:
                valid_jacobian = False
                break

        if not valid_jacobian or not np.all(np.isfinite(G)):
            continue

        # Solve QP over simplex: min 0.5 * alpha^T H alpha, with H = 2 * (G^T G)
        H = 2.0 * (G.T @ G)
        alpha_init = np.ones(m, dtype=float) / m
        bounds = [(0.0, 1.0)] * m
        constraints = {"type": "eq", "fun": lambda a: np.sum(a) - 1.0}

        norm_H = float(np.linalg.norm(H, ord=np.inf))
        scale = norm_H if norm_H > 1e-12 else 1.0

        try:
            res = minimize(
                lambda a: 0.5 * float(a @ H @ a) / scale,
                x0=alpha_init,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
                options={"disp": False, "maxiter": 150, "ftol": 1e-12},
            )
            if res.success and np.all(np.isfinite(res.x)) and np.all(res.x >= -1e-8):
                alpha = np.clip(res.x, 0.0, 1.0)
                sum_a = np.sum(alpha)
                alpha = alpha / sum_a if sum_a > 0 else alpha_init
            else:
                alpha = alpha_init
        except Exception:
            alpha = alpha_init

        q = G @ alpha
        residuals[i] = float(np.sum(q**2))

    return residuals


def calc_h_old(
    pop_dec: np.ndarray,
    problem,
    residuals: Optional[np.ndarray] = None,
) -> float:
    """Calculate the original H_old convergence indicator with fixed saturation.

    Formula:
        t_i = min(1/e, s_i)
        H_old = - (1/N) * sum_{i=1}^N t_i * log(t_i)
        with convention 0 * log(0) = 0.

    Parameters
    ----------
    pop_dec : np.ndarray
        Population decision vectors (N, n).
    problem : Problem
        EmoPyLab Problem instance.
    residuals : Optional[np.ndarray]
        Precomputed KKT residuals s(x) to avoid re-evaluating derivatives.

    Returns
    -------
    float
        H_old score in [0, 1/e]. Lower is better.
    """
    if residuals is None:
        s = compute_kkt_residuals(pop_dec, problem)
    else:
        s = np.asarray(residuals, dtype=float)

    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float(np.nan)

    sat = 1.0 / np.e
    t = np.clip(s, 0.0, sat)
    term = np.zeros_like(t)
    nonzero = t > 0.0
    term[nonzero] = t[nonzero] * np.log(t[nonzero])
    return float(-np.mean(term))


def calc_h_adap(
    pop_dec: np.ndarray,
    problem,
    alpha: float = 0.10,
    beta: float = 0.90,
    eps: float = 1e-12,
    residuals: Optional[np.ndarray] = None,
) -> float:
    """Calculate the adaptive H_adap convergence indicator via quantile winsorization.

    Formula:
        Q_alpha = quantile(s, alpha)
        Q_beta  = quantile(s, beta)
        shat_i  = min(max(s_i, Q_alpha), Q_beta)
        z_i     = (shat_i - Q_alpha) / (Q_beta - Q_alpha + eps)  in [0, 1]
        H_adap  = - (1/N) * sum_{i=1}^N z_i * log(z_i + eps)
        with convention 0 * log(0) = 0.

    Parameters
    ----------
    pop_dec : np.ndarray
        Population decision vectors (N, n).
    problem : Problem
        EmoPyLab Problem instance.
    alpha : float, optional
        Lower quantile probability in (0, 0.5) (default is 0.10).
    beta : float, optional
        Upper quantile probability in (0.5, 1.0) (default is 0.90).
    eps : float, optional
        Numerical regularizer to avoid division by zero and log(0) (default is 1e-12).
    residuals : Optional[np.ndarray]
        Precomputed KKT residuals s(x) to avoid re-evaluating derivatives.

    Returns
    -------
    float
        H_adap score in [0, 1/e]. Lower is better.
    """
    if residuals is None:
        s = compute_kkt_residuals(pop_dec, problem)
    else:
        s = np.asarray(residuals, dtype=float)

    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float(np.nan)

    ql = float(np.quantile(s, alpha))
    qu = float(np.quantile(s, beta))
    denom = qu - ql

    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0

    shat = np.clip(s, ql, qu)
    z = (shat - ql) / (denom + eps)
    z = np.clip(z, 0.0, 1.0)

    term = np.zeros_like(z)
    nonzero = z > 0.0
    term[nonzero] = z[nonzero] * np.log(z[nonzero] + eps)
    return float(-np.mean(term))


def analyze_quantile_sensitivity(
    pop_dec: np.ndarray,
    problem,
    alpha_levels: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    beta_levels: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
    residuals: Optional[np.ndarray] = None,
) -> Dict[Tuple[float, float], float]:
    """Perform sensitivity analysis of H_adap under varying quantile thresholds.

    Evaluates H_adap across all combinations of (alpha, beta) without repeating
    the expensive Jacobian and QP steps.

    Parameters
    ----------
    pop_dec : np.ndarray
        Population decision vectors (N, n).
    problem : Problem
        EmoPyLab Problem instance.
    alpha_levels : tuple of float
        Grid of lower quantile thresholds.
    beta_levels : tuple of float
        Grid of upper quantile thresholds.
    residuals : Optional[np.ndarray]
        Precomputed KKT residuals.

    Returns
    -------
    Dict[Tuple[float, float], float]
        Mapping (alpha, beta) -> H_adap score.
    """
    if residuals is None:
        s = compute_kkt_residuals(pop_dec, problem)
    else:
        s = residuals

    results: Dict[Tuple[float, float], float] = {}
    for a in alpha_levels:
        for b in beta_levels:
            if a < b:
                score = calc_h_adap(pop_dec, problem, alpha=a, beta=b, residuals=s)
                results[(round(a, 2), round(b, 2))] = score
    return results
