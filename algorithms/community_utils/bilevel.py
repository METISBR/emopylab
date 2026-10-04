"""Shared pieces of the bilevel ports: quadratic response-surface approximation (with BIC model choice), SQP on the
approximations, and per-level evaluation helpers (the lower level is evaluated without charging the upper budget)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["x2fx", "quad_approx", "approx_value", "sqp_min", "level_fitness"]


def x2fx(X, model):
    X = np.atleast_2d(X)
    n, d = X.shape
    cols = [np.ones(n), *X.T]
    if model == "quadratic":
        cols += [X[:, i] * X[:, j] for i in range(d) for j in range(i + 1, d)]
    if model in ("quadratic", "purequadratic"):
        cols += list((X ** 2).T)
    return np.column_stack(cols)


def quad_approx(y, X):
    """Least-squares linear / pure-quadratic / quadratic response surface chosen by BIC. As in the reference, the
    square and cross terms of a selected pure-quadratic or quadratic model are read from the last fitted (quadratic)
    coefficient vector."""
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).reshape(-1)
    d, n = X.shape[1], X.shape[0]
    if n < d:
        raise ValueError("Cannot compute model as the datasetSize is smaller than dimensions.")
    models = ["linear", "purequadratic", "quadratic"]
    betas, bic = [], []
    for m in models:
        XX = x2fx(X, m)
        b = np.linalg.lstsq(XX, y, rcond=None)[0]
        mse = float(np.real(((y - XX @ b) ** 2).sum() / len(y)))
        betas.append(b)
        with np.errstate(divide="ignore"):
            bic.append(XX.shape[1] * np.log(XX.shape[0]) / XX.shape[0] + 2 * np.log(mse))
    k = int(np.argmin(bic))
    b = betas[k]
    last = betas[-1]
    sq = np.zeros((d, d))
    if models[k] == "purequadratic":
        sq[np.diag_indices(d)] = last[-d:]
    elif models[k] == "quadratic":
        cross = last[1 + d: len(last) - d]
        sq[np.diag_indices(d)] = last[-d:]
        kk = 0
        for i in range(d):
            for j in range(i + 1, d):
                sq[i, j] = sq[j, i] = cross[kk] / 2
                kk += 1
    XX = x2fx(X, models[k])
    with np.errstate(all="ignore"):
        sx = XX.std(0, ddof=1) if n > 1 else np.zeros(XX.shape[1])
        sy = y.std(ddof=1) if n > 1 else 0.0
        sx[sx == 0] = -np.finfo(float).tiny
        sy = -np.finfo(float).tiny if sy == 0 else sy
        XN = (XX - XX.mean(0)) / sx
        yN = (y - y.mean()) / sy
        bN = np.linalg.lstsq(XN, yN, rcond=None)[0]
        mse_norm = float(((yN - XN @ bN) ** 2).sum() / len(yN))
    mse = float(np.real(((y - XX @ b) ** 2).sum() / len(y)))
    if np.isnan(mse):
        raise ValueError("The approximation ended up with NaN values (under-defined system).")
    return dict(constant=float(b[0]), linear=np.asarray(b[1:1 + d], float), sq=sq, mse=mse, mseNorm=mse_norm)


def approx_value(x, m):
    x = np.asarray(x, float)
    return m["constant"] + x @ m["linear"] + x @ m["sq"] @ x


def sqp_min(fun, x0, lb, ub, ineq=()):
    """Bounded SQP (SLSQP) with approximated inequality constraints c(x) <= 0; returns (x, n_function_calls)."""
    cons = [{"type": "ineq", "fun": (lambda x, c=c: -approx_value(x, c))} for c in ineq]
    res = minimize(fun, np.clip(np.asarray(x0, float), lb, ub), method="SLSQP", bounds=list(zip(lb, ub)), constraints=cons,
                   options={"maxiter": 400})
    return np.clip(res.x, lb, ub), int(res.nfev)


def level_fitness(obj, con):
    """Feasible: objective; infeasible: total violation + 1e10."""
    obj = np.asarray(obj, float).reshape(-1)
    if con is None or np.size(con) == 0:
        return obj
    cv = np.maximum(0, con).sum(1)
    return np.where(cv <= 0, obj, cv + 1e10)
