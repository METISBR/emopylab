# emopylab 2026
"""Lightweight surrogate models used by the expensive-optimisation ports (NumPy, no extra packages).

* :class:`RBFExact`   - exact Gaussian radial-basis interpolation with a bias term.
* :class:`RBFNet`     - Gaussian RBF with centres chosen from the data and a regularised linear read-out.
* :class:`Kriging`    - ordinary Kriging (Gaussian correlation, constant trend) with a coarse-to-fine
                        maximum-likelihood search over the length scales (DACE-style).
* :class:`SVRLite`    - epsilon-insensitive kernel ridge regression (RBF), the closed-form relaxation of SVR.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.kernels import pdist2

__all__ = ["RBFExact", "RBFPoly", "RBFNet", "Kriging", "gp_linear_predict", "SVMClassifier"]


class RBFExact:
    """Exact RBF interpolation ``y(x) = sum_i w_i exp(-(c * ||x - x_i|| / spread)^2) + b`` with ``c = 0.8326``."""

    def __init__(self, spread: float = 1.0):
        self.spread = float(max(spread, 1e-12))

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        self.X = X
        A = np.exp(-(0.8326 * pdist2(X, X) / self.spread) ** 2)
        n = len(X)
        done = False
        if n > 40:
            try:  # K is positive definite: w = K^-1 (y - b 1); the bias b minimises ||w||^2 + b^2 (minimum-norm solution)
                from scipy.linalg import cho_factor, cho_solve
                cf = cho_factor(A + 1e-10 * np.eye(n))
                u, v = cho_solve(cf, y), cho_solve(cf, np.ones((n, 1)))
                b = (np.sum(u * v, axis=0) / (np.sum(v * v) + 1.0))
                self.w, self.b = u - v * b, b
                done = True
            except Exception:  # noqa: BLE001
                done = False
        if not done:
            Pp = np.hstack([A, np.ones((n, 1))])                   # (Q, Q+1): unknowns are [w, b]
            sol, *_ = np.linalg.lstsq(Pp, y, rcond=None)            # minimum-norm exact solution
            self.w, self.b = sol[:-1], sol[-1]
        return self

    def predict(self, Xn):
        Xn = np.atleast_2d(np.asarray(Xn, dtype=float))
        A = np.exp(-(0.8326 * pdist2(Xn, self.X) / self.spread) ** 2)
        out = A @ self.w + self.b
        return out.reshape(-1) if out.shape[1] == 1 else out


class RBFNet:
    """Gaussian RBF regression with all training points as centres and a ridge-regularised read-out."""

    def __init__(self, spread: float | None = None, ridge: float = 1e-8):
        self.spread, self.ridge = spread, ridge

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        d = pdist2(X, X)
        self.spread_ = self.spread if self.spread else max(float(np.median(d[d > 0])) if np.any(d > 0) else 1.0, 1e-12)
        self.X = X
        K = np.exp(-(d / self.spread_) ** 2)
        self.w = np.linalg.solve(K + self.ridge * np.eye(len(X)), y)
        return self

    def predict(self, Xn):
        K = np.exp(-(pdist2(np.atleast_2d(Xn), self.X) / self.spread_) ** 2)
        out = K @ self.w
        return out.reshape(-1) if out.shape[1] == 1 else out


class Kriging:
    """Ordinary Kriging with an isotropic-or-ARD Gaussian correlation ``exp(-sum_k theta_k (x_k - x'_k)^2)``.

    ``fit`` maximises the concentrated likelihood over ``log10(theta)`` by a coarse-to-fine pattern search
    (robust and deterministic; adequate for the small training sets used by surrogate-assisted EAs).
    ``predict(X, return_std=True)`` also returns the Kriging standard error (for EI / LCB acquisition)."""

    def __init__(self, ard: bool = True, theta_bounds=(-3.0, 3.0), n_iter: int = 40):
        self.ard, self.bounds, self.n_iter = ard, theta_bounds, n_iter

    def _corr(self, A, B, theta):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2 * theta).sum(axis=2)
        return np.exp(-d2)

    def _nll(self, log_theta, X, y):
        theta = 10.0 ** log_theta
        n = len(X)
        R = self._corr(X, X, theta) + 1e-10 * np.eye(n)
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return np.inf, None
        Li1 = np.linalg.solve(L, np.ones(n))
        Liy = np.linalg.solve(L, y)
        beta = (Li1 @ Liy) / (Li1 @ Li1)
        r = Liy - beta * Li1
        sigma2 = max(float(r @ r) / n, 1e-300)
        return n * np.log(sigma2) + 2.0 * np.sum(np.log(np.diag(L))), (L, beta, sigma2)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        Xn = (X - self.lo) / self.span
        self.ym, self.ys = y.mean(), y.std() if y.std() > 0 else 1.0
        yn = (y - self.ym) / self.ys
        D = Xn.shape[1] if self.ard else 1
        best = np.zeros(D)
        f, _ = self._nll(best if self.ard else np.full(Xn.shape[1], best[0]), Xn, yn)
        step = 1.0
        for _ in range(self.n_iter):
            improved = False
            for k in range(D):
                for s in (step, -step):
                    cand = best.copy()
                    cand[k] = np.clip(cand[k] + s, *self.bounds)
                    lt = cand if self.ard else np.full(Xn.shape[1], cand[0])
                    fc, _ = self._nll(lt, Xn, yn)
                    if fc < f:
                        best, f, improved = cand, fc, True
            if not improved:
                step /= 2.0
                if step < 0.05:
                    break
        lt = best if self.ard else np.full(Xn.shape[1], best[0])
        _, aux = self._nll(lt, Xn, yn)
        self.theta = 10.0 ** lt
        self.Xn = Xn
        if aux is None:  # degenerate: fall back to nugget-regularised isotropic model
            self.theta = np.full(Xn.shape[1], 1.0)
            R = self._corr(Xn, Xn, self.theta) + 1e-6 * np.eye(len(Xn))
            L = np.linalg.cholesky(R)
            Li1 = np.linalg.solve(L, np.ones(len(Xn)))
            beta = (Li1 @ np.linalg.solve(L, yn)) / (Li1 @ Li1)
            aux = (L, beta, 1.0)
        self.L, self.beta, self.sigma2 = aux
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yn - self.beta))
        return self

    def predict(self, Xq, return_std: bool = False):
        Xq = (np.atleast_2d(np.asarray(Xq, dtype=float)) - self.lo) / self.span
        r = self._corr(Xq, self.Xn, self.theta)
        mu = (self.beta + r @ self.alpha) * self.ys + self.ym
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, r.T)
        var = self.sigma2 * np.maximum(1.0 - np.sum(v * v, axis=0), 0.0) * self.ys ** 2
        return mu, np.sqrt(var)


def gp_linear_predict(x, y, xs, noise: float = 0.01):
    """Exact GP regression with zero mean, linear covariance ``k(a, b) = a * b`` and Gaussian noise
    (``sigma = noise``) on scalar inputs.  Returns predictive mean and (noisy) variance at ``xs``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    K = np.outer(x, x) + noise ** 2 * np.eye(len(x))
    Ks = np.outer(xs, x)
    sol = np.linalg.solve(K, np.column_stack([y, Ks.T]))
    mu = Ks @ sol[:, 0]
    var = xs * xs - np.sum(Ks * sol[:, 1:].T, axis=1) + noise ** 2
    return mu, np.maximum(var, 0.0)


from util.svm import SVMClassifier  # noqa: E402  (kept importable from here)


class RBFPoly:
    """Interpolating RBF with a linear polynomial tail on data scaled to [-1, 1] (inputs and targets);
    kernel ``'gaussian'`` = exp(-(sqrt(ln 2) r)^2) or ``'cubic'`` = r^3.  Targets may be a matrix (one column per output)."""

    def __init__(self, kernel: str = "gaussian"):
        self.kernel = kernel

    def _phi(self, r):
        return np.exp(-(np.sqrt(np.log(2.0)) * r) ** 2) if self.kernel == "gaussian" else r ** 3

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        N, D = X.shape
        self.xmin, self.xmax = X.min(axis=0), X.max(axis=0)
        self.ymin, self.ymax = y.min(axis=0), y.max(axis=0)
        with np.errstate(all="ignore"):
            ax = 2.0 / (self.xmax - self.xmin) * (X - self.xmin) - 1
            ay = 2.0 / (self.ymax - self.ymin) * (y - self.ymin) - 1
        P = np.hstack([np.ones((N, 1)), ax])
        A = np.block([[self._phi(pdist2(ax, ax)), P], [P.T, np.zeros((D + 1, D + 1))]])
        b = np.vstack([ay, np.zeros((D + 1, y.shape[1]))])
        A, b = np.nan_to_num(A), np.nan_to_num(b)
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A, b, rcond=None)[0]
        self.alpha, self.beta, self.nodes = theta[:N], theta[N:], ax
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        with np.errstate(all="ignore"):
            x = 2.0 / (self.xmax - self.xmin) * (X - self.xmin) - 1
        y = self._phi(pdist2(x, self.nodes)) @ self.alpha + np.hstack([np.ones((len(x), 1)), x]) @ self.beta
        y = (self.ymax - self.ymin) / 2 * (y + 1) + self.ymin
        return y[:, 0] if y.shape[1] == 1 else y
