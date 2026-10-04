"""Binary soft-margin SVM solved by SMO (no extra packages), shared by problems and algorithms."""

from __future__ import annotations

import numpy as np


class SVMClassifier:
    """Binary soft-margin SVM with an RBF kernel ``exp(-||x-z||^2 / scale^2)`` (labels -1/+1).

    Dual problem solved by SMO with LIBSVM's second-order working-set selection (deterministic, no extra
    packages).  ``decision_function(X) > 0`` is the +1 class; the value is the signed margin score."""

    def __init__(self, C: float = 1.0, kernel_scale: float = 1.0, tol: float = 1e-3, max_iter: int = 5000, kernel: str = "rbf"):
        self.C, self.scale, self.tol, self.max_iter, self.kernel = float(C), float(kernel_scale), float(tol), int(max_iter), kernel

    def _k(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        d2 = np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)
        return np.exp(-d2 / self.scale ** 2)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.where(np.asarray(y).reshape(-1) > 0, 1.0, -1.0)
        n = len(X)
        K = self._k(X, X)
        alpha = np.zeros(n)
        G = -np.ones(n)
        C = self.C
        for _ in range(self.max_iter):
            yG = -y * G
            up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
            if not up.any() or not low.any():
                break
            i = int(np.argmax(np.where(up, yG, -np.inf)))
            gmax = yG[i]
            gmin = np.min(np.where(low, yG, np.inf))
            if gmax - gmin < self.tol:
                break
            cand = np.where(low & (yG < gmax))[0]
            if len(cand) == 0:
                break
            b = gmax - yG[cand]
            a = np.maximum(K[i, i] + K[cand, cand] - 2.0 * K[i, cand], 1e-12)
            j = int(cand[np.argmin(-(b * b) / a)])
            ai, aj = alpha[i], alpha[j]
            quad = max(K[i, i] + K[j, j] - 2.0 * K[i, j], 1e-12)
            if y[i] != y[j]:
                delta = (-G[i] - G[j]) / quad
                diff = ai - aj
                ai_n, aj_n = ai + delta, aj + delta
                if diff > 0:
                    if aj_n < 0:
                        aj_n, ai_n = 0.0, diff
                else:
                    if ai_n < 0:
                        ai_n, aj_n = 0.0, -diff
                if diff > 0 and ai_n > C:
                    ai_n, aj_n = C, C - diff
                elif diff <= 0 and aj_n > C:
                    aj_n, ai_n = C, C + diff
            else:
                delta = (G[i] - G[j]) / quad
                s_ = ai + aj
                ai_n, aj_n = ai - delta, aj + delta
                if s_ > C:
                    if ai_n > C:
                        ai_n, aj_n = C, s_ - C
                else:
                    if aj_n < 0:
                        aj_n, ai_n = 0.0, s_
                if s_ > C and aj_n > C:
                    aj_n, ai_n = C, s_ - C
                elif s_ <= C and ai_n < 0:
                    ai_n, aj_n = 0.0, s_
            di, dj = ai_n - ai, aj_n - aj
            alpha[i], alpha[j] = ai_n, aj_n
            G += y * (K[:, i] * y[i] * di + K[:, j] * y[j] * dj)
        free = (alpha > 1e-12) & (alpha < C - 1e-12)
        yG = -y * G
        self.rho = float(np.mean(yG[free])) if free.any() else float((np.max(np.where((y > 0) & (alpha < C) | (y < 0) & (alpha > 0), yG, -np.inf)) + np.min(np.where((y > 0) & (alpha > 0) | (y < 0) & (alpha < C), yG, np.inf))) / 2.0)
        sv = alpha > 1e-12
        self.sv, self.coef = X[sv], (alpha * y)[sv]
        if not np.isfinite(self.rho):
            self.rho = 0.0
        return self

    def decision_function(self, Xq):
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        if len(self.sv) == 0:
            return np.full(len(Xq), -self.rho)
        return self._k(Xq, self.sv) @ self.coef + self.rho

    def predict(self, Xq):
        return np.where(self.decision_function(Xq) > 0, 1, -1)


class SVR:
    """Epsilon-insensitive support vector regression with an RBF kernel, solved by SMO (no extra packages).

    Options follow the usual defaults of regression SVMs: predictors are standardised, the box constraint is
    ``iqr(y)/1.349``, the tube half-width ``iqr(y)/13.49`` (``1`` and ``0.1`` when ``iqr(y) == 0``) and the kernel scale
    ``'auto'`` uses the median pairwise distance of the standardised predictors.  Kernel ``exp(-||x-z||^2 / scale^2)``."""

    def __init__(self, C=None, epsilon=None, kernel_scale="auto", standardize: bool = True, tol: float = 1e-3, max_iter: int = 200000,
                 kernel: str = "rbf", gamma=None, coef0: float = 0.0):
        """``kernel="sigmoid"``: ``tanh(gamma <x, z> + coef0)`` (LIBSVM ``-t 3``; ``gamma`` defaults to 1/n_features)."""
        self.C, self.epsilon, self.kernel_scale = C, epsilon, kernel_scale
        self.standardize, self.tol, self.max_iter = bool(standardize), float(tol), int(max_iter)
        self.kernel, self.gamma, self.coef0 = kernel, gamma, float(coef0)

    def _kern(self, A, B):
        if self.kernel == "sigmoid":
            return np.tanh(self.gamma_ * (A @ B.T) + self.coef0)
        return np.exp(-self._d2(A, B) / self.scale ** 2)

    @staticmethod
    def _d2(A, B):
        return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(X)
        if self.standardize:
            self.mu, self.sd = X.mean(axis=0), X.std(axis=0, ddof=1) if n > 1 else np.ones(X.shape[1])
            self.sd = np.where(self.sd > 0, self.sd, 1.0)
        else:
            self.mu, self.sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = (X - self.mu) / self.sd
        q1, q3 = np.percentile(y, [25, 75], method="hazen")           # the usual (n p + 1/2) percentile convention
        iqr = float(q3 - q1)
        C = float(self.C) if self.C is not None else (iqr / 1.349 if iqr > 0 else 1.0)
        eps = float(self.epsilon) if self.epsilon is not None else (iqr / 13.49 if iqr > 0 else 0.1)
        if self.kernel_scale == "auto":
            sub = Z if n <= 1000 else Z[np.linspace(0, n - 1, 1000).astype(int)]
            d = np.sqrt(self._d2(sub, sub))[np.triu_indices(len(sub), 1)]
            scale = float(np.median(d)) if d.size and np.median(d) > 0 else 1.0
        else:
            scale = float(self.kernel_scale)
        self.scale = scale
        self.gamma_ = float(self.gamma) if self.gamma is not None else 1.0 / max(1, Z.shape[1])
        K = self._kern(Z, Z)
        Kd = np.diag(K).copy()
        # dual on 2n variables: alpha = [alpha+, alpha-], sign s = [+1, -1], linear term p
        s = np.concatenate([np.ones(n), -np.ones(n)])
        G = np.concatenate([eps - y, eps + y])
        a = np.zeros(2 * n)
        for _ in range(self.max_iter):
            yG = -s * G
            up = ((s > 0) & (a < C)) | ((s < 0) & (a > 0))
            low = ((s > 0) & (a > 0)) | ((s < 0) & (a < C))
            if not up.any() or not low.any():
                break
            i = int(np.argmax(np.where(up, yG, -np.inf)))
            gmax = yG[i]
            gmin = np.min(np.where(low, yG, np.inf))
            if gmax - gmin < self.tol:
                break
            cand = np.where(low & (yG < gmax))[0]
            if len(cand) == 0:
                break
            bi, bc = i % n, cand % n
            quad = np.maximum(Kd[bi] + Kd[bc] - 2.0 * K[bi, bc], 1e-12)
            b = gmax - yG[cand]
            j = int(cand[np.argmin(-(b * b) / quad)])
            bj = j % n
            q = max(Kd[bi] + Kd[bj] - 2.0 * K[bi, bj], 1e-12)
            t = (s[j] * G[j] - s[i] * G[i]) / q
            # keep both variables inside [0, C] (alpha_i += s_i t, alpha_j -= s_j t)
            lo_t, hi_t = -np.inf, np.inf
            for idx, sg in ((i, s[i]), (j, -s[j])):
                l, h = (0.0 - a[idx]) * sg, (C - a[idx]) * sg
                if sg < 0:
                    l, h = h, l
                lo_t, hi_t = max(lo_t, min(l, h)), min(hi_t, max(l, h))
            t = min(max(t, lo_t), hi_t)
            if abs(t) < 1e-15:
                break
            a[i] += s[i] * t
            a[j] -= s[j] * t
            ci = np.concatenate([K[:, bi], K[:, bi]]) * s * s[i]
            cj = np.concatenate([K[:, bj], K[:, bj]]) * s * s[j]
            G += ci * (s[i] * t) - cj * (s[j] * t)
        free = (a > 1e-12) & (a < C - 1e-12)
        yG = -s * G
        if free.any():
            rho = float(np.mean(yG[free]))
        else:
            up = ((s > 0) & (a < C)) | ((s < 0) & (a > 0))
            low = ((s > 0) & (a > 0)) | ((s < 0) & (a < C))
            ub = np.max(np.where(up, yG, -np.inf)) if up.any() else 0.0
            lb = np.min(np.where(low, yG, np.inf)) if low.any() else 0.0
            rho = float((ub + lb) / 2.0)
        beta = a[:n] - a[n:]
        keep = np.abs(beta) > 1e-12
        self.sv, self.coef, self.bias = Z[keep], beta[keep], rho  # rho here is the intercept (mean of -s*G)
        return self

    def predict(self, Xq):
        Xq = (np.atleast_2d(np.asarray(Xq, dtype=float)) - self.mu) / self.sd
        if len(self.sv) == 0:
            return np.full(len(Xq), self.bias)
        return self._kern(Xq, self.sv) @ self.coef + self.bias
