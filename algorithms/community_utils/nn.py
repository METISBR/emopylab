# emopylab 2026
"""Small neural components (restricted Boltzmann machine, denoising autoencoder) implemented with NumPy only.

Both keep the training procedure of the models used by the mask/decision-space learning operators of the sparse
large-scale algorithms: contrastive-divergence training for the RBM and single-hidden-layer back-propagation with
input masking for the autoencoder."""

from __future__ import annotations

import numpy as np

__all__ = ["RBM", "DAE"]


def _sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


class RBM:
    def __init__(self, n_visible, n_hidden, epoch, batch_size, penalty, momentum, learn_rate, rng):
        self.nv, self.nh, self.epoch, self.bs = int(n_visible), int(n_hidden), int(epoch), int(batch_size)
        self.penalty, self.momentum, self.lr, self.rng = penalty, momentum, learn_rate, rng
        self.W = 0.1 * rng.standard_normal((self.nv, self.nh))
        self.vb = np.zeros(self.nv)
        self.hb = np.zeros(self.nh)

    def train(self, X):
        X = np.asarray(X, dtype=float)
        rng, bs = self.rng, self.bs
        dW, dv, dh = np.zeros_like(self.W), np.zeros_like(self.vb), np.zeros_like(self.hb)
        for _ in range(self.epoch):
            if self.epoch > 5:
                self.momentum = 0.9
            kk = rng.permutation(len(X))
            for b in range(len(X) // bs):
                x = X[kk[b * bs:(b + 1) * bs]]
                ph = _sigmoid(x @ self.W + self.hb)
                hs = ph > rng.random((bs, self.nh))
                nd_p = _sigmoid(hs @ self.W.T + self.vb)
                nd = nd_p > rng.random((bs, self.nv))
                nh = _sigmoid(nd @ self.W + self.hb)
                pos, neg = x.T @ ph, nd_p.T @ nh
                dW = self.momentum * dW + self.lr * ((pos - neg) / bs - self.penalty * self.W)
                dv = self.momentum * dv + (self.lr / bs) * (x.sum(axis=0) - nd.sum(axis=0))
                dh = self.momentum * dh + (self.lr / bs) * (ph.sum(axis=0) - nh.sum(axis=0))
                self.W, self.vb, self.hb = self.W + dW, self.vb + dv, self.hb + dh

    def reduce(self, X):
        return _sigmoid(np.asarray(X, dtype=float) @ self.W + self.hb) > self.rng.random((len(X), self.nh))

    def recover(self, H):
        return _sigmoid(np.asarray(H, dtype=float) @ self.W.T + self.vb) > self.rng.random((len(H), self.nv))


class DAE:
    def __init__(self, n_visible, n_hidden, epoch, batch_size, masked_fraction, momentum, learn_rate, rng):
        self.nv, self.nh, self.epoch, self.bs = int(n_visible), int(n_hidden), int(epoch), int(batch_size)
        self.frac, self.momentum, self.lr, self.rng = masked_fraction, momentum, learn_rate, rng
        self.WA = (rng.random((self.nh, self.nv + 1)) - 0.5) * 8 * np.sqrt(6 / (self.nh + self.nv))
        self.WB = (rng.random((self.nv, self.nh + 1)) - 0.5) * 8 * np.sqrt(6 / (self.nv + self.nh))
        self.lower = self.upper = None

    def train(self, X):
        X = np.asarray(X, dtype=float)
        rng, bs = self.rng, self.bs
        self.lower, self.upper = X.min(axis=0), X.max(axis=0)
        with np.errstate(all="ignore"):
            X = (X - self.lower) / (self.upper - self.lower)
        vA, vB = np.zeros_like(self.WA), np.zeros_like(self.WB)
        theta = (rng.random(X.shape) > self.frac) if self.frac != 0 else np.ones(X.shape, bool)
        Xt = np.hstack([np.ones((len(X), 1)), X * theta])
        for _ in range(self.epoch):
            kk = rng.permutation(len(X))
            for b in range(len(X) // bs):
                idx = kk[b * bs:(b + 1) * bs]
                bx, by = Xt[idx], X[idx]
                h1 = np.hstack([np.ones((bs, 1)), _sigmoid(bx @ self.WA.T)])
                h2 = _sigmoid(h1 @ self.WB.T)
                d3 = -(by - h2) * (h2 * (1 - h2))
                d2 = (d3 @ self.WB) * (h1 * (1 - h1))
                gA = self.lr * (d2[:, 1:].T @ bx) / len(d2)
                gB = self.lr * (d3.T @ h1) / len(d3)
                if self.momentum > 0:
                    vA, vB = self.momentum * vA + gA, self.momentum * vB + gB
                    gA, gB = vA, vB
                self.WA, self.WB = self.WA - gA, self.WB - gB

    def reduce(self, X):
        with np.errstate(all="ignore"):
            X = (np.asarray(X, dtype=float) - self.lower) / (self.upper - self.lower)
        return _sigmoid(X @ self.WA[:, 1:].T + self.WA[:, 0])

    def recover(self, H):
        X = _sigmoid(np.asarray(H, dtype=float) @ self.WB[:, 1:].T + self.WB[:, 0])
        return X * (self.upper - self.lower) + self.lower


class LMNet:
    """Feed-forward regression network (tanh hidden layers, linear output) trained with the Levenberg-Marquardt rule.

    Mirrors the usual behaviour of a Levenberg-Marquardt trained network: inputs and targets are scaled to [-1, 1] with
    the ranges seen when the network is configured (constant inputs are dropped), weights start from the Nguyen-Widrow
    rule, and training splits the data 70/15/15 into training/validation/test parts, keeping the weights with the best
    validation error and stopping at ``epochs``, at the error ``goal``, after ``max_fail`` validation failures or when
    the damping factor overflows."""

    def __init__(self, rng, hidden=(10, 10, 10), mu=1e-3, mu_dec=0.1, mu_inc=10.0, mu_max=1e10, min_grad=1e-7, max_fail=6):
        self.rng, self.hidden = rng, tuple(hidden)
        self.mu0, self.mu_dec, self.mu_inc, self.mu_max, self.min_grad, self.max_fail = mu, mu_dec, mu_inc, mu_max, min_grad, max_fail
        self.theta = None

    # -- configuration ------------------------------------------------------------------------------------
    def configure(self, X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
        self.xmin, self.xmax = X.min(axis=0), X.max(axis=0)
        self.keep = (self.xmax - self.xmin) > 0
        self.ymin, self.ymax = Y.min(axis=0), Y.max(axis=0)
        self.sizes = (int(self.keep.sum()),) + self.hidden + (Y.shape[1],)
        self._init_weights()
        return self

    def _scale_x(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))[:, self.keep]
        span = (self.xmax - self.xmin)[self.keep]
        return 2 * (X - self.xmin[self.keep]) / span - 1

    def _scale_y(self, Y):
        span = np.where(self.ymax > self.ymin, self.ymax - self.ymin, 1.0)
        return np.where(self.ymax > self.ymin, 2 * (Y - self.ymin) / span - 1, Y - self.ymin)

    def _unscale_y(self, Z):
        span = np.where(self.ymax > self.ymin, self.ymax - self.ymin, 1.0)
        return np.where(self.ymax > self.ymin, (Z + 1) / 2 * span + self.ymin, Z + self.ymin)

    # -- parameters -----------------------------------------------------------------------------------------
    def _shapes(self):
        return [(self.sizes[i + 1], self.sizes[i]) for i in range(len(self.sizes) - 1)]

    def _init_weights(self):
        parts = []
        for li, (o, i) in enumerate(self._shapes()):
            if li < len(self.sizes) - 2 and i > 0:                  # Nguyen-Widrow for the tanh layers
                beta = 0.7 * o ** (1.0 / i)
                W = self.rng.uniform(-1, 1, (o, i))
                W = beta * W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
                b = beta * np.linspace(-1, 1, o) * np.sign(self.rng.uniform(-1, 1, o))
            else:
                W, b = self.rng.uniform(-0.5, 0.5, (o, i)), self.rng.uniform(-0.5, 0.5, o)
            parts += [W.ravel(), b]
        self.theta = np.concatenate(parts)

    def _unpack(self, theta):
        out, k = [], 0
        for o, i in self._shapes():
            W = theta[k: k + o * i].reshape(o, i)
            k += o * i
            b = theta[k: k + o]
            k += o
            out.append((W, b))
        return out

    def _forward(self, theta, X):
        acts = [X]
        layers = self._unpack(theta)
        for li, (W, b) in enumerate(layers):
            z = acts[-1] @ W.T + b
            acts.append(z if li == len(layers) - 1 else np.tanh(z))
        return acts

    def _jacobian(self, theta, X):
        """Jacobian of the outputs w.r.t. the parameters; rows ordered sample-major / output-minor (the order of
        ``(out - Y).reshape(-1)``)."""
        n_out = self.sizes[-1]
        if n_out == 1:
            return self._jacobian_k(theta, X, None)
        parts, out = [], None
        for k in range(n_out):
            Jk, out = self._jacobian_k(theta, X, k)
            parts.append(Jk)
        return np.stack(parts, axis=1).reshape(len(X) * n_out, -1), out

    def _jacobian_k(self, theta, X, k):
        layers = self._unpack(theta)
        acts = self._forward(theta, X)
        n = len(X)
        J = []
        if k is None:
            g = np.ones((n, 1))                                    # d out / d z_last
        else:
            g = np.zeros((n, self.sizes[-1]))
            g[:, k] = 1.0
        for li in range(len(layers) - 1, -1, -1):
            W, _ = layers[li]
            A = acts[li]
            J.append((g[:, :, None] * A[:, None, :]).reshape(n, -1))
            J.append(g)
            if li > 0:
                g = (g @ W) * (1 - acts[li] ** 2)
        blocks = []
        for li in range(len(layers)):
            blocks += [J[2 * (len(layers) - 1 - li)], J[2 * (len(layers) - 1 - li) + 1]]
        return np.hstack(blocks), acts[-1]

    # -- public API -------------------------------------------------------------------------------------------
    def sim(self, X):
        if self.theta is None:
            raise RuntimeError("network not configured")
        return self._unscale_y(self._forward(self.theta, self._scale_x(X))[-1])

    def train(self, X, Y, epochs=100, goal=1e-3):
        X = self._scale_x(X)
        Y = self._scale_y(np.asarray(Y, dtype=float).reshape(len(X), -1))
        n = len(X)
        perm = self.rng.permutation(n)
        n_val, n_test = int(round(0.15 * n)), int(round(0.15 * n))
        tr, va = perm[: max(n - n_val - n_test, 1)], perm[max(n - n_val - n_test, 1): n - n_test]
        theta, mu = self.theta.copy(), self.mu0
        Xt, Yt = X[tr], Y[tr]

        def mse(th, XX, YY):
            return float(np.mean((self._forward(th, XX)[-1] - YY) ** 2)) if len(XX) else np.inf

        perf = mse(theta, Xt, Yt)
        best_val, best_theta, fails = (mse(theta, X[va], Y[va]) if len(va) else np.inf), theta.copy(), 0
        for _ in range(epochs):
            if perf <= goal:
                break
            J, out = self._jacobian(theta, Xt)
            e = (out - Yt).reshape(-1)
            JtJ, Jte = J.T @ J, J.T @ e
            if np.linalg.norm(Jte) / len(e) < self.min_grad:
                break
            improved = False
            while mu <= self.mu_max:
                try:
                    step = np.linalg.solve(JtJ + mu * np.eye(len(theta)), Jte)
                except np.linalg.LinAlgError:
                    mu *= self.mu_inc
                    continue
                cand = theta - step
                p2 = mse(cand, Xt, Yt)
                if p2 < perf:
                    theta, perf, improved = cand, p2, True
                    mu = max(mu * self.mu_dec, 1e-20)
                    break
                mu *= self.mu_inc
            if not improved:
                break
            if len(va):
                v = mse(theta, X[va], Y[va])
                if v < best_val:
                    best_val, best_theta, fails = v, theta.copy(), 0
                else:
                    fails += 1
                    if fails >= self.max_fail:
                        break
        self.theta = best_theta if len(va) else theta
        return self


class BNClassifierNet:
    """Feature z-score -> dense(h) -> batch normalisation -> ReLU -> dense(1) -> sigmoid, trained with Adam on half-MSE
    (mini-batches, reshuffled every epoch, incomplete last batch dropped); inference uses full-training-set BN statistics."""

    def __init__(self, n_in, n_hidden, rng, lr=1e-3, epochs=100, batch=32, beta1=0.9, beta2=0.999, eps=1e-8, bn_eps=1e-5):
        self.rng, self.lr, self.epochs, self.batch = rng, lr, int(epochs), int(batch)
        self.b1, self.b2, self.eps, self.bn_eps = beta1, beta2, eps, bn_eps
        g1, g2 = np.sqrt(6.0 / (n_in + n_hidden)), np.sqrt(6.0 / (n_hidden + 1))
        self.p = {"W1": rng.uniform(-g1, g1, (n_in, n_hidden)), "b1": np.zeros(n_hidden),
                  "gam": np.ones(n_hidden), "bet": np.zeros(n_hidden),
                  "W2": rng.uniform(-g2, g2, (n_hidden, 1)), "b2": np.zeros(1)}

    def _forward(self, X, mu=None, var=None):
        p = self.p
        Z = (X - self.xm) / self.xs
        H = Z @ p["W1"] + p["b1"]
        if mu is None:
            mu, var = H.mean(axis=0), H.var(axis=0)
        Hn = (H - mu) / np.sqrt(var + self.bn_eps)
        A = np.maximum(0.0, Hn * p["gam"] + p["bet"])
        O = 1.0 / (1.0 + np.exp(-(A @ p["W2"] + p["b2"])))
        return O[:, 0], (Z, H, Hn, A, mu, var)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.xm, self.xs = X.mean(axis=0), X.std(axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
        self.xs = np.where(self.xs == 0, 1.0, self.xs)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        t, n = 0, len(X)
        bs = min(self.batch, n)
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n - bs + 1, bs):
                idx = perm[s:s + bs]
                O, (Z, H, Hn, A, mu, var) = self._forward(X[idx])
                B = len(idx)
                dO = (O - y[idx]) / B
                dL = (dO * O * (1 - O))[:, None]
                g = {"W2": A.T @ dL, "b2": dL.sum(axis=0)}
                dA = dL @ self.p["W2"].T
                dA = dA * ((Hn * self.p["gam"] + self.p["bet"]) > 0)
                g["gam"], g["bet"] = (dA * Hn).sum(axis=0), dA.sum(axis=0)
                dHn = dA * self.p["gam"]
                inv = 1.0 / np.sqrt(var + self.bn_eps)
                dH = inv / B * (B * dHn - dHn.sum(axis=0) - Hn * (dHn * Hn).sum(axis=0))
                g["W1"], g["b1"] = Z.T @ dH, dH.sum(axis=0)
                t += 1
                for k in self.p:
                    m[k] = self.b1 * m[k] + (1 - self.b1) * g[k]
                    v2[k] = self.b2 * v2[k] + (1 - self.b2) * g[k] ** 2
                    self.p[k] -= self.lr * (m[k] / (1 - self.b1 ** t)) / (np.sqrt(v2[k] / (1 - self.b2 ** t)) + self.eps)
        H = ((X - self.xm) / self.xs) @ self.p["W1"] + self.p["b1"]
        self.mu, self.var = H.mean(axis=0), H.var(axis=0)
        return self

    def predict(self, X):
        return self._forward(np.asarray(X, float), self.mu, self.var)[0]


def minmax_fit(X):
    """Column-wise map to [-1, 1] (constant columns map to -1); returns (xmin, gain)."""
    X = np.asarray(X, float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    with np.errstate(divide="ignore"):
        gain = 2.0 / (hi - lo)
    gain[~np.isfinite(gain)] = 1.0
    return lo, gain


def minmax_apply(X, st):
    return (np.asarray(X, float) - st[0]) * st[1] - 1.0


class PatternNet:
    """Feed-forward classifier: tanh hidden layers, softmax output, mean cross-entropy loss, trained by scaled conjugate
    gradient (Moller) on a random 70/15/15 train/validation/test split with validation early stopping (6 failures)."""

    def __init__(self, hidden, rng, epochs=1000, max_fail=6, sigma=5e-5, lam=5e-7, min_grad=1e-6):
        self.hidden, self.rng = [int(h) for h in hidden], rng
        self.epochs, self.max_fail, self.sigma0, self.lam0, self.min_grad = int(epochs), int(max_fail), sigma, lam, min_grad

    def _unpack(self, w):
        out, i = [], 0
        for a, b in self.shapes:
            W = w[i:i + a * b].reshape(a, b); i += a * b
            c = w[i:i + b]; i += b
            out.append((W, c))
        return out

    def _forward(self, w, X):
        acts = [X]
        L = self._unpack(w)
        for W, c in L[:-1]:
            acts.append(np.tanh(acts[-1] @ W + c))
        Z = acts[-1] @ L[-1][0] + L[-1][1]
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True), acts, L

    def _loss_grad(self, w, X, T):
        P, acts, L = self._forward(w, X)
        n = X.size and len(X)
        loss = -np.sum(T * np.log(np.maximum(P, 1e-300))) / T.size
        d = (P - T) / T.size
        grads = []
        for li in range(len(L) - 1, -1, -1):
            grads.append((acts[li].T @ d, d.sum(axis=0)))
            if li:
                d = (d @ L[li][0].T) * (1 - acts[li] ** 2)
        g = np.concatenate([np.concatenate([G.ravel(), c]) for G, c in grads[::-1]])
        return loss, g

    def fit(self, X, T):
        X, T = np.asarray(X, float), np.asarray(T, float)
        sizes = [X.shape[1]] + self.hidden + [T.shape[1]]
        self.shapes = list(zip(sizes[:-1], sizes[1:]))
        w = np.concatenate([np.concatenate([self.rng.uniform(-1, 1, a * b) * np.sqrt(6.0 / (a + b)), self.rng.uniform(-1, 1, b) * 0.1])
                            for a, b in self.shapes])
        n = len(X)
        perm = self.rng.permutation(n)
        ntr, nva = int(round(0.7 * n)), int(round(0.15 * n))
        tr, va = perm[:ntr], perm[ntr:ntr + nva]
        Xt, Tt = X[tr], T[tr]
        f, g = self._loss_grad(w, Xt, Tt)
        r = -g
        p = r.copy()
        lam, lam_bar, success = self.lam0, 0.0, True
        best_w, best_v, fails = w.copy(), np.inf, 0
        nw = len(w)
        for k in range(self.epochs):
            if success:
                pp = p @ p
                if pp == 0:
                    break
                sig = self.sigma0 / np.sqrt(pp)
                _, g2 = self._loss_grad(w + sig * p, Xt, Tt)
                s = (g2 - g) / sig
                delta = p @ s
            delta = delta + (lam - lam_bar) * pp
            if delta <= 0:
                lam_bar = 2 * (lam - delta / pp)
                delta = -delta + lam * pp
                lam = lam_bar
            mu = p @ r
            alpha = mu / delta
            wn = w + alpha * p
            fn, gn = self._loss_grad(wn, Xt, Tt)
            Delta = 2 * delta * (f - fn) / (mu * mu) if mu else 0.0
            if Delta >= 0:
                w, f, g = wn, fn, gn
                rn = -g
                lam_bar, success = 0.0, True
                if (k + 1) % nw == 0:
                    p = rn
                else:
                    beta = (rn @ rn - rn @ r) / mu
                    p = rn + beta * p
                r = rn
                if Delta >= 0.75:
                    lam *= 0.25
            else:
                lam_bar, success = lam, False
            if Delta < 0.25:
                lam += delta * (1 - Delta) / pp
            if len(va):
                v, _ = self._loss_grad(w, X[va], T[va])
                if v < best_v:
                    best_v, best_w, fails = v, w.copy(), 0
                else:
                    fails += 1
                    if fails >= self.max_fail:
                        break
            else:
                best_w = w
            if np.linalg.norm(g) < self.min_grad:
                break
        self.w = best_w
        return self

    def predict_proba(self, X):
        return self._forward(self.w, np.asarray(X, float))[0]
