"""Dropout MLP (dropout 0.2 -> ReLU(40) -> dropout 0.5 -> tanh(40) -> linear) trained by plain SGD with weight decay,
predicting with dropout active (Monte-Carlo dropout); used by the dropout-network surrogate/DRL ports."""

from __future__ import annotations

import numpy as np

__all__ = ["DropoutNet", "minmax_fit", "minmax_apply", "minmax_reverse"]


def minmax_fit(X):
    """Column-wise map to [-1, 1] (constant columns -> -1); returns (xmin, gain)."""
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    with np.errstate(divide="ignore"):
        g = 2.0 / (hi - lo)
    g[~np.isfinite(g)] = 1.0
    return lo, g


def minmax_apply(X, st):
    return (np.asarray(X, float) - st[0]) * st[1] - 1.0


def minmax_reverse(Y, st):
    return (np.asarray(Y, float) + 1.0) / st[1] + st[0]


class DropoutNet:
    def __init__(self, n_in, n_out, rng, neurons=40, drop=(0.2, 0.5), learn_rate=0.01, decay=1e-5, bias1=0.1):
        self.rng, self.drop, self.lr, self.decay = rng, drop, float(learn_rate), float(decay)
        n = neurons
        g = rng.standard_normal
        self.W = [g((n_in, n)) * np.sqrt(1 / n_in), g((n, n)) * np.sqrt(1 / n), g((n, n_out)) * np.sqrt(1 / n)]
        self.B = [np.full(n, bias1) * np.sqrt(1 / n), g(n) * np.sqrt(1 / n), g(n_out) * np.sqrt(1 / n_out)]

    def _dropout(self, x, p):
        keep = self.rng.random(x.shape) >= p
        return np.where(keep, x, 0.0) / (1 - p), keep

    def forward(self, x):
        x1, _ = self._dropout(x, self.drop[0])
        x3 = np.maximum(0, x1 @ self.W[0] + self.B[0])
        x4, keep = self._dropout(x3, self.drop[1])
        x6 = np.tanh(x4 @ self.W[1] + self.B[1])
        return x6 @ self.W[2] + self.B[2], (x1, x3, x4, keep, x6)

    predict = lambda self, x: self.forward(np.atleast_2d(np.asarray(x, float)))[0]

    def sgd_step(self, x, y):
        N = len(x)
        out, (x1, x3, x4, keep, x6) = self.forward(x)
        e = out - y                                          # implicit expansion when y has fewer columns
        dW = [None] * 3
        dB = [None] * 3
        dW[2], dB[2] = x6.T @ e, e.sum(0)
        dx5 = (e @ self.W[2].T) * (1 - x6 ** 2)
        dW[1], dB[1] = x4.T @ dx5, dx5.sum(0)
        dx4 = dx5 @ self.W[1].T
        dx3 = np.where(keep, dx4 / (1 - self.drop[1]), 0.0)
        dx3[x3 <= 0] = 0
        dW[0], dB[0] = x1.T @ dx3, dx3.sum(0)
        for k in range(3):
            self.W[k] = self.W[k] - (self.decay * self.W[k] + dW[k]) / N * self.lr
            self.B[k] = self.B[k] - dB[k] / N * self.lr

    def train(self, X, Y, rounds, batch=None):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        N = len(X)
        bs = X.shape[1] if batch is None else int(batch)
        idx = np.round(self.rng.random(rounds * bs) * (N - 1)).astype(int)
        for j in range(rounds):
            b = idx[j * bs:(j + 1) * bs]
            self.sgd_step(X[b], Y[b])
        return self


def mc_estimate(net, X, ps, qs, n_pass=100):
    """Monte-Carlo dropout prediction: mean and standard deviation over ``n_pass`` stochastic forward passes."""
    x = minmax_apply(X, ps)
    Y = np.stack([minmax_reverse(net.predict(x), qs) for _ in range(n_pass)])
    mu = Y.mean(0)
    s2 = (Y ** 2).mean(0)
    return mu, np.sqrt(np.maximum(s2 - mu ** 2, 0.0))
