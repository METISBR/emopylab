"""Small deep Q-network (fully connected ReLU layers, Glorot-uniform weights, zero biases) with replay memory, a target
network refreshed every few learning steps, and plain SGD on the half mean squared TD error."""

from __future__ import annotations

import numpy as np

__all__ = ["MLP", "DQN"]


class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for a, c in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6.0 / (a + c))
            self.W.append(rng.uniform(-lim, lim, (a, c)))
            self.b.append(np.zeros(c))

    def copy_from(self, o):
        self.W = [w.copy() for w in o.W]
        self.b = [b.copy() for b in o.b]

    def forward(self, X):
        acts = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            Z = acts[-1] @ W + b
            acts.append(np.maximum(0, Z) if k < len(self.W) - 1 else Z)
        return acts

    def predict(self, X):
        return self.forward(np.atleast_2d(X))[-1]

    def sgd(self, X, dOut, lr):
        acts = self.forward(X)
        d = dOut
        for k in range(len(self.W) - 1, -1, -1):
            gW, gb = acts[k].T @ d, d.sum(0)
            if k:
                d = (d @ self.W[k].T) * (acts[k] > 0)
            self.W[k] -= lr * gW
            self.b[k] -= lr * gb


class DQN:
    def __init__(self, n_states, n_actions, rng, hidden=(128, 256, 128, 64, 32), capacity=512, batch=16, lr=0.01,
                 gamma=0.95, target_every=7):
        self.rng, self.nA = rng, int(n_actions)
        sizes = [n_states, *hidden, n_actions]
        self.net, self.target = MLP(sizes, rng), MLP(sizes, rng)
        self.target.copy_from(self.net)
        self.cap, self.batch, self.lr, self.gamma, self.every = capacity, batch, lr, gamma, target_every
        self.S = np.zeros((capacity, n_states))
        self.A = np.zeros(capacity, dtype=int)
        self.R = np.zeros(capacity)
        self.S2 = np.zeros((capacity, n_states))
        self.counter = 0
        self.learn_steps = 0

    def store(self, s, a, r, s2):
        i = self.counter % self.cap
        self.S[i], self.A[i], self.R[i], self.S2[i] = s, a, r, s2
        self.counter += 1

    def learn(self):
        self.learn_steps += 1
        if self.learn_steps % self.every == 0:
            self.target.copy_from(self.net)
        idx = self.rng.integers(0, min(self.cap, self.counter), self.batch)
        S, A, R, S2 = self.S[idx], self.A[idx], self.R[idx], self.S2[idx]
        q = self.net.predict(S)
        qa = q[np.arange(self.batch), A]
        tgt = R + self.gamma * self.target.predict(S2).max(1)
        d = np.zeros_like(q)
        d[np.arange(self.batch), A] = (qa - tgt) / self.batch           # d/dq of 1/(2B) sum (q - t)^2
        self.net.sgd(S, d, self.lr)


class TanhMLP:
    """ReLU hidden layers and a tanh output (Glorot-uniform weights, zero biases) trained with Adam; ``adam_t`` is the
    iteration number fed to the bias correction (the reference passes the epoch number)."""

    def __init__(self, sizes, rng, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.W, self.b = [], []
        for a, c in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6.0 / (a + c))
            self.W.append(rng.uniform(-lim, lim, (a, c)))
            self.b.append(np.zeros(c))
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.reset_adam()

    def reset_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        acts = [np.atleast_2d(X)]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            Z = acts[-1] @ W + b
            acts.append(np.maximum(0, Z) if k < len(self.W) - 1 else np.tanh(Z))
        return acts

    def predict(self, X):
        return self.forward(X)[-1]

    def backward(self, acts, dOut):
        """Gradients of a loss with d(loss)/d(output) = ``dOut``; returns (grads_W, grads_b, d(loss)/d(input))."""
        d = dOut * (1 - acts[-1] ** 2)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k], gb[k] = acts[k].T @ d, d.sum(0)
            d = d @ self.W[k].T
            if k:
                d = d * (acts[k] > 0)
        return gW, gb, d

    def adam(self, gW, gb, t):
        c1, c2 = 1 - self.b1 ** t, 1 - self.b2 ** t
        for k in range(len(self.W)):
            for P, G, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[k] = self.b1 * m[k] + (1 - self.b1) * G[k]
                v[k] = self.b2 * v[k] + (1 - self.b2) * G[k] ** 2
                P[k] = P[k] - self.lr * (m[k] / c1) / (np.sqrt(v[k] / c2) + self.eps)

    def fit_regression(self, X, Y, epochs=100, batch=60, half_mse=True):
        """Supervised training; loss = half mean squared error (``half_mse``) or the plain sum of squared errors."""
        self.reset_adam()
        X, Y = np.asarray(X, float), np.asarray(Y, float).reshape(len(X), -1)
        for ep in range(1, epochs + 1):
            for s in range(0, len(X), batch):
                x, y = X[s:s + batch], Y[s:s + batch]
                acts = self.forward(x)
                e = acts[-1] - y
                dOut = e / len(x) if half_mse else 2 * e
                gW, gb, _ = self.backward(acts, dOut)
                self.adam(gW, gb, ep)
        return self
