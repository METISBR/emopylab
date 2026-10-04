# emopylab 2026
"""LCMEA (large-scale constrained multi-objective evolutionary algorithm).

Reference:
L. Si, X. Zhang, Y. Zhang, S. Yang, and Y. Tian. An efficient sampling approach to offspring
generation for evolutionary large-scale constrained multi-objective optimization. IEEE Transactions
on Emerging Topics in Computational Intelligence, 2025, 9(3): 2080-2092.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils import spea
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'LCMEA': {'constrained', 'large', 'multi', 'real'}}


def ada_fitness(F, cv=None, var=None):
    """SPEA2 fitness under constrained dominance on the total violation (violations <= ``var`` count as zero)."""
    C = None
    if cv is not None:
        c = np.maximum(0, np.asarray(cv, float).reshape(len(F), -1)).sum(1)
        if var is not None:
            c = np.where(c <= var, 0.0, c)
        C = c[:, None]
    return spea.cal_fitness(F, C)


def cal_sde(F):
    N = len(F)
    with np.errstate(all="ignore"):
        P = (F - F.min(0)) / (F.max(0) - F.min(0))
    S = np.maximum(P[None, :, :], P[:, None, :])
    d = np.sqrt(((P[:, None, :] - S) ** 2).sum(-1))
    k = int(np.floor(np.sqrt(N)))
    return 1.0 / (np.sort(d, 1)[:, k] + 2)


def _select(pop, fit, N):
    F = objs(pop)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[spea.truncation(F[idx], int(nxt.sum()) - N)]] = False
    idx = np.where(nxt)[0]
    return pop[idx[np.argsort(fit[idx], kind="stable")]]


def env_cdp(pop, N):
    return _select(pop, ada_fitness(objs(pop), cons(pop) if cons(pop).shape[1] else None), N)


def env_eps(pop, N, eps):
    return _select(pop, ada_fitness(objs(pop), cons(pop) if cons(pop).shape[1] else None, eps), N)


def env_mop(pop, N, eps):
    C = cons(pop)
    nc = C.shape[1]
    tot = np.maximum(0, C).sum(1) if nc else np.zeros(len(pop))
    ok = (tot <= eps).astype(int) == nc          # literal: summed violation compared with the constraint count
    if ok.sum() > N:
        pop = pop[ok]
        C = cons(pop)
        cv = np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(len(pop))
        fit = ada_fitness(np.column_stack([objs(pop), cv]))
    else:
        fit = ada_fitness(np.column_stack([cal_sde(objs(pop)), tot]))
    return _select(pop, fit, N)


class _BNTanhNet:
    """Batch-normalised tanh network of the reference evaluated in training mode (batch statistics): on a single state
    every normalised activation is zero, so the critic returns 0 and the actor returns a uniform distribution."""

    def __init__(self, n_in, n_hidden, n_out, rng, out_scale):
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2)
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(out_scale)
        self.b1, self.b2 = np.zeros(n_hidden), np.zeros(n_out)

    @staticmethod
    def _bn(x):
        return (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)

    def forward(self, X, softmax):
        h = np.tanh(self._bn(np.atleast_2d(X)))
        h = np.tanh(self._bn(h @ self.W1 + self.b1))
        z = self._bn(h @ self.W2 + self.b2)
        if softmax:
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)
        return np.tanh(z)


class PPOAgent:
    """PPO operator of the reference: the action is random for the first generation and with probability 0.2, otherwise
    sampled from the actor's output for the current state. Because the actor normalises a single state with batch
    statistics (see :class:`_BNTanhNet`) that output is always uniform, so the clipped-surrogate updates the reference
    performs every 20 steps cannot change any decision and are not repeated here; the transition buffers are kept."""

    def __init__(self, n_actions, n_states, rng):
        self.rng, self.nA = rng, n_actions
        self.actor = _BNTanhNet(n_states, 60, n_actions, rng, 0.01)
        self.critic = _BNTanhNet(n_states, 60, 1, rng, 1.0)
        self.gen = 1
        self.rewards, self.states, self.values = [], [], []

    def get_action(self, state, reward):
        self.rewards.append(reward)
        s = np.asarray(state, float)[None]
        self.states.append(s[0])
        self.values.append(float(self.critic.forward(s, False)[0, 0]))
        probs = self.actor.forward(s, True)[0]
        if self.gen == 1 or self.rng.random() < 0.2:
            a = int(self.rng.integers(0, self.nA))
        else:
            p = probs.copy()
            p[-1] = 1 - p[:-1].sum()
            a = int(self.rng.choice(self.nA, p=np.clip(p, 0, 1) / np.clip(p, 0, 1).sum()))
        self.gen += 1
        return a + 1


class _EnvRL:
    def __init__(self, archive, rng):
        self.ppo = PPOAgent(3, 10, rng)
        self._refresh(archive, initial=True)

    def _stats(self, A, eps_range=False):
        F, C = objs(A), cons(A)
        n = len(A)
        sc = (np.maximum(C, 0) / self.cmax).sum(1) if C.shape[1] else np.zeros(n)
        rng_ = np.maximum(self.zmax - self.zmin, np.finfo(float).eps) if eps_range else (self.zmax - self.zmin)
        with np.errstate(all="ignore"):
            so = ((F - self.zmin) / rng_).sum(1)
        return so, sc

    def _bounds(self, A):
        F, C = objs(A), cons(A)
        self.zmin, self.zmax = F.min(0), F.max(0)
        self.cmax = np.maximum(np.maximum(C, 0).max(0), 1e-6) if C.shape[1] else np.zeros(0)

    def _centres(self, so, sc):
        self.z1min, self.z1max, self.c1max = np.nanmin(so), np.nanmax(so), sc.max()
        cv = sc.min()
        self.center = np.array([np.nanmin(so[sc == cv]), cv])
        o, c = self.norm(so, sc)
        i1 = (o >= 0) & (c > 0)
        self.cI = np.array([o[i1].mean(), c[i1].mean()]) if i1.any() else np.array([1.0, 1.0])
        i2 = (o < 0) & (c > 0)
        self.cII = np.array([o[i2].mean(), c[i2].mean()]) if i2.any() else np.array([0.0, 0.0])
        i4 = (o > 0) & (c <= 0)
        self.cIV = np.array([o[i4].min(), c[i4].min()]) if i4.any() else np.array([1.0, 0.0])
        return o, c, i1, i2, i4

    def _refresh(self, A, initial=False):
        self._bounds(A)
        so, sc = self._stats(A, eps_range=not initial)
        return self._centres(so, sc)

    def norm(self, O, Cv):
        eps = np.finfo(float).eps
        SO, SC = O.copy(), Cv.copy()
        c0, c1 = self.center
        with np.errstate(all="ignore"):
            m = (O >= c0) & (Cv > c1)
            if m.sum() > 1:
                SO[m] = (O[m] - c0) / max(self.z1max - c0, eps)
                SC[m] = (Cv[m] - c1) / max(self.c1max - c1, eps)
            m = (O < c0) & (Cv > c1)
            if m.sum() > 1:
                SO[m] = (O[m] - self.z1min) / max(c0 - self.z1min, eps) - 1
                SC[m] = (Cv[m] - c1) / max(self.c1max - c1, eps)
            m = (O <= c0) & (Cv <= c1)
            m[(O == c0) & (Cv == c1)] = False
            if m.sum() > 1:
                SO[m] = (O[m] - self.z1min) / max(c0 - self.z1min, eps) - 1
                SC[m] = (Cv[m] + self.c1max) / max(c1 + self.c1max, eps) - 1
            m = (SO < c0) & (Cv <= c1)                     # literal: tests the already rescaled objective value
            if m.sum() > 1:
                SO[m] = (O[m] - c0) / max(self.z1max - c0, eps)
                SC[m] = (Cv[m] + self.c1max) / max(c1 + self.c1max, eps) - 1
        if np.any(SO == np.inf):
            SO[SO == np.inf] = 1
        elif np.any(SO == -np.inf):
            SO[SO == -np.inf] = 0
        r = (O == c0) & (Cv == c1)
        SO[r], SC[r] = 0, 0
        return SO, SC

    def state_reward(self, A):
        so, sc = self._stats(A)
        o, c = self.norm(so, sc)
        i1 = (o >= 0) & (c > 0)
        cI = np.array([o[i1].mean(), c[i1].mean()]) if i1.any() else np.array([1.0, 1.0])
        r1 = (self.cI[1] - cI[1]) + (self.cI[0] - cI[0])
        i2 = (o < 0) & (c > 0)
        cII = np.array([o[i2].mean(), c[i2].mean()]) if i2.any() else np.array([0.0, 0.0])
        r2 = self.cII[1] - cII[1]
        i3 = (o <= 0) & (c <= 0)
        i3[(o == 0) & (c == 0)] = False
        cIII = np.array([o[i3].min(), c[i3].min()]) if i3.any() else np.array([0.0, 0.0])
        r3 = -cIII[0] - cIII[1]
        i4 = (o > 0) & (c <= 0)
        cIV = np.array([o[i4].min(), c[i4].min()]) if i4.any() else np.array([1.0, 0.0])
        r4 = self.cIV[0] - cIV[0]
        reward = float(np.tanh(r1 + r2 + r3 + r4))
        o, c, i1, i2, i4 = self._refresh(A)
        n = len(A)
        with np.errstate(all="ignore"):
            rel = np.corrcoef(o, c)[0, 1]
        rel = 0.0 if np.isnan(rel) else rel
        state = [*self.cI, *self.cII, *self.cIV, i1.sum() / n, i2.sum() / n, i4.sum() / n, rel]
        return state, reward

    def form_pop(self, A, rng):
        so, sc = self._stats(A)
        o, c = self.norm(so, sc)
        i2 = (o < 0) & (c > 0)
        o = o.copy()
        o[i2] = -o[i2]
        fr, maxf = nd_sort(np.column_stack([o, c]), None, len(A))
        dl = np.where(fr == maxf)[0]
        total = len(fr)
        n = total // 2
        num = total - n
        if total >= 2 * num:
            cand = rng.permutation(total)[: 2 * num]
            a, b = cand[:num], cand[num:]
            sel = np.concatenate([a[fr[a] <= fr[b]], b[fr[b] < fr[a]]])
            rest = np.setdiff1d(np.arange(total), cand)
            idx = np.concatenate([sel, rest])
        else:
            cand = rng.permutation(total)[: 2 * n]
            a, b = cand[:n], cand[n:]
            idx = np.concatenate([a[fr[a] <= fr[b]], b[fr[b] < fr[a]]])
        if len(dl) == 1:
            idx[-1] = dl[0]
        return idx.astype(int)


class LCMEA(LoopAlgorithm):
    """An archive of 2N solutions; a population of N drawn from it by tournaments on a centre-normalised objective /
    violation space produces N offspring with an evolutionary sampling operator (lines between good and poor solutions,
    Gaussian spread, DE-like moves, perturbations) and is reselected by one of three environmental selections
    (constrained dominance, epsilon-relaxed dominance, or density/violation bi-objective) chosen by a PPO agent."""

    def _initialize_infill(self):
        return self.evaluate(self.random_decs(2 * self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills                      # archive
        self.env = _EnvRL(infills, self.rng)
        cv = np.maximum(0, cons(infills)).sum(1) if cons(infills).shape[1] else np.zeros(len(infills))
        self.var0 = cv.max() if cv.max() != 0 else 1.0
        self.x = 0.0
        self.var = self.var0 * (1 - self.x) ** ((-np.log(self.var0) - 6) / np.log(0.5))
        self.idx = self.env.form_pop(self.pop, self.rng)
        self.P = self.pop[self.idx]
        self.prev = None
        self._set_optimum()

    def _esp(self, P):
        rng, N, D = self.rng, self.N, self.D
        ref = 10
        sub = min(int(np.ceil(N / ref)), N)
        F, C = objs(P), cons(P)
        cvv = np.maximum(C, 0).sum(1) if C.shape[1] else np.zeros(len(P))
        fnum = int((cvv == 0).sum())
        fidx = np.where(cvv == 0)[0]
        fit = ada_fitness(F, cvv)
        order = np.argsort(fit, kind="stable")
        half = N // 2
        worse = lambda: order[half + rng.permutation(N - half)[:ref]]
        if rng.random() < np.exp(-ref * fnum / N):
            nd = np.where(fit <= 1)[0]
            S = order[:ref] if fnum < ref else fidx[rng.permutation(fnum)[:ref]]
            E = worse()
            R = nd if fnum > 1 else order[:ref]
        else:
            fr, _ = nd_sort(F, C if C.shape[1] else None, np.inf)
            nI, dI = np.where(fr == 1)[0], np.where(fr > 1)[0]
            nn, dn = len(nI), len(dI)
            if nn < 2 * ref or dn < ref:
                if nn < ref:
                    S, E = order[:ref], worse()
                elif dn < ref:
                    sel = rng.permutation(nn)[: 2 * ref - dn]
                    S, E = nI[sel[:ref]], np.concatenate([dI, nI[sel[ref:]]])
                else:
                    S, E = nI[rng.permutation(nn)[:ref]], dI[rng.permutation(dn)[:ref]]
            elif rng.random() < np.exp(-0.5 * ref * nn / N):
                sel = rng.permutation(nn)[: 2 * ref]
                S, E = nI[sel[:ref]], nI[sel[ref:]]
            else:
                S, E = nI[rng.permutation(nn)[:ref]], dI[rng.permutation(dn)[:ref]]
            R = nI if fnum > 1 else S
        X = decs(P)
        V = X[E] - X[S]
        with np.errstate(all="ignore"):
            Dr = V / np.sqrt((V ** 2).sum(1, keepdims=True))
        rows = []
        for i in range(ref):
            lam = (X[R] - X[S[i]]) @ Dr[i]
            beta = np.std(lam, ddof=1) if len(lam) > 1 else 0.0
            r = rng.normal(0, beta, sub) * (1 / (1 + ((N - fnum + 1) / N))) ** 2
            rows.append(r[:, None] * Dr[i] + X[S[i]])
        PD = np.nan_to_num(np.vstack(rows))[:N]
        BU, BD = X.max(0), X.min(0)
        delta = ((self.max_FE - self.FE + 1) / self.max_FE) ** 2
        delta = 0.0 if delta < 0.1 else delta
        p = int(np.ceil(0.1 * N))
        Pi = rng.permutation(N)
        best = order
        if rng.random() < 0.1:
            rn = N - 2 * p
            blk = Pi[rn:rn + p]
            PD[blk] = PD[blk] + rng.standard_normal((p, 1)) * PD[blk]
            blk2 = Pi[rn + p:]
            q = len(blk2)
        else:
            p = 2 * p
            rn = N - p
            blk2 = Pi[rn:]
            q = len(blk2)
        par = X[rng.permutation(N)[:q]]
        site = rng.random((q, D)) < rng.random((q, 1))
        bdec = np.repeat(X[[best[rng.integers(0, ref)]]], q, 0)
        P2 = PD[blk2]
        P2[site] = P2[site] + 0.5 * (bdec[site] - par[site])
        PD[blk2] = P2
        siteR = rng.random((rn, D)) < 2 * rng.random((rn, 1))
        PR = (BU - BD) / ref * (2 * rng.standard_normal((rn, 1))) * delta
        P1 = PD[Pi[:rn]]
        P1[siteR] = P1[siteR] + np.broadcast_to(PR, P1.shape)[siteR]
        PD[Pi[:rn]] = P1
        lo, up = self.lower, self.upper
        PD = np.clip(PD, lo, up)
        s, mu = rng.random((N, D)) < 1 / D, rng.random((N, D))
        L, U = np.broadcast_to(lo, PD.shape), np.broadcast_to(up, PD.shape)
        with np.errstate(all="ignore"):
            t = s & (mu <= 0.5)
            PD[t] = PD[t] + (U[t] - L[t]) * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (PD[t] - L[t]) / (U[t] - L[t])) ** 21) ** (1 / 21) - 1)
            t = s & (mu > 0.5)
            PD[t] = PD[t] + (U[t] - L[t]) * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (U[t] - PD[t]) / (U[t] - L[t])) ** 21) ** (1 / 21))
        return self.evaluate(PD)

    def step(self):
        N = self.N
        off = self._esp(self.P)
        state, reward = self.env.state_reward(self.pop)
        a = self.env.ppo.get_action(state, reward)
        merged = Population.merge(self.P, off)
        if a == 1:
            self.P = env_cdp(merged, N)
        elif a == 2:
            self.P = env_eps(merged, N, self.var)
        else:
            self.P = env_mop(merged, N, self.var)
        arc = self.pop
        sel = np.arange(len(arc))
        m = Population.merge(arc, self.P)
        sel[self.idx[: len(self.P)]] = len(arc) + np.arange(min(len(self.P), len(self.idx)))
        self.pop = m[sel]
        if a != self.prev:
            self.idx = self.env.form_pop(self.pop, self.rng)
            self.P = self.pop[self.idx]
        self.prev = a
        self.var = self.var0 * (1 - self.x) ** ((-np.log(self.var0) - 6) / np.log(0.5))
        if self.var < 1e-6:
            self.var = 0.0
        self.x += 1 / (self.max_FE / N)
