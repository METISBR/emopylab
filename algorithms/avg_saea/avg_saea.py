# emopylab 2026
"""AVG-SAEA (adaptive variable grouping based surrogate-assisted evolutionary algorithm).

Reference:
Y. Li, X. Feng, and H. Yu. Solving high-dimensional expensive multiobjective optimization problems
by adaptive decision variable grouping. IEEE Transactions on Evolutionary Computation, 2025, 29(4):
1041-1054.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, nd_sort, objs, tournament
from algorithms.community_utils.surrogates import RBFExact
from core.population import Population

ALGORITHM_FLAGS = {'AVGSAEA': {'expensive', 'integer', 'large', 'multi', 'real'}}


def _strength(F):
    lt = (F[:, None] < F[None]).any(-1)
    gt = (F[:, None] > F[None]).any(-1)
    dom = lt & ~gt
    S = dom.sum(1)
    return np.array([S[dom[:, i]].sum() for i in range(len(F))], float)


def cal_con(F):
    front, _ = nd_sort(F, None, np.inf)
    s = F.sum(1)
    return front * (s.max() - s.min()) + s


def _cos_sim(F):
    with np.errstate(all="ignore"):
        n = np.linalg.norm(F, axis=1)
        c = (F @ F.T) / (n[:, None] * n[None])
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 0.0)
    return c


def _cos_truncation(F, K, rng):
    with np.errstate(all="ignore"):
        F = (F - F.min(0)) / (F.max(0) - F.min(0))
    cos = _cos_sim(F)
    ch = np.zeros(len(F), bool)
    ch[np.argmax(np.where(np.isnan(F), -np.inf, F), axis=0)] = True
    if ch.sum() > K:
        sel = np.where(ch)[0]
        out = np.zeros(len(F), bool)
        out[sel[rng.permutation(len(sel))[:K]]] = True
        return out
    while ch.sum() < K:
        un = np.where(~ch)[0]
        ch[un[int(np.argmin(cos[np.ix_(un, np.where(ch)[0])].max(1)))]] = True
    return ch


def _mc_hv(F, rng, n=10000):
    ref = F.max(0) * 1.1
    F = F[~np.any(F > ref, axis=1)]
    lo = F.min(0)
    S = lo + rng.random((n, F.shape[1])) * (ref - lo)
    dom = np.zeros(n, bool)
    for f in F:
        dom |= np.all(f <= S, axis=1)
    return np.prod(ref - lo) * dom.sum() / n


class _MinMax01:
    def __init__(self, X):
        self.lo = X.min(0)
        with np.errstate(divide="ignore"):
            g = 1.0 / (X.max(0) - self.lo)
        g[~np.isfinite(g)] = 1.0
        self.g = g

    def apply(self, X):
        return (X - self.lo) * self.g

    def reverse(self, Y):
        return Y / self.g + self.lo


class AVGSAEA(LoopAlgorithm):
    """Variables are ranked by how much their means differ between non-dominated and worst-converged evaluated solutions and
    split into ``n_esp`` groups; one exact RBF network per group and objective predicts from that group's variables only. Each
    group evolves its own sub-population on the surrogates; the evaluated solutions are assembled group-wise, with the
    assembly rule (best, most diverse, or most disputed among group models) switched by hypervolume/convergence feedback."""

    def __init__(self, pop_size: int = 100, num_train: int = 300, wmax: int = 20, n_esp: int = 5, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.num_train, self.wmax, self.n_esp, self.mu = int(num_train), int(wmax), int(n_esp), int(mu)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        P, _ = UniformPoint(self.num_train, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.tX, self.tF = decs(infills), objs(infills)
        self.flag = 1
        self._set_optimum()

    def _operator(self, P, grp, g):
        n = len(P) // 2
        A, B = P[:n], P[n:2 * n]
        rng, d = self.rng, P.shape[1]
        mu = rng.random((n, d))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (-1.0) ** rng.integers(0, 2, (n, d))
        beta[rng.random((n, d)) < 0.5] = 1
        off = np.vstack([(A + B) / 2 + beta * (A - B) / 2, (A + B) / 2 - beta * (A - B) / 2])
        lo, up = self.lower[grp == g], self.upper[grp == g]
        N2 = len(off)
        site = rng.random((N2, d)) < 1.0 / d
        mu = rng.random((N2, d))
        L, U = np.broadcast_to(lo, off.shape), np.broadcast_to(up, off.shape)
        off = np.minimum(np.maximum(off, L), U)
        with np.errstate(all="ignore"):
            t = site & (mu <= 0.5)
            off[t] = off[t] + (U[t] - L[t]) * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - L[t]) / (U[t] - L[t])) ** 21) ** (1 / 21) - 1)
            t = site & (mu > 0.5)
            off[t] = off[t] + (U[t] - L[t]) * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (U[t] - off[t]) / (U[t] - L[t])) ** 21) ** (1 / 21))
        return off

    def _dv_select(self, F, X, N):
        front, maxf = nd_sort(F, None, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        nxt[last[_cos_truncation(F[last], N - int(nxt.sum()), self.rng)]] = True
        return F[nxt], X[nxt]

    def step(self):
        rng, D, M, N, E = self.rng, self.D, self.M, self.N, self.n_esp
        arc = self.pop
        AF, AX = objs(arc), decs(arc)
        conv = _strength(AF)
        better, bad = AX[conv == 0], AX[np.argsort(conv, kind="stable")[len(conv) - len(arc) // 4:]]
        diff = np.abs(better.mean(0) - bad.mean(0))
        order = np.argsort(-diff, kind="stable")
        per = D // E
        nor = np.concatenate([np.repeat(np.arange(1, E), per), np.full(D - per * (E - 1), E)])
        grp = np.empty(D, int)
        grp[order] = nor[:D]
        models = {}
        for i in range(1, E + 1):
            Xi = self.tX[:, grp == i]
            for m in range(M):
                ps, qs = _MinMax01(Xi), _MinMax01(self.tF[:, [m]])
                models[i, m] = (ps, qs, RBFExact(spread=1.0).fit(ps.apply(Xi), qs.apply(self.tF[:, [m]])))
        # survivors of NSGA-II selection seed every group
        front, maxf = nd_sort(AF, None, N)
        nxt = front < maxf
        cd = crowding(AF, front)
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
        PX, PF = AX[nxt], AF[nxt]
        sub = {i: PX[:, grp == i] for i in range(1, E + 1)}
        pobj = {i: PF.copy() for i in range(1, E + 1)}

        def predict(i, X):
            out = np.zeros((len(X), M))
            for m in range(M):
                ps, qs, net = models[i, m]
                out[:, m] = qs.reverse(np.asarray(net.predict(ps.apply(X))).reshape(-1, 1))[:, 0]
            return out

        for _ in range(self.wmax):
            for i in range(1, E + 1):
                con = cal_con(pobj[i])
                mate = tournament(2, int(np.ceil(N / 2)) * 2, con, rng=rng)
                od = self._operator(sub[i][mate], grp, i)
                of = predict(i, od)
                n = len(od)
                if self.flag == 2:
                    pobj[i], sub[i] = self._dv_select(np.vstack([pobj[i], of]), np.vstack([sub[i], od]), n)
                else:
                    allc = cal_con(np.vstack([pobj[i], of]))
                    upd = allc[:n] > allc[n:]
                    sub[i][upd], pobj[i][upd] = od[upd], of[upd]
        new = np.zeros((self.mu, D))
        if self.flag == 1:
            for i in range(1, E + 1):
                new[:, grp == i] = sub[i][np.argsort(cal_con(pobj[i]), kind="stable")[: self.mu]]
        elif self.flag == 2:
            for i in range(1, E + 1):
                ch = _cos_truncation(pobj[i], self.mu, rng)
                new[:, grp == i] = sub[i][ch]
        else:
            merged = np.zeros((N, D))
            for i in range(1, E + 1):
                merged[:, grp == i] = sub[i]
            sd = np.column_stack([np.std(np.column_stack([pobj[i][:, m] for i in range(1, E + 1)]), axis=1, ddof=1) for m in range(M)])
            new = merged[np.argsort(-sd.mean(1), kind="stable")[: self.mu]]
        last_pop = arc
        self.pop = Population.merge(arc, self.evaluate(new))
        self.flag = self._cal_flag(self.pop, last_pop)
        self.tX, self.tF = self._train_data(self.pop, self.num_train, len(new))

    def _cal_flag(self, pop, last):
        F, LF = objs(pop), objs(last)
        f1, _ = nd_sort(F, None, np.inf)
        f0, _ = nd_sort(LF, None, np.inf)
        nd, lnd = F[f1 == 1], LF[f0 == 1]
        lo, hi = lnd.min(0), lnd.max(0)
        with np.errstate(all="ignore"):
            lcon = ((lnd - lo) / (hi - lo)).sum(1)
            con = ((nd - lo) / (hi - lo)).sum(1)
        if _mc_hv(lnd, self.rng) < _mc_hv(nd, self.rng):
            return 1 if con.min() < lcon.min() else 2
        return 0

    @staticmethod
    def _train_data(pop, N1, N2):
        F = objs(pop)
        NA = len(F)
        F = F - F.min(0)
        cos = _cos_sim(F)
        ch = np.zeros(NA, bool)
        ch[NA - N2:] = True
        while ch.sum() < min(N1, NA):
            un = np.where(~ch)[0]
            ch[un[int(np.argmin(cos[np.ix_(un, np.where(ch)[0])].max(1)))]] = True
        return decs(pop)[ch], F[ch]
