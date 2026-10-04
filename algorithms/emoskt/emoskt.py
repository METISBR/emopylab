# emopylab 2026
"""EMOSKT (evolutionary multi-objective optimization with sparsity knowledge transfer).

Reference:
C. Wu, Y. Tian, L. Zhang, X. Xiang, and X. Zhang. A sparsity knowledge transfer-based evolutionary
algorithm for large-scale multitasking multi- objective optimization. IEEE Transactions on
Evolutionary Computation, 2025, 29(6): 2582-2595.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, nd_sort, objs, tournament
from algorithms.community_utils.sparse_mask import ts
from core.population import Population

ALGORITHM_FLAGS = {'EMOSKT': {'binary', 'constrained', 'large', 'multi', 'multitask', 'real', 'sparse'}}


def _nd_obj_cons(pop):
    F, C = objs(pop), cons(pop)
    return nd_sort(np.hstack([F, C]) if C.shape[1] else F, None, np.inf)[0]


def _sparse_select(pop, dec, mask, N, unique=True):
    if unique:
        u = np.unique(objs(pop), axis=0, return_index=True)[1]
        pop, dec, mask = pop[u], dec[u], mask[u]
    N = min(N, len(pop))
    F, C = objs(pop), cons(pop)
    front, maxf = nd_sort(F, C if C.shape[1] else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    nxt[last[np.argsort(-cd[last], kind="stable")[: N - int(nxt.sum())]]] = True
    return pop[nxt], dec[nxt], mask[nxt], front[nxt], cd[nxt]


class EMOSKT(LoopAlgorithm):
    """One SparseEA subpopulation per task (variable scores from stratified single-variable probes; the initial values of
    every variable are drawn from its best-scoring stratum). Each generation, the non-dominated members of every task
    receive masks transferred from a randomly chosen source task with a different score profile: sparse masks are filled
    up to the task's current median sparsity with variables active in the source, others take one source variable; the
    number of transferred solutions adapts to their survival rate."""

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)

    def _task_info(self):
        sd = getattr(self.problem, "sub_d", [self.problem.n_var])
        self.subD = [int(v) for v in sd]
        self.T = len(self.subD)
        self.maxD = max(self.subD)
        self.buq = [self.maxD - d for d in self.subD]

    def _solution(self, dec, mask, t):
        n = len(dec)
        sol = np.hstack([dec * mask, np.zeros((n, self.buq[t]))])
        if self.T > 1:
            sol = np.hstack([sol, np.full((n, 1), t + 1.0)])
        return sol

    def _initialize_infill(self):
        self._task_info()
        rng, T = self.rng, self.T
        lo, up = self.lower, self.upper
        enc = self.encoding
        self.REAL = bool(np.all(enc != 4))
        S = 1 + 4 * int(self.REAL)
        self.fit = [np.zeros(d) for d in self.subD]
        self.fitdec = [np.zeros((S, d)) for d in self.subD]
        self.probe = [[None] * T for _ in range(S)]
        allpops = []
        for i in range(S):
            for j in range(T):
                d = self.subD[j]
                if self.REAL:
                    a = lo[:d] + (up[:d] - lo[:d]) * (i / S)
                    b = lo[:d] + (up[:d] - lo[:d]) * ((i + 1) / S)
                    dec = a + rng.random((d, d)) * (b - a)
                else:
                    dec = np.ones((d, d))
                mask = np.eye(d)
                p = self.evaluate(self._solution(dec, mask, j))
                r = _nd_obj_cons(p)
                self.fit[j] += r
                self.fitdec[j][i] = r
                self.probe[i][j] = (p, dec, mask)
                allpops.append(p)
        return Population.merge(*allpops)

    def _initialize_advance(self, infills=None, **kwargs):
        rng, T, N = self.rng, self.T, self.N
        S = len(self.probe)
        lo, up = self.lower, self.upper
        self.EachN = int(np.ceil(N / T))
        E = self.EachN
        best = [[np.where(self.fitdec[j][:, d] == self.fitdec[j][:, d].min())[0] for d in range(self.subD[j])] for j in range(T)]
        self.sub, self.dec, self.mask, self.front, self.cd = [], [], [], [], []
        theta = np.zeros(T)
        for i in range(T):
            d = self.subD[i]
            if self.REAL:
                dec = np.zeros((E, d))
                for n in range(E):
                    for v in range(d):
                        k = best[i][v][rng.integers(0, len(best[i][v]))]
                        dec[n, v] = lo[v] + (up[v] - lo[v]) * (k / S) + rng.random() * (up[v] - lo[v]) / S
            else:
                dec = np.ones((E, d))
            mask = np.zeros((E, d))
            rank1 = np.argsort(self.fit[i], kind="stable")
            idx = np.round(np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * d).astype(int)
            for j in range(min(5, E)):
                mask[j, rank1[: idx[j]]] = 1
            for j in range(5, E):
                mask[j, tournament(2, int(np.ceil(rng.random() * d)), self.fit[i], rng=rng)] = 1
            pop = self.evaluate(self._solution(dec, mask, i))
            allp = Population.merge(pop, *[self.probe[s][i][0] for s in range(S)])
            alld = np.vstack([dec] + [self.probe[s][i][1] for s in range(S)])
            allm = np.vstack([mask] + [self.probe[s][i][2] for s in range(S)])
            _, _, _, tfront, _ = _sparse_select(allp, alld, allm, len(allp), unique=False)
            p, dd, mm, fr, c = _sparse_select(allp, alld, allm, E)
            self.sub.append(p); self.dec.append(dd); self.mask.append(mm); self.front.append(fr); self.cd.append(c)
            first5 = tfront[:5] == 1
            theta[i] = idx[: len(first5)][first5].mean() if first5.any() else mm[fr == 1].sum(1).mean()
        self.num_up = np.full(T, E // 10)
        self.theta_hist = [theta]
        self._source()
        self.pop = Population.merge(*self.sub)
        self._set_optimum()

    def _source(self):
        rng, T = self.rng, self.T
        TF = np.zeros((0, self.maxD))
        for i in range(T):
            TF = np.vstack([TF, np.concatenate([self.fit[i] / 5, np.zeros(self.buq[i])])])
            TF[TF == 0] = (self.fit[i] / 5).max() + 1
        self.TF = TF
        mse = ((TF[:, None, :] - TF[None, :, :]) ** 2).mean(-1)
        self.src = np.zeros(T, dtype=int)
        for i in range(T):
            arr = np.where(mse[i] != 0)[0]
            self.src[i] = arr[rng.permutation(len(arr))[0]] if len(arr) else i

    def _ga_half(self, P, t):
        rng = self.rng
        d = self.subD[t]
        n = len(P) // 2
        A, B = P[:n], P[n:2 * n]
        mu = rng.random((n, d))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (-1.0) ** rng.integers(0, 2, (n, d))
        beta[rng.random((n, d)) < 0.5] = 1
        off = (A + B) / 2 + beta * (A - B) / 2
        lo, up = np.broadcast_to(self.lower[:d], off.shape), np.broadcast_to(self.upper[:d], off.shape)
        s, m = rng.random((n, d)) < 1.0 / d, rng.random((n, d))
        off = np.minimum(np.maximum(off, lo), up)
        with np.errstate(all="ignore"):
            t_ = s & (m <= 0.5)
            off[t_] = off[t_] + (up[t_] - lo[t_]) * ((2 * m[t_] + (1 - 2 * m[t_]) * (1 - (off[t_] - lo[t_]) / (up[t_] - lo[t_])) ** 21) ** (1 / 21) - 1)
            t_ = s & (m > 0.5)
            off[t_] = off[t_] + (up[t_] - lo[t_]) * (1 - (2 * (1 - m[t_]) + 2 * (m[t_] - 0.5) * (1 - (up[t_] - off[t_]) / (up[t_] - lo[t_])) ** 21) ** (1 / 21))
        return off

    def _sparse_gen(self, t):
        rng, E = self.rng, self.EachN
        fit = self.fit[t]
        mate = tournament(2, 2 * E, self.front[t], -self.cd[t], rng=rng)
        pd, pm = self.dec[t][mate], self.mask[t][mate] > 0
        h = len(pd) // 2
        P1, P2 = pm[:h], pm[h:]
        off = P1.astype(float)
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(P1[i] & ~P2[i])[0]
                k = ts(-fit[idx], rng)
                if k is not None:
                    off[i, idx[k]] = 0
            else:
                idx = np.where(~P1[i] & P2[i])[0]
                k = ts(fit[idx], rng)
                if k is not None:
                    off[i, idx[k]] = P2[i, idx[k]]
        for i in range(h):
            if rng.random() < 0.5:
                idx = np.where(off[i] > 0)[0]
                k = ts(-fit[idx], rng)
                if k is not None:
                    off[i, idx[k]] = 0
            else:
                idx = np.where(off[i] == 0)[0]
                k = ts(fit[idx], rng)
                if k is not None:
                    off[i, idx[k]] = 1
        od = self._ga_half(pd, t) if self.REAL else np.ones((h, self.subD[t]))
        o = self.evaluate(self._solution(od, off, t))
        out = _sparse_select(Population.merge(self.sub[t], o), np.vstack([self.dec[t], od]), np.vstack([self.mask[t], off]), E)
        self.sub[t], self.dec[t], self.mask[t], self.front[t], self.cd[t] = out

    def _transfer(self):
        rng, T = self.rng, self.T
        NS = []
        for i in range(T):
            f1 = self.front[i] == 1
            pad = np.zeros((int(f1.sum()), self.buq[i]))
            NS.append((self.sub[i][f1], np.hstack([self.dec[i][f1], pad]), np.hstack([self.mask[i][f1], pad]), self.cd[i][f1]))
        cur_theta = np.zeros(T)
        pops = [None] * T
        offs = [None] * T
        for i in range(T):
            s = self.src[i]
            tp, tdec, tmask, tcd = NS[i]
            sp, sdec, smask, scd = NS[s]
            D = self.subD[i]
            n = len(tdec)
            if n > self.num_up[i]:
                n = int(self.num_up[i])
                tm = tournament(2, n, -tcd, rng=rng) if n > 0 else np.zeros(0, int)
                TD, TM = tdec[tm], tmask[tm]
            else:
                TD, TM = tdec, tmask
            th = self.theta_hist[-1][i]
            sm = tournament(2, n, -scd, rng=rng) if n > 0 and len(scd) else np.zeros(0, int)
            SM = smask[sm] if len(sm) else np.zeros((0, self.maxD))
            s1 = SM.sum(0) > 0 if len(SM) else np.zeros(self.maxD, bool)
            f1, f2 = self.TF[i], self.TF[s]
            OM = TM.copy()
            for r in range(min(n, len(SM))):
                if OM[r].sum() <= th:
                    i0 = np.where((TM[r] == 0) & s1)[0]
                    i0 = i0[i0 < D]
                    k = int(np.floor(th - OM[r].sum()))
                    if len(i0) and k > 0:
                        OM[r, i0[tournament(2, k, f1[i0], f2[i0], rng=rng)]] = 1
                else:
                    idx = np.where((TM[r] == 0) & (SM[r] > 0))[0]
                    idx = idx[idx < D]
                    if len(idx):
                        k = int(tournament(2, 1, f1[idx], f2[idx], rng=rng)[0])
                        OM[r, idx[k]] = SM[r, idx[k]]
            for r in range(len(OM)):
                idx = np.where(OM[r] > 0)[0]
                idx = idx[idx < D]
                k = ts(-f1[idx], rng)
                if k is not None:
                    OM[r, idx[k]] = 0
            OD = TD.copy()
            OD[:, np.where(self.encoding[: self.maxD] == 4)[0]] = 1
            if len(OD):
                sol = np.hstack([OD * OM, np.full((len(OD), 1), i + 1.0)])
                pops[i] = self.evaluate(sol)
            offs[i] = (OD, OM)
            # survival of the transferred solutions among the task's non-dominated set
            pool = Population.merge(tp, pops[i]) if pops[i] is not None else tp
            pd_, pm_ = np.vstack([tdec, OD]), np.vstack([tmask, OM])
            u = np.unique(objs(pool), axis=0, return_index=True)[1]
            if len(u) == 1:
                u = np.unique(pool.get("X"), axis=0, return_index=True)[1]
            pool, pd_, pm_ = pool[u], pd_[u], pm_[u]
            Nk = min(len(tp), len(pool))
            F, C = objs(pool), cons(pool)
            fr, maxf = nd_sort(F, C if C.shape[1] else None, Nk)
            nxt = fr < maxf
            cdv = crowding(F, fr)
            last = np.where(fr == maxf)[0]
            nxt[last[np.argsort(-cdv[last], kind="stable")[: Nk - int(nxt.sum())]]] = True
            sr = min(max((nxt[Nk:].sum() + 1e-6) / (self.num_up[i] + 1e-6), 0), 1)
            self.num_up[i] = int(np.round(self.num_up[i] * (1 + sr) / 2))
            kept_m, kept_f = pm_[nxt], fr[nxt]
            cur_theta[i] = kept_m[kept_f == 1].sum(1).mean() if (kept_f == 1).any() else 0.0
        self.theta_hist.append(cur_theta)
        for i in range(T):
            D = self.subD[i]
            if pops[i] is None:
                continue
            OD, OM = offs[i]
            out = _sparse_select(Population.merge(self.sub[i], pops[i]), np.vstack([self.dec[i], OD[:, :D]]),
                                 np.vstack([self.mask[i], OM[:, :D]]), self.EachN)
            self.sub[i], self.dec[i], self.mask[i], self.front[i], self.cd[i] = out

    def step(self):
        for t in range(self.T):
            self._sparse_gen(t)
        self._transfer()
        self._source()
        self.pop = Population.merge(*self.sub)
