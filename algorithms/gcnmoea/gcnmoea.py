# emopylab 2026
"""GCNMOEA (graph convolutional network based multi-objective evolutionary algorithm).

Reference:
P. Yan, Y. Tian, and Y. Liu. An indicator-based multi-objective evolutionary algorithm assisted by
improved graph convolutional networks. Swarm and Evolutionary Computation, 2025, 94: 101892.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, de, decs, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population
from util.hv import hypervolume

ALGORITHM_FLAGS = {'GCNMOEA': {'integer', 'multi', 'real'}}


# ----------------------------------------------------------------------------------------------- graph utilities
def pagerank(A, d=0.85, tol=1e-4, max_iter=100):
    n = len(A)
    deg = A.sum(1)
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dang = x[deg == 0].sum()
        with np.errstate(all="ignore"):
            share = np.where(deg > 0, x / deg, 0.0)
        xn = (1 - d) / n + d * (A.T @ share + dang / n)
        if np.abs(xn - x).sum() < tol:
            x = xn
            break
        x = xn
    return x


def betweenness(A):
    """Brandes betweenness of an unweighted undirected graph (each unordered pair counted once)."""
    n = len(A)
    nb = [np.where(A[i] > 0)[0] for i in range(n)]
    cb = np.zeros(n)
    for s in range(n):
        S, P = [], [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1
        dist = np.full(n, -1)
        dist[s] = 0
        Q = [s]
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in nb[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    return cb / 2


def _find_degree(P, adj):
    deg = np.array([len(np.intersect1d(P, adj[node - 1])) for node in P])
    return list(np.asarray(P)[np.argsort(-deg, kind="stable")])


def bron_kerbosch(adj, R, P, X):
    """Literal (1-based) maximum-clique recursion of the reference."""
    if len(P) == 0 and len(X) == 0:
        return R
    best = np.array([], dtype=int)
    for node in _find_degree(P, adj):
        nr = np.union1d(R, [node])
        npp = np.intersect1d(P, adj[node - 1])
        nx = np.intersect1d(X, adj[node - 1])
        cl = bron_kerbosch(adj, nr, npp, nx)
        if len(cl) > len(best):
            best = cl
        P = np.setdiff1d(P, [node])
        X = np.union1d(X, [node])
    return best


def _cv(C):
    return np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(len(C))


class GCNMOEA(LoopAlgorithm):
    """A correlation graph over the population (at most 15 links per individual, strongest or weakest correlations) drives
    the variation: sampled nodes recombine with neighbour cliques through a graph self-attention layer whose weights are
    tuned by PSO with annealing on the hypervolume (evaluations not charged, as in the reference), then with their most
    central neighbours (PageRank + betweenness); otherwise a level-based NSGA-style generation is run. Selection sorts by
    non-dominance with a PBI tie-breaker inside each reference direction."""

    UNCHARGED_EVALS = True

    def _initialize_infill(self):
        self.W1, self.pop_size = uniform_point(self.pop_size, self.M)
        return self.evaluate(self.random_decs(self.N))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.Z = objs(infills).min(0)
        self.eps_k = 0.0
        self.pop, self.front, self.d2 = self._env(infills)
        self._set_optimum()

    # ------------------------------------------------------------------------------------------------ selection
    def _env(self, pop):
        N, W = self.N, self.W1
        F = objs(pop)
        with np.errstate(all="ignore"):
            P = (F - F.min(0)) / (F.max(0) - F.min(0))
        nP = np.linalg.norm(P, axis=1)
        with np.errstate(all="ignore"):
            cos = (P @ W.T) / (nP[:, None] * np.linalg.norm(W, axis=1)[None])
        d1a, d2a = nP[:, None] * cos, nP[:, None] * np.sqrt(np.maximum(1 - cos ** 2, 0))
        d2a = np.where(np.isnan(d2a), np.inf, d2a)
        RP = np.argmin(d2a, 1)
        d2 = d2a[np.arange(len(P)), RP]
        d1 = d1a[np.arange(len(P)), RP]
        f, _ = nd_sort(P, None, 1)
        nd = np.where(f == 1)[0]
        ext = nd[np.argmax(np.where(np.isnan(P[nd]), -np.inf, P[nd]), 0)]
        d1[ext], d2[ext] = 0, 0
        front, maxf = self._spd_sort(P, d1, d2, RP, N)
        nxt = front < maxf
        last = np.where(front == maxf)[0]
        nxt[last[np.argsort(d2[last], kind="stable")[: N - int(nxt.sum())]]] = True
        return pop[nxt], front[nxt], d2[nxt]

    @staticmethod
    def _spd_sort(F, d1, d2, RP, n_sort):
        N = len(F)
        front, maxf = np.full(N, np.inf), 0
        lt = (F[:, None] < F[None]).any(-1)
        gt = (F[:, None] > F[None]).any(-1)
        score = d1 + 5 * d2
        while (front < np.inf).sum() < min(n_sort, N):
            maxf += 1
            Dm = front != np.inf
            for i in range(N):
                if Dm[i]:
                    continue
                for j in range(i + 1, N):
                    if Dm[j]:
                        continue
                    domi = 1 if (lt[i, j] and not gt[i, j]) else -1 if (gt[i, j] and not lt[i, j]) else 0
                    if domi == 0 and RP[i] == RP[j]:
                        domi = 1 if score[i] < score[j] else -1 if score[i] > score[j] else 0
                    if domi == 1:
                        Dm[j] = True
                    elif domi == -1:
                        Dm[i] = True
                        break
                if not Dm[i]:
                    front[i] = maxf
        return front, maxf

    def _env1(self, pop):
        N, M, W = self.N, self.M, self.W1
        n = 2 * M
        F = objs(pop)
        fr, maxf = nd_sort(F, None, np.inf)
        keep = np.zeros(len(pop), bool)
        for i in range(1, int(maxf) + 1):
            keep |= fr == i
            if keep.sum() >= N:
                break
        P = pop[np.where(keep)[0]]
        F = objs(P)
        zmin, zmax = F.min(0), F.max(0)
        inter = (zmax - zmin) / n
        lvl = np.zeros(len(P), int)
        for i, f in enumerate(F):
            t = 0
            while True:
                t += 1
                if not np.any(f > zmin + (t + 1) * inter):
                    break
                if t > 10 ** 6:
                    break
            lvl[i] = t
        idx = np.concatenate([np.where(lvl == i)[0] for i in range(1, n + 1)])    # every level is kept (see reference)
        pool = list(idx)
        out = []
        for w in W:
            if not pool:
                break
            Fp = objs(P[np.array(pool)])
            with np.errstate(all="ignore"):
                S = (Fp - Fp.min(0)) / (Fp.max(0) - Fp.min(0))
                c = (S @ w) / (np.linalg.norm(S, axis=1) * np.linalg.norm(w))
            k = int(np.argmax(np.where(np.isnan(c), -np.inf, c)))
            out.append(pool.pop(k))
        return P[np.array(out)]

    def _density(self, pop):
        F = objs(pop)
        with np.errstate(all="ignore"):
            S = (F - F.min(0)) / (F.max(0) - F.min(0))
            c = (S @ self.W1.T) / (np.linalg.norm(S, axis=1)[:, None] * np.linalg.norm(self.W1, axis=1)[None])
        region = np.argmax(np.where(np.isnan(c), -np.inf, c), 1)
        cnt = np.bincount(region, minlength=len(self.W1))
        return cnt[region].astype(float)

    # ------------------------------------------------------------------------------------------------ helpers
    def _replace(self, idx, new):
        merged = Population.merge(self.pop, new)
        sel = np.arange(len(self.pop))
        sel[np.asarray(idx, int)] = len(self.pop) + np.arange(len(new))
        self.pop = merged[sel]

    def _attention(self, X, adj, Wm, a, charged):
        XW = X @ Wm
        D = XW.shape[1]
        s = np.tanh((XW @ a[:D])[:, None] + (XW @ a[D:])[None, :])
        e = np.where(adj == 1, np.exp(s), 0.0)
        with np.errstate(all="ignore"):
            alpha = e / e.sum(1, keepdims=True)
            H = alpha @ XW
        H = np.fmax(np.fmin(H, self.upper), self.lower)      # MATLAB min/max ignore NaN (isolated rows -> bounds)
        return self.evaluate(H) if charged else self.solutions_uncharged(H)

    def _optimize_weights(self, parent, adj):
        rng = self.rng
        X = decs(parent)
        D = X.shape[1]
        ref = objs(parent).max(0) + 0.1

        def hv(p):
            F = objs(p)
            F = F[~np.any(F > ref, 1)]
            return float(hypervolume(F, ref)) if len(F) else 0.0

        n, T = 30, 100.0
        W = rng.random((n, D, D))
        a = rng.random((n, 2 * D))
        vW, va = rng.random((n, D, D)), rng.random((n, 2 * D))
        pW, pa = W.copy(), a.copy()
        phv = np.array([hv(self._attention(X, adj, W[i], a[i], False)) for i in range(n)])
        g = int(np.argmax(phv))
        ghv, gW, ga_ = phv[g], W[g].copy(), a[g].copy()
        for it in range(1, 101):
            w = 0.9 - 0.5 * it / 100
            for i in range(n):
                vW[i] = w * vW[i] + 3.0 * rng.random() * (pW[i] - W[i]) + 3.0 * rng.random() * (gW - W[i])
                va[i] = w * va[i] + 3.0 * rng.random() * (pa[i] - a[i]) + 3.0 * rng.random() * (ga_ - a[i])
                W[i] += vW[i]
                a[i] += va[i]
                cur = hv(self._attention(X, adj, W[i], a[i], False))
                dl = cur - phv[i]
                with np.errstate(over="ignore"):
                    if dl > 0 or np.exp(dl / T) > rng.random():
                        pW[i], pa[i], phv[i] = W[i].copy(), a[i].copy(), cur
                if cur > ghv:
                    ghv, gW, ga_ = cur, W[i].copy(), a[i].copy()
            T *= 0.99
            if T <= 1e-3:
                break
        return gW, ga_

    def _graph(self):
        rng, X = self.rng, decs(self.pop)
        n = len(X)
        R = np.corrcoef(X)
        order = np.argsort(-R.ravel(order="F"), kind="stable") if rng.random() < 0.3 else np.argsort(R.ravel(order="F"), kind="stable")
        A = np.zeros((n, n))
        rs, cs = np.zeros(n), np.zeros(n)
        for k in order:
            i, j = k % n, k // n
            if i != j and rs[i] < 15 and cs[j] < 15:
                if A[i, j] == 0:
                    A[i, j] = A[j, i] = 1
                    rs = A.sum(1)
                    cs = A.sum(0)
        return A

    # ------------------------------------------------------------------------------------------------ main loop
    def step(self):
        rng, N = self.rng, self.N
        A = self._graph()
        nbrs = [np.where(A[i] > 0)[0] for i in range(len(A))]
        if rng.random() < 0.4 or self.FE < 0.5 * self.max_FE:
            reds, greens = [], []
            for _ in range(50):
                red = int(rng.integers(0, len(A)))
                green = nbrs[red]
                if len(green) > 8:
                    green = rng.permutation(green)[:8]
                reds.append(red)
                greens.append(np.asarray(green))
                flag, Wm, av = 1, None, None
                for gnode in green:
                    black = nbrs[gnode][nbrs[gnode] != red]
                    if len(black) > 10:
                        black = rng.permutation(black)[:10]
                    if len(black) > 1:
                        index = np.concatenate([[gnode], black])
                        sub = A[np.ix_(index, index)]
                        adj = [list(np.where(sub[r] == 1)[0] + 1) for r in range(len(index))]
                        mc = bron_kerbosch(adj, np.array([], int), np.arange(1, len(black) + 1), np.array([], int))
                        mc = np.asarray(mc, int)[1:]
                        b1 = black[mc - 1] if len(mc) else np.array([], int)
                        if len(b1) < 2:
                            k = 2 if len(black) == 2 else 3 if len(black) == 3 else 4
                            b2 = rng.permutation(black)[: min(k, len(black))]
                            if len(b1) == 1:
                                b2 = b2[b2 != b1[0]]
                        else:
                            b2 = np.array([], int)
                        if len(b1) == 0:
                            b1 = rng.permutation(black)[: min(4, len(black))]
                        black2 = np.concatenate([b1, b2]).astype(int)
                        index = np.concatenate([[gnode], black2]).astype(int)
                        parent = self.pop[index]
                        if len(parent) == 2:
                            off = self.evaluate(ga(self.problem, decs(parent), (1, 10, 1, 10), rng=rng))
                        else:
                            adjm = A[np.ix_(index, index)]
                            if flag == 1:
                                Wm, av = self._optimize_weights(parent, adjm)
                                flag = 2
                            off = self._attention(decs(parent), adjm, Wm, av, True)
                        self.Z = np.minimum(self.Z, objs(off).min(0))
                        W1 = self.W1[index]
                        g_old = np.max(np.abs(objs(parent) - self.Z) * W1, 1)
                        g_new = np.max(np.abs(objs(off) - self.Z) * W1, 1)
                        co, cn = _cv(cons(parent)), _cv(cons(off)) * np.ones(len(index))
                        e = self.eps_k
                        cond = ((g_old >= g_new) & (((co <= e) & (cn <= e)) | (co == cn))) | (cn < co)
                        pos = np.where(cond)[0][:2]
                        if len(pos):
                            ii = index[pos]
                            keep = np.isin(index, ii)
                            # Population(ii) = Offspring(ismember(index, ii)): assignment in order of appearance
                            self._replace(ii, off[np.where(keep)[0][: len(ii)]])
                    elif len(black) == 1:
                        par = np.array([gnode, int(black[0])])
                        off = self.evaluate(ga(self.problem, decs(self.pop)[par], rng=rng))
                        self.pop, _, _ = self._env(Population.merge(self.pop, off))
                    self.pop, self.front, self.d2 = self._env(self.pop)
            pr = pagerank(A)
            bt = betweenness(A)
            wgt = 0.5 * pr + 0.5 * bt
            for red, green in zip(reds, greens):
                n_g = len(green)
                if n_g in (1, 2):
                    b = green[int(np.argmax(wgt[green]))]
                    off = self.evaluate(ga(self.problem, decs(self.pop)[[red, b]], (1, 10, 1, 10), rng=rng))
                    self._replace([red], off[[0]])
                    self._replace([b], off[[1]])
                elif n_g > 0:
                    o = green[np.argsort(-wgt[green], kind="stable")]
                    X = decs(self.pop)
                    off = self.evaluate(de(self.problem, X[[red]], X[[o[0]]], X[[o[1]]], rng=rng))
                    self._replace([red], off)
                self.pop, self.front, self.d2 = self._env(self.pop)
        else:
            mate = tournament(2, N, self.front, self.d2, rng=rng)
            off = self.evaluate(ga(self.problem, decs(self.pop)[mate], rng=rng))
            self.pop = self._env1(Population.merge(self.pop, off))
            self.front, _ = nd_sort(objs(self.pop), None, np.inf)
            self.d2 = self._density(self.pop)
