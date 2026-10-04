"""Two-archive helpers of the Kriging-assisted Two_Arch2 family (KTA2 / KTS): influential-point-insensitive models, IBEA-style
convergence archive, Lp-diversity archive, pure-diversity score, Wilcoxon signed-rank convergence test, adaptive sampling."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import nd_sort
from algorithms.community_utils.dace import norm_cdf

__all__ = ["signrank", "pure_diversity", "ibea_keep", "lp_greedy", "cal_convergence", "adaptive_sampling", "k_update_ca", "k_update_da"]


def _minkowski(A, B, p):
    return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(-1) ** (1.0 / p)


def _tiedrank(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    sa = a[order]
    i, adj = 0, 0.0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        t = j - i + 1
        adj += (t ** 3 - t) / 2
        i = j + 1
    return r, adj


def signrank(x, y, alpha=0.05):
    """Two-sided Wilcoxon signed-rank test; returns (p, h, r1, r2) with r1 the rank sum of negative differences."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    eps = np.spacing(np.abs(x)) + np.spacing(np.abs(y))
    keep = ~np.isnan(d)
    d, eps = d[keep], eps[keep]
    keep = ~(np.abs(d) < eps)
    d = d[keep]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0.0, 0.0
    r, adj = _tiedrank(np.abs(d))
    w = r[d < 0].sum()
    r1, r2 = w, n * (n + 1) / 2 - w
    w = min(w, n * (n + 1) / 2 - w)
    if n <= 15:
        s = np.round(2 * r).astype(int)
        dist = np.zeros(s.sum() + 1)
        dist[0] = 1.0
        for v in s:
            dist[v:] = dist[v:] + dist[:-v].copy() if v else dist * 2
        dist /= 2.0 ** n
        p = min(1.0, 2 * dist[: int(np.floor(2 * w + 1e-9)) + 1].sum())
    else:
        z = (w - n * (n + 1) / 4) / np.sqrt((n * (n + 1) * (2 * n + 1) - adj) / 24)
        p = 2 * float(norm_cdf(z))
    return p, int(p <= alpha), r1, r2


def pure_diversity(F):
    """Pure diversity score (sum of the dissimilarities added while greedily linking solutions into a tree)."""
    N = len(F)
    C = np.eye(N, dtype=bool)
    D = _minkowski(F, F, 0.1)
    np.fill_diagonal(D, np.inf)
    score = 0.0
    for _ in range(N - 1):
        while True:
            J = np.argmin(D, axis=1)
            d = D[np.arange(N), J]
            i = int(np.argmax(d))
            j = int(J[i])
            if D[j, i] != -np.inf:
                D[j, i] = np.inf
            if D[i, j] != -np.inf:
                D[i, j] = np.inf
            P = C[i].copy()
            while not P[j]:
                newP = C[P].any(0)
                if np.array_equal(P, newP):
                    break
                P = newP
            if not P[j]:
                break
        C[i, j] = C[j, i] = True
        D[i, :] = -np.inf
        score += d[i]
    return score


def ibea_keep(F, K):
    """Indices kept by iteratively removing the solution with the smallest indicator-based fitness (kappa = 0.05)."""
    N = len(F)
    with np.errstate(all="ignore"):
        Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
    I = np.max(Fn[:, None, :] - Fn[None, :, :], axis=2)
    C = np.abs(I).max(0)
    with np.errstate(all="ignore"):
        fit = (-np.exp(-I / C[None, :] / 0.05)).sum(0) + 1
    choose = list(range(N))
    while len(choose) > K:
        x = int(np.argmin(fit[choose]))
        with np.errstate(all="ignore"):
            fit = fit + np.exp(-I[choose[x], :] / C[choose[x]] / 0.05)
        del choose[x]
    return np.array(choose, dtype=int)


def lp_greedy(P, choose, K, p):
    """Complete ``choose`` greedily (max-min Minkowski-p distance) until ``K`` rows are chosen."""
    Dm = _minkowski(P, P, p)
    np.fill_diagonal(Dm, np.inf)
    added = []
    while choose.sum() < K:
        rem = np.where(~choose)[0]
        x = rem[int(np.argmax(Dm[np.ix_(rem, np.where(choose)[0])].min(1)))]
        choose[x] = True
        added.append(x)
    return choose, added


def cal_convergence(F1, F2, zmin):
    if len(F1) != len(F2):
        return 0
    P = np.vstack([F1, F2]) - zmin
    with np.errstate(all="ignore"):
        P = P / (P.max(0) - zmin)
        d1, d2 = np.sqrt(P[: len(F1)].sum(1)), np.sqrt(P[len(F1):].sum(1))
    _, h, r1, r2 = signrank(d1, d2)
    return 0 if (h == 1 and r1 - r2 < 0) else h


def k_update_ca(F, X, V, K):
    if len(F) <= K:
        return F, X, V
    k = ibea_keep(F, K)
    return F[k], X[k], V[k]


def k_update_da(F, X, V, K, p, rng):
    with np.errstate(all="ignore"):
        pre = (F - F.min(0)) / (F.max(0) - F.min(0))
    nd, _ = nd_sort(F, None, 1)
    nd = nd == 1
    F, X, V, pre = F[nd], X[nd], V[nd], pre[nd]
    N = len(F)
    if N <= K:
        return F, X, V
    ch = np.zeros(N, bool)
    ch[rng.permutation(F.shape[1])[0]] = True        # the reference seeds with a random index in 1..M
    ch, _ = lp_greedy(pre, ch, K, p)
    return F[ch], X[ch], V[ch]


def adaptive_sampling(CAo, DAo, CAd, DAd, DAv, DA_F, DA_X, mu, p, phi, rng, small_shortcut=False):
    """``small_shortcut`` (KTS variant): with at most ``mu`` predicted diversity-archive members, sample convergence first and
    otherwise return the whole predicted archive; only the first ``M`` uncertainty columns are averaged."""
    ideal = np.vstack([CAo, DAo]).min(0)
    flag = 1 if (small_shortcut and len(DAd) <= mu) else cal_convergence(CAo, DAo, ideal)
    if flag == 1:
        return CAd[ibea_keep(CAo, mu)]
    if small_shortcut and len(DAd) <= mu:
        return DAd
    DAv = DAv[:, : DAo.shape[1]]
    if pure_diversity(DAo) < pure_diversity(DA_F):
        An = len(DAv)
        ch = []
        for _ in range(mu):
            a = rng.permutation(An)
            unc = DAv[a[: int(np.ceil(phi * An))]].mean(1)
            ch.append(a[int(np.argmax(unc))])
        return DAd[ch]
    allo = np.vstack([DAo, DA_F])
    lo, hi = allo.min(0), allo.max(0)
    with np.errstate(all="ignore"):
        A = (DA_F - lo) / (hi - lo)
        B = (DAo - lo) / (hi - lo)
    P = np.vstack([A, B])
    X = np.vstack([DA_X, DAd])
    ch = np.zeros(len(P), bool)
    ch[: len(A)] = True
    Dm = _minkowski(P, P, p)
    np.fill_diagonal(Dm, np.inf)
    out = []
    while ch.sum() < len(A) + mu and (~ch).any():
        rem = np.where(~ch)[0]
        x = rem[int(np.argmax(Dm[np.ix_(rem, np.where(ch)[0])].min(1)))]
        ch[x] = True
        out.append(x)
    return X[out]
