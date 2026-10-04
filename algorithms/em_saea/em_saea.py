# emopylab 2026
"""EM-SAEA (ensemble-based surrogate model-assisted evolutionary algorithm).

Reference:
Y. Li, X. Feng, and H. Yu. Enhancing landscape approximation with ensemble-based surrogate model for
expensive constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation,
2025.
"""

from __future__ import annotations

import numpy as np

from algorithms.ab_saea.ab_saea import ds_merge
from algorithms.community_utils import nsga3_ref
from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, ga_half, kmeans, nd_sort, objs, uniform_point
from algorithms.community_utils.dace import DaceModel
from algorithms.k_rvea.k_rvea import _angles, _argmin_rows, _env_selection, _no_active
from core.population import Population

ALGORITHM_FLAGS = {'EM_SAEA': {'constrained', 'expensive', 'many', 'multi', 'real'}}


def _unique_stable(A):
    _, idx = np.unique(A, axis=0, return_index=True)
    return np.sort(idx)


def _in_rows(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros(len(A), bool)
    return np.array([np.any(np.all(B == a, 1)) for a in A])


def _mse_tf(M):
    M = np.maximum(M, 0)
    return np.where(M <= 1, np.sqrt(M), M)


def archive_update(X, F, C, N, rng):
    feas = np.all(C <= 0, 1) if C.shape[1] else np.ones(len(X), bool)
    X, F, C = X[feas], F[feas], C[feas]
    if len(X) == 0:
        return np.zeros((0, X.shape[1]))
    f, _ = nd_sort(F, None, 1)
    X, F = X[f == 1], F[f == 1]
    r = rng.permutation(len(X))
    X, F = X[r], F[r]
    if len(X) > N:
        with np.errstate(all="ignore"):
            Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
        d = np.sqrt(((Fn[:, None] - Fn[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        sd = np.sort(d, 1)
        rr = np.median(sd[:, min(Fn.shape[1], sd.shape[1]) - 1])
        R = np.minimum(d / rr, 1)
        keep = list(range(len(X)))
        while len(keep) > N:
            w = int(np.argmax(1 - np.prod(R[np.ix_(keep, keep)], 1)))
            del keep[w]
        X = X[keep]
    return X


def kriging_select(X, F, V, mu, theta, C, rng):
    cv = np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(len(X))
    nva, va = _no_active(F, V)
    nc = min(mu, len(V) - nva)
    Va = V[va]
    idx = kmeans(Va, nc, rng)
    Fs = F - F.min(0)
    cosv = np.cos(_angles(Va, Va))
    np.fill_diagonal(cosv, 0)
    gamma = np.arccos(cosv).min(1)
    ang = _angles(Fs, Va)
    assoc = _argmin_rows(ang)
    apd = np.ones(len(F))
    for i in np.unique(assoc):
        cur = assoc == i
        apd[cur] = (1 + F.shape[1] * theta * ang[cur, i] / gamma[i]) * np.sqrt((Fs[cur] ** 2).sum(1))
    cidx = idx[assoc]
    nxt = []
    for i in np.unique(cidx):
        b1, b2 = [], []
        for t in np.unique(assoc[cidx == i]):
            s1 = np.where((assoc == t) & (cv == 0))[0]
            s2 = np.where((assoc == t) & (cv != 0))[0]
            if len(s1):
                b1.append(s1[int(np.argmin(apd[s1]))])
            elif len(s2):
                b2.append(s2[int(np.argmin(cv[s2]))])
        if b1:
            nxt.append(b1[int(np.argmin(apd[b1]))])
        elif b2:
            nxt.append(b2[int(np.argmin(cv[b2]))])
    return X[np.array(nxt, dtype=int)]


def _igd(P, opt):
    return np.mean(np.sqrt(((opt[:, None] - P[None]) ** 2).sum(-1)).min(1))


def kriging_select_rvmm(X, F, X1, F1, A2, rng):
    u = _unique_stable(F1)
    F1, X1 = F1[u], X1[u]
    u = _unique_stable(F)
    F, X = F[u], X[u]
    if len(F):
        f, _ = nd_sort(F, None, 1)
        F, X = F[f == 1], X[f == 1]
    if len(F1):
        f, _ = nd_sort(F1, None, 1)
        F1, X1 = F1[f == 1], X1[f == 1]
    whole = np.vstack([A2, F1, F]) if len(F1) or len(F) else A2
    zmin, Zmin = whole.min(0), A2.min(0)
    zmin1 = whole.min(0)
    scale = A2.max(0) - A2.min(0)
    scale[scale == 0] = 1e-6
    if len(F1):
        zmin1 = np.minimum(zmin1, F1.min(0))
    cd = []
    for a in F1:
        d = np.sqrt((((a - zmin1) - (A2 - zmin1)) ** 2).sum(1))
        j = int(np.argmin(d))
        k = int(np.any(a < A2[j])) - int(np.any(a > A2[j]))
        cd.append(d[j] if k == 1 else 0.0)
    cd = np.array(cd)
    cbest = None
    if len(cd):
        cid = np.where(cd == cd.max())[0]
        cbest = int(cid[rng.integers(0, len(cid))])
    dbest = None
    zmin = np.minimum(zmin, zmin1)
    if len(F):
        zmin = np.minimum(zmin, F.min(0))
        R = np.maximum(F - zmin, 0) / scale
        a1, a2 = np.maximum(A2 - zmin, 0) / scale, np.maximum(A2 - Zmin, 0) / scale
        ang = _angles(R, a1 if _igd(R, a1) < _igd(R, a2) else a2)
        dd = ang.min(1)
        dd = dd / dd.max() if dd.max() > 0 else dd
        dbest = int(np.argmax(dd))
    if cbest is not None:
        fr, _ = nd_sort(np.vstack([F1[[cbest]], A2]), None, len(A2))
        if fr[0] == 1:
            return X1[[cbest]]
    if dbest is not None:
        return X[[dbest]]
    return np.zeros((0, X.shape[1] if X.ndim == 2 else 0))


class EM_SAEA(LoopAlgorithm):
    """Stage 2 (at the start and in the last half of the budget): MOEA/D-style search on Kriging objectives with local
    (per objective-space cluster) and global constraint models, sampling the feasible non-dominated survivors. Stage 1:
    two RVEA-style surrogate searches (optimistic objectives f + 0.5 s) around a few representative reference vectors and
    around all active vectors; one solution is sampled by convergence/diversity gain over the evaluated front."""

    def __init__(self, pop_size: int = 100, wmax: int = 20, lc_num: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.lc_num = int(wmax), int(lc_num)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.V0, self.NI = uniform_point(self.pop_size, self.M)
        self.W, self.pop_size = uniform_point(self.pop_size, self.M)
        self.ClW, _ = uniform_point(self.lc_num, self.M)
        self.lc_num = len(self.ClW)
        P, _ = UniformPoint(self.NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        N, D, M = self.N, self.D, self.M
        self.pop = infills
        self.Vbb = self.V0.copy()
        self.T, self.nr = int(np.ceil(N / 10)), int(np.ceil(N / 100))
        self.B = np.argsort(np.sqrt(((self.W[:, None] - self.W[None]) ** 2).sum(-1)), 1, kind="stable")[:, : self.T]
        self.numCon = cons(infills).shape[1]
        self.oth = 5.0 * np.ones((M, D))
        self.th_lc = [5.0 * np.ones((self.numCon, D)) for _ in range(self.lc_num)]
        self.th_gc = 5.0 * np.ones((self.numCon, D))
        self.Z = objs(infills).min(0)
        ang = _angles(self.W, self.W)
        np.fill_diagonal(ang, np.inf)
        self.theta = ang.min(1) * 0.5
        self.stage = 2
        self._set_optimum()

    def _dace(self, X, y, th):
        D = self.D
        m = DaceModel(X, y, "regpoly0", th, 1e-5 * np.ones(D), 100 * np.ones(D))
        return m, m.theta

    def _predict(self, models, X):
        p = [m.predict(X, mse=True) for m in models]
        return np.column_stack([q[0] for q in p]), np.column_stack([q[1] for q in p])

    def step(self):
        TA = self.pop[np.unique(decs(self.pop), axis=0, return_index=True)[1]]
        TX, TF, TC = decs(TA), objs(TA), cons(TA)
        self.om = []
        for i in range(self.M):
            sX, sY = ds_merge(TX, TF[:, i])
            m, self.oth[i] = self._dace(sX, sY, self.oth[i])
            self.om.append(m)
        new = self._stage1(TA) if self.stage == 1 else self._stage2(TA, TX, TF, TC)
        if new is not None and len(new):
            self.pop = Population.merge(self.pop, self.evaluate(new))
        self.stage = 1 if self.FE < np.ceil(self.NI + 0.5 * (self.max_FE - self.NI)) else 2

    # ---------------------------------------------------------------- stage 2
    def _stage2(self, TA, TX, TF, TC):
        rng, N, M, nc = self.rng, self.N, self.M, self.numCon
        feas = np.all(TC <= 0, 1) if nc else np.ones(len(TX), bool)
        arc_fz = TF[feas].min(0) if feas.any() else np.ones(M)
        fr, maxf = nd_sort(TF, TC if nc else None, N)
        nxt = fr < maxf
        last = np.where(fr == maxf)[0]
        ch = nsga3_ref.last_selection(TF[nxt], TF[last], N - int(nxt.sum()), self.W, arc_fz, rng)
        nxt[last[ch]] = True
        PX = TX[nxt]
        zmin = TF.min(0)
        n = len(TX)
        NO = TF - zmin
        cl1, cl2 = np.full((n, self.lc_num), np.inf), np.full((n, self.lc_num), np.inf)
        size = int(np.ceil(n / self.lc_num))
        for i in range(self.lc_num):
            a = _angles(NO, self.ClW[[i]])[:, 0]
            cl1[np.argsort(np.where(np.isnan(a), np.inf, a), kind="stable")[:size], i] = i
        left, NO2 = n, NO.copy()
        while left > 0:
            for i in range(self.lc_num):
                a = _angles(NO2, self.ClW[[i]])[:, 0]
                a = np.where(np.isnan(a) | np.isinf(NO2).any(1), np.inf, a)
                loc = int(np.argmin(a))
                cl2[loc, i] = i
                NO2[loc] = np.inf
                left -= 1
                if left == 0:
                    break
        cl = np.minimum(cl1, cl2)
        mlc = [[None] * nc for _ in range(self.lc_num)]
        for i in range(self.lc_num):
            sel = cl[:, i] == i
            for j in range(nc):
                x, y = ds_merge(TX[sel], TC[sel, j])
                mlc[i][j], self.th_lc[i][j] = self._dace(x, y, self.th_lc[i][j])
        mgc = []
        for j in range(nc):
            x, y = ds_merge(TX, TC[:, j])
            m, self.th_gc[j] = self._dace(x, y, self.th_gc[j])
            mgc.append(m)
        Z = np.minimum(self.Z, TF.min(0))
        PF, _ = self._predict(self.om, PX)
        PC = self._predict(mgc, PX)[0] if nc else np.zeros((len(PX), 0))
        W, B = self.W, self.B
        for _ in range(self.wmax):
            for i in range(N):
                P = B[i][rng.permutation(self.T)] if rng.random() < 0.9 else rng.permutation(N)
                od = ga_half(self.problem, PX[P[:2]], rng=rng)[:1]
                of, _ = self._predict(self.om, od)
                of = of[0]
                zmin = np.minimum(zmin, of)
                Z = np.minimum(Z, of)
                loc = int(np.nanargmin(_angles((of - zmin)[None], self.ClW)[0]))
                if nc:
                    cl_, cl_m = self._predict(mlc[loc], od)
                    cg_, cg_m = self._predict(mgc, od)
                    oc = cg_[0] if np.mean(cg_m) < np.mean(cl_m) else cl_[0]
                else:
                    oc = np.zeros(0)
                cvo = np.maximum(0, oc).sum()
                cvp = np.maximum(0, PC[P]).sum(1) if nc else np.zeros(len(P))
                with np.errstate(all="ignore"):
                    nW = np.sqrt((W[P] ** 2).sum(1))
                    nP = np.sqrt(((PF[P] - Z) ** 2).sum(1))
                    nO = np.sqrt(((of - Z) ** 2).sum())
                    cP = ((PF[P] - Z) * W[P]).sum(1) / nW / nP
                    cO = ((of - Z) * W[P]).sum(1) / nW / nO
                    g_old = nP * cP + 5 * nP * np.sqrt(1 - cP ** 2)
                    g_new = nO * cO + 5 * nO * np.sqrt(1 - cO ** 2)
                if np.all(cvo + cvp == 0):
                    rep = np.where(g_old >= g_new)[0][: self.nr]
                else:
                    # literal: the reference tests ~isempty(...) of a logical vector, which is always true
                    rep = np.where(((g_old >= g_new) & (cvp == cvo)) | (cvp > cvo))[0][: self.nr]
                if len(rep):
                    PX[P[rep]], PF[P[rep]] = od[0], of
                    if nc:
                        PC[P[rep]] = oc
        self.Z = Z
        new = archive_update(PX, PF, PC, 5, rng)
        if len(new) == 0:
            new = kriging_select(PX, PF, W, 5, (self.FE / self.max_FE) ** 2, PC, rng)
        return new

    # ---------------------------------------------------------------- stage 1
    def _filter_new(self, off, ref):
        keep = []
        for x in off:
            pool = np.vstack([ref] + keep) if keep else ref
            if np.sqrt(((pool - x) ** 2).sum(1)).min() > 1e-6:
                keep.append(x[None])
        return np.vstack(keep) if keep else np.zeros((0, off.shape[1]))

    def _search(self, X0, V, A1X, TX, mating_all, accumulate, adapt_V0=None):
        rng, N, kk, wmax = self.rng, self.N, 0.5, self.wmax
        X = X0.copy()
        WX, WF, WM = [], [], []
        scale = None
        V0 = adapt_V0
        F = M_ = np.zeros((0, self.M))
        for w in range(1, wmax + 1):
            if mating_all and len(X) >= N:
                off = ga(self.problem, X, rng=rng)
            else:
                off = ga(self.problem, X[rng.integers(0, len(X), N)], rng=rng)
            off = self._filter_new(off, A1X)
            X = np.vstack([X, off])
            F, Ms = self._predict(self.om, X)
            Ms = _mse_tf(Ms)
            Fb, Mb = F.copy(), Ms.copy()
            F = F + kk * Ms
            if V0 is not None and w == 1:
                f, _ = nd_sort(F, None, 1)
                Fnd = F[f == 1]
                sc = Fnd.max(0) - Fnd.min(0)
                sc[sc == 0] = 1e-6
                V = V0 * sc
                active = np.unique(_argmin_rows(_angles(Fnd, V)))
                if len(active) < N:
                    tmp = (Fnd - Fnd.min(0)) / sc
                    add = tmp[rng.permutation(len(tmp))[: min(N - len(active), len(tmp))]]
                    V0 = np.vstack([V0, add])
                    V = V0 * sc
            idx = _env_selection(F, V, (w / wmax) ** 2)
            X, F, Ms = X[idx], F[idx], Ms[idx]
            if _in_rows(X, TX).all() and len(off):
                of, om = Fb[len(Fb) - len(off):], Mb[len(Mb) - len(off):]
                f, _ = nd_sort(of, None, len(of))
                s = f == 1
                X, F, Ms = np.vstack([X, off[s]]), np.vstack([F, of[s] + kk * om[s]]), np.vstack([Ms, om[s]])
            if w % int(np.ceil(wmax * 0.1)) == 0 and len(np.unique(F, axis=0)) > 2:
                if V0 is not None:
                    V = V0 * (F.max(0) - F.min(0))
                else:
                    if scale is None:
                        scale = self._scale1
                    V = V / scale
                    scale = F.max(0) - F.min(0)
                    V = V * scale
            if accumulate:
                WX.append(X); WF.append(F); WM.append(Ms)
        if accumulate:
            X, F, Ms = np.vstack(WX), np.vstack(WF), np.vstack(WM)
        F = F - kk * Ms
        u = _unique_stable(F)
        X, F = X[u], F[u]
        keep = ~_in_rows(X, TX)
        return X[keep], F[keep]

    def _stage1(self, TA):
        rng, M = self.rng, self.M
        A1F, A1X = objs(TA), decs(TA)
        u = _unique_stable(A1F)
        A1F, A1X = A1F[u], A1X[u]
        ok = np.isfinite(A1F).all(1)
        A1F, A1X = A1F[ok], A1X[ok]
        A1X = A1X[np.unique(A1X, axis=0, return_index=True)[1]]
        V0 = self.Vbb.copy()
        A2 = objs(self.pop)
        A2 = A2[_unique_stable(A2)]
        f, _ = nd_sort(A2, None, 1)
        A2 = A2[f == 1]
        zmin = A2.min(0)
        if len(A2) >= 2:
            self._scale1 = A2.max(0) - A2.min(0)
            V_1 = V0 * self._scale1
        else:
            self._scale1, V_1 = np.ones(M), V0
        ang = _angles(A2 - zmin, V_1)
        assoc = _argmin_rows(ang)
        _, first = np.unique(assoc, return_index=True)
        active = assoc[np.sort(first)]
        Va = V_1[active]
        ncl = min(5, len(Va))
        lab = kmeans(V0[active], ncl, rng)
        ids = [int(rng.choice(np.where(lab == i)[0])) for i in range(ncl) if (lab == i).any()]
        V1 = Va[ids]
        if len(V1) < 5:
            notS = np.setdiff1d(np.arange(len(V_1)), active[ids])
            V1 = np.vstack([V1, V_1[notS[rng.permutation(len(notS))[: 5 - len(V1)]]]])
        TX = decs(self.pop)
        X1, F1 = self._search(A1X, V1, A1X, TX, mating_all=False, accumulate=False)
        X, F = self._search(A1X, None, A1X, TX, mating_all=True, accumulate=True, adapt_V0=V0)
        new = kriging_select_rvmm(X, F, X1, F1, A2, rng)
        if len(new):
            new = new[~_in_rows(new, TX)]
        return new
