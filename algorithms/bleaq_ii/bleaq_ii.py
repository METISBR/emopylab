# emopylab 2026
"""BLEAQ-II (bilevel evolutionary algorithm based on quadratic approximations II).

Reference:
A. Sinha, Z. Lu, K. Deb, and P. Malo. Bilevel optimization based on iterative approximation of
mappings. Journal of Heuristics, 2020, 26: 151-185.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from algorithms.community_utils.base import LoopAlgorithm, kmeans, tournament
from algorithms.community_utils.bilevel import approx_value, quad_approx
from core.population import Population

ALGORITHM_FLAGS = {'BLEAQII': {'bilevel', 'constrained', 'multi', 'real'}}


def pcx_bleaq(parent, lower, upper, rng, proC=0.9, proM=1.0, disM=20.0):
    """Parent-centric recombination (step scaled by the mean deviation from the centroid) + polynomial mutation."""
    N, D = parent.shape
    W = parent.mean(axis=0)
    p1 = np.r_[np.arange(1, N), 0]
    p2 = np.r_[np.arange(2, N), 0, 1]
    off = parent + 0.1 * (parent - W) + np.mean(np.abs(parent - W), axis=1)[:, None] * (parent[p2] - parent[p1]) / 2
    keep = np.repeat(rng.random((N, 1)) > proC, D, axis=1)
    off[keep] = parent[keep]
    Lo, Up = np.tile(lower, (N, 1)), np.tile(upper, (N, 1))
    site = rng.random((N, D)) < proM / D
    mu = rng.random((N, D))
    off = np.minimum(np.maximum(off, Lo), Up)
    span = Up - Lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (off[t] - Lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (Up[t] - off[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return off


def _fitness(C, F, G):
    """Lower-level solutions (NaN upper objective) are ranked by the follower objective and constraints, the others
    by the leader's; infeasible ones by total violation + 1e10."""
    F = np.atleast_2d(F)
    G = np.zeros((len(F), 0)) if G is None else np.atleast_2d(G)
    if np.any(np.isnan(F[:, 0])):
        obj, con = F[:, 1], G[:, C:]
    else:
        obj, con = F[:, 0], G[:, :C]
    cv = np.zeros(len(obj)) if con.shape[1] == 0 else np.maximum(0, con).sum(1)
    feas = cv <= 0
    return np.where(feas, obj, cv + 1e10)


def _fmincon(fun, x0, lb, ub, cfun=None):
    """SQP with bounds and nonlinear inequality constraints c(x) <= 0."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    cons = [] if cfun is None else [{"type": "ineq", "fun": lambda x: -np.atleast_1d(cfun(x))}]
    with np.errstate(all="ignore"):
        res = minimize(fun, np.clip(np.asarray(x0, float), lb, ub), method="SLSQP", bounds=list(zip(lb, ub)),
                       constraints=cons, options={"maxiter": 400})
    return np.clip(res.x, lb, ub)


def _models(Y, X):
    """One quadratic approximation per column of ``Y`` (empty list without columns)."""
    return [quad_approx(Y[:, i], X) for i in range(Y.shape[1])]


class _Set:
    """Decision vectors with their objectives and constraints (a light population container)."""

    def __init__(self, X, F, G):
        self.X, self.F, self.G = np.atleast_2d(X), np.atleast_2d(F), np.atleast_2d(G)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return _Set(self.X[i], self.F[i], self.G[i])

    def cat(self, other):
        if other is None or len(other) == 0:
            return self
        return _Set(np.vstack([self.X, other.X]), np.vstack([self.F, other.F]), np.vstack([self.G, other.G]))


class BLEAQII(LoopAlgorithm):
    """Bilevel EA with quadratic approximations of the lower-level reaction (psi) and optimal-value (phi) mappings:
    offspring leaders get their followers from the mappings once the archive of reliably solved lower levels is large
    enough (otherwise from a lower-level search that first tries SQP on a quadratic model and falls back to a
    PCX-based GA), and every generation the best leader is refined by a local search on the mappings."""

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)

    # ------------------------------------------------------------------------------------------ evaluations
    def _ncon(self):
        return int(getattr(self.problem, "n_ieq_constr", 0) or 0)

    def _wrap(self, pop):
        X = np.asarray(pop.get("X"), float)
        F = np.asarray(pop.get("F"), float)
        G = pop.get("G")
        G = np.zeros((len(X), 0)) if G is None or np.size(G) == 0 or self._ncon() == 0 else np.asarray(G, float)
        return _Set(X, F, G)

    def _eval(self, X):
        return self._wrap(self.evaluate(np.atleast_2d(X)))

    def _eval_lower(self, X):
        if hasattr(self.problem, 'evaluation_lower'):
            return self._wrap(self.problem.evaluation_lower(np.atleast_2d(X)))
        return self._wrap(self.evaluate(np.atleast_2d(X)))

    def _pop(self, S):
        pop = Population.new("X", S.X, "F", S.F)
        if S.G.shape[1]:
            pop.set("G", S.G)
        return pop

    def _env_sel(self, P, O):
        sel = self.rng.permutation(len(P))[:2]
        pool = P[sel].cat(O)
        rank = np.argsort(_fitness(getattr(self.problem, "C", 0), pool.F, pool.G), kind="stable")[: len(sel)]
        P.X[sel], P.F[sel], P.G[sel] = pool.X[rank], pool.F[rank], pool.G[rank]
        return P

    def _archive_update(self, A, new):
        DU = getattr(self.problem, "DU", max(1, self.problem.n_var // 2))
        size = ((DU + 1) * (DU + 2) / 2 + 3 * DU) * 10
        A = new if A is None else A.cat(new)
        if len(A) > size:                                  # literal: only the oldest member leaves
            A = A[np.arange(1, len(A))]
        return A

    # ------------------------------------------------------------------------------------------ lower-level search
    def _ll_search(self, xu, member=None, ll_pop=None):
        pr, rng, N = self.problem, self.rng, self.N
        DU, DL, C = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "DL", pr.n_var - getattr(pr, "DU", max(1, pr.n_var // 2))), getattr(pr, "C", 0)
        lo, up = self.lower[DU:], self.upper[DU:]
        tag = 0
        n = int((DL + 1) * (DL + 2) / 2 + DL)
        P = lo + rng.random((n, DL)) * (up - lo)
        if member is not None:
            P[-1] = member
        else:
            member = P[-1].copy()
        S = self._eval_lower(np.hstack([np.tile(xu, (n, 1)), P]))
        obj, con = S.F[:, 1], S.G[:, C:]
        try:
            fm = quad_approx(obj, P)
            cm = _models(con, P)
            xl = _fmincon(lambda x: approx_value(x, fm), member, lo, up,
                          (lambda x: np.array([approx_value(x, c) for c in cm])) if cm else None)
            E = self._eval_lower(np.concatenate([xu, xl]))
            f = approx_value(E.X[0, DU:], fm)
            c = np.array([approx_value(xl, m) for m in cm])
            d = np.sqrt((f - E.F[0, 1]) ** 2 + np.sum((c - E.G[0, C:]) ** 2))
            if d < 1e-6:
                return E.X[0, DU:].copy(), 1
        except (ValueError, np.linalg.LinAlgError):
            xl = np.asarray(member, float)                 # the quadratic step could not be built
        if ll_pop is not None:
            if len(ll_pop) > N:
                P = ll_pop[rng.permutation(len(ll_pop))[:N]]
            else:
                P = lo + rng.random((N, DL)) * (up - lo)
                P[: len(ll_pop)] = ll_pop
        else:
            P = np.vstack([lo + rng.random((N, DL)) * (up - lo), xl])
        L = self._eval_lower(np.hstack([np.tile(xu, (len(P), 1)), P]))
        fe = 0
        alpha0 = np.sum(P.var(0, ddof=1))
        while fe < getattr(pr, "maxFElower", 1000):
            pool = tournament(2, 3, _fitness(C, L.F, L.G), rng=rng)
            off = pcx_bleaq(L.X[pool][:, DU:], lo, up, rng)
            O = self._eval_lower(np.hstack([np.tile(xu, (len(off), 1)), off]))
            fe += len(O)
            L = self._env_sel(L, O)
            with np.errstate(all="ignore"):
                alpha = np.sum(L.X[:, DU:].var(0, ddof=1)) / alpha0
            if alpha < 1e-4:
                tag = 1
                break
        best = int(np.argmin(_fitness(C, L.F, L.G)))
        return L.X[best, DU:].copy(), tag

    # ------------------------------------------------------------------------------------------ mappings
    def _mappings(self, indv, A):
        pr, rng = self.problem, self.rng
        DU, DL = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "DL", pr.n_var - getattr(pr, "DU", max(1, pr.n_var // 2)))
        n_eval = DU
        n_min = int((DU + 1) * (DU + 2) / 2 + 2 * DU) + n_eval
        up, lw = A.X[:, :DU], A.X[:, DU:]
        n = len(A)
        # literal: the distance matrix is n-by-n and only its first entry is filled, so the column-wise sort yields
        # the order of [d1, 0, ..., 0] followed by the identity order of the all-zero columns
        col = np.zeros(n)
        col[0] = np.sum((indv - up[0]) ** 2)
        I = np.concatenate([np.argsort(col, kind="stable"), np.tile(np.arange(n), n - 1)])[:n_min]
        permut = rng.permutation(len(I))
        members = I[permut[: len(I) - n_eval]]
        rest = np.setdiff1d(I, members)
        psi = [quad_approx(lw[members, j], up[members]) for j in range(DL)]
        pred = np.array([[approx_value(up[k], m) for m in psi] for k in rest]).reshape(len(rest), DL)
        with np.errstate(all="ignore"):
            psi_map = dict(function=psi, sumMSE=sum(m["mseNorm"] for m in psi),
                           validMSE=float(np.mean((pred - lw[rest]) ** 2)) if len(rest) else np.nan)
        llv = A.F[:, 1]
        phi = quad_approx(llv[members], up[members])
        predp = np.array([approx_value(up[k], phi) for k in rest])
        with np.errstate(all="ignore"):
            phi_map = dict(function=phi, sumMSE=phi["mseNorm"],
                           validMSE=float(np.mean((predp - llv[rest]) ** 2)) if len(rest) else np.nan)
        lies = int(not np.any((indv >= up[members].max(0)) | (indv <= up[members].min(0))))
        return psi_map, phi_map, lies

    def _phi_models(self, Z, F, G, C):
        return dict(function=quad_approx(F[:, 0], Z), ineq=_models(G[:, :C], Z), llFunction=quad_approx(F[:, 1], Z),
                    llIneq=_models(G[:, C:], Z))

    def _phi_cons(self, pop, ap, phi, DU):
        c = [approx_value(pop, m) for m in ap["ineq"]] + [approx_value(pop, m) for m in ap["llIneq"]]
        c.append(approx_value(pop[:DU], phi) - approx_value(pop, ap["llFunction"]))
        return np.array(c)

    def _ll_from_mapping(self, off, psi_map, phi_map, A1, A0):
        pr, rng = self.problem, self.rng
        DU, DL, C = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "DL", pr.n_var - getattr(pr, "DU", max(1, pr.n_var // 2))), getattr(pr, "C", 0)
        min_phi = int((DU + DL + 1) * (DU + DL + 2) / 2 + 3 * DU)
        A = A1.cat(A0)
        run_phi = False
        if len(A) >= min_phi:
            run_phi = ((psi_map["sumMSE"] >= phi_map["sumMSE"]) and (psi_map["validMSE"] >= phi_map["validMSE"])) \
                or (rng.random() < 0.25)
        xl = np.array([approx_value(off, m) for m in psi_map["function"]])
        valid = psi_map["validMSE"]
        if run_phi:
            d = np.sum((off - A.X[:, :DU]) ** 2, axis=1)
            I = np.argsort(d, kind="stable")[:min_phi]
            Z = A.X[I]
            ap = self._phi_models(Z, A.F[I], A.G[I], C)
            lb, ub = Z[:, DU:].min(0), Z[:, DU:].max(0)
            xl = _fmincon(lambda z: -approx_value(np.concatenate([off, z]), ap["function"]), xl, lb, ub,
                          lambda z: self._phi_cons(np.concatenate([off, z]), ap, phi_map["function"], DU))
            valid = phi_map["validMSE"]
        return xl, valid

    # ------------------------------------------------------------------------------------------ local search
    def _local_search(self, init, A1, A0, stopping, gen):
        pr, rng = self.problem, self.rng
        DU, DL, C = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "DL", pr.n_var - getattr(pr, "DU", max(1, pr.n_var // 2))), getattr(pr, "C", 0)
        min_psi = int((DU + 1) * (DU + 2) / 2 + 2 * DU) + DU
        min_phi = int((DU + DL + 1) * (DU + DL + 2) / 2 + 3 * DU)
        term = (self.FE > self.max_FE) or stopping == 1
        freq = self.N
        ls = 0
        if len(A1) >= min_psi:
            if term:
                ls = 1
            elif (gen + freq - 1) % freq == 0:
                ls = 2
        xu, xl = init[:DU].copy(), init[DU:].copy()
        ulmin, ulmax, llmin, llmax = self.lower[:DU], self.upper[:DU], self.lower[DU:], self.upper[DU:]
        dont_phi = len(A1) + len(A0) < min_phi
        psi_map, phi_map, _ = self._mappings(xu, A1)
        psi_f, phi_f = psi_map["function"], phi_map["function"]
        if (psi_map["validMSE"] <= phi_map["validMSE"]) or dont_phi or (rng.random() <= 0.5):
            method = 1 if ls == 1 else 2
        else:
            method = 3 if ls == 1 else 4
        A = A1.cat(A0)
        order = np.argsort(np.sum((xu - A.X[:, :DU]) ** 2, axis=1), kind="stable")

        def bound(v, lo, hi):
            diff = 0.05 * (hi - lo)
            return np.clip(v - diff, lo, hi), np.clip(v + diff, lo, hi)

        def psi_x(x):
            return np.clip(np.array([approx_value(x, m) for m in psi_f]), llmin, llmax)

        if method == 1:                                    # exact leader objective on the psi mapping (charged)
            lb, ub = bound(xu, ulmin, ulmax)
            xu = _fmincon(lambda x: -self._eval(np.concatenate([x, psi_x(x)])).F[0, 0], xu, lb, ub,
                          (lambda x: self._eval(np.concatenate([x, psi_x(x)])).G[0, :C]) if C else None)
            return xu, xl
        if method == 2:                                    # approximated leader objective in the leader space
            I = order[: min(min_psi, len(order))]
            U = A.X[I, :DU]
            fm = quad_approx(A.F[I, 0], U)
            cm = _models(A.G[I, :C], U)
            xu = _fmincon(lambda x: -approx_value(x, fm), xu, ulmin, ulmax,
                          (lambda x: np.array([approx_value(x, m) for m in cm])) if cm else None)
            return xu, xl
        if method == 3:                                    # exact leader objective, phi constraint (charged)
            lb, ub = bound(np.concatenate([xu, xl]), np.concatenate([ulmin, llmin]), np.concatenate([ulmax, llmax]))

            def cons3(z):
                S = self._eval(z)
                return np.concatenate([S.G[0], [approx_value(z[:DU], phi_f) - S.F[0, 1]]])

            z = _fmincon(lambda z: -self._eval(z).F[0, 0], np.concatenate([xu, xl]), lb, ub, cons3)
            return z[:DU], z[DU:]
        I = order[:min_phi]                                # method 4: everything approximated
        Z = A.X[I]
        ap = self._phi_models(Z, A.F[I], A.G[I], C)
        z = _fmincon(lambda z: -approx_value(z, ap["function"]), np.concatenate([xu, xl]),
                     np.concatenate([ulmin, llmin]), np.concatenate([ulmax, llmax]),
                     lambda z: self._phi_cons(z, ap, phi_f, DU))
        return z[:DU], z[DU:]

    # ------------------------------------------------------------------------------------------ main loop
    def _initialize_infill(self):
        pr, rng, N = self.problem, self.rng, self.N
        DU = getattr(pr, "DU", max(1, pr.n_var // 2))
        lo, up = self.lower[:DU], self.upper[:DU]
        ul = lo + rng.random((N, DU)) * (up - lo)
        res = [self._ll_search(ul[i]) for i in range(N)]
        ll = np.array([r[0] for r in res])
        k = int(max(4, N / 10))
        lab = kmeans(ll, k, rng)
        cent = np.array([ll[lab == j].mean(0) for j in range(lab.max() + 1) if np.any(lab == j)])
        res = [self._ll_search(ul[i], None, cent) for i in range(N)]
        ll = np.array([r[0] for r in res])
        self.tag_ul = np.array([r[1] for r in res])
        self.ul0 = ul
        S = self._eval(np.hstack([ul, ll]))
        self.S = S
        return self._pop(S)

    def _initialize_advance(self, infills=None, **kwargs):
        S, t = self.S, self.tag_ul == 1
        self.A1, self.A0 = self._archive_update(None, S), None
        with np.errstate(all="ignore"):
            self.alpha0 = np.sum(np.hstack([self.ul0[t], S.X[t, getattr(self.problem, "DU", max(1, self.problem.n_var // 2)):]]).var(0, ddof=1)) / self.D \
                if t.sum() > 1 else (0.0 if t.sum() == 1 else np.nan)
        self.stopping, self.gen = 0, 0
        self.pop = self._pop(S)
        self._set_optimum()

    def step(self):
        pr, rng, N = self.problem, self.rng, self.N
        DU, C = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "C", 0)
        self.gen += 1
        S = self.S
        pool = tournament(2, 3, _fitness(C, S.F, S.G), rng=rng)
        PD = S.X[pool]
        ul = pcx_bleaq(PD[:, :DU], self.lower[:DU], self.upper[:DU], rng)
        ll, tags = [], []
        closest = np.argmin(((ul[:, None, :] - PD[None, :, :DU]) ** 2).sum(-1), axis=1)
        n_arch = (DU + 1) * (DU + 2) / 2 + 3 * DU
        for i in range(len(ul)):
            if (np.sum(self.tag_ul == 1) < N / 2) or (len(self.A1) < n_arch):
                x, t = self._ll_search(ul[i], PD[closest[i], DU:])
            else:
                psi_map, phi_map, lies = self._mappings(ul[i], self.A1)
                x, valid = self._ll_from_mapping(ul[i], psi_map, phi_map, self.A1, self.A0)
                t = int(lies == 1 and valid < 1e-4)
            ll.append(x)
            tags.append(t)
        tags = np.array(tags)
        O = self._eval(np.hstack([ul, np.array(ll)]))
        S = self._env_sel(S, O)
        if np.any(tags == 1):
            self.A1 = self._archive_update(self.A1, O[np.where(tags == 1)[0]])
        if np.any(tags == 0):
            self.A0 = self._archive_update(self.A0, O[np.where(tags == 0)[0]])
        fit = _fitness(C, S.F, S.G)
        t = self.tag_ul == 1
        idx = int(np.argmin(fit)) if not t.any() else int(np.where(t)[0][int(np.argmin(fit[t]))])
        init = S[[idx]]
        with np.errstate(all="ignore"):
            # literal: the leader part comes from the initial population, the follower part from the current one
            a = np.sum(np.hstack([self.ul0[t], S.X[t, DU:]]).var(0, ddof=1)) / self.D if t.sum() > 1 else \
                (0.0 if t.sum() == 1 else np.nan)
            if np.float64(a) / np.float64(self.alpha0) < 1e-4:
                self.stopping = 1
        A0 = self.A0 if self.A0 is not None else _Set(np.zeros((0, self.D)), np.zeros((0, 2)), np.zeros((0, S.G.shape[1])))
        try:
            xu, xl = self._local_search(init.X[0], self.A1, A0, self.stopping, self.gen)
        except (ValueError, np.linalg.LinAlgError):
            xu, xl = init.X[0, :DU].copy(), init.X[0, DU:].copy()
        xl, _ = self._ll_search(xu, xl)
        E = self._eval(np.concatenate([xu, xl]))
        if _fitness(C, E.F, E.G)[0] > _fitness(C, init.F, init.G)[0]:
            E = init
        self.S = self._env_sel(S, E)
        self.pop = self._pop(self.S)
