# emopylab 2026
"""BL-SAEA (bi-level surrogate modelling based evolutionary algorithm).

Reference:
H. Jiang, K. Qiu, Y. Tian, X. Zhang, and Y. Jin. Efficient surrogate modeling method for
evolutionary algorithm to solve bilevel optimization problems. IEEE Transactions on Cybernetics,
2024, 54(7): 4335-4347
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, nd_sort, tournament
from algorithms.community_utils.bilevel import approx_value, level_fitness, quad_approx, sqp_min
from algorithms.community_utils.dace import DaceModel
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

ALGORITHM_FLAGS = {'BLSAEA': {'bilevel', 'multi', 'real'}}


def _cos_sim(a, B):
    with np.errstate(all="ignore"):
        return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a))


class BLSAEA(LoopAlgorithm):
    """Nested bilevel GA: every upper-level candidate gets its follower from a lower-level GA refined by SQP on quadratic
    response surfaces (started near the follower of the most similar leader); a Kriging model of the leader objective
    screens the upper-level offspring (non-dominated in predicted value and uncertainty) before the expensive nested
    search, and every generation the best leader is refined by SQP on response surfaces of both levels."""

    def __init__(self, pop_size: int = 100, ll_max_gens: int = 2000, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.ll_max_gens = int(ll_max_gens)

    # ------------------------------------------------------------------------------------------ level evaluations
    def _ll(self, xu, XL):
        pr = self.problem
        XL = np.atleast_2d(XL)
        X = np.clip(np.hstack([np.tile(xu, (len(XL), 1)), XL]), pr.xl, pr.xu)
        if hasattr(pr, '_calc_f'):
            F = pr._calc_f(X)
            G = pr._calc_g(X)
        else:
            pop = self.evaluate(X)
            F = np.asarray(pop.get("F"), float)
            G = pop.get("G")
        g = None if G is None or np.shape(G)[1] == 0 else np.asarray(G, float)[:, getattr(pr, "C", 0):]
        g = None if g is None or g.shape[1] == 0 else g
        return F[:, 1], g

    def _ul(self, XU, XL):
        pr = self.problem
        pop = self.evaluate(np.hstack([np.atleast_2d(XU), np.atleast_2d(XL)]))
        F = np.asarray(pop.get("F"), float)
        G = pop.get("G")
        n_l = 0 if G is None else max(0, np.shape(G)[1] - getattr(pr, "C", 0))
        con = None if G is None or n_l == 0 else np.asarray(G, float)[:, :n_l]    # literal: first l columns
        return F[:, 0], con, pop

    def _ga(self, P, lo, up):
        rng = self.rng
        n = len(P) // 2
        A, B = P[:n], P[n:2 * n]
        d = P.shape[1]
        mu = rng.random((n, d))
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        beta = beta * (-1.0) ** rng.integers(0, 2, (n, d))
        beta[rng.random((n, d)) < 0.5] = 1
        O = np.vstack([(A + B) / 2 + beta * (A - B) / 2, (A + B) / 2 - beta * (A - B) / 2])
        L, U = np.broadcast_to(lo, O.shape), np.broadcast_to(up, O.shape)
        s, m = rng.random(O.shape) < 1 / d, rng.random(O.shape)
        O = np.minimum(np.maximum(O, L), U)
        with np.errstate(all="ignore"):
            t = s & (m <= 0.5)
            O[t] = O[t] + (U[t] - L[t]) * ((2 * m[t] + (1 - 2 * m[t]) * (1 - (O[t] - L[t]) / (U[t] - L[t])) ** 21) ** (1 / 21) - 1)
            t = s & (m > 0.5)
            O[t] = O[t] + (U[t] - L[t]) * (1 - (2 * (1 - m[t]) + 2 * (m[t] - 0.5) * (1 - (U[t] - O[t]) / (U[t] - L[t])) ** 21) ** (1 / 21))
        return O

    def _ll_search(self, xu, xl0=None, ul_pop=None, ll_pop=None):
        rng, pr = self.rng, self.problem
        DU, DL = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "DL", pr.n_var - getattr(pr, "DU", max(1, pr.n_var // 2)))
        lo, up = self.lower[DU:], self.upper[DU:]
        n = self.llN
        if xl0 is None:
            P = lo + lhs_design(rng, n, DL) * (up - lo)
        else:
            k = 13 if DL <= 3 else 23
            P = lo + lhs_design(rng, k, DL) * (up - lo)
            Y = ll_pop[int(np.nanargmax(_cos_sim(xu, ul_pop)))]
            sig = 0.2 * (up - lo)
            m = rng.random(DL) < 0.5
            Yd = Y.copy()
            Yd[m] = Yd[m] + sig[m] * rng.standard_normal(m.sum())
            P = np.vstack([P, np.clip(Yd, lo, up), Y])
            for _ in range(14 if DL <= 3 else 24):
                m = rng.random(DL) < 0.5
                Yd = np.array(xl0, float).copy()
                Yd[m] = Yd[m] + sig[m] * rng.standard_normal(m.sum())
                P = np.vstack([P, np.clip(Yd, lo, up)])
            P = np.vstack([P, np.clip(xl0, lo, up)])
        obj, con = self._ll(xu, P)
        fit = level_fitness(obj, con)
        hist = [obj[int(np.argmin(fit))]]
        v0 = P.var(0, ddof=1)
        alpha, gen = 1.0, 0
        while alpha > 1e-5 and gen <= self.ll_max_gens:
            mate = tournament(2, n, fit, rng=rng)
            off = self._ga(P[mate], lo, up)
            oo, oc = self._ll(xu, off)
            of = level_fitness(oo, oc)
            P, obj, fit = np.vstack([P, off]), np.concatenate([obj, oo]), np.concatenate([fit, of])
            r = np.argsort(fit, kind="stable")[:n]
            P, obj, fit = P[r], obj[r], fit[r]
            try:
                fm = quad_approx(obj, P)
                cm = [quad_approx(oc[:, i], P) for i in range(oc.shape[1])] if oc is not None and len(oc) == len(P) else []
                x, _ = sqp_min(lambda z: approx_value(z, fm), P[0], lo, up, cm)
                lo_, lc_ = self._ll(xu, x)
                lf = level_fitness(lo_, lc_)[0]
                if fit[-1] > lf:
                    P[-1], obj[-1], fit[-1] = x, lo_[0], lf
            except (ValueError, np.linalg.LinAlgError):
                pass
            with np.errstate(all="ignore"):
                alpha = min(1.0, float(np.nansum(P.var(0, ddof=1) / v0)))
            gen += 1
            hist.append(obj[int(np.argmin(fit))])
            if gen > 9 and abs(hist[-1] - hist[-6]) < 1e-5:
                break
        k = np.where(np.abs(fit - fit.min()) < 1e-9)[0]
        A = [self._ul(xu[None], P[[i]])[0][0] for i in k]
        b = k[int(np.argmin(A))]
        bo, bc = self._ll(xu, P[[b]])
        return P[b], obj[b], (np.zeros(1) if bc is None else bc[0])

    def _ul_fit(self, obj, ucon, lcon):
        if ucon is None and np.all(lcon == 0):
            return obj.copy()
        pc = (np.maximum(0, ucon).sum(1) if ucon is not None else 0) + np.maximum(0, lcon).sum(1)
        return np.where(pc <= 0, obj, pc + 1e10)

    # ------------------------------------------------------------------------------------------ main loop
    def _initialize_infill(self):
        pr, rng = self.problem, self.rng
        self.ulN = self.N // 2
        self.llN = self.N - self.ulN
        DU = getattr(pr, "DU", max(1, pr.n_var // 2))
        lo, up = self.lower[:DU], self.upper[:DU]
        self.ulP = lo + lhs_design(rng, self.ulN, DU) * (up - lo)
        res = [self._ll_search(self.ulP[i]) for i in range(self.ulN)]
        self.llP = np.array([r[0] for r in res])
        self.llF = np.array([r[1] for r in res])
        self.lcon = np.array([np.atleast_1d(r[2]) for r in res])
        obj, ucon, pop = self._ul(self.ulP, self.llP)
        self.ulO = obj
        self.ulFit = self._ul_fit(obj, ucon, self.lcon)
        self.Axu, self.Axl, self.AF = self.ulP.copy(), self.llP.copy(), obj.copy()
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.evaluate(np.hstack([self.ulP, self.llP]))
        self._set_optimum()

    def _krg(self, X, Y):
        return DaceModel(X, Y, "regpoly0", np.full(X.shape[1], len(X) ** (-1.0 / X.shape[1])))

    def step(self):
        rng, pr = self.rng, self.problem
        DU, DL = getattr(pr, "DU", max(1, pr.n_var // 2)), getattr(pr, "DL", pr.n_var - getattr(pr, "DU", max(1, pr.n_var // 2)))
        lo, up = self.lower[:DU], self.upper[:DU]
        off = np.vstack([self._ga(self.ulP[tournament(2, self.ulN, self.ulFit, rng=rng)], lo, up),
                         self._ga(self.ulP[tournament(2, self.ulN, self.ulFit, rng=rng)], lo, up)])
        X, Y = self.Axu, self.AF
        surr = self._krg(X, Y)
        pm, pv = surr.predict(off, mse=True)
        fr, _ = nd_sort(np.column_stack([pm, pv]), None, 2 * self.ulN)
        num = np.where(fr == 1)[0]
        nd_x, nd_f, nd_c = [], [], []
        xl0 = None
        for j in num:
            # literal: the "lower-level" models are trained on the upper-level archive objective
            xl0 = np.full(DL, float(self._krg(X, Y).predict(off[[j]])[0]))
            bx, bf, bc = self._ll_search(off[j], xl0, self.ulP, self.llP)
            nd_x.append(bx); nd_f.append(bf); nd_c.append(np.atleast_1d(bc))
        if len(num):
            nd_x, nd_f, nd_c = np.array(nd_x), np.array(nd_f), np.array(nd_c)
            o, uc, _ = self._ul(off[num], nd_x)
            fit = self._ul_fit(o, uc, nd_c)
            self.Axu = np.vstack([self.Axu, off[num]])
            self.Axl = np.vstack([self.Axl, nd_x])
            self.AF = np.concatenate([self.AF, o])
            first = np.unique(np.round(self.Axu / 1e9) * 1e9, axis=0, return_index=True)[1]   # round(.,-9)
            self.Axu, self.Axl, self.AF = self.Axu[first], self.Axl[first], self.AF[first]
            if len(self.AF) > 900:
                r = np.argsort(self.AF, kind="stable")[:900]
                self.Axu, self.Axl, self.AF = self.Axu[r], self.Axl[r], self.AF[r]
            self.ulP = np.vstack([self.ulP, off[num]])
            self.llP = np.vstack([self.llP, nd_x])
            self.ulO = np.concatenate([self.ulO, o])
            self.ulFit = np.concatenate([self.ulFit, fit])
            self.llF = np.concatenate([self.llF, nd_f])
        r = np.argsort(self.ulFit, kind="stable")[: self.ulN]
        self.ulP, self.llP, self.ulO, self.ulFit, self.llF = self.ulP[r], self.llP[r], self.ulO[r], self.ulFit[r], self.llF[r]
        best = self.ulP[0]
        n_xl0 = xl0 if xl0 is not None else np.full(DL, float(self._krg(X, Y).predict(best[None])[0]))
        self.pop = self.evaluate(np.hstack([self.ulP, self.llP]))
        F = np.asarray(self.pop.get("F"), float)
        G = self.pop.get("G")
        lcons = []
        for i in range(len(self.ulP)):
            _, g = self._ll(self.ulP[i], self.llP[[i]])
            lcons.append(0 if g is None else g.shape[1])
        l = lcons[0] if lcons else 0
        ulcon = None if G is None or l == 0 else np.asarray(G, float)[:, :l]
        try:
            ul_m = quad_approx(F[:, 0], self.ulP)
            con_m = [quad_approx(ulcon[:, i], self.ulP) for i in range(ulcon.shape[1])] if ulcon is not None else []
        except ValueError:
            ul_m, con_m = None, []
        llo_all = np.array([self._ll(self.ulP[i], self.llP[[i]])[0][0] for i in range(len(self.ulP))])
        try:
            ll_m = quad_approx(llo_all, self.llP)
        except ValueError:
            ll_m = None

        def nested(xa):
            if ll_m is None:
                xl = n_xl0
            else:
                xl, _ = sqp_min(lambda z: approx_value(z, ll_m), n_xl0, self.lower[DU:], self.upper[DU:])
            X_ = np.clip(np.concatenate([xa, xl])[None], pr.xl, pr.xu)
            return float(pr._calc_f(X_)[0, 0])                 # CalObj: not charged

        x, _ = sqp_min(nested, best, lo, up, con_m)
        bx, bf, bc = self._ll_search(x)
        o, uc, _ = self._ul(x[None], bx[None])
        f = self._ul_fit(o, uc, np.atleast_2d(bc))
        self.ulP[-1], self.ulO[-1], self.llP[-1], self.llF[-1], self.ulFit[-1] = x, o[0], bx, bf, f[0]
