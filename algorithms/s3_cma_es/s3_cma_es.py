# emopylab 2026
"""S3-CMA-ES (scalable small subpopulations based covariance matrix adaptation).

Reference:
H. Chen, R. Cheng, J. Wen, H. Li, and J. Weng. Solving large-scale many-objective optimization
problems by covariance matrix adaptation evolution strategy with scalable small subpopulations.
Information Sciences, 2020, 509: 457-469.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, nd_sort, objs
from core.population import Population

ALGORITHM_FLAGS = {'S3CMAES': {'integer', 'large', 'many', 'multi', 'real'}}

_EPS = np.finfo(float).eps


def _minkowski(A, B, p):
    return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(axis=2) ** (1.0 / p)


def _update_archive(N, pop):
    front, _ = nd_sort(objs(pop), None, 1)
    arc = pop[front == 1]
    if len(arc) > N:
        F = objs(arc)
        choose = np.zeros(len(F), bool)
        with np.errstate(all="ignore"):
            cos = 1 - (F @ np.eye(F.shape[1]).T) / (np.linalg.norm(F, axis=1)[:, None])
        choose[np.argmin(np.where(np.isnan(cos), np.inf, cos), axis=0)] = True
        lp = _minkowski(F, F, 0.5)
        while choose.sum() < N:
            remain = np.where(~choose)[0]
            rho = int(np.argmax(np.min(lp[np.ix_(remain, np.where(choose)[0])], axis=1)))
            choose[remain[rho]] = True
        arc = arc[choose]
    return arc


def _cma_params(N, lam, xmean, sigma):
    mu = lam / 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, int(mu) + 1))
    mueff = w.sum() ** 2 / np.sum(w ** 2)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cs = (mueff + 2) / (N + mueff + 5)
    return dict(N=N, lam=lam, mu=int(mu), weights=w / w.sum(), mueff=mueff,
                cc=(4 + mueff / N) / (N + 4 + 2 * mueff / N), cs=cs, c1=c1,
                cmu=min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff)),
                damps=1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs,
                chiN=N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2)),
                xmean=xmean.copy(), sigma=sigma, pc=np.zeros(N), ps=np.zeros(N), B=np.eye(N), D=np.ones(N),
                C=np.eye(N), eigeneval=0, counteval=0)


def _copy_params(p):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.items()}


class S3CMAES(LoopAlgorithm):
    """Variables are split into position-related ones (few, decided by how many fronts a sweep of one variable creates) and
    distance-related ones, grouped by an interaction analysis; several small CMA-ES populations (one per position vector)
    optimise the distance groups against the squared objective norm, and after the first stage the position variables are
    diversified by DE on an archive and CMA-ES restarts from every archive member."""

    def _initialize_infill(self):
        return self._prepare()

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.archive
        self._set_optimum()

    # -- variable analysis ---------------------------------------------------------------------------------------
    def _control_variable_analysis(self, nca=50):
        D, M = self.D, self.M
        fno = np.zeros(D)
        span = self.upper - self.lower
        tempA = np.array([0.05 + k * (0.95 - 0.05) / (nca - 1) for k in range(nca)])
        for i in range(D):
            x = 0.2 * np.ones(D) * span + self.lower
            S = np.tile(x, (nca, 1))
            S[:, i] = tempA * span[i] + self.lower[i]
            pop = self.evaluate(S)
            fno[i] = nd_sort(objs(pop), None, np.inf)[1]
        I = np.argsort(fno, kind="stable")
        return np.sort(I[: M - 1]), np.sort(I[M - 1:])

    def _group_dv(self, DV, PV, n_per_group):
        M = self.M
        ub, lb = self.upper[DV], self.lower[DV]
        dim = len(DV)
        fix_pv = (self.lower[PV] + self.upper[PV]) / 2
        mean_dv = (ub + lb) / 2
        fhat_x = lb[None, :] + np.eye(dim) * mean_dv[None, :]
        fhat = objs(self.evaluate(np.hstack([np.tile(fix_pv, (dim, 1)), fhat_x])))
        fp1 = objs(self.evaluate(np.concatenate([fix_pv, lb])[None, :]))[0]
        lam = np.zeros((M, dim, dim))
        f4 = np.zeros((M, dim, dim))
        for i in range(dim - 1):
            fp2 = fhat[i]
            for j in range(i + 1, dim):
                p4 = lb.copy()
                p4[i], p4[j] = mean_dv[i], mean_dv[j]
                fp4 = objs(self.evaluate(np.concatenate([fix_pv, p4])[None, :]))[0]
                f4[:, i, j] = fp4
                lam[:, i, j] = np.abs((fp2 - fp1) - (fp4 - fhat[j]))
        big = np.zeros((dim, dim))
        mu_m = _EPS / 2
        n = np.sqrt(dim)
        gamma = n * mu_m / (1 - n * mu_m)
        for m in range(M):
            fa = f4[m] + f4[m].T
            F1 = np.full((dim, dim), fp1[m])
            F2 = np.tile(fhat[:, m][None, :], (dim, 1))
            F3 = np.tile(fhat[:, m][:, None], (1, dim))
            Fmax = np.max(np.stack([F1, F2, F3, fa]), axis=0)
            big += (lam[m] >= gamma * Fmax)
        big = big > 0
        labels = np.zeros(dim, dtype=int)
        ccc = 0
        while np.any(labels == 0):
            fue = int(np.where(labels == 0)[0][0])
            ccc += 1
            labels[fue] = ccc
            lst = [fue]
            while lst:
                new = []
                for p in lst:
                    cp = np.where(big[p])[0]
                    cp1 = cp[labels[cp] == 0]
                    labels[cp1] = ccc
                    new.extend(cp1.tolist())
                lst = new
        comps = [np.where(labels == k)[0] for k in range(1, ccc + 1)]
        seps = np.concatenate([c for c in comps if len(c) == 1]) if any(len(c) == 1 for c in comps) else np.zeros(0, int)
        groups = [DV[c] for c in comps if len(c) > 1]
        for g in range(0, len(seps), n_per_group):
            groups.append(DV[seps[g: g + n_per_group]])
        return groups

    def _load_params(self, groups, pop_size):
        rng = self.rng
        out = []
        for idx in groups:
            N = len(idx)
            xmean = self.lower[idx] + (self.upper[idx] - self.lower[idx]) * rng.random(N)
            out.append(_cma_params(N, pop_size, xmean, 0.5))
        return out

    def _prepare(self):
        rng, D = self.rng, self.D
        PV, DV = self._control_variable_analysis(50)
        self.PV, self.DV = PV, DV
        self.groups = self._group_dv(DV, PV, 35)
        popN = 5
        V = 0.05 + 0.9 * rng.random((popN, len(PV)))
        V = self.lower[PV] + V * (self.upper[PV] - self.lower[PV])
        self.pop_size_cma = 6 + int(np.floor(3 * np.log(35)))
        one = self._load_params(self.groups, self.pop_size_cma)
        self.cma = [[_copy_params(p) for p in one] for _ in range(popN)]
        self.big = []
        for i in range(popN):
            t = np.zeros((self.pop_size_cma, D))
            t[:, PV] = V[i]
            t[:, DV] = np.tile(self.lower[DV] + (self.upper[DV] - self.lower[DV]) * rng.random(len(DV)), (self.pop_size_cma, 1))
            self.big.append(t)
        self.archive = self.evaluate(self.big[0])
        self.last_best = np.zeros(popN)
        self.stop = np.zeros(popN, bool)
        self.first_tag = True
        self.converged = None
        return self.archive

    # -- CMA-ES generation on one group ----------------------------------------------------------------------------
    def _operator(self, para, bestmem, dim_index):
        rng = self.rng
        N, lam, mu = para["N"], para["lam"], para["mu"]
        xmean, sigma, w = para["xmean"], para["sigma"], para["weights"]
        mueff, cc, cs, c1, cmu, damps, chiN = (para[k] for k in ("mueff", "cc", "cs", "c1", "cmu", "damps", "chiN"))
        pc, ps, B, D, C = para["pc"], para["ps"], para["B"], para["D"], para["C"]
        invsqrt = B @ np.diag(1.0 / D) @ B.T
        counteval, eigeneval = para["counteval"], para["eigeneval"]
        arx = xmean[:, None] + sigma * (B @ (D[:, None] * rng.standard_normal((N, lam))))
        arx = np.clip(arx, 0, 10)
        X = np.tile(bestmem, (lam, 1))
        X[:, dim_index] = arx.T
        pop = self.evaluate(X)
        fit = np.sum(objs(pop) ** 2, axis=1)
        counteval += lam
        order = np.argsort(fit, kind="stable")
        xold = xmean
        xmean = arx[:, order[:mu]] @ w
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrt @ (xmean - xold) / sigma
        hsig = float(np.sum(ps ** 2) / (1 - (1 - cs) ** (2 * counteval / lam)) / N < 2 + 4 / (N + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        artmp = (arx[:, order[:mu]] - xold[:, None]) / sigma
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp @ np.diag(w) @ artmp.T
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if counteval - eigeneval > lam / (c1 + cmu) / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            if np.any(np.isnan(C)) or np.any(np.isinf(C)):
                C = B @ np.diag(D ** 2) @ B.T
                C = np.triu(C) + np.triu(C, 1).T
            ev, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(ev, 0))
        para.update(xmean=xmean, sigma=sigma, pc=pc, ps=ps, B=B, D=D, C=C, eigeneval=eigeneval, counteval=counteval)
        return arx.T, float(fit[order[0]]), pop[[order[0]]]

    def _generate_big(self, archive):
        cdec = decs(archive)
        popsize = 16
        big, cma = [], []
        for ci in range(len(cdec)):
            t = np.tile(cdec[ci], (popsize, 1))
            big.append(t)
            cma.append([_cma_params(len(idx), popsize, t[:, idx].mean(axis=0), 0.1) for idx in self.groups])
        return big, cma

    def step(self):
        rng, D, N = self.rng, self.D, self.N
        PV = self.PV
        arch = self.converged
        new = []
        popN = len(self.big)
        for p in range(popN):
            if self.stop[p]:
                continue
            t = self.big[p]
            pvd = t[:, PV]
            bestmem = t[0].copy()
            t = np.zeros((self.cma[p][0]["lam"], D))
            t[:, PV] = pvd
            best_ind, best_val = None, None
            for g, idx in enumerate(self.groups):
                pop_vals, best_val, best_ind = self._operator(self.cma[p][g], bestmem, idx)
                t[:, idx] = pop_vals
            new.append(best_ind)
            self.big[p] = t
            if abs(self.last_best[p] - best_val) < 1e-10:
                self.stop[p] = True
                self.converged = best_ind if self.converged is None else Population.merge(self.converged, best_ind)
            self.last_best[p] = best_val
        arch = self.converged
        parts = ([arch] if arch is not None else []) + new
        archive = Population.merge(*parts) if parts else self.archive
        tag = self.FE > 0.6 * self.max_FE and self.first_tag
        if self.stop.all() or tag:
            self.first_tag = False
            for _ in range(200):
                ex = decs(archive)
                ex_pv = ex[:, PV]
                n = len(ex)
                p2, p3 = ex_pv[rng.permutation(n)], ex_pv[rng.permutation(n)]
                off = ex_pv.copy()
                site = rng.random(off.shape) < 0.2
                off[site] += 0.5 * (p2[site] - p3[site])
                off = np.clip(off, self.lower[PV], self.upper[PV])
                nd = ex.copy()
                nd[:, PV] = off
                archive = _update_archive(N, Population.merge(self.evaluate(nd), archive))
            self.big, self.cma = self._generate_big(archive)
            popN = len(self.big)
            self.last_best = 1e20 * np.ones(popN)
            self.stop = np.zeros(popN, bool)
            self.converged = None
        if self.FE >= self.max_FE:
            final = self.evaluate(np.array([b[0] for b in self.big]))
            archive = _update_archive(N, Population.merge(final, archive))
        self.archive = archive
        self.pop = archive
