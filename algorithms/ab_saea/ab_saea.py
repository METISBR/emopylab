# emopylab 2026
"""AB-SAEA (adaptive Bayesian based surrogate-assisted evolutionary algorithm).

Reference:
X. Wang, Y. Jin, S. Schmitt S, and M. Olhofer. An adaptive Bayesian approach to surrogate-assisted
evolutionary multi-objective optimization. Information Sciences, 2020, 519: 317-331.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga, nd_sort, objs, uniform_point
from algorithms.community_utils.dace import DaceModel
from algorithms.k_rvea.k_rvea import _angles, _argmin_rows, _gamma
from core.population import Population

ALGORITHM_FLAGS = {'ABSAEA': {'expensive', 'integer', 'many', 'multi', 'real'}}


def f_selection(F, V, theta, flag):
    """One solution per active reference vector: angle-penalised distance (flag 2) or the scaled angle alone (flag 1)."""
    M = F.shape[1]
    F = F - F.min(axis=0)
    gamma = _gamma(V)
    ang = _angles(F, V)
    assoc = _argmin_rows(ang)
    nxt = []
    for i in np.unique(assoc):
        cur = np.where(assoc == i)[0]
        if flag == 2:
            ad = (1 + M * theta * ang[cur, i] / gamma[i]) * np.sqrt((F[cur] ** 2).sum(1))
        else:
            ad = M * ang[cur, i] / gamma[i]
        nxt.append(cur[int(np.argmin(ad))])
    return np.array(nxt, dtype=int)


def ds_merge(S, Y, ds=1e-14):
    """Merge design sites closer than ``ds`` (Euclidean, on standardised data): each group is replaced by its mean site and
    mean response; unmerged sites keep their order and merged ones are appended in merge order."""
    S, Y = np.asarray(S, float).copy(), np.asarray(Y, float).copy()
    while True:
        m = len(S)
        with np.errstate(all="ignore"):
            sc = (S - S.mean(0)) / S.std(0, ddof=1)
        Dm = np.sqrt(((sc[:, None] - sc[None]) ** 2).sum(-1))
        np.fill_diagonal(Dm, 0.0)
        mult = (Dm < ds).sum(0)
        if mult.max() == 1:
            return S, Y
        ladr = []
        while mult.max() > 1:
            jj = int(np.argmax(mult))
            ngb = np.where(Dm[:, jj] < ds)[0]
            S[jj], Y[jj] = S[ngb].mean(0), Y[ngb].mean()
            ladr.append(jj)
            mult[ngb] = 0
        act = np.concatenate([np.where(mult > 0)[0], ladr]).astype(int)
        S, Y = S[act], Y[act]


class ABSAEA(LoopAlgorithm):
    """Kriging-assisted RVEA whose infill criterion blends predicted objectives and predicted uncertainty with weights that
    move from exploitation to exploration-free convergence along the budget (cosine schedule); up to ``mu`` reference-vector
    representatives of the blend are evaluated per iteration. Training data are capped at ``L`` points."""

    def __init__(self, pop_size: int = 100, alpha: float = 2, wmax: int = 20, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.alpha, self.wmax, self.mu = float(alpha), int(wmax), int(mu)

    def _initialize_infill(self):
        from operators.utility_functions.UniformPoint import UniformPoint
        self.V0, self.pop_size = uniform_point(self.pop_size, self.M)
        self.V, self.V1 = self.V0.copy(), self.V0.copy()
        NI = 11 * self.D - 1
        self.L = NI + 25
        P, _ = UniformPoint(NI, self.D, "Latin", rng=self.rng)
        return self.evaluate(self.lower + np.asarray(P) * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.theta = 5.0 * np.ones((self.M, self.D))
        self._set_optimum()

    def step(self):
        rng, D, M, L = self.rng, self.D, self.M, self.L
        idx = np.unique(decs(self.pop), axis=0, return_index=True)[1]
        X, F = decs(self.pop)[idx], objs(self.pop)[idx]
        if len(X) > L:
            front, _ = nd_sort(F, None, len(X))
            o = np.argsort(front, kind="stable")
            h = L // 2
            X1, F1 = X[o[:h]], F[o[:h]]
            X2, F2 = X[o[: L - h]], F[o[: L - h]]
            p = rng.permutation(len(X2))[: L - h]
            X, F = np.vstack([X1, X2[p]]), np.vstack([F1, F2[p]])
        models = []
        for i in range(M):
            mS, mY = ds_merge(X, F[:, i])
            dm = DaceModel(mS, mY, "regpoly0", self.theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            models.append(dm)
            self.theta[i] = dm.theta
        pop_dec = X
        for w in range(1, self.wmax + 1):
            pop_dec = np.vstack([pop_dec, ga(self.problem, pop_dec, rng=rng)])
            preds = [m.predict(pop_dec, mse=True) for m in models]
            PF = np.column_stack([q[0] for q in preds])
            MSE = np.column_stack([q[1] for q in preds])
            s = f_selection(PF, self.V, (w / self.wmax) ** self.alpha, 2)
            pop_dec, PF, MSE = pop_dec[s], PF[s], MSE[s]
            if w % int(np.ceil(self.wmax * 0.1)) == 0:
                self.V = self.V0 * (F.max(axis=0) - F.min(axis=0))
        ratio = self.FE / self.max_FE
        a, b = -0.5 * np.cos(ratio * np.pi) + 0.5, 0.5 * np.cos(ratio * np.pi) + 0.5
        with np.errstate(all="ignore"):
            fit = PF / PF.max(axis=0) * b + MSE / MSE.max(axis=0) * a
        s = f_selection(fit, self.V1, ratio ** self.alpha, 2 if a > 0.5 else 1)
        new = pop_dec[s]
        if len(new) >= self.mu:
            new = new[rng.permutation(len(new))[:5]]
        if self.FE % int(np.ceil(self.max_FE * 0.1)) == 0:
            self.V1 = self.V0 * (F.max(axis=0) - F.min(axis=0))
        self.pop = Population.merge(self.pop, self.evaluate(new))
