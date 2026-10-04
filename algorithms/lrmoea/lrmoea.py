# emopylab 2026
"""LRMOEA (large-scale robust multi-objective evolutionary algorithm).

Reference:
S. Shao, Y. Tian, L. Zhang, K. C. Tan, and X. Zhang. An evolutionary algorithm for solving large-
scale robust multi-objective optimization problems. IEEE Transactions on Evolutionary Computation,
2025, 29(6): 2476-2490.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, ga_half, nd_sort, cons, objs, tournament, uniform_point
from algorithms.sparseea.sparseea import _env_selection
from core.population import Population

ALGORITHM_FLAGS = {'LRMOEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'robust', 'sparse'}}


class _Arch:
    """Archive of decision vectors with the history of their (perturbed) objective vectors."""

    def __init__(self, dec, mask, obj):
        n = len(dec)
        self.dec, self.mask, self.obj = np.array(dec, float), np.array(mask, float), np.array(obj, float)
        self.mobj = [self.obj[i:i + 1].copy() for i in range(n)]
        self.mr, self.tno, self.Gn = np.zeros(n), np.zeros(n, dtype=int), np.ones(n, dtype=int)

    def __len__(self):
        return len(self.dec)

    def take(self, idx):
        out = _Arch.__new__(_Arch)
        idx = np.asarray(idx)
        out.dec, out.mask, out.obj = self.dec[idx], self.mask[idx], self.obj[idx]
        out.mobj = [self.mobj[i] for i in idx]
        out.mr, out.tno, out.Gn = self.mr[idx].copy(), self.tno[idx].copy(), self.Gn[idx].copy()
        return out

    def concat(self, other):
        out = _Arch.__new__(_Arch)
        out.dec, out.mask, out.obj = np.vstack([self.dec, other.dec]), np.vstack([self.mask, other.mask]), np.vstack([self.obj, other.obj])
        out.mobj = self.mobj + other.mobj
        out.mr, out.tno, out.Gn = np.concatenate([self.mr, other.mr]), np.concatenate([self.tno, other.tno]), np.concatenate([self.Gn, other.Gn])
        return out

    def memorize(self, k, f):
        self.mobj[k] = np.vstack([self.mobj[k], f])
        self.Gn[k] += 1
        ob = self.mobj[k]
        with np.errstate(all="ignore"):
            self.mr[k] = np.mean(np.mean(np.abs(ob - ob.min(axis=0)), axis=0) / ob.mean(axis=0))

    @property
    def x(self):
        return self.dec * self.mask


def _fitness_cal(Mask, front, score):
    N, d = Mask.shape
    score = score.copy()
    b = 0
    for i in range(N):
        if front[i] <= 1:
            b += 1
            score = score + 1.0 / (2 + np.sqrt(N) * Mask[i])
    return score / b


def _lr_operator(algo, ParentDec, ParentMask, Fit):
    rng = algo.rng
    n, D = ParentDec.shape
    h = n // 2
    P1, P2 = ParentMask[:h], ParentMask[h:]
    Off = P1.copy()

    def ts(f):
        return None if len(f) == 0 else int(tournament(2, 1, f, rng=rng)[0])

    for i in range(h):
        if rng.random() < 0.5:
            idx = np.where((P1[i] != 0) & (P2[i] == 0))[0]
            k = ts(Fit[idx])
            if k is not None:
                Off[i, idx[k]] = 0
        else:
            idx = np.where((P1[i] == 0) & (P2[i] != 0))[0]
            k = ts(-Fit[idx])
            if k is not None:
                Off[i, idx[k]] = P2[i, idx[k]]
    for i in range(h):
        if rng.random() < 0.5:
            idx = np.where(Off[i] != 0)[0]
            k = ts(Fit[idx])
            if k is not None:
                Off[i, idx[k]] = 0
        else:
            idx = np.where(Off[i] == 0)[0]
            k = ts(-Fit[idx])
            if k is not None:
                Off[i, idx[k]] = 1
    enc = algo.encoding
    if np.any(enc != 4):
        OffDec = ga_half(algo.problem, ParentDec, rng=rng)
        OffDec[:, enc == 4] = 1
    else:
        OffDec = np.ones((h, D))
    return OffDec, Off


class LRMOEA(LoopAlgorithm):
    """Large-scale robust sparse EA: the non-dominated solutions are kept in an archive together with the history of
    their objective values under decision perturbation; members whose relative variation exceeds a threshold are
    dropped and, at the end, one archive member per weight vector is re-evaluated as the final population."""

    def __init__(self, pop_size: int = 100, thea: float = 0.2, eta: float = 1.15, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.thea, self.eta = float(thea), float(eta)

    def _initialize_infill(self):
        N, D, rng = self.N, self.D, self.rng
        self.W, _ = uniform_point(N, self.M)
        if self.encoding[0] == 4:
            Dec = np.ones((N, D))
        else:
            Dec = self.lower + rng.random((N, D)) * (self.upper - self.lower)
        Mask = np.zeros((N, D), bool)
        for i in range(N):
            Mask[i, rng.permutation(D)[: int(np.ceil(rng.random() ** 2 * D))]] = True
        pop = self.evaluate(Dec * Mask)
        pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(pop, Dec, Mask, N)
        self.score = _fitness_cal(self.Mask.astype(float), self.front, np.ones(D))
        self.arch = _Arch(self.Dec, self.Mask, objs(pop))
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def _perturbed_objs(self, X):
        if not hasattr(self.problem, "perturb"):
            raise TypeError("LRMOEA estimates robustness through Problem.perturb: use a robust problem family (e.g. TP)")
        return self.problem.perturb(X, 1, rng=self.rng)[0][0]

    def _arch_update(self, tem):
        arch, thea, eta = self.arch, self.thea, self.eta
        P = self._perturbed_objs(arch.x)
        for k in range(len(arch)):
            arch.memorize(k, P[k:k + 1])
        arch = arch.take(np.where(arch.mr <= thea)[0])
        if len(arch) == 0:
            return tem
        Td, Ad = tem.x, arch.x
        a = np.ones(len(Td), bool)
        for i in range(len(Td)):
            hit = np.where(np.all(Ad == Td[i], axis=1))[0]
            if len(hit):
                a[i] = False
                arch.memorize(int(hit[0]), tem.obj[i:i + 1])
        t = tem.take(np.where(a)[0])
        if len(t):
            maf = t.obj.max(axis=0)
            arch = arch.take(np.where(arch.obj.sum(axis=1) < eta * maf.sum())[0])
        return arch.concat(t)

    def _final(self):
        N, W, arch, score = self.N, self.W.copy(), self.arch, self.score
        if len(arch) <= N:
            return self.evaluate(arch.x)
        RF, MF = [], []
        while len(W) > 0 and len(arch) > len(W):
            F = arch.obj
            with np.errstate(all="ignore"):
                Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
                cos = (Fn @ W.T) / np.sqrt(np.sum(W * W, axis=1)[None, :] * np.sum(Fn * Fn, axis=1)[:, None])
            ang = np.arccos(np.clip(cos, -1, 1))
            arch.tno = np.argmin(np.where(np.isnan(ang), np.inf, ang), axis=1)
            sp = np.array([m.sum(axis=1).mean() for m in arch.mobj])
            with np.errstate(all="ignore"):
                a_sp = 1.0 / (np.sum(arch.mask.astype(bool) & (score > 0.5), axis=1) / len(score))
            rch, wr = [], []
            for r in range(len(W)):
                members = np.where(arch.tno == r)[0]
                if len(members):
                    s = sp[members] + a_sp[members]
                    best = int(members[np.argmin(s)])
                    RF.insert(0, arch.dec[best])
                    MF.insert(0, arch.mask[best])
                    rch.append(best)
                    wr.append(r)
            if not rch:
                break
            arch = arch.take(np.delete(np.arange(len(arch)), rch))
            W = np.delete(W, wr, axis=0)
        return self.evaluate(np.array(RF) * np.array(MF))

    def step(self):
        N, rng = self.N, self.rng
        pool = tournament(2, 2 * N, self.front, -self.crowd, rng=rng)
        OffDec, OffMask = _lr_operator(self, self.Dec[pool], self.Mask[pool], self.score)
        off = self.evaluate(OffDec * OffMask)
        self.pop, self.Dec, self.Mask, self.front, self.crowd = _env_selection(
            Population.merge(self.pop, off), np.vstack([self.Dec, OffDec]), np.vstack([self.Mask, OffMask]), N)
        self.score = self.arch.mask.sum(axis=0) / len(self.arch)
        first = self.front == 1
        tem = _Arch(self.Dec[first], self.Mask[first], objs(self.pop)[first])
        self.arch = self._arch_update(tem)
        if self.FE >= self.max_FE:
            self.pop = self._final()
