"""Kriging-assisted GA with constraint-aware surrogate handling (constraint models M1: one per constraint, M2: one for the summed violation)."""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, crowding, decs, ga, nd_sort, objs, tournament
from algorithms.community_utils.dace import DaceModel
from algorithms.community_utils.sparse_mask import lhs_design
from core.population import Population

__all__ = ["RGABase"]


def _first_unique(F):
    return np.unique(F, axis=0, return_index=True)[1]


def _normalise_cols(C):
    C = np.maximum(0.0, C)
    if C.shape[1] == 0:
        return C
    with np.errstate(all="ignore"):
        C = (C - C.min(axis=0)) / (C.max(axis=0) - C.min(axis=0))
    C[:, np.isnan(C[0])] = 0.0
    return C


def _truncate(pop, N):
    """Crowding-distance truncation of ``pop`` to ``N`` (constraint-aware ND sort)."""
    F = objs(pop)
    cv = np.maximum(0.0, cons(pop)).sum(axis=1)
    front, maxf = nd_sort(F, cv if cons(pop).shape[1] else None, N)
    nxt = front < maxf
    cd = crowding(F, front)
    last = np.where(front == maxf)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt]


def update_archive(pop, N):
    pop = pop[_first_unique(objs(pop))]
    return _truncate(pop, N) if len(pop) > N else pop


def update_population(pop, new, N):
    fo, fn = objs(pop), objs(new)
    keep = np.array([not np.any(np.all(fn == r, axis=1)) for r in fo], dtype=bool)
    idx = np.where(keep)[0]
    idx = idx[_first_unique(fo[idx])] if len(idx) else idx
    pop = pop[idx]
    if len(pop) > N:
        pop = _truncate(pop, N)
    return Population.merge(pop, new)


class RGABase(LoopAlgorithm):
    """Surrogate GA: ``wmax`` generations on Kriging models of the objectives and of the (normalised) constraint violation, the
    ``mu`` best predicted solutions are evaluated for real; an archive of ``N`` solutions is returned."""

    CV_MODE = 1     # 1: one model per constraint, 2: one model for the summed violation

    def __init__(self, pop_size: int = 100, wmax: int = 20, mu: int = 5, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.wmax, self.mu = int(wmax), int(mu)

    def _initialize_infill(self):
        self.NI = 11 * self.D - 1
        P = lhs_design(self.rng, self.NI, self.D)
        return self.evaluate(self.lower + P * (self.upper - self.lower))

    def _initialize_advance(self, infills=None, **kwargs):
        self.P = infills
        self.pop = update_archive(infills, self.N)
        ncon = cons(infills).shape[1]
        self.theta = 5.0 * np.ones((self.M + (ncon if self.CV_MODE == 1 else 1), self.D))
        self._set_optimum()

    def _env_selection(self, Dec, Obj, N):
        M = self.M
        idx = _first_unique(Obj[:, :M])
        Dec, Obj = Dec[idx], Obj[idx]
        real = Obj[:, :M]
        if self.CV_MODE == 1:
            cv = np.maximum(0.0, Obj[:, M:]).sum(axis=1)
            has_c = Obj.shape[1] > M
        else:
            cv, has_c = np.maximum(0.0, Obj[:, -1]), True
        front, maxf = nd_sort(real, cv if has_c else None, N)
        nxt = front < maxf
        cd = crowding(real, front)
        last = np.where(front == maxf)[0]
        rank = np.argsort(-cd[last], kind="stable")
        nxt[last[rank[: N - int(nxt.sum())]]] = True
        return Dec[nxt], Obj[nxt], front[nxt], cd[nxt]

    def step(self):
        rng, D, NI, M0 = self.rng, self.D, self.NI, self.M
        Pdec, C = decs(self.P), np.maximum(0.0, cons(self.P))
        if self.CV_MODE == 1:
            C = _normalise_cols(C)
        else:
            C = _normalise_cols(C).sum(axis=1, keepdims=True)
        PopObj = np.hstack([objs(self.P), C])
        M = PopObj.shape[1]
        models = []
        for i in range(M):
            dm = DaceModel(Pdec, PopObj[:, i], "regpoly0", self.theta[i], 1e-5 * np.ones(D), 100 * np.ones(D))
            models.append(dm)
            self.theta[i] = dm.theta
        raw = np.maximum(0.0, cons(self.P)).sum(axis=1)
        cv = C.sum(axis=1) if self.CV_MODE == 1 else raw
        front, _ = nd_sort(objs(self.P), cv if cons(self.P).shape[1] else None, np.inf)
        cd = crowding(objs(self.P), front)
        Dec = Pdec
        for _ in range(self.wmax):
            mating = tournament(2, NI, front, -cd, rng=rng)
            Dec = np.vstack([Dec, ga(self.problem, Dec[mating], rng=rng)])
            Obj = np.column_stack([m.predict(Dec) for m in models])
            Dec, Obj, front, cd = self._env_selection(Dec, Obj, NI)
        new_dec, _, _, _ = self._env_selection(Dec, Obj, self.mu)
        new = self.evaluate(new_dec)
        self.P = update_population(self.P, new, NI - self.mu)
        self.pop = update_archive(Population.merge(self.pop, new), self.N)
