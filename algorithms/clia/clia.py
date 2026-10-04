# emopylab 2026
"""CLIA (evolutionary algorithm with cascade clustering and reference point incremental learning).

Reference:
H. Ge, M. Zhao, L. Sun, Z. Wang, G. Tan, Q. Zhang, and C. L. P. Chen. A many-objective evolutionary
algorithm with two interacting processes: Cascade clustering and reference point incremental
learning. IEEE Transactions on Evolutionary Computation, 2019, 23(4): 572-586.
"""

from __future__ import annotations

import math

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, cons, decs, ga, nd_sort, objs, tournament, uniform_point
from core.population import Population
from util.svm import SVMClassifier

ALGORITHM_FLAGS = {'CLIA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}

# problem-specific settings of the reference implementation (stable thresholds for M = 5/10/15, disabled archive,
# final crowding pick, intercept normalisation); unlisted problems use the defaults
_MAF = {
    "MaF1": dict(st=[20, 20, 15], dis={5: False, 10: False}, cp={5: False, 10: True}, norm=""),
    "MaF2": dict(dis={5: False, 10: False}, cp={5: False, 10: True}, norm={10: ""}),
    "MaF3": dict(st=[np.inf] * 3, cp=False, dis=True, norm=""),
    "MaF4": dict(st=[5, 10, 15], dis={5: False, 10: False}, cp={5: False, 10: False}, norm=""),
    "MaF5": dict(cp=False, dis=True, norm="normalize"),
    "MaF6": dict(st=[5, 10, 15], dis={5: False, 10: False}, cp={5: True, 10: False}, norm=""),
    "MaF7": dict(st=[10, 15, 20], dis={5: False, 10: False}, cp={5: False, 10: False}, norm="normalize"),
    "MaF8": dict(st=[5, 5, 5], dis={5: False, 10: False}, cp={5: False, 10: True}, norm="normalize"),
    "MaF9": dict(st=[5, 5, 5], dis={5: False, 10: False}, cp={5: False, 10: False}, norm=""),
    "MaF10": dict(cp=False, dis=True), "MaF11": dict(cp=False, dis=True), "MaF12": dict(cp=False, dis=True),
    "MaF13": dict(cp=False, dis=False, norm=""), "MaF14": dict(cp=False, dis=True, norm=""),
    "MaF15": dict(st=[20, 20, 20], dis=False, cp=False, norm=""),
}


def _pick(v, M, default):
    if isinstance(v, dict):
        return v.get(M, default)
    return default if v is None else v


def _sin_pair(O, Z, distance=False):
    with np.errstate(all="ignore"):
        cos = (O @ Z.T) / (np.linalg.norm(O, axis=1)[:, None] * np.linalg.norm(Z, axis=1)[None])
        err = np.sqrt(np.maximum(1 - cos ** 2, 0))
        if distance:
            err = np.linalg.norm(O, axis=1)[:, None] * err
    err = np.where(np.isnan(err), np.inf, err)
    alloc = np.argmin(err, 1)
    return err[np.arange(len(O)), alloc], alloc


def _crowd(O):
    N, M = O.shape
    cd = np.zeros(N)
    fmax, fmin = O.max(0), O.min(0)
    with np.errstate(all="ignore"):
        for i in range(M):
            r = np.argsort(O[:, i], kind="stable")
            cd[r[0]] = cd[r[-1]] = np.inf
            for j in range(1, N - 1):
                cd[r[j]] += (O[r[j + 1], i] - O[r[j - 1], i]) / (fmax[i] - fmin[i])
    return cd


def _density_sequence(M):
    seq, step = [], 1
    R = math.comb(M + step - 1, step)
    while R < 1e6:
        seq.append(R)
        step += 1
        R = math.comb(M + step - 1, step)
    return seq


class CLIA(LoopAlgorithm):
    """Cascade clustering keeps one solution per reference vector (sorted by a penalised distance) and fills the rest
    round-robin; a non-dominated archive feeds an SVM that learns which reference vectors lie in active regions, so the
    reference density can grow (with the predicted-inactive vectors pruned) or shrink once the active set is stable."""

    def __init__(self, pop_size: int = 100, stable_threshold=(0, 0, 0), delta=None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.stable_threshold = list(stable_threshold)
        self.delta = delta

    # -- settings / reference generation ----------------------------------
    def _settings(self):
        M = self.M
        name = type(self.problem).__name__
        cfg = _MAF.get(name, {})
        st = self.stable_threshold
        if "st" in cfg and np.linalg.norm(st) < 1e-2:
            st = list(cfg["st"])
        self.st = st
        self.disable = bool(_pick(cfg.get("dis"), M, False))
        self.crowd_flag = bool(_pick(cfg.get("cp"), M, False))
        self.norm_str = _pick(cfg.get("norm"), M, "") or ""
        self.delta_ = 2 * self.N if self.delta is None else self.delta
        seq = _density_sequence(M)
        self.seq = [s for s in seq if s >= self.N]
        self.pointer, self.ref_init = 1, True
        self.max_ref_flag = False

    def _gen_z(self, action):
        if self.ref_init:
            self.max_ref_flag = False
            self.pointer = 1
            self.ref_init = False
        p = self.pointer + action
        if p >= len(self.seq):
            self.max_ref_flag = True
        p = int(max(1, min(len(self.seq), p)))
        self.pointer = p
        Z, _ = uniform_point(self.seq[p - 1], self.M)
        self.current_density = len(Z)
        return Z

    def _normalization(self, O):
        if self.norm_str != "normalize":
            return O
        N, M = O.shape
        ch = list(np.argmin(O, 0))
        L = np.column_stack([(O[:, [j for j in range(M) if j != i]] ** 2).sum(1) for i in range(M)])
        ch += list(np.argmin(L, 0))
        ext = np.unique(np.array(ch)[np.argmax(O[ch], 0)])
        a = O.max(0)
        if len(ext) == M:
            try:
                a = 1.0 / np.linalg.solve(O[ext], np.ones(M))
            except np.linalg.LinAlgError:
                a = O.max(0)
        return (O - O.min(0)) / a

    # -- building blocks ----------------------------------------------------
    def _frontier(self, F, C):
        Fp = F.copy()
        if C.shape[1]:
            inf = np.any(C > 0, 1)
            Fp[inf] = F.max(0) + np.maximum(0, C[inf]).sum(1)[:, None]
        f, _ = nd_sort(Fp, None, 1)
        fi = np.where(f == 1)[0]
        return fi, np.setdiff1d(np.arange(len(F)), fi)

    def _crowding_pick(self, pop, picks, mode):
        if len(pop) < picks:
            return pop
        F = objs(pop)
        if mode == "precise":
            keep = np.ones(len(pop), bool)
            with np.errstate(all="ignore"):
                O = (F - F.mean(0)) / F.std(0, ddof=1)
            while keep.sum() > picks:
                cd = np.full(len(pop), np.nan)
                cd[keep] = _crowd(O)
                keep[int(np.nanargmin(cd))] = False
                O = self._normalization(F[keep])
            return pop[keep]
        with np.errstate(all="ignore"):
            O = (F - F.mean(0)) / F.std(0, ddof=1)
        return pop[np.argsort(-_crowd(O), kind="stable")[:picks]]

    def _update_archive(self, A, P, Z, picks):
        if self.disable:
            return A
        P = Population.merge(A, P) if A is not None and len(A) else P
        P = P[np.unique(objs(P), axis=0, return_index=True)[1]]
        fi, _ = self._frontier(objs(P), cons(P))
        A = P[fi]
        if len(A) > picks:
            O = self._normalization(objs(A))
            mm, alloc = _sin_pair(O, Z)
            centers = []
            for c in np.unique(alloc):
                idx = np.where(alloc == c)[0]
                centers.append(idx[int(np.argmin(mm[idx]))])
            A = Population.merge(A[np.array(centers)], self._crowding_pick(A, picks, "fast"))
            A = A[np.unique(objs(A), axis=0, return_index=True)[1]]
        return A

    def _cascade(self, P, Z, N, cat_flag):
        rng = self.rng
        P = P[np.unique(objs(P), axis=0, return_index=True)[1]]
        F = objs(P)
        fi, nfi = self._frontier(F, cons(P))
        O = self._normalization(F)
        mm, alloc = _sin_pair(O[fi], Z)
        active = np.unique(alloc)
        NC = len(active)
        queues, centers = [], []
        for c in active:
            sm = alloc == c
            idx = fi[sm]
            if len(idx) == 1:
                order = [0]
            else:
                fm = 5 * (mm[sm] * np.sqrt((O[idx] ** 2).sum(1))) + O[idx].mean(1)
                order = [int(np.argmin(fm))] if NC >= N else list(np.argsort(fm, kind="stable"))
            q = list(idx[order])
            queues.append(q)
            centers.append(q[0])
        centers = np.array(centers, dtype=int)
        if NC > N:
            P = self._crowding_pick(P[centers], N, "precise")
        elif NC == N:
            P = P[centers]
        else:
            if cat_flag or len(fi) < N:
                if len(nfi):
                    Cc = O[centers]
                    d = np.sqrt(((O[nfi][:, None] - Cc[None]) ** 2).sum(-1))
                    al, md = np.argmin(d, 1), d.min(1)
                    for k in np.unique(al):
                        s = al == k
                        queues[k] += list(nfi[s][np.argsort(md[s], kind="stable")])
            perm = rng.permutation(NC)
            queues = [queues[i] for i in perm]
            nxt, j = [], 0
            total = sum(len(q) for q in queues)
            for _ in range(min(N, total)):
                while not queues[j]:
                    j = (j + 1) % NC
                nxt.append(queues[j].pop(0))
                j = (j + 1) % NC
            P = P[np.array(nxt, dtype=int)]
        return P, active, np.setdiff1d(np.arange(len(Z)), active)

    # -- incremental learning ----------------------------------------------
    def _project(self, O):
        M = O.shape[1]
        U, s, _ = np.linalg.svd(np.eye(M) - 1.0 / M)
        return O @ U[:, s > 1e-10]

    def _svm_predict(self, X):
        if self.svm_X is None:
            return np.full(len(X), np.nan), np.full(len(X), np.nan)
        f = self.svm.decision_function(self._project(X))
        score = 0.5 + 0.5 * f / (1 + np.abs(f))                         # elliotsig
        return score, np.where(score >= 0.5, 1, -1)

    def _learn(self, Pp, Nn):
        if len(Pp) == 0 or len(Nn) == 0:
            return
        X = np.vstack([Pp, Nn])
        y = np.concatenate([np.ones(len(Pp)), -np.ones(len(Nn))])
        # the incremental SVM keeps the exact solution on all data learned so far: train on the union
        self.svm_X = X if self.svm_X is None else np.vstack([self.svm_X, X])
        self.svm_y = y if self.svm_y is None else np.concatenate([self.svm_y, y])
        self.svm = SVMClassifier(C=10.0, kernel_scale=np.sqrt(2 * 0.056)).fit(self._project(self.svm_X), self.svm_y)

    def _reduce(self, Z, thr):
        n0 = len(Z)
        if n0 <= self.N:
            return Z
        score, _ = self._svm_predict(Z)
        if len(score) == 1:
            keep = np.arange(n0)
        elif thr is None:
            keep = np.array([], dtype=int)
        elif 0 < thr <= 1:
            keep = np.where(score >= min(thr, 0.5))[0]
        else:
            S = np.sort(score)[::-1]
            keep = np.where(score >= S[min(max(self.N, int(thr)), len(S)) - 1])[0]
        if len(keep) < self.N:
            S = np.sort(score)[::-1]
            keep = np.where(score >= S[min(self.N, len(S)) - 1])[0]
        return Z[keep]

    def _check_status(self, total, active):
        M = self.M
        if self.status_init:
            self.status_init = False
            self.cons_counter = 0
            self.density = self.current_density
            self.history = np.zeros(total)
            ptr = {5: 1, 10: 2, 15: 3}.get(M, 0)
            if ptr > 0 and self.st[ptr - 1] > 0:
                self.threshold = self.st[ptr - 1]
            else:
                self.threshold = min(20, max(5, int(np.ceil(self.max_FE / 2e4))))
        fl = 3 if self.norm_str == "normalize" else 1e-2
        if self.density != self.current_density:
            self.density = self.current_density
            self.cons_counter = 0
            self.history = np.zeros(total)
        cur = np.zeros(max(len(self.history), total, (active.max() + 1) if len(active) else 0))
        cur[active] = 1
        hist = np.zeros(len(cur))
        hist[: len(self.history)] = self.history
        if np.linalg.norm(hist - cur) <= fl:
            self.cons_counter += 1
        else:
            self.history = cur
            self.cons_counter = 0
        if self.cons_counter >= self.threshold:
            self.cons_counter = 0
            self.threshold = max(5, self.threshold - 1)
            return "stable"
        return "unstable"

    def _incremental(self, Z, ica, icn, A):
        N, M = self.N, self.M
        nA = 0 if A is None else len(A)
        if (self.disable and nA > 0.9 * self.max_arc) or (len(ica) > 0.95 * N and len(Z) == N):
            return Z
        if self._check_status(len(Z), ica) == "unstable" or nA < 0.9 * self.max_arc:
            return Z
        if len(ica) > N and len(ica) == len(Z):
            return self._gen_z(-1)
        if len(ica) < 0.95 * N:
            if self.max_ref_flag:
                return Z
            FA = objs(A)
            _, al = _sin_pair(FA, Z)
            z_old = Z[np.unique(al)]
            Z = self._gen_z(1)
            trunc = lambda X: X / X.sum(1, keepdims=True)
            if self.norm_str == "normalize":
                if len(Z) > 4 * N:
                    self.norm_str = ""
                    _, al = _sin_pair(FA, Z)
                    ina = np.setdiff1d(np.arange(len(Z)), np.unique(al))
                    self._learn(trunc(FA), trunc(Z[ina]))
                    zn = self._reduce(Z, self.delta_)
                else:
                    zn = Z
            elif len(Z) > 2 * N:
                picked = self._crowding_pick(A, 2 * N, "precise")
                _, al = _sin_pair(FA, Z)
                ina = np.setdiff1d(np.arange(len(Z)), np.unique(al))
                Pp, Nn = trunc(objs(picked)), trunc(Z[ina])
                if len(Pp) and len(Nn):
                    _, ya = self._svm_predict(Pp)
                    _, yi = self._svm_predict(Nn)
                    Pp, Nn = Pp[ya != 1], Nn[yi != -1]
                    self._learn(Pp, Nn)
                    zn = self._reduce(Z, self.delta_)
                else:
                    zn = Z
            else:
                zn = Z
            _, al = _sin_pair(FA, Z)
            Z = np.unique(np.vstack([z_old, zn, Z[np.unique(al)]]), axis=0)
        return Z

    # -- main loop ---------------------------------------------------------------
    def _initialize_advance(self, infills=None, **kwargs):
        self._settings()
        self.Z = self._gen_z(-10 ** 9)
        self.ref_init = True
        self.status_init = True
        self.max_arc = int(np.ceil(0.33 * self.M * self.N))
        self.svm = self.svm_X = self.svm_y = None
        self.P, self.A = infills, infills
        self.pop = infills
        self._set_optimum()

    def step(self):
        rng, N = self.rng, self.N
        C = cons(self.P)
        cv = np.maximum(0, C).sum(1) if C.shape[1] else np.zeros(len(self.P))
        off = self.evaluate(ga(self.problem, decs(self.P)[tournament(2, N, cv, rng=rng)], rng=rng))
        merged = Population.merge(self.P, off)
        self.A = self._update_archive(self.A, merged, self.Z, self.max_arc)
        self.P, ica, icn = self._cascade(merged, self.Z, N, self.FE < self.max_FE)
        self.Z = self._incremental(self.Z, ica, icn, self.A)
        if self.FE >= self.max_FE and self.crowd_flag:
            self.pop = self._crowding_pick(self._update_archive(self.A, self.P, self.Z, np.inf), N, "precise")
        else:
            self.pop = self.P
