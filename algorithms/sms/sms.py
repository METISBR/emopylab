"""EmoPyLab SMS-EMOA and tournament selection helpers."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, ga_half, nd_sort, objs
from algorithms.hype.hype import cal_hv
from core.population import Population

__all__ = [
    "cv_and_dom_tournament",
    "SMSEMOA",
]


def cv_and_dom_tournament(pop: Population, P: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Tournament selection function comparing constraint violation (CV) and dominance."""
    n_tournaments, n_parents = P.shape
    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)
    random_state = kwargs.get("random_state", None)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        ind_a, ind_b = pop[a], pop[b]

        cv_a = float(getattr(ind_a, "cv", 0.0) or 0.0)
        cv_b = float(getattr(ind_b, "cv", 0.0) or 0.0)

        if cv_a != cv_b:
            S[i] = a if cv_a < cv_b else b
        else:
            rank_a = getattr(ind_a, "rank", None)
            rank_b = getattr(ind_b, "rank", None)
            if rank_a is not None and rank_b is not None and rank_a != rank_b:
                S[i] = a if rank_a < rank_b else b
            else:
                f_a = getattr(ind_a, "F", None)
                f_b = getattr(ind_b, "F", None)
                if f_a is not None and f_b is not None:
                    f_a = np.asarray(f_a, dtype=float).reshape(-1)
                    f_b = np.asarray(f_b, dtype=float).reshape(-1)
                    a_dom_b = np.all(f_a <= f_b) and np.any(f_a < f_b)
                    b_dom_a = np.all(f_b <= f_a) and np.any(f_b < f_a)
                    if a_dom_b and not b_dom_a:
                        S[i] = a
                    elif b_dom_a and not a_dom_b:
                        S[i] = b
                    else:
                        if random_state is not None:
                            S[i] = random_state.choice([a, b])
                        else:
                            S[i] = np.random.choice([a, b])
                else:
                    if random_state is not None:
                        S[i] = random_state.choice([a, b])
                    else:
                        S[i] = np.random.choice([a, b])

    return S[:, None].astype(int, copy=False)


def _weak_dom(A, B):
    """``D[j, i]`` is True when ``A[j]`` is no worse than ``B[i]`` in every objective."""
    return np.all(A[:, None, :] <= B[None, :, :], axis=2)


def update_front(F, front, x=None):
    """Incremental non-dominated sorting: insert the last row of ``F`` (``x`` None) or delete row ``x``.

    Weak dominance (all objectives <=) is used, so a duplicate of a front member is pushed to the next front."""
    F = np.asarray(F, dtype=float)
    N = len(F)
    if x is None:
        front = np.append(np.asarray(front, dtype=float), 0)
        dom_new = np.all(F[:-1] <= F[-1], axis=1)
        cur = 1
        while np.any((front[:-1] == cur) & dom_new):
            cur += 1
        move = np.zeros(N, bool)
        move[-1] = True
        while move.any():
            nxt = (front == cur) & np.any(_weak_dom(F[move], F), axis=0)
            front[move] = cur
            cur += 1
            move = nxt
        return front
    front = np.asarray(front, dtype=float).copy()
    move = np.zeros(N, bool)
    move[x] = True
    cur = front[x] + 1
    while move.any():
        nxt = (front == cur) & np.any(_weak_dom(F[move], F), axis=0)
        prev = (front == cur - 1) & ~move
        if nxt.any():
            nxt[nxt] = ~np.any(_weak_dom(F[prev], F[nxt]), axis=0)
        front[move] = cur - 2
        cur += 1
        move = nxt
    return np.delete(front, x)


class SMSEMOA(LoopAlgorithm):
    """S-metric selection EMOA: steady state, one offspring of two random parents per step; the member of the worst
    front with the smallest exclusive hypervolume contribution is discarded (exact in two objectives, Monte-Carlo
    otherwise)."""

    ALGO_FLAGS = {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"}

    def __init__(self, pop_size: int = 100, sampling=None, n_sample: int = 10000, **kwargs: Any) -> None:
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.n_sample = int(n_sample)

    def start(self):
        self.front = nd_sort(objs(self.pop), None, np.inf)[0].astype(float)

    def _reduce(self, pop):
        F = objs(pop)
        self.front = update_front(F, self.front)
        last = np.where(self.front == self.front.max())[0]
        P = F[last]
        n, M = P.shape
        delta = np.full(n, np.inf)
        if M == 2:
            r = np.lexsort((P[:, 1], P[:, 0]))
            for i in range(1, n - 1):
                delta[r[i]] = (P[r[i + 1], 0] - P[r[i], 0]) * (P[r[i - 1], 1] - P[r[i], 1])
        elif n > 1:
            delta = cal_hv(P, P.max(axis=0) * 1.1, 1, self.n_sample, self.rng)
        worst = int(last[int(np.argmin(delta))])
        self.front = update_front(F, self.front, worst)
        return pop[np.delete(np.arange(len(pop)), worst)]

    def step(self):
        for _ in range(self.N):
            parents = decs(self.pop[self.rng.permutation(len(self.pop))[:2]])
            off = self.evaluate(ga_half(self.problem, parents, rng=self.rng))
            self.pop = self._reduce(Population.merge(self.pop, off))
