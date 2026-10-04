# emopylab 2026
"""SMOA (supervised multi-objective optimization algorithm).

Reference:
T. Takagi, K. Takadama, and H. Sato. Supervised multi-objective optimization algorithm using
estimation. Proceedings of the IEEE Congress on Evolutionary Computation, 2022.
"""

from __future__ import annotations

import os

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, decs, objs, uniform_point
from algorithms.community_utils.sparse_mask import lhs_design
from algorithms.community_utils.surrogates import RBFExact
from core.population import Population

ALGORITHM_FLAGS = {'SMOA': {'integer', 'many', 'multi', 'real'}}


def _load(data):
    if isinstance(data, (str, os.PathLike)):
        p = str(data)
        if p.endswith(".npy"):
            return np.load(p)
        return np.loadtxt(p)
    return np.asarray(data, float)


def subset_selection(F, Fhat, N):
    """Greedy max-min (Euclidean) completion of the given objective vectors ``F`` with rows of ``Fhat`` until ``N`` points are
    selected; returns the mask over ``Fhat``.  Distances are kept incrementally (no full distance matrix)."""
    L = len(F)
    sel = np.zeros(len(Fhat), bool)
    dmin = np.full(len(Fhat), np.inf)
    for f in F:
        dmin = np.minimum(dmin, np.sqrt(((Fhat - f) ** 2).sum(1)))
    while L + sel.sum() < N:
        cand = np.where(~sel)[0]
        r = cand[int(np.argmax(dmin[cand]))]
        sel[r] = True
        dmin = np.minimum(dmin, np.sqrt(((Fhat - Fhat[r]) ** 2).sum(1)))
    return sel


class SMOA(LoopAlgorithm):
    """One-shot supervised optimisation: from a supervised set of evaluated solutions, exact RBF networks map each L1-unit
    objective direction to its L1 norm and to every decision variable; a dense set of ``H`` uniformly spread directions (thinned
    by greedy max-min selection on the predicted objectives to fit the budget) is mapped to decision vectors and evaluated.

    ``data`` is the supervised set: a decision matrix or a path to a text/``.npy`` file (the reference asks the user to pick
    the file). Without it a Latin hypercube sample of ``pop_size`` points is used as the supervised set."""

    def __init__(self, pop_size: int = 100, H: int = 26000, data=None, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.H, self.data = int(H), data

    def _initialize_infill(self):
        if self.data is not None:
            X = _load(self.data)
        else:
            X = self.lower + lhs_design(self.rng, self.N, self.D) * (self.upper - self.lower)
        return self.evaluate(X)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self._set_optimum()

    def step(self):
        W, _ = uniform_point(self.H, self.M, "ILD")
        W[W == 1e-6] = 0
        F, X0 = objs(self.pop), decs(self.pop)
        Y = np.abs(F).sum(1)
        U = F / Y[:, None]
        if self.max_FE < len(W):
            Fhat = W * np.asarray(RBFExact(1.0).fit(U, Y).predict(W)).reshape(-1, 1)
            W = W[subset_selection(F, Fhat, self.max_FE)]
        net = RBFExact(1.0).fit(U, X0)
        dec = np.asarray(net.predict(W)).reshape(len(W), -1)
        self.pop = Population.merge(self.pop, self.evaluate(dec))
        if self.termination is not None:
            self.termination.terminate()
