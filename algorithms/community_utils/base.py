# emopylab 2026
"""Base class and helpers for algorithm ports that follow the classic ``while not terminated`` loop.

An algorithm written on :class:`LoopAlgorithm` implements

* ``start()``  - one-off state built from the evaluated initial population (``self.pop``);
* ``step()``   - the body of the main loop; every new solution is created through
  :meth:`LoopAlgorithm.evaluate`, so function evaluations are charged to the problem
  exactly once and whole generations are executed (termination is checked between steps).

All helpers here are 0-based and return NumPy arrays; the 1-based community utility
functions in ``operators.utility_functions`` are wrapped accordingly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions import _common as _u
from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorDE import OperatorDE
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.OperatorGAhalf import OperatorGAhalf
from operators.utility_functions.OperatorPSO import OperatorPSO
from operators.utility_functions.UniformPoint import UniformPoint
from algorithms.community_utils.kernels import angle_matrix, cosine_distance, dominance_matrix, pdist2, sde_distance
from util.optimum import filter_optimum

__all__ = [
    "LoopAlgorithm", "Terminated",
    "decs", "objs", "cons", "cv",
    "tournament", "roulette", "nd_sort", "crowding", "uniform_point",
    "ga", "ga_half", "de", "pso",
    "pdist2", "sort_rows_rank", "angle_matrix", "first_front",
    "sde_distance", "truncate_lexi", "cosine_distance", "dominance_matrix",
    "normalization", "first_index_reaching", "velocity", "neighbors_of", "stm_select", "grid_locations", "kmeans", "unique_individuals", "chebyshev_dist", "adds", "gde3_selection", "gradient_direction", "polynomial_mutation",
]


# ----------------------------------------------------------------------------
# Population accessors (MATLAB ``Population.decs / objs / cons``)
# ----------------------------------------------------------------------------
def decs(pop) -> np.ndarray:
    return np.asarray(pop.get("X"), dtype=float)


def objs(pop) -> np.ndarray:
    return np.asarray(pop.get("F"), dtype=float)


def cons(pop) -> np.ndarray:
    """Constraint matrix (g <= 0 feasible); shape (N, 0) when the problem is unconstrained."""
    g = pop.get("G")
    n = len(pop)
    parts = []
    if g is not None:
        g = np.asarray(g, dtype=float)
        if g.size:
            parts.append(g.reshape(n, -1))
    h = pop.get("H")
    if h is not None:
        h = np.asarray(h, dtype=float)
        if h.size:
            parts.append(np.abs(h.reshape(n, -1)) - 1e-4)
    return np.hstack(parts) if parts else np.zeros((n, 0), dtype=float)


def cv(pop) -> np.ndarray:
    c = cons(pop)
    return np.sum(np.maximum(0.0, c), axis=1) if c.size else np.zeros(len(pop), dtype=float)


# ----------------------------------------------------------------------------
# 0-based wrappers of the community utility functions
# ----------------------------------------------------------------------------
def tournament(K: int, N: int, *fitness, rng=None) -> np.ndarray:
    """K-tournament selection, lower fitness wins (lexicographic across several fitness vectors)."""
    return _u.tournament_selection(K, N, *fitness, rng=rng) - 1


def roulette(N: int, fitness, rng=None) -> np.ndarray:
    return _u.roulette_wheel_selection(N, fitness, rng=rng) - 1


def nd_sort(F, C=None, n_sort=np.inf):
    """Return ``(front_no, max_front_no)``; ``front_no`` is 1-based (inf = not ranked)."""
    if C is None or np.size(C) == 0:
        return NDSort(np.asarray(F, dtype=float), n_sort)
    return NDSort(np.asarray(F, dtype=float), np.asarray(C, dtype=float), n_sort)


def crowding(F, front_no=None) -> np.ndarray:
    return np.asarray(CrowdingDistance(np.asarray(F, dtype=float), front_no), dtype=float)


def uniform_point(N: int, M: int, method: str = "NBI"):
    W, n = UniformPoint(int(N), int(M), method)
    return np.asarray(W, dtype=float), int(n)


def ga(problem, parents, params=None, rng=None) -> np.ndarray:
    """SBX + polynomial mutation, 2*floor(n/2) offspring (decision matrix in, matrix out)."""
    return np.asarray(OperatorGA(problem, np.asarray(parents), params, rng=rng), dtype=float)


def ga_half(problem, parents, params=None, rng=None) -> np.ndarray:
    """Same as :func:`ga` but only ``floor(n/2)`` offspring."""
    return np.asarray(OperatorGAhalf(problem, np.asarray(parents), params, rng=rng), dtype=float)


def de(problem, p1, p2, p3, params=None, rng=None) -> np.ndarray:
    return np.asarray(OperatorDE(problem, np.asarray(p1), np.asarray(p2), np.asarray(p3), params, rng=rng), dtype=float)


def pso(problem, particle, pbest, gbest, w: float = 0.4, rng=None):
    return OperatorPSO(problem, particle, pbest, gbest, w, rng=rng)


def first_front(F, C=None) -> np.ndarray:
    """Boolean mask of the first non-dominated front."""
    fn, _ = nd_sort(F, C, 1)
    return fn == 1


def truncate_lexi(Dist, K) -> np.ndarray:
    """Repeatedly delete the row whose ascending-sorted distance vector is lexicographically smallest.

    Returns the boolean deletion mask (``K`` True entries)."""
    N = len(Dist)
    dele = np.zeros(N, bool)
    while dele.sum() < K:
        remain = np.where(~dele)[0]
        temp = np.sort(Dist[np.ix_(remain, remain)], axis=1)
        dele[remain[np.lexsort(temp.T[::-1])[0]]] = True
    return dele


def normalization(F, z, znad):
    """Hyperplane-intercept normalisation (extreme points by ASF). Returns ``(F_norm, z, znad)``."""
    F = np.asarray(F, dtype=float)
    N, M = F.shape
    z = np.minimum(z, F.min(axis=0))
    W = np.full((M, M), 1e-6)
    np.fill_diagonal(W, 1.0)
    with np.errstate(all="ignore"):
        asf = np.stack([np.max(np.abs((F - z) / (znad - z)) / W[i], axis=1) for i in range(M)], axis=1)
        extreme = np.argmin(asf, axis=0)
        try:
            hyper = np.linalg.solve(F[extreme] - z, np.ones(M))
        except np.linalg.LinAlgError:
            hyper = np.full(M, np.nan)
        a = 1.0 / hyper + z
    if np.any(np.isnan(a)) or np.any(a <= z):
        a = F.max(axis=0)
    znad = a
    with np.errstate(all="ignore"):
        Fn = (F - z) / (znad - z)
    return Fn, z, znad


def first_index_reaching(front_no, N) -> int:
    """Smallest front number f with ``#(front_no <= f) >= N`` (front numbers are positive integers)."""
    fn = np.asarray(front_no)
    fin = fn[np.isfinite(fn)].astype(int)
    cum = np.cumsum(np.bincount(fin, minlength=int(fin.max()) + 1)[1:])
    return int(np.argmax(cum >= N)) + 1


def velocity(pop) -> np.ndarray:
    """Particle velocities stored on the individuals (zeros when absent)."""
    v = pop.get("V")
    try:
        v = np.asarray(v, dtype=float)
        if v.ndim == 2 and v.shape[0] == len(pop):
            return v
    except (TypeError, ValueError):
        pass
    return np.zeros((len(pop), np.asarray(pop.get("X")).shape[1]))


def neighbors_of(W, T) -> np.ndarray:
    """Indices of the T nearest weight vectors of every weight vector (itself first)."""
    return np.argsort(pdist2(W, W), axis=1, kind="stable")[:, :T]


def stm_select(F, W, z, znad, rng) -> np.ndarray:
    """Stable-matching selection between solutions and subproblems; returns one solution index per W row."""
    N, NW = len(F), len(W)
    with np.errstate(all="ignore"):
        g = np.max(np.abs(F - z)[:, None, :] / W[None, :, :], axis=2)                 # (N, NW)
        G = (F - z) / (znad - z)
    cos = 1.0 - cosine_distance(G, W)
    dist = np.linalg.norm(G, axis=1)[:, None] * np.sqrt(np.maximum(0.0, 1.0 - cos ** 2))
    Fp = np.full(NW, -1)
    FX = np.full(N, -1)
    phi = np.zeros((NW, N), bool)
    while np.any(Fp < 0):
        remain_w = np.where(Fp < 0)[0]
        i = int(remain_w[rng.integers(0, len(remain_w))])
        remain_x = np.where(~phi[i])[0]
        j = int(remain_x[np.argmin(g[remain_x, i])])
        phi[i, j] = True
        if FX[j] < 0:
            Fp[i], FX[j] = j, i
        elif dist[j, i] < dist[j, FX[j]]:
            Fp[FX[j]] = -1
            Fp[i], FX[j] = j, i
    return Fp


def grid_locations(F, div):
    """Adaptive-grid cell of every solution: ``(unique_cells, site_index, crowding_per_cell)``."""
    F = np.asarray(F, dtype=float)
    d = (F.max(axis=0) - F.min(axis=0)) / div
    with np.errstate(all="ignore"):
        loc = np.floor((F - F.min(axis=0)) / d)
    loc[loc >= div] = div - 1
    loc[np.isnan(loc)] = 0
    uniq, site = np.unique(loc, axis=0, return_inverse=True)
    site = site.reshape(-1)
    return uniq, site, np.bincount(site, minlength=len(uniq))


def kmeans(X, K, rng, n_iter: int = 100) -> np.ndarray:
    """Lloyd's k-means with k-means++ seeding; returns labels in ``0..K-1`` (empty clusters are re-seeded)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    K = int(min(K, n))
    C = X[[int(rng.integers(0, n))]]
    for _ in range(1, K):
        d2 = np.min(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        tot = d2.sum()
        C = np.vstack([C, X[[int(rng.choice(n, p=d2 / tot)) if tot > 0 else int(rng.integers(0, n))]]])
    lab = np.full(n, -1)
    for _ in range(n_iter):
        new = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for k in range(K):
            m = lab == k
            C[k] = X[m].mean(axis=0) if m.any() else X[int(rng.integers(0, n))]
    return lab


def adds(pop, key: str, default) -> np.ndarray:
    """Per-individual extra attribute matrix; individuals that do not carry ``key`` get the matching ``default`` row."""
    default = np.asarray(default, dtype=float)
    rows = []
    for i, ind in enumerate(pop):
        v = ind.get(key) if hasattr(ind, "get") else None
        rows.append(np.asarray(v, dtype=float).reshape(-1) if v is not None else default[i])
    return np.vstack(rows) if rows else default


def unique_individuals(pop):
    """Population without repeated individuals (identity, like unique() on a handle-object array)."""
    seen, keep = set(), []
    for i, ind in enumerate(pop):
        if id(ind) not in seen:
            seen.add(id(ind))
            keep.append(i)
    return pop[np.asarray(keep, dtype=int)]


def chebyshev_dist(A, B=None) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = A if B is None else np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[1] == 0:
        return np.zeros((len(A), len(B)))
    return np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)


def sort_rows_rank(*cols) -> np.ndarray:
    """Indices that sort lexicographically by ``cols`` (first column is the primary key); stable."""
    return np.lexsort(tuple(np.asarray(c, dtype=float).reshape(-1) for c in reversed(cols)))


# ----------------------------------------------------------------------------
# Base algorithm
# ----------------------------------------------------------------------------
class Terminated(Exception):
    """Raised by :meth:`LoopAlgorithm.not_terminated` once the evaluation budget is exhausted."""


class LoopAlgorithm(Algorithm):
    """MATLAB-style algorithm skeleton (see module docstring)."""

    def __init__(self, pop_size: int = 100, sampling: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling

    # -- problem shortcuts ------------------------------------------------
    @property
    def N(self) -> int:
        return self.pop_size

    @property
    def M(self) -> int:
        return int(self.problem.n_obj)

    @property
    def D(self) -> int:
        return int(self.problem.n_var)

    @property
    def lower(self) -> np.ndarray:
        return _u._problem_lower(self.problem, self.D)

    @property
    def upper(self) -> np.ndarray:
        return _u._problem_upper(self.problem, self.D)

    @property
    def encoding(self) -> np.ndarray:
        return _u._problem_encoding(self.problem, self.D)

    @property
    def rng(self) -> np.random.Generator:
        r = self.random_state
        if not isinstance(r, np.random.Generator):
            r = np.random.default_rng(self.seed)
            self.random_state = r
        return r

    @property
    def FE(self) -> int:
        return int(self.evaluator.n_eval)

    @property
    def max_FE(self) -> int:
        term = getattr(self, "termination", None)
        for attr in ("n_max_evals", "n_max_eval", "max_evals"):
            v = getattr(term, attr, None)
            if v is not None:
                return int(v)
        return max(1, self.pop_size * 100)

    # -- decision-space initialisation (uniform random, per encoding) ------
    def random_decs(self, n: int) -> np.ndarray:
        n, D, rng = int(n), self.D, self.rng
        lo, up, enc = self.lower, self.upper, self.encoding
        X = np.zeros((n, D), dtype=float)
        m12 = np.isin(enc, (1, 2))
        if m12.any():
            X[:, m12] = lo[m12] + rng.random((n, int(m12.sum()))) * (up[m12] - lo[m12])
            m2 = enc == 2
            if m2.any():
                X[:, m2] = np.clip(np.round(X[:, m2]), lo[m2], up[m2])
        m3 = enc == 3
        if m3.any():
            X[:, m3] = rng.integers(lo[m3].astype(int), up[m3].astype(int) + 1, size=(n, int(m3.sum())))
        m4 = enc == 4
        if m4.any():
            X[:, m4] = rng.random((n, int(m4.sum()))) < 0.5
        m5 = enc == 5
        if m5.any():
            k = int(m5.sum())
            X[:, m5] = np.argsort(rng.random((n, k)), axis=1) + 1
        return X

    def cal_dec(self, X) -> np.ndarray:
        """Repair a decision matrix like the reference platform: clip to the bounds (NaN falls back to
        a bound instead of propagating) and round integer / label / binary variables."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X = np.fmax(np.fmin(X, self.upper), self.lower)
        r = np.isin(self.encoding, (2, 3, 4))
        if r.any():
            X = X.copy()
            X[:, r] = np.round(X[:, r])
        return X

    def evaluate(self, X, **extras) -> Population:
        """Evaluate a decision matrix (charges ``len(X)`` function evaluations) -> evaluated Population.

        Extra per-solution attributes (e.g. ``V=velocities``) are stored on the individuals."""
        X = self.cal_dec(X)
        kv = [x for k, v in extras.items() for x in (k, v)]
        pop = Population.new("X", X, *kv)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        return pop

    def fep(self, pop) -> Population:
        """Fast evolutionary programming mutation (Cauchy step with self-adapted per-variable ``eta``)."""
        X = decs(pop)
        N, D = X.shape
        rng = self.rng
        eta = adds(pop, "eta", rng.random((N, D)))
        tau, tau1 = 1.0 / np.sqrt(2 * np.sqrt(D)), 1.0 / np.sqrt(2 * D)
        gi = np.repeat(rng.standard_normal((N, 1)), D, axis=1)
        gj = rng.standard_normal((N, D))
        return self.evaluate(X + eta * rng.standard_cauchy((N, D)), eta=eta * np.exp(tau1 * gi + tau * gj))

    def pso(self, particle, pbest, gbest, w: float = 0.4) -> Population:
        """Canonical PSO move (inertia ``w``, unit acceleration, per-variable random weights)."""
        X, P, G, V = decs(particle), decs(pbest), decs(gbest), velocity(particle)
        r1, r2 = self.rng.random(X.shape), self.rng.random(X.shape)
        vel = w * V + r1 * (P - X) + r2 * (G - X)
        return self.evaluate(X + vel, V=vel)

    def cal_obj(self, X) -> np.ndarray:
        """Raw objective values of ``X`` that are *not* charged to the evaluation budget (analysis phases)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(self.problem.do(X, ["F"])["F"], dtype=float)

    def solutions_uncharged(self, X) -> Population:
        """Evaluated solutions of ``X`` whose evaluations are *not* charged to the budget (objectives and constraints
        computed directly from the problem, like building solutions from ``CalObj``/``CalCon``)."""
        X = self.cal_dec(X)
        names = ["F", "G", "H"]
        n0 = getattr(self.problem, "n_fe", 0)
        out = self.problem.evaluate(X, return_values_of=names, return_as_dictionary=True)
        pop = Population.new("X", X)
        for k, v in out.items():
            if v is not None:
                pop.set(k, v)
        pop.apply(lambda ind: ind.evaluated.update(k for k, v in out.items() if v is not None))
        self.problem.n_fe = n0                     # the problem-level counter is not charged either
        return pop

    def cal_grad(self, dec):
        """Forward finite-difference gradients at ``dec``: ``(ObjGrad[M, D], ConGrad[C, D])``.

        Charges ``D + 1`` function evaluations (the base point and one perturbed point per variable)."""
        x = np.asarray(dec, dtype=float).reshape(-1).copy()
        x[x == 0] = 1e-12
        D = x.size
        p1 = self.evaluate(x[None, :])
        p2 = self.evaluate(np.tile(x, (D, 1)) * (1 + np.eye(D) * 1e-6))
        obj_grad = (objs(p2) - objs(p1)).T / x / 1e-6
        c1, c2 = cons(p1), cons(p2)
        con_grad = (c2 - c1).T / x / 1e-6 if c1.shape[1] else np.zeros((0, D))
        return obj_grad, con_grad

    # -- framework glue ----------------------------------------------------
    def initial_size(self) -> int:
        return self.pop_size

    def _initialize_infill(self) -> Population:
        return Population.new("X", self.random_decs(self.initial_size()))

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        self.pop = infills
        self.start()
        self._set_optimum()

    def _infill(self):
        return None  # solutions are created and evaluated inside step()

    PER_STEP_OPTIMUM = True   # steady-state algorithms (one offspring per step) set this to False

    def not_terminated(self, pop) -> bool:
        """Multi-phase algorithms call this where the reference calls its termination test: the population is
        recorded as the current result and the whole run ends (``Terminated``) as soon as the budget is spent."""
        self.pop = pop
        if self.FE >= self.max_FE:
            raise Terminated()
        return True

    def _advance(self, infills=None, **kwargs) -> None:
        try:
            self.step()
        except Terminated:
            pass
        if self.PER_STEP_OPTIMUM:
            self._set_optimum()

    def _finalize(self):
        self._set_optimum()

    def _set_optimum(self) -> None:
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    # -- to be overridden --------------------------------------------------
    def start(self) -> None:  # pragma: no cover - optional hook
        pass

    def step(self) -> None:  # pragma: no cover - must be provided
        raise NotImplementedError


def gde3_selection(pop, off, N) -> Population:
    """GDE3 survivor selection: one-to-one replacement (constraint-domination) plus non-dominated truncation."""
    PF, PC = objs(pop), cons(pop)
    OF, OC = objs(off), cons(off)
    fp = np.all(PC <= 0, axis=1)
    fo = np.all(OC <= 0, axis=1)
    updated = (~fp & fo) | (~fp & ~fo & np.all(PC >= OC, axis=1)) | (fp & fo & np.all(PF >= OF, axis=1))
    selected = fp & fo & np.any(PF < OF, axis=1) & np.any(PF > OF, axis=1)
    pop = pop.copy(deep=False)
    pop[updated] = off[updated]
    pop = Population.merge(pop, off[selected])
    F, C = objs(pop), cons(pop)
    feasible = np.all(C <= 0, axis=1)
    front = np.full(len(pop), np.inf)
    max_f = 0
    if feasible.any():
        front[feasible], max_f = nd_sort(F[feasible], None, np.inf)
    if (~feasible).any():
        front[~feasible] = nd_sort(C[~feasible], None, np.inf)[0] + max_f
    max_f = first_index_reaching(front, N)
    last = list(np.where(front == max_f)[0])
    n_before = int(np.sum(front < max_f))
    while len(last) > N - n_before:
        last.pop(int(np.argmin(crowding(F[last]))))
    return pop[np.concatenate([np.where(front < max_f)[0], np.asarray(last, dtype=int)]).astype(int)]


def gradient_direction(algo, pop_i, w):
    """Finite-difference descent gradient of one solution: the weighted objective gradient when it is feasible,
    otherwise the summed gradient of its violated constraints.  Returns ``(gradient[D], site[D])``; ``site`` marks
    the variables on which the objectives have conflicting gradient signs (feasible case)."""
    x = np.asarray(pop_i.X, dtype=float)
    con = np.asarray(pop_i.G, dtype=float).reshape(-1) if pop_i.G is not None else np.zeros(0)
    og, cg = algo.cal_grad(x)
    if con.size and not np.all(con <= 0):
        cg = cg.copy()
        cg[con <= 0] = 0.0
        return cg.sum(axis=0), np.zeros(len(x), bool)
    df = og.T                                                     # (D, M)
    site = np.any(df < 0, axis=1) & np.any(df > 0, axis=1)
    return df @ np.asarray(w, dtype=float), site


def polynomial_mutation(X, lower, upper, rng, disM: float = 20.0, prob=None) -> np.ndarray:
    """Clip to the bounds, then apply polynomial mutation to every variable with probability ``prob`` (default 1/D)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, D = X.shape
    lo = np.broadcast_to(np.asarray(lower, dtype=float), X.shape)
    up = np.broadcast_to(np.asarray(upper, dtype=float), X.shape)
    site = rng.random((n, D)) < (1.0 / D if prob is None else prob)
    mu = rng.random((n, D))
    X = np.maximum(np.minimum(X, up), lo)
    span = up - lo
    with np.errstate(all="ignore"):
        t = site & (mu <= 0.5)
        X[t] = X[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - (X[t] - lo[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        t = site & (mu > 0.5)
        X[t] = X[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (up[t] - X[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return X
