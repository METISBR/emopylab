# emopylab 2026
"""Community algorithm utility functions (MATLAB-style signatures) for EmoPyLab.

These helpers provide Python/NumPy implementations of common utility routines
used by community MATLAB multi-objective algorithms. They are designed for
compatibility with EmoPyLab algorithm ports and reuse EmoPyLab primitives where
appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any

import numpy as np
from core.population import Population
from util.nds.non_dominated_sorting import NonDominatedSorting
from operators.survival.rank_and_crowding.metrics import calc_crowding_distance


def _rng(rng=None):
    return rng if rng is not None else np.random.default_rng()


def _is_solution_array(x: Any) -> bool:
    if hasattr(x, "decs"):
        return True
    if hasattr(x, "get") and callable(x.get):
        try:
            return x.get("X") is not None
        except Exception:
            return True
    if hasattr(x, '__len__') and len(x) > 0:
        first = x[0]
        return hasattr(first, 'decs') or hasattr(first, 'X') or (hasattr(first, "get") and callable(first.get))
    return False


def _extract_decs(x: Any) -> np.ndarray:
    if hasattr(x, 'decs'):
        return np.asarray(x.decs)
    if hasattr(x, 'get'):
        X = x.get('X')
        if X is not None:
            return np.asarray(X)
    return np.asarray(x)


def _extract_objs(x: Any) -> np.ndarray:
    if hasattr(x, 'objs'):
        return np.asarray(x.objs)
    if hasattr(x, 'get'):
        F = x.get('F')
        if F is not None:
            return np.asarray(F)
    return np.asarray(x)


def _extract_adds(x: Any, default=None) -> np.ndarray:
    if hasattr(x, 'adds') and callable(x.adds):
        out = x.adds(default)
        if out is not None:
            return np.asarray(out)
    if hasattr(x, 'get'):
        for key in ('V', 'vel', 'velocity'):
            val = x.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.dtype == object and arr.size > 0 and all(v is None for v in arr.reshape(-1)):
                    continue
                return np.asarray(val)
    return np.asarray(default) if default is not None else None


def _problem_lower(problem: Any, D: int | None = None) -> np.ndarray:
    if hasattr(problem, 'lower'):
        arr = np.asarray(problem.lower, dtype=float)
        if arr.ndim == 0 and D is not None:
            arr = np.full(D, float(arr))
        return arr
    if hasattr(problem, 'xl'):
        arr = np.asarray(problem.xl, dtype=float)
        if arr.ndim == 0 and D is not None:
            arr = np.full(D, float(arr))
        return arr
    if D is None:
        raise AttributeError('Problem lower bound not available')
    return np.zeros(D, dtype=float)


def _problem_upper(problem: Any, D: int | None = None) -> np.ndarray:
    if hasattr(problem, 'upper'):
        arr = np.asarray(problem.upper, dtype=float)
        if arr.ndim == 0 and D is not None:
            arr = np.full(D, float(arr))
        return arr
    if hasattr(problem, 'xu'):
        arr = np.asarray(problem.xu, dtype=float)
        if arr.ndim == 0 and D is not None:
            arr = np.full(D, float(arr))
        return arr
    if D is None:
        raise AttributeError('Problem upper bound not available')
    return np.ones(D, dtype=float)


def _problem_encoding(problem: Any, D: int) -> np.ndarray:
    enc = getattr(problem, 'encoding', None)
    if enc is None:
        vtype = getattr(problem, 'vtype', float)
        if isinstance(vtype, np.ndarray):
            # best-effort: bool/int/float map to binary/integer/real
            out = np.ones(D, dtype=int)
            for i, t in enumerate(vtype[:D]):
                if t in (bool, np.bool_):
                    out[i] = 4
                elif t in (int, np.int32, np.int64):
                    out[i] = 2
                else:
                    out[i] = 1
            return out
        if vtype in (bool, np.bool_):
            return np.full(D, 4, dtype=int)
        if vtype in (int, np.int32, np.int64):
            return np.full(D, 2, dtype=int)
        return np.full(D, 1, dtype=int)
    arr = np.asarray(enc, dtype=int).reshape(-1)
    if arr.size == 1:
        arr = np.full(D, int(arr[0]), dtype=int)
    return arr[:D]


def _problem_evaluation(problem: Any, *args) -> Any:
    if hasattr(problem, 'Evaluation') and callable(problem.Evaluation):
        return problem.Evaluation(*args)
    if not (hasattr(problem, "evaluate") and callable(problem.evaluate)):
        raise AttributeError('Problem.Evaluation or problem.evaluate is required for these community utility functions.')
    if len(args) < 1:
        raise ValueError("At least decision variables X must be provided for problem evaluation.")
    X = np.asarray(args[0], dtype=float)
    out = problem.evaluate(X, return_as_dictionary=True)
    pop = Population.new("X", X)
    if isinstance(out, dict):
        for k, v in out.items():
            if v is not None:
                pop.set(k, v)
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))
    return pop

def nd_sort(PopObj, PopCon=None, n_sort=np.inf):
    F = np.asarray(PopObj, dtype=float)
    if F.ndim != 2:
        F = np.atleast_2d(F)
    N, M = F.shape

    if PopCon is not None:
        C = np.asarray(PopCon, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        infeasible = np.any(C > 0, axis=1)
        if np.any(infeasible):
            penalty = np.sum(np.maximum(0.0, C[infeasible]), axis=1)[:, None]
            fmax = np.max(F, axis=0, keepdims=True)
            F = F.copy()
            F[infeasible, :] = fmax + penalty

    if not np.isfinite(float(n_sort)):
        stop = N
    else:
        stop = int(max(1, min(N, int(n_sort))))

    nds = NonDominatedSorting(method="efficient_non_dominated_sort")
    fronts = nds.do(F, n_stop_if_ranked=stop)

    front_no = np.full(N, np.inf, dtype=float)
    max_f = 0
    ranked = 0
    for i, front in enumerate(fronts, start=1):
        if len(front) == 0:
            continue
        front = np.asarray(front, dtype=int)
        front_no[front] = float(i)
        max_f = i
        ranked += len(front)
        if ranked >= stop:
            break
    return front_no, int(max_f)


def crowding_distance(PopObj, FrontNo=None):
    F = np.asarray(PopObj, dtype=float)
    if F.ndim != 2:
        F = np.atleast_2d(F)
    N, _M = F.shape
    if FrontNo is None:
        FrontNo = np.ones(N, dtype=float)
    FrontNo = np.asarray(FrontNo, dtype=float).reshape(-1)
    cd = np.zeros(N, dtype=float)

    finite_fronts = [f for f in np.unique(FrontNo) if np.isfinite(f)]
    for f in finite_fronts:
        idx = np.where(FrontNo == f)[0]
        if idx.size == 0:
            continue
        if idx.size <= 2:
            cd[idx] = np.inf
            continue
        cd[idx] = calc_crowding_distance(F[idx])
    return cd


def fitness_single(Population):
    """Fitness calculation for single-objective optimization (Deb constraint rule).

    MATLAB-compatible behavior:
    - Feasible solutions use the objective value directly.
    - Infeasible solutions use summed positive constraint violation plus a large penalty.
    """
    if hasattr(Population, "objs"):
        pop_obj = np.asarray(Population.objs, dtype=float)
    elif hasattr(Population, "get"):
        pop_obj = Population.get("F")
        if pop_obj is None:
            raise AttributeError("Population objective values are required (objs/F).")
        pop_obj = np.asarray(pop_obj, dtype=float)
    else:
        raise AttributeError("Population-like input with objs/F is required.")

    if pop_obj.ndim == 1:
        obj = pop_obj.reshape(-1)
    elif pop_obj.shape[1] == 1:
        obj = pop_obj[:, 0]
    else:
        # MATLAB FitnessSingle is for single-objective; use the first column by convention if provided.
        obj = pop_obj[:, 0]

    if hasattr(Population, "cons"):
        cons = np.asarray(Population.cons, dtype=float)
    elif hasattr(Population, "get"):
        cons = Population.get("G")
        if cons is None:
            cons = np.zeros((obj.shape[0], 0), dtype=float)
        else:
            cons = np.asarray(cons, dtype=float)
    else:
        cons = np.zeros((obj.shape[0], 0), dtype=float)

    if cons.ndim == 1:
        cons = cons[:, None]

    pop_con = np.sum(np.maximum(0.0, cons), axis=1) if cons.size else np.zeros(obj.shape[0], dtype=float)
    feasible = pop_con <= 0.0
    fitness = feasible.astype(float) * obj + (~feasible).astype(float) * (pop_con + 1e10)
    return fitness


def tournament_selection(K, N, *fitness_args, rng=None):
    if len(fitness_args) == 0:
        raise ValueError('At least one fitness vector is required.')
    K = int(K)
    N = int(N)
    rng = _rng(rng)

    cols = [np.asarray(f, dtype=float).reshape(-1, 1) for f in fitness_args]
    n_pop = cols[0].shape[0]
    for c in cols[1:]:
        if c.shape[0] != n_pop:
            raise ValueError('All fitness vectors must have the same length.')
    fit = np.hstack(cols)

    uniq, inv = np.unique(fit, axis=0, return_inverse=True)
    order = np.lexsort(tuple(uniq[:, j] for j in range(uniq.shape[1]-1, -1, -1)))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    rank_by_ind = rank[inv]

    parents = rng.integers(0, n_pop, size=(K, N))
    local_ranks = rank_by_ind[parents]
    best = np.argmin(local_ranks, axis=0)
    return parents[best, np.arange(N)] + 1  # MATLAB-style 1-based indices


def roulette_wheel_selection(N, Fitness, rng=None):
    rng = _rng(rng)
    fit = np.asarray(Fitness, dtype=float).reshape(-1)
    fit = fit - min(np.min(fit), 0.0) + 1e-6
    probs = 1.0 / fit
    probs = probs / np.sum(probs)
    idx = rng.choice(fit.size, size=int(N), replace=True, p=probs)
    return idx + 1  # MATLAB-style 1-based indices


def _nbi_points(N, M):
    H1 = 1
    while math.comb(H1 + M, M - 1) <= N:
        H1 += 1
    combs = np.array(list(combinations(range(1, H1 + M), M - 1)), dtype=int)
    if combs.size == 0:
        W = np.ones((1, M), dtype=float) / M
    else:
        offsets = np.arange(M - 1, dtype=int)
        W = combs - offsets - 1
        W = (np.hstack([W, np.full((W.shape[0], 1), H1)]) - np.hstack([np.zeros((W.shape[0], 1), dtype=int), W])) / H1
    if H1 < M:
        H2 = 0
        while math.comb(H1 + M - 1, M - 1) + math.comb(H2 + M, M - 1) <= N:
            H2 += 1
        if H2 > 0:
            combs2 = np.array(list(combinations(range(1, H2 + M), M - 1)), dtype=int)
            offsets2 = np.arange(M - 1, dtype=int)
            W2 = combs2 - offsets2 - 1
            W2 = (np.hstack([W2, np.full((W2.shape[0], 1), H2)]) - np.hstack([np.zeros((W2.shape[0], 1), dtype=int), W2])) / H2
            W = np.vstack([W, W2 / 2.0 + 1.0 / (2.0 * M)])
    W = np.maximum(W, 1e-6)
    return W, int(W.shape[0])


def _ild_points(N, M):
    I = M * np.eye(M, dtype=int)
    W = np.zeros((1, M), dtype=int)
    edgeW = W.copy()
    while W.shape[0] < N:
        edgeW = np.repeat(edgeW, M, axis=0) + np.tile(I, (edgeW.shape[0], 1))
        edgeW = np.unique(edgeW, axis=0)
        edgeW = edgeW[np.min(edgeW, axis=1) == 0]
        W = np.vstack([W + 1, edgeW])
    W = W / np.sum(W, axis=1, keepdims=True)
    W = np.maximum(W, 1e-6)
    return W, int(W.shape[0])


def _cal_cd2(UT):
    N, S = UT.shape
    X = (2 * UT - 1) / (2 * N)
    cs1 = np.sum(np.prod(2 + np.abs(X - 0.5) - (X - 0.5) ** 2, axis=1))
    cs2 = 0.0
    for i in range(N):
        Xi = np.tile(X[i], (N, 1))
        term = 1 + 0.5 * np.abs(Xi - 0.5) + 0.5 * np.abs(X - 0.5) - 0.5 * np.abs(Xi - X)
        cs2 += np.sum(np.prod(term, axis=1))
    return (13 / 12) ** S - (2 ** (1 - S)) / N * cs1 + cs2 / (N ** 2)


def _good_lattice_point(N, M):
    hm = np.array([i for i in range(1, N + 1) if math.gcd(i, N) == 1], dtype=int)
    udt = (np.outer(np.arange(1, N + 1), hm) % N)
    udt[udt == 0] = N
    n_comb = math.comb(len(hm), M) if len(hm) >= M else 0
    if 0 < n_comb < 10_000:
        best_cd2 = np.inf
        best_data = None
        for comb_idx in combinations(range(len(hm)), M):
            UT = udt[:, comb_idx]
            cd2 = _cal_cd2(UT)
            if cd2 < best_cd2:
                best_cd2 = cd2
                best_data = UT
        data = best_data
    else:
        best_cd2 = np.inf
        best_i = 1
        for i in range(1, N + 1):
            exps = np.array([i ** p for p in range(M)], dtype=object)
            UT = (np.outer(np.arange(1, N + 1), exps) % N).astype(int)
            cd2 = _cal_cd2(UT)
            if cd2 < best_cd2:
                best_cd2 = cd2
                best_i = i
        exps = np.array([best_i ** p for p in range(M)], dtype=object)
        data = (np.outer(np.arange(1, N + 1), exps) % N).astype(int)
        data[data == 0] = N
    data = (data - 1) / max(N - 1, 1)
    return data


def _mud_points(N, M):
    X = _good_lattice_point(int(N), int(M - 1))
    powers = np.arange(M - 1, 0, -1, dtype=float)
    X = np.power(np.maximum(X, 1e-12), 1.0 / powers)
    X = np.maximum(X, 1e-6)
    W = np.zeros((N, M), dtype=float)
    W[:, :-1] = (1 - X) * np.cumprod(X, axis=1) / X
    W[:, -1] = np.prod(X, axis=1)
    return W, int(N)


def _grid_points(N, M):
    g = int(np.ceil(N ** (1.0 / M)))
    gap = np.linspace(0.0, 1.0, g)
    mesh = np.meshgrid(*([gap] * M), indexing='ij')
    W = np.column_stack([m.reshape(-1) for m in mesh])
    return W, int(W.shape[0])


def _latin_points(N, M, rng=None):
    rng = _rng(rng)
    ranks = np.zeros((N, M), dtype=float)
    for j in range(M):
        perm = rng.permutation(N)
        ranks[:, j] = perm + 1
    W = (rng.random((N, M)) + ranks - 1) / N
    return W, int(N)


def uniform_point(N, M, method='NBI', rng=None):
    method = str(method or 'NBI').strip().lower()
    N = int(N)
    M = int(M)
    if method == 'nbi':
        return _nbi_points(N, M)
    if method == 'ild':
        return _ild_points(N, M)
    if method == 'mud':
        return _mud_points(N, M)
    if method == 'grid':
        return _grid_points(N, M)
    if method == 'latin':
        return _latin_points(N, M, rng=rng)
    raise ValueError(f'Unsupported UniformPoint method: {method}')


def _split_parents(parent_decs: np.ndarray):
    n = parent_decs.shape[0] // 2
    return parent_decs[:n].copy(), parent_decs[n:2*n].copy()


def _ga_real(parent1, parent2, lower, upper, proC, disC, proM, disM, half=False, rng=None):
    rng = _rng(rng)
    N, D = parent1.shape
    mu = rng.random((N, D))
    beta = np.zeros((N, D), dtype=float)
    mask = mu <= 0.5
    beta[mask] = (2 * mu[mask]) ** (1.0 / (disC + 1))
    beta[~mask] = (2 - 2 * mu[~mask]) ** (-1.0 / (disC + 1))
    beta *= np.where(rng.integers(0, 2, size=(N, D)) == 0, -1.0, 1.0)
    beta[rng.random((N, D)) < 0.5] = 1.0
    beta[np.repeat((rng.random((N, 1)) > proC), D, axis=1)] = 1.0

    c1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    if half:
        offspring = c1
    else:
        c2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
        offspring = np.vstack([c1, c2])

    lower = np.asarray(lower, dtype=float).reshape(1, -1)
    upper = np.asarray(upper, dtype=float).reshape(1, -1)
    lower = np.repeat(lower, offspring.shape[0], axis=0)
    upper = np.repeat(upper, offspring.shape[0], axis=0)

    offspring = np.minimum(np.maximum(offspring, lower), upper)
    site = rng.random(offspring.shape) < (proM / max(D, 1))
    mu = rng.random(offspring.shape)

    temp = site & (mu <= 0.5)
    if np.any(temp):
        denom = np.maximum(upper[temp] - lower[temp], 1e-32)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
            (2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - (offspring[temp] - lower[temp]) / denom) ** (disM + 1)) ** (1.0 / (disM + 1)) - 1
        )
    temp = site & (mu > 0.5)
    if np.any(temp):
        denom = np.maximum(upper[temp] - lower[temp], 1e-32)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - (upper[temp] - offspring[temp]) / denom) ** (disM + 1)) ** (1.0 / (disM + 1))
        )
    return np.minimum(np.maximum(offspring, lower), upper)


def _ga_binary(parent1, parent2, proC, proM, half=False, rng=None):
    rng = _rng(rng)
    N, D = parent1.shape
    parent1 = parent1.astype(bool, copy=True)
    parent2 = parent2.astype(bool, copy=True)
    k = rng.random((N, D)) < 0.5
    k[np.repeat((rng.random((N, 1)) > proC), D, axis=1)] = False
    o1 = parent1.copy()
    o1[k] = parent2[k]
    if half:
        offspring = o1
    else:
        o2 = parent2.copy(); o2[k] = parent1[k]
        offspring = np.vstack([o1, o2])
    site = rng.random(offspring.shape) < (proM / max(D, 1))
    offspring[site] = ~offspring[site]
    return offspring.astype(parent1.dtype)


def _ga_permutation(parent1, parent2, proC, half=False, rng=None):
    rng = _rng(rng)
    N, D = parent1.shape

    def _single_ox(a, b, cut):
        c = a.copy()
        if rng.random() < proC:
            prefix = a[:cut]
            tail = [x for x in b.tolist() if x not in set(prefix.tolist())]
            c[cut:] = np.asarray(tail, dtype=a.dtype)
        s = int(rng.integers(0, D))
        k = int(rng.integers(0, D))
        if s < k:
            c = c[np.r_[0:s, k, s:k, k+1:D]]
        elif s > k:
            c = c[np.r_[0:k, k+1:s, k, s:D]]
        return c

    cuts = rng.integers(0, D, size=N)
    o1 = np.vstack([_single_ox(parent1[i], parent2[i], int(cuts[i])) for i in range(N)])
    if half:
        return o1
    o2 = np.vstack([_single_ox(parent2[i], parent1[i], int(cuts[i])) for i in range(N)])
    return np.vstack([o1, o2])


def _ga_label(parent1, parent2, lower, upper, proC, proM, half=False, rng=None):
    # Best-effort label operator consistent with the MATLAB utility intent.
    rng = _rng(rng)
    N, D = parent1.shape
    parent1 = np.asarray(parent1, copy=True)
    parent2 = np.asarray(parent2, copy=True)
    k = rng.random((N, D)) < 0.5
    k[np.repeat((rng.random((N, 1)) > proC), D, axis=1)] = False
    o1 = parent1.copy(); o1[k] = parent2[k]
    if half:
        offspring = o1
    else:
        o2 = parent2.copy(); o2[k] = parent1[k]
        offspring = np.vstack([o1, o2])
    site = rng.random(offspring.shape) < (proM / max(D, 1))
    randv = rng.integers(int(np.min(lower)), int(np.max(upper)) + 1, size=offspring.shape)
    offspring[site] = randv[site]
    return offspring


def operator_ga(problem, Parent, Parameter=None, half=False, rng=None):
    rng = _rng(rng)
    if Parameter is not None:
        proC, disC, proM, disM = [float(x) for x in Parameter]
    else:
        proC, disC, proM, disM = 1.0, 20.0, 1.0, 20.0

    evaluated = _is_solution_array(Parent)
    parent = _extract_decs(Parent)
    parent = np.asarray(parent)
    p1, p2 = _split_parents(parent)
    if p1.size == 0:
        return _problem_evaluation(problem, parent) if evaluated else parent

    D = parent.shape[1]
    enc = _problem_encoding(problem, D)
    offspring = np.zeros(((p1.shape[0] if half else 2 * p1.shape[0]), D), dtype=parent.dtype if parent.dtype != object else float)
    lower = _problem_lower(problem, D)
    upper = _problem_upper(problem, D)

    idx_real_int = np.where(np.isin(enc, [1, 2]))[0]
    idx_label = np.where(enc == 3)[0]
    idx_binary = np.where(enc == 4)[0]
    idx_perm = np.where(enc == 5)[0]

    if idx_real_int.size:
        pm = proM * len(idx_real_int) / max(D, 1) if not half else proM
        offspring[:, idx_real_int] = _ga_real(p1[:, idx_real_int].astype(float), p2[:, idx_real_int].astype(float), lower[idx_real_int], upper[idx_real_int], proC, disC, pm, disM, half=half, rng=rng)
        # integer variables rounded after variation
        int_mask_local = enc[idx_real_int] == 2
        if np.any(int_mask_local):
            cols = idx_real_int[int_mask_local]
            offspring[:, cols] = np.round(offspring[:, cols])
            offspring[:, cols] = np.clip(offspring[:, cols], lower[cols], upper[cols])
    if idx_label.size:
        pm = proM * len(idx_label) / max(D, 1) if not half else proM
        offspring[:, idx_label] = _ga_label(p1[:, idx_label], p2[:, idx_label], lower[idx_label], upper[idx_label], proC, pm, half=half, rng=rng)
    if idx_binary.size:
        pm = proM * len(idx_binary) / max(D, 1) if not half else proM
        offspring[:, idx_binary] = _ga_binary(p1[:, idx_binary], p2[:, idx_binary], proC, pm, half=half, rng=rng)
    if idx_perm.size:
        offspring[:, idx_perm] = _ga_permutation(p1[:, idx_perm], p2[:, idx_perm], proC, half=half, rng=rng)

    if evaluated:
        return _problem_evaluation(problem, offspring)
    return offspring


def operator_de(problem, Parent1, Parent2, Parent3, Parameter=None, rng=None):
    rng = _rng(rng)
    if Parameter is not None:
        CR, F, proM, disM = [float(x) for x in Parameter]
    else:
        CR, F, proM, disM = 1.0, 0.5, 1.0, 20.0

    evaluated = _is_solution_array(Parent1)
    P1 = _extract_decs(Parent1).astype(float)
    P2 = _extract_decs(Parent2).astype(float)
    P3 = _extract_decs(Parent3).astype(float)
    N, D = P1.shape

    site = rng.random((N, D)) < CR
    off = P1.copy()
    off[site] = off[site] + F * (P2[site] - P3[site])

    lower = np.repeat(_problem_lower(problem, D).reshape(1, -1), N, axis=0)
    upper = np.repeat(_problem_upper(problem, D).reshape(1, -1), N, axis=0)
    off = np.minimum(np.maximum(off, lower), upper)

    site = rng.random((N, D)) < (proM / max(D, 1))
    mu = rng.random((N, D))
    temp = site & (mu <= 0.5)
    if np.any(temp):
        denom = np.maximum(upper[temp] - lower[temp], 1e-32)
        off[temp] = off[temp] + (upper[temp] - lower[temp]) * (
            (2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - (off[temp] - lower[temp]) / denom) ** (disM + 1)) ** (1.0 / (disM + 1)) - 1
        )
    temp = site & (mu > 0.5)
    if np.any(temp):
        denom = np.maximum(upper[temp] - lower[temp], 1e-32)
        off[temp] = off[temp] + (upper[temp] - lower[temp]) * (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - (upper[temp] - off[temp]) / denom) ** (disM + 1)) ** (1.0 / (disM + 1))
        )
    off = np.minimum(np.maximum(off, lower), upper)

    if evaluated:
        return _problem_evaluation(problem, off)
    return off


def operator_pso(problem, Particle, Pbest, Gbest, W=0.4, rng=None):
    rng = _rng(rng)
    particle_dec = _extract_decs(Particle).astype(float)
    pbest_dec = _extract_decs(Pbest).astype(float)
    gbest_dec = _extract_decs(Gbest).astype(float)
    N, D = particle_dec.shape
    particle_vel = _extract_adds(Particle, np.zeros((N, D), dtype=float)).astype(float)

    r1 = rng.random((N, D))
    r2 = rng.random((N, D))
    off_vel = float(W) * particle_vel + r1 * (pbest_dec - particle_dec) + r2 * (gbest_dec - particle_dec)
    off_dec = particle_dec + off_vel
    off = _problem_evaluation(problem, off_dec)
    if hasattr(off, "set"):
        off.set("V", off_vel)
    return off


def operator_fep(problem, Population, rng=None):
    rng = _rng(rng)
    dec = _extract_decs(Population).astype(float)
    N, D = dec.shape
    eta = _extract_adds(Population, rng.random((N, D))).astype(float)

    tau = 1.0 / np.sqrt(2 * np.sqrt(D))
    tau1 = 1.0 / np.sqrt(2 * D)
    gaussian_i = np.repeat(rng.standard_normal((N, 1)), D, axis=1)
    gaussian_j = rng.standard_normal((N, D))
    cauchy_j = rng.standard_cauchy((N, D))
    off_dec = dec + eta * cauchy_j
    off_eta = eta * np.exp(tau1 * gaussian_i + tau * gaussian_j)
    off = _problem_evaluation(problem, off_dec)
    if hasattr(off, "set"):
        off.set("eta", off_eta)
    return off
