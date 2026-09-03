# emopylab 2026
"""LERD (large-scale evolutionary algorithm with reformulated decision variable analysis).

Reference:
C. He, R. Cheng, L. Li, K. C. Tan, and Y. Jin. Large-scale multiobjective
optimization via reformulated decision variable analysis. IEEE Transactions
on Evolutionary Computation, 2024, 28(1): 47-61.
"""

from __future__ import annotations

from util.array_backend import (
    get_array_module,
    to_device,
    to_numpy,
)
from util.array_backend import xp as np

from algorithms.community_utils.moead_family import (
    max_fe,
    set_optimum_from_pop,
    tchebycheff_values,
    weight_vectors,
)
from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorDE import OperatorDE
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from operators.utility_functions.UniformPoint import UniformPoint
from operators.sampling.rnd import FloatRandomSampling


ALGORITHM_FLAGS = {"LERD": {"large", "many", "multi", "real"}}


def _rng(algo):
    rng = getattr(algo, "random_state", None)
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        rng = np.random.default_rng()
        algo.random_state = rng
        return rng
    return np.random.default_rng(int(rng))


def _sample_initial(problem, n, sampling, rng):
    if sampling is None:
        sampling = FloatRandomSampling()
    return sampling.do(problem, int(n), random_state=rng)


def _remaining_eval_budget(evaluator, max_evals: int | None) -> int:
    """Return objective evaluations still available under a fixed FE budget."""
    if max_evals is None or int(max_evals) <= 0:
        return 10**12
    used = int(getattr(evaluator, "n_eval", 0) or 0)
    return max(0, int(max_evals) - used)


def _can_evaluate(evaluator, max_evals: int | None, n: int = 1) -> bool:
    return _remaining_eval_budget(evaluator, max_evals) >= int(max(1, n))


def _environment_selection(population: Population, n: int) -> Population:
    """NSGA-II environmental selection."""
    pop_obj = np.asarray(population.get("F"), dtype=float)
    front_no, max_f_no = NDSort(pop_obj, n)
    selected = front_no < max_f_no
    crowd_dis = CrowdingDistance(pop_obj, front_no)
    last = np.where(front_no == max_f_no)[0]
    rank = np.argsort(-crowd_dis[last])
    n_last = n - int(np.sum(selected))
    if n_last > 0:
        selected[last[rank[:n_last]]] = True
    return population[selected]


def _fitness_selection(fitness: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-II selection on fitness vectors."""
    front_no, max_f_no = NDSort(fitness, n)
    selected = front_no < max_f_no
    crowd_dis = CrowdingDistance(fitness, front_no)
    last = np.where(front_no == max_f_no)[0]
    rank = np.argsort(-crowd_dis[last])
    n_last = n - int(np.sum(selected))
    if n_last > 0:
        selected[last[rank[:n_last]]] = True
    return selected, front_no[selected], crowd_dis[selected]


def _evolve_by_moead(problem, population: Population, w: np.ndarray,
                     delta_g: int, evaluator, rng,
                     max_evals: int | None = None) -> Population:
    """MOEA/D-DE style uniformity optimization used by LERD."""
    pop_size = len(population)
    if pop_size == 0:
        return population

    f = np.asarray(population.get("F"), dtype=float)
    z = f.min(axis=0)

    # Weight scaling and neighbor detection.
    scaled_w = w * (f.max(axis=0) - f.min(axis=0))
    dist = np.linalg.norm(scaled_w[:, None, :] - scaled_w[None, :, :], axis=2)
    t = max(1, int(np.ceil(pop_size / 10)))
    b = np.argsort(dist, axis=1)[:, :t]

    # Associate each subproblem with one solution.
    g = np.zeros((pop_size, pop_size), dtype=float)
    for i in range(pop_size):
        g[i, :] = np.max(np.abs(f[i] - z) / np.maximum(w, 1e-12), axis=1)
    rank = np.argsort(g, axis=1)
    associate = np.full(pop_size, -1, dtype=int)
    for i in range(pop_size):
        for idx in rank[i]:
            if associate[idx] == -1:
                associate[idx] = i
                break
    # ``population[associate]`` already returns a fresh container of references,
    # so we copy it once up front and then mutate slots in place.  The previous
    # code copied the entire population on every neighbour replacement
    # (O(pop_size) per update, O(pop_size^2) per generation); copying once is
    # behaviourally identical and removes that overhead.
    pop = population[associate].copy()

    for _ in range(max(1, int(delta_g))):
        for i in range(pop_size):
            if not _can_evaluate(evaluator, max_evals, 1):
                return pop
            if rng.random() < 0.9:
                p = b[i, rng.permutation(b.shape[1])]
            else:
                p = rng.permutation(pop_size)

            parent_i = pop[[i]]
            parent_p1 = pop[[int(p[0])]]
            parent_p2 = pop[[int(p[1])]]
            # Pass decision vectors so the operator does not evaluate internally
            # (that path bypasses the evaluator and leaves the offspring
            # uncounted).  Evaluate explicitly through the evaluator so every
            # MOEA/D offspring is charged to the n_eval budget.
            off_dec = OperatorDE(
                problem, parent_i.get("X"), parent_p1.get("X"), parent_p2.get("X"), rng=rng
            )
            if hasattr(off_dec, "get"):
                off_dec = off_dec.get("X")
            off_dec = np.atleast_2d(np.asarray(off_dec, dtype=float))
            offspring = evaluator.eval(problem, Population.new("X", off_dec))
            off_f = np.asarray(offspring.get("F"), dtype=float).reshape(-1)
            z = np.minimum(z, off_f)

            pool = pop[p]
            pool_f = np.asarray(pool.get("F"), dtype=float)
            g_old = np.max(np.abs(pool_f - z) / np.maximum(w[p, :], 1e-12), axis=1)
            g_new = np.max(np.abs(off_f - z) / np.maximum(w[p, :], 1e-12), axis=1)
            replace = np.where(g_old >= g_new)[0]
            for r in replace:
                pop[p[r]] = offspring[0]
    return pop


def _cal_con(range_obj: np.ndarray, pop_obj: np.ndarray) -> np.ndarray:
    """Convergence measure used inside LERD."""
    norm = (pop_obj - range_obj[0]) / np.maximum(range_obj[1] - range_obj[0], 1e-6)
    return np.sum(norm, axis=1)


def _sparse_fit(problem, population: Population, par_dec: np.ndarray,
                s_n: int, evaluator, rng,
                max_evals: int | None = None) -> tuple[Population, np.ndarray]:
    """Evaluate a set of sparse variable masks."""
    f = np.asarray(population.get("F"), dtype=float)
    range_obj = np.vstack([f.min(axis=0), f.max(axis=0)])

    d_min = np.asarray(population.get("X"), dtype=float).min(axis=0)
    d_max = np.asarray(population.get("X"), dtype=float).max(axis=0)
    upper = np.tile(d_max, (s_n, 1))
    lower = np.tile(d_min, (s_n, 1))

    con = _cal_con(range_obj, f)
    best_idx = int(np.argmin(con))
    arc_dec = np.tile(np.asarray(population[best_idx].get("X"), dtype=float), (s_n, 1))

    n_masks, d = par_dec.shape
    alpha = 0.25
    all_pop = Population()
    fitness = np.full((n_masks, 2), np.inf, dtype=float)

    for i in range(n_masks):
        remaining = _remaining_eval_budget(evaluator, max_evals)
        if remaining <= 0:
            break
        n_trials = min(int(s_n), remaining)
        mask = par_dec[i].astype(bool)
        temp_dec = arc_dec[:n_trials].copy()
        noise = alpha * rng.random((n_trials, d)) * (upper[:n_trials] - lower[:n_trials])
        temp_dec[:, mask] = noise[:, mask]
        temp = evaluator.eval(problem, Population.new("X", temp_dec))
        temp_f = np.asarray(temp.get("F"), dtype=float)
        fitness[i, 0] = float(np.min(_cal_con(range_obj, temp_f)))
        fitness[i, 1] = float(np.sum(mask))
        all_pop = Population.merge(all_pop, temp)

    return all_pop, fitness


def _binary_variation(parent1: np.ndarray, parent2: np.ndarray, rng) -> np.ndarray:
    """One-point crossover and bitwise mutation for binary masks."""
    pro_c, pro_m = 1.0, 1.0
    n, d = parent1.shape
    cut = rng.integers(0, d, size=n)
    k = np.tile(np.arange(d), (n, 1)) > cut[:, None]
    skip = rng.random((n, 1)) > pro_c
    k = k & (~skip)

    off1 = parent1.copy()
    off2 = parent2.copy()
    off1[k] = parent2[k]
    off2[k] = parent1[k]
    offspring = np.vstack([off1, off2])

    site = rng.random((2 * n, d)) < pro_m / max(d, 1)
    offspring[site] = ~offspring[site]
    return offspring.astype(parent1.dtype)


def _second_optimization(problem, population: Population, variables: np.ndarray,
                         flag: int, evaluator, rng,
                         max_evals: int | None = None) -> Population:
    """Convergence/distribution optimization on a variable subset."""
    n = len(population)
    if n == 0:
        return population
    f = np.asarray(population.get("F"), dtype=float)
    range_obj = np.vstack([f.min(axis=0), f.max(axis=0)])

    mating_pool = TournamentSelection(2, n, _cal_con(range_obj, f), rng=rng) - 1
    off_dec = np.asarray(population[mating_pool].get("X"), dtype=float)

    dec = np.asarray(population.get("X"), dtype=float)
    if flag == 0:
        new_dec = OperatorGA(problem, population[rng.integers(0, n, size=n + 1)].get("X"), rng=rng)
        if hasattr(new_dec, "get"):
            new_dec = new_dec.get("X")
        new_dec = np.asarray(new_dec, dtype=float)[:n, :]
    else:
        next_idx = rng.integers(0, n, size=2 * n)
        p1 = dec[next_idx[:n]]
        p2 = dec[next_idx[n:]]
        param = [1.0, 0.5, range_obj.shape[1] / max(len(variables), 1) / 2.0, 20.0]
        new_dec = OperatorDE(problem, dec, p1, p2, Parameter=param, rng=rng)
        if hasattr(new_dec, "get"):
            new_dec = new_dec.get("X")
        new_dec = np.asarray(new_dec, dtype=float)

    off_dec[:, variables] = new_dec[:, variables]
    remaining = _remaining_eval_budget(evaluator, max_evals)
    if remaining <= 0:
        return population
    if len(off_dec) > remaining:
        off_dec = off_dec[:remaining]
    offspring = evaluator.eval(problem, Population.new("X", off_dec))
    return _environment_selection(Population.merge(population, offspring), n)


def _reformulated_optimization(problem, population: Population, par_dec: np.ndarray,
                               par_fit: np.ndarray, s_n: int, gen: int,
                               evaluator, rng,
                               max_evals: int | None = None) -> tuple[Population, np.ndarray, np.ndarray]:
    """LERD mask-level optimization."""
    all_pop, fit = _sparse_fit(problem, population, par_dec, s_n, evaluator, rng, max_evals)
    n_masks = par_dec.shape[0]
    if len(all_pop) == 0 or not np.any(np.isfinite(fit[:, 0])):
        return all_pop, par_dec, par_fit
    selected, front_no, crowd_dis = _fitness_selection(fit, n_masks)
    par_fit = fit

    for _ in range(gen):
        if _remaining_eval_budget(evaluator, max_evals) <= 0:
            break
        mating_pool = TournamentSelection(2, 2 * n_masks, front_no, -crowd_dis, rng=rng) - 1
        off_mask = _binary_variation(par_dec[mating_pool[:n_masks]], par_dec[mating_pool[n_masks:]], rng)
        off_pop, mask_fit = _sparse_fit(problem, population, off_mask, s_n, evaluator, rng, max_evals)
        if len(off_pop) == 0 or not np.any(np.isfinite(mask_fit[:, 0])):
            break

        all_pop = Population.merge(all_pop, off_pop)
        fitness = np.vstack([par_fit, mask_fit])
        pop_dec = np.vstack([par_dec, off_mask])

        selected, front_no, crowd_dis = _fitness_selection(fitness, n_masks)
        par_dec = pop_dec[selected]
        par_fit = fitness[selected]

    return all_pop, par_dec, par_fit


class LERD(Algorithm):
    def __init__(self, pop_size: int = 100, sampling=None,
                 s_n: int = 3, n_dva: int = 10, gen_dva: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling
        self.s_n = int(s_n)
        self.n_dva = int(n_dva)
        self.gen_dva = int(gen_dva)

    def _eval_budget_limit(self) -> int | None:
        try:
            return int(max_fe(self))
        except Exception:
            return None

    def _remaining_eval_budget(self) -> int:
        return _remaining_eval_budget(self.evaluator, self._eval_budget_limit())

    def _initialize_infill(self):
        rng = _rng(self)
        n_initial = min(self.pop_size, self._remaining_eval_budget())
        if n_initial <= 0:
            return Population()
        return _sample_initial(self.problem, n_initial, self.sampling, rng)

    def _initialize_advance(self, infills=None, **kwargs):
        rng = _rng(self)
        if infills is None or len(infills) == 0:
            self.pop = Population()
            return
        self.pop = infills
        self.pop_size = min(self.pop_size, len(self.pop))
        self.w, _ = weight_vectors(self.pop_size, self.problem.n_obj)
        self.pop_size = self.w.shape[0]
        if len(self.pop) > self.pop_size:
            self.pop = _environment_selection(self.pop, self.pop_size)
        self.pop = _evolve_by_moead(
            self.problem, self.pop, self.w, 20, self.evaluator, rng,
            max_evals=self._eval_budget_limit(),
        )

        d = self.problem.n_var
        self.par_dec = rng.random((self.n_dva, d)) > 0.5
        self.par_fit = np.zeros((self.n_dva, 2), dtype=float)
        self.par = {"N": self.n_dva, "sN": self.s_n, "gen": self.gen_dva, "t": 1}

    def _infill(self):
        rng = _rng(self)
        self._prev_pop = self.pop.copy()
        self._start_fe = int(self.evaluator.n_eval)
        if self._remaining_eval_budget() <= 0:
            self._new_pop = Population()
            return self._new_pop

        all_pop, self.par_dec, self.par_fit = _reformulated_optimization(
            self.problem, self.pop, self.par_dec, self.par_fit,
            self.s_n, self.gen_dva, self.evaluator, rng,
            max_evals=self._eval_budget_limit(),
        )
        self._new_pop = all_pop
        return all_pop

    def _advance(self, infills=None, **kwargs):
        rng = _rng(self)
        merged = Population.merge(self._prev_pop, self._new_pop)
        self.pop = _environment_selection(merged, self.pop_size) if len(merged) > self.pop_size else merged

        for g in range(self.par_dec.shape[0]):
            if self._remaining_eval_budget() <= 0:
                break
            dv = np.where(self.par_dec[g])[0]
            pv = np.where(~self.par_dec[g])[0]
            if len(dv) > 0:
                self.pop = _second_optimization(
                    self.problem, self.pop, dv, 1, self.evaluator, rng,
                    max_evals=self._eval_budget_limit(),
                )
            if self._remaining_eval_budget() <= 0:
                break
            if len(pv) > 0:
                self.pop = _second_optimization(
                    self.problem, self.pop, pv, 1, self.evaluator, rng,
                    max_evals=self._eval_budget_limit(),
                )

        delta_fe = max(1, int(np.ceil((self.evaluator.n_eval - self._start_fe) / self.pop_size)))
        if self._remaining_eval_budget() > 0:
            self.pop = _evolve_by_moead(
                self.problem, self.pop, self.w, delta_fe, self.evaluator, rng,
                max_evals=self._eval_budget_limit(),
            )

    def _set_optimum(self):
        set_optimum_from_pop(self)
