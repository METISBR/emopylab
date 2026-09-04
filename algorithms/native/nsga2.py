"""EmoPyLab Native NSGA-II Solver in Pure Tensors."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.engine.runner import OptimizationResult
from core.nds.ens import efficient_non_dominated_sort
from core.nds.gpu_nds import boolean_matrix_nds
from core.operators.crossover.sbx import sbx_crossover_tensor
from core.operators.mutation.polynomial import polynomial_mutation_tensor
from core.operators.sampling.lhs import latin_hypercube_sampling
from core.operators.selection.tournament import binary_tournament_selection
from core.tensor.backend import get_array_module, index_tensor, init_tensor_backend, to_device, to_numpy, vstack
from core.tensor.population import TensorPopulation
from core.tensor.problem import TensorProblem
from metrics.evaluator import evaluate_front


def calc_crowding_distance(F_matrix: np.ndarray) -> np.ndarray:
    """Calculates Deb's Crowding Distance for a Pareto front."""
    N, M = F_matrix.shape
    if N <= 2:
        return np.full(N, np.inf, dtype=np.float32)

    crowding = np.zeros(N, dtype=np.float32)

    for m in range(M):
        order = np.argsort(F_matrix[:, m])
        crowding[order[0]] = np.inf
        crowding[order[-1]] = np.inf

        f_min = F_matrix[order[0], m]
        f_max = F_matrix[order[-1], m]
        denom = f_max - f_min
        if denom <= 1e-8:
            continue

        for i in range(1, N - 1):
            if np.isfinite(crowding[order[i]]):
                crowding[order[i]] += (F_matrix[order[i + 1], m] - F_matrix[order[i - 1], m]) / denom

    return crowding


class NativeNSGA2:
    """Pure Tensor-Native NSGA-II Solver with GPU/CPU auto-dispatch."""

    def __init__(
        self,
        pop_size: int = 100,
        crossover_eta: float = 15.0,
        crossover_prob: float = 0.9,
        mutation_eta: float = 20.0,
        mutation_prob: float | None = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.crossover_eta = float(crossover_eta)
        self.crossover_prob = float(crossover_prob)
        self.mutation_eta = float(mutation_eta)
        self.mutation_prob = mutation_prob

    def solve(
        self,
        problem: TensorProblem,
        n_gen: int = 250,
        seed: int = 42,
    ) -> OptimizationResult:
        t_start = time.perf_counter()
        init_tensor_backend()

        N = self.pop_size
        D = problem.n_var
        M = problem.n_obj

        # 1. Initialize population on device via LHS
        X_init = latin_hypercube_sampling(N, D, problem.xl_cpu, problem.xu_cpu, seed=seed)
        F_init, G_init = problem.evaluate(X_init)

        pop = TensorPopulation(N, D, M, problem.n_constr, X=X_init, F=F_init, G=G_init)

        # 2. Main generational loop
        for gen in range(1, n_gen + 1):
            # Evaluate fronts via ENS or Boolean NDS
            F_cpu = to_numpy(pop.F)
            fronts = efficient_non_dominated_sort(F_cpu)

            # Assign ranks and crowding
            ranks = np.zeros(N, dtype=np.int32)
            crowdings = np.zeros(N, dtype=np.float32)
            for r, front in enumerate(fronts):
                ranks[front] = r
                if len(front) > 0:
                    crowdings[front] = calc_crowding_distance(F_cpu[front])

            pop.rank = ranks
            pop.crowding = crowdings

            # Selection: Mating Pool
            mating_idx = binary_tournament_selection(ranks, crowdings, N, seed=seed + gen)
            p1_idx = mating_idx[::2]
            p2_idx = mating_idx[1::2]
            if len(p1_idx) != len(p2_idx):
                p1_idx = p1_idx[:len(p2_idx)]

            parent1 = index_tensor(pop.X, p1_idx)
            parent2 = index_tensor(pop.X, p2_idx)

            # Vectorized Crossover & Mutation in GPU
            off1, off2 = sbx_crossover_tensor(
                parent1, parent2, problem.xl_dev, problem.xu_dev,
                eta=self.crossover_eta, prob=self.crossover_prob, seed=seed + gen * 2,
            )
            offspring_X = vstack([off1, off2])
            offspring_X = polynomial_mutation_tensor(
                offspring_X, problem.xl_dev, problem.xu_dev,
                eta=self.mutation_eta, prob_var=self.mutation_prob, seed=seed + gen * 3,
            )

            offspring_F, offspring_G = problem.evaluate(offspring_X)
            offspring_pop = TensorPopulation(
                len(offspring_X), D, M, problem.n_constr,
                X=offspring_X, F=offspring_F, G=offspring_G,
            )

            # Merge 2N pool
            merged = TensorPopulation.merge(pop, offspring_pop)
            merged_F_cpu = to_numpy(merged.F)
            merged_fronts = efficient_non_dominated_sort(merged_F_cpu)

            # Environmental selection (2N -> N)
            survivor_indices = []
            for front in merged_fronts:
                if len(survivor_indices) + len(front) <= N:
                    survivor_indices.extend(front.tolist())
                else:
                    needed = N - len(survivor_indices)
                    cd = calc_crowding_distance(merged_F_cpu[front])
                    best_in_front = front[np.argsort(-cd)[:needed]]
                    survivor_indices.extend(best_in_front.tolist())
                    break

            pop = merged.slice(survivor_indices[:N])

        t_elapsed = time.perf_counter() - t_start
        F_final = to_numpy(pop.F)
        X_final = to_numpy(pop.X)

        # Quality metrics
        pf_true = problem.pareto_front()
        metrics = evaluate_front(F_final, pf_true=pf_true)

        return OptimizationResult(
            algorithm_name="NativeNSGA2",
            problem_name=problem.name,
            seed=seed,
            n_gen=n_gen,
            pop_size=N,
            X=X_final,
            F=F_final,
            metrics=metrics,
            runtime_seconds=t_elapsed,
            success=True,
        )
