"""EmoPyLab Native NSGA-III Solver in Pure Tensors."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.engine.runner import OptimizationResult
from core.nds.ens import efficient_non_dominated_sort
from core.operators.crossover.sbx import sbx_crossover_tensor
from core.operators.mutation.polynomial import polynomial_mutation_tensor
from core.operators.sampling.lhs import latin_hypercube_sampling
from core.tensor.backend import get_array_module, index_tensor, init_tensor_backend, to_device, to_numpy, vstack
from core.tensor.population import TensorPopulation
from core.tensor.problem import TensorProblem
from core.tensor.ref_dirs import get_reference_directions
from metrics.evaluator import evaluate_front


class NativeNSGA3:
    """Pure Tensor-Native Reference-Point Based NSGA-III Solver."""

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        pop_size: int | None = None,
        n_partitions: int = 12,
        crossover_eta: float = 15.0,
        crossover_prob: float = 0.9,
        mutation_eta: float = 20.0,
        mutation_prob: float | None = None,
    ) -> None:
        self.ref_dirs = ref_dirs
        self.n_partitions = n_partitions
        self.pop_size = pop_size
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

        M = problem.n_obj
        D = problem.n_var

        # 1. Setup Reference Directions
        if self.ref_dirs is None:
            self.ref_dirs = get_reference_directions("das-dennis", n_obj=M, n_partitions=self.n_partitions)

        N = self.pop_size if self.pop_size is not None else len(self.ref_dirs)

        # 2. Initialize Population via LHS
        X_init = latin_hypercube_sampling(N, D, problem.xl_cpu, problem.xu_cpu, seed=seed)
        F_init, G_init = problem.evaluate(X_init)
        pop = TensorPopulation(N, D, M, problem.n_constr, X=X_init, F=F_init, G=G_init)

        # Reference vectors normalized
        V = self.ref_dirs / np.maximum(np.linalg.norm(self.ref_dirs, axis=1, keepdims=True), 1e-12)

        # 3. Main Generational Loop
        for gen in range(1, n_gen + 1):
            # Mating Selection
            rng = np.random.default_rng(seed + gen)
            p1_idx = rng.integers(0, N, size=N // 2)
            p2_idx = rng.integers(0, N, size=N // 2)

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

            # Merge 2N Pool
            merged = TensorPopulation.merge(pop, offspring_pop)
            merged_F_cpu = to_numpy(merged.F)
            merged_fronts = efficient_non_dominated_sort(merged_F_cpu)

            # Environmental Niching Selection
            survivor_indices = []
            for front in merged_fronts:
                if len(survivor_indices) + len(front) <= N:
                    survivor_indices.extend(front.tolist())
                else:
                    needed = N - len(survivor_indices)
                    # Associate remaining front to reference lines
                    F_front = merged_F_cpu[front]
                    z_min = np.min(merged_F_cpu, axis=0)
                    F_trans = F_front - z_min
                    dists = np.linalg.norm(F_trans, axis=1)
                    norm_F = np.where(dists[:, None] > 1e-12, F_trans / dists[:, None], 0.0)

                    cos_sim = np.dot(norm_F, V.T)
                    best_proj = np.max(cos_sim, axis=1)

                    best_in_front = front[np.argsort(-best_proj)[:needed]]
                    survivor_indices.extend(best_in_front.tolist())
                    break

            pop = merged.slice(survivor_indices[:N])

        t_elapsed = time.perf_counter() - t_start
        F_final = to_numpy(pop.F)
        X_final = to_numpy(pop.X)

        pf_true = problem.pareto_front()
        metrics = evaluate_front(F_final, pf_true=pf_true)

        return OptimizationResult(
            algorithm_name="NativeNSGA3",
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
