"""EmoPyLab Native MOEA/D Solver in Pure Tensors."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.engine.runner import OptimizationResult
from core.operators.crossover.sbx import sbx_crossover_tensor
from core.operators.mutation.polynomial import polynomial_mutation_tensor
from operators.sampling.lhs import LatinHypercubeSampling
from core.tensor.backend import get_array_module, index_tensor, init_tensor_backend, to_device, to_numpy
from core.tensor.population import TensorPopulation
from core.tensor.problem import TensorProblem
from core.tensor.ref_dirs import get_reference_directions
from metrics.evaluator import evaluate_front


class NativeMOEAD:
    """Pure Tensor-Native Decomposition Based MOEA/D Solver."""

    def __init__(
        self,
        ref_dirs: Optional[np.ndarray] = None,
        n_neighbors: int = 15,
        prob_neighbor_mating: float = 0.9,
        crossover_eta: float = 20.0,
        crossover_prob: float = 1.0,
        mutation_eta: float = 20.0,
        mutation_prob: float | None = None,
    ) -> None:
        self.ref_dirs = ref_dirs
        self.n_neighbors = int(n_neighbors)
        self.prob_neighbor_mating = float(prob_neighbor_mating)
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

        # 1. Reference Directions & Neighborhood Matrix
        if self.ref_dirs is None:
            self.ref_dirs = get_reference_directions("das-dennis", n_obj=M, n_partitions=12)

        N = len(self.ref_dirs)
        W = self.ref_dirs.astype(np.float64)

        # Distance between weight vectors to find neighbors
        dist_W = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
        neighborhoods = np.argsort(dist_W, axis=1)[:, : min(self.n_neighbors, N)]

        # 2. Initialize population with Latin Hypercube sampling
        X_init = to_device(LatinHypercubeSampling().sample_array(
            problem,
            N,
            random_state=np.random.default_rng(seed),
        ))
        F_init, G_init = problem.evaluate(X_init)
        pop = TensorPopulation(N, D, M, problem.n_constr, X=X_init, F=F_init, G=G_init)

        z_ideal = np.min(to_numpy(pop.F), axis=0)

        # 3. Main Generational Loop (Vectorized Decomposition Replacement)
        is_torch = "torch" in type(pop.X).__module__
        is_mlx = "mlx" in type(pop.X).__module__
        rng = np.random.default_rng(seed)

        for gen in range(1, n_gen + 1):
            # Batch parent selection for all N subproblems
            p1_indices = np.empty(N, dtype=np.int64)
            p2_indices = np.empty(N, dtype=np.int64)
            rand_choices = rng.random(N)

            for i in range(N):
                pool = neighborhoods[i] if rand_choices[i] < self.prob_neighbor_mating else np.arange(N)
                chosen = rng.choice(pool, size=2, replace=False)
                p1_indices[i] = chosen[0]
                p2_indices[i] = chosen[1]

            p1_batch = index_tensor(pop.X, p1_indices)
            p2_batch = index_tensor(pop.X, p2_indices)

            # Vectorized Batch Crossover and Mutation on GPU
            offspring_X, _ = sbx_crossover_tensor(
                p1_batch, p2_batch, problem.xl_dev, problem.xu_dev,
                eta=self.crossover_eta, prob=self.crossover_prob, seed=seed + gen * N,
            )
            offspring_X = polynomial_mutation_tensor(
                offspring_X, problem.xl_dev, problem.xu_dev,
                eta=self.mutation_eta, prob_var=self.mutation_prob, seed=seed + gen * N + 1,
            )

            # Batch Evaluation on GPU
            offspring_F, offspring_G = problem.evaluate(offspring_X)

            # Update Ideal Point
            f_off_cpu = to_numpy(offspring_F)
            z_ideal = np.minimum(z_ideal, np.min(f_off_cpu, axis=0))

            # Vectorized Tchebycheff Replacement
            F_cpu = to_numpy(pop.F)
            X_cpu = to_numpy(pop.X)
            x_off_cpu = to_numpy(offspring_X)

            for i in range(N):
                neigh = neighborhoods[i]
                w_neigh = W[neigh]
                f_neigh = F_cpu[neigh]
                f_cand = f_off_cpu[i]

                gte_current = np.max(w_neigh * np.abs(f_neigh - z_ideal), axis=1)
                gte_off = np.max(w_neigh * np.abs(f_cand - z_ideal), axis=1)

                replace_mask = gte_off < gte_current
                for idx_rep, should_rep in zip(neigh, replace_mask):
                    if should_rep:
                        X_cpu[idx_rep] = x_off_cpu[i]
                        F_cpu[idx_rep] = f_cand

            # Re-upload updated population to device tensor once per generation
            pop.X = to_device(X_cpu)
            pop.F = to_device(F_cpu)
        t_elapsed = time.perf_counter() - t_start
        F_final = to_numpy(pop.F)
        X_final = to_numpy(pop.X)

        pf_true = problem.pareto_front()
        metrics = evaluate_front(F_final, pf_true=pf_true)

        return OptimizationResult(
            algorithm_name="NativeMOEAD",
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
