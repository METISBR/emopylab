"""EmoPyLab Native MOEA/D Solver in Pure Tensors."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.engine.runner import OptimizationResult
from core.operators.crossover.sbx import sbx_crossover_tensor
from core.operators.mutation.polynomial import polynomial_mutation_tensor
from core.operators.sampling.lhs import latin_hypercube_sampling
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

        # 2. Initialize Population via LHS
        X_init = latin_hypercube_sampling(N, D, problem.xl_cpu, problem.xu_cpu, seed=seed)
        F_init, G_init = problem.evaluate(X_init)
        pop = TensorPopulation(N, D, M, problem.n_constr, X=X_init, F=F_init, G=G_init)

        z_ideal = np.min(to_numpy(pop.F), axis=0)

        # 3. Main Generational Loop (Decomposition Replacement)
        rng = np.random.default_rng(seed)

        for gen in range(1, n_gen + 1):
            perm = rng.permutation(N)
            for i in perm:
                # Select mating pool: neighborhood or whole population
                if rng.random() < self.prob_neighbor_mating:
                    pool = neighborhoods[i]
                else:
                    pool = np.arange(N)

                parents_idx = rng.choice(pool, size=2, replace=False)
                p1 = index_tensor(pop.X, parents_idx[0:1])
                p2 = index_tensor(pop.X, parents_idx[1:2])

                # Generate offspring
                off1, _ = sbx_crossover_tensor(
                    p1, p2, problem.xl_dev, problem.xu_dev,
                    eta=self.crossover_eta, prob=self.crossover_prob, seed=seed + gen * N + i,
                )
                off1 = polynomial_mutation_tensor(
                    off1, problem.xl_dev, problem.xu_dev,
                    eta=self.mutation_eta, prob_var=self.mutation_prob, seed=seed + gen * N + i + 1,
                )

                f_off, g_off = problem.evaluate(off1)
                f_off_cpu = to_numpy(f_off)[0]

                # Update Ideal Point
                z_ideal = np.minimum(z_ideal, f_off_cpu)

                # Tchebycheff Scalarization Replacement
                neigh = neighborhoods[i]
                W_neigh = W[neigh]
                F_neigh = to_numpy(pop.F)[neigh]

                # Current Tchebycheff value
                gte_current = np.max(W_neigh * np.abs(F_neigh - z_ideal), axis=1)
                # Offspring Tchebycheff value
                gte_off = np.max(W_neigh * np.abs(f_off_cpu - z_ideal), axis=1)

                # Replace better solutions
                replace_mask = gte_off < gte_current
                for idx_replace, should_replace in zip(neigh, replace_mask):
                    if should_replace:
                        try:
                            pop.X[idx_replace] = off1[0]
                            pop.F[idx_replace] = f_off[0]
                        except Exception:
                            if hasattr(pop.X, "at") and hasattr(pop.X.at[idx_replace], "set"):
                                pop.X = pop.X.at[idx_replace].set(off1[0])
                                pop.F = pop.F.at[idx_replace].set(f_off[0])
                            else:
                                X_np = to_numpy(pop.X).copy()
                                F_np = to_numpy(pop.F).copy()
                                X_np[idx_replace] = to_numpy(off1[0])
                                F_np[idx_replace] = to_numpy(f_off[0])
                                pop.X = to_device(X_np)
                                pop.F = to_device(F_np)

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
