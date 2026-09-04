"""Mixed variable operators and mating for EmoPyLab (zero-pymoo)."""

from __future__ import annotations
import numpy as np
from core.mating import Mating
from core.population import Population
from core.individual import Individual
from core.duplicate import NoDuplicateElimination, DuplicateElimination


class MixedVariableMating(Mating):
    """Mating operator for problems with mixed variable types."""

    def __init__(self, crossover=None, mutation=None, eliminate_duplicates=None, **kwargs):
        if eliminate_duplicates is None:
            eliminate_duplicates = NoDuplicateElimination()
        super().__init__(crossover=crossover, mutation=mutation, eliminate_duplicates=eliminate_duplicates, **kwargs)
        self.crossover_dict = crossover if isinstance(crossover, dict) else {}
        self.mutation_dict = mutation if isinstance(mutation, dict) else {}

    def _do(self, problem, pop, n_offsprings, parents=None, random_state=None, **kwargs):
        # If parents are provided directly as pairs
        if parents is True or isinstance(parents, (list, tuple, np.ndarray)):
            parent_pairs = pop if parents is True else parents
        else:
            parent_pairs = pop

        vars_dict = getattr(problem, "vars", {})
        if not vars_dict:
            return super()._do(problem, pop, n_offsprings, parents=parents, random_state=random_state, **kwargs)

        # Process each variable type separately
        offspring_list = []
        n_matings = len(parent_pairs)

        off_dict = {k: [] for k in vars_dict.keys()}
        
        for k, vtype_cls in vars_dict.items():
            # Find matching crossover and mutation
            cx = None
            for cls, op in self.crossover_dict.items():
                if isinstance(vtype_cls, cls) or (isinstance(vtype_cls, type) and issubclass(vtype_cls, cls)) or vtype_cls == cls:
                    cx = op
                    break
            
            mut = None
            for cls, op in self.mutation_dict.items():
                if isinstance(vtype_cls, cls) or (isinstance(vtype_cls, type) and issubclass(vtype_cls, cls)) or vtype_cls == cls:
                    mut = op
                    break

            # Extract variable values from parents
            # parent_pairs is list of pairs of individuals
            var_parents = []
            for pair in parent_pairs:
                p1_val = pair[0].get("X")
                p2_val = pair[1].get("X")
                if isinstance(p1_val, dict):
                    v1 = p1_val.get(k)
                    v2 = p2_val.get(k)
                else:
                    v1 = p1_val
                    v2 = p2_val
                var_parents.append([v1, v2])

            var_parents = np.array(var_parents)
            
            if cx is not None:
                # Do crossover for this variable
                try:
                    vals = cx._do(problem, var_parents, random_state=random_state, **kwargs)
                except Exception:
                    vals = var_parents[:, 0]
            else:
                vals = var_parents[:, 0]

            if mut is not None:
                try:
                    vals = mut._do(problem, vals, random_state=random_state, **kwargs)
                except Exception:
                    pass

            if hasattr(vals, "shape") and len(vals) == n_matings:
                off_dict[k] = vals
            else:
                off_dict[k] = [vals] * n_matings

        # Reconstruct individuals
        offspring = []
        for i in range(min(n_matings, n_offsprings)):
            x_ind = {k: off_dict[k][i] for k in vars_dict.keys()}
            offspring.append(Individual(X=x_ind))

        return Population.create(*offspring)
