"""
Core optimization framework components for EmoPyLab (2026).
Completely standalone native implementations of evolutionary algorithm building blocks.
"""
from __future__ import annotations

from core.individual import Individual, calc_cv, constr_to_cv
from core.population import Population, pop_from_array_or_individual, merge
from core.problem import Problem, ElementwiseProblem
from core.algorithm import Algorithm, LoopwiseAlgorithm, MetaAlgorithm
from core.operator import Operator, default_random_state
from core.variable import Variable, Real, Integer, Binary, Choice, BoundedVariable, get
from core.callback import Callback, CallbackCollection
from core.evaluator import Evaluator, VoidEvaluator
from core.duplicate import (
    DuplicateElimination,
    DefaultDuplicateElimination,
    ElementwiseDuplicateElimination,
    HashDuplicateElimination,
    NoDuplicateElimination,
)
from core.infill import InfillCriterion
from core.mating import Mating
from core.repair import Repair, NoRepair
from core.survival import Survival, ToReplacement, split_by_feasibility
from core.crossover import Crossover
from core.mutation import Mutation
from core.selection import Selection
from core.sampling import Sampling
from core.result import Result, Meta
from core.parameters import (
    get_params,
    flatten,
    hierarchical,
    set_params,
    apply_to_params,
    deactivate_params,
)

__all__ = [
    "Individual",
    "calc_cv",
    "constr_to_cv",
    "Population",
    "pop_from_array_or_individual",
    "merge",
    "Problem",
    "ElementwiseProblem",
    "Algorithm",
    "LoopwiseAlgorithm",
    "MetaAlgorithm",
    "Operator",
    "default_random_state",
    "Variable",
    "Real",
    "Integer",
    "Binary",
    "Choice",
    "BoundedVariable",
    "get",
    "Callback",
    "CallbackCollection",
    "Evaluator",
    "VoidEvaluator",
    "DuplicateElimination",
    "DefaultDuplicateElimination",
    "ElementwiseDuplicateElimination",
    "HashDuplicateElimination",
    "NoDuplicateElimination",
    "InfillCriterion",
    "Mating",
    "Repair",
    "NoRepair",
    "Survival",
    "ToReplacement",
    "split_by_feasibility",
    "Crossover",
    "Mutation",
    "Selection",
    "Sampling",
    "Result",
    "Meta",
    "get_params",
    "flatten",
    "hierarchical",
    "set_params",
    "apply_to_params",
    "deactivate_params",
]
