# emopylab 2026
"""NSGA-II-DTI (nSGA-II of Deb's type I robust version).

Reference:
K. Deb and H. Gupta. Introducing robustness in multi-objective optimization. Evolutionary
Computation, 2006, 14(4): 463-494.
"""

from __future__ import annotations

import numpy as np

from algorithms.community_utils.base import LoopAlgorithm, crowding, decs, ga, nd_sort, tournament
from core.population import Population

ALGORITHM_FLAGS = {'NSGAIIDTI': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real', 'robust'}}


def _mean_effective(problem, pop, rng):
    """Mean objective/constraint values over the problem's disturbance samples (not charged to FE)."""
    if hasattr(problem, "perturb"):
        F, G = problem.perturb(decs(pop), rng=rng)
        return F.mean(axis=0), (G.mean(axis=0) if G is not None and G.size > 0 else None)
    else:
        F, G = problem.evaluate(decs(pop), return_values_of=["F", "G"])
        return F, (G if G is not None and G.size > 0 else None)


def _environmental_selection(pop, N, obj_v, con_v):
    front_no, max_f = nd_sort(obj_v, con_v, N)
    nxt = front_no < max_f
    cd = crowding(obj_v, front_no)
    last = np.where(front_no == max_f)[0]
    rank = np.argsort(-cd[last], kind="stable")
    nxt[last[rank[: N - int(nxt.sum())]]] = True
    return pop[nxt], front_no[nxt], cd[nxt], obj_v[nxt], (None if con_v is None else con_v[nxt])


class NSGAIIDTI(LoopAlgorithm):
    def start(self):
        obj_v, con_v = _mean_effective(self.problem, self.pop, self.rng)
        self.pop, self.front_no, self.crowd, self.obj_v, self.con_v = _environmental_selection(self.pop, self.N, obj_v, con_v)

    def step(self):
        pool = tournament(2, self.N, self.front_no, -self.crowd, rng=self.rng)
        off = self.evaluate(ga(self.problem, decs(self.pop[pool]), (1, 10, 1, 50), rng=self.rng))
        off_obj, off_con = _mean_effective(self.problem, off, self.rng)
        merged = Population.merge(self.pop, off)
        con_v = None if self.con_v is None else np.vstack([self.con_v, off_con])
        self.pop, self.front_no, self.crowd, self.obj_v, self.con_v = _environmental_selection(
            merged, self.N, np.vstack([self.obj_v, off_obj]), con_v)
