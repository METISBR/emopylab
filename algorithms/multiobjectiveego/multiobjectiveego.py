# emopylab 2026
"""Multi-objective efficient global optimization.\n\nReference:\nR. Hussein and K. Deb. A generative Kriging surrogate model for constrained and unconstrained multi-objective optimization. Proceedings of the Genetic and Evolutionary Computation Conference, 2016, 573-580. -------------------------------------------------------------------------- This function is written by Youwei He (email: 1554748356@qq.com) Parameter setting parameter for AASF in equation (10) Generate the initial design points number of design variables number of initial design points 11*D-1 step1-1: generate initial design points using Latin Hypercube sampling step1-2: evaluate initial design points step 2: Generate the reference direction set Optimization step 2 diversity_preserver procedure: neighborhood approach determine whether the problem is constrained or not step 3: Points_Selector procedure orthogonal distance of each point to the given reference direction step 4-1 step 4-2 step 4-3 step 5: Optimization infill point too close (not used in the paper, but this happens sometimes especially for low dimensional problem)\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MultiObjectiveEGO': {'constrained', 'expensive', 'integer', 'multi', 'real'}}


class MultiObjectiveEGO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
