# emopylab 2026
"""Inverse modeling based multiobjective evolutionary algorithm.\n\nReference:\nR. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff. A multiobjective evolutionary algorithm using Gaussian process-based inverse modeling. IEEE Transactions on Evolutionary Computation, 2015, 19(6): 838-856.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'IMMOEA': {'integer', 'large', 'multi', 'real'}}


class IMMOEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
