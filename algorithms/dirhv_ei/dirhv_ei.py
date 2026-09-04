# emopylab 2026
"""Expected direction-based hypervolume improvement.\n\nReference:\nL. Zhao and Q. Zhang. Hypervolume-guided decomposition for parallel expensive multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2024, 28(2): 432-444.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'DirHVEI': {'expensive', 'integer', 'many', 'multi', 'real'}}


class DirHVEI(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
