# emopylab 2026
"""Direction guided evolutionary algorithm.\n\nReference:\nC. He, R. Cheng, and D. Yazdani. Adaptive offspring generation for evolutionary large-scale multiobjective optimization. IEEE Transactions on System, Man, and Cybernetics: Systems, 2022, 52(2): 786-798.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'DGEA': {'integer', 'large', 'many', 'multi', 'real'}}


class DGEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
