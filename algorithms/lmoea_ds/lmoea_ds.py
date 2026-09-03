# emopylab 2026
"""Large-scale evolutionary multi-objective optimization assisted by directed sampling.\n\nReference:\nS. Qin, C. Sun, Y. Jin, Y. Tan, and J. Fieldsend. Large-scale evolutionary multi-objective optimization assisted by directed sampling. IEEE Transactions on Evolutionary Computation, 2021, 25(4): 724-738.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'LMOEADS': {'integer', 'large', 'multi', 'real'}}


class LMOEADS(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
