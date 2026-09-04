# emopylab 2026
"""Weight vector based multi-objective optimization algorithm with preference.\n\nReference:\nX. Zhang, X. Jiang, and L. Zhang. A weight vector based multi-objective optimization algorithm with preference. Acta Electronica Sinica (Chinese), 2016, 44(11): 2639-2645.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'WVMOEAP': {'integer', 'multi', 'real'}}


class WVMOEAP(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
