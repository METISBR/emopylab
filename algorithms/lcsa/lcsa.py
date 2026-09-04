# emopylab 2026
"""Linear combination-based search algorithm.\n\nReference:\nH. Zille. Large-scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art. PhD Thesis, Otto von Guericke University Magdeburg, 2019. --------------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'LCSA': {'integer', 'large', 'many', 'multi', 'real'}}


class LCSA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
