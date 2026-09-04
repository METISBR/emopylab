# emopylab 2026
"""SIBEA with minimum objective subset of size k with minimum error.\n\nReference:\nD. Brockhoff and E. Zitzler. Improving hypervolume-based multiobjective evolutionary algorithms by using objective reduction methods. Proceedings of the IEEE Congress on Evolutionary Computation, 2007, 2086-2093.\n"""

from __future__ import annotations

from algorithms.e_moea.e_moea import eMOEA


ALGORITHM_FLAGS = {'SIBEAkEMOSS': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class SIBEAkEMOSS(eMOEA):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
