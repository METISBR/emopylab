# emopylab 2026
"""Multi-objective evolutionary algorithm based on an enhanced IGD.\n\nReference:\nY. Tian, X. Zhang, R. Cheng, and Y. Jin. A multi-objective evolutionary algorithm based on an enhanced inverted generational distance metric. Proceedings of the IEEE Congress on Evolutionary Computation, 2016, 5222-5229.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MOEAIGDNS': {'binary', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MOEAIGDNS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
