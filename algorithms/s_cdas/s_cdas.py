# emopylab 2026
"""Self-controlling dominance area of solutions.\n\nReference:\nH. Sato, H. E. Aguirre, and K. Tanaka. Self-controlling dominance area of solutions in evolutionary many-objective optimization. Proceedings of the Asia-Pacific Conference on Simulated Evolution and Learning, 2010, 455-465.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'SCDAS': {'binary', 'integer', 'label', 'many', 'permutation', 'real'}}


class SCDAS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
