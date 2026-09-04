# emopylab 2026
"""Decision space based niching NSGA-II.\n\nReference:\nJ. Liang, C. Yue, and B. Qu. Multimodal multi-objective optimization: A preliminary study. Proceedings of the IEEE Congress on Evolutionary Computation, 2016, 2454-2461.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'DNNSGAII': {'integer', 'multi', 'multimodal', 'real'}}


class DNNSGAII(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
