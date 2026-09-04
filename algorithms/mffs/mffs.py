# emopylab 2026
"""Multiform feature selection.\n\nReference:\nR. Jiao, B. Xue, and M. Zhang. Benefiting from single-objective feature selection to multiobjective feature selection: A multiform approach. IEEE Transactions on Cybernetics, 2023, 53(12): 7773-7786.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MFFS': {'binary', 'multi'}}


class MFFS(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
