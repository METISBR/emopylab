# emopylab 2026
"""Multimodal multi-objective evolutionary algorithm assisted by Pareto set learning.\n\nReference:\nF. Ming, W. Gong, and Y. Jin. Growing neural gas network-based surrogate-assisted Pareto set learning for multimodal multi-objective optimization. Swarm and Evolutionary Computation, 2024, 87: 101541.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'MMEAPSL': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


class MMEAPSL(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
