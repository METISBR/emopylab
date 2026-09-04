# emopylab 2026
"""Many-objective evolutionary algorithm using a one-by-one selection.\n\nReference:\nY. Liu, D. Gong, J. Sun, and Y. Jin. A many-objective evolutionary algorithm using a one-by-one selection strategy. IEEE Transactions on Cybernetics, 2017, 47(9): 2689-2702.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'onebyoneEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class onebyoneEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
