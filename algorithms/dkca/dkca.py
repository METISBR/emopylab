# emopylab 2026
"""Dynamic knowledge-guided coevolutionary algorithm.\n\nReference:\nY. Li, X. Feng, and H. Yu. A dynamic knowledge-guided coevolutionary algorithm for large-scale sparse multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(11): 7054-7064.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DKCA': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class DKCA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
