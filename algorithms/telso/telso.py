# emopylab 2026
"""Two-layer encoding learning swarm optimizer.\n\nReference:\nS. Qi, R. Wang, T. Zhang, X. Yang, R. Sun, and L. Wang. A two-layer encoding learning swarm optimizer based on frequent itemsets for sparse large-scale multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2024, 11(6): 1342-1357.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'TELSO': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class TELSO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
