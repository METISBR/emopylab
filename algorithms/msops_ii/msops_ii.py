# emopylab 2026
"""Multiple single objective Pareto sampling II.\n\nReference:\nE. J. Hughes. MSOPS-II: A general-purpose many-objective optimiser. Proceedings of the IEEE Congress on Evolutionary Computation, 2007, 3944-3951.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MSOPSII': {'constrained', 'integer', 'many', 'multi', 'real'}}


class MSOPSII(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
