# emopylab 2026
"""Gradient based particle swarm optimization algorithm (for multi-objective optimization).\n\nReference:\nM. M. Noel. A new gradient based particle swarm optimization algorithm for accurate computation of global minimum. Applied Soft Computing, 2012, 12: 353-359.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'GPSOM': {'constrained', 'large', 'many', 'multi', 'real'}}


class GPSOM(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
