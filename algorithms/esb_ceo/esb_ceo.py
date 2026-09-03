# emopylab 2026
"""Bayesian co-evolutionary optimization based entropy search.\n\nReference:\nH. Bian, J. Tian, J. Yu, and H. Yu. Bayesian co-evolutionary optimization based entropy search for high-dimensional many-objective optimization. Knowledge-Based Systems, 2023, 274: 110630.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'ESBCEO': {'expensive', 'multi', 'real'}}


class ESBCEO(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
