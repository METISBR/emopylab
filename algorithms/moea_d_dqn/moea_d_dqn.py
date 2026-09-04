# emopylab 2026
"""MOEA/D based on deep Q-network (Enhanced with proper Target Network).\n\nReference:\nY. Tian, X. Li, H. Ma, X. Zhang, K. C. Tan, and Y. Jin, Deep reinforcement learning based adaptive operator selection for evolutionary multi-objective optimization, IEEE Transactions on Emerging Topics in Computational Intelligence, 2023, 7(4): 1051-1064.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'MOEADDQN': {'integer', 'many', 'multi', 'real'}}


class MOEADDQN(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
