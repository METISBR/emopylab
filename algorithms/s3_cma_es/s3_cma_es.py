# emopylab 2026
"""Scalable small subpopulations based covariance matrix adaptation.\n\nReference:\nH. Chen, R. Cheng, J. Wen, H. Li, and J. Weng. Solving large-scale many-objective optimization problems by covariance matrix adaptation evolution strategy with scalable small subpopulations. Information Sciences, 2020, 509: 457-469.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'S3CMAES': {'integer', 'large', 'many', 'multi', 'real'}}


class S3CMAES(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
