# emopylab 2026
"""CNSDE-DVC (Constrained NSDE with decision variable classification).

Reference:
W. Du et al. High-dimensional robust multi-objective optimization for order scheduling: A decision variable classification approach. IEEE Transactions on Industrial Informatics, 2019, 15(1): 293-304.
"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'CNSDEDVC': {'integer', 'multi', 'real', 'robust'}}


class CNSDEDVC(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, SN=4, PN=6, TN=15, theta=0.001, eta=0.001, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        self.SN = SN
        self.PN = PN
        self.TN = TN
        self.theta = theta
        self.eta = eta
