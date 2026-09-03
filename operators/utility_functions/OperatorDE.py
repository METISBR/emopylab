# emopylab 2026
# Reference: H. Li and Q. Zhang, IEEE TEC, 2009.
from ._common import operator_de

def OperatorDE(Problem, Parent1, Parent2, Parent3, Parameter=None, **kwargs):
    return operator_de(Problem, Parent1, Parent2, Parent3, Parameter=Parameter, **kwargs)
