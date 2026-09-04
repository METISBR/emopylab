# emopylab 2026
from ._common import operator_ga

def OperatorGAhalf(Problem, Parent, Parameter=None, **kwargs):
    return operator_ga(Problem, Parent, Parameter=Parameter, half=True, **kwargs)
