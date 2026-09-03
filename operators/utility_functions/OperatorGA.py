# emopylab 2026
# References include Deb et al. (SBX/PM), Davis (1985), Fogel (1988).
from ._common import operator_ga

def OperatorGA(Problem, Parent, Parameter=None, **kwargs):
    return operator_ga(Problem, Parent, Parameter=Parameter, half=False, **kwargs)
