# emopylab 2026
# Reference: X. Yao, Y. Liu, and G. Lin, IEEE TEC, 1999.
from ._common import operator_fep

def OperatorFEP(Problem, Population, **kwargs):
    return operator_fep(Problem, Population, **kwargs)
