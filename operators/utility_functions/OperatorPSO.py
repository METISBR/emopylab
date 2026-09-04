# emopylab 2026
# Reference: C. A. Coello Coello and M. S. Lechuga, CEC 2002.
from ._common import operator_pso

def OperatorPSO(Problem, Particle, Pbest, Gbest, W=0.4, **kwargs):
    return operator_pso(Problem, Particle, Pbest, Gbest, W=W, **kwargs)
