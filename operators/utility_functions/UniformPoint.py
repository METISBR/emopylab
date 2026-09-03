# emopylab 2026
# References:
# [1] Y. Tian et al., CEC 2018.
# [2] T. Takagi et al., GECCO Companion 2020.
from ._common import uniform_point

def UniformPoint(N, M, method='NBI', **kwargs):
    return uniform_point(N, M, method=method, **kwargs)
