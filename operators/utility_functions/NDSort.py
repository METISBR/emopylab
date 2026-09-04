# emopylab 2026
# References:
# [1] X. Zhang, Y. Tian, R. Cheng, and Y. Jin, IEEE TEC, 2015.
# [2] X. Zhang, Y. Tian, R. Cheng, and Y. Jin, IEEE TEC, 2018.
from ._common import nd_sort

def NDSort(*args):
    if len(args) == 2:
        return nd_sort(args[0], None, args[1])
    if len(args) == 3:
        return nd_sort(args[0], args[1], args[2])
    raise TypeError('NDSort expects (PopObj, nSort) or (PopObj, PopCon, nSort)')
