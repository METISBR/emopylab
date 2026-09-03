# emopylab 2026
# Reference: S. Kukkonen and K. Deb, CEC 2006.
from ._common import crowding_distance

def CrowdingDistance(PopObj, FrontNo=None):
    return crowding_distance(PopObj, FrontNo)
