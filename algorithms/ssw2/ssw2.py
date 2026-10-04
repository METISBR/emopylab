# Legacy backward compatibility bridge: points to the official gsa_ssw_mud module
from algorithms.gsa_ssw_mud import GSASSWMUD, GSA_SSW_MUD, SSW2
from algorithms.gsa_ssw_mud.gsa_ssw_mud import *

__all__ = ["GSASSWMUD", "GSA_SSW_MUD", "SSW2"]
