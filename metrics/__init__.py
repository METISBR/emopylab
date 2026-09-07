import numpy as np
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# -*- coding: utf-8 -*-
"""EmoPyLab Metrics Package."""

from metrics.indicators import HV, IGD, IGDPlus, GD, GDPlus, Indicator, R2
from metrics.hv_fast_mc import HV_fast_MC
from metrics.evaluator import (
    hypervolume,
    inverted_generational_distance,
    generational_distance,
    averaged_hausdorff_distance,
    spacing,
    MetricEvaluator,
    evaluate_front,
    r2_indicator,
)
from metrics.kkt_indicators import (
    compute_kkt_residuals,
    calc_h_old,
    calc_h_adap,
    analyze_quantile_sensitivity,
)

# Aliases
igd = inverted_generational_distance
gd = generational_distance

__all__ = [
    "HV",
    "IGD",
    "IGDPlus",
    "GD",
    "GDPlus",
    "Indicator",
    "R2",
    "HV_fast_MC",
    "hypervolume",
    "inverted_generational_distance",
    "generational_distance",
    "averaged_hausdorff_distance",
    "spacing",
    "MetricEvaluator",
    "evaluate_front",
    "igd",
    "gd",
    "r2_indicator",
    "compute_kkt_residuals",
    "calc_h_old",
    "calc_h_adap",
    "analyze_quantile_sensitivity",
]
