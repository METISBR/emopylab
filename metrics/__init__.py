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
]
