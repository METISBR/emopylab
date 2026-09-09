"""Official canonical SMS-EMOA package for EmoPyLab."""

from __future__ import annotations

from .sms import SMSEMOA, cv_and_dom_tournament

SMS = SMSEMOA
SMS_EMOA = SMSEMOA

ALGORITHMS = {
    "SMSEMOA": SMSEMOA,
    "SMS-EMOA": SMSEMOA,
}

ALGORITHM_FLAGS = {
    "SMSEMOA": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
    "SMS-EMOA": {"multi", "many", "real", "integer", "binary", "permutation", "label", "constrained"},
}

__all__ = [
    "SMSEMOA",
    "SMS",
    "SMS_EMOA",
    "cv_and_dom_tournament",
    "ALGORITHMS",
    "ALGORITHM_FLAGS",
]
