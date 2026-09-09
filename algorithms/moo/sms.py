"""Compatibility facade: re-exports canonical SMSEMOA from algorithms.sms."""

from __future__ import annotations

from algorithms.sms.sms import *
from algorithms.sms import SMSEMOA, cv_and_dom_tournament, ALGORITHM_FLAGS

__all__ = ["SMSEMOA", "cv_and_dom_tournament", "ALGORITHM_FLAGS"]
