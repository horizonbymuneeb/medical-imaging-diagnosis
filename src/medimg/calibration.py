"""Probability calibration: reliability curves, ECE and temperature scaling.

A CNN trained with a balanced patch sampler outputs probabilities that are
confident relative to the true prevalence, so the numbers are not probabilities
yet.  This module measures that (reliability curve, ECE, Brier) and fixes it
with a single temperature parameter fitted on held-out logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special
