# SPDX-License-Identifier: Apache-2.0
"""Fit error classes, defined in alloy.core.errors (SCR-WP11-1, accepted b15d8ab) and re-exported here."""

from __future__ import annotations

from alloy.core.errors import FitError, HeldOutGapError, SplitError

__all__ = ["FitError", "SplitError", "HeldOutGapError"]
