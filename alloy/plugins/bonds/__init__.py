# SPDX-License-Identifier: Apache-2.0
"""Cohesive bonds between a film and rigid surfaces (``Bonds``): formation under pressure, peel at a resolved front.
The model, its limits and the sources are in ``plugin.py`` and ``adhesives.py``; laws in ``laws.py``."""

from .adhesives import ROWS_7818, TAPE_7818, Adhesive, adhesive
from .plugin import Bonds

__all__ = ["ROWS_7818", "TAPE_7818", "Adhesive", "Bonds", "adhesive"]
