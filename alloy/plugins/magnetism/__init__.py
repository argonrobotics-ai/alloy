# SPDX-License-Identifier: Apache-2.0
"""Magnetism (Gym-Anything coverage gap, 2026-11-06): ``Magnetism`` point dipoles on bodies, driven rotating sources
(magnetic stirrers), ferromagnetic plates by image dipoles and a uniform field; ``field`` holds the closed forms."""

from alloy.plugins.magnetism import field
from alloy.plugins.magnetism.plugin import Magnetism

__all__ = ["Magnetism", "field"]
