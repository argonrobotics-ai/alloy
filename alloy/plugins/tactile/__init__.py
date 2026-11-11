# SPDX-License-Identifier: Apache-2.0
"""Tactile sensors (engine item 10): ``Tactile`` taxel arrays from contact forces, with a noise layer and
real-sensor presets (v1); ``GelSensor`` gel depth and marker fields for vision-based sensors (v2 physics).
See ``sensor.py`` and ``gel.py`` for frames and channels, and ``presets.py`` for the sources."""

from alloy.plugins.tactile.gel import GEL_PRESETS, GelSensor
from alloy.plugins.tactile.presets import PRESETS, Preset, preset
from alloy.plugins.tactile.sensor import Tactile

__all__ = ["GEL_PRESETS", "PRESETS", "GelSensor", "Preset", "Tactile", "preset"]
