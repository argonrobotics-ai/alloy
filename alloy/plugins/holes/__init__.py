# SPDX-License-Identifier: Apache-2.0
"""Holes as geometry (drilling v2): the drill's hole table, the union-of-stages geometry and the sourced drilling
work materials."""

from alloy.plugins.holes.table import Hole, Stage, holes_of
from alloy.plugins.holes.work import WORK_MATERIALS, WorkMaterial, work_material

__all__ = ["Hole", "Stage", "WORK_MATERIALS", "WorkMaterial", "holes_of", "work_material"]
