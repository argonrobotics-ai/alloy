# SPDX-License-Identifier: Apache-2.0
"""Surface flow (thin films on surfaces): a 2.5-D depth-averaged film on charts fixed to shapes and bodies.

Design: docs/notes/surface-flow-design.md; the film / supply interface: docs/notes/surface-flow-interface.md.
Phase 1 (WP15): charts, the film step, wetting, sources and sinks, the core laws. The supply side (nozzles, tanks,
drains, jets) is DC5's. Off by default: nothing changes unless a simulation installs the plug-in.
"""

from alloy.plugins.surface_flow.charts import PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.film import FilmMaterial
from alloy.plugins.surface_flow.plugin import SurfaceFlow

__all__ = ["FilmMaterial", "PlaneChart", "RevolutionChart", "SurfaceFlow"]
