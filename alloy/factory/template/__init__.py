# SPDX-License-Identifier: Apache-2.0
"""Factory template plug-in: the smallest package that passes the Alloy solver plug-in contract (SPEC §3.8).

Importing the package registers its ``SolverCapability`` (``plugin.py``), its B0 bully scene (``scenes.py``) and,
when the bench library is present, its laws (``laws.py``). ``manifest.json`` names them for ``evaluate_plugin``.
"""

from . import laws, plugin, scenes
from .solver import PLUGIN_NAME, SolverTemplate, TemplateConfig

__all__ = ["PLUGIN_NAME", "SolverTemplate", "TemplateConfig", "laws", "plugin", "scenes"]
