# SPDX-License-Identifier: Apache-2.0
"""Entry point for Isaac Lab's ``isaaclab.tasks`` group: Isaac Lab's ``isaaclab`` command imports this before it
resolves a task, which makes ``physics=alloy`` resolve (``alloy.isaaclab.presets``). Registers no task of its own.
Only Isaac Lab loads this group; without Isaac Lab the import does nothing (``alloy.isaaclab`` names then raise
``FeatureUnavailableError`` with the install line when used)."""

import alloy.isaaclab  # noqa: F401  (installs the preset name on import when Isaac Lab is there)
