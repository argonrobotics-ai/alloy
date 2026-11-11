# SPDX-License-Identifier: Apache-2.0
"""Registers the template plug-in's ``SolverCapability`` on import (§3.8 item 1).

Unshipped plug-ins serve only the route ``contrib:<name>``, which the router never produces, so importing an
agent-written package can never change what a scene routes to. Shipping a plug-in onto a real route is a
coordinator decision after ``evaluate_plugin`` says ``ship``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alloy._vendor.newton.solvers import SolverBase

from alloy.core.newton_compat import ModelView
from alloy.core.registry import SolverCapability, register_solver

from .solver import PLUGIN_NAME, SolverTemplate, TemplateConfig


def make_solver(view: ModelView, kwargs: Mapping[str, Any]) -> SolverBase:
    """Capability factory: build the solver on a coupled entry's model view (or a plain Model)."""
    return SolverTemplate(view, TemplateConfig(**dict(kwargs)))  # type: ignore[arg-type]


CAPABILITY = SolverCapability(
    name=PLUGIN_NAME,
    factory=make_solver,
    routes=SolverTemplate.routes,
    owns=frozenset({"particle"}),
    coupling=True,
    uniform_worlds=True,
    per_contact_friction=False,
    determinism=SolverTemplate.determinism,
    friction_mix="none",
    register_attributes=SolverTemplate.register_custom_attributes,
)

register_solver(CAPABILITY, replace=True)
