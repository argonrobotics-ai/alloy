# SPDX-License-Identifier: Apache-2.0
"""A scene whose only dynamics are plug-ins: a surface film on a static plate, a plug-in solver beside static shapes.

The engine steps its dynamic entries (rigid, cloth, liquid, ...); with none it refuses to build ("scene has no
dynamic entry to step"), so a film on a static plate needed a dummy free body (WP13's combination sweep, F6). Here such
a scene steps with ``StaticStepper``: each exchange copies the state (static shapes have none; kinematic bodies hold
unless driven), and the plug-ins that wrap the stepper (``alloy.plugins.host.PluginHost``) do the work. The exchange
is the rigid step a dummy body would get, so the plug-ins see the same clock as with the dummy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NAME = "static"


class StaticStepper:
    """``StepperLike`` with nothing to integrate (module docstring). Example: ``StaticStepper(built)``."""

    def __init__(self, built: Any) -> None:
        self.worlds = int(built.world_count)
        self.dt_x = built.rigid_dt          # the scene's rigid step (alloy.scene.rigid_step), today's when unchosen

    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        """Carry the state to the next buffer unchanged. Example: ``stepper.step(s0, s1, None, None, 1e-3)``."""
        if state_out is not state_in:
            state_out.assign(state_in)

    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        """The rigid step on the control period's grid. Example: ``stepper.choose_clock(0.01, None)``."""
        from alloy.core.timebase import snap_to_grid  # noqa: PLC0415

        return snap_to_grid(self.dt_x, control_dt), {NAME: 1}

    def report(self) -> Any:
        """A quiet coupling report. Example: ``stepper.report().dt_x``."""
        from alloy.coupling.types import CouplingReport  # noqa: PLC0415

        rep = CouplingReport.quiet(self.dt_x, {NAME: 1}, self.worlds)
        rep.notes.update({"stepper": "static (plug-ins only)"})
        return rep

    def state_arrays(self) -> Mapping[str, Any]:
        """No state of its own. Example: ``stepper.state_arrays() == {}``."""
        return {}

    def state_world_maps(self) -> Mapping[str, Any]:
        """No state of its own. Example: ``stepper.state_world_maps() == {}``."""
        return {}

    def rollback_requested(self, report: Any) -> bool:
        """Never: there is no coupling to escalate. Example: ``stepper.rollback_requested(rep)``."""
        return False

    def on_rollback(self, world_mask: Any = None) -> None:
        """Nothing to re-plan. Example: ``stepper.on_rollback()``."""
        return None

    def solvers(self) -> list[Any]:
        """No solvers. Example: ``stepper.solvers() == []``."""
        return []


def stepper_for(built: Any) -> StaticStepper | None:
    """A ``StaticStepper`` when ``built`` has no dynamic entry, else None (the engine builds its own).

    Example: ``wiring.wire(built, p, stepper=stepper_for(built))``."""
    from alloy.integration.stepper import dynamic_entries  # noqa: PLC0415

    return None if dynamic_entries(built) else StaticStepper(built)
