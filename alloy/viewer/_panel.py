# SPDX-License-Identifier: Apache-2.0
"""Alloy's own controls in a viewer's panel (the browser viewer and the desktop window): which world is shown, and the
contact overlay's points and forces. The controls only record what was asked; the session applies it before it draws
the next frame, on the simulation thread.

Example::

    panel = Panel(worlds=sim.worlds, points=True, forces=True)
    panel.attach(backend)              # False for a backend without a panel (rerun, usd, null)
    ask = panel.take()                 # None, or the newest request
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

log = logging.getLogger("alloy.viewer")

ALL_WORLDS = -1
WORLD_LABEL = "World shown (-1: all)"
POINTS_LABEL = "Alloy contact points"
FORCES_LABEL = "Alloy contact forces"


@dataclass(frozen=True)
class PanelState:
    """What the panel shows: one world (``ALL_WORLDS`` for every world) and the overlay's two parts."""

    world: int = ALL_WORLDS
    points: bool = True
    forces: bool = True


class Panel:
    """The panel's controls and the newest request made with them."""

    def __init__(self, worlds: int, points: bool, forces: bool) -> None:
        self.worlds = int(worlds)
        self.state = PanelState(ALL_WORLDS, bool(points), bool(forces))
        self._asked: PanelState | None = None

    def attach(self, backend: Any) -> bool:
        """Add the controls to ``backend``'s panel; False when it has none."""
        register = getattr(backend, "register_ui_callback", None)
        if not callable(register):
            return False
        try:
            register(self.render)
        except Exception as exc:  # a backend whose panel cannot take controls keeps working without them
            log.debug("viewer panel controls not added: %s", exc)
            return False
        return True

    def render(self, ui: Any) -> None:
        """Draw the controls with the viewer's widget set and record a change."""
        new, touched = self.state, False
        if self.worlds > 1:
            changed, world = ui.slider_int(WORLD_LABEL, new.world, ALL_WORLDS, self.worlds - 1)
            if changed:
                new, touched = replace(new, world=int(world)), True
        changed, points = ui.checkbox(POINTS_LABEL, new.points)
        if changed:
            new, touched = replace(new, points=bool(points)), True
        changed, forces = ui.checkbox(FORCES_LABEL, new.forces)
        if changed:
            new, touched = replace(new, forces=bool(forces)), True
        if touched and new != self.state:
            self.state = self._asked = new

    def show(self, *, world: int | None = None, points: bool | None = None, forces: bool | None = None) -> None:
        """Make the controls show what the program set (``view.show_worlds``, ``view.show_contacts``)."""
        s = self.state
        self.state = PanelState(s.world if world is None else int(world), s.points if points is None else bool(points),
                                s.forces if forces is None else bool(forces))

    def take(self) -> PanelState | None:
        """The newest request made in the panel since the last call, or None."""
        asked, self._asked = self._asked, None
        return asked


__all__ = ["ALL_WORLDS", "Panel", "PanelState"]
