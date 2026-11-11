# SPDX-License-Identifier: Apache-2.0
"""Opt-in interactive tools of a viewer session (joint sliders, mouse dragging). A tool may change the simulation,
but only through the public ``Sim`` calls (``set_targets``, ``apply_wrench``) and only when the user acts in the
viewer: a session whose tools are not touched steps the same simulation, bit for bit, as one without tools.

A tool is an object with these methods; the session calls them on the simulation thread, holding its frame lock:

- ``attach(view)``: once, after the viewer has its model (read names and limits, register nothing itself);
- ``render(ui)``: every time the viewer draws its panel, with the viewer's widget set (``slider_float``,
  ``checkbox``, ``button``, ``text``, ``separator``); record what the user changed, do not touch the sim here;
- ``apply(view)``: once per frame, before the frame's step (also while paused): make the recorded changes with
  ``view.sim`` calls;
- ``draw(view)``: inside the frame, after the state is logged: extra drawing on ``view.backend`` (may do nothing);
- ``detach(view)``: once, when the viewer closes: remove anything the tool still holds on the sim (a drag force).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

log = logging.getLogger("alloy.viewer")


class Tool(Protocol):
    """The interface of an interactive tool (module docstring)."""

    def attach(self, view: Any) -> None: ...

    def render(self, ui: Any) -> None: ...

    def apply(self, view: Any) -> None: ...

    def draw(self, view: Any) -> None: ...

    def detach(self, view: Any) -> None: ...


class Tools:
    """The tools of one session, called in order."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = list(tools)

    def __bool__(self) -> bool:
        return bool(self.tools)

    def attach(self, view: Any) -> None:
        for tool in self.tools:
            tool.attach(view)
        register = getattr(view.backend, "register_ui_callback", None)
        if self.tools and callable(register):
            register(self.render)

    def render(self, ui: Any) -> None:
        for tool in self.tools:
            tool.render(ui)

    def apply(self, view: Any) -> None:
        for tool in self.tools:
            tool.apply(view)

    def draw(self, view: Any) -> None:
        for tool in self.tools:
            tool.draw(view)

    def detach(self, view: Any) -> None:
        for tool in self.tools:
            try:
                tool.detach(view)
            except Exception as exc:  # closing must not fail on a tool's clean-up
                log.warning("viewer tool %s did not detach cleanly: %s", type(tool).__name__, exc)


def make_tools(joints: bool, drag: bool) -> Tools:
    """The tools ``launch(joints=, drag=)`` asks for (each module loads only when asked)."""
    tools: list[Tool] = []
    if joints:
        from alloy.viewer._joints import JointSliders  # noqa: PLC0415

        tools.append(JointSliders())
    if drag:
        from alloy.viewer._drag import MouseDrag  # noqa: PLC0415

        tools.append(MouseDrag())
    return Tools(tools)


__all__ = ["Tool", "Tools", "make_tools"]
