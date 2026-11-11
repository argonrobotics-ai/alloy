# SPDX-License-Identifier: Apache-2.0
"""tool-joints: what the port reads from the film's books for its verdict (audit AB-gym-7, 2026-11-09).

The off-wall part of ``smear`` (Gym-Anything's off-region catch, task.py:159-167) is the mortar that left the trowel
other than by staying on it, and any film spilled over the chart's edges. A pass of it is not counted
(``concrete_ports.TOOL_JOINTS``): on the wall laid flat the lifted trowel keeps more than it can carry, so the gob that
falls from their blade on the standing wall cannot fall here.
"""

from __future__ import annotations

import numpy as np

OFF_KEYS = ("tool_dropped", "tool_dripped", "tool_released")   # [W, tools] m³ off a tool: always in the books
OFF_BANDS = "tool_overflow_off"                                 # the same past the chart's edge: with end bands only


def off_wall(read: dict) -> np.ndarray:
    """[W] m³ of mortar off the wall, from the film's books: what left the trowel other than staying on it (a lifted
    gob, drips, releases, overflow past the chart) and any film spilled over the chart's edges. Not counted: the
    drains' take (the mortar board, ``concrete_joints.boards``)."""
    spilled = np.asarray(read["spilled_volume"], float)
    keys = (*OFF_KEYS, OFF_BANDS) if OFF_BANDS in read else OFF_KEYS       # a renamed book is a KeyError, not a 0
    return spilled + sum(np.asarray(read[k], float).reshape(len(spilled), -1).sum(axis=1) for k in keys)


def window_differs(window: tuple[float, float, float, float] | None) -> tuple[str, ...]:
    """A window's verdict says it is not the whole infill's."""
    if window is None:
        return ()
    return (f"window: x {window[0]:g}-{window[1]:g} m, z {window[2]:g}-{window[3]:g} m of the infill, not the whole "
            f"wall: the shares count the window's joints only",)


__all__ = ["OFF_BANDS", "OFF_KEYS", "off_wall", "window_differs"]
