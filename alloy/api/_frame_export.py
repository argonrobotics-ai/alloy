# SPDX-License-Identifier: Apache-2.0
"""``frame_export(sim, worlds, max_points)``: one synchronous frame per picked world (SCR-WP13-EXPORT).

For a live viewer next to a training run, keep a ``FrameExporter`` instead: ``capture()`` between steps and ``poll()``
from the publishing thread never block the run. This helper is the simple form (it waits for its copy). It keeps the
last exporter it made on the Sim (one, for the last worlds and max_points asked) and replaces it when those change.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.export import Frame, FrameExporter
from alloy.export.exporter import _picks


def frame_export(sim: Any, worlds: list[int] | int, max_points: int = 20000) -> list[Frame]:
    """The current frame of each picked world. Example: ``frame_export(sim, [0, 3])[0].body_pos``."""
    picked = _picks(sim, worlds)                         # checked, and an int (numpy too) is one world
    key = (tuple(picked), int(max_points))
    last = sim.__dict__.get("_frame_exporter")
    if last is None or last[0] != key:
        last = (key, FrameExporter(sim, picked, max_points))
        sim.__dict__["_frame_exporter"] = last
    exp = last[1]
    exp.flush()
    exp.poll()                                           # an older unpolled frame is not this one
    frames = None
    if exp.capture():                                    # False only while the device records a CUDA graph
        exp.flush()
        frames = exp.poll()
    if frames is None:
        from alloy.api.errors import SimulationError  # noqa: PLC0415

        raise SimulationError("frame_export got no frame: the device was capturing a CUDA graph",
                              "call frame_export between steps, not inside a graph capture", field="worlds")
    return frames


__all__ = ["frame_export"]
