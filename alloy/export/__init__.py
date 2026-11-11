# SPDX-License-Identifier: Apache-2.0
"""Frame export for live viewers (SCR-WP13-EXPORT): picked worlds' body poses, deformable mesh vertices, downsampled
particles and rod centre lines as plain numpy arrays, gathered on the device and copied without blocking the run.
Guide: docs/guide/frame-export.md."""

from typing import Any

from alloy.export.exporter import MAX_WORLDS, FrameExporter
from alloy.export.frames import Frame, Topology


def frame_export(sim: Any, worlds: Any, max_points: int = 20000) -> list[Frame]:
    """The current frame of each picked world, waiting for its copy. For a live viewer use ``FrameExporter``.

    Example::

        frames = frame_export(sim, [0, 3])          # or one world: frame_export(sim, 0)
        frames[0].body_pos, frames[0].body_quat_xyzw, frames[0].topology.body_names
    """
    from alloy.api._frame_export import frame_export as run  # noqa: PLC0415

    return run(sim, worlds, max_points)


__all__ = ["MAX_WORLDS", "Frame", "FrameExporter", "Topology", "frame_export"]
