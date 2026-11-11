# SPDX-License-Identifier: Apache-2.0
"""The frame_change_pre handler that poses every entity at scene.frame_current + scene.frame_subframe.

Cycles calls frame_change_pre at each motion-blur step (verified on Blender 4.5: a render of frame 5 with a
1-frame shutter calls it at 5.0, 4.5, 5.5, 5.0), so one handler drives both frames and motion-blur sub-steps.
"""

from __future__ import annotations

import logging
from typing import Protocol

import bpy

log = logging.getLogger(__name__)


class Updater(Protocol):
    def update(self, fr: float, center: int) -> None:
        """Pose the entity at fractional frame fr; `center` is the frame being rendered."""


class Runtime:
    """Handler state: the registered updaters, the frame being rendered, and the first handler error."""

    def __init__(self) -> None:
        self.updaters: list[Updater] = []
        self.center: int | None = None
        self.error: BaseException | None = None
        self.calls: list[float] = []


RUNTIME = Runtime()


def on_frame(scene: bpy.types.Scene, depsgraph=None) -> None:
    fr = float(scene.frame_current) + float(scene.frame_subframe)
    center = RUNTIME.center if RUNTIME.center is not None else int(round(fr))
    RUNTIME.calls.append(fr)
    try:
        for u in RUNTIME.updaters:
            u.update(fr, center)
    except Exception as exc:  # noqa: BLE001 - recorded, re-raised by the render loop
        if RUNTIME.error is None:
            RUNTIME.error = exc
        log.exception("frame handler failed at frame %.3f", fr)


def install(updaters: list[Updater]) -> None:
    RUNTIME.updaters = list(updaters)
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") == "on_frame":
            bpy.app.handlers.frame_change_pre.remove(h)
    bpy.app.handlers.frame_change_pre.append(on_frame)
