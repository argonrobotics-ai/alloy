# SPDX-License-Identifier: Apache-2.0
"""Camera rig: Catmull-Rom eye/target keys over bundle time, lens, depth of field and focus tracking."""

from __future__ import annotations

import logging
from typing import Callable

import bpy
import numpy as np
from mathutils import Matrix

import motion
from io_bundle import Bundle
from look import Look, LookError

log = logging.getLogger(__name__)


CLIP_END_M = 200.0


def clip_end_m(look: Look) -> float:
    """Camera far clip: 200 m, or past the end of world.fade so the faded ground is not cut off at a hard line."""
    fade = look.world.get("fade") or {}
    end = float(fade.get("start_m", 0.0)) + float(fade.get("depth_m", 0.0)) if fade else 0.0
    return max(CLIP_END_M, 1.1 * end)


class CameraRig:
    """The look camera `name`, posed per (sub)frame; focus follows a label, a point or a fixed distance."""

    def __init__(self, scene: bpy.types.Scene, bundle: Bundle, look: Look, name: str,
                 origin_fn: Callable[[str], Callable[[float], np.ndarray]] | None):
        if name not in look.cameras:
            raise LookError(f"--camera {name!r} not in look.cameras ({sorted(look.cameras)})")
        cfg = look.cameras[name]
        self.bundle = bundle
        keys = cfg["keys"]
        self.times = np.array([float(k.get("t", 0.0)) for k in keys])
        self.eyes = np.array([k["eye"] for k in keys], np.float64)
        self.targets = np.array([k["target"] for k in keys], np.float64)
        data = bpy.data.cameras.new(name)
        data.lens = float(cfg.get("lens_mm", 50.0))
        data.sensor_width = float(cfg.get("sensor_mm", 36.0))
        data.sensor_fit = "AUTO"
        data.clip_start, data.clip_end = 0.002, clip_end_m(look)
        self.obj = bpy.data.objects.new(f"camera_{name}", data)
        scene.collection.objects.link(self.obj)
        scene.camera = self.obj
        res = cfg.get("resolution", look.render["resolution"])
        scene.render.resolution_x, scene.render.resolution_y = int(res[0]), int(res[1])
        scene.render.resolution_percentage = 100
        self.focus_point: Callable[[float], np.ndarray] | None = None
        self._setup_dof(data, cfg, origin_fn)
        self._last = None
        self.update(0.0, 0)

    def _setup_dof(self, data, cfg: dict, origin_fn) -> None:
        fstop = float(cfg.get("fstop", 0.0))
        data.dof.use_dof = fstop > 0.0
        if fstop <= 0.0:
            return
        data.dof.aperture_fstop = fstop
        data.dof.aperture_blades = int(cfg.get("blades", 0))
        focus = cfg.get("focus") or {}
        if "label" in focus:
            if origin_fn is None:
                raise LookError("focus.label needs rigid shapes in the bundle")
            self.focus_point = origin_fn(focus["label"])
        elif "point" in focus:
            pt = np.asarray(focus["point"], np.float64)
            self.focus_point = lambda fr: pt
        else:
            data.dof.focus_distance = float(focus.get("distance", 1.0))

    def pose(self, fr: float) -> np.ndarray:
        """4x4 camera-to-world matrix at fractional bundle frame fr."""
        t = self.bundle.time_at(fr)
        eye = motion.catmull_rom(self.times, self.eyes, t)
        target = motion.catmull_rom(self.times, self.targets, t)
        return motion.look_at(eye, target)

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        m = self.pose(fr)
        self.obj.matrix_world = Matrix(m.tolist())
        if self.focus_point is not None:
            # Blender focuses on the plane at this distance along the view axis (-Z of the camera).
            dist = float(np.dot(self.focus_point(fr) - m[:3, 3], -m[:3, 2]))
            self.obj.data.dof.focus_distance = max(dist, 0.001)
