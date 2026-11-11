# SPDX-License-Identifier: Apache-2.0
"""Tangential motion of the gel under the touching object: a stick-slip estimate per image, on the GPU.

While an object touches the gel, the gel surface sticks to it: the contact patch is dragged by the object's in-plane
motion since the touch began (shear dx, dy and twist about the gel normal), as in FOTS (Zhao et al. 2024). Beyond a
friction-limited travel (``cap_mm``, ``cap_rad``) the object slips: the anchor follows the object and the gel keeps
the capped displacement. The touch ends when nothing presses the gel. The result per image is
``(dx_mm, dy_mm, twist_rad, active)`` and the centre of the patch ``(cx_mm, cy_mm)``, all in the sensor frame.

This is a kinematic estimate; a physical tangential field (gel displacement per pixel) replaces it when the engine
provides one (:meth:`alloy.camera.tactile.TactileSensor.render` ``shear_field=``).

Example::

    from alloy.camera.tactile.shear import StickSlip
    ss = StickSlip(num_images=4, device="cpu")
    # each frame, after PressMap.compute and GelOptics.spread(press, touched) (which fills optics.stats):
    # ss.update(optics.stats, press_map.gel_xf, model.shape_body, state.body_q)
    # ss.motion -> (4,) vec4 (dx, dy, twist, active); ss.center -> (4,) vec2 [mm]
"""
from __future__ import annotations

import warp as wp

from .stats import PressStats


@wp.func
def twist_about_z(q: wp.quat) -> float:
    """Rotation angle of q about +Z (its swing part removed)."""
    return 2.0 * wp.atan2(q[2], q[3])


@wp.func
def clamp_len(v: wp.vec2, cap: float) -> wp.vec2:
    n = wp.length(v)
    if n > cap and n > 0.0:
        return v * (cap / n)
    return v


@wp.kernel
def stick_slip(sums: wp.array(dtype=wp.vec3), lead: wp.array(dtype=wp.int32), shape_body: wp.array(dtype=wp.int32),
               gel_xf: wp.array(dtype=wp.transform), body_q: wp.array(dtype=wp.transform), cap_mm: float,
               cap_rad: float, anchor: wp.array(dtype=wp.transform), anchor_body: wp.array(dtype=wp.int32),
               offset: wp.array(dtype=wp.vec3), center: wp.array(dtype=wp.vec2), motion: wp.array(dtype=wp.vec4)):
    """Advance each image's stick-slip state by one frame."""
    i = wp.tid()
    s = sums[i]
    shp = lead[i]
    body = int(-1)
    if shp >= 0:
        body = shape_body[shp]
    if s[0] <= 0.0 or body < 0:
        anchor_body[i] = -1
        motion[i] = wp.vec4(0.0)
        return
    rel = wp.transform_multiply(wp.transform_inverse(gel_xf[i]), body_q[body])
    cen = wp.vec2(s[1] / s[0], s[2] / s[0])
    if anchor_body[i] != body:  # a new touch: stick here
        anchor_body[i] = body
        anchor[i] = rel
        offset[i] = wp.vec3(0.0)
        center[i] = cen
    a = anchor[i]
    c0 = center[i]
    q_obj = wp.transform_point(wp.transform_inverse(a), wp.vec3(c0[0], c0[1], 0.0) * 0.001)
    moved = wp.transform_point(rel, q_obj) * 1000.0
    o = offset[i]
    d = wp.vec2(moved[0] - c0[0], moved[1] - c0[1]) + wp.vec2(o[0], o[1])
    th = twist_about_z(wp.mul(wp.transform_get_rotation(rel), wp.quat_inverse(wp.transform_get_rotation(a)))) + o[2]
    if wp.length(d) > cap_mm or wp.abs(th) > cap_rad:  # slip: keep the capped travel, re-anchor on the object
        d = clamp_len(d, cap_mm)
        th = wp.clamp(th, -cap_rad, cap_rad)
        anchor[i] = rel
        offset[i] = wp.vec3(d[0], d[1], th)
        center[i] = cen - d
    motion[i] = wp.vec4(d[0], d[1], th, 1.0)


class StickSlip:
    """Per-image stick-slip state (device arrays) and its per-frame update."""

    def __init__(self, num_images: int, device: str, cap_mm: float = 0.6, cap_rad: float = 0.35):
        n, dev = int(num_images), str(device)
        self.device, self.cap_mm, self.cap_rad = dev, float(cap_mm), float(cap_rad)
        self.anchor = wp.zeros(n, dtype=wp.transform, device=dev)
        self.anchor_body = wp.full(n, -1, dtype=wp.int32, device=dev)
        self.offset = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.center = wp.zeros(n, dtype=wp.vec2, device=dev)
        self.motion = wp.zeros(n, dtype=wp.vec4, device=dev)

    def reset(self) -> None:
        """Forget every touch (call after resetting worlds)."""
        self.anchor_body.fill_(-1)
        self.motion.zero_()

    def update(self, stats: PressStats, gel_xf: wp.array, shape_body: wp.array, body_q: wp.array) -> wp.array:
        """One frame from the press summary (patch sums, touched shape): returns ``motion`` (images,) vec4
        (dx_mm, dy_mm, twist_rad, active)."""
        wp.launch(stick_slip, dim=stats.maxp.shape[0],
                  inputs=[stats.sums, stats.lead, shape_body, gel_xf, body_q, self.cap_mm, self.cap_rad],
                  outputs=[self.anchor, self.anchor_body, self.offset, self.center, self.motion], device=self.device)
        return self.motion
