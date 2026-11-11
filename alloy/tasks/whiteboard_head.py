# SPDX-License-Identifier: Apache-2.0
"""A pen head for the whiteboard tasks: an impedance-controlled tool holder that follows a polyline with pen up/down.

Like a plotter or a force-controlled hand on an arm: the head body tracks the path in x, y at a set speed (PD on
position and velocity). It presses down with a set force while the segment's pen is down, and hovers at ``hover_z``
(PD with gravity compensation) while it is up. It keeps the tool upright (attitude PD) and starts after ``start``
seconds. The path is a device array evaluated on the device clock (``ctx.time``), so a control period can be
captured as a graph. Gains are bandwidths on the head's own mass and inertia, so the explicit forces stay stable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench, com_world


@wp.struct
class HeadGains:
    kp: wp.float32               # N/m (x, y, and z while the pen is up)
    kd: wp.float32               # N·s/m
    k_tilt: wp.float32           # N·m/rad
    d_tilt: wp.float32           # N·m·s/rad
    press: wp.float32            # N down while the pen is down
    weight: wp.float32           # N: gravity compensation while the pen is up
    hover_z: wp.float32          # m: body height while the pen is up
    speed: wp.float32            # m/s along the path
    start: wp.float32            # s before the head moves


@wp.func
def path_point(points: wp.array[wp.vec3], arc: wp.array[wp.float32], s: wp.float32) -> wp.vec4:
    """(x, y, pen, segment index) at arc length ``s``: points[k] = (x, y, pen of the segment starting at k)."""
    n = points.shape[0]
    k = int(0)
    while k < n - 2 and s > arc[k + 1]:
        k += 1
    a, b = points[k], points[k + 1]
    span = wp.max(arc[k + 1] - arc[k], 1.0e-9)
    u = wp.clamp((s - arc[k]) / span, 0.0, 1.0)
    return wp.vec4(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2], float(k))


@wp.kernel
def _head(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
          head: wp.array[wp.int32], time: wp.array[wp.float64], points: wp.array[wp.vec3],
          arc: wp.array[wp.float32], g: HeadGains, body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    b = head[w]
    total = arc[arc.shape[0] - 1]
    s = g.speed * (wp.float32(time[w]) - g.start)
    target = path_point(points, arc, wp.clamp(s, 0.0, total))
    k = int(target[3])
    tangent = wp.vec3(points[k + 1][0] - points[k][0], points[k + 1][1] - points[k][1], 0.0)
    moving = wp.where(s > 0.0 and s < total, g.speed, 0.0)
    v_ref = moving * wp.normalize(tangent + wp.vec3(1.0e-12, 0.0, 0.0))
    down = target[2] > 0.5 and s > 0.0 and s < total
    com = com_world(body_q, body_com, b)
    qd = body_qd[b]
    v = wp.vec3(qd[0], qd[1], qd[2])
    fx = g.kp * (target[0] - com[0]) + g.kd * (v_ref[0] - v[0])
    fy = g.kp * (target[1] - com[1]) + g.kd * (v_ref[1] - v[1])
    fz = -g.press - g.kd * v[2]                       # damped descent; the steady press is unchanged
    if not down:
        fz = g.kp * (g.hover_z - com[2]) - g.kd * v[2] + g.weight
    up = wp.transform_vector(body_q[b], wp.vec3(0.0, 0.0, 1.0))
    tau = g.k_tilt * wp.cross(up, wp.vec3(0.0, 0.0, 1.0)) - g.d_tilt * wp.vec3(qd[3], qd[4], qd[5])
    add_wrench(body_f, b, wp.spatial_vector(fx, fy, fz, tau[0], tau[1], tau[2]))


class PenHead(ForcePlugin):
    """Follows a polyline with pen up/down: presses while drawing, hovers while travelling, keeps the tool upright."""

    name = "pen_head"
    summary = "whiteboard pen head: polyline path with pen up/down, press force, hover height, upright hold"

    @dataclass
    class Config:
        head: str = "marker"                         # body label of the head (the tool body)
        path: tuple[Any, ...] = ((0.0, 0.0, 1.0), (0.1, 0.0, 0.0))   # (x, y, pen of the segment starting here)
        speed: float = 0.1                           # m/s along the path
        press: float = 1.0                           # N while drawing, on top of the weight
        hover_z: float = 0.05                        # m: body height while the pen is up
        start: float = 0.0                           # s
        rate: float = 60.0                           # rad/s: position bandwidth (kp = m·rate², kd = 2·m·rate)
        tilt_rate: float = 200.0                     # rad/s: attitude bandwidth, critically damped

        def __post_init__(self) -> None:
            if len(self.path) < 2 or any(len(p) != 3 for p in self.path):
                raise ValueError("path needs at least two (x, y, pen) points")
            if not (self.speed > 0.0 and self.rate > 0.0 and self.tilt_rate > 0.0 and self.press >= 0.0):
                raise ValueError("speed, rate and tilt_rate must be > 0 and press >= 0")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.head = ctx.bodies(c.head)
        ids = self.head.numpy()
        mass = float(model.body_mass.numpy()[ids].min())
        inertia = min(float(np.linalg.eigvalsh(np.asarray(model.body_inertia.numpy()[b], np.float64)).min())
                      for b in ids)
        pts = np.asarray(c.path, np.float32)
        self.points = ctx.array(pts, wp.vec3)
        self.arc = ctx.array(np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]))
        self.gains = HeadGains()
        g = self.gains
        g.kp, g.kd = mass * c.rate**2, 2.0 * mass * c.rate
        g.k_tilt, g.d_tilt = inertia * c.tilt_rate**2, 2.0 * inertia * c.tilt_rate
        g.press, g.weight, g.hover_z = float(c.press), float(mass * abs(float(ctx.gravity[2]))), float(c.hover_z)
        g.speed, g.start = float(c.speed), float(c.start)

    @property
    def duration(self) -> float:
        """Seconds from the start until the head reaches the path's end."""
        pts = np.asarray(self.config.path, np.float64)
        return self.config.start + float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum()) / self.config.speed

    def check_dt(self, dt: float) -> None:
        c = self.config
        worst = {"rate": 2.0 * c.rate * dt, "tilt_rate": 2.0 * c.tilt_rate * dt}
        bad = {k: round(v, 3) for k, v in worst.items() if v >= 1.0}
        if bad:
            raise PluginError(f"{self.label}: 2·bandwidth·dt must be < 1 at dt={dt:.3g} s: {bad}; lower them")

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        wp.launch(_head, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.head, self.ctx.time, self.points,
                          self.arc, self.gains],
                  outputs=[forces.body_f], device=self.ctx.device)


def strokes_to_path(strokes: Sequence[Sequence[tuple[float, float]]], start: tuple[float, float],
                    end: tuple[float, float], lead: float = 0.006) -> tuple[tuple[float, float, float], ...]:
    """The head's ``path`` for polylines drawn in order: pen-up travel from ``start`` to each stroke and on to ``end``.
    The pen goes down ``lead`` metres before each stroke so the tip lands at its start."""
    path: list[tuple[float, float, float]] = []
    cur = (float(start[0]), float(start[1]))
    for stroke in strokes:
        first = (float(stroke[0][0]), float(stroke[0][1]))
        path.append((*cur, 0.0))
        gap = math.hypot(first[0] - cur[0], first[1] - cur[1])
        if gap > lead:
            k = lead / gap
            path.append((first[0] + k * (cur[0] - first[0]), first[1] + k * (cur[1] - first[1]), 1.0))
        path += [(float(x), float(y), 1.0) for x, y in stroke[:-1]]
        cur = (float(stroke[-1][0]), float(stroke[-1][1]))
    path += [(*cur, 0.0), (float(end[0]), float(end[1]), 0.0)]
    return tuple(path)


__all__ = ["PenHead", "strokes_to_path"]
