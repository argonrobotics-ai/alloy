# SPDX-License-Identifier: Apache-2.0
"""The sanding head's controller for task_sand_plate: a Cartesian impedance head that rasters a pad over a plate.

Like a gantry sanding cell or a force-controlled flange on an arm: the head body follows a serpentine raster in x, y
(PD on position and velocity), presses down with a constant force, keeps its pad level (attitude PD) and spins the
pad at a set rate (velocity servo about the pad axis). All gains act explicitly on the head's own mass and inertia;
``check_dt`` refuses gains that would be unstable at the exchange step. The path runs on the device clock
(``ctx.time``), so a control period can be captured as a graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench, com_world

STABLE = 1.0


@wp.struct
class Raster:
    start: wp.vec2           # first row's start (x, y) [m]
    length: wp.float32       # row length along +x [m]
    spacing: wp.float32      # row spacing along +y [m]
    rows: wp.int32
    speed: wp.float32        # [m/s] along the path


@wp.struct
class Gains:
    kp: wp.float32           # [N/m]
    kd: wp.float32           # [N·s/m]
    k_tilt: wp.float32       # [N·m/rad]
    d_tilt: wp.float32       # [N·m·s/rad]
    k_spin: wp.float32       # [N·m·s/rad]
    spin: wp.float32         # pad rate [rad/s], about the pad axis pointing up
    downforce: wp.float32    # [N]


@wp.func
def raster_point(r: Raster, s: wp.float32) -> wp.vec4:
    """(x, y, dx/ds, dy/ds) at arc length ``s`` of the serpentine: rows along +x / -x joined by +y steps."""
    period = r.length + r.spacing
    total = float(r.rows) * r.length + float(r.rows - 1) * r.spacing
    s = wp.clamp(s, 0.0, total)
    k = wp.min(int(wp.floor(s / period)), r.rows - 1)
    rest = s - float(k) * period
    forward = k % 2 == 0
    y = r.start[1] + float(k) * r.spacing
    if rest <= r.length:
        along = wp.where(forward, rest, r.length - rest)
        return wp.vec4(r.start[0] + along, y, wp.where(forward, 1.0, -1.0), 0.0)
    x = wp.where(forward, r.start[0] + r.length, r.start[0])
    return wp.vec4(x, y + rest - r.length, 0.0, 1.0)


@wp.kernel
def _head_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 head: wp.array[wp.int32], time: wp.array[wp.float64], raster: Raster, gains: Gains,
                 body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    b = head[w]
    total = float(raster.rows) * raster.length + float(raster.rows - 1) * raster.spacing
    s = raster.speed * wp.float32(time[w])
    target = raster_point(raster, s)
    moving = wp.where(s < total, raster.speed, 0.0)
    com = com_world(body_q, body_com, b)
    qd = body_qd[b]
    v = wp.vec3(qd[0], qd[1], qd[2])
    omega = wp.vec3(qd[3], qd[4], qd[5])
    fx = gains.kp * (target[0] - com[0]) + gains.kd * (moving * target[2] - v[0])
    fy = gains.kp * (target[1] - com[1]) + gains.kd * (moving * target[3] - v[1])
    up = wp.transform_vector(body_q[b], wp.vec3(0.0, 0.0, 1.0))
    tilt = wp.cross(up, wp.vec3(0.0, 0.0, 1.0))                         # rotates the pad axis back to vertical
    spin_rate = wp.dot(omega, up)
    wobble = omega - spin_rate * up
    torque = gains.k_tilt * tilt - gains.d_tilt * wobble + gains.k_spin * (gains.spin - spin_rate) * up
    add_wrench(body_f, b, wp.spatial_vector(fx, fy, -gains.downforce, torque[0], torque[1], torque[2]))


class SandingHead(ForcePlugin):
    """Impedance-controlled sanding head: rasters a body over a plate, presses it down and spins its pad."""

    name = "sanding_head"
    summary = "Cartesian impedance head: serpentine raster, constant downforce, level pad, pad spin servo"

    @dataclass
    class Config:
        head: str = "sander"                               # body label of the head
        start: tuple[float, float] = (-0.1, -0.06)         # first row start [m]
        length: float = 0.2                                # row length [m]
        spacing: float = 0.03                              # row spacing [m]
        rows: int = 5
        speed: float = 0.1                                 # m/s along the path
        downforce: float = 20.0                            # N on top of the head's weight
        spin: float = 60.0                                 # rad/s pad rate
        kp: float = 3000.0                                 # N/m
        kd: float = 130.0                                  # N·s/m
        k_tilt: float = 20.0                               # N·m/rad
        d_tilt: float = 0.25                               # N·m·s/rad
        k_spin: float = 0.5                                # N·m·s/rad

        def __post_init__(self) -> None:
            if self.rows < 1 or self.length <= 0.0 or self.spacing < 0.0 or self.speed < 0.0:
                raise ValueError("rows >= 1, length > 0, spacing >= 0 and speed >= 0 are required")
            gains = (self.kp, self.kd, self.k_tilt, self.d_tilt, self.k_spin, self.downforce)
            if not all(math.isfinite(g) and g >= 0.0 for g in gains):
                raise ValueError("gains and downforce must be finite and >= 0")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.head = ctx.bodies(c.head)
        b = int(self.head.numpy()[0])
        self.mass = float(model.body_mass.numpy()[b])
        inertia = np.linalg.eigvalsh(np.asarray(model.body_inertia.numpy()[b], np.float64).reshape(3, 3))
        self.inertia = float(inertia.min())
        self.raster, self.gains = Raster(), Gains()
        self.raster.start = wp.vec2(*c.start)
        self.raster.length, self.raster.spacing = float(c.length), float(c.spacing)
        self.raster.rows, self.raster.speed = int(c.rows), float(c.speed)
        for name in ("kp", "kd", "k_tilt", "d_tilt", "k_spin", "spin", "downforce"):
            setattr(self.gains, name, float(getattr(c, name)))

    @property
    def duration(self) -> float:
        """Seconds to run the whole raster."""
        c = self.config
        return (c.rows * c.length + (c.rows - 1) * c.spacing) / max(c.speed, 1e-12)

    def check_dt(self, dt: float) -> None:
        c = self.config
        checks = {"kp": math.sqrt(c.kp / self.mass) * dt, "kd": c.kd * dt / self.mass,
                  "k_tilt": math.sqrt(c.k_tilt / self.inertia) * dt, "d_tilt": c.d_tilt * dt / self.inertia,
                  "k_spin": c.k_spin * dt / self.inertia}
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: gains unstable at dt={dt:.3g} s: {bad} (each < {STABLE}); lower them")

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        wp.launch(_head_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.head, self.ctx.time,
                          self.raster, self.gains],
                  outputs=[forces.body_f], device=self.ctx.device)


__all__ = ["SandingHead", "raster_point"]
