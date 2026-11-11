# SPDX-License-Identifier: Apache-2.0
"""A test rig for the deposition laws: a carriage that drives one tool body along +x at a set speed.

It holds the body's y and attitude, and presses it onto the board with a set force on top of its weight. A marker
balanced on a domed nib is an inverted pendulum, so a law scene needs this hold, as a hand or an arm would give.
Gains are set from the body's own mass and inertia as bandwidths (velocity ``rate``, tilt ``tilt_rate`` at critical
damping), so the explicit forces stay stable whatever the tool weighs. Speed and press are per-world inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench


@wp.struct
class Gains:
    kv: wp.float32               # N·s/m (x speed and y velocity)
    kp: wp.float32               # N/m (y hold)
    k_tilt: wp.float32           # N·m/rad
    d_tilt: wp.float32           # N·m·s/rad


@wp.kernel
def _carriage(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body: wp.array[wp.int32],
              y0: wp.array[wp.float32], speed: wp.array[wp.float32], press: wp.array[wp.float32], g: Gains,
              body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    b = body[w]
    p = wp.transform_get_translation(body_q[b])
    qd = body_qd[b]
    fx = g.kv * (speed[w] - qd[0])
    fy = g.kp * (y0[w] - p[1]) - g.kv * qd[1]
    up = wp.transform_vector(body_q[b], wp.vec3(0.0, 0.0, 1.0))
    tau = g.k_tilt * wp.cross(up, wp.vec3(0.0, 0.0, 1.0)) - g.d_tilt * wp.vec3(qd[3], qd[4], qd[5])
    add_wrench(body_f, b, wp.spatial_vector(fx, fy, -press[w], tau[0], tau[1], tau[2]))


class Carriage(ForcePlugin):
    """Drives one tool body along +x at ``speed``, holds its y and attitude, presses it down by ``press``."""

    name = "deposition_carriage"
    summary = "test rig: drives a tool along +x at a set speed, holds y and attitude, presses it down"

    @dataclass
    class Config:
        body: str = "marker"
        speed: float = 0.1             # m/s along +x (per world: set_input("speed", ...))
        press: float = 1.0             # N down on top of the weight (per world: set_input("press", ...))
        rate: float = 400.0            # 1/s: velocity servo bandwidth (kv = mass·rate)
        tilt_rate: float = 200.0       # rad/s: attitude hold, critically damped

        def __post_init__(self) -> None:
            if not (self.rate > 0.0 and self.tilt_rate > 0.0):
                raise ValueError("rate and tilt_rate must be > 0")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.body = ctx.bodies(c.body)
        ids = self.body.numpy()
        mass = float(model.body_mass.numpy()[ids].min())
        inertia = min(float(np.linalg.eigvalsh(np.asarray(model.body_inertia.numpy()[b], np.float64)).min())
                      for b in ids)
        self.gains = Gains()
        self.gains.kv, self.gains.kp = mass * c.rate, mass * (0.25 * c.rate) ** 2
        self.gains.k_tilt, self.gains.d_tilt = inertia * c.tilt_rate**2, 2.0 * inertia * c.tilt_rate
        self.y0 = ctx.array(model.body_q.numpy()[ids, 1].astype(np.float32))
        self.speed = self.add_input("speed", ctx.full(ctx.world_count, float(c.speed)))
        self.press = self.add_input("press", ctx.full(ctx.world_count, float(c.press)))

    def check_dt(self, dt: float) -> None:
        c = self.config
        worst = max(c.rate * dt, 2.0 * c.tilt_rate * dt)
        if worst >= 1.0:
            raise PluginError(f"{self.label}: rate·dt = {c.rate * dt:.3g}, 2·tilt_rate·dt = {2 * c.tilt_rate * dt:.3g} "
                              f"must be < 1 at dt={dt:.3g} s: lower rate or tilt_rate")

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        wp.launch(_carriage, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.body, self.y0, self.speed, self.press, self.gains],
                  outputs=[forces.body_f], device=self.ctx.device)


__all__ = ["Carriage"]
