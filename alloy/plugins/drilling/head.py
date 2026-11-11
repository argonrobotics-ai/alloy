# SPDX-License-Identifier: Apache-2.0
"""A feed-controlled drilling spindle: the companion controller of the drilling laws and task_drill_plate.

Like a drilling end effector or a quill on a drill press: the head body holds its start position across its feed
direction and keeps its +z axis against that direction (PD). It feeds along the direction at a set rate (PD on a ramp
x_ref = x0 + direction·min(feed_rate·t, travel), plus gravity feed-forward), and spins at a set rate (velocity servo
about its +z axis). The direction is straight down by default; a tilted one drills a tilted hole (the axis
law), and ``travel`` stops the feed (a blind hole). Feed rate, spin, direction and travel are per-world inputs
(``set_input("feed_rate", ...)``, ``set_input("spin", ...)``, ``set_direction(...)``, ``set_input("travel", ...)``),
so one batch can drill at several feeds and angles. All gains act explicitly on the head's own mass and inertia;
``check_dt`` refuses unstable gains. The ramp runs on the device clock (``ctx.time``), so a control period can be
captured as a graph.

At steady state the head lags its ramp by thrust / kp but moves at the ramp's rate, so the feed per revolution is
feed_rate·2π / spin.
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
from alloy.plugins.npgeom import quat_rotate

STABLE = 1.0


@wp.struct
class HeadGains:
    kp: wp.float32           # [N/m]
    kd: wp.float32           # [N·s/m]
    k_tilt: wp.float32       # [N·m/rad]
    d_tilt: wp.float32       # [N·m·s/rad]
    k_spin: wp.float32       # [N·m·s/rad]
    weight: wp.float32       # gravity feed-forward [N], along +z
    delay: wp.float32        # s before the feed starts


@wp.kernel
def _feed_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 head: wp.array[wp.int32], hold: wp.array[wp.vec3], time: wp.array[wp.float64],
                 feed_rate: wp.array[wp.float32], spin: wp.array[wp.float32], direction: wp.array[wp.vec3],
                 travel: wp.array[wp.float32], gains: HeadGains,
                 body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    b = head[w]
    t = wp.max(wp.float32(time[w]) - gains.delay, 0.0)
    ramp = feed_rate[w] * t
    rate = wp.where(wp.float32(time[w]) >= gains.delay and ramp < travel[w], feed_rate[w], 0.0)
    down = direction[w]
    target = hold[w] + down * wp.min(ramp, travel[w])
    com = com_world(body_q, body_com, b)
    qd = body_qd[b]
    v = wp.vec3(qd[0], qd[1], qd[2])
    omega = wp.vec3(qd[3], qd[4], qd[5])
    f = gains.kp * (target - com) + gains.kd * (down * rate - v) + wp.vec3(0.0, 0.0, gains.weight)
    up = wp.transform_vector(body_q[b], wp.vec3(0.0, 0.0, 1.0))
    tilt = wp.cross(up, -down)                                   # rotates the head axis back against the feed
    spin_rate = wp.dot(omega, up)
    wobble = omega - spin_rate * up
    torque = gains.k_tilt * tilt - gains.d_tilt * wobble + gains.k_spin * (spin[w] - spin_rate) * up
    add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], torque[0], torque[1], torque[2]))


class FeedHead(ForcePlugin):
    """Feed-controlled drilling spindle: holds its axis against a feed direction (down by default), feeds along it
    at a set rate and spins at a set rate."""

    name = "feed_head"
    summary = "drilling spindle controller: axis hold, feed ramp along a direction, spin servo (per-world inputs)"

    @dataclass
    class Config:
        head: str = "spindle"                     # body label of the head
        feed_rate: float = 5.0e-3                 # m/s down (per world: set_input("feed_rate", ...))
        spin: float = 100.0 * math.pi             # rad/s about the head's +z (3000 rpm)
        delay: float = 0.0                        # s before the feed starts
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0)   # unit feed direction, world (per world: input)
        travel: float = 1.0e9                     # m along the direction, then the feed stops (per world: input)
        kp: float = 1.0e6                         # N/m
        kd: float = 3.0e3                         # N·s/m
        k_tilt: float = 200.0                     # N·m/rad
        d_tilt: float = 2.0                       # N·m·s/rad
        k_spin: float = 1.0                       # N·m·s/rad

        def __post_init__(self) -> None:
            gains = (self.kp, self.kd, self.k_tilt, self.d_tilt, self.k_spin, self.delay)
            if not all(math.isfinite(g) and g >= 0.0 for g in gains):
                raise ValueError("gains and delay must be finite and >= 0")
            if not (math.isfinite(self.feed_rate) and math.isfinite(self.spin)):
                raise ValueError("feed_rate and spin must be finite")
            n = float(np.linalg.norm(self.direction))
            if not (math.isfinite(n) and abs(n - 1.0) < 1.0e-6):
                raise ValueError(f"direction must be a unit vector, got {self.direction}")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.head = ctx.bodies(c.head)
        ids = self.head.numpy()
        b = int(ids[0])
        self.mass = float(model.body_mass.numpy()[b])
        if self.mass <= 0.0:
            raise PluginError(f"head body {c.head!r} has no mass")
        self.inertia = float(np.linalg.eigvalsh(np.asarray(model.body_inertia.numpy()[b], np.float64)
                                                .reshape(3, 3)).min())
        q, com = model.body_q.numpy(), model.body_com.numpy()
        hold = np.stack([q[i, :3] + quat_rotate(q[i, 3:], com[i]) for i in ids]).astype(np.float32)
        self.hold = ctx.array(hold, wp.vec3)
        W = ctx.world_count
        self.feed_rate = self.add_input("feed_rate", ctx.full(W, float(c.feed_rate), wp.float32))
        self.spin = self.add_input("spin", ctx.full(W, float(c.spin), wp.float32))
        self.travel = self.add_input("travel", ctx.full(W, float(c.travel), wp.float32))
        self.direction = self.add_input("direction", ctx.array(np.tile(np.asarray(c.direction, np.float32), (W, 1)),
                                                               wp.vec3))
        self.gains = HeadGains()
        for name in ("kp", "kd", "k_tilt", "d_tilt", "k_spin", "delay"):
            setattr(self.gains, name, float(getattr(c, name)))
        self.gains.weight = float(-self.mass * ctx.gravity[2])

    def check_dt(self, dt: float) -> None:
        c = self.config
        checks = {"kp": math.sqrt(c.kp / self.mass) * dt, "kd": c.kd * dt / self.mass,
                  "k_tilt": math.sqrt(c.k_tilt / self.inertia) * dt, "d_tilt": c.d_tilt * dt / self.inertia,
                  "k_spin": c.k_spin * dt / self.inertia}
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: gains unstable at dt={dt:.3g} s: {bad} (each < {STABLE}); lower them")

    def set_direction(self, direction: Any) -> None:
        """Feed direction per world (or one for all), host side between periods; normalized here."""
        d = np.broadcast_to(np.asarray(direction, np.float64), (self.ctx.world_count, 3))
        n = np.linalg.norm(d, axis=1, keepdims=True)
        if not (np.isfinite(d).all() and (n > 0.0).all()):
            raise PluginError(f"{self.label}: directions must be finite and non-zero")
        self.set_input("direction", d / n)

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        wp.launch(_feed_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.head, self.hold, self.ctx.time,
                          self.feed_rate, self.spin, self.direction, self.travel, self.gains],
                  outputs=[forces.body_f], device=self.ctx.device)


__all__ = ["FeedHead"]
