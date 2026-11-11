# SPDX-License-Identifier: Apache-2.0
"""A tilt table for vehicle stability and grade laws: gravity tilted slowly toward a ``downhill`` direction of the
ground plane, as ISO 22915's tilting platform tilts a truck (sideways, or along it for a grade).

The solver keeps its own gravity ``(0, 0, -g)``. This plug-in adds the difference to tilted gravity
``g·(sin β·d_x, sin β·d_y, -cos β)`` on each named body at its centre of mass, ``m·g·(sin β·d, 1 - cos β)``, with ``d``
the unit downhill direction (default -y). The tilt holds at ``start`` for ``hold`` seconds (the vehicle settles on its
tyres), grows at ``rate`` up to ``stop``, stays there for ``dwell`` seconds, then falls back at ``rate`` to ``end``
(``end`` >= ``stop``: it stays). ``beta`` [W] records the tilt every exchange. The ground stays level in the solver's
frame: with the vehicle plug-in's flat ground this is exactly a tilted ground under upright gravity.
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
from alloy.plugins.funcs import add_wrench


@wp.struct
class Schedule:
    start: wp.float32            # rad
    hold: wp.float32             # s
    rate: wp.float32             # rad/s
    stop: wp.float32             # rad
    dwell: wp.float32            # s at stop
    end: wp.float32              # rad after falling back (>= stop: no fall)
    down: wp.vec2                # unit downhill direction in the ground plane


@wp.func
def tilt_angle(t: wp.float64, s: Schedule) -> wp.float32:
    tt = wp.float32(t) - s.hold
    if tt <= 0.0:
        return s.start
    if s.rate <= 0.0:
        return s.start
    up = (s.stop - s.start) / s.rate
    if tt <= up:
        return s.start + s.rate * tt
    if tt <= up + s.dwell or s.end >= s.stop:
        return s.stop
    return wp.max(s.end, s.stop - s.rate * (tt - up - s.dwell))


@wp.kernel
def tilt_kernel(time: wp.array[wp.float64], bodies: wp.array2d[wp.int32], mass: wp.array[wp.float32], g: wp.float32,
                s: Schedule, body_f: wp.array[wp.spatial_vector]):
    """ADD each named body's share of tilted gravity minus the solver's (one thread per world)."""
    w = wp.tid()
    beta = tilt_angle(time[w], s)
    sb = wp.sin(beta)
    for k in range(bodies.shape[1]):
        b = bodies[w, k]
        f = mass[b] * g * wp.vec3(sb * s.down[0], sb * s.down[1], 1.0 - wp.cos(beta))
        add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0))


@wp.kernel
def angle_kernel(time: wp.array[wp.float64], s: Schedule, beta: wp.array[wp.float32]):
    w = wp.tid()
    beta[w] = tilt_angle(time[w], s)


class TiltTable(ForcePlugin):
    """Tilts gravity toward ``downhill`` (default -y, sideways) on the named bodies, slowly: a tilting platform.

    Example (hold 0.4 s, then 3°/s up to 40°)::

        TiltTable(bodies=("forklift", "axle"), hold=0.4, rate=math.radians(3.0), stop=math.radians(40.0))
    """

    name = "tilt_table"
    summary = "tilting platform: gravity turned sideways about world x on named bodies, ramped slowly"

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("chassis",)      # body labels the tilted gravity acts on (every body of the vehicle)
        start: float = 0.0                          # rad: tilt during the hold
        hold: float = 0.0                           # s at ``start`` before the ramp
        rate: float = math.radians(2.0)             # rad/s
        stop: float = math.radians(45.0)            # rad: largest tilt
        dwell: float = 0.0                          # s at ``stop`` before falling back
        end: float = math.pi                        # rad to fall back to (>= stop: stays at stop)
        downhill: tuple[float, float] = (0.0, -1.0)  # ground-plane direction gravity tilts toward

        def __post_init__(self) -> None:
            values = (self.start, self.hold, self.rate, self.stop, self.dwell, self.end, *self.downhill)
            if not all(math.isfinite(v) for v in values) or min(self.hold, self.rate, self.dwell, self.end) < 0.0:
                raise ValueError("tilt_table: start, hold, rate, stop, dwell, end and downhill must be finite, >= 0")
            if not (0.0 <= self.start <= self.stop < 0.5 * math.pi) or not self.bodies:
                raise ValueError("tilt_table: need 0 <= start <= stop < 90° and at least one body")
            if len(self.downhill) != 2 or math.hypot(*self.downhill) == 0.0:
                raise ValueError("tilt_table: downhill is a non-zero (x, y) direction")

    def setup(self, ctx: Any) -> None:
        c = self.config
        missing = [b for b in c.bodies if not ctx.has(b)]
        if missing:
            raise PluginError(f"{self.label}: no body labelled {missing}; bodies: {ctx.labels('body')[:20]}")
        index = np.stack([ctx.bodies(b).numpy() for b in c.bodies], axis=1).astype(np.int32)
        self.index = ctx.array(index, wp.int32)
        self.g = float(np.linalg.norm(ctx.gravity))
        self.schedule = self._schedule()
        self.beta = self.add_state("beta", ctx.zeros(ctx.world_count))

    def _schedule(self) -> Schedule:
        c = self.config
        s = Schedule()
        s.start, s.hold, s.rate, s.stop, s.dwell, s.end = c.start, c.hold, c.rate, c.stop, c.dwell, c.end
        n = math.hypot(*c.downhill)
        s.down = wp.vec2(c.downhill[0] / n, c.downhill[1] / n)
        return s

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        wp.launch(tilt_kernel, dim=self.ctx.world_count,
                  inputs=[self.ctx.time, self.index, self.ctx.model.body_mass, self.g, self.schedule],
                  outputs=[forces.body_f], device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        wp.launch(angle_kernel, dim=self.ctx.world_count, inputs=[self.ctx.time, self.schedule],
                  outputs=[self.beta], device=self.ctx.device)


__all__ = ["TiltTable"]
