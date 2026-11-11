# SPDX-License-Identifier: Apache-2.0
"""Laws of the vehicle plug-in's drivable surfaces (SCR-WP8-12 F1b, WP13 review), against closed forms.

- **R-3 profile**: the test cart drives at 0.5 m/s over a polyline of static boxes. The polyline runs over the ground
  plane: the foot of a 13.6 % ramp (GA's yard ramp grade), its crest onto a level deck, a seam where two coplanar decks
  meet, and a 25 mm step down.
  - A disc's centre follows the upper envelope of the faces offset by R and of circles of radius R about every vertex.
    The front wheel's centre stays on it within 1 mm plus its radial deflection F_z/k.
  - Over the crest, where that path is smooth, the cart's vertical acceleration stays within 5 % of g of the one the
    two wheels' paths give it (v²/R on the arc). A face switch there would jolt it.
  - Across the seam, the front wheel's load stays within 2 % of the level value: two coplanar faces are one surface.
  - The foot and the step have kinks in the path, so they really jolt the wheel; only the path is checked there.
- **R-4 sag**: a 2,500 kg floor rides on a second vehicle, four braked "wheels" of 1e5 N/m (the trailer's air
  springs in F4). With the cart parked at its centre it sinks ``(m_floor + m_cart)·g/(4·k)``, within 3 %.
- **R-4 momentum**: the cart drives, then brakes, on a 2 t floor that slides freely in x on a prismatic joint. The
  tyre's reaction goes on the floor, so cart plus floor keep zero x momentum, to 5e-6 of the cart's (float32 over
  2,400 steps: 1.2e-7 to 1.2e-6 measured).
- **Seam under a parked wheel** (WP13): the cart drives off a static deck onto the sprung floor (coplanar at rest) and
  parks with its front wheel 5 mm past the seam; then a slowly ramped force pushes the floor down 5 mm more, so the
  seam goes off coplanar under the wheel. The blended rule keeps the front wheel's load change per period within 5 %
  and the treads settle. (Rolling onto the floor is a real step down of its sag: not checked.) A geometric
  sweep (the neighbour face rising 0 -> 5 mm and tilting 0 -> 8° under a parked wheel) keeps the weighted contact depth
  changing no faster than the motion.
- **R-2 on a ramp face**: GA's H2.5A truck, front brakes set, parked on a static 13.6 % ramp box, both ways round:
  it creeps at most 1 mm in 10 s. This is the tilt-table law on a real face.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.funcs import add_wrench
from alloy.plugins.harness.scene import body_xform
from alloy.plugins.vehicle import laws as vl
from alloy.plugins.vehicle import laws_axle as ax

GRADE, RISE, STEP_DOWN = 0.136, 0.3, 0.025
FOOT, DECK = 1.5, 2.0                                  # m: ramp foot x; deck length (>= the cart's wheelbase)
RUN = RISE / GRADE
SEGMENTS = (("ramp", (FOOT, 0.0), (FOOT + RUN, RISE)),
            ("deck_a", (FOOT + RUN, RISE), (FOOT + RUN + DECK, RISE)),
            ("deck_b", (FOOT + RUN + DECK, RISE), (FOOT + RUN + 2 * DECK, RISE)),
            ("deck_c", (FOOT + RUN + 2 * DECK, RISE - STEP_DOWN), (FOOT + RUN + 3 * DECK, RISE - STEP_DOWN)))
THICK, WIDE = 0.04, 1.0                                # box thickness, half width [m]
CART = {"radial_stiffness": (2.0e6,), "radial_damping": (1.0e4,), "surfaces": tuple(s[0] for s in SEGMENTS)}


# --- geometry -------------------------------------------------------------------------------------------------------

def segment_box(b: newton.ModelBuilder, label: str, p0: tuple[float, float], p1: tuple[float, float],
                body: int = -1, half_width: float = WIDE) -> None:
    """A box whose top face runs from ``p0`` to ``p1`` (x, z) across y; static unless ``body``."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    length, theta = math.hypot(dx, dz), math.atan2(dz, dx)
    normal = np.array([-math.sin(theta), 0.0, math.cos(theta)])
    mid = np.array([0.5 * (p0[0] + p1[0]), 0.0, 0.5 * (p0[1] + p1[1])]) - 0.5 * THICK * normal
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -theta)
    b.add_shape_box(body, xform=wp.transform(wp.vec3(*mid), rot), hx=0.5 * length, hy=half_width, hz=0.5 * THICK,
                    label=label)


def centre_path(x: np.ndarray, radius: float, segments=SEGMENTS) -> np.ndarray:
    """Height of a disc's centre rolling over the ground plane and the segments: the upper envelope of the faces
    offset by R and of circles of radius R about their ends."""
    z = np.full_like(x, radius, dtype=np.float64)
    for _, p0, p1 in segments:
        d = np.subtract(p1, p0) / math.dist(p0, p1)
        n = np.array([-d[1], d[0]])
        t = (x - p0[0] - radius * n[0]) / d[0]
        on = (t >= 0.0) & (t <= math.dist(p0, p1))
        z = np.where(on, np.maximum(z, p0[1] + t * d[1] + radius * n[1]), z)
        for vx, vz in (p0, p1):
            near = np.abs(x - vx) <= radius
            z = np.where(near, np.maximum(z, vz + np.sqrt(np.clip(radius**2 - (x - vx) ** 2, 0.0, None))), z)
    return z


def profile_build(c: Any):
    cart = vl.cart(c)

    def build(b: newton.ModelBuilder) -> None:
        cart(b)
        for label, p0, p1 in SEGMENTS:
            segment_box(b, label, p0, p1)
    return build


def cruise(speed: float, gain: float = 3.0, feed: float = 0.1):
    """A speed controller: throttle (or brake) from the chassis's x speed."""
    def control(t: float, last: dict[str, np.ndarray]) -> dict[str, float]:
        u = feed + gain * (speed - float(last["qd"][0]))
        return {"throttle": float(np.clip(u, 0.0, 1.0)), "brake": float(np.clip(-u, 0.0, 1.0))}
    return control


# --- floors -----------------------------------------------------------------------------------------------------------

FLOOR_MASS, FLOOR_K, FLOOR_R = 2500.0, 1.0e5, 0.2
FLOOR_HALF = (1.5, 1.0, 0.05)
SUPPORTS = ((1.2, 0.8), (1.2, -0.8), (-1.2, 0.8), (-1.2, -0.8))


def floor_vehicle():
    """The floor's carrier: four braked, undriven-in-effect 'wheels' of stiffness FLOOR_K under the floor."""
    from alloy.plugins.vehicle import Vehicle  # noqa: PLC0415

    z = -FLOOR_HALF[2] - 0.05                       # centres R above the ground at the floor's rest height
    return Vehicle(label="floor_carrier", chassis="floor", ground_plane=True,
                   positions=tuple((x, y, z) for x, y in SUPPORTS), radius=(FLOOR_R,), steer=(0.0,),
                   driven=(True, True, True, True), motor_torque=0.0, radial_stiffness=(FLOOR_K,),
                   radial_damping=(2.0e4,), brake_torque=(1.0e4,), mu=1.0)


def floor_rest() -> float:
    """The floor's centre height with its supports undeflected."""
    return FLOOR_HALF[2] + FLOOR_R + 0.05


def sprung_floor_build(c: Any):
    """The floor on its carrier (free body), the cart parked at its centre, on top."""
    radius = float(np.atleast_1d(c.radius)[0])

    def build(b: newton.ModelBuilder) -> None:
        top = floor_rest() + FLOOR_HALF[2]
        floor = b.add_body(xform=body_xform(0.0, 0.0, floor_rest()), label="floor")
        b.add_shape_box(floor, hx=FLOOR_HALF[0], hy=FLOOR_HALF[1], hz=FLOOR_HALF[2], label="floor_top",
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(FLOOR_MASS / (8.0 * np.prod(FLOOR_HALF)))))
        body = b.add_body(xform=body_xform(0.0, 0.0, top + radius), label=c.chassis)
        b.add_shape_box(body, hx=vl.HALF[0], hy=vl.HALF[1], hz=vl.HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(vl.MASS / (8.0 * np.prod(vl.HALF)))))
    return build


def floor_rest_top() -> float:
    """The sprung floor's top face height, settled under its own weight."""
    return floor_rest() - FLOOR_MASS * 9.81 / (4.0 * FLOOR_K) + FLOOR_HALF[2]


SEAM_PARK = 0.0                                        # m: the seam between the static deck and the sprung floor
PARK_PAST = 0.005                                      # m: the front axle parks this far past the seam


def seam_floor_build(c: Any):
    """A static deck (x < 0) and the sprung floor (x > 0), coplanar at rest, the cart on the deck 2.5 m back."""
    radius = float(np.atleast_1d(c.radius)[0])

    def build(b: newton.ModelBuilder) -> None:
        top = floor_rest_top()
        segment_box(b, "deck", (-4.0, top), (SEAM_PARK, top))
        floor = b.add_body(xform=body_xform(SEAM_PARK + FLOOR_HALF[0], 0.0, top - FLOOR_HALF[2]), label="floor")
        b.add_shape_box(floor, hx=FLOOR_HALF[0], hy=FLOOR_HALF[1], hz=FLOOR_HALF[2], label="floor_top",
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(FLOOR_MASS / (8.0 * np.prod(FLOOR_HALF)))))
        body = b.add_body(xform=body_xform(-2.5, 0.0, top + radius), label=c.chassis)
        b.add_shape_box(body, hx=vl.HALF[0], hy=vl.HALF[1], hz=vl.HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(vl.MASS / (8.0 * np.prod(vl.HALF)))))
    return build


def park_at(front_x: float, speed: float, decel: float = 0.4, ahead: float = 0.6, hold: float = math.inf):
    """Drive the cart until its front axle is at ``front_x``, then hold it with the brakes (from ``hold`` [s] on in
    any case)."""
    def control(t: float, last: dict[str, np.ndarray]) -> dict[str, float]:
        rem = front_x - (float(last["q"][0]) + ahead)
        v = float(last["qd"][0])
        if rem < 0.003 or t >= hold:
            return {"throttle": 0.0, "brake": 1.0}
        ref = min(speed, math.sqrt(2.0 * decel * rem))
        u = 0.1 + 3.0 * (ref - v)
        return {"throttle": float(np.clip(u, 0.0, 1.0)), "brake": float(np.clip(-u, 0.0, 1.0))}
    return control


class Press(ForcePlugin):
    """A downward force on one body that ramps from 0 at ``start`` by ``rate`` [N/s] up to ``force``: it moves a
    sprung floor slowly under a parked wheel (the seam law)."""

    name = "vehicle_press"
    summary = "test support: a slowly ramped downward force on one body"

    @dataclasses.dataclass
    class Config:
        body: str = "floor"
        force: float = 2000.0                           # N
        rate: float = 800.0                             # N/s
        start: float = 0.0                              # s

    def setup(self, ctx: Any) -> None:
        self.index = ctx.bodies(self.config.body)

    def apply(self, state: Any, forces: Any, dt: float) -> None:
        c = self.config
        wp.launch(_press, dim=self.ctx.world_count, inputs=[self.ctx.time, self.index, float(c.force), float(c.rate),
                                                            float(c.start)],
                  outputs=[forces.body_f], device=self.ctx.device)


@wp.kernel
def _press(time: wp.array[wp.float64], body: wp.array[wp.int32], force: float, rate: float, start: float,
           body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    f = wp.min(wp.max(wp.float32(time[w]) - start, 0.0) * rate, force)
    add_wrench(body_f, body[w], wp.spatial_vector(0.0, 0.0, -f, 0.0, 0.0, 0.0))


SLIDE_MASS, SLIDE_HALF = 2000.0, (5.0, 1.0, 0.05)


def sliding_floor_build(c: Any):
    """A 2 t floor on a frictionless prismatic x joint (held in z), the cart at rest on it."""
    radius = float(np.atleast_1d(c.radius)[0])

    def build(b: newton.ModelBuilder) -> None:
        floor = b.add_link(xform=body_xform(0.0, 0.0, 0.0), label="floor")
        b.add_shape_box(floor, hx=SLIDE_HALF[0], hy=SLIDE_HALF[1], hz=SLIDE_HALF[2], label="floor_top",
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(SLIDE_MASS / (8.0 * np.prod(SLIDE_HALF)))))
        slide = b.add_joint_prismatic(-1, floor, axis=(1.0, 0.0, 0.0), label="floor_slide")
        b.add_articulation([slide], label="floor")
        body = b.add_body(xform=body_xform(-3.0, 0.0, SLIDE_HALF[2] + radius), label=c.chassis)
        b.add_shape_box(body, hx=vl.HALF[0], hy=vl.HALF[1], hz=vl.HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(vl.MASS / (8.0 * np.prod(vl.HALF)))))
    return build


# --- a ramp face ------------------------------------------------------------------------------------------------------

def ramp_truck_build(front_uphill: bool):
    """GA's truck parked on a static 13.6 % ramp box (its face through the origin), facing up or down it."""
    theta = math.atan(GRADE)

    def build(b: newton.ModelBuilder) -> None:
        segment_box(b, "ramp", (-6.0 * math.cos(theta), -6.0 * math.sin(theta)),
                    (6.0 * math.cos(theta), 6.0 * math.sin(theta)), half_width=2.0)
        yaw = 0.0 if front_uphill else math.pi
        rot = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -theta),
                     wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
        ax.build_truck(b, wp.transform(wp.vec3(0.0, 0.0, 0.0), rot))
    return build


def on_ramp(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**cfg, "surfaces": ("ramp",), "ground_plane": False}


__all__ = ["CART", "PARK_PAST", "Press", "SEAM_PARK", "SEGMENTS", "centre_path", "cruise", "floor_rest",
           "floor_rest_top", "floor_vehicle", "on_ramp", "park_at", "profile_build", "ramp_truck_build",
           "seam_floor_build", "segment_box", "sliding_floor_build", "sprung_floor_build"]
