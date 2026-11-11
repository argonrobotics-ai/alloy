# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's 53 ft dry-van trailer for the forklift tasks (SCR-WP8-12 F4): a free body on its tandem axle group and
landing gear, as GA's (``F/`` = gym-anything-robotics benchmarks/roboworld/environments/forklift/; F/dock.py:1-24,
:29-75, :205-380).

Trailer frame: x from the floor's rear edge (the rear sill) toward the nose, y left, z up, origin on the floor top.
- The body: 6,800 kg, centre of mass at 0.52 of the inside length and 0.9 m up (F/params.py:719, :922). Box shapes for
  the oak floor and the rear sill (both drivable), the side walls, the nose, the roof, the rear posts and header, and
  the rear impact (ICC) bar.
- The supports are two more vehicle plug-ins on the trailer body, always braked (``Parked``):
  - the tandem: one wheel per side at 2.6 m (:719), 4.11e5 N/m and 2e4 N·s/m per side (:722, :733), braked tyres at
    μ 0.70 (:739). Its trailing arms (``arm_slope`` 0.15, :725) move each axle 0.15 m rearward per metre of
    compression, so a truck at the sill walks a held trailer toward its nose (GA's dock walk);
  - the landing gear: two legs at 12.6 m, 0.9 of the track apart (:921), 4e6 N/m and 2e4 N·s/m each (:734, :918),
    steel sand shoes at μ 0.40 (:740), as locked wheels.
  - Presliding: each support's tread relaxation length makes its tread spring GA's 1.5e6 N/m (:741) at the empty
    trailer's static load.
- The supports' rest lengths stand the floor at the spotted height under its static load (``Spot``, ``rest``): the
  trailer's own weight plus extra loads (a deployed leveler's tip, a ramp's lip plates, pallets in it), as GA's
  ``TrailerSupports._rest`` (F/dock.py:302-354).
- The restraint hook (optional) is a static box on the nose side of the ICC bar: a one-sided stop by native contact.

Limits: the chocks are not modelled (the braked tandem and the shoes hold about 0.70·W_t + 0.40·W_g, some 40 kN, and a
laden truck braking at 0.38 g pushes about 19 kN); the restraint and the supports are stops and springs without GA's
capacities (60 kN chocks, 142 kN hook); a sand shoe slides at the magic formula's locked share of μ (65 %), not at μ.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.plugins.harness.scene import body_xform
from alloy.plugins.vehicle import Vehicle

IN, FT, G = 0.0254, 0.3048, 9.81
LABEL = "trailer"
MASS = 6800.0                                              # F/params.py:719
INSIDE_L, INSIDE_W, INSIDE_H = 52.5 * FT, 101 * IN, 110.25 * IN    # :619
WALL, SUBFRAME, OUTSIDE_W = 0.025, 0.22, 102 * IN          # :719-720
CG = np.array([0.52 * INSIDE_L, 0.0, 0.9])                 # :922
INERTIA_H = 2.6                                            # :922, the inertia box's height
TANDEM_X, GEAR_X, TRACK = 2.6, 12.6, 1.83                  # :719-720
GEAR_Y = 0.9 * TRACK / 2.0                                 # :921
K_TANDEM, C_TANDEM, ARM_SLOPE = 4.11e5, 2.0e4, 0.15        # per side: :722, :733, :725
K_GEAR, C_GEAR = 4.0e6, 2.0e4                              # per leg: :734, :918
MU_TANDEM, MU_SHOE, STICK_K = 0.70, 0.40, 1.5e6            # :739-741
R_TANDEM, R_SHOE = 0.5, 0.15                               # m: the tyres; a shoe as a small locked wheel
SLIP_B, SLIP_C = 10.0, 1.65                                # the vehicle plug-in's magic formula (tread slope B·C·D)
REAR_FRAME = 0.10                                          # posts behind the floor's rear edge (:870)
MU_STRUCTURE = 0.50                                        # :794
ICC = (np.array([0.25, 0.0, -0.55]), (0.04, 1.1, 0.05))    # F/dock.py:73: centre, half extents


@dataclasses.dataclass(frozen=True)
class Spot:
    """Where the trailer stands: its floor's rear edge at (x, y) and floor height ``bed`` [m] above the ground at
    ``ground``; ``yaw`` turns the nose from +x."""

    x: float = 0.0
    y: float = 0.0
    bed: float = 48 * IN
    ground: float = 0.0
    yaw: float = 0.0

    def xform(self) -> wp.transform:
        return wp.transform(wp.vec3(self.x, self.y, self.bed), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0),
                                                                                         self.yaw))


Loads = tuple[tuple[float, float], ...]                    # (force [N], x in the trailer frame [m]) beyond its weight


def support_loads(extra: Loads = ()) -> tuple[float, float]:
    """(tandem, landing gear) loads [N], each summed over its two sides, under the weight and ``extra``."""
    loads = ((MASS * G, float(CG[0])), *extra)
    gear = sum(f * (x - TANDEM_X) for f, x in loads) / (GEAR_X - TANDEM_X)
    return sum(f for f, _ in loads) - gear, gear


def compressions(extra: Loads = ()) -> tuple[float, float]:
    """(tandem, landing gear) spring compressions [m] under the weight and ``extra``."""
    tandem, gear = support_loads(extra)
    return 0.5 * tandem / K_TANDEM, 0.5 * gear / K_GEAR


def floor_drop(force: float, at: float, x: float) -> float:
    """How far the floor at ``x`` sinks [m] when ``force`` [N] more stands on it at ``at`` (a rigid body on the
    support springs)."""
    d_t, d_g = (b - a for a, b in zip(compressions(), compressions(((force, at),)), strict=True))
    return d_t + (d_g - d_t) * (x - TANDEM_X) / (GEAR_X - TANDEM_X)


def relaxation(mu: float, load: float) -> float:
    """The tread relaxation length [m] that makes a wheel's tread spring ``STICK_K`` at ``load`` [N]."""
    return mu * load * SLIP_B * SLIP_C / STICK_K


def tread_force(mu: float, load: float, sigma: float, travel: float) -> float:
    """A braked wheel's tread force [N] after its contact slid ``travel`` [m] one way from rest: Pacejka's transient
    tread fills as u = σ·(1 - exp(-travel/σ)), and the force is the magic formula's at u/σ."""
    u = sigma * (1.0 - math.exp(-travel / sigma))
    return mu * load * math.sin(SLIP_C * math.atan(SLIP_B * u / sigma))


def walk(force: float, at: float, rest: Loads = ()) -> float:
    """How far a held trailer's ground-level points move toward its nose [m] when ``force`` [N] more is pressed slowly
    onto it at ``at``. The trailing arms move the tandem's contacts back by ARM_SLOPE times its extra compression; the
    tandem's treads then push the body forward and the landing gear's hold it, at their loads after (solved by
    bisection). The floor's rear edge moves that less ``bed·pitch``, the body's nose-up turn about it."""
    tandem0, gear0 = support_loads(rest)
    tandem1, gear1 = support_loads((*rest, (force, at)))
    sig_t, sig_g = relaxation(MU_TANDEM, 0.5 * tandem0), relaxation(MU_SHOE, 0.5 * gear0)
    shift = ARM_SLOPE * 0.5 * (tandem1 - tandem0) / K_TANDEM
    lo, hi = 0.0, shift
    for _ in range(60):
        x = 0.5 * (lo + hi)
        push = tread_force(MU_TANDEM, tandem1, sig_t, shift - x)
        hold = tread_force(MU_SHOE, gear1, sig_g, x)
        lo, hi = (x, hi) if push > hold else (lo, x)
    return 0.5 * (lo + hi)


def pitch(force: float, at: float) -> float:
    """The body's nose-up turn [rad] when ``force`` [N] more stands on it at ``at``."""
    d_t, d_g = (b - a for a, b in zip(compressions(), compressions(((force, at),)), strict=True))
    return (d_t - d_g) / (GEAR_X - TANDEM_X)


class Parked(Vehicle):
    """A vehicle whose brakes are on from the start (the trailer's spring brakes and its sand shoes)."""

    def setup(self, ctx: Any) -> None:
        super().setup(ctx)
        self.brake.fill_(1.0)


def _support(label: str, x: float, y: float, radius: float, k: float, c: float, mu: float, load: float,
             spot: Spot, slope: float) -> Parked:
    """Two braked support wheels at (x, ±y), their centres placed so each spring holds ``load``/2 at the spot."""
    z = spot.ground + radius - 0.5 * load / k - spot.bed
    sigma = relaxation(mu, 0.5 * load)
    return Parked(label=label, chassis=LABEL, positions=((x, y, z), (x, -y, z)), radius=(radius,), steer=(0.0,),
                  driven=(True, True), motor_torque=0.0, radial_stiffness=(k,), radial_damping=(c,),
                  brake_torque=(1.0e6,), mu=mu, rolling_resistance=0.0, relaxation_x=sigma, relaxation_y=sigma,
                  arm_slope=(slope,), ground=spot.ground, ground_plane=True, surfaces=())


def supports(spot: Spot = Spot(), rest: Loads = (), mu_scale: float = 1.0) -> list[Parked]:
    """The tandem and landing gear for a trailer spotted at ``spot`` under ``rest`` (extra static loads).
    ``mu_scale`` scales both frictions (a frictionless-support law uses a tiny one)."""
    tandem, gear = support_loads(rest)
    return [_support("trailer_tandem", TANDEM_X, TRACK / 2.0, R_TANDEM, K_TANDEM, C_TANDEM, MU_TANDEM * mu_scale,
                     tandem, spot, ARM_SLOPE),
            _support("trailer_gear", GEAR_X, GEAR_Y, R_SHOE, K_GEAR, C_GEAR, MU_SHOE * mu_scale, gear, spot, 0.0)]


def _boxes() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], bool]]:
    """(label, centre, half extents, drivable) of the trailer's shapes, trailer frame."""
    L, W, H = INSIDE_L, INSIDE_W, INSIDE_H
    return [("trailer_floor", (L / 2, 0.0, -SUBFRAME / 2), (L / 2, W / 2, SUBFRAME / 2), True),
            ("trailer_sill", (-REAR_FRAME / 2, 0.0, -0.10), (REAR_FRAME / 2, OUTSIDE_W / 2, 0.10), True),
            ("trailer_wall_l", (L / 2, W / 2 + WALL / 2, H / 2), (L / 2, WALL / 2, H / 2), False),
            ("trailer_wall_r", (L / 2, -W / 2 - WALL / 2, H / 2), (L / 2, WALL / 2, H / 2), False),
            ("trailer_nose", (L + WALL / 2, 0.0, H / 2), (WALL / 2, OUTSIDE_W / 2, H / 2), False),
            ("trailer_roof", (L / 2, 0.0, H + 0.03), (L / 2, OUTSIDE_W / 2, 0.03), False),
            ("trailer_post_l", (-0.05, W / 2 + 0.04, H / 2), (0.05, 0.05, H / 2), False),
            ("trailer_post_r", (-0.05, -W / 2 - 0.04, H / 2), (0.05, 0.05, H / 2), False),
            ("trailer_header", (-0.05, 0.0, H + 0.08), (0.05, OUTSIDE_W / 2, 0.08), False),
            ("trailer_icc", tuple(ICC[0]), ICC[1], False)]


DRIVABLE = tuple(label for label, *_, drive in _boxes() if drive)


def add_trailer(b: newton.ModelBuilder, spot: Spot = Spot(), restraint: bool = False) -> int:
    """The trailer body at ``spot`` (mass properties set, shapes massless), and the restraint hook if asked."""
    inertia = np.diag([MASS * (OUTSIDE_W**2 + INERTIA_H**2) / 12.0, MASS * (INSIDE_L**2 + INERTIA_H**2) / 12.0,
                       MASS * (INSIDE_L**2 + OUTSIDE_W**2) / 12.0])
    body = b.add_link(xform=spot.xform(), mass=MASS, com=wp.vec3(*CG), inertia=wp.mat33(*inertia.ravel()),
                      lock_inertia=True, label=LABEL)
    b.add_articulation([b.add_joint_free(body, label=LABEL)], label=LABEL)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU_STRUCTURE)
    for label, centre, half, _ in _boxes():
        b.add_shape_box(body, xform=body_xform(*centre), hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=label)
    if restraint:
        _restraint(b, spot)
    return body


def _restraint(b: newton.ModelBuilder, spot: Spot) -> None:
    """A static hook on the nose side of the ICC bar, 1 mm clear: it stops the trailer creeping away from the dock."""
    centre, half = ICC
    local = np.array([centre[0] + half[0] + 0.001 + 0.03, 0.0, centre[2]])
    c, s = math.cos(spot.yaw), math.sin(spot.yaw)
    world = np.array([spot.x + c * local[0] - s * local[1], spot.y + s * local[0] + c * local[1], spot.bed + local[2]])
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(*world), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), spot.yaw)),
                    hx=0.03, hy=0.15, hz=0.08, label="restraint_hook")


__all__ = ["ARM_SLOPE", "DRIVABLE", "GEAR_X", "K_GEAR", "K_TANDEM", "LABEL", "MASS", "Parked", "Spot", "TANDEM_X",
           "add_trailer", "compressions", "floor_drop", "pitch", "relaxation", "support_loads", "supports",
           "tread_force", "walk"]
