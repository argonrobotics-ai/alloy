# SPDX-License-Identifier: Apache-2.0
"""The three-bowl pot sink of Gym-Anything's surface-rinsing station, for the rinse_* proof tasks.

The bowls are open tanks of the lumped-fluid core (``alloy.plugins.fluid``, ``medium="tank"``): each has a basket
drain (an orifice to the building drain) opened by a lever waste on the front apron, and an overflow at its rim that
spills onto the floor. A swing faucet (a source edge) fills the wash bowl when opened. The lever wastes are rigid
bodies on revolute joints; ``rinse_levers.SinkPlumbing`` opens each drain by its lever's angle every exchange step.

Sources (Gym-Anything's RoboWorld ``surface_rinsing`` environment; file:line):
- bowls 18 x 24 x 14 in, 2 mm walls, 35 mm drain bore, filled to 10 in: ``params.py:170-175`` (POT_SINK);
  pool boxes inset by the wall at each end in x: ``layout.py:341-342``;
- drain law ``Q = Cd·a·f·sqrt(2·g·h)`` with Cd 0.6 and the overflow over the front rim: ``water/pools.py:213-220``,
  ``params.py:251`` (POOL Cd);
- lever waste: a 90° twist lever (axis front to back), range 0 to π/2, joint friction 0.30, armature 2e-3, a chrome knob
  (r 12 mm, 20 mm, 0.04 kg) and a bar (90 x 12 x 12 mm, 0.05 kg) centred 45 mm out: ``scene.py:399-414``;
  the opening is ``f = clip(q/(π/2), 0, 1)``, read every tick: ``water/system.py:514-519``;
- fills: wash 43 °C with 0.75 g/L surfactant, rinse 43 °C, sanitizer 24 °C with 100 ppm chlorine, drains closed:
  ``layout.py:433-443``; swing faucet 2.2 gpm: ``params.py:64-70``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton

from alloy.plugins.fluid import Edge, Node, tank_overflow

INCH = 0.0254
GPM = 3.785411784e-3 / 60.0           # m³/s per US gallon per minute
BOWLS = ("wash", "rinse", "sanitize")
BOWL = (18 * INCH, 24 * INCH, 14 * INCH)   # x (along the sink), y (front to back), depth [m]
WALL = 0.002                          # m, 14 gauge
FILL = 10 * INCH                      # m, the start depth of every bowl
DRAIN_D = 0.035                       # m, basket drain bore
DRAIN_CD = 0.6                        # discharge coefficient
SURFACTANT = 0.75                     # kg/m³ in the wash bowl (0.75 g/L)
FAUCET = 2.2 * GPM                    # m³/s, swing faucet with a laminar regulator
LEVER_RANGE = (0.0, 0.5 * math.pi)    # rad, closed to fully open
LEVER_FRICTION = 0.30                 # N·m (joint friction loss)
LEVER_ARMATURE = 2.0e-3               # kg·m²
KNOB = (0.012, 0.010, 0.04)           # radius, half height [m], mass [kg]; centred 10 mm out of the apron
BAR = ((0.045, 0.006, 0.006), (0.045, 0.0, 0.022), 0.05)   # half extents, centre (lever frame) [m], mass [kg]
# MuJoCo's dry friction is a soft constraint: under a steady torque below the friction loss a joint still creeps at
# about ((1 - d)/d)·(τ/I)·(timeconst/2). At MuJoCo's defaults (d 0.9, 0.02 s) an open lever slid shut under its own
# weight at 0.33°/s (0.013 N·m against 0.30 N·m: a real twist waste holds). A stiff friction constraint on the lever
# (d 0.99, the shortest stable time constant at the 3.125 ms exchange step) cuts that about 35-fold.
FRICTION_SOLREF = (0.00625, 1.0)
FRICTION_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
DP_LAM = 1.0                          # Pa: the drain orifice follows Torricelli down to a 0.1 mm head
# the lever frame: z (the joint axis) points out of the apron toward the user (-y), x along the sink, y up
LEVER_ROT = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))     # x, y, z, w: +90° about x


def drain_edge(bowl: str) -> str:
    return f"waste_{bowl}"


def overflow_edge(bowl: str) -> str:
    return f"{bowl}_overflow"


def lever_joint(bowl: str) -> str:
    return f"lever_{bowl}_joint"


@dataclass(frozen=True)
class PotSink:
    """The sink: ``fill`` depth of each bowl [m] (wash, rinse, sanitize); levers on the apron at ``apron_y``,
    ``rim_z - 0.055`` high, under each bowl's centre (``layout.py:28-44``: the wash bowl starts 24 in from the sink's
    left end, after the soiled drainboard)."""

    fill: tuple[float, float, float] = (FILL, FILL, FILL)
    x0: float = 2.0 + 24 * INCH       # m, the wash bowl's left wall
    apron_y: float = 0.0              # m, the front apron's face
    rim_z: float = 0.940              # m

    @property
    def area(self) -> float:
        """Free-surface area of one bowl's pool [m²]."""
        return (BOWL[0] - 2.0 * WALL) * BOWL[1]

    @property
    def capacity(self) -> float:
        """Volume to the rim of one bowl [m³]."""
        return self.area * BOWL[2]

    def bowl_x(self, k: int) -> float:
        return self.x0 + (k + 0.5) * BOWL[0]

    def nodes(self) -> tuple[Node, ...]:
        return tuple(Node(b, medium="tank", volume=self.area * d, area=self.area, capacity=self.capacity,
                          species=(SURFACTANT if b == "wash" else 0.0,))
                     for b, d in zip(BOWLS, self.fill, strict=True))

    def edges(self) -> tuple[Edge, ...]:
        drain = DRAIN_CD * math.pi * DRAIN_D ** 2 / 4.0
        out = [Edge(drain_edge(b), "orifice", b, cd_area=drain, density=998.0, dp_lam=DP_LAM) for b in BOWLS]
        out += [tank_overflow(n, name=overflow_edge(n.name)) for n in self.nodes()]
        out.append(Edge("faucet", "source", "wash", flow=FAUCET, species=(0.0,)))
        return tuple(out)

    def openings(self) -> tuple[tuple[str, float], ...]:
        """Start openings: drains closed (levers at 0), faucet off; overflows always open."""
        return (*((drain_edge(b), 0.0) for b in BOWLS), ("faucet", 0.0))

    def build_levers(self, b: newton.ModelBuilder) -> None:
        """Add the three lever wastes (bodies and revolute joints, one articulation each) to builder ``b``."""
        cfg = newton.ModelBuilder.ShapeConfig
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        stiff = {"mujoco:solreffriction": [wp.vec2(*FRICTION_SOLREF)], "mujoco:solimpfriction": [FRICTION_SOLIMP]}
        knob_rho = KNOB[2] / (math.pi * KNOB[0] ** 2 * 2.0 * KNOB[1])
        bar_rho = BAR[2] / (8.0 * BAR[0][0] * BAR[0][1] * BAR[0][2])
        for k, bowl in enumerate(BOWLS):
            pivot = wp.transform(wp.vec3(self.bowl_x(k), self.apron_y - 0.012, self.rim_z - 0.055), wp.quat(*LEVER_ROT))
            body = b.add_link(xform=pivot, label=f"lever_{bowl}")
            b.add_shape_cylinder(body, xform=wp.transform(wp.vec3(0.0, 0.0, KNOB[1]), wp.quat_identity()),
                                 radius=KNOB[0], half_height=KNOB[1], cfg=cfg(density=knob_rho), label=f"knob_{bowl}")
            b.add_shape_box(body, xform=wp.transform(wp.vec3(*BAR[1]), wp.quat_identity()), hx=BAR[0][0],
                            hy=BAR[0][1], hz=BAR[0][2], cfg=cfg(density=bar_rho), label=f"bar_{bowl}")
            j = b.add_joint_revolute(-1, body, parent_xform=pivot, child_xform=wp.transform_identity(),
                                     axis=(0.0, 0.0, 1.0), limit_lower=LEVER_RANGE[0], limit_upper=LEVER_RANGE[1],
                                     friction=LEVER_FRICTION, armature=LEVER_ARMATURE, label=lever_joint(bowl),
                                     custom_attributes=stiff)
            b.add_articulation([j], label=f"lever_{bowl}_articulation")


__all__ = ["BOWL", "BOWLS", "DRAIN_CD", "DRAIN_D", "FAUCET", "FILL", "GPM", "LEVER_RANGE", "PotSink", "drain_edge",
           "lever_joint", "overflow_edge"]
