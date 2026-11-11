# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, robot stage: the fixture WP15 ruled for rig (b) (11-07), disclosed, not in Gym-Anything's
scene: a passive turntable under the can (``Turntable``).

Why: one rim pinch can neither turn the can on the tile nor hold it in the air ([internal note]
``notes/job3/can_arm_reach.md``). Turning it on the tile takes the floor's friction torque, mu m g r_bot = 4.45 N·m,
17 N at the rim; that force, 0.69 m up, tips the can about its foot first (F h / (m g r_bot) = mu h / R = 1.18).
Lifted clear, its weight hangs 0.26 m off the pinch (11.5 N·m against about 4 N·m that the pads hold). On the table the
UR turns the can against the bearing and the inertia alone.

The table: a plate whose top is ``top`` over the tile, its radius the can's foot plus ``margin``, on a vertical
revolute joint to the floor with the bearing's dry friction and viscous damping stated below. The plate rides ``gap``
clear of the tile. The can rests on it by contact, never welded: the tip lifts it off and pivots on the foot's edge at
the table's top, ``top`` higher than on the tile. The plate is not a chart or a pool: water that lands on it would
leave the books, so the dump's stream must clear it (the tipped rim is far past the plate's edge).

Disclosed for v1 (WP15, 11-08):
- the turntable: plate top 10 mm over the tile, 1.5 kg, rubber top (mu 0.8), bearing 0.10 N·m dry and 0.05 N·m·s/rad
  viscous (estimates); its radius is the can's collider footprint plus 12 mm (254.7 mm: the floor's 8 radial boxes
  reach 242.7 mm, past the 224 mm foot);
- a 5 mm lip on the plate's outer 8 mm, 4 mm clear of those colliders, that bounds the can's walk to 4 mm while the UR
  turns it. Without it the can walked 31 mm off its axis in 12 strokes on the MuJoCo Warp route, one stroke at a tip
  ratio of 0.53; MuJoCo C kept it within 2 mm (a route difference left for after the launch). At first 2 mm clear
  (WP15, 11-07); opened to 4 mm (WP15, 11-09) when the can's creep of up to 2.5 mm on MuJoCo C rested it on the lip;
- the brake (``Turntable.brake``), a stand-in until DC5 fixes G2: it holds the plate whenever the UR is not turning
  the can. With the pool aboard as G2's mass (``carry="mass"``) the free plate and can gained yaw on their own
  (-29° in 36 s standing with 7.5 kg aboard, up to 8 rad/s with 9.5 kg, then the can was thrown off), and with
  ``carry="weight"`` they did not (0.07°). The plate is free while the UR turns it;
- v1 dumps with stage 2's pose hand: the UR's own dump failed with every fixture tried, a split chock for the tipped
  rim included ([internal note] ``notes/job3/can_arm_reach.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

TABLE = "turntable"
TABLE_JOINT = "turntable_bearing"


@dataclass(frozen=True)
class Turntable:
    """A lazy-susan: the plate and its bearing (SI)."""

    top: float = 0.010            # m, the plate's top over the tile (WP15: 10 mm or less)
    gap: float = 0.002            # m, the plate's underside over the tile (it rides on its bearing)
    margin: float = 0.012         # m, the plate's radius past the can's collider footprint (``footprint``): the
                                  # lip 4 mm clear of the colliders (WP15, 11-09: at 2 mm the can's creep of up to
                                  # 2.5 mm on MuJoCo C rested it on the lip, 0.31-0.53x the load in 6 of 15 bands)
    mass: float = 1.5             # kg, the plate (a 12 mm-class steel lazy-susan with a rubber top: an estimate)
    friction: float = 0.10        # N·m, the bearing's dry friction torque under the can (estimate: ball bearing)
    damping: float = 0.05         # N·m·s/rad, the bearing's viscous damping (estimate)
    mu: float = 0.8               # the plate's rubber top against the can's foot
    lip: float = 0.005            # m, a lip on the plate's edge that keeps the can centred (0: none). Without it
                                  # the can walked 31 mm off the axis in 12 strokes on the Warp route; with it 2 mm
    lip_wall: float = 0.008       # m, its thickness (inside the plate's radius: 4 mm clear of the footprint)

    def radius(self, can: Any) -> float:
        """The plate's radius under ``can``: its collider footprint plus ``margin``."""
        return footprint(can) + self.margin

    def inertia(self, can: Any) -> float:
        """The plate's moment of inertia about its axis [kg·m²]."""
        return 0.5 * self.mass * self.radius(can) ** 2

    def brake(self) -> Any:
        """The table's brake (a lock on the lazy-susan, disclosed; module doc): a pose hold about the bearing's axis
        (``rinse_hand.PoseHand`` on the plate: 20 N·m/rad, 3 N·m·s/rad, 20 N·m at most; no force, so the plate's
        weight stays on its bearing). Free, the table let the can yaw 25° and then spin 200° under the UR's tip (with
        G2's pool mass aboard: the same spurious yaw)."""
        from alloy.tasks.rinse_hand import PoseHand  # noqa: PLC0415

        return PoseHand(bodies=(TABLE,), kp=0.0, kd=0.0, kr=20.0, dr=3.0, f_max=0.0, t_max=20.0, label="table_brake")

    def build(self, b: Any, at: tuple[float, float], can: Any) -> int:
        """Add the plate under ``at`` (its axis), on its bearing to the world, as its own articulation; returns its
        body."""
        import warp as wp  # noqa: PLC0415

        from alloy._vendor import newton  # noqa: PLC0415

        r, half = self.radius(can), 0.5 * (self.top - self.gap)
        pivot = wp.transform(wp.vec3(float(at[0]), float(at[1]), self.gap + half), wp.quat_identity())
        rho = self.mass / (math.pi * r * r * 2.0 * half)
        body = b.add_link(xform=pivot, label=TABLE)
        b.add_shape_cylinder(body, radius=r, half_height=half, label=f"{TABLE}_plate",
                             cfg=newton.ModelBuilder.ShapeConfig(density=rho, mu=self.mu))
        if self.lip > 0.0:
            self._build_lip(b, body, r, half)
        j = b.add_joint_revolute(-1, body, parent_xform=pivot, child_xform=wp.transform_identity(),
                                 axis=(0.0, 0.0, 1.0), friction=self.friction, damping=self.damping, label=TABLE_JOINT)
        b.add_articulation([j], label=f"{TABLE}_articulation")
        return body


    def _build_lip(self, b: Any, body: int, r: float, half: float, n: int = 48) -> None:
        """The lip: ``n`` boxes round the plate's edge, ``lip`` over its top, ``lip_wall`` thick."""
        import warp as wp  # noqa: PLC0415

        from alloy._vendor import newton  # noqa: PLC0415

        rc, seg = r - 0.5 * self.lip_wall, 2.0 * math.pi * r / n
        for i in range(n):
            a = 2.0 * math.pi * i / n
            pos = wp.vec3(rc * math.cos(a), rc * math.sin(a), half + 0.5 * self.lip)
            b.add_shape_box(body, xform=wp.transform(pos, wp.quat(0.0, 0.0, math.sin(0.5 * a), math.cos(0.5 * a))),
                            hx=0.5 * self.lip_wall, hy=0.5 * seg, hz=0.5 * self.lip, label=f"{TABLE}_lip{i}",
                            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=self.mu))

def footprint(can: Any) -> float:
    """The radius of the can's collider footprint [m]: its floor is ``ring // 2`` radial boxes from the axis to the
    foot (``r_bot``), each 1.05 x its sector wide (``rinse_vessels.Vessel.build``), so their outer corners reach past
    the foot (242.7 mm against 224 on COARSE's can)."""
    nf = can.ring // 2
    r = float(can.radius(0.0)) + can.wall
    return math.hypot(r, 0.5 * 2.0 * math.pi * r / nf * 1.05)


@dataclass(frozen=True)
class TurnLog:
    """One turn's books (WP15's condition 4): the UR's tangential force at the rim [N] (the torque balance about the
    axis), the tip ratio F h / (m g r_bot), the can's slip on the table [m, at the foot], the axis's drift [m], the
    can's rise [m] and the rim's slip against the pads [m], each the worst over the turn; the angle turned [rad]; the
    lip's peak force on the can and the share of periods it touches (WP15: it centres the can, it carries nothing);
    the can's turn while it was not held before this turn [rad]; the water aboard [kg]. The tip ratio's weight is the
    can's and the water's aboard at the turn's start (the water's centroid sits on the upright can's axis)."""

    force: float
    tip_ratio: float
    slip: float
    drift: float
    rise: float
    pad_slip: float
    turned: float
    lip_force: float = float("nan")     # N, the lip's peak normal force on the can (NaN: no lip, or no Warp data)
    lip_touch: float = float("nan")     # the share of the turn's periods the can touches the lip
    idle: float = 0.0                   # rad, the can's turn since the UR's last turn ended, while it was not held
    aboard: float = 0.0                 # kg, the water aboard at the turn's start (its weight is in the tip ratio)
    braked: float = 0.0                 # the plate brake's largest grip over the turn's periods (0: released
                                        # throughout, so no torque on the plate: ``hand_kernel`` skips grip <= 0)
    offset: float = 0.0                 # m, the can's axis off the table's axis, the most over the turn


def rim_force(alpha_can: float, alpha_table: float, omega_table: float, i_can: float, i_table: float,
              table: Turntable, rim: float) -> float:
    """The tangential force at the rim [N] that turns can and table: (I_can a_can + I_table a_table + c w + f sign w)
    / R."""
    torque = i_can * alpha_can + i_table * alpha_table + table.damping * omega_table
    torque += table.friction * float(np.sign(omega_table))
    return abs(torque) / rim


def tip_ratio(force: float, height: float, mass: float, g: float, foot: float) -> float:
    """F h / (m g r_bot): 1 tips the can about its foot's edge."""
    return force * height / (mass * g * foot)


__all__ = ["TABLE", "TABLE_JOINT", "TurnLog", "Turntable", "footprint", "rim_force", "tip_ratio"]
