# SPDX-License-Identifier: Apache-2.0
"""The syringe tasks' rig (SCR-WP8-11 Part A): a barrel in a holder that tilts, and its plunger moved by a hand.

Per syringe ``label``:
- ``<label>_tilt``: a revolute joint (world x) under the barrel, position-driven; 0 = the head (nozzle) up, π = down;
- ``<label>_plunger``: the plunger's prismatic joint (the ``syringe`` plug-in reads it); its armature is the needle's
  liquid column (``syringe.inertance``), which keeps the plug-in's explicit force stable (docs: syringe.py). The
  hand's grip on the flange is this joint's drive, to a position and a speed (velocity-led: ``GRIP_TAU``;
  force-limited at ``FINGER_EFFORT``), a grip with no play. A finger body pushing the flange through contact rang:
  the column's armature rattled in the grip's play and on its contact spring, swinging the barrel pressure ±50 %.

``Drive`` sets each joint's targets every control period (host side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

ROD = 0.075                              # m, the stopper face to the flange
FLANGE = (0.008, 0.008, 0.001)           # m, half sizes
PLUNGER_MASS = 2.0e-3                    # kg (GA: 3.4 g for the whole empty 3 mL syringe)
FINGER_KE = 3.0e4                        # N/m: the grip's position gain
FINGER_EFFORT = 40.0                     # N: its force limit (GA's fast bolus needs 16-20 N)
GRIP_TAU = 0.5                           # s: kd/ke, velocity-led: the drive tracks the hand's speed (implicit damping)
                                         # and corrects its position over this time, so a target that steps once per
                                         # control period still moves the plunger smoothly
TILT_KE, TILT_KD = 50.0, 5.0             # N·m/rad, N·m·s/rad: the holder's tilt drive


@dataclass(frozen=True)
class Rig:
    """One syringe's rig: ``x0`` the stopper's start travel from the head, ``armature`` the needle column's mass."""

    label: str
    stroke: float
    x0: float
    armature: float
    at: tuple[float, float, float] = (0.0, 0.0, 0.3)   # m, the head's position in the world (built head up;
                                                       # a script tilts it with ``Drive.set(tilt=...)``)

    def build(self, b: newton.ModelBuilder) -> None:
        flip = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)   # local z from the head toward the thumb
        base = wp.transform(wp.vec3(*self.at), flip)
        barrel = b.add_link(xform=base, label=f"{self.label}_barrel")
        b.add_shape_box(barrel, xform=_at(0.5 * self.stroke), hx=0.006, hy=0.006, hz=0.5 * self.stroke,
                        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, has_shape_collision=False))
        jt = b.add_joint_revolute(-1, barrel, parent_xform=base, axis=(1.0, 0.0, 0.0), target_ke=TILT_KE,
                                  target_kd=TILT_KD, armature=0.01, label=f"{self.label}_tilt")
        plunger = b.add_link(xform=wp.transform_multiply(base, _at(self.x0)), label=f"{self.label}_plunger_body")
        b.add_shape_box(plunger, xform=_at(ROD), hx=FLANGE[0], hy=FLANGE[1], hz=FLANGE[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=PLUNGER_MASS / (8.0 * math.prod(FLANGE)),
                                                            has_shape_collision=False))
        jp = b.add_joint_prismatic(barrel, plunger, parent_xform=_at(self.x0), axis=(0.0, 0.0, 1.0),
                                   limit_lower=-self.x0, limit_upper=self.stroke - self.x0, armature=self.armature,
                                   target_ke=FINGER_KE, target_kd=FINGER_KE * GRIP_TAU, effort_limit=FINGER_EFFORT,
                                   actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,   # kd tracks target_qd
                                   label=f"{self.label}_plunger")
        b.add_articulation([jt, jp], label=self.label)


def _at(z: float) -> wp.transform:
    return wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity())


def build(rigs: tuple[Rig, ...]):
    """``build(b)`` of a world holding every rig."""
    def fn(b: newton.ModelBuilder) -> None:
        for r in rigs:
            r.build(b)
    return fn


class Drive:
    """Host-side targets of each rig's grip (the plunger's travel change from its start, + toward the thumb; with its
    speed, which the velocity-led drive tracks) and tilt [rad]."""

    def __init__(self, sim: Any, labels: tuple[str, ...]) -> None:
        self.sim = sim
        m = sim.model
        qs, qds = m.joint_q_start.numpy(), m.joint_qd_start.numpy()
        self.q, self.qd = {}, {}
        for lab in labels:
            for j in ("plunger", "tilt"):
                ks = [k for k, n in enumerate(m.joint_label) if n.endswith(f"{lab}_{j}")]
                self.q[(lab, j)], self.qd[(lab, j)] = qs[ks], qds[ks]
        self.targets = {k: 0.0 for k in self.q}
        self.speeds = {k: 0.0 for k in self.q}

    def set(self, label: str, finger: float | None = None, tilt: float | None = None, speed: float = 0.0) -> None:
        for j, v in (("plunger", finger), ("tilt", tilt)):
            if v is not None:
                self.targets[(label, j)] = v
        self.speeds[(label, "plunger")] = speed
        t, td = self.sim.control.joint_target_q.numpy(), self.sim.control.joint_target_qd.numpy()
        for k, v in self.targets.items():
            t[self.q[k]] = v
            td[self.qd[k]] = self.speeds[k]
        self.sim.control.joint_target_q.assign(t.astype(np.float32))
        self.sim.control.joint_target_qd.assign(td.astype(np.float32))


__all__ = ["Drive", "FINGER_EFFORT", "PLUNGER_MASS", "Rig", "build"]
