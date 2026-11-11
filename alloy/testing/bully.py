# SPDX-License-Identifier: Apache-2.0
"""Bully test targets (WP7 helpers, D17): a seeded-instability FakeSolver target and a stable twin.

``seeded_fake_target``: one free body on the FakeSolver floor per world, pushed in x and y by applied forces (the
actions, ``body_f``). FakeSolver's seeded instability multiplies a world's velocities by ``blowup_gain`` every step
once its fastest body exceeds ``blowup_speed``, so a world blows up (SPEED_FAIL, then non-finite) only if the actions
push the body past that speed. With ``f_max = 5 N`` per axis on 2 kg the body reaches at most 5 m/s per axis
(7.1 m/s diagonally, below the 10 m/s impossible-motion limit) in the 2 s horizon, and crossing ``blowup_speed =
4 m/s`` needs force held in one direction for most of it: a random legal trace rarely does that, a search that follows
the speed ratio does.

``stable_fake_target``: no instability and half the force (``STABLE_F_MAX``), so the body also stays inside the 4 m
frame-warn radius (at most 2.5 m per axis in 2 s; DirectSim's health would flag FAR_FROM_ORIGIN, S3, past it) and the
twin is clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.bully.targets import register_target
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.testing.fakes import FakeSolver


@dataclass(frozen=True)
class SeededSpec:
    """Seeded FakeSolver target parameters (SI)."""

    blowup_speed: float | None = 4.0  # [m/s]; None = the stable twin
    blowup_gain: float = 1.5
    f_max: float = 5.0  # [N] per axis
    mass: float = 2.0  # [kg]
    body_radius: float = 0.1  # [m]
    control_dt: float = 0.02
    substeps: int = 4


def _make_sim(spec: SeededSpec, worlds: int, device: str) -> DirectSim:
    b = newton.ModelBuilder()
    b.add_body(xform=wp.transform((0.0, 0.0, spec.body_radius), wp.quat_identity()), mass=spec.mass)
    scene = newton.ModelBuilder()
    scene.replicate(b, copy_count=worlds)
    model = scene.finalize(device=device)
    cfg = FakeSolver.Config(blowup_speed=spec.blowup_speed, blowup_gain=spec.blowup_gain, floor_z=0.0,
                            body_radius=spec.body_radius)
    return DirectSim(model, FakeSolver(model, cfg), control_dt=spec.control_dt, substeps=spec.substeps, collide=False)


def seeded_fake_target(spec: SeededSpec | None = None, name: str = "fake_seeded") -> BullyTarget:
    """Target whose only defect is FakeSolver's seeded instability."""
    spec = spec or SeededSpec()
    rate = 1.0 / spec.control_dt
    f = spec.f_max
    actions = ActionSpace(low=np.array([-f, -f]), high=np.array([f, f]), rate_hz=rate,
                          max_delta_per_tick=np.array([f, f]) * 0.5, initial=np.zeros(2))
    return BullyTarget(name=name, scene=None, make_sim=lambda w, d: _make_sim(spec, w, d), actions=actions,
                       action_kind="body_f", action_index=(0, 1))


STABLE_F_MAX = 2.5  # [N] per axis: <= 2.5 m per axis in 2 s, inside FRAME_WARN_M = 4 m


def stable_fake_target() -> BullyTarget:
    """Same scene without the seeded instability, pushed with half the force (no finding should survive)."""
    return seeded_fake_target(SeededSpec(blowup_speed=None, f_max=STABLE_F_MAX), name="fake_stable")


register_target("fake_seeded", seeded_fake_target)
register_target("fake_stable", stable_fake_target)
