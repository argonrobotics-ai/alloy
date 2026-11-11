# SPDX-License-Identifier: Apache-2.0
"""Fixtures for alloy.io tests: small hand-made ``IOModel``s with known answers.

Example::

    from alloy.testing.io import arm_model
    io = arm_model()          # base + two revolute links, joint frames off the body frames
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from alloy.io.geometry import compose, inverse
from alloy.io.model import IOBody, IOJoint, IOMaterial, IOModel, IOShape


def _rot(axis, angle: float) -> tuple[float, float, float, float]:
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    return (*(a * math.sin(angle / 2)), math.cos(angle / 2))


def _box_inertia(m: float, h) -> tuple:
    hx, hy, hz = h
    d = (m / 3.0 * (hy * hy + hz * hz), m / 3.0 * (hx * hx + hz * hz), m / 3.0 * (hx * hx + hy * hy))
    return ((d[0], 0.0, 0.0), (0.0, d[1], 0.0), (0.0, 0.0, d[2]))


def arm_model() -> IOModel:
    """A fixed base and two links on revolute joints. The joint frames are rotated and offset from the child
    body frames, and the second axis is not a coordinate axis, so writers must re-express frames correctly."""
    mat = IOMaterial("alu", 2700.0, 0.6, 0.5, 0.3, 0.001, 0.0)
    base_pose = (0.1, -0.2, 0.3, *_rot((0, 0, 1), 0.4))
    jp1 = (0.0, 0.0, 0.05, *_rot((1, 0, 0), 0.3))           # joint 1 frame in the base
    jc1 = (0.0, 0.0, -0.1, *_rot((0, 1, 0), -0.2))          # joint 1 frame in link 1
    jp2 = (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0)
    jc2 = (0.02, 0.0, -0.08, *_rot((1, 1, 0), 0.5))
    l1_pose = compose(compose(base_pose, jp1), inverse(jc1))
    l2_pose = compose(compose(l1_pose, jp2), inverse(jc2))
    shapes = {n: (IOShape(f"{n}_geom", "box", h, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), "alu"),)
              for n, h in (("base", (0.05, 0.05, 0.02)), ("link1", (0.02, 0.02, 0.1)), ("link2", (0.015, 0.015, 0.08)))}
    masses = {"base": 1.0, "link1": 0.4, "link2": 0.2}
    bodies = [IOBody(n, p, masses[n], (0.0, 0.0, 0.0), _box_inertia(masses[n], shapes[n][0].size), shapes[n],
                     group="arm") for n, p in (("base", base_pose), ("link1", l1_pose), ("link2", l2_pose))]
    joints = [IOJoint("base_fixed", "fixed", None, "base", base_pose, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
              IOJoint("j1", "revolute", "base", "link1", jp1, jc1, (0.0, 0.0, 1.0), -1.5, 1.5, 20.0, 3.0),
              IOJoint("j2", "revolute", "link1", "link2", jp2, jc2, tuple(np.array([0.0, 0.6, 0.8])), -2.0, 2.0,
                      10.0, 4.0, damping=0.1)]
    return IOModel("arm", bodies, [], joints, [], {"alu": mat}, {"arm": "robot"})


RAMP_ANGLE = math.radians(20.0)
RAMP_MU = 0.2
BLOCK_HALF = 0.02


def _card_q(v: float):
    from alloy.core.quantity import Q  # noqa: PLC0415

    return Q(float(v), "model", note="alloy.io test fixture")


def ramp_library(root: Path) -> tuple["object", str]:
    """A card library folder with a static ramp (20 deg, mu 0.2) and a 40 mm block resting near its top, and the
    scene name. The block slides with a = g (sin t - mu cos t) = 1.51 m/s^2."""
    from alloy.cards import types as ct  # noqa: PLC0415
    from alloy.cards.library import CardLibrary, default_root  # noqa: PLC0415
    from alloy.cards.loader import dump_card  # noqa: PLC0415
    from alloy.io.geometry import compose, quat_rotate  # noqa: PLC0415

    surf = lambda e: ct.Surface(_card_q(RAMP_MU), _card_q(RAMP_MU), _card_q(e))  # noqa: E731
    cards = {"materials": [ct.MaterialCard("ramp_wood", _card_q(700.0), surf(0.2)),
                           ct.MaterialCard("block_abs", _card_q(1050.0), surf(0.2))],
             "objects": [ct.ObjectCard("ramp", ct.Primitive("box", (0.3, 0.1, 0.01)), "ramp_wood", motion="static"),
                         ct.ObjectCard("block", ct.Primitive("box", (BLOCK_HALF,) * 3), "block_abs")]}
    tilt = (0.0, math.sin(-RAMP_ANGLE / 2), 0.0, math.cos(-RAMP_ANGLE / 2))   # surface rises towards -x
    ramp_pose = (0.0, 0.0, 0.2, *tilt)
    up, along = quat_rotate(tilt, (0, 0, 1)), quat_rotate(tilt, (1, 0, 0))
    offset = up * (0.01 + BLOCK_HALF + 1e-4) - along * 0.2
    block_pose = compose((*offset, 0, 0, 0, 1), ramp_pose)
    scene = ct.SceneCard(robots=(), objects=(ct.Placement("ramp", ramp_pose), ct.Placement("block", block_pose)),
                         worlds=1)
    for kind, items in cards.items():
        (root / kind).mkdir(parents=True, exist_ok=True)
        for c in items:
            (root / kind / f"{c.name}.yaml").write_text(dump_card(c))
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    (root / "scenes" / "ramp_slide.yaml").write_text(dump_card(scene))
    return CardLibrary([root, default_root()]), "ramp_slide"
