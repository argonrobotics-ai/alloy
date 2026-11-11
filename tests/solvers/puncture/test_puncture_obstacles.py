# SPDX-License-Identifier: Apache-2.0
"""PUN-10 thread on a rigid obstacle (obstacles.py): a box tilted 25 deg rises under a hanging suture string and
lifts its end (a straight string pushed exactly end-on cannot pick a side to buckle to; a tilted face gives it one).
The string's particles never sink into the box by more than 0.03 mm (penalty k = 1e5 N/m), and the lowest one rides
up on the box's top face."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import BOX, Obstacle, ObstacleConfig, PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import Block, Thread, build_block

pytestmark = pytest.mark.compile_heavy
TH = Thread(count=12, spacing=0.5e-3, diameter=0.28e-3, EI=1.97e-8, EA=227.0)
BL = Block(size=4e-3, thick=2e-3, material=TissueMaterial(fp=0.3, fcut=0.1, flen=50.0), fix_sides=True,
           needle_length=2.5e-3, thread=TH, from_below=True)
DT, HALF, RISE, SPEED, TILT = 2.5e-4, (2e-3, 2e-3, 1e-3), 2.0e-3, 2.0e-2, np.radians(25.0)
Q = (0.0, float(np.sin(0.5 * TILT)), 0.0, float(np.cos(0.5 * TILT)))      # about y


def _box_sdf(p: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    d = p - c
    cs, sn = np.cos(TILT), np.sin(TILT)
    loc = np.stack([cs * d[:, 0] - sn * d[:, 2], d[:, 1], sn * d[:, 0] + cs * d[:, 2]], 1)   # into the box frame
    q = np.abs(loc) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


def test_pun10_thread_rests_on_a_rising_box() -> None:
    swage_z = -BL.gap - BL.needle_length
    end_z = swage_z - (TH.count - 1) * TH.spacing
    z0 = end_z - 0.5e-3 - HALF[2] / np.cos(TILT)     # box top 0.5 mm below the string's end (at its centre)
    body = []
    with wp.ScopedDevice("cpu"):
        def extra(b):
            body.append(b.add_body(xform=wp.transform(wp.vec3(BL.x_off, 0.0, z0), wp.quat(*Q)),
                                   mass=0.0, is_kinematic=True))
        m, specs = build_block(BL, extra=extra)
        ob = ObstacleConfig(items=(Obstacle(BOX, world=0, body=body[0], size=HALF),))
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, obstacles=ob, iterations=20,
                           integrate_with_external_rigid_solver=True)
        s0, s1 = m.state(), m.state()
        X = s0.body_q.numpy().copy()
        r = float(m.particle_radius.numpy()[specs[0].thread_start])
        rise_steps, hold = round(RISE / SPEED / DT), 200
        worst, lowest0 = np.inf, None
        for k in range(rise_steps + hold):
            for st, kk in ((s0, k), (s1, k + 1)):
                q = X.copy()
                q[body[0], 2] = z0 + SPEED * DT * min(kk, rise_steps)
                st.body_q.assign(q)
            s.step(s0, s1, None, None, DT)
            s0, s1 = s1, s0
            x = s0.particle_q.numpy()[specs[0].thread_start: specs[0].thread_start + TH.count]
            c = s0.body_q.numpy()[body[0], :3]
            worst = min(worst, float((_box_sdf(x, c, np.asarray(HALF)) - r).min()))
            if lowest0 is None:
                lowest0 = float(x[:, 2].min())
        d_end = float(_box_sdf(x[-1:], c, np.asarray(HALF))[0]) - r
    assert worst > -3e-5, worst                               # never sinks more than 0.03 mm
    assert float(x[:, 2].min()) > lowest0 + 0.5e-3            # the end rode up (and slid down the slope)
    assert abs(d_end) < 5e-5, d_end                           # and rests on its face
