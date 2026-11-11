# SPDX-License-Identifier: Apache-2.0
"""PUN-4 thread pull-through: a needle with a suture string runs up through a slab from below; at the swage every
track anchor hands over to the thread, which is then drawn through against friction per length (Chentanez's
fperlen·l): drag = thread_flen x track length (the slab thickness, stretched), and nothing gains energy.
The string hangs below the swage in tension (a standing string pushed into a hole buckles; design note §2.5)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import Block, Drive, Thread, build_block
from alloy.solvers.puncture.terms import FREE, NEEDLE, THREAD

pytestmark = pytest.mark.compile_heavy
TH = Thread(count=10, flen=50.0)
BL = Block(size=4e-3, thick=2e-3, material=TissueMaterial(fp=0.3, fcut=0.1, flen=50.0), fix_sides=True,
           needle_length=2.5e-3, thread=TH, from_below=True)
DT, SPEED = 2.5e-4, 2e-2


def test_pun4_thread_drawn_through_at_friction_per_length() -> None:
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(BL)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=20,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, DT, up_axis=True)
        travel = BL.gap + BL.thick + BL.needle_length + 1.5e-3
        drag = []
        for _ in range(round(travel / SPEED / DT)):
            d.step(lambda t: SPEED * t, BL.gap)
            drag.append(float(s.loads.thread.numpy()[0][2]))
        st = s.anchors.state.numpy()
        rep = s.ledger.report()
    assert (st == NEEDLE).sum() == 0 and (st == THREAD).sum() >= 4 and int(s.tracks.flags.numpy()[0]) == 0
    expect = TH.flen * BL.thick                                       # whole track on the thread
    tail = np.asarray(drag[-40:])
    assert np.all(tail < 0.0)                                           # the tissue holds the thread back
    assert abs(-tail.mean() / expect - 1.0) < 0.15, (tail.mean(), expect)
    assert rep["net_rise"][0] < 1e-5 and rep["step_gain_max"][0] < 1e-6
