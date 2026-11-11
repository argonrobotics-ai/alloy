# SPDX-License-Identifier: Apache-2.0
"""Law: FAR_FROM_ORIGIN and DET_DOWNGRADED are notices, listed but leaving a world OK (TRAIN-FIX 4c, physics
2026-11-11; audit TA-3 F4, TA-1 F7).

Before: FAR_FROM_ORIGIN made a world DEGRADED past 4 m, which a walking robot passes in seconds (760-1,400 of 4,096 Go2
worlds DEGRADED from iteration 25, 256 / 256 in a 20 s eval at 0.5 m/s; most rough-terrain tiles lie past 4 m), so
DEGRADED carried no signal for locomotion or any mobile robot. DET_DOWNGRADED, on every GPU rigid world, was made a
notice on 2026-11-10 (tests/precision/test_health_notice_law.py).

Laws:
- a world far from the origin with no other flag lists FAR_FROM_ORIGIN, reads OK, is in no degraded list and passes
  sim.check(); a world near the origin does not list it (CPU);
- a real PEN_WARN in a far world still reads DEGRADED (CPU, the monitor);
- on a GPU, a box at rest, one near and one far, lists DET_DOWNGRADED (and FAR_FROM_ORIGIN far away) and reads OK.
Past r_fail (16 m) under the precision preset a far world is still INVALID: tests/precision/test_time_origin.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.testing.precision import HealthLoop, set_body, world_bodies
from alloy.testing.scenes import box_on_plane

FAR_X = 6.0                                   # past r_warn (4 m), short of r_fail (16 m)
STARTS = np.array([[0.0, 0.0, 0.05], [FAR_X, 0.0, 0.05]])


def _boxes(device: str):
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=STARTS, mass=1.0)
    sim = alloy.build(scene, device=device)
    sim.run(0.3)
    return sim


@pytest.mark.law
def test_a_far_world_with_no_other_flag_reads_ok() -> None:
    sim = _boxes("cpu")
    h, c = sim.health(), sim.check()
    near, far = h["worlds"]
    assert "FAR_FROM_ORIGIN" in far["flags"], far                     # listed: the user still sees it
    assert far["severity"] == "OK" and "problem" not in far, far      # but it is not a problem
    assert "FAR_FROM_ORIGIN" not in near["flags"] and near["severity"] == "OK", near
    assert h["degraded_worlds"] == [] and h["invalid_worlds"] == []
    assert c["ok"] is True and not any("FAR_FROM_ORIGIN" in str(p) for p in c["problems"]), c["problems"]


@pytest.mark.law
def test_a_real_pen_warn_in_a_far_world_still_reads_degraded() -> None:
    loop = HealthLoop(box_on_plane(2, gap=-0.004, device="cpu"), checks=CheckPolicy(level="full", full_every=1))
    b = int(world_bodies(loop.scene.model, 1)[0])                       # both boxes 4 mm into the ground: PEN_WARN
    q = loop.state.body_q.numpy()[b].copy()
    q[0] = FAR_X                                                        # world 1's box far away
    set_body(loop.state.body_q, b, q)
    loop.step(1)
    rep = loop.monitor.poll()
    pen, far = int(HealthFlags.PEN_WARN), int(HealthFlags.FAR_FROM_ORIGIN)
    assert rep.flags[1] & pen and rep.flags[1] & far, hex(int(rep.flags[1]))
    assert rep.severity[1] == Severity.DEGRADED                         # the notice never hides a real flag
    assert rep.flags[0] & pen and not rep.flags[0] & far and rep.severity[0] == Severity.DEGRADED


@pytest.mark.gpu
@pytest.mark.law
def test_on_a_gpu_a_box_at_rest_near_and_far_reads_ok() -> None:
    sim = _boxes("cuda:0")
    h = sim.health()
    near, far = h["worlds"]
    assert "DET_DOWNGRADED" in near["flags"] and near["severity"] == "OK", near
    assert {"DET_DOWNGRADED", "FAR_FROM_ORIGIN"} <= set(far["flags"]) and far["severity"] == "OK", far
    assert h["degraded_worlds"] == [] and sim.check()["ok"] is True
