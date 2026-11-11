# SPDX-License-Identifier: Apache-2.0
"""A cube resting exactly on a static box's diagonal keeps its box-box contact (XB4-10's MuJoCo Warp 4.0 report).

MuJoCo Warp's box-box routine loses the contact when the cube's offset from the table centre has |dx| = |dy| in float32
(XB4-10, bench/xbench/xb4/mujoco4/gpu/NOTES.md item 4): on 4.0 the FR3 pack's cube falls through the table (96 of 100
steps without contact). Alloy's pin (86e2fae0) has a milder form. Raw, with the pack's values (2 ms, solref 0.02 1, no
gap), on CPU it loses the contact on 8-10 of 249 steps at (0.2, +-0.2) and comes back to MuJoCo C's height. At
(0.35, -0.35) it loses it on 65 of 249 and ends 1.06 mm lower ([internal note]).

Through alloy.Scene (native gains, 1.25 ms) the cube keeps 4 contacts on every period and rests 4 um in at every
offset, with Alloy's 5 mm contact gap or without it. Two guards for a MuJoCo Warp bump: the raw pack case must not let
the cube fall, and Alloy's route must hold it. The suite runs the CPU's Warp route; ALLOY_DEVICE=cuda:0 runs both on
the GPU."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import warp as wp

import alloy

OFFSETS = [(0.2, -0.2), (0.2, 0.2), (0.35, -0.35), (0.2, -0.15)]       # the last is off the diagonal


def _pack_xml(dx: float, dy: float) -> str:
    """XB4-10's box_on_table_min.py: the FR3 pack's option and geom values."""
    geom = 'friction="1 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"'
    return (f'<mujoco><option timestep="0.002" impratio="1" iterations="100" ls_iterations="50" '
            f'integrator="implicitfast" cone="pyramidal" jacobian="sparse" solver="Newton" tolerance="1e-8"/>'
            f'<worldbody><body pos="0.3 0 -0.025"><geom type="box" size="0.6 0.6 0.025" {geom}/></body>'
            f'<body pos="{0.3 + dx} {dy} 0.025"><freejoint/><geom type="box" size="0.025 0.025 0.025" mass="0.1" '
            f'{geom}/></body></worldbody></mujoco>')


@pytest.mark.parametrize("offset", OFFSETS)
def test_raw_pin_keeps_the_cube_on_the_table(device, offset):
    """The pinned MuJoCo Warp, alone, 0.5 s: the cube ends within 1.5 mm of MuJoCo C and loses the contact on fewer
    than 100 of 249 steps (pin, CPU: at most 65, 1.06 mm low; 4.0 on the GPU falls through)."""
    from alloy._vendor import mujoco_warp as mjw

    m = mujoco.MjModel.from_xml_string(_pack_xml(*offset))
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    with wp.ScopedDevice(device):
        mm, dd = mjw.put_model(m), mjw.put_data(m, d0, nworld=1, nconmax=64, njmax=200)
        dc, lost = mujoco.MjData(m), 0
        for k in range(250):
            mjw.step(mm, dd)
            mujoco.mj_step(m, dc)
            lost += int(k >= 1 and int(dd.nacon.numpy()[0]) == 0)
        z = float(dd.xpos.numpy()[0, 2, 2])
    assert abs(z - float(dc.xpos[2, 2])) < 1.5e-3 and lost < 100, (z, float(dc.xpos[2, 2]), lost)


@pytest.mark.law
@pytest.mark.parametrize("offset", OFFSETS)
def test_cube_on_the_tables_diagonal_stays_on_it(device, offset):
    """Through alloy.Scene: a 5 cm, 0.1 kg cube at rest on a 1.2 x 1.2 x 0.05 m static box (centre x 0.3, as the
    pack) for 0.5 s stays within 20 um of the table top and never goes 50 um into it (every offset: -4 um)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_box("table", size=(1.2, 1.2, 0.05), pos=(0.3, 0.0, -0.025), static=True)
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.3 + offset[0], offset[1], 0.025), mass=0.1)
    sim = alloy.build(scene, device=device)
    bottom = sim.run(0.5, record=["cube"]).world(0)["cube.pos"][:, 2] - 0.025
    assert bottom.min() > -5e-5 and abs(bottom[-1]) < 2e-5, np.round(bottom, 6).tolist()
    assert sim.check()["ok"], sim.check()["problems"]
