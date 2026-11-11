# SPDX-License-Identifier: Apache-2.0
"""The FR3 holding the aerosol can (CPU): the can in the arm's MJCF, the tool point that puts the nozzle over an aim
pointing down, the nozzle's path against the stand-in's aim points, and a plan of two rows the arm can follow."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.tasks import paint_arm as pa
from alloy.tasks import paint_fr3_can as pc
from alloy.tasks.paint_spray_stencil import BOX, Can, SprayLayout, schedule


def test_the_can_is_in_the_hand() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_string(pc.can_xml())
    can = m.body("can")
    assert m.body(can.parentid[0]).name == "hand" and can.mass[0] == pytest.approx(pc.CAN_MASS)
    for g in range(m.ngeom):                                              # the can does not collide
        if m.geom_bodyid[g] == can.id:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def test_the_tool_point_puts_the_nozzle_over_the_aim_pointing_down() -> None:
    """IK to ``tcp_for`` and MuJoCo's forward kinematics of the arm on its base: the can's body (the nozzle) at the
    aim to 10 µm, the can upright to 1e-4 (its -z, the nozzle's axis, down). They agree to 1.3 µm; the press-sized
    1.5 mm of a misplaced body fails."""
    import mujoco

    lay, can = SprayLayout(), Can()
    aim = np.array([0.12, -0.12, pc.nozzle_z(lay, can)])
    q = pa.ik_starts(pc.rig(), pc.tcp_for(aim)[None])
    assert q
    from alloy.tasks.paint_spray_supply import box_shapes

    m, _, _ = pa.mounted_model(pc.can_xml(), (*pc.CAN_BASE, *pc.base_quat()), box_shapes(lay))
    d = mujoco.MjData(m)
    d.qpos[:7] = q[0][0, 0]
    mujoco.mj_kinematics(m, d)
    body = m.body("can").id
    assert np.allclose(d.xpos[body], aim, rtol=0.0, atol=1e-5)
    assert d.xmat[body].reshape(3, 3)[:, 2] == pytest.approx((0.0, 0.0, 1.0), abs=1e-4)


def test_the_nozzle_path_sprays_the_stand_ins_aim_points() -> None:
    """With the valve open the nozzle is at each of the kinematic can's aim points at its sub-period's middle, in
    order; it runs up and out with the valve shut, inside the box, at most at 0.4 m/s."""
    lay, can = SprayLayout(), Can()
    path = pc.nozzle_path(lay, can)
    pts, valve = np.array(path.points), np.array(path.valve)
    aims = np.array([a for aims in schedule(can) if aims for a in aims])
    mids = 0.5 * (pts[:-1] + pts[1:])
    assert valve.sum() == len(aims) and not valve[-1]
    assert np.abs(mids[valve[:-1], :2] - aims).max() < 1e-12
    assert np.allclose(pts[:, 2], pc.nozzle_z(lay, can))
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / pc.sub_dt(can)
    assert speed.max() < 0.4 and np.abs(speed[valve[:-1]] - can.speed).max() < 1e-9
    assert np.abs(pts[:, 0]).max() < 0.5 * BOX[0] - 0.1 and np.abs(pts[:, 1]).max() < 0.5 * BOX[1] - 0.05


@pytest.mark.compile_heavy
def test_two_rows_plan_inside_the_arms_limits() -> None:
    can = dataclasses.replace(Can(), rows=(-0.12, 0.12), coats=1)
    plan = pc.plan_can(SprayLayout(), can)
    assert plan.ok, plan.report()
    assert plan.clearance > 0.04 and plan.peak_speed < 2.0


@pytest.mark.compile_heavy
def test_the_driver_keeps_the_nozzle_on_its_plan() -> None:
    """Two rows' driver in its arm scene without the film, its first 160 sub-periods (out to the first row and along
    it at 0.19 m/s): the nozzle within 0.3 mm of its planned place (measured 0.18 mm; the next sub-period's targets
    put it 2.4 mm ahead, unled targets 20 mm behind)."""
    can = dataclasses.replace(Can(), rows=(-0.12, 0.12), coats=1)
    d = pc.Fr3Can(SprayLayout(), can)
    err = pa.dry_run(d, d.scene(lambda sim: None, pc.sub_dt(can)), 160, lambda k: d.steps[k].nozzle)
    assert np.isfinite(err).all() and err.max() < 3e-4
