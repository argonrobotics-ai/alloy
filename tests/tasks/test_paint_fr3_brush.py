# SPDX-License-Identifier: Apache-2.0
"""The FR3 holding the round brush (CPU): the brush in the arm's MJCF, the tool point that puts the tip's point where
asked with the brush level, and a plan of the whole script (dip, hold, S, lift) the arm can follow."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import paint_arm as pa
from alloy.tasks import paint_fr3_brush as fb


def test_the_brush_is_in_the_hand() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_string(fb.brush_xml())
    brush = m.body("brush")
    assert m.body(brush.parentid[0]).name == "hand" and brush.mass[0] == pytest.approx(fb.BRUSH_MASS)
    for g in range(m.ngeom):                                              # the brush does not collide
        if m.geom_bodyid[g] == brush.id:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def test_the_tool_point_puts_the_tip_on_the_point_level() -> None:
    """IK to ``down_tcp(point, GRIP)`` and MuJoCo's forward kinematics of the arm on its base: the brush body (the
    tip's point) there to 10 µm, its z up to 1e-4 (the applicator's normal, its -z, down)."""
    import mujoco

    from alloy.tasks.paint_brush_stroke import BOARD_Z

    point = np.array([0.10, 0.03, BOARD_Z - 0.004])
    q = pa.ik_starts(fb.rig(), pa.down_tcp(point, fb.GRIP)[None])
    assert q
    m, _, _ = pa.mounted_model(fb.brush_xml(), (*fb.BRUSH_BASE, *fb.base_quat()), fb.board_boxes())
    d = mujoco.MjData(m)
    d.qpos[:7] = q[0][0, 0]
    mujoco.mj_kinematics(m, d)
    body = m.body("brush").id
    assert np.allclose(d.xpos[body], point, rtol=0.0, atol=1e-5)
    assert d.xmat[body].reshape(3, 3)[:, 2] == pytest.approx((0.0, 0.0, 1.0), abs=1e-4)


@pytest.mark.compile_heavy
def test_the_whole_script_plans_inside_the_arms_limits() -> None:
    from alloy.tasks.paint_brush_stroke import brush_script

    plan = fb.plan_brush(fb.ease_script(brush_script()))
    assert plan.ok, plan.report()
    assert plan.clearance > 0.05 and plan.peak_speed < 1.0
    assert sum(st.stroke for st in plan.steps) > 200
    assert plan.within_fr3, plan.report()                   # every move eased: inside 10 rad/s², 5,000 rad/s³


def test_the_eased_script_keeps_the_path_from_rest_to_rest() -> None:
    """Every move and the S keep their places' path: the S from the landing to its last point, each move from rest
    to rest; the steps' velocities are their places' differences, and the tip's acceleration stays under 4 m/s² (a
    kinematic tip changes velocity in one period: 2.5 m/s² down into the well, at the landing and the lift)."""
    from alloy.tasks import paint_brush as pb
    from alloy.tasks.paint_brush_stroke import brush_script

    raw = brush_script()
    eased = fb.ease_script(raw)
    stroke, raw_stroke = [s.pose.pos for s in eased if s.stroke], [s.pose.pos for s in raw if s.stroke]
    assert np.array_equal(stroke[-1], raw_stroke[-1]) and np.array_equal(eased[-1].pose.pos, raw[-1].pose.pos)
    pos = np.array([s.pose.pos for s in eased])
    assert np.allclose(np.diff(pos, axis=0), np.array([s.pose.vel for s in eased[1:]]) * pb.DT, rtol=0.0, atol=1e-12)
    k = [s.stroke for s in eased].index(True)
    assert np.linalg.norm(pos[k] - pos[k - 1]) < 1e-4 and np.linalg.norm(pos[k - 1] - pos[k - 2]) < 1e-4
    assert np.linalg.norm(np.diff(pos, 2, axis=0), axis=1).max() / pb.DT ** 2 < 4.0


@pytest.mark.compile_heavy
def test_the_driver_keeps_the_tip_on_its_plan() -> None:
    """The brush's driver in its arm scene without the film, its first 60 control periods (down toward the well at
    0.98 mm a period): the tip within 0.45 mm of its plan (measured 0.30 mm; unled targets leave it 5.6 mm behind)."""
    d = fb.Fr3Brush()
    err = pa.dry_run(d, d.scene(lambda sim: None), 60, lambda k: d.steps[k].pose.pos)
    assert np.isfinite(err).all() and err.max() < 4.5e-4
