# SPDX-License-Identifier: Apache-2.0
"""The FR3 holding the real roller (CPU): the tool in the arm's MJCF, the tool point that puts the cover where asked,
the handle's pitch over the strip, and a plan of one lane's script that the arm can follow."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import paint_arm as pa
from alloy.tasks import paint_fr3 as pf
from alloy.tasks import paint_roller as pr
from alloy.tasks.paint_scene import Band, WallLayout


def test_the_roller_is_in_the_hand() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_string(pf.roller_xml())
    frame, cover = m.body("roller_frame"), m.body("cover")
    assert m.body(frame.parentid[0]).name == "hand" and m.body(cover.parentid[0]).name == "roller_frame"
    axle = m.joint("axle")
    assert axle.type[0] == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(axle.axis, (1.0, 0.0, 0.0))
    assert cover.mass[0] == pytest.approx(pf.COVER_MASS) and frame.mass[0] == pytest.approx(pf.FRAME_MASS)
    assert np.allclose(cover.pos, (0.0, 0.0, pf.HANDLE))
    for g in range(m.ngeom):                                              # the tool does not collide
        if m.geom_bodyid[g] in (frame.id, cover.id):
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


@pytest.mark.parametrize(("z", "pitch"), [(0.256, -30.0), (0.756, 0.0), (1.256, 30.0)])
def test_the_tool_point_puts_the_cover_where_asked(z: float, pitch: float) -> None:
    """IK to ``tcp_for`` and MuJoCo's forward kinematics of the arm on its base: the cover's body at the asked axis
    centre to 10 µm, its axis along world x and the handle at the wall's pitch to 1e-4. They agree to 1.3 µm; a
    frame misplaced by the press (1.5 mm) fails. The plan's own cover kinematics (for its clearances) agree too."""
    import mujoco

    assert math.degrees(pf.wall_pitch(z)) == pytest.approx(pitch, abs=1e-9)
    rig = pf.rig()
    centre = np.array([0.0, -pr.R + pr.PRESS, z])
    q = pa.ik_starts(rig, pf.tcp_for(centre, pf.wall_pitch(z))[None])
    assert q
    m, _, _ = pa.mounted_model(pf.roller_xml(), (*pf.BASE, *pf.base_quat()), [])
    d = mujoco.MjData(m)
    d.qpos[:7] = q[0][0, 0]
    mujoco.mj_kinematics(m, d)
    cover, frame = m.body("cover").id, m.body("roller_frame").id
    assert np.allclose(d.xpos[cover], centre, rtol=0.0, atol=1e-5)
    assert d.xmat[cover].reshape(3, 3)[:, 0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-4)
    p = math.radians(pitch)
    assert d.xmat[frame].reshape(3, 3)[:, 2] == pytest.approx((0.0, math.cos(p), math.sin(p)), abs=1e-4)
    assert np.allclose(pf._cover_fk(rig)(q[0][0, 0]), d.xpos[cover], rtol=0.0, atol=1e-5)


@pytest.mark.compile_heavy
def test_one_lane_plans_inside_the_arms_limits() -> None:
    lay = WallLayout()
    moves = pr.load_moves(lay)[-1:] + pr.lane_moves(lay, pr.lanes(lay)[0], 1) + pr.load_moves(lay)[:1]
    x, y, z = pr.well_floor(lay)
    plan = pf.plan_roller(lay, list(pr.poses((x, y - 0.5 * pr.WELL[1] + pr.R, z + 0.30), moves)))
    assert plan.ok, plan.report()
    assert plan.clearance > 0.04 and plan.speed_share == pytest.approx(0.844, abs=0.01)    # the transfers' 0.85
    assert plan.within_fr3, plan.report()                   # every move eased: inside 10 rad/s², 5,000 rad/s³


def test_led_targets_hold_a_ramp_on_the_plan() -> None:
    """MuJoCo C, no gravity: joint 1 ramps at 0.5 rad/s for 0.5 s. Targets led by ``kv/kp`` keep it on the plan;
    the plan's own angles leave it 0.1 s behind (the cover 26.7 mm short at 0.25 m/s)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(pf.roller_xml())
    m.opt.gravity[:] = 0.0
    m.opt.timestep = pf.DT
    arm = [m.actuator(f"fr3_joint{i}").id for i in range(1, 8)]
    lead = pa.servo_lead(m.actuator_biasprm[arm])
    assert lead == pytest.approx(0.1)
    from alloy.clips.arms import FR3

    home = np.array(FR3.home)
    n, rate = 40, 0.5
    qs = np.repeat(home[None], n + 1, axis=0)
    qs[:, 0] += rate * pr.SUB * np.arange(n + 1)
    errors = []
    for targets in (pa.led_targets(qs, lead, pr.SUB), qs):
        d = mujoco.MjData(m)
        d.qpos[:7] = home
        mujoco.mj_forward(m, d)
        for k in range(n):
            d.ctrl[arm] = targets[k]
            mujoco.mj_step(m, d, int(round(pr.SUB / pf.DT)))
        errors.append(qs[n, 0] - d.qpos[0])
    assert abs(errors[0]) < 1e-3
    assert errors[1] == pytest.approx(rate * (0.1 + 0.5 * pr.SUB), rel=0.05)


def test_every_move_is_eased() -> None:
    """A lane with its approach and lift: each move keeps its line and its ends, starts from rest and stops at its end;
    place and velocity agree step to step; the cover's acceleration stays under 2 m/s² (a kinematic cover changes
    velocity in one sub-period); it spins as it rolls on the wall."""
    lay = WallLayout()
    moves = pr.lane_moves(lay, pr.lanes(lay)[0], 3)[1:]                  # onto the wall, up, down, up, off it
    start = np.asarray(pr.lane_moves(lay, pr.lanes(lay)[0], 3)[0].to)
    raw = list(pr.poses(start, moves))
    eased = pf.ease_corners(raw)
    assert len(eased) > len(raw)
    wall = [p for p in eased if p.roll == "wall"]
    z, raw_z = [p.pos[2] for p in wall], [p.pos[2] for p in raw if p.roll == "wall"]
    assert max(z) == pytest.approx(max(raw_z), abs=1e-9) and min(z) == pytest.approx(min(raw_z), abs=1e-9)
    ends = [np.asarray(m.to) for m in moves]
    pos = np.array([p.pos for p in eased] + [eased[-1].pos + eased[-1].vel * pr.SUB])
    for e in ends:                                                       # every move's end is reached
        assert np.linalg.norm(pos - e, axis=1).min() < 1e-9
    vel = np.array([p.vel for p in eased])
    assert np.allclose(pos[1:] - pos[:-1], vel * pr.SUB, rtol=0.0, atol=1e-12)
    assert np.linalg.norm(np.diff(vel, axis=0), axis=1).max() / pr.SUB < 2.0
    assert np.linalg.norm(vel[0]) < 0.01 and np.linalg.norm(vel[-1]) < 0.01      # from rest, to rest
    for p in wall:
        assert p.spin == pytest.approx(-float(p.vel @ np.cross((1.0, 0.0, 0.0), pr.CONTACT["wall"])) / pr.R)


def test_a_ramped_path_starts_and_stops_at_rest() -> None:
    """Along a smooth path (a quarter circle of 0.1 m as 200 chords): the places keep to it, from its first point to
    its last, at rest at both ends, the cruise at the asked speed, the speed's change at most the asked acceleration
    (1.875·V/T over a ramp T)."""
    th = np.linspace(0.0, 0.5 * np.pi, 201)
    path = np.stack([0.1 * np.cos(th), 0.1 * np.sin(th), np.zeros_like(th)], axis=1)
    dt, speed, accel = 0.0125, 0.25, 1.5
    pts = np.array(pa.ramped_path(path, speed, dt, accel) + [path[-1]])
    assert np.array_equal(pts[0], path[0]) and np.array_equal(pts[-1], path[-1])
    assert np.abs(np.linalg.norm(pts[:, :2], axis=1) - 0.1).max() < 1e-5                 # on the arc (chords)
    v = np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt
    assert v[0] < 0.01 and v[-1] < 0.01 and v.max() == pytest.approx(speed, rel=0.02)
    assert (np.abs(np.diff(v)) / dt).max() < 1.05 * accel


def test_the_cover_rolls_where_it_is() -> None:
    """The film's age follows the cover's place: the driver's rolling pose is the planned one moved to where the cover
    is (2 mm along and 0.5 mm off the plan: still pressing the wall), and on the wall only while it presses it."""
    from types import SimpleNamespace

    lay = WallLayout()
    d = object.__new__(pf.Fr3Roller)                                      # its plan is not needed here
    d.layout, d.body = lay, 0
    planned = pr.Pose(np.array([0.0, lay.face_y - pr.R + pr.PRESS, 0.6]), np.array([0.0, 0.0, 0.25]), 0.0, 8.0,
                      "wall")  # fmt: skip

    def at(pos: np.ndarray) -> SimpleNamespace:
        q = np.zeros((1, 7), np.float32)
        q[0, :3] = pos
        return SimpleNamespace(state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: q)))

    near = planned.pos + (0.0, -0.0005, 0.002)
    p = d.rolling(at(near), pf.Step(np.zeros(7), planned, 0.0))
    assert p.roll == "wall" and np.allclose(p.pos, near, rtol=0.0, atol=1e-6) and p.spin == planned.spin
    clear = planned.pos + (0.0, -pr.PRESS - 0.001, 0.0)                  # 1 mm clear of the wall
    assert d.rolling(at(clear), pf.Step(np.zeros(7), planned, 0.0)).roll is None
    assert d.rolling(at(near), pf.Step(np.zeros(7), None, float("nan"))) is None


def test_a_short_ramped_path_is_all_ramp() -> None:
    """40 mm at 0.25 m/s and 1.5 m/s² is shorter than 1.875·V²/a (78 mm): the two ramps take the whole path, and the
    peak acceleration is about 1.875·V²/L (2.9 m/s²), above the asked one, as the docstring says."""
    length, speed, accel, dt = 0.04, 0.25, 1.5, 0.0125
    pts = np.array(pa.ramped_path(np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]), speed, dt, accel) + [[length, 0, 0]])
    peak = (np.abs(np.diff(pts[:, 0], 2)) / dt ** 2).max()
    assert accel < peak <= 1.875 * speed ** 2 / length


def test_a_plan_is_the_fr3s_seven_joints() -> None:
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="7 joint angles"):
        pa.Plan(steps=[SimpleNamespace(q=np.zeros(6)), SimpleNamespace(q=np.ones(6))]).finish(pr.SUB)


def test_a_smooth_move_keeps_each_joint_under_its_peak() -> None:
    """A transfer's joint move at 0.85 of each joint's FR3 speed (joint 1 swings 1.5 rad, so its speed binds, not the
    acceleration): joint 1 peaks at that share (1.875·Δ/T), none above it, and the acceleration stays under
    8 rad/s²."""
    a = np.zeros(7)
    b = np.array([1.5, 0.3, 0.0, 0.8, 1.0, 0.4, 1.0])
    peak = pa.SPEED_SHARE * pa.FR3_SPEED
    qs = np.array([a] + pa.smooth(a, b, pr.SUB, math.inf, peak=peak))
    v = np.abs(np.diff(qs, axis=0)) / pr.SUB
    assert (v / peak).max() <= 1.0 + 1e-9 and (v / peak).max() > 0.97
    assert (np.abs(np.diff(qs, 2, axis=0)) / pr.SUB ** 2).max() < 0.8 * pa.FR3_ACCEL


def test_a_plan_books_its_peak_acceleration_and_jerk() -> None:
    """Joint 2 at 4 rad/s² for 40 periods, 80 rad/s² for 3, then coasting: the plan books those peaks from its angles'
    differences, and the steps past the FR3's 10 rad/s² and 5,000 rad/s³ (Franka's specifications)."""
    from types import SimpleNamespace

    dt, acc = pr.SUB, np.array([4.0] * 40 + [80.0] * 3 + [0.0] * 20)
    vel = np.concatenate([[0.0], np.cumsum(acc) * dt])
    q2 = np.concatenate([[0.0], np.cumsum(vel) * dt])
    qs = np.zeros((len(q2), 7))
    qs[:, 1] = q2
    plan = pa.Plan(steps=[SimpleNamespace(q=q) for q in qs]).finish(dt)
    assert plan.peak_accel == pytest.approx(80.0, rel=1e-9) and plan.over_accel == 3
    assert plan.peak_jerk == pytest.approx(80.0 / dt, rel=1e-9) and plan.over_jerk == 2
    assert plan.peak_speed == pytest.approx(vel.max(), rel=1e-9) and not plan.within_fr3
    assert plan.report()["steps_over_fr3_accel"] == 3


@pytest.mark.compile_heavy
def test_the_driver_keeps_the_cover_on_its_plan() -> None:
    """The small wall's driver in its arm scene without the film, its first 220 sub-periods: still over the tray, down
    into the well at 0.2 m/s, rolling in it. From 25 sub-periods after a change of velocity to 12 before one (the
    servo targets lead the plan by 8), the cover stays within 0.45 mm of its plan (measured 0.20 mm going down; the
    next sub-period's targets put it 2.5 mm ahead), and rolling at a steady rate, the axle turns at the plan's spin
    (to 0.1 of 3.25 rad/s)."""
    small = WallLayout(width=0.4, height=0.352, cell=0.008, strip=Band(-0.1, 0.1, 0.076, 0.276))
    d, spins, axle = pf.Fr3Roller(small), [], []
    assert d.plan.within_fr3, d.plan.report()           # the whole script, its last pull-back cut short by the reach
    assert d.plan.speed_share == pytest.approx(0.767, abs=0.01)

    def turning(sim, k: int) -> None:  # noqa: ANN001
        if not axle:
            m = sim.model
            j = next(i for i, lbl in enumerate(m.joint_label) if str(lbl).endswith("/axle"))
            axle.append(int(m.joint_qd_start.numpy()[j]))
        pose, last = d.steps[k].pose, d.steps[k - 1].pose
        if pose is not None and pose.roll is not None and last is not None and abs(pose.spin - last.spin) < 1e-9:
            spins.append(float(sim.state.joint_qd.numpy()[axle[0]]) - pose.spin)     # rolling at a steady rate

    n = 220
    err = pa.dry_run(d, d.scene(lambda sim: None), n, lambda k: None if d.steps[k].pose is None else
                     d.steps[k].pose.pos, turning)  # fmt: skip
    poses = [st.pose for st in d.steps[: n + 12]]
    steady = [k for k in range(25, n) if all(p is not None and np.linalg.norm(p.vel) > 0.0 and
                                              np.allclose(p.vel, poses[k].vel, rtol=0.0, atol=1e-9)
                                              for p in poses[k - 25: k + 12])]
    assert len(steady) > 40
    assert err[steady].max() < 4.5e-4
    assert len(spins) > 20 and np.abs(spins[2:]).max() < 0.1
