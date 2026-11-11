# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can's robot stage: the turntable's books (fast) and one quarter turn of the empty can by the UR10e on
the CPU (WP15's condition 5)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks.can_wash_table import Turntable, footprint, rim_force, tip_ratio
from alloy.tasks.can_wash_tip import COARSE, Empty, rotate, tip_pose
from alloy.tasks.can_wash_turn import Lathe, eased

TURN = math.radians(90.0)


def can_corners(only: str = "") -> np.ndarray:
    """``[N, 3]`` the corners of COARSE's can's box colliders in its body frame, as ``rinse_vessels.Vessel.build``
    makes them, of the shapes whose label holds ``only``."""
    import itertools  # noqa: PLC0415

    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.tasks.can_wash_scene import CAN  # noqa: PLC0415

    b = newton.ModelBuilder()
    COARSE.can.build(b, CAN, wp.transform_identity())
    corners = []
    for i, label in enumerate(b.shape_label):
        if only not in str(label):
            continue
        assert b.shape_type[i] == newton.GeoType.BOX, label
        p, q = np.asarray(b.shape_transform[i][:3], float), np.asarray(b.shape_transform[i][3:], float)
        h = np.asarray(b.shape_scale[i], float)
        corners += [p + rotate(q, np.array(s) * h) for s in itertools.product((-1, 1), repeat=3)]
    return np.asarray(corners)


def test_the_rim_force_balances_both_inertias_the_damping_and_the_bearing():
    """(I_can a_can + I_table a_table + c w + f sign w) / R, its size whichever way the can turns."""
    t = Turntable(friction=0.1, damping=0.05)
    f = rim_force(2.0, 2.0, 1.0, 0.2, 0.04, t, 0.25)
    assert f == pytest.approx((0.2 * 2.0 + 0.04 * 2.0 + 0.05 + 0.1) / 0.25, rel=1e-12, abs=0)
    assert rim_force(-2.0, -2.0, -1.0, 0.2, 0.04, t, 0.25) == pytest.approx(f, rel=1e-12, abs=0)


def test_the_tip_ratio_is_one_where_the_rim_force_tips_the_can():
    """F h = m g r_bot at 1: on the tile, mu m g r_bot / R at the rim gives mu h / R (1.18 for this can)."""
    can = COARSE.can
    f_tile = can.mu * can.mass * 9.81 * can.r_bot / can.r_top
    assert tip_ratio(f_tile, can.height, can.mass, 9.81, can.r_bot) == pytest.approx(can.mu * can.height / can.r_top)
    assert can.mu * can.height / can.r_top > 1.0


def test_the_table_is_low_and_just_wider_than_the_cans_footprint():
    """WP15's condition 1: its top 10 mm or less over the tile, its radius the can's collider footprint plus 12 mm
    (the floor's 8 radial boxes reach 242.7 mm, past the 224 mm foot; the lip 4 mm clear); the can stands on its
    top."""
    t, can = Turntable(), COARSE.can
    assert footprint(can) == pytest.approx(0.2427, rel=0, abs=1e-4)
    assert t.top <= 0.010 and 0.0 < t.gap < t.top
    assert t.radius(can) == pytest.approx(footprint(can) + 0.012, rel=0, abs=1e-12)
    task = dataclasses.replace(COARSE, table=t)
    assert task.stand == t.top and COARSE.stand == 0.0


def test_the_turn_eases_in_and_out_and_reaches_its_angle():
    """The angles step by at most omega dt, start and end slowly (cosine ramps) and end exactly at the total."""
    a = eased(TURN, 1.0, 0.02)
    steps = np.diff(np.r_[0.0, a])
    assert a[-1] == pytest.approx(TURN, rel=1e-12, abs=0) and (steps >= 0).all()
    assert steps.max() <= 1.02 * 1.0 * 0.02, steps.max()          # omega dt, scaled to land on the total
    assert steps[0] < 0.1 * steps.max() and steps[-1] < 0.1 * steps.max()


def test_the_pinch_turns_with_the_can_about_its_axis():
    """The tool point stays on the rim's circle at the pinch's height whatever the turn."""
    lathe = Lathe(dataclasses.replace(COARSE, table=Turntable()))
    c = lathe.centre()
    r0 = np.linalg.norm(lathe.grip_pose(0.0)[:2] - c[:2])
    for phi in (0.3, 1.2, -2.0):
        p = lathe.grip_pose(phi)
        assert np.linalg.norm(p[:2] - c[:2]) == pytest.approx(r0, rel=0, abs=1e-12)
        assert p[2] == pytest.approx(lathe.grip_pose(0.0)[2], rel=0, abs=1e-12)


@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
def test_the_ur_turns_the_empty_can_a_quarter_on_the_turntable():
    """The UR10e pinches the empty can's rim and turns it 90° at 1 rad/s on the turntable: the can turns its 90°
    within 2°, its centre rises under 1 mm, its axis drifts and it slips on the table under 5 mm (WP15: a few mm),
    the rim slips under 2 mm in the pads, and the rim force stays under half of what tips it. The suite's Warp CPU
    route, with the table's lip: 90.4°, 6.0 N (tip ratio 0.42), slip 2.0 mm, drift 2.2 mm, pads 0.4 mm, rise 0.2 mm."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.tasks.can_wash_arms import ArmsBay, Rigs  # noqa: PLC0415
    from alloy.tasks.can_wash_turn import LatheRun  # noqa: PLC0415
    from alloy.tasks.paint_arm import Plan, Tracker  # noqa: PLC0415

    task = dataclasses.replace(COARSE, table=Turntable())
    lathe, rigs, plan = Lathe(task), Rigs.make(), Plan()
    tracker = Tracker(rigs.ur, plan)
    down = lathe.approach()
    qs = tracker.track(down, tracker.start(down))
    turn_q = tracker.track(lathe.turn(TURN, 1.0), qs[-1][None, None])
    assert plan.misses == 0
    bay = ArmsBay(task, rigs, qs[0], np.asarray(rigs.fr3.specs[0].home, float))
    run = LatheRun(task, rigs.ur, build_simulation(bay.scene(lambda s: None, 1), [], "cpu"))
    run.grip(qs)
    log = run.turn(turn_q)
    print(log)
    assert abs(log.turned - TURN) < math.radians(2.0), log
    assert log.rise < 1e-3 and log.drift < 5e-3 and log.slip < 5e-3 and log.pad_slip < 2e-3, log
    assert log.tip_ratio < 0.5, log


def test_the_plate_is_sized_from_the_floor_boxes_the_vessel_builds():
    """WP15's condition: the footprint is measured on ``rinse_vessels.Vessel.build``'s floor boxes (their outer
    corners: 242.7 mm on COARSE's can, past its 224 mm foot); the plate is that + 12 mm and the lip's inner face 4 mm
    or more past it (WP15, 11-09), so a change to the floor boxes fails here instead of seating the can on the lip."""
    corners = can_corners("_floor")
    measured = float(np.linalg.norm(corners[:, :2], axis=1).max())
    t = Turntable()
    assert len(corners) == 8 * (COARSE.can.ring // 2)
    # the builder keeps shapes in float32: 1 um
    assert measured == pytest.approx(footprint(COARSE.can), rel=0, abs=1e-6) and measured > COARSE.can.r_bot + 0.015
    assert t.radius(COARSE.can) == pytest.approx(measured + 0.012, rel=0, abs=1e-6)
    assert t.lip > 0.0 and t.radius(COARSE.can) - t.lip_wall >= measured + 0.004 - 1e-6



def least_clearance(task, foot: float | None) -> tuple[float, float]:
    """The robot stage's lifted dump (raised ``lip + LIFT_CLEAR`` over the plate, ``tip_pose`` about ``foot``) at
    every tilt to stage 2's angle and every yaw 0-45° (the colliders repeat every 45°): the least height [m] of the
    can's collider corners over the lip's top (corners over its ring) and over the plate's top (inside the ring)."""
    from alloy.tasks.can_wash_robot import LIFT_CLEAR, quat_mul  # noqa: PLC0415
    from alloy.tasks.can_wash_run import can_z  # noqa: PLC0415

    t, corners, at = task.table, can_corners(), np.asarray(task.at, float)
    r_in, r_out = t.radius(task.can) - t.lip_wall, t.radius(task.can)
    origin = np.array([*at, can_z(task) + t.lip + LIFT_CLEAR])
    lip = plate = math.inf
    for yaw in np.radians(np.arange(0.0, 45.1, 3.0)):
        qy = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        for th in np.linspace(0.0, Empty().angle, 106):
            pos, q = tip_pose(task, origin, float(th), foot=foot)
            w = pos + rotate(quat_mul(q, qy), corners)
            rho = np.hypot(w[:, 0] - at[0], w[:, 1] - at[1])
            over_lip, inside = (rho >= r_in) & (rho <= r_out), rho < r_in
            if over_lip.any():
                lip = min(lip, float(w[over_lip, 2].min()) - (t.top + t.lip))
            if inside.any():
                plate = min(plate, float(w[inside, 2].min()) - t.top)
    return lip, plate


def test_the_lifted_dump_tips_the_can_clear_of_the_lip_and_the_plate():
    """WP15's condition 4 (11-09): raised 3 mm over the 5 mm lip and tipped about its colliders' footprint, the can's
    colliders stay 3 mm or more over the lip's top and over the plate through the whole tip (3.4 and 8.1 mm). Tipped
    about the foot's edge (``r_bot``, stage 2's pivot) the floor boxes' corners swing under the pivot into the plate:
    the dump that brought the can back with its foot on the lip."""
    task = dataclasses.replace(COARSE, table=Turntable())
    lip, plate = least_clearance(task, footprint(task.can))
    assert lip >= 0.003 and plate >= 0.003, (lip, plate)
    assert least_clearance(task, None)[1] < 0.0


def test_the_release_drives_the_jaws_open():
    """The 2F-85's servo target is kept within force/kp of the measured driver (``Robotiq2F85.ctrl``): at a zero cap
    an open command leaves it on the driver and the jaws hang limp (on MuJoCo Warp they stayed shut, 15 mm, between
    strokes, and came down on the rim's top). ``RELEASE_N`` leads the driver toward open from any closed position."""
    from alloy.clips.arms.gripper import Robotiq2F85  # noqa: PLC0415
    from alloy.tasks.can_wash_turn import OPEN, RELEASE_N  # noqa: PLC0415

    g = Robotiq2F85()
    for q in (0.2, 0.5, 0.7):                         # rad, the driver: 66, 33 and 10 mm open
        held = g.ctrl(np.array([OPEN]), np.array([0.0]), np.array([q]))[0]
        assert held == pytest.approx(q, abs=1e-12)
        assert g.ctrl(np.array([OPEN]), np.array([RELEASE_N]), np.array([q]))[0] < q - 0.02

def test_the_released_brake_puts_no_torque_on_the_plate():
    """The brake is ``rinse_hand``'s pose hold on the plate: released (grip 0) it adds no wrench, held (grip 1) it
    turns the plate back toward its angle, capped at its 20 N·m (WP15's condition: the UR does all the turning)."""
    import warp as wp  # noqa: PLC0415

    from alloy.tasks.rinse_hand import Gains, hand_kernel  # noqa: PLC0415

    c = Turntable().brake().config
    g = Gains()
    for name in ("kp", "kd", "kr", "dr", "f_max", "t_max"):
        setattr(g, name, float(getattr(c, name)))
    yaw = 0.5                                                  # the plate turned 0.5 rad off its held angle
    body_q = wp.array([wp.transform(wp.vec3(0.0), wp.quat(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))],
                      dtype=wp.transform, device="cpu")
    body_qd = wp.array([wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.spatial_vector, device="cpu")
    torque = {}
    for grip in (0.0, 1.0):
        body_f = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
        arr = lambda v, t: wp.array(v, dtype=t, device="cpu")  # noqa: E731
        vec0 = wp.zeros((1, 1), dtype=wp.vec3, device="cpu")
        wp.launch(hand_kernel, dim=1, device="cpu", inputs=[
            body_q, body_qd, wp.zeros(1, dtype=wp.vec3, device="cpu"), arr([1.5], wp.float32), arr([[0]], wp.int32),
            vec0, arr([[wp.quat_identity()]], wp.quat), vec0, arr([[grip]], wp.float32), wp.vec3(0.0, 0.0, -9.81), g,
        ], outputs=[wp.zeros((1, 1), dtype=wp.float32, device="cpu"), body_f])
        torque[grip] = np.asarray(body_f.numpy()[0], np.float64)
    assert np.all(torque[0.0] == 0.0)
    assert torque[1.0][5] == pytest.approx(-min(c.kr * yaw + c.dr * 1.0, c.t_max), rel=1e-5)
    assert np.all(torque[1.0][:3] == 0.0)                     # no force: the plate's weight stays on its bearing


def test_the_ur_runs_world_0s_commands_in_every_world():
    """``LatheRun.write`` puts the UR's joint targets and its gripper's command into each world's actuators."""
    from types import SimpleNamespace  # noqa: PLC0415

    from alloy.tasks.can_wash_turn import LatheRun  # noqa: PLC0415

    class Arr:
        def __init__(self, a: np.ndarray) -> None:
            self.a = np.asarray(a, np.float32)

        def numpy(self) -> np.ndarray:
            return self.a

        def assign(self, v: np.ndarray) -> None:
            self.a = np.asarray(v, np.float32)

    act = np.array([[[0, 1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12, 13]]])        # [W, A, J + 1]: two worlds
    ctrl = Arr(np.full(14, -9.0))
    run = LatheRun.__new__(LatheRun)
    run.bind = SimpleNamespace(act=act)
    run.rig = SimpleNamespace(grip_ctrl=lambda bind, width, force, jq: width * 10.0 + force)
    run.sim = SimpleNamespace(control=SimpleNamespace(mujoco=SimpleNamespace(ctrl=ctrl)),
                              state=SimpleNamespace(joint_q=Arr(np.zeros(4))))
    q = np.arange(6) * 0.1
    run.write(q, 0.05, 2.0)
    for w in range(2):
        assert ctrl.a[act[w, 0, :6]] == pytest.approx(q, abs=1e-7)
        assert ctrl.a[act[w, 0, 6]] == pytest.approx(2.5, abs=1e-6)
