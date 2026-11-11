# SPDX-License-Identifier: Apache-2.0
"""GA's forklift mast (SCR-WP8-12 F2) against the closed forms in alloy.tasks.forklift_mast: axle loads (M-1), the
tip-over load (M-2), the two-stage lift (M-3) and the tilt (M-4), on the H2.5A truck at the task step (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alloy.plugins import fluid
from alloy.plugins.vehicle import laws_grip as lg
from alloy.tasks import forklift_mast as fm
from alloy.tasks import forklift_truck as ft

pytestmark = pytest.mark.law
SLOW = pytest.mark.timeout(240)                      # 20-40 s alone on the CPU; a loaded gate host (-n 8) runs ~2x
FWD = math.radians(6.0)
K = lg.TRUCK["radial_stiffness"][0]
RAISED = ft.Pose(tilt=FWD, stage=1.12, free=0.96)                           # forks 3.2 m up, 6° forward


def _at_height(tilt: float, height: float) -> ft.Pose:
    """The mast at ``tilt`` with the forks ``height`` above their lowest: free lift first, then the stages."""
    free = min(height, ft.FREE)
    return ft.Pose(tilt=tilt, stage=0.5 * (height - free), free=free)


GRID = [(_at_height(t, h), m) for t in (-FWD, 0.0, FWD) for h in (0.0, 1.6, 3.2) for m in (0.0, 1000.0, 2000.0)]


def _truck(pose: ft.Pose, load: ft.Load | None, locked: bool = False):
    """The truck built at its closed-form rest on the tyres, so the laws start without a drop."""
    return ft.build(pose, load, locked=locked, at=fm.rest_at(pose, load, K))


def _parked(pose: ft.Pose, load: ft.Load | None, seconds: float, **kw):
    return lg.run(lg.TRUCK, seconds, {"brake": 1.0}, build=_truck(pose, load, locked=True), **kw)


@pytest.mark.parametrize("pose, mass", GRID, ids=lambda v: f"{v}" if isinstance(v, float) else
                         f"t{math.degrees(v.tilt):+.0f}h{v.free + 2.0 * v.stage:.1f}")
def test_m1_axle_loads_match_the_statics(pose, mass):
    load = ft.Load(mass) if mass > 0.0 else None
    r = _parked(pose, load, 1.0)
    fz = r["force"][r["t"] >= 0.5, :, 2].mean(axis=0)
    front, rear = fm.axle_loads(pose, load, K)
    assert fz[:2].sum() == pytest.approx(front, rel=0.02), (fz, front, rear)
    assert fz[2:].sum() == pytest.approx(rear, rel=0.02), (fz, front, rear)


def _drive_then_brake(go: float, brake: float, ramp: float):
    """Brakes held until ``go``, full throttle until ``brake``, then throttle off and brakes on over ``ramp`` seconds
    (a step would ring the truck's pitch on its tyres)."""
    def control(t: float, last: dict) -> dict[str, float]:
        s = min(max((t - brake) / ramp, 0.0), 1.0)
        return {"throttle": 0.0, "brake": 1.0} if t < go else {"throttle": 1.0 - s, "brake": s}
    return control


def test_m1_braking_moves_load_to_the_front_axle_by_m_a_h_over_l():
    load = ft.Load(1000.0)
    go, brake, ramp = 0.6, 2.0, 0.4
    r = lg.run(lg.TRUCK, 4.0, {"brake": 1.0}, build=_truck(ft.Pose(), load, locked=True),
               control=_drive_then_brake(go, brake, ramp))
    mass, c = fm.centre(ft.Pose(), load)
    height = float(c[2]) - 0.5 * fm.axle_loads(ft.Pose(), load, K)[0] / K   # the front sinks with the truck's frame
    t, front = r["t"], r["force"][:, :2, 2].sum(axis=1)
    rest = front[(t > 0.3) & (t <= go)].mean()
    moving = r["qd"][:, 0] > 0.3                                              # world x speed of the chassis
    window = (t > brake + ramp + 0.2) & moving
    decel = -r["force"][window, :, 0].sum(axis=1).mean() / mass
    assert window.sum() >= 20 and decel > 2.0, (window.sum(), decel)
    assert front[window].mean() - rest == pytest.approx(mass * decel * height / fm.WHEELBASE, rel=0.03)


@pytest.mark.parametrize("factor, tips", [(0.94, False), (0.98, False), (1.02, True), (1.06, True)])
def test_m2_the_truck_tips_forward_past_the_closed_form_load(factor, tips):
    load = ft.Load(factor * fm.tip_load(RAISED, ft.Load(1000.0), K))
    rest = Rotation.from_quat(np.asarray(fm.rest_at(RAISED, load, K).q)).as_euler("ZYX")[1]
    r = _parked(RAISED, load, 2.5)
    rear = r["force"][:, 2:, 2].sum(axis=1)
    lifted = float((rear <= 1.0).sum() * lg.PERIOD)
    pitch = Rotation.from_quat(r["q"][:, 3:]).as_euler("ZYX")[:, 1]
    if tips:
        assert lifted > 0.3, lifted                                          # GA's no_wheel_lift would fail
    else:
        assert rear.min() > 0.0 and abs(float(pitch[-1]) - rest) < math.radians(0.2), (rear.min(), pitch[-1], rest)


def _ramp(t: float, start: float, seconds: float) -> float:
    """A valve opening over ``seconds`` from ``start``, as a proportional valve does (a step rings the oil column)."""
    return min(max((t - start) / seconds, 0.0), 1.0)


def _in_outer(r: dict, body: str) -> np.ndarray:
    """``body``'s origin in the outer upright's frame, from where it started [K, 3]."""
    outer = r["mast_outer.q"]
    rel = Rotation.from_quat(outer[:, 3:]).inv().apply(r[f"{body}.q"][:, :3] - outer[:, :3])
    return rel - rel[0]


def _hydraulic_run(seconds: float, openings, track: tuple[str, ...], device: str = "cpu"):
    """The truck with 1 t on its lowered forks and the hydraulics; ``openings(t)`` sets valve openings. Returns the
    run, every node's gauge pressure [K-1, nodes] and each actuator's piston gauge pressures [K-1, A, 2] at the run's
    times."""
    net, gauge, piston = ft.hydraulics(), [], []

    def control(t: float, last: dict) -> dict[str, float]:
        for name, value in openings(t).items():
            net.set_opening(name, value)
        read = net.read()
        gauge.append(read["pressure"][0].copy() - fluid.P_ATM)
        piston.append(read["piston_pressure"][0].copy())
        return {}

    r = lg.run(lg.TRUCK, seconds, {"brake": 1.0}, build=_truck(ft.Pose(), LOAD), extra=lambda: [net],
               control=control, track=track, device=device)
    return r, np.asarray(gauge), np.asarray(piston)


LOAD = ft.Load(1000.0)


def _valve(t: float, plan: tuple[tuple[float, float], ...], ramp: float = 0.3) -> float:
    """A valve following ``plan`` ((time, opening), ...), each change ramped over ``ramp`` seconds."""
    value = 0.0
    for start, target in plan:
        value += (target - value) * _ramp(t, start, ramp)
    return value


# lift, hold mid free lift, lift, hold mid staging, lift to the top (times from the closed-form speeds)
LIFT_PLAN = ((0.5, 1.0), (1.5, 0.0), (2.5, 1.0), (4.9, 0.0), (5.9, 1.0))


@SLOW
def test_m3_free_lift_goes_first_then_the_stages_at_the_closed_form_pressures_and_speeds():
    r, gauge, piston = _hydraulic_run(10.5, lambda t: {"lift_up": _valve(t, LIFT_PLAN)},
                                      ("mast_outer", "carriage", "mast_int"))
    n = len(gauge)
    t, carriage, stage = r["t"][:n], _in_outer(r, "carriage")[:n, 2], _in_outer(r, "mast_int")[:n, 2]
    free, lift = carriage - 2.0 * stage, gauge[:, ft.NODES.index("lift")]
    want_p, want_v = fm.lift_pressures(LOAD), fm.lift_speeds(LOAD)
    assert stage[free < 0.95].max() < 1.0e-3, stage[free < 0.95].max()     # free lift first
    held_free, held_stage = (t > 2.0) & (t < 2.5), (t > 5.4) & (t < 5.9)
    assert 0.3 < free[held_free].mean() < 0.8 and 0.3 < stage[held_stage].mean() < 1.4
    moving_free = (t > 0.9) & (t < 1.5)                                        # valve fully open
    moving_stage = (t > 6.4) & (stage < 1.6)
    assert float(np.polyfit(t[moving_free], carriage[moving_free], 1)[0]) == pytest.approx(want_v[0], rel=0.02)
    assert float(np.polyfit(t[moving_stage], carriage[moving_stage], 1)[0]) == pytest.approx(want_v[1], rel=0.02)
    assert float(lift[held_free].mean()) == pytest.approx(want_p[0], rel=0.01)
    assert float(lift[held_stage].mean()) == pytest.approx(want_p[1], rel=0.01)
    main_piston = piston[:, 1, 0]                       # what the main pistons act with while lifting (3 s, ~10 rings)
    assert float(main_piston[moving_stage].mean()) == pytest.approx(want_p[1], rel=0.01)
    assert float(carriage[-1]) == pytest.approx(fm.top_height(), abs=1.0e-3)


@SLOW
def test_m4_tilt_vane_holds_the_moment_and_turns_at_q_over_d():
    go = 2.0
    r, gauge, _ = _hydraulic_run(2.8, lambda t: {"tilt_fwd": _ramp(t, go, 0.3), "tilt_b_drain": _ramp(t, go, 0.3)},
                              ("mast_outer",))
    n = len(gauge)
    t = r["t"][:n]
    rel = Rotation.from_quat(r["q"][:n, 3:]).inv() * Rotation.from_quat(r["mast_outer.q"][:n, 3:])
    tilt = rel.as_euler("ZYX")[:, 1]                                         # relative to the chassis
    dp = gauge[:, ft.NODES.index("tilt_b")] - gauge[:, ft.NODES.index("tilt_a")]
    held = (t > 1.0) & (t < go)
    assert float(dp[held].mean()) == pytest.approx(fm.tilt_moment(ft.Pose(), LOAD) / ft.TILT_D, rel=0.01)
    turning = (t > go + 0.3) & (tilt < 0.8 * FWD)
    assert turning.sum() >= 5, turning.sum()
    assert float(np.polyfit(t[turning], tilt[turning], 1)[0]) == pytest.approx(fm.tilt_rate(), rel=0.02)


@pytest.mark.gpu
def test_m3_free_lift_on_cuda():
    """L4 smoke of the shared lift node on CUDA (MuJoCo Warp, float32, graphs): free lift first, at the closed-form
    speed, then held at the closed-form pressure."""
    r, gauge, _ = _hydraulic_run(2.5, lambda t: {"lift_up": _valve(t, LIFT_PLAN)},
                                 ("mast_outer", "carriage", "mast_int"), device="cuda:0")
    n = len(gauge)
    t, carriage, stage = r["t"][:n], _in_outer(r, "carriage")[:n, 2], _in_outer(r, "mast_int")[:n, 2]
    want_p, want_v = fm.lift_pressures(LOAD), fm.lift_speeds(LOAD)
    assert np.abs(stage).max() < 1.0e-3, np.abs(stage).max()
    moving, held = (t > 0.9) & (t < 1.5), (t > 2.0) & (t < 2.5)
    assert float(np.polyfit(t[moving], carriage[moving], 1)[0]) == pytest.approx(want_v[0], rel=0.02)
    assert float(gauge[held, ft.NODES.index("lift")].mean()) == pytest.approx(want_p[0], rel=0.01)
