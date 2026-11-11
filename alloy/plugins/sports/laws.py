# SPDX-License-Identifier: Apache-2.0
"""PressureBall laws: FIFA's drop tests and ball roll in the simulation, against ``model.py``.

Each law checks the simulation against the float64 model (the kernels and their time stepping). The FIFA numbers
are calibration, not validation: ``SHELL_C``, ``TURF_C`` and ``TURF_ROLL`` were fitted to them, so those checks
pass by construction and only guard that the fits still hold. No law tests the model against independent data.

| Law | Kind | Scene | Reference | Bound |
|---|---|---|---|---|
| rebound_vs_pressure | sim vs model | 2.00 m drops, rigid plate, 0.608 / 0.8 / 1.115 bar, in air | drop_rebound | 1 % |
| calibration_ball_rebound | calibration | the 0.8 bar world | 1.45 ± 0.03 m (the fit's datum) | in band |
| impact_duration | sim vs model | the 0.8 bar world's contact time and peak force | impact | 5 % |
| turf_rebound | sim vs model (+ calibration band) | 2.00 m drop on the pitch | drop_rebound(turf) | 2 % |
| ball_roll | sim vs model (+ calibration band) | FIFA ramp speed, rolling on the pitch | v0²/(2a) | 2 % |
"""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.sports import model as M
from alloy.plugins.sports.scene import add_ball, launch, with_inputs

DT = 2.5e-4                                          # s: ~33 steps per contact at 0.8 bar
PRESSURES = (M.P_IFAB[0], M.P_TEST, M.P_IFAB[1])     # Pa gauge
PLATE = "plate"
RIGID = {"pitch": False, "colliders": (PLATE,)}


def _companions() -> list:
    from alloy.plugins.sports.aero import BallAero  # noqa: PLC0415

    return [BallAero()]


def _drop_build(z0: float, plate: bool):
    def build(b: newton.ModelBuilder) -> None:
        if plate:
            b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.05), hx=0.3, hy=0.3, hz=0.05,
                            cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False), label=PLATE)
        add_ball(b, (0.0, 0.0, z0))
    return build


def default_scene() -> TestScene:
    """Three identical worlds: a spinning ball bounces along the pitch towards a wall (the harness perturbs worlds
    1.. itself)."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(1.2, 0.0, 0.3), hx=0.05, hy=0.5, hz=0.3, label="wall")
        add_ball(b, (0.0, 0.0, 0.4))
    starts = [((3.0, 0.2, 0.0), (0.0, 10.0, 5.0))]
    return TestScene(build=build, worlds=3, periods=40, dt=DT, ground=False, init=launch(starts),
                     companions=_companions)


def drop_scene(pressures: tuple[float, ...], plate: bool) -> TestScene:
    """FIFA's 2.00 m drop, one world per gauge pressure [Pa], onto the rigid plate or the pitch."""
    init = with_inputs(None, "pressure_ball", pressure=np.array(pressures, np.float32)[:, None])
    return TestScene(build=_drop_build(M.FIFA_DROP + M.R_BALL, plate), worlds=len(pressures), periods=70, dt=DT,
                     ground=False, init=init, companions=_companions)


def bounce(run, label: str = "ball") -> np.ndarray:
    """Per world: the first rebound's apex [m] of the ball's bottom (NaN if it never rose and fell again)."""
    z = run.trace.body_q[:, run.bodies(label), 2].astype(np.float64)
    vz = run.trace.body_qd[:, run.bodies(label), 2].astype(np.float64)
    apex = np.full(z.shape[1], np.nan)
    for w in range(z.shape[1]):
        up = np.flatnonzero(vz[:, w] > 0.0)
        down = np.flatnonzero(vz[up[0]:, w] < 0.0) if len(up) else []
        if len(down):
            k = int(up[0] + down[0])
            apex[w] = _apex(z[k - 1:k + 1, w], vz[k - 1:k + 1, w]) - M.R_BALL
    return apex


def _apex(z: np.ndarray, vz: np.ndarray) -> float:
    """The apex between two samples straddling vz = 0 under gravity (the drag is tiny there)."""
    k = int(np.argmin(np.abs(vz)))
    return float(z[k] + vz[k] ** 2 / (2.0 * M.G))


def check_rebound(run) -> LawResult:
    h = bounce(run)
    want = np.array([M.drop_rebound(p=p)["rebound_m"] for p in PRESSURES[:len(h)]])
    err = h / want - 1.0
    k = int(np.argmax(np.abs(err)))
    rises = bool(np.all(np.diff(h) > 0.0))
    detail = ", ".join(f"{p / 1e5:.3f} bar {a:.3f} vs {b:.3f} m" for p, a, b in zip(PRESSURES, h, want))
    return LawResult(bool(abs(err[k]) <= 0.01 and rises), float(err[k]), 0.01,
                     f"rebound from 2.00 m on a rigid plate: {detail}; rises with pressure: {rises}",
                     "check shell_force / the dashpot in kernels.ball_contacts against model.py")


def check_calibration(run) -> LawResult:
    h = bounce(run)
    lo, hi = M.FIFA_REBOUND_TEST_BALL
    off = max(lo - h[1], h[1] - hi, 0.0)
    return LawResult(bool(off == 0.0), float(off), 0.0,
                     f"calibration: the 0.8 bar ball still rebounds {h[1]:.3f} m from 2.00 m, the datum SHELL_C was "
                     f"fitted to (FIFA test ball 1.45 ± 0.03 m)", "refit SHELL_C (model.py) to 1.45 m")


def check_impact(run) -> LawResult:
    touch = run.trace.plugin("pressure_ball", "touch")            # [K, W, B, 1 + C, 2]
    dur, peak = float(touch[-1, 1, 0, 1, 0]), abs(float(touch[-1, 1, 0, 1, 1]))
    ref = M.impact(M.fall_speed(M.FIFA_DROP))
    e_t = (dur - ref["contact_s"]) / ref["contact_s"]
    e_f = peak / ref["peak_n"] - 1.0
    worst = e_t if abs(e_t) > abs(e_f) else e_f
    return LawResult.within(worst, 0.05, f"0.8 bar impact at {M.fall_speed(M.FIFA_DROP):.2f} m/s: contact "
                            f"{dur * 1e3:.2f} vs {ref['contact_s'] * 1e3:.2f} ms, peak {peak:.0f} vs "
                            f"{ref['peak_n']:.0f} N", "contact time follows π·sqrt(m/k): check the stiffness")


def check_turf(run) -> LawResult:
    h = bounce(run)
    want = M.drop_rebound(surface="turf")["rebound_m"]
    err = h[0] / want - 1.0
    lo, hi = M.FIFA_TURF_REBOUND
    ok = abs(err) <= 0.02 and lo <= h[0] <= hi
    return LawResult(bool(ok), float(err), 0.02, f"2.00 m drop on the pitch rebounds {h[0]:.3f} m vs {want:.3f} m "
                     f"(series model); calibration band {lo:.2f}-{hi:.2f} m (TURF_C was fitted to its middle)",
                     "check kernels.turf_step against model.py")


def roll_scene() -> TestScene:
    v0 = M.ramp_exit_speed()
    return TestScene(build=_drop_build(M.R_BALL - 2e-4, False), worlds=1, periods=230, dt=2.0 * DT, ground=False,
                     init=launch([((v0, 0.0, 0.0), (0.0, v0 / M.R_BALL, 0.0))]))


def check_roll(run) -> LawResult:
    b = run.bodies("ball")[0]
    x, v = run.trace.body_q[:, b, 0].astype(np.float64), run.trace.body_qd[:, b, 0].astype(np.float64)
    stop = np.flatnonzero(np.abs(v) < 0.02)
    d = float(x[stop[0]] if len(stop) else x[-1]) - float(x[0])
    want = M.roll_distance(M.ramp_exit_speed())
    lo, hi = M.FIFA_BALL_ROLL
    err = d / want - 1.0
    return LawResult(bool(abs(err) <= 0.02 and lo <= d <= hi), float(err), 0.02,
                     f"rolls {d:.2f} m from {M.ramp_exit_speed():.2f} m/s vs {want:.2f} m "
                     f"(C_rr {M.TURF_ROLL}); calibration band {lo:.1f}-{hi:.1f} m (TURF_ROLL was fitted to its "
                     "middle)", "check the rolling torque in kernels")


def ball_laws() -> list[Law]:
    rigid = drop_scene(PRESSURES, True)
    return [
        Law("rebound_vs_pressure", check_rebound, rigid, config=RIGID,
            doc="2.00 m drops on a rigid plate at the IFAB pressure limits and FIFA's 0.8 bar match model.py"),
        Law("calibration_ball_rebound", check_calibration, rigid, config=RIGID,
            doc="calibration: the fit of SHELL_C still holds (1.45 m, the datum it was fitted to)"),
        Law("impact_duration", check_impact, rigid, config=RIGID,
            doc="contact time and peak force of the 0.8 bar impact match model.impact"),
        Law("turf_rebound", check_turf, drop_scene((M.P_TEST,), False),
            doc="a 2.00 m drop on the pitch rebounds as the series model (calibration band 0.60-1.00 m)"),
        Law("ball_roll", check_roll, roll_scene(),
            doc="a ball leaving FIFA's ramp rolls v0²/(2a) on the pitch (calibration band 4.0-10.0 m)"),
    ]


__all__ = ["DT", "PRESSURES", "ball_laws", "bounce", "default_scene", "drop_scene", "roll_scene"]
