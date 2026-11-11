# SPDX-License-Identifier: Apache-2.0
"""A weld that holds: stretch of an MJCF weld / connect under a static pull against the row's own compliance (GAP-W).

Rig (GAP-E3's i03, the audit's ERR-11 rig): a 5 kg body ``a`` on an x slide (position servo) and a free 20 kg body
``b`` joined by an equality at b's centre; a held ``sim.apply_wrench`` pulls b sideways at 3 / 10 / 30 kN.

The bound is derived (``alloy.route.weld_stiff``, from MuJoCo "Solver parameters" and engine_core_constraint.c): a
soft row carrying F rests at r = F W (1 - d) / (d² k), W = the row's diagApprox = 1/m_b + 1/(3 m_a) here. At MuJoCo's
defaults that is 2.33 mm per kN: 7.0 / 23.3 / 70.0 mm. Under ``alloy.build(scene, welds="stiff")`` the row takes the
scene-attachment rule (``alloy.scene.weld_impedance``: direct solref at ω h = 0.6, impedance 0.999), k = ω² / d², so
r = F W (1 - d) / ω² = 5.06e-7 mm per N at the 1.25 ms step: 0.0015 / 0.0051 / 0.0152 mm.

Laws:
- compliance: with either setting the simulated stretch is the formula's, read from the row the engine built;
- a stiff weld holds: at 3 / 10 / 30 kN the stretch stays under ``HOLD_BUDGET_M`` (0.1 mm, the attachment rule's own
  separation target) and the world reads OK with no JOINT_FAIL;
- the default path (welds left out: ``weld_stiff.DEFAULT_WELDS``) holds when that default is "stiff", and is an
  xfail until then; welds="mujoco" keeps MuJoCo's row, whose 7-70 mm is asserted as the 'before' reading;
- dynamic (a tool welded to a link spun about z at constant omega by a velocity servo): the radial error is the
  semi-implicit step's own lag, r = B h a / (2 K d) inward, less the centripetal force's compliance outward
  (``weld_stiff.spin_lag``; derived, and checked to within 1 % over 72 runs in plain MuJoCo C), and
  the sag under the tool's weight is its compliance; within 5 % of the two terms' sum. The stiff row's error is the
  lag alone, (5/3) a h² = 2.6 um per m/s² at 1.25 ms: under ``HOLD_BUDGET_M`` at omega 8 rad/s, 0.4 m (25.6 m/s²),
  where MuJoCo's own row is 0.44 mm off. (A stiff position servo stepping its target once per 10 ms control period
  kicks the link every period; plain MuJoCo C stepping the same model with the same per-period controls gives the
  same weld error to 7 digits, so that error is MuJoCo's, not Alloy's.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import alloy
from alloy.api._capacity import mujoco_solver
from alloy.route.weld_stiff import DEFAULT_WELDS, RowParams, rest_stretch, row_compliance, spin_lag
from alloy.scene.weld_impedance import WELD_IMP_CAP, WELD_OMEGA_H, WELD_SEP_TOL

# robot builds + sim.check(): cold kernels on a loaded gate host take minutes (suite default 60 s)
pytestmark = pytest.mark.timeout(600)

M_A, M_B, G = 5.0, 20.0, 9.81
W_ROW = 1.0 / M_B + 1.0 / (3.0 * M_A)                   # diagApprox of a translation row [1/kg]
WEIGHT = M_B * G
OFFSET = np.array([0.1, 0.0, 0.0])                      # b - a as built: the equality's target
PULL_N = (3000.0, 10000.0, 30000.0)
HOLD_BUDGET_M = WELD_SEP_TOL                            # 0.1 mm
FLOAT32_NOISE_M = 2.0e-7                                # rest noise of the Warp route on this rig: <= 0.05 um measured
EQUALITIES = {"weld": '<weld body1="a" body2="b"/>',
              "connect": '<connect body1="a" body2="b" anchor="0.1 0 0"/>'}   # at b's centre: no swing
CASES = [("mjc", "weld", "stiff"), ("warp", "weld", "stiff"), ("mjc", "connect", "stiff"), ("warp", "connect", "stiff"),
         ("mjc", "weld", None), ("warp", "weld", None), ("mjc", "weld", "mujoco"),
         pytest.param(("cuda", "weld", "stiff"), marks=pytest.mark.gpu)]


@dataclass(frozen=True)
class RigRun:
    welds: str | None             # as asked of alloy.build (None: the default)
    params: RowParams             # the row as the engine built it
    stretch_m: np.ndarray         # per world, |(b - a) - OFFSET| after 1 s of the held pull
    ripple_m: np.ndarray          # peak to peak over the last 0.2 s
    health: dict[str, Any]


def _mjcf(equality: str) -> str:
    return f"""<mujoco><compiler angle="radian"/><worldbody>
<body name="a" pos="0 0 0.6"><joint name="ja" type="slide" axis="1 0 0" range="-1 1"/>
  <geom type="box" size="0.03 0.03 0.03" mass="{M_A}"/></body>
<body name="b" pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="{M_B}"/></body>
</worldbody><equality>{equality}</equality>
<actuator><position joint="ja" kp="5000" kv="200"/></actuator></mujoco>"""


def _run(route: str, kind: str, welds: str | None) -> RigRun:
    scene = alloy.Scene(worlds=len(PULL_N), dt=0.01)
    scene.add_ground()
    scene.add_robot("rig", mjcf=_mjcf(EQUALITIES[kind]))
    extra = {} if welds is None else {"welds": welds}
    sim = alloy.build(scene, device="cuda:0" if route == "cuda" else "cpu", **extra)
    sim.step(5)
    sim.apply_wrench("rig/b", force=np.array([[0.0, f, 0.0] for f in PULL_N]))
    traj = sim.run(1.0, record=["rig/a", "rig/b"])
    gap = np.linalg.norm(traj["rig/b.pos"] - traj["rig/a.pos"] - OFFSET, axis=2)      # (frames, worlds)
    mj = sim.engine.model.mujoco
    params = RowParams(tuple(float(x) for x in mj.eq_solref.numpy()[0]),
                       tuple(float(x) for x in mj.eq_solimp.numpy()[0]))
    return RigRun(welds, params, gap[-1], np.ptp(gap[-20:], axis=0), sim.health())


@pytest.fixture(scope="module", params=CASES, ids=["-".join(map(str, getattr(c, "values", (c,))[0])) for c in CASES])
def rig(request) -> RigRun:
    route, kind, welds = request.param
    with pytest.MonkeyPatch.context() as mp:
        if route != "cuda":
            mp.setenv("ALLOY_CPU_ROUTE", route)    # users' CPU default (MuJoCo C) and the suite's (MuJoCo Warp)
        return _run(route, kind, welds)


def _expected(params: RowParams) -> np.ndarray:
    return np.array([rest_stretch(params, W_ROW, math.hypot(f, WEIGHT)) for f in PULL_N])


def test_the_budget_is_derived() -> None:
    omega = WELD_OMEGA_H / 1.25e-3
    per_newton = W_ROW * (1.0 - WELD_IMP_CAP) / omega**2                   # r / F of the stiff row at this step
    assert per_newton == pytest.approx(5.064e-10, rel=1e-3)
    assert 30000.0 * per_newton < HOLD_BUDGET_M / 6.0                      # 0.0152 mm: the budget is not tight
    assert _expected(RowParams())[0] > 50.0 * HOLD_BUDGET_M                # the default row misses it at 3 kN


@pytest.mark.law
def test_the_stretch_is_the_rows_own_compliance(rig: RigRun) -> None:
    np.testing.assert_allclose(rig.stretch_m, _expected(rig.params), rtol=0.02, atol=FLOAT32_NOISE_M)
    assert (rig.ripple_m < 0.01 * rig.stretch_m + FLOAT32_NOISE_M).all()   # at rest, not ringing


def _holds_or_skip(welds: str | None) -> None:
    """Skip MuJoCo's own row; xfail the default path while its default keeps that row."""
    if welds == "mujoco":
        pytest.skip("MuJoCo's row: its soft reading is asserted separately")
    if (welds or DEFAULT_WELDS) != "stiff":
        pytest.xfail("the default path keeps MuJoCo's soft row (docs/known_gaps/weld-stretch-under-load.md)")


@pytest.mark.law
def test_a_stiff_weld_holds_30_kn(rig: RigRun) -> None:
    _holds_or_skip(rig.welds)
    assert rig.params.solref[0] < 0.0 and rig.params.solimp[0] == pytest.approx(WELD_IMP_CAP)   # the rule reached it
    assert (rig.stretch_m < HOLD_BUDGET_M).all(), rig.stretch_m
    for w, row in enumerate(rig.health["worlds"]):
        assert row["severity"] == "OK" and "JOINT_FAIL" not in row["flags"], (w, row)


@pytest.mark.law
def test_mujocos_row_does_not_hold(rig: RigRun) -> None:
    """The 'before' reading: MuJoCo's default row (welds="mujoco", and the default path while it is "mujoco")."""
    if (rig.welds or DEFAULT_WELDS) != "mujoco":
        pytest.skip("MuJoCo's row only")
    assert (rig.stretch_m > 50.0 * HOLD_BUDGET_M).all()
    assert all("JOINT_FAIL" in row["flags"] for row in rig.health["worlds"])


# --- dynamic: a welded tool on a spinning link ---------------------------------------------------------------------

RT, M_TOOL, STEP = 0.4, 1.0, 1.25e-3
OMEGAS = (2.0, 4.0, 8.0)                                # rad/s, one world each: 1.6 / 6.4 / 25.6 m/s² at the tool
SPIN_CASES = [("mjc", None), ("warp", None), ("mjc", "stiff"), ("warp", "stiff"), ("mjc", "mujoco")]
SPIN_MJCF = f"""<mujoco><compiler angle="radian"/><worldbody>
<body name="link" pos="0 0 0.3"><joint name="spin" type="hinge" axis="0 0 1"/>
  <geom type="capsule" fromto="0 0 0 {RT} 0 0" size="0.02" mass="2"/><geom type="cylinder" size="0.05 0.02" mass="1"/></body>
<body name="tool" pos="{RT} 0 0.3"><freejoint/>
  <geom type="box" size="0.03 0.03 0.03" mass="{M_TOOL}" contype="0" conaffinity="0"/></body>
</worldbody><equality><weld body1="link" body2="tool"/></equality>
<actuator><velocity joint="spin" kv="50"/></actuator></mujoco>"""


@dataclass(frozen=True)
class SpinRun:
    welds: str | None
    params: RowParams
    inv_weight: float             # the row's diagApprox, from the engine's own MuJoCo model
    omega: np.ndarray             # measured, per world
    error_m: np.ndarray           # (worlds, 3): radial, tangential, vertical, mean over 0.2 s of steady spin
    health: dict[str, Any]


def _spin(welds: str | None) -> SpinRun:
    scene = alloy.Scene(worlds=len(OMEGAS), dt=0.01)
    scene.add_ground()
    scene.add_robot("rig", mjcf=SPIN_MJCF)
    sim = alloy.build(scene, device="cpu", **({} if welds is None else {"welds": welds}))
    mj = sim.engine.model.mujoco
    params = RowParams(tuple(float(x) for x in mj.eq_solref.numpy()[0]),
                       tuple(float(x) for x in mj.eq_solimp.numpy()[0]))
    iw = mujoco_solver(sim).mjw_model.body_invweight0.numpy()
    iw = iw.reshape(-1, iw.shape[-2], iw.shape[-1])[0]              # (bodies, 2): translation, rotation
    sim.set_targets({"spin": np.array(OMEGAS)}, kind="velocity")
    sim.run(1.0)                                                     # spin up (I / kv = 5 ms) and settle
    traj = sim.run(0.2, record=["rig/link", "rig/tool"])
    qw = traj["rig/link.quat_wxyz"]
    yaw = 2.0 * np.arctan2(qw[..., 3], qw[..., 0])
    zero = np.zeros_like(yaw)
    radial = np.stack([np.cos(yaw), np.sin(yaw), zero], -1)
    tangent = np.stack([-np.sin(yaw), np.cos(yaw), zero], -1)
    err = traj["rig/tool.pos"] - traj["rig/link.pos"] - RT * radial
    parts = np.stack([(err * radial).sum(-1), (err * tangent).sum(-1), err[..., 2]], -1).mean(0)
    unwrapped = np.unwrap(yaw, axis=0)
    omega = (unwrapped[-1] - unwrapped[0]) / (sim.dt * (len(yaw) - 1))
    return SpinRun(welds, params, float(iw[-2, 0] + iw[-1, 0]), omega, parts, sim.health())


@pytest.fixture(scope="module", params=SPIN_CASES, ids=[f"{r}-{w or 'default'}" for r, w in SPIN_CASES])
def spin(request) -> SpinRun:
    route, welds = request.param
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ALLOY_CPU_ROUTE", route)
        return _spin(welds)


@pytest.mark.law
def test_the_spin_error_is_the_steps_lag_less_the_compliance(spin: SpinRun) -> None:
    np.testing.assert_allclose(spin.omega, OMEGAS, rtol=1e-3)
    for w in range(len(OMEGAS)):
        viol = float(np.linalg.norm(spin.error_m[w]))                # MuJoCo's impedance reads the whole violation
        accel = spin.omega[w] ** 2 * RT
        lag = spin_lag(spin.params, STEP, accel, viol)
        compliance = row_compliance(spin.params, spin.inv_weight, viol, STEP)
        push = compliance * M_TOOL * accel
        assert abs(spin.error_m[w, 0] - (push - lag)) <= 0.05 * (lag + push) + FLOAT32_NOISE_M, (w, spin.error_m[w])
        sag = compliance * M_TOOL * G
        assert spin.error_m[w, 2] == pytest.approx(-sag, rel=0.02, abs=FLOAT32_NOISE_M)


@pytest.mark.law
def test_a_stiff_weld_holds_a_spun_tool(spin: SpinRun) -> None:
    total = np.linalg.norm(spin.error_m, axis=1)
    if (spin.welds or DEFAULT_WELDS) == "mujoco":
        assert total[-1] > 3.0 * HOLD_BUDGET_M                       # the 'before' reading: 0.44 mm at 25.6 m/s²
    _holds_or_skip(spin.welds)
    assert (total < HOLD_BUDGET_M).all(), total
    assert all(row["severity"] == "OK" for row in spin.health["worlds"])
