# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(scene, welds="stiff")`` rows in use (GAP-W; reviews WP3, WP13, AUDIT-STACK; CPU).

- The rows follow the MuJoCo step that runs: the rule's ω h = 0.6 (``alloy.scene.weld_impedance``) at h = the control
  period over the physics steps per period, for the default step and for ``scene.rigid_step`` pinned at 2.5 / 5 ms.
- Stability at the largest steps for light, heavy and lopsided pairs (1 g to 1 t, mass ratios to 1e6): a held step load, then its
  release. The row is a critically damped spring-damper in acceleration space (ω h = 0.6 < 2(√2 - 1), the
  semi-implicit bound; module docstring of weld_impedance), so mass does not enter its stability. Law: the loaded
  stretch is the row's compliance (``alloy.route.weld_stiff.rest_stretch``, W from the engine's own MuJoCo model)
  within 2 %, at rest (ripple under 1 %), back to the weight sag after the release with no overshoot past the loaded
  stretch (x 1.2), finite and not INVALID.
- welds="mujoco" leaves MuJoCo's rows; leaving welds out steps bit for bit as welds=DEFAULT_WELDS.
- ``rerun_worlds``' side simulation is built inside the same request and carries the stiff rows.
- An opt-in that finds no default row says so once; the default (welds left out) says nothing in a scene with no
  default row (AGENTS.md's quick start, a weld-free arm).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

import alloy
from alloy.api._capacity import mujoco_solver
from alloy.route.weld_stiff import DEFAULT_SOLREF, DEFAULT_WELDS, RowParams, rest_stretch
from alloy.scene.weld_impedance import WELD_IMP_CAP, WELD_OMEGA_H

# robot builds + sim.check(): cold kernels on a loaded gate host take minutes (suite default 60 s)
pytestmark = pytest.mark.timeout(600)

G = 9.81
NOISE_M = 2.0e-7                     # float32 positions near 0.6 m: a few ulp
OFFSET = np.array([0.1, 0.0, 0.0])


def _mjcf(m_a: float, m_b: float, step: float, weld: str = '<weld body1="a" body2="b"/>') -> str:
    """A slide body ``a`` (servo sized to hold a + b at step ``step``) and a free body ``b`` welded at b's centre."""
    total, omega = m_a + m_b, 0.3 / step
    return f"""<mujoco><compiler angle="radian"/><worldbody>
<body name="a" pos="0 0 0.6"><joint name="ja" type="slide" axis="1 0 0" range="-1 1"/>
  <geom type="box" size="0.03 0.03 0.03" mass="{m_a}" contype="0" conaffinity="0"/></body>
<body name="b" pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="{m_b}" contype="0"
  conaffinity="0"/></body></worldbody><equality>{weld}</equality>
<actuator><position joint="ja" kp="{total * omega**2:.6g}" kv="{2.0 * total * omega:.6g}"/></actuator></mujoco>"""


def _build(m_a: float, m_b: float, rigid_step: float | None, **kw):
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_robot("rig", mjcf=_mjcf(m_a, m_b, rigid_step or 1.25e-3, kw.pop("weld", '<weld body1="a" body2="b"/>')))
    if rigid_step is not None:
        scene.rigid_step = rigid_step
    return alloy.build(scene, device="cpu", **kw)


def _step_run(sim) -> float:
    before = sim.physics_steps
    sim.step(1)
    return sim.dt / (sim.physics_steps - before)


def _row(sim) -> RowParams:
    mj = sim.engine.model.mujoco
    return RowParams(tuple(float(x) for x in mj.eq_solref.numpy()[0]), tuple(float(x) for x in mj.eq_solimp.numpy()[0]))


@pytest.mark.parametrize("rigid_step", [None, 0.0025, 0.005], ids=["default-step", "2.5ms", "5ms"])
def test_the_rows_follow_the_step_that_runs(rigid_step: float | None) -> None:
    sim = _build(5.0, 20.0, rigid_step, welds="stiff")
    h = _step_run(sim)
    omega = WELD_OMEGA_H / h
    row = _row(sim)
    assert row.solref == pytest.approx((-omega * omega, -2.0 * omega), rel=1e-6)
    assert row.solimp[:2] == pytest.approx((WELD_IMP_CAP, WELD_IMP_CAP), rel=1e-6)
    if rigid_step is not None:
        assert h == pytest.approx(rigid_step, rel=1e-9)


STABILITY = [  # rigid step, m_a (slide, servo), m_b (free), held load
    (1.25e-3, 5.0, 20.0, 3.0e4),
    (0.005, 5.0, 20.0, 3.0e4),
    (0.01, 1000.0, 1000.0, 1.0e5),       # the coarsest step there is at dt = 0.01: one per control period
    (0.005, 1000.0, 0.001, 0.05),        # a 1 g body on a 1 t carrier
    (0.005, 0.001, 1.0, 10.0),           # a 1 kg body on a 1 g carrier: W = 334 /kg, the row is soft in force
]
# Not a case: a 1 t body on a 1 g carrier. The row's force stiffness is about ω² / ((1 - d) W), 43 kN/m at 5 ms
# with W = 333 /kg, so the 1 t weight alone would stretch it 0.23 m: far outside the small-violation regime the
# formula and the weld are for (measured: 96-121 mm, swinging; MuJoCo's default row 5-7 m). The page states the limit.


@pytest.mark.law
@pytest.mark.parametrize("route", ["mjc", "warp"])
@pytest.mark.parametrize("case", STABILITY, ids=[f"{s * 1e3:g}ms-{a:g}kg-{b:g}kg" for s, a, b, _ in STABILITY])
def test_a_stiff_weld_is_stable_at_the_largest_steps(case, route: str, monkeypatch) -> None:
    step, m_a, m_b, load = case
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _build(m_a, m_b, step, welds="stiff")
    row = _row(sim)
    iw = mujoco_solver(sim).mjw_model.body_invweight0.numpy()
    iw = iw.reshape(-1, iw.shape[-2], iw.shape[-1])[0]
    inv_weight = float(iw[-2, 0] + iw[-1, 0])                        # bodies a, b: the row's diagApprox
    sim.step(5)
    sim.apply_wrench("rig/b", force=np.array([[0.0, load, 0.0]]))
    loaded = _gap(sim.run(1.0, record=["rig/a", "rig/b"]))
    sim.apply_wrench("rig/b", force=np.zeros((1, 3)))
    released = _gap(sim.run(1.0, record=["rig/a", "rig/b"]))
    held = rest_stretch(row, inv_weight, math.hypot(load, m_b * G), step)
    sag = rest_stretch(row, inv_weight, m_b * G, step)
    assert np.isfinite(loaded).all() and np.isfinite(released).all()
    assert loaded[-1] == pytest.approx(held, rel=0.02, abs=NOISE_M)
    assert np.ptp(loaded[-20:]) < 0.01 * held + NOISE_M                 # at rest under the load
    assert released[-1] == pytest.approx(sag, rel=0.02, abs=NOISE_M)
    assert released.max() < 1.2 * held + NOISE_M                         # the release does not ring past the load
    worlds = sim.health()["worlds"]
    assert worlds[0]["severity"] != "INVALID" and sim.check()["ok"], worlds


def _gap(traj) -> np.ndarray:
    return np.linalg.norm(traj["rig/b.pos"] - traj["rig/a.pos"] - OFFSET, axis=2)[:, 0]


def test_mujoco_keeps_the_row_and_the_default_is_its_mode_bit_for_bit() -> None:
    """welds="mujoco" leaves MuJoCo's row as written; leaving welds out steps exactly as welds=DEFAULT_WELDS."""
    runs = {}
    for welds in (None, DEFAULT_WELDS, "mujoco"):
        sim = _build(5.0, 20.0, None, **({} if welds is None else {"welds": welds}))
        if (welds or DEFAULT_WELDS) == "mujoco":
            assert _row(sim).solref == pytest.approx(DEFAULT_SOLREF)
        sim.apply_wrench("rig/b", force=np.array([[0.0, 3000.0, 0.0]]))
        runs[welds] = sim.run(0.5, record=["rig/a", "rig/b"])
    for key in ("rig/a.pos", "rig/b.pos", "rig/b.quat_wxyz"):
        assert np.array_equal(runs[None][key], runs[DEFAULT_WELDS][key]), key


def test_leaving_welds_out_stiffens_a_default_row() -> None:
    """The shipped default is the stiff rule (coordinator ruling 06:02Z, merged 0f647f2a2): this fails if
    DEFAULT_WELDS goes back to "mujoco", where the laws that read it would turn into xfails (AUDIT-BENCH 16:17Z)."""
    assert DEFAULT_WELDS == "stiff"
    row = _row(_build(5.0, 20.0, None))
    assert row.solref[0] < 0.0 and row.solref != pytest.approx(DEFAULT_SOLREF)
    assert row.solimp[:2] == pytest.approx((WELD_IMP_CAP, WELD_IMP_CAP), rel=1e-6)


def test_the_rerun_pack_carries_the_stiff_rows() -> None:
    scene = alloy.Scene(worlds=4, dt=0.01)
    scene.add_ground()
    scene.add_robot("rig", mjcf=_mjcf(5.0, 20.0, 1.25e-3))
    sim = alloy.build(scene, device="cpu", welds="stiff", rerun_worlds=1)
    fine = getattr(sim.engine, "fine_rerun", None)
    if fine is None:
        pytest.skip("nothing in this scene re-runs a period: no side simulation was built")
    side = fine.pack()
    assert side is not None
    assert side.model.mujoco.eq_solref.numpy()[0] == pytest.approx(_row(sim).solref)


def test_an_opt_in_that_changes_nothing_says_so(caplog) -> None:
    authored = '<weld body1="a" body2="b" solref="0.005 1" solimp="0.95 0.99 0.001"/>'
    with caplog.at_level(logging.WARNING):
        sim = _build(5.0, 20.0, None, welds="stiff", weld=authored)
    assert _row(sim).solref == pytest.approx((0.005, 1.0))
    assert any("changed no row" in r.getMessage() for r in caplog.records)


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _quick_start() -> alloy.Scene:
    """AGENTS.md section 2's scene: ground, a static table, a free block."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)
    return scene


def _arm() -> alloy.Scene:
    """AGENTS.md section 5's scene: a robot with no equality at all."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    return scene


@pytest.mark.law
@pytest.mark.parametrize("make", [_quick_start, _arm], ids=["agents-quick-start", "weld-free-arm"])
def test_the_default_says_nothing_about_welds_a_scene_does_not_have(make, caplog, capsys) -> None:
    """Before the fix every build without a default weld row printed the opt-in's "changed no row" line (rc2
    MUST-FIX, main a1f4795e6): only a welds="stiff" the caller passed may say it changed nothing."""
    with caplog.at_level(logging.WARNING):
        sim = alloy.build(make(), device="cpu")
        sim.step(1)
    said = [r.getMessage() for r in caplog.records if "weld" in r.getMessage()] + \
        [line for line in capsys.readouterr().err.splitlines() if "weld" in line]
    assert not said, said
    with caplog.at_level(logging.WARNING):
        alloy.build(make(), device="cpu", welds="stiff")
    assert any("changed no row" in r.getMessage() for r in caplog.records)
