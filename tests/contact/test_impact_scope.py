# SPDX-License-Identifier: Apache-2.0
"""Where impact-matched gains go (physics rulings 2026-11-13 option A, 2026-11-14 option (1);
alloy.scene.native_gains.impact_applies).

Only balls (one-point contacts, no resting floor) in free-body entries whose dynamic bodies are all balls take them:
rk4 entries, and free-body clutter on implicitfast with rk4's gains and the friction cap (Warp; MuJoCo C keeps rk4
there). One free box, capsule, cylinder or mesh keeps the whole entry static (``native_gains.static_entry``), so a ball
on a free box is static. Everything else keeps the static rule bit for bit. Multi-point contacts rock under the larger
kd's uncapped friction pull: a 30 cm cube pushed to
4.6 m/s on rk4 rocked 8.6 deg against 3.9 and slid 4 % fast (tests/bench/test_rocking_law.py found it), and beside a
robot at a 2.5 ms step a sliding 27 kg cube hopped 4-15 mm against 0.6-2.0. A ball rolls or skids on one point.
- static bit for bit (state digests against the static rule everywhere, ``native_gains.planning_rule``): a robot
  entry, a kinematic entry, a chain-grip entry and a mixed scene (free bodies beside a robot); a box, a capsule, a
  cylinder and a convex mesh pushed to about 4.6 m/s, alone (rk4) and a box among 8 more (clutter); XB4-9's stack
  settling. Planted, the robot scene with impact gains on implicitfast hashes differently.
- balls on impact gains: pushed to 10 m/s (rolling) and kicked to 4.6 m/s skidding, alone (rk4, both CPU routes) and
  among 8 more (clutter, Warp): they rise and bounce no more than on the static rule;
- a ball resting on, and dropped onto, a free box, by a geom contact and by a pair row: static bit for bit (taking the
  box's gains by solmix instead, sized for its 212 kg, let the 4.1 kg ball sink past 1 cm).
"""

from __future__ import annotations

import contextlib
import hashlib

import mujoco
import numpy as np
import pytest

import alloy
from alloy.contact import mjw_friction_cap as FC
from alloy.scene import integrator_route as IR
from alloy.scene import native_gains as NG

pytestmark = [pytest.mark.law, pytest.mark.compile_heavy, pytest.mark.timeout(3000)]
MU, G, BALL_R = 0.5, 9.81, 0.05
ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="b" pos="0 0 0.05"><joint name="j" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/></body></worldbody>
  <actuator><position joint="j" kp="200" kv="10"/></actuator></mujoco>"""
CUBE_V = [(x, y, z) for x in (-0.15, 0.15) for y in (-0.15, 0.15) for z in (-0.15, 0.15)]
CUBE_F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
          (1, 5, 7), (1, 7, 3)]


def entry(sim) -> tuple[str, float | None, bool]:
    """(integrator, friction cap in m/s2 or None, MuJoCo C) of the scene's MuJoCo entry."""
    st = sim.engine.stepper
    s = next(s for s in (getattr(e, "solver", e) for e in (getattr(st, "entries", None) or [st]))
             if hasattr(s, "mj_model"))
    return (mujoco.mjtIntegrator(int(s.mj_model.opt.integrator)).name, getattr(s, "_alloy_friction_cap", None),
            bool(getattr(s, "use_mujoco_cpu", False)))


def _build(scene: alloy.Scene, today: bool, impact_everywhere: bool = False):
    with contextlib.ExitStack() as stack:
        if today:
            stack.enter_context(NG.planning_rule())
        if impact_everywhere:                 # planted: impact gains on implicitfast too
            mp = stack.enter_context(pytest.MonkeyPatch.context())
            mp.setattr(NG, "IMPACT_MODES", (*NG.IMPACT_MODES, "implicitfast"))
        return alloy.build(scene, device="cpu")


def _hash_run(sim, act, steps: int) -> str:
    """sha256 of the model's shape gains, its MuJoCo pair rows and every state value over ``steps`` periods."""
    h, m = hashlib.sha256(), sim.engine.model
    for a in (m.shape_material_ke, m.shape_material_kd):
        h.update(a.numpy().tobytes())
    for name in ("pair_solref", "pair_solreffriction", "pair_solimp"):
        a = getattr(getattr(m, "mujoco", None), name, None)
        h.update(b"" if a is None else a.numpy().tobytes())
    act(sim)
    for _ in range(steps):
        sim.step(1)
        st = sim.state()
        for k in sorted(st):
            h.update(np.ascontiguousarray(st[k]).tobytes())
    return h.hexdigest()


def _entry_scene(kind: str) -> alloy.Scene:
    """``kind``: robot, kinematic, chain or mixed (a steel brick and a dropping ball in each), or robot_ball (the arm
    and the ball only: the planted check, as a brick keeps an entry static whatever its gains mode)."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    if kind != "robot_ball":
        s.add_box("brick", size=(0.1,) * 3, pos=(0.5, 0.25, 0.05 + 1e-4), material="steel_carbon")   # 7.85 kg
    s.add_sphere("ball", BALL_R, pos=(0.5, -0.25, 0.15), material="steel_carbon")                  # 4.1 kg, drops
    if kind == "kinematic":
        s.add_open_box("tray", (0.3, 0.3, 0.05), wall=0.01, pos=(-0.4, 0.0, 0.0), kinematic=True)
        s.add_box("load", size=(0.08,) * 3, pos=(-0.4, 0.0, 0.05), material="steel_carbon")
        return s
    s.add_robot("arm", mjcf=ARM)
    if kind == "mixed":                       # 9 more free bodies: clutter, were it not beside a robot
        for k in range(9):
            s.add_sphere(f"s{k}", 0.03, pos=(-0.5, -0.4 + 0.1 * k, 0.03 + 1e-4), material="steel_carbon")
    if kind == "chain":
        s.add_rod("wire", length=0.24, radius=0.004, pos=(0.05, -0.12, 0.004), direction=(1, 0, 0), segments=6)
    return s


def _entry_digest(kind: str, today: bool, impact_everywhere: bool = False) -> tuple[str, tuple]:
    sim = _build(_entry_scene(kind), today, impact_everywhere)
    try:
        act = ((lambda s: s.drive("tray", velocity=(0.1, 0.0, 0.0), duration=0.3)) if kind == "kinematic"
               else (lambda s: s.set_targets({"j": 1.2})))
        digest = _hash_run(sim, act, 30)
        assert sim.check()["ok"], sim.check()["problems"]
        return digest, tuple(sim.built.report.scene_effects)
    finally:
        sim.close()


@pytest.mark.parametrize("kind", ["robot", "kinematic", "chain", "mixed"])
def test_entries_beside_a_robot_kinematic_object_or_chain_keep_the_static_gains_bit_for_bit(kind):
    now, effects = _entry_digest(kind, today=False)
    assert IR.RK4_EFFECT not in effects and IR.R1_EFFECT not in effects, effects
    assert now == _entry_digest(kind, today=True)[0]


def test_the_digest_sees_impact_gains_planted_on_a_robot_entry():
    assert (_entry_digest("robot_ball", today=False, impact_everywhere=True)[0]
            != _entry_digest("robot_ball", today=True)[0])


def _add(s: alloy.Scene, shape: str, name: str, pos) -> None:
    """A 30 cm body of 1000 kg/m3 (mu 0.5) resting at ``pos`` (its centre's x, y)."""
    x, y = pos
    if shape == "box":
        s.add_box(name, size=(0.3,) * 3, pos=(x, y, 0.15 + 1e-4), density=1000.0, friction=MU)
    elif shape == "capsule":
        s.add_capsule(name, 0.1, 0.3, pos=(x, y, 0.1 + 1e-4), rpy=(np.pi / 2, 0.0, 0.0), density=1000.0, friction=MU)
    elif shape == "cylinder":
        s.add_cylinder(name, 0.15, 0.3, pos=(x, y, 0.15 + 1e-4), density=1000.0, friction=MU)
    else:
        s.add_mesh(name, vertices=CUBE_V, faces=CUBE_F, collision="convex", pos=(x, y, 0.15 + 1e-4), density=1000.0,
                   friction=MU)


@pytest.mark.parametrize("shape,clutter", [("box", False), ("capsule", False), ("cylinder", False), ("mesh", False),
                                           ("box", True)])
def test_floored_shapes_pushed_fast_keep_the_static_gains_bit_for_bit(shape, clutter):
    """Pushed at 1.5 mu m g for 2 s (to about 4.6 m/s), the rocking law's push, alone (rk4) or among 8 more far off."""
    digests = []
    for today in (False, True):
        s = alloy.Scene(dt=0.01)
        s.add_ground(friction=MU)
        _add(s, shape, "blk", (0.0, 0.0))
        for i in range(IR.R1_MIN_FREE if clutter else 0):
            _add(s, shape, f"rest{i}", (5.0 + 2.0 * i, 5.0))
        sim = _build(s, today)
        try:                                  # 1.5 mu m g for the 27 kg cube; the same force on every shape
            digests.append(_hash_run(sim, lambda x: x.apply_wrench("blk", force=(1.5 * MU * 27.0 * G, 0.0, 0.0)), 250))
        finally:
            sim.close()
    assert digests[0] == digests[1], shape


def test_the_rocking_law_stack_settles_on_the_static_gains_bit_for_bit():
    """XB4-9's resting stack (3 x 3 x 3 cubes of 1 m at 10 kg/m3: free-body clutter) settling for 1 s."""
    digests = []
    for today in (False, True):
        s = alloy.Scene(dt=0.01)
        s.add_ground(friction=1.0)
        for i, j, k in ((i, j, k) for i in range(3) for j in range(3) for k in range(3)):
            s.add_box(f"c{i}{j}{k}", size=(1.0,) * 3, pos=(2.0 * i, 2.0 * j, 0.5 + 1.002 * k), density=10.0,
                      friction=1.0)
        sim = _build(s, today)
        try:
            digests.append(_hash_run(sim, lambda x: None, 100))
        finally:
            sim.close()
    assert digests[0] == digests[1]


def _ball(case: str, clutter: bool, today: bool) -> tuple[tuple, float, float, float]:
    """A 4.1 kg steel ball (r 5 cm, mu 0.5) at rest 0.3 s, then pushed at its centre by 1.5 mu m g for 2 s (it rolls
    to about 10 m/s) or kicked for one 10 ms period to 4.6 m/s with no spin (it skids, then rolls): (entry, rise of
    its centre above rest (mm), largest |vz| (m/s), end speed)."""
    s = alloy.Scene(dt=0.01)
    s.add_ground(friction=MU)
    s.add_sphere("ball", BALL_R, pos=(0.0, 0.0, BALL_R + 1e-4), material="steel_carbon", friction=MU)
    for k in range(IR.R1_MIN_FREE if clutter else 0):
        s.add_sphere(f"rest{k}", BALL_R, pos=(-2.0, -1.4 + 0.4 * k, BALL_R + 1e-4), material="steel_carbon")
    sim = _build(s, today)
    try:
        m = 4.0 / 3.0 * np.pi * BALL_R**3 * 7850.0
        sim.run(0.3)
        if case == "push":
            sim.apply_wrench("ball", force=(1.5 * MU * m * G, 0.0, 0.0))
            tr = sim.run(2.0, record=["ball"]).world(0)
        else:
            sim.apply_wrench("ball", force=(4.6 * m / 0.01, 0.0, 0.0))
            sim.step(1)
            sim.apply_wrench("ball", force=(0.0, 0.0, 0.0))
            tr = sim.run(1.5, record=["ball"]).world(0)
        assert sim.check()["ok"], sim.check()["problems"]
        z, vz = tr["ball.pos"][:, 2] - BALL_R, tr["ball.vel"][:, 2]
        return entry(sim), float(1e3 * z.max()), float(np.abs(vz).max()), float(tr["ball.vel"][-1, 0])
    finally:
        sim.close()


@pytest.mark.parametrize("clutter,route", [(False, "mjc"), (False, "warp"), (True, "warp")],
                         ids=["rk4-mjc", "rk4-warp", "clutter-warp"])
@pytest.mark.parametrize("case", ["push", "skid"])
def test_a_ball_on_impact_gains_rises_and_bounces_no_more_than_on_the_static_rule(case, clutter, route, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    (ent, rise, vz, v), (ent0, rise0, vz0, v0) = _ball(case, clutter, False), _ball(case, clutter, True)
    print(f"{route} {'clutter' if clutter else 'rk4'} {case} {ent}: rise {rise:.3f} mm (static rule {rise0:.3f}), "
          f"|vz| {vz:.4f} m/s ({vz0:.4f}), end speed {v:.3f} m/s ({v0:.3f})")
    want = ("mjINT_IMPLICITFAST", FC.CAP_DEFAULT, False) if clutter else ("mjINT_RK4", None, route == "mjc")
    assert ent == ent0 == want, (ent, ent0)
    assert rise <= max(rise0 + 0.1, 1.05 * rise0) and vz <= 1.05 * vz0 + 0.01, (rise, rise0, vz, vz0)   # clutter's
    # kick hops both rules ~17 mm on implicitfast (16.2 against 17.3 mm; |vz| 0.586 against 0.562): within 5 %
    assert v == pytest.approx(v0, rel=0.01)


def _on_box(today: bool, pair: bool) -> str:
    """A 4.1 kg steel ball resting on a free 0.3 m steel box (world 0) and dropped onto it at 1 m/s (world 1), 0.6 s:
    the digest. Equal friction (0.5) leaves a geom contact; 0.5 against 0.8 makes a pair row."""
    s = alloy.Scene(worlds=2, dt=0.00125)
    s.add_ground(friction=MU)
    s.add_box("box", size=(0.3,) * 3, pos=(0.0, 0.0, 0.15 + 1e-4), material="steel_carbon", friction=MU)
    top, h = 0.3 + 1e-4, 1.0 / (2 * G)
    s.add_sphere("ball", BALL_R, pos=np.array([[0.0, 0.0, top + BALL_R], [0.0, 0.0, top + BALL_R + h]]),
                 material="steel_carbon", friction=0.8 if pair else MU)
    sim = _build(s, today)
    try:
        rows = getattr(getattr(sim.engine.model, "mujoco", None), "pair_geom1", None)
        assert (rows is not None and rows.numpy().size > 0) == pair
        digest = _hash_run(sim, lambda x: None, 480)
        assert sim.check()["ok"], sim.check()["problems"]
        return digest
    finally:
        sim.close()


@pytest.mark.parametrize("pair", [False, True], ids=["geom", "pair-row"])
def test_a_ball_on_a_free_box_keeps_the_static_gains_bit_for_bit(pair):
    """Static when either side has a floor (the whole entry: ``native_gains.static_entry``). The same ball on a static
    slab takes impact gains (tests/contact/test_impact_laws.py, rebound)."""
    assert _on_box(False, pair) == _on_box(True, pair)
