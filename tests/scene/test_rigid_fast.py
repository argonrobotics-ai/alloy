# SPDX-License-Identifier: Apache-2.0
"""SPEED-NOW S1: the opt-in rigid fast mode (``scene.rigid_step = "fast"``, or ``profile="fast"`` on a scene that never
set it; alloy.scene.rigid_step, Fast mode; coordinator ruling 2026-11-09 09:18Z) on CPU. It takes the scenes "auto"
takes, steps them at the largest step whose predicted rest sag is within SAG_SHARE of each body's thinnest member
(the other caps stand, creep included), and tags the MuJoCo entry with FAST_EFFECT; every other scene and profile
keeps its step."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import alloy
from alloy.api import _rigid_step as API
from alloy.scene import integrator_route as IR
from alloy.contact.mjw_boxbox_reduce import REDUCE_KEY
from alloy.scene import rigid_step as RS

TODAY = 1.25e-3
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def stack(edge: float = 1.0, density: float = 10.0, layers: int = 3, worlds: int = 1) -> alloy.Scene:
    """One resting column of cubes (XB4-9's box stack at its defaults: 1 m, 10 kg/m^3)."""
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground(friction=1.0)
    for k in range(layers):
        s.add_box(f"c{k}", size=(edge,) * 3, pos=(0.0, 0.0, edge * (0.5 + k) + 1e-4 * k), density=density,
                  friction=1.0)
    return s


def step_of(sim) -> RS.RigidStep:
    return sim.engine.parts.built.report.rigid_step


def effects_of(sim) -> list[str]:
    return list(sim.engine.parts.built.report.scene_effects)


def mujoco_kwargs(sim) -> dict:
    return next(e for e in sim.engine.parts.built.entries if e.name == "mujoco").solver_kwargs


def build(scene: alloy.Scene, mode: str | None = "fast", profile: str = "standard", **kw):
    if mode is not None:                # None: never set (the profile decides)
        scene.rigid_step = mode
    return alloy.build(scene, device="cpu", profile=profile, **kw)


def test_fast_mode_steps_the_box_stack_once_per_period() -> None:
    """XB4-9's 3-cube column: "auto" stops at 2.5 ms on rk4 (never softer than the fixed step); fast takes the 10 ms
    period on implicitfast with step 2c's own gains (the ruling: fast mode keeps implicitfast), the entry tagged and
    WP3's 4-point box-box faces on, impratio as "auto"'s."""
    auto, fast = build(stack(), "auto"), build(stack())
    assert step_of(auto).dt == 2.5e-3 and not step_of(auto).fast and RS.FAST_EFFECT not in effects_of(auto)
    step = step_of(fast)
    assert step.dt == 1e-2 and step.fast and step.reason.startswith("fast mode")
    assert RS.FAST_EFFECT in effects_of(fast)
    assert mujoco_kwargs(fast).get(REDUCE_KEY) is True and REDUCE_KEY not in mujoco_kwargs(auto)
    assert mujoco_kwargs(fast)["impratio"] == mujoco_kwargs(auto)["impratio"] == 300.0
    assert any(r.startswith("mujoco integrator rk4") for r in auto.routes()["c0"]["reasons"])
    line = next(r for r in fast.routes()["c0"]["reasons"] if r.startswith("mujoco integrator"))
    assert line.startswith("mujoco integrator implicitfast") and "fast mode: implicitfast" in line, line
    assert "own gains at 10 ms" in line and IR.FAST_ROCKS in line


def test_profile_fast_means_fast_mode_and_says_so(caplog) -> None:
    """Ruling 09:18Z (A): profile="fast" on a scene that never set rigid_step; one build log line; sim.routes() and
    the report's warnings carry what it gives up."""
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        sim = build(stack(), None, profile="fast")
    assert step_of(sim).dt == 1e-2 and step_of(sim).fast
    assert sum("rigid fast mode on" in r.message for r in caplog.records) == 1
    assert RS.FAST_COST in sim.engine.parts.built.report.warnings
    assert all(sum(RS.FAST_COST in r for r in row["reasons"]) == 1 for row in sim.routes().values()
               if row["solver"] == "mujoco")


@pytest.mark.parametrize("mode", ["auto", "fixed", 0.0025])
def test_an_explicit_rigid_step_wins_over_the_profile(mode, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        sim = build(stack(), mode, profile="fast")
    assert not step_of(sim).fast and step_of(sim).dt == {"auto": 2.5e-3, "fixed": TODAY, 0.0025: 0.0025}[mode]
    assert not any("rigid fast mode on" in r.message for r in caplog.records)


def test_scene_json_round_trip_keeps_never_set_and_explicit_auto() -> None:
    """A default scene's JSON has no rigid_step and comes back never set (profile="fast" still applies, as in a
    copy for other world counts); an explicit "auto" is saved and still wins."""
    default = alloy.Scene.from_dict(stack().to_dict())
    assert "rigid_step" not in stack().to_dict() and API.is_default(default.rigid_step)
    assert default.rigid_step == "auto"
    assert step_of(alloy.build(default, worlds=2, device="cpu", profile="fast")).fast       # a from_dict copy inside
    s = stack()
    s.rigid_step = "auto"
    back = alloy.Scene.from_dict(s.to_dict())
    assert s.to_dict()["rigid_step"] == "auto" and not API.is_default(back.rigid_step)
    assert not step_of(alloy.build(back, device="cpu", profile="fast")).fast


@pytest.mark.parametrize("mode", [None, "auto"])
def test_processes_workers_follow_the_profile(mode, capfd) -> None:
    """processes= workers get the scene as JSON: never set -> fast mode there too, and one warning for the build
    (the first worker's); explicit "auto" -> neither."""
    s = stack(worlds=2)
    if mode is not None:
        s.rigid_step = mode
    with alloy.build(s, device="cpu", profile="fast", processes=2) as sim:
        reasons = sim.routes()["c0"]["reasons"]
        assert any(RS.FAST_COST in r for r in reasons) == (mode is None), reasons
    assert capfd.readouterr().err.count("rigid fast mode on") == (1 if mode is None else 0)


def robot_scene() -> alloy.Scene:
    s = stack(layers=1)
    s.add_robot("arm", mjcf=ARM, pos=(3.0, 0.0, 0.5))
    return s


def pusher_scene() -> alloy.Scene:
    s = stack(layers=1)
    s.add_box("pusher", size=(0.2, 0.2, 0.2), pos=(2.0, 0.0, 0.1), kinematic=True)
    return s


@pytest.mark.parametrize("scene", [robot_scene, pusher_scene], ids=["robot", "kinematic"])
def test_profile_fast_is_silent_where_it_cannot_apply(scene, caplog) -> None:
    """OSS-3's must-fix: a scene the fast mode cannot step keeps the fixed step under profile="fast" with no warning
    about the mode (neither "on" nor "ignored"): the user never set it."""
    with caplog.at_level(logging.INFO):
        sim = build(scene(), None, profile="fast")
    assert step_of(sim).dt == TODAY and not step_of(sim).fast
    assert RS.FAST_COST not in sim.engine.parts.built.report.warnings
    loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("fast" in m and "rigid" in m for m in loud), loud


def test_profile_standard_never_asks_for_fast() -> None:
    """The default stays: a never-set scene under profile="standard" asks for "auto"; its entry has no FAST_EFFECT."""
    s = stack()
    req = API.request(s, "standard", (), {})
    assert req.mode == "auto" and not req.from_profile and req.announce is None
    sim = build(s, None)
    assert step_of(sim).dt == 2.5e-3 and RS.FAST_EFFECT not in effects_of(sim)


@pytest.mark.parametrize("edge", [0.05, 0.3])
def test_small_bodies_step_finer_by_their_sag(edge: float) -> None:
    """Bodies at the stability cap sag about the same at any size, so a smaller cube gets a finer step: the cap reads
    the predicted rest sag against SAG_SHARE of the thinnest member, and never goes below "auto"'s step."""
    auto, fast = (step_of(build(stack(edge, 1000.0), m)) for m in ("auto", "fast"))
    assert auto.dt <= fast.dt <= 1e-2 and fast.fast and fast.reason.startswith("fast mode")
    if edge == 0.05:
        assert fast.dt == 5e-3 and dict(fast.caps)["sag"] == 5e-3, fast       # own gains: stiffer than rk4's


def test_the_creep_cap_stays_on() -> None:
    """Cap 6 holds A5.speed's slow-sliding model in fast mode as under "auto" (design change accepted 09:18Z): it
    refuses 10 cm cubes 10 ms (saturation speed x1.05 of A5's pass point), as the sag bound does."""
    fast = step_of(build(stack(0.1, 1000.0)))
    assert fast.dt == 5e-3 and dict(fast.caps)["creep"] == 5e-3 and "creep cap" in fast.reason, fast


@pytest.mark.parametrize("profile", ["precision", "debug"])
def test_precision_and_debug_profiles_keep_the_fixed_step(profile: str) -> None:
    sim = build(stack(), profile=profile)
    assert step_of(sim).dt == TODAY and not step_of(sim).fast and RS.FAST_EFFECT not in effects_of(sim)


def test_ineligible_scene_keeps_the_fixed_step() -> None:
    sim = build(pusher_scene())
    assert step_of(sim).dt == TODAY and not step_of(sim).fast and RS.FAST_EFFECT not in effects_of(sim)


def test_scene_json_keeps_the_mode() -> None:
    s = stack()
    s.rigid_step = "fast"
    assert alloy.Scene.from_dict(s.to_dict()).rigid_step == "fast"


def test_fast_stack_rests_within_its_sag_bound() -> None:
    """2 s at 10 ms: the column stands, no world unhealthy, and every cube sits within SAG_SHARE of its edge (the
    sum of the contacts under it) of its ideal rest height."""
    sim = build(stack())
    sim.run(2.0)
    st = sim.state(world=0)
    z = np.array([st[f"c{k}.pos"][2] for k in range(3)])
    err = np.abs(z - (0.5 + np.arange(3)))
    assert sim.check()["ok"] and err.max() <= RS.SAG_SHARE * 1.0, err




MU, G, SIZE = 0.5, 9.81, (0.4, 0.4, 0.1)
MASS = 1000.0 * float(np.prod(SIZE))                    # 16 kg


def pushed(share: float, mode: str = "fast", size: tuple[float, float, float] = SIZE):
    """A box (default 40 x 40 x 10 cm, 16 kg) resting on the ground (mu 0.5 on both), settled 0.5 s, then pushed along
    x at its COM by ``share`` x mu m g. A tall box rocks and lifts as it slides, under "auto" too:
    docs/known_gaps/rigid-sliding-box-hops.md (MuJoCo's soft friction, amplified by the integrator rule R2)."""
    s = alloy.Scene(dt=0.01)
    s.add_ground(friction=MU)
    s.add_box("blk", size=size, pos=(0.0, 0.0, 0.5 * size[2] + 1e-4), density=1000.0, friction=MU)
    sim = build(s, mode)
    sim.run(0.5)
    sim.apply_wrench("blk", force=(share * MU * 1000.0 * float(np.prod(size)) * G, 0.0, 0.0))
    return sim


def pitch_p2p(sim, seconds: float = 1.0) -> float:
    """Peak-to-peak pitch [deg] of the pushed box over ``seconds``, from 0.15 s on (the slide under way)."""
    tr = sim.run(seconds, record=["blk"]).world(0)
    k = tr["t"] > tr["t"][0] + 0.15
    return float(np.ptp(np.degrees(2.0 * np.arcsin(tr["blk.quat_wxyz"][k, 2]))))


def test_fast_box_holds_under_half_its_friction_limit() -> None:
    """Static friction at the fast step (creep cap and impratio as "auto"'s): pushed at 0.5 mu m g the box creeps
    under 0.1 mm over 1 s (measured 0.003 mm on the CPU; WP3's rows in scr/WP3.md, 0.005-0.03 mm/s)."""
    sim = pushed(0.5)
    assert step_of(sim).dt == 1e-2 and step_of(sim).fast
    sim.run(0.5)
    x0 = float(sim.state(["blk"], world=0)["blk.pos"][0])
    sim.run(1.0)
    assert abs(float(sim.state(["blk"], world=0)["blk.pos"][0]) - x0) < 1e-4 and sim.check()["ok"]


def test_fast_box_slides_at_the_coulomb_rate() -> None:
    """Sliding friction at the fast step: pushed at 1.5 mu m g the box accelerates at 0.5 mu g (2.45 m/s^2) within
    1 % (measured -0.02 %; "auto" at 2.5 ms -0.00 %), fitted over 0.85 s once it slides, and stays flat (pitch under
    1 deg peak to peak; measured 0.72 deg, against 0.24 under "auto")."""
    sim = pushed(1.5)
    tr = sim.run(1.0, record=["blk"]).world(0)
    t, p, q = tr["t"], tr["blk.pos"], tr["blk.quat_wxyz"]
    k = t > t[0] + 0.15
    acc = 2.0 * np.polyfit(t[k], p[k, 0], 2)[0]
    pitch = np.degrees(2.0 * np.arcsin(q[k, 2]))
    assert acc == pytest.approx(0.5 * MU * G, rel=0.01) and np.ptp(pitch) < 1.0 and sim.check()["ok"]


@pytest.mark.law
@pytest.mark.parametrize("route, reason, ceiling", [("warp", "fast mode: implicitfast", (3.0, 0.01)),
                                                    ("c", "fast mode keeps rk4", (15.0, 0.15))])
def test_a_cube_sliding_at_5_m_s_in_fast_mode_stays_under_its_ceiling(route, reason, ceiling, monkeypatch) -> None:
    """Fast mode's stated cost per CPU route (FAST_COST; audit AB-gaps-1's protocol, 2026-11-09): a 30 cm cube pushed
    at 1.5 mu m g for 2 s, to about 5 m/s, then kept sliding there for 2 s more by a push of mu m g.
    - MuJoCo Warp (implicitfast with WP3's friction-pull cap): pitch within 3 deg, lift within 10 mm, the Coulomb
      speed after the push within 2 % (measured: 0.59 deg, 1.1 mm).
    - The CPU's MuJoCo C route (no cap): the fast mode keeps rk4 there; the cube rocks and hops
      (rigid-sliding-box-hops.md) but does not topple: within 15 deg and 0.15 m (5.8 deg, 79 mm). Under implicitfast it
      toppled: 25 deg and 162 mm by the end of the push, 90 deg and 245 mm within the 2 s slide (the audit's row)."""
    if route == "c":
        monkeypatch.delenv("ALLOY_CPU_ROUTE", raising=False)
    sim = pushed(1.5, "fast", (0.3, 0.3, 0.3))
    assert any(reason in r for r in sim.routes()["blk"]["reasons"]), sim.routes()["blk"]["reasons"]
    first = sim.run(2.0, record=["blk"]).world(0)
    speed = float(np.linalg.norm(first["blk.vel"][-1, :3]))
    sim.apply_wrench("blk", force=(MU * 1000.0 * 0.3 ** 3 * G, 0.0, 0.0))
    held = sim.run(2.0, record=["blk"]).world(0)
    w, x, y, z = np.concatenate([first["blk.quat_wxyz"], held["blk.quat_wxyz"]]).T
    pitch = np.abs(np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))).max()
    lift = max(float(tr["blk.pos"][:, 2].max()) for tr in (first, held)) - 0.15
    assert pitch <= ceiling[0] and lift <= ceiling[1], (route, pitch, lift)
    assert (abs(speed - 2.0 * 0.5 * MU * G) < 0.02 * speed if route == "warp" else speed > 4.5) and sim.check()["ok"]


@pytest.mark.law
def test_fast_mode_keeps_rk4_on_the_mujoco_c_route(monkeypatch) -> None:
    """Audit AB-gaps-1: the CPU's default route (MuJoCo C) has no friction-pull cap, so the fast mode keeps rk4 there
    (the routes say so) at the fast step; XB4-9's stack still stands within 1 % of an edge (1.56 mm), at rest."""
    monkeypatch.delenv("ALLOY_CPU_ROUTE", raising=False)
    s = alloy.Scene(dt=0.01)
    s.add_ground(friction=1.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s.add_box(f"b{i}{j}{k}", size=(1.0,) * 3, pos=(2.0 * i, 2.0 * j, 0.5 + k + 1e-4 * k), density=10.0,
                          friction=1.0)
    sim = build(s)
    reasons = sim.routes()["b000"]["reasons"]
    assert type(sim.engine.stepper.solvers()[0]).__name__ == "MjcCoreSolver" and step_of(sim).dt == 1e-2
    assert any(r.startswith("mujoco integrator rk4") and "fast mode keeps rk4" in r for r in reasons), reasons
    sim.run(2.0)
    st = sim.state(world=0)
    err = [abs(float(st[f"b{i}{j}{k}.pos"][2]) - (0.5 + k)) for i in range(3) for j in range(3) for k in range(3)]
    speed = [float(np.linalg.norm(st[f"b{i}{j}{k}.vel"][:3])) for i in range(3) for j in range(3) for k in range(3)]
    assert max(err) <= RS.SAG_SHARE * 1.0 and max(speed) < 0.01 and sim.check()["ok"], (max(err), max(speed))


@pytest.mark.law
def test_a_resting_27_cube_stack_stands_in_fast_mode() -> None:
    """The ruling's resting-pile row on the CPU: XB4-9's 3 x 3 x 3 stack of 1 m cubes (27 bodies: R1's path, own gains
    at 10 ms) for 2 s: every cube within 1 % of its edge of its rest height, none moving."""
    s = alloy.Scene(dt=0.01)
    s.add_ground(friction=1.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s.add_box(f"b{i}{j}{k}", size=(1.0,) * 3, pos=(2.0 * i, 2.0 * j, 0.5 + k + 1e-4 * k), density=10.0,
                          friction=1.0)
    sim = build(s)
    assert step_of(sim).dt == 1e-2 and any("own gains at 10 ms" in r for r in sim.routes()["b000"]["reasons"])
    sim.run(2.0)
    st = sim.state(world=0)
    err = [abs(float(st[f"b{i}{j}{k}.pos"][2]) - (0.5 + k)) for i in range(3) for j in range(3) for k in range(3)]
    speed = [float(np.linalg.norm(st[f"b{i}{j}{k}.vel"][:3])) for i in range(3) for j in range(3) for k in range(3)]
    assert max(err) <= RS.SAG_SHARE * 1.0 and max(speed) < 0.01 and sim.check()["ok"], (max(err), max(speed))
