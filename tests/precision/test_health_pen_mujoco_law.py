# SPDX-License-Identifier: Apache-2.0
"""GAP-E1 / ERR-1 law: rigid and robot (MuJoCo-route) penetration raises PEN_WARN / PEN_FAIL through the public API.

A 0.3 m box rests on a light 5 cm cube on the ground (ERR-1's repro). Its mass sets how deep the cube sinks into
the ground and how deep the box sinks into the cube (soft contacts: both depths come out equal).

  L1  bands by load (rule: PEN_WARN at once past 1 mm; PEN_FAIL when the depth stays past 1 cm for 0.1 s of
      simulated time, or passes 2 cm at any check; coordinator ruling 2026-11-10 00:08Z):
      5 kg, 0.35 mm -> OK; 80 kg, 2.9 mm -> PEN_WARN, DEGRADED, check ok, still moving; 200 kg (the audit's case:
      rests 7.0 mm deep, past 1 cm only for a few ms of the landing) -> PEN_WARN, DEGRADED, not frozen; 500 kg and
      2,500 kg on a 2.5 g cube (the audit's 13.1 and 30.4 mm) -> PEN_FAIL, INVALID, LATCHED, check not ok with a
      problem naming the world, ``traj.meta["health"]["ok"]`` False, the world frozen.
  L1b a bounce is not a resting overlap: 5 cm balls dropped 0.3 / 0.6 / 0.9 m with restitution 0.6 (the shipped
      example basics/11) dip up to 10.1 mm for a few ms: PEN_WARN while deep, never INVALID, OK at rest.
  L2  health's ``max_penetration_m`` equals the depth measured from body heights alone (cube z against its rest
      height; box and cube z for the other contact) within TOL_M; a PEN flag is raised iff that depth is past a
      threshold, away from the thresholds by MARGIN (a dip between two checks is not asserted).
  L3  an arm servo pressed into a table: PEN_FAIL, INVALID, frozen; ``profile="precision"`` raises
      ``SimulationError`` from run / step in the period the world fails (AGENTS.md: it "checks every step and
      raises"), once per failed world; ``sim.health()`` and ``sim.check()`` still read afterwards.
  L4  the same on the MuJoCo C route (``mjc``, the user default) and on Warp (``warp``, the suite's); a GPU case.
Calibration: ``CheckPolicy.native_pen_flags`` has no public switch, so masses were read from runs below the failing
depth (steady depth is about linear in mass up to 130 kg: 0.07 mm/kg 5 kg to 100 kg, then saturates near 8 mm).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pytest

import alloy
from alloy.api.health import INFO_FLAGS
from alloy.core.defaults import PEN_LIMIT_M, PEN_S1_M

CUBE, BOX = 0.05, 0.3
GAP = 2e-4                                  # start gap: no START_PEN
MASS = {"ok": 5.0, "warn": 80.0, "rest": 200.0, "fail": 500.0, "crush": 2500.0}
LIGHT = {"crush": 0.0025}                   # [kg] the cube under the load (default 0.05)
WIDE = {"crush": 0.5}                       # [m] the load's edge (default BOX; density cap 25,000 kg/m^3)
TOL_M = 1.5e-4                              # health vs geometry, steady state
MARGIN = 0.5                                # no flag below MARGIN * limit, flag above limit / MARGIN
ROUTES = ("mjc", "warp")
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20000" kv="200"/></actuator></mujoco>"""


@dataclass
class Run:
    sim: Any
    traj: Any
    health: dict[str, Any]
    depth_m: np.ndarray        # (frames, worlds): deepest of the two geometric contacts, from body heights alone


def _geometry_depth(traj: Any, box: float = BOX) -> np.ndarray:
    zc, zb = traj["light.pos"][..., 2], traj["heavy.pos"][..., 2]
    ground = CUBE / 2 - zc
    box_on_cube = (zc + CUBE / 2) - (zb - box / 2)
    return np.maximum(ground, box_on_cube)


def _scene(mass: float, worlds: int, light: float = 0.05, box: float = BOX) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("light", size=(CUBE,) * 3, pos=(0, 0, CUBE / 2 + GAP), mass=light)
    scene.add_box("heavy", size=(box,) * 3, pos=(0, 0, CUBE + 2 * GAP + box / 2), mass=mass)
    return scene


def _build(scene: alloy.Scene, route: str, device: str, **kw: Any) -> Any:
    """Build on ``route`` (the CPU route is read at build; the suite forces warp, so set it for this call only)."""
    old = os.environ.get("ALLOY_CPU_ROUTE")
    os.environ["ALLOY_CPU_ROUTE"] = route
    try:
        return alloy.build(scene, device=device, **kw)
    finally:
        if old is None:
            os.environ.pop("ALLOY_CPU_ROUTE", None)
        else:
            os.environ["ALLOY_CPU_ROUTE"] = old


def _run_box(band: str, route: str, device: str = "cpu", worlds: int = 1) -> Run:
    box = WIDE.get(band, BOX)
    sim = _build(_scene(MASS[band], worlds, LIGHT.get(band, 0.05), box), route, device)
    traj = sim.run(1.0, record=["light", "heavy"])
    return Run(sim, traj, sim.health(), _geometry_depth(traj, box))


@lru_cache(maxsize=None)
def _cached(band: str, route: str) -> Run:
    return _run_box(band, route)


def _frozen(sim: Any, key: str) -> bool:
    """True when the world's state is identical across 5 more control periods."""
    before = sim.state(world=0)[key].copy()
    sim.step(5)
    return bool(np.array_equal(before, sim.state(world=0)[key]))


def _assert_band(run: Run, band: str, world: int = 0) -> None:
    w = run.health["worlds"][world]
    flags = set(w["flags"])
    if band == "ok":
        assert not flags & {"PEN_WARN", "PEN_FAIL"}, w
        assert w["severity"] == "OK" or (w["severity"] == "DEGRADED" and flags <= INFO_FLAGS), w   # GPU: DET_DOWNGRADED
    elif band in ("warn", "rest"):
        assert w["severity"] == "DEGRADED" and "PEN_WARN" in flags and "PEN_FAIL" not in flags, w
    else:
        assert w["severity"] == "INVALID" and {"PEN_WARN", "PEN_FAIL", "LATCHED"} <= flags, w


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l1_light_load_is_ok(route):
    run = _cached("ok", route)
    _assert_band(run, "ok")
    assert run.sim.check()["ok"] and run.traj.meta["health"]["ok"]


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l1_moderate_load_warns_and_degrades(route):
    run = _cached("warn", route)
    _assert_band(run, "warn")
    assert run.depth_m.max() < PEN_S1_M, "calibration: the landing transient must stay under the failing depth"
    check = run.sim.check()
    assert check["ok"] and run.traj.meta["health"]["ok"], check
    before = run.sim.state(world=0)["heavy.pos"].copy()          # a DEGRADED world is not frozen: it still moves
    run.sim.apply_wrench("heavy", force=(0, 0, -1.0e3))
    run.sim.step(20)
    assert not np.array_equal(before, run.sim.state(world=0)["heavy.pos"])


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l1_a_landing_past_1_cm_that_rests_shallower_only_warns(route):
    """The audit's 200 kg box: 7.0 mm deep at rest (flagged PEN_WARN), past 1 cm only while it lands."""
    run = _cached("rest", route)
    _assert_band(run, "rest")
    assert 5.0e-3 < run.depth_m[-1, 0] < PEN_S1_M and run.sim.check()["ok"]
    assert not _frozen(run.sim, "heavy.vel") or float(np.abs(run.sim.state(world=0)["heavy.vel"]).max()) < 1e-3


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("band", ["fail", "crush"])
def test_l1_deep_load_fails_latches_and_freezes(route, band):
    run = _cached(band, route)
    _assert_band(run, "fail")
    check = run.sim.check()
    assert not check["ok"] and run.traj.meta["health"]["ok"] is False
    assert any("world 0" in p["what"] for p in check["problems"]), check["problems"]
    w = run.health["worlds"][0]                      # frozen at its last good state: the peak is the failing depth
    assert w["peak_penetration_m"] > PEN_S1_M and w["peak_penetration_m"] >= w["max_penetration_m"], w
    assert _frozen(run.sim, "light.pos") and _frozen(run.sim, "heavy.pos"), "INVALID world must be frozen"


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("band", ["ok", "warn"])
def test_l2_health_depth_matches_geometry(route, band):
    run = _cached(band, route)
    geometric = float(run.depth_m[-1, 0])
    measured = run.health["worlds"][0]["max_penetration_m"]
    assert abs(measured - geometric) < TOL_M, f"health {measured:.6f} m vs geometry {geometric:.6f} m"


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("band", ["ok", "warn", "rest", "fail"])
def test_l2_flag_iff_geometry_past_threshold(route, band):
    run = _cached(band, route)
    flags = set(run.health["worlds"][0]["flags"])
    steady, deepest = float(run.depth_m[-1, 0]), float(run.depth_m.max())
    if deepest < MARGIN * PEN_LIMIT_M:
        assert "PEN_WARN" not in flags, (steady, flags)
    if steady > PEN_LIMIT_M / MARGIN:
        assert "PEN_WARN" in flags, (steady, deepest, flags)
    if steady < PEN_S1_M:                         # at rest under 1 cm (frames every 10 ms: a landing dip may pass it)
        assert "PEN_FAIL" not in flags, (steady, deepest, flags)
    else:
        assert "PEN_FAIL" in flags, (steady, deepest, flags)


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l1b_a_bounce_warns_and_never_fails(route):
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.05, pos=np.array([[0, 0, 0.3], [0, 0, 0.6], [0, 0, 0.9]]), restitution=0.6)
    sim = _build(scene, route, "cpu")
    peak, seen = np.zeros(3), set()
    for _ in range(100):
        sim.step(1)
        worlds = sim.health()["worlds"]
        peak = np.maximum(peak, [w["max_penetration_m"] for w in worlds])
        seen |= {(w["severity"], flag) for w in worlds for flag in w["flags"] or [""]}
    assert peak[2] > 5.0e-3, "calibration: the 0.9 m drop dips several mm at the bounce"
    assert ("DEGRADED", "PEN_WARN") in seen and not any(sev == "INVALID" for sev, _ in seen), seen
    assert sim.check()["ok"] and sim.health()["invalid_worlds"] == []
    rest = sim.health()["worlds"][2]                 # at rest the current depth is small: the peak keeps the dip
    assert rest["max_penetration_m"] < 1.0e-3 and rest["peak_penetration_m"] >= peak[2] > 5.0e-3, rest


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l1b_a_dropped_cube_warns_while_it_lands_and_never_fails(route):
    """Audit API-13: a 10 cm, 5 kg cube dropped 0.4 m dips past 1 cm as it lands (11.74 mm in the audit's run).
    Every period-end reading is PEN_WARN exactly when it is deeper than 1 mm; the world is never INVALID, rests OK,
    and the peak keeps the landing depth."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.45), mass=5.0)
    sim = _build(scene, route, "cpu")
    for _ in range(100):
        sim.step(1)
        w = sim.health()["worlds"][0]
        assert w["severity"] != "INVALID", w
        assert ("PEN_WARN" in w["flags"]) == (w["max_penetration_m"] > PEN_LIMIT_M), w
    assert w["severity"] == "OK" and sim.check()["ok"], w
    assert w["peak_penetration_m"] > 5.0e-3 > w["max_penetration_m"], w


@pytest.mark.law
def test_precision_names_the_world_that_just_failed_and_settle_restarts_the_peak():
    """OSS-3 review: a later failure names its own world, not the first INVALID world on record; ``sim.settle``
    makes the settled state the start, so the peak depth restarts there."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.05, pos=np.array([[0, 0, 0.9], [0, 0, 0.06]]), restitution=0.6)
    sim = _build(scene, "warp", "cpu", profile="precision")
    sim.settle(0.6)                                  # world 0 bounces (a dip of about 1 cm) while settling
    rows = sim.health()["worlds"]
    assert all(r["peak_penetration_m"] == pytest.approx(r["max_penetration_m"], abs=1e-9) for r in rows), rows
    sim.set_pose("ball", pos=np.array([[0, 0, 30.0], [0, 0, 0.06]]))    # world 0 leaves the workspace
    with pytest.raises(alloy.SimulationError, match="world 0"):
        sim.step(1)
    sim.set_pose("ball", pos=np.array([[0, 0, 30.0], [0, 0, 30.0]]))
    with pytest.raises(alloy.SimulationError, match="world 1") as err:
        sim.step(1)
    assert "world 0" not in err.value.message.split("world 1")[0], err.value.message


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l1c_at_a_coarse_5_ms_step_a_landing_goes_on_and_a_body_held_through_a_plate_freezes(route):
    """Ruling 2026-11-10 07:08Z: the hard limit counts checks, so it scales with the rigid step. At a pinned 5 ms
    step: 1 m cubes (10 kg/m3) dropped 0.5 m onto the ground and onto a 1 cm static plate are never INVALID and rest
    OK (a static shape sets no size limit: the cube's 10 cm holds); a 5 cm box pressed through a 5 mm static plate by
    a held 2 kN load is INVALID with PEN_FAIL."""
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(1.6, 1.6, 0.01), pos=(3.0, 0, 0.3), static=True)
    scene.add_box("on_ground", size=(1.0,) * 3, pos=(0, 0, 0.5 + 0.5), density=10.0)
    scene.add_box("on_plate", size=(1.0,) * 3, pos=(3.0, 0, 0.305 + 0.5 + 0.5), density=10.0)
    scene.rigid_step = 0.005
    sim = _build(scene, route, "cpu")
    for _ in range(100):
        sim.step(1)
        assert sim.health()["worlds"][0]["severity"] != "INVALID", sim.health()["worlds"][0]
    assert sim.physics_steps == 200, sim.physics_steps                        # the step really was 5 ms
    assert sim.health()["worlds"][0]["severity"] == "OK" and sim.check()["ok"], sim.health()["worlds"][0]
    held = alloy.Scene(worlds=1, dt=0.01)
    held.add_ground()
    held.add_box("plate", size=(0.4, 0.4, 0.005), pos=(0, 0, 0.3), static=True)
    held.add_box("box", size=(0.05,) * 3, pos=(0, 0, 0.3025 + 0.025), mass=0.2)
    held.rigid_step = 0.005
    sim = _build(held, route, "cpu")
    sim.apply_wrench("box", force=(0, 0, -2000.0))
    sim.step(30)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and "PEN_FAIL" in w["flags"], w


def _arm_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    scene.add_box("table", size=(0.2, 0.3, 0.5), pos=(0.25, 0, 0.25), static=True)   # top z = 0.5
    return scene


def _press_arm(sim: Any) -> Any:
    traj = sim.run(1.0, record=["arm/shoulder"], targets={"shoulder": (0.0, 1.0)})
    sim.step(50)
    return traj


def _capsule_depth_m(q: float) -> float:
    xs = np.linspace(0, 0.3, 301)
    under = xs * np.cos(q) >= 0.15            # the part of the capsule over the table
    return float(0.5 - (0.6 - xs * np.sin(q) - 0.02)[under].min())


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(("kp", "kv", "thick", "z"), [(200, 10, 0.005, 0.42), (5000, 100, 0.005, 0.42),
                                                      (200, 10, 0.02, 0.45), (200, 10, 0.05, 0.45)])
def test_l3_an_arm_driven_through_a_static_plate_fails_before_it_is_through(route, kp, kv, thick, z):
    """Audit ERR2R-5: the AGENTS.md arm, one target jump to 1.8 rad, a static plate in the link's way. A link pushed
    through a plate stays past its hard depth for 4-10 checks (5-12.5 ms; a ball landing on a thin plate 1-3
    checks, a tunnel dip 1): PEN_FAIL at the fourth check freezes the world, at kp 200 before the link is through
    (q under 1 rad). The kp 5000 jump (FAST_CONTACT, 8.7 m/s) is TUNNEL's since GAP-E2 round 3 (points inside each
    robot link): INVALID at check 7, 19.7 mm in, before the fourth deep check; PEN_FAIL caught it 4 checks in, most of
    the way through, before that round. Either flag freezes the world (the 07:26Z ruling needs both with ERR-1)."""
    arm = ARM.replace('kp="20000" kv="200"', f'kp="{kp}" kv="{kv}"')
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=arm, pos=(0, 0, 0.5))
    scene.add_box("plate", size=(0.4, 0.4, thick), pos=(0.25, 0, z), static=True)
    sim = _build(scene, route, "cpu")
    sim.set_targets({"shoulder": 1.8})
    q = []
    for _ in range(60):
        sim.step(1)
        q.append(float(sim.state(world=0)["arm/shoulder.q"]))
    w = sim.health()["worlds"][0]
    caught = {"PEN_FAIL"} if kp == 200 else {"PEN_FAIL", "TUNNEL"}
    assert w["severity"] == "INVALID" and caught & set(w["flags"]), (w, max(q))
    assert not sim.check()["ok"]
    if kp == 200:                                    # a servo-speed push freezes before the link is through
        assert max(q) < 1.0, (max(q), w)


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l3_arm_pressed_into_table_fails_and_freezes(route):
    sim = _build(_arm_scene(), route, "cpu")
    traj = _press_arm(sim)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and {"PEN_WARN", "PEN_FAIL", "LATCHED"} <= set(w["flags"]), w
    assert not sim.check()["ok"] and traj.meta["health"]["ok"] is False
    assert _frozen(sim, "arm/shoulder.q"), "joint q must stop changing"
    geometric = _capsule_depth_m(float(sim.state(world=0)["arm/shoulder.q"]))
    assert abs(w["max_penetration_m"] - geometric) < 1e-3, (w["max_penetration_m"], geometric)


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_l3_precision_profile_raises_while_stepping(route):
    """AGENTS.md section 8: the precision profile checks every step and raises when a world becomes INVALID. The
    raise comes from run / step (a caller who never calls check() is stopped), once per world that failed; health
    and check stay readable afterwards and name the world."""
    sim = _build(_arm_scene(), route, "cpu", profile="precision")
    with pytest.raises(alloy.SimulationError) as err:
        _press_arm(sim)
    assert "PEN_FAIL" in str(err.value) and "profile='standard'" in err.value.fix
    assert sim.t < 0.5, "raised in the period the world failed (t 0.355 s), not at the end of the run"
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and "PEN_FAIL" in w["flags"], w
    check = sim.check()                              # reads, does not raise
    assert not check["ok"] and any("world 0" in p["what"] for p in check["problems"]), check
    assert "steps" in sim.report()                   # reads, does not raise
    sim.step(2)                                      # already reported: no second raise for the same world
    sim.reset()
    with pytest.raises(alloy.SimulationError):       # a reset world that fails again raises again
        _press_arm(sim)


@pytest.mark.law
@pytest.mark.gpu
@pytest.mark.parametrize("band", ["ok", "warn", "rest", "fail"])
def test_l4_gpu_four_worlds_per_band(band):
    """Four identical worlds per load band on cuda:0 (one mass per sim: per-world masses are not an API field)."""
    wp = pytest.importorskip("warp")
    if not wp.get_cuda_device_count():
        pytest.skip("no CUDA device")
    run = _run_box(band, "warp", device="cuda:0", worlds=4)
    for world in range(4):
        _assert_band(run, band, world)
        if band in ("ok", "warn"):
            geometric = float(run.depth_m[-1, world])
            assert abs(run.health["worlds"][world]["max_penetration_m"] - geometric) < 3 * TOL_M
    assert run.sim.check()["ok"] is (band != "fail")
