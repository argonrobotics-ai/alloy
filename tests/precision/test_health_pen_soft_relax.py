# SPDX-License-Identifier: Apache-2.0
"""Laws for contacts made soft on purpose (GAP-E1, 0.1.1; physics ruling D, 2026-11-10 23:41Z).

Menagerie ANYmal C gives its feet ``solimp="0.015 1 0.03"``: standing still they rest 17.2 mm deep (plain MuJoCo C
reads the same), past the PEN rule's 1 cm for 0.1 s, so health froze it INVALID. ``scene.relax_contact_depth(name)``
judges the contacts of ``name`` on their depth past their own solimp width (30 mm there), or past a stated ``depth``;
every other contact keeps the rule. Measured on the Menagerie file itself (CPU, MuJoCo C route, 2 worlds, 1 s; not
shipped, so not a test here): as shipped INVALID (PEN_FAIL + START_PEN), relaxed OK with ``check()["ok"]``, 17.2 mm
at rest and 19.1 mm peak still reported. ``DOG`` below has the same feet (sphere r 3 cm, priority 1, condim 6, that
solimp) on hinged legs held by position servos: it rests 21.0 mm deep on both CPU routes. A file that keeps MuJoCo's
default solimp (the library Go2) is judged as before, fallen or not.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.precision.health import HealthMonitor
from alloy.precision.health_pen import relax_soft_contacts
from alloy.precision.types import CheckPolicy, HealthFlags

sys.path.insert(0, str(Path(__file__).parent))
from test_health_pen_measured import DT, _boxes  # noqa: E402

ROUTES = ("mjc", "warp")
FOOT = 'priority="1" solimp="0.015 1 0.03" condim="6"'


def _leg(n: str, x: float, y: float) -> str:
    return (f'<body name="{n}_thigh" pos="{x} {y} -0.05"><joint name="{n}_hfe" type="hinge" axis="0 1 0" damping="1"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" mass="2"/>'
            f'<body name="{n}_shank" pos="0 0 -0.25"><joint name="{n}_kfe" type="hinge" axis="0 1 0" damping="1"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.1"/>'
            f'<geom name="{n}_foot" type="sphere" size="0.03" pos="0 0 -0.25" {FOOT} mass="0.02"/></body></body>')


_LEGS = (("lf", 0.3, 0.15), ("rf", 0.3, -0.15), ("lh", -0.3, 0.15), ("rh", -0.3, -0.15))
DOG = ('<mujoco><compiler angle="radian"/><worldbody><body name="base" pos="0 0 0.58"><freejoint/>'
       '<geom type="box" size="0.3 0.12 0.06" mass="30"/>' + "".join(_leg(*leg) for leg in _LEGS) +
       '</body></worldbody><actuator>' +
       "".join(f'<position joint="{n}_{j}" kp="300" kv="20"/>' for n, _, _ in _LEGS for j in ("hfe", "kfe")) +
       '</actuator></mujoco>')


def _build(scene: alloy.Scene, route: str) -> Any:
    """Build on the CPU ``route`` (read at build; the suite forces warp, so set it for this call only)."""
    old = os.environ.get("ALLOY_CPU_ROUTE")
    os.environ["ALLOY_CPU_ROUTE"] = route
    try:
        return alloy.build(scene, device="cpu")
    finally:
        if old is None:
            os.environ.pop("ALLOY_CPU_ROUTE", None)
        else:
            os.environ["ALLOY_CPU_ROUTE"] = old


def _dog_scene(z: float = 0.0, worlds: int = 1) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_robot("dog", mjcf=DOG, pos=(0, 0, z))
    return scene


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_a_robot_standing_on_feet_soft_on_purpose_reads_ok_once_relaxed(route: str) -> None:
    """Unrelaxed the dog rests past 1 cm and is frozen INVALID (the ANYmal C report); with
    ``relax_contact_depth("dog")`` it stands 1 s OK, ``check()`` ok, the true depth (10-30 mm) still reported."""
    sim = _build(_dog_scene(), route)
    sim.step(100)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and "PEN_FAIL" in w["flags"], w
    scene = _dog_scene(worlds=2)
    scene.relax_contact_depth("dog")
    sim = _build(scene, route)
    sim.step(100)
    for w in sim.health()["worlds"]:
        assert w["severity"] == "OK" and not w["flags"] and not w["seen_flags"], w
        assert 0.010 < w["max_penetration_m"] < 0.030, w
    assert sim.check()["ok"], sim.check()


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_a_real_sink_still_fails_beside_a_relaxed_robot(route: str) -> None:
    """The relax is per name: a second, identical soft-footed robot in the same world, not relaxed, resting 21 mm
    deep, freezes the world INVALID with PEN_FAIL (both relaxed: OK); the dog relaxed to a stated 2 mm only, resting
    19 mm past it, is INVALID too."""
    for both in (False, True):
        scene = _dog_scene()
        scene.add_robot("cat", mjcf=DOG, pos=(2.0, 0, 0))
        scene.relax_contact_depth("dog")
        if both:
            scene.relax_contact_depth("cat")
        sim = _build(scene, route)
        sim.step(100)
        w = sim.health()["worlds"][0]
        assert (w["severity"], "PEN_FAIL" in w["flags"]) == (("OK", False) if both else ("INVALID", True)), w
    scene = _dog_scene()
    scene.relax_contact_depth("dog", depth=0.002)
    sim = _build(scene, route)
    sim.step(100)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and "PEN_FAIL" in w["flags"], w


@pytest.mark.law
def test_the_rule_judges_the_depth_past_the_allowance() -> None:
    """Engine level (``relax_soft_contacts``), the measured rig of ``test_health_pen_measured``: a 10 cm box held 2 cm
    in the ground in four worlds, four checks 30 ms apart. Unmarked, and marked with the default solimp (its 1 mm
    width gives no allowance): PEN_FAIL. Marked 5 mm: 15 mm past it, still PEN_FAIL (the 0.1 s dwell). Marked 25 mm:
    OK. Every world reports the true 20 mm."""
    model, s0 = _boxes("cpu", press=(0.02,) * 4)
    s1, control = model.state(), model.control()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), solvers=[solver])
    boxes = np.flatnonzero(model.shape_body.numpy() >= 0)
    relax_soft_contacts(monitor, [boxes[1]])
    relax_soft_contacts(monitor, [boxes[2]], depth=0.005)
    relax_soft_contacts(monitor, [boxes[3]], depth=0.025)
    for k in range(4):
        solver.step(s0, s1, control, None, DT)
        monitor.update(k, s0, s1, None, dt=0.03)
    report = monitor.poll()
    flags = [int(f) for f in report.flags]
    assert [bool(f & int(HealthFlags.PEN_FAIL)) for f in flags] == [True, True, True, False]
    assert [bool(f & int(HealthFlags.PEN_WARN)) for f in flags] == [True, True, True, False]
    np.testing.assert_allclose(report.max_pen_m, 0.02, atol=2.0e-5)


@pytest.mark.law
def test_a_start_overlap_within_the_allowance_raises_no_start_pen() -> None:
    """The dog placed 8 mm into the ground (ANYmal C's file starts its feet 10.5 mm in): START_PEN unrelaxed, none
    relaxed."""
    flags = []
    for relax in (False, True):
        scene = _dog_scene(z=-0.008)
        if relax:
            scene.relax_contact_depth("dog")
        sim = _build(scene, "mjc")
        sim.step(1)
        flags.append(sim.health()["worlds"][0]["flags"])
    assert "START_PEN" in flags[0] and "START_PEN" not in flags[1], flags


def test_relax_contact_depth_checks_its_arguments_and_round_trips() -> None:
    scene = _dog_scene()
    for bad in (-0.001, float("nan"), "deep"):
        with pytest.raises(alloy.SceneError):
            scene.relax_contact_depth("dog", depth=bad)
    with pytest.raises(alloy.SceneError):
        scene.relax_contact_depth("")
    scene.relax_contact_depth("dog")
    scene.relax_contact_depth("dog/lf_shank", depth=0.004)
    again = alloy.Scene.from_dict(scene.to_dict())
    assert again.contact_relax == {"dog": None, "dog/lf_shank": 0.004}
    assert "contact_relax" not in _dog_scene().to_dict()
    bad = _dog_scene()
    bad.relax_contact_depth("cat")
    with pytest.raises(alloy.UnknownNameError):
        _build(bad, "mjc")


def test_scene_save_warns_that_its_cards_drop_the_relax(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """OSS-3's review: cards carry no relax, so ``scene.save()`` says so (as it does for clamps and sysid)."""
    scene = _dog_scene()
    scene.relax_contact_depth("dog")
    with caplog.at_level("WARNING", logger="alloy.api"):
        scene.save(tmp_path / "relaxed")
    assert any("relax_contact_depth for dog" in r.getMessage() for r in caplog.records), caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="alloy.api"):
        _dog_scene().save(tmp_path / "plain")
    assert not any("relax_contact_depth" in r.getMessage() for r in caplog.records), caplog.text


def _go2_on_its_back(base_z: float, relax: bool, route: str) -> Any:
    """The library Go2 as ``alloy.envs.loco`` builds it (no obstacle), on its back, base ``base_z`` m up, 1.5 s."""
    from alloy.cards.library import default_root  # noqa: PLC0415
    from alloy.envs.mjcf import without_bodies  # noqa: PLC0415

    text = without_bodies(default_root() / "robots" / "emb_e8_go2" / "scene.xml", ["obstacle"]).read_text()
    text = re.sub(r'<body name="base" pos="[^"]*" quat="[^"]*"',
                  f'<body name="base" pos="0 0 {base_z}" quat="0 1 0 0"', text, count=1)
    scene = alloy.Scene(dt=0.02)
    scene.add_robot("go2", mjcf=text)
    if relax:
        scene.relax_contact_depth("go2")
    sim = _build(scene, route)
    sim.step(75)
    return sim


@pytest.mark.law
@pytest.mark.parametrize("route", ROUTES)
def test_a_fallen_go2_at_rest_reads_ok_and_a_hard_landing_still_fails(route: str) -> None:
    """Fallen Go2 worlds of loco training read INVALID (SAMPLE-LEARN; physics 2026-11-11). Lying on its back (dropped
    9 cm) the library Go2 rests 0.12 mm deep and reads OK, relaxed or not: its file keeps MuJoCo's default solimp
    (1 mm wide), so the relax changes nothing for it. Dropped onto its back from 1 m, its trunk sinks 26.8 mm (plain
    MuJoCo C 3.17, same file and 1.25 ms step: 29.8 mm, 34 ms past 2 cm): INVALID with PEN_FAIL, relaxed or not."""
    for relax in (False, True):
        sim = _go2_on_its_back(0.15, relax, route)
        lying = sim.health()["worlds"][0]
        assert lying["severity"] == "OK" and not lying["flags"] and lying["max_penetration_m"] < 0.001, lying
        assert sim.check()["ok"], sim.check()
        landed = _go2_on_its_back(1.0, relax, route).health()["worlds"][0]
        assert landed["severity"] == "INVALID" and "PEN_FAIL" in landed["flags"], landed
        assert landed["peak_penetration_m"] > 0.020, landed


@pytest.mark.gpu
@pytest.mark.law
def test_relaxed_and_unrelaxed_on_a_gpu_under_graph_capture() -> None:
    """The same on MuJoCo Warp on a GPU (the step captured in a CUDA graph), 4 worlds: unrelaxed INVALID with
    PEN_FAIL, relaxed OK with the true depth (10-30 mm) reported."""
    pytest.importorskip("warp").init()
    import warp as wp  # noqa: PLC0415

    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")
    for relax in (False, True):
        scene = _dog_scene(worlds=4)
        if relax:
            scene.relax_contact_depth("dog")
        sim = alloy.build(scene, device="cuda:0")
        sim.step(100)
        rows = sim.health()["worlds"]
        if relax:
            assert all(w["severity"] == "OK" and 0.010 < w["max_penetration_m"] < 0.030 for w in rows), rows
            assert sim.check()["ok"], sim.check()
        else:
            assert all(w["severity"] == "INVALID" and "PEN_FAIL" in w["flags"] for w in rows), rows
