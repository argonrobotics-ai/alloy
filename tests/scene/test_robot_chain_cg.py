# SPDX-License-Identifier: Apache-2.0
"""SAMPLE-CLASSIC C1: a robot file that is a long chain of light bodies steps on CG (`alloy.scene.robot_chain`), since
MuJoCo Warp's float32 Newton solver goes non-finite at its first contact; short chains and the shipped robots keep
Newton, and a file's own `<option solver=...>` stands. The law: a 10-bead chain dropped 1 cm onto a bench stays finite
on MuJoCo Warp (the CPU's Warp route), and goes INVALID on Newton (CPU)."""

from __future__ import annotations

import pytest

import alloy
from alloy.cards.library import default_root
from alloy.scene import robot_chain as RC
from alloy.scene.entries import ROBOT_CG_EFFECT

BENCH_TOP = 0.30


def chain_mjcf(beads: int, option: str = "") -> str:
    """The sample's steel ball chain ([internal note] samples/classic_chain), straight, 1 cm above the bench: 8 mm beads
    10 mm apart, bead 0 free, each next bead on a ball joint at the midpoint."""
    r, pitch, z = 0.004, 0.010, BENCH_TOP + 0.004 + 0.01
    x0 = -0.05 - beads * pitch
    body = f'<body name="bead0" pos="{x0:.4f} 0 {z:.4f}"><freejoint name="free"/>'
    geom = f'<geom name="ball{{i}}" type="sphere" size="{r}" density="7850" friction="0.4 0.005 0.0001"/>'
    parts = [body + geom.format(i=0)]
    for i in range(1, beads):
        parts.append(f'<body name="bead{i}" pos="{pitch} 0 0"><joint name="j{i}" type="ball" pos="{-pitch / 2} 0 0"/>'
                     + geom.format(i=i))
    opt = f"<option {option}/>" if option else ""
    return (f'<mujoco model="chain"><compiler angle="radian"/>{opt}<worldbody>{"".join(parts)}'
            f'{"</body>" * beads}</worldbody></mujoco>')


def bench_scene(beads: int, option: str = "") -> alloy.Scene:
    scene = alloy.Scene(dt=1 / 300)
    scene.add_ground(z=0.0)
    scene.add_box("bench", size=(0.9, 0.5, BENCH_TOP), pos=(-0.45, 0.0, BENCH_TOP / 2), static=True)
    scene.add_robot("chain", mjcf=chain_mjcf(beads, option), contact_gap=0.0005)
    return scene


def mujoco_solver(sim) -> str:
    entry = next(e for e in sim.engine.built.entries if e.name == "mujoco")
    return str(entry.solver_kwargs.get("solver", "newton"))


def test_bound_follows_the_impratio():
    assert RC.bound(300.0) == pytest.approx(RC.LAM_AT_FILL)
    assert RC.bound(100.0) == pytest.approx(RC.LAM_AT_FILL / 3.0)
    assert RC.bound(1.0) == RC.bound(RC.IMP_FLOOR) == pytest.approx(RC.LAM_AT_FILL * RC.IMP_FLOOR / 300.0)


@pytest.mark.parametrize(("beads", "below"), [(5, False), (6, True), (10, True), (30, True)])
def test_the_screen_puts_6_beads_and_more_below_the_bound(tmp_path, beads, below):
    path = tmp_path / f"chain{beads}.xml"
    path.write_text(chain_mjcf(beads))
    s = RC.screen(str(path))
    assert s is not None and s.nv == 6 + 3 * (beads - 1)
    assert (s.lam < RC.bound(300.0)) is below, s          # measured: 5 beads 1.22e-3, 6 beads 6.1e-4


@pytest.mark.parametrize("robot", ["emb_e1_fr3", "emb_e2_panda", "emb_e3_ur10e_2f85", "emb_e4_fr3_bimanual",
                                   "emb_e5_fr3_duo", "emb_e6_g1", "emb_e7_h1_2", "emb_e8_go2", "emb_e9_leap"])
def test_shipped_robots_stay_well_above_the_bound(robot):
    s = RC.screen(str(default_root() / "robots" / robot / "scene.xml"))
    assert s is not None and s.lam > 10.0 * RC.bound(300.0), s       # G1 the lowest: 0.0119


def test_a_long_chain_steps_on_cg_and_a_short_one_on_newton():
    sim = alloy.build(bench_scene(10), device="cpu")
    assert mujoco_solver(sim) == "cg"
    assert ROBOT_CG_EFFECT in sim.engine.built.report.scene_effects
    line = next(x for x in sim.engine.built.report.entry_options if x.startswith("mujoco solver: CG for robot"))
    assert "chain" in line and RC.GAP_PAGE in line
    short = alloy.build(bench_scene(5), device="cpu")
    assert mujoco_solver(short) == "newton"
    assert ROBOT_CG_EFFECT not in short.engine.built.report.scene_effects


def test_a_files_own_solver_stands():
    assert mujoco_solver(alloy.build(bench_scene(10, 'solver="Newton"'), device="cpu")) == "newton"


def test_a_low_impratio_in_the_file_moves_the_bound():
    """10 beads (lambda_min 9.9e-5) at impratio 1 step finite on Newton (MuJoCo Warp, measured): they keep it."""
    assert mujoco_solver(alloy.build(bench_scene(10, 'impratio="1"'), device="cpu")) == "newton"


def _landing_flags(scene) -> tuple[list[str], bool]:
    sim = alloy.build(scene, device="cpu", contact_capacity=1024)
    sim.step(36)                                                   # 0.12 s; the chain lands at 0.047 s
    world = sim.health()["worlds"][0]
    return list(world.get("seen_flags") or []), sim.check()["ok"]


def test_a_dropped_10_bead_chain_stays_finite_on_mujoco_warp(monkeypatch):
    """The law: on the CPU's Warp route (MuJoCo Warp, float32, the GPU's code path) the chain lands finite on CG; the
    same chain held on Newton by its file goes non-finite at the landing (the failure the rule avoids). 7 s warm."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    flags, ok = _landing_flags(bench_scene(10))
    assert ok and not any(f.startswith("NONFINITE") for f in flags), flags
    flags, ok = _landing_flags(bench_scene(10, 'solver="Newton"'))
    assert not ok and "NONFINITE_Q" in flags, flags


@pytest.mark.gpu
def test_a_dropped_30_bead_chain_stays_finite_on_a_gpu():
    """The sample's own failing row (30 beads, dropped 1 cm; INVALID at the landing on an L4 before the rule)."""
    sim = alloy.build(bench_scene(30), device="cuda:0", contact_capacity=1024)
    assert mujoco_solver(sim) == "cg"
    sim.step(60)                                                   # 0.2 s
    flags = list(sim.health()["worlds"][0].get("seen_flags") or [])
    print(f"GPU, 30 beads on CG: flags {flags}, ok {sim.check()['ok']}")
    assert sim.check()["ok"] and not any(f.startswith("NONFINITE") for f in flags), flags


def test_a_robot_built_in_code_is_not_screened():
    """A robot built in code (``RobotRef`` path "": the UQD halves of ``alloy.tasks.dc_cool_uqd``) has no file to
    screen; the screen once resolved its empty path to the card roots and every such scene raised a CardError."""
    from alloy.cards.types import RobotRef
    from alloy.route.types import Route, RouteDecision
    from alloy.scene.template import TemplateObject

    dec = RouteDecision(route=Route.RIGID, entry="mujoco", numbers={}, margin=float("inf"), near_boundary=False,
                        reasons=("test",))
    pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    in_code = TemplateObject("half", None, dec, RobotRef("", "mjcf", pose, "half"), "robot:half")
    from_file = TemplateObject("arm", None, dec, RobotRef("arm.xml", "mjcf", pose, "arm"), "robot:arm")
    assert not RC._screened(in_code) and RC._screened(from_file)
