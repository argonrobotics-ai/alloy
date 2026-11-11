# SPDX-License-Identifier: Apache-2.0
"""PHYSA-1 law (CPU): a rigid body on Newton's contact pipeline comes to rest and never gains height.

A scene with an ``add_mesh(collision="sdf" | "mesh")`` shape builds with Newton's contact pipeline (``alloy.build``'s
``contacts="auto"``), which makes the contacts once per step. Free bodies stepped on rk4 there (F6) saw every contact
at the step's first distance through rk4's four stages, so the contact spring was integrated explicitly and each bounce
added energy: a 10 cm, 0.5 kg, 12-triangle sdf cube dropped 1 cm rose to 61 mm and kept bouncing 49-57 mm for 3 s with
health OK (audit PHYSA-1, rc8; an ``add_box`` under ``contacts="newton"`` did the same, 61.7 mm). The MuJoCo factory
(`alloy.scene.builtin_plugins`) now runs such an entry on implicitfast.

Law (pre-registered, coordinator 2026-11-09 22:45Z), over 3 s on the CPU:
- the body never rises above its drop height (0.1 mm allowed);
- it ends within 1 mm of its rest height (the documented contact accuracy of these shapes);
- over the last second its height moves less than 0.1 mm and its speed stays under SPEED_MAX;
- ``sim.check()`` is ok.

Rows (CPU, main 264484a50 -> with the fix; zmax / end height / range over the last second, mm): sdf cube on the ground
61.0 (bouncing) -> 60.00 / 49.987 / 0.024; mesh cube on the ground -> 60.00 / 49.987 / 0.024; sdf and mesh cube on a
static box (rest 150) -> 160.00 / 149.993 / 0.014; a 4 cm box on a static sdf or mesh cube (rest 120; thrown to 162
before) -> 130.00 / 119.998 / 0.027; a 4 cm ball on a static sdf slab (rest 120; INVALID, ENERGY_JUMP before) -> 150.00
/ 120.003 / 0.011. One L4 (europe-west2, 2026-11-09): the same heights within 0.02 mm, ranges 0.015-0.048 mm, none
above its drop; main there 61.72 mm (cube) and 161.65 mm (box on the sdf cube). What is left is a vibration of
0.01-0.03 mm at 5-18 mm/s on the pair table's contact damping (known gap rigid-sdf-mesh-contact-limits).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.scene import integrator_route as IR

A = 0.05                       # the cube's half edge (m)
RISE_MM, REST_MM, RANGE_MM = 0.1, 1.0, 0.1
SPEED_MAX = 0.03               # m/s over the last second (measured 0.005-0.018)
RK4 = 1                        # mujoco_warp IntegratorType.RK4
FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def _box_mesh(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def _scene(mode: str, case: str) -> tuple[alloy.Scene, str, float, float]:
    """(scene, the body to watch, its rest height, its drop height) for one case of the law."""
    cube = _box_mesh((-A, -A, -A), (A, A, A))
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    if case == "under_a_box":                  # the mesh is the static support
        s.add_mesh("m", vertices=cube, faces=FACES, collision=mode, pos=(0, 0, A), static=True)
        s.add_box("b", size=(0.04,) * 3, pos=(0, 0, 0.13), mass=0.1)
        return s, "b", 0.12, 0.13
    if case == "ball_on_a_slab":               # audit AB-fairness-14
        s.add_mesh("slab", vertices=_box_mesh((-0.1, -0.1, 0.0), (0.1, 0.1, 0.1)), faces=FACES, collision=mode,
                   static=True)
        s.add_sphere("ball", 0.02, pos=(0, 0, 0.15), mass=0.05)
        return s, "ball", 0.12, 0.15
    top = 0.0
    if case == "on_a_box":
        s.add_box("table", size=(0.5, 0.5, 0.1), pos=(0, 0, 0.05), static=True)
        top = 0.1
    if mode == "box":                          # a primitive on the same pipeline
        s.add_box("m", size=(2 * A,) * 3, pos=(0, 0, top + A + 0.01), mass=0.5)
    else:
        s.add_mesh("m", vertices=cube, faces=FACES, collision=mode, pos=(0, 0, top + A + 0.01), mass=0.5)
    return s, "m", top + A, top + A + 0.01


def _integrator(sim) -> int:
    return int(sim.engine.solvers[0].mjw_model.opt.integrator)


def _once_per_step(sim) -> list[int]:
    """The integrators of the MuJoCo solvers whose contacts are made once per step (Newton's pipeline)."""
    return [int(s.mjw_model.opt.integrator) for s in sim.engine.solvers
            if hasattr(s, "mjw_model") and not s._use_mujoco_contacts]


def test_an_entry_on_contacts_made_once_per_step_is_not_multi_stage(monkeypatch):
    """The mechanism, on the route: Newton's pipeline makes the contacts once per step, so no entry on it runs rk4,
    whose stages would see them frozen. The same box on MuJoCo's own contacts (rebuilt at each stage) keeps rk4."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    native = alloy.build(_scene("box", "ground")[0], device="cpu")
    newton = alloy.build(_scene("box", "ground")[0], device="cpu", contacts="newton")
    assert _integrator(native) == RK4 and _once_per_step(native) == []
    assert _once_per_step(newton) and RK4 not in _once_per_step(newton)
    assert any("integrator rk4" in r for r in native.routes()["m"]["reasons"])
    assert not any("integrator rk4" in r for r in newton.routes()["m"]["reasons"])      # routes say what runs
    native.close()
    newton.close()


def test_newtons_pipeline_keeps_rk4_off():
    """PHYSA-1: a build on Newton's pipeline (contacts once per step; alloy.build picks it for add_mesh sdf / mesh
    shapes) gets no rk4 from the rule, which says why and names the shapes."""
    from types import SimpleNamespace as NS

    def obj(name: str, collision: str):
        return NS(name=name, obj=NS(card=NS(shape=NS(collision=collision), motion="dynamic")),
                  decision=NS(entry="mujoco", scene_effects=()), ranges=NS(joint=[]))

    for mode in IR.NEWTON_COLLISIONS:
        choice = IR.choose_integrator(None, [obj("cup", mode)], precision=False, noslip=False, newton_contacts=True)
        assert not choice.rk4 and not choice.r1 and "cup" in choice.reason and "once per step" in choice.reason
    plain = IR.choose_integrator(None, [obj("box", "")], precision=False, noslip=False, newton_contacts=True)
    assert not plain.rk4 and "once per step" in plain.reason                 # contacts="newton" without such a shape


def test_newtons_pipeline_keeps_the_fixed_rigid_step():
    """Contact rows on Newton's pipeline take the pair table's gains, not the native gains the coarser rigid steps are
    sized for, so the scene keeps the fixed step, a static sdf / mesh support included. Fixed step against the step the
    planner picked (CPU, 5 s): a 5 kg sdf cube 0.12 mm / 15 mm/s against 1.03 mm / 87 mm/s at 5 ms; 20 kg 0.18 mm /
    8 mm/s against 2.48 mm / 116 mm/s and PEN_WARN at 10 ms."""
    from types import SimpleNamespace as NS

    from alloy.scene import rigid_step as RS

    def obj(name: str, collision: str, motion: str):
        return NS(name=name, obj=NS(card=NS(shape=NS(collision=collision), motion=motion)), decision=NS())

    for mode in IR.NEWTON_COLLISIONS:
        for motion in ("dynamic", "static"):
            why = RS.why_not(None, [obj("part", mode, motion)], NS(), RS.RigidStepRequest(newton_contacts=True))
            assert why is not None and "part" in why and "fixed step" in why, (mode, motion, why)
    assert RS.RigidStepRequest().newton_contacts is False and RS.why_not(None, [], NS()) is None


def test_only_a_build_on_newtons_pipeline_loses_rk4_and_the_auto_step():
    """The rule follows the build's contact pipeline, not the card: ``add_object`` of a card whose collision is sdf
    (23 shipped objects: the mug, bolts, nuts, bins ...) stays on MuJoCo's own contacts, rebuilt at each rk4 stage, so
    it keeps rk4 and the auto rigid step; the same scene with an ``add_mesh(collision="sdf")`` shape goes to Newton's
    pipeline and loses both. (105c33f6c keyed on the card: the estimate's bin rows, timed on rk4, read 2.8-5x off.)"""
    def facts(scene: alloy.Scene) -> tuple[bool, str, bool]:
        est = alloy.estimate(scene, worlds=1, seconds=0.1, device="cpu")
        return (est["rigid_step"]["eligible"], est["rigid_step"]["reason"],
                any("integrator rk4" in r for r in est["routes"]["mug"]["reasons"]))

    def scene() -> alloy.Scene:
        s = alloy.Scene(dt=0.01)
        s.add_ground()
        s.add_object("mug", "mug_stoneware", pos=(0.0, 0.0, 0.05))
        return s

    eligible, reason, rk4 = facts(scene())
    assert eligible and rk4 and "Newton" not in reason, reason
    sim = alloy.build(scene(), device="cpu")              # and the build runs what its routes name
    assert _integrator(sim) == RK4 and _once_per_step(sim) == []
    assert any("integrator rk4" in r for r in sim.routes()["mug"]["reasons"])
    sim.close()
    concave = scene()
    concave.add_mesh("slab", vertices=_box_mesh((0.2, -0.1, 0.0), (0.4, 0.1, 0.1)), faces=FACES, collision="sdf",
                     static=True)
    eligible, reason, rk4 = facts(concave)
    assert not eligible and not rk4 and "Newton's contact pipeline" in reason and "slab" in reason, reason


CASES = [("sdf", "ground"), ("mesh", "ground"), ("box", "ground"), ("sdf", "on_a_box"), ("mesh", "on_a_box"),
         ("sdf", "under_a_box"), ("mesh", "under_a_box"), ("sdf", "ball_on_a_slab")]


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode,case", CASES, ids=[f"{m}-{c}" for m, c in CASES])
def test_a_body_on_newtons_contacts_comes_to_rest(mode, case):
    scene, name, rest, drop = _scene(mode, case)
    sim = alloy.build(scene, device="cpu", contacts="newton" if mode == "box" else "auto")
    assert _once_per_step(sim) and RK4 not in _once_per_step(sim)
    assert not any("integrator rk4" in r for r in sim.routes()[name]["reasons"]), sim.routes()[name]["reasons"]
    tr = sim.run(3.0, record=[name]).world(0)
    z, speed = 1e3 * tr[f"{name}.pos"][:, 2], np.linalg.norm(tr[f"{name}.vel"], axis=1)
    print(f"{mode} {case}: zmax {z.max():.3f} end {z[-1]:.3f} range {np.ptp(z[-100:]):.4f} mm "
          f"speed {speed[-100:].max():.4f} m/s")
    assert z.max() <= 1e3 * drop + RISE_MM, z.max()
    assert abs(z[-1] - 1e3 * rest) <= REST_MM, z[-1]
    assert np.ptp(z[-100:]) <= RANGE_MM and speed[-100:].max() <= SPEED_MAX, (np.ptp(z[-100:]), speed[-100:].max())
    assert sim.check()["ok"], sim.check()["problems"]
    sim.close()
