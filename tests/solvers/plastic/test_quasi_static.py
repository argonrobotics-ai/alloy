# SPDX-License-Identifier: Apache-2.0
"""GAP #6 slice 2b step 2: ``Scene.set_quasi_static_mass_scale`` (opt-in quasi-static mass scaling of a j2 solid on
MPM; physics ruling 2026-11-10; ``alloy.solvers.plastic.quasi_static``, ``alloy.api._quasi_static``).

Fast: the numbers, the argument checks, the refusals (a solid in the air, a solid that is not j2 on MPM, a share over
5 %), the warning over 1 %, the true weight at rest, and that a scene without the setting gets nothing of it. Slow
(run by hand; numbers in notes/wp7-dent/slice2/STEP2-CHECK.md; physics ruling 2026-11-11): brass with scale "auto"
reads its converged punch force; the run-time flag (QUASI_STATIC) stays off at the declared speed and at 10x (the
force within about 9 %), and is on for a tool striking at 100x and for a block pushed off its table."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api.errors import SceneError
from alloy.precision.types import HealthFlags
from alloy.solvers.plastic import quasi_static as qs
from tests.solvers.plastic.test_mpm_solid_reg_laws import BRASS, DT, GAP, LX, LZ, TOY_N, YIELD_RATIO, R, _card

BRASS_SHARE_2MM = {16000: 0.0234, 2913: 0.0100}      # v sqrt(E rho s) / yield at 2 mm/s


def _scene(tmp_path, material=None, z=LZ / 2, *, table=False, tool=False):
    import alloy.route.checks as ck  # noqa: F401  (the voxel is pinned by the callers' monkeypatch)

    (tmp_path / "materials").mkdir(parents=True, exist_ok=True)
    (tmp_path / "materials" / "reg_law_j2.yaml").write_text(_card(material or BRASS))
    s = alloy.Scene(dt=DT, roots=[str(tmp_path)])
    s.add_ground(friction=1.0)
    if table:
        s.add_box("table", size=(0.1, 0.1, 0.02), pos=(0, 0, 0.01), static=True)
    s.add_box("blk", size=(LX, LX, LZ), pos=(0, 0, z + (0.02 if table else 0.0)), material="reg_law_j2",
              card_extra={"route": {"route": "paste_mpm", "reason": "step 2 test"}})
    if tool:
        s.add_sphere("tool", R, pos=(0, 0, LZ + R + GAP), kinematic=True, friction=1e-4)
    return s


@pytest.fixture
def voxel3(monkeypatch):
    import alloy.route.checks as ck

    monkeypatch.setattr(ck, "mpm_voxel", lambda d, vmax: 0.003)


def test_the_numbers():
    e, rho, y = BRASS["youngs"], BRASS["density"], BRASS["yield"]
    assert qs.auto_scale(e, rho, y, 0.002) == 2913.0
    for s, share in BRASS_SHARE_2MM.items():
        assert qs.inertial_share(e, rho, y, s, 0.002) == pytest.approx(share, rel=3e-3)
    assert qs.auto_scale(3.0e6, 1076.0, 9.6e3, 0.002) == 1.0       # a soft card: never below 1
    assert qs.MOTION_MARGIN == 0.5 and qs.CONTACT_MARGIN == 12.0      # physics ruling 2026-11-11


def test_the_setting_is_checked_and_saved():
    s = alloy.Scene(dt=0.01)
    s.add_box("blk", size=(0.03, 0.03, 0.015), pos=(0, 0, 0.0075))
    for kw, field in (({"scale": 0.5, "tool_speed": 0.002}, "scale"), ({"scale": "big", "tool_speed": 0.002}, "scale"),
                      ({"tool_speed": 0.0}, "tool_speed"), ({"tool_speed": float("nan")}, "tool_speed")):
        with pytest.raises(SceneError) as e:
            s.set_quasi_static_mass_scale("blk", **kw)
        assert e.value.field == field
    with pytest.raises(SceneError, match="no object named 'nope'"):
        s.set_quasi_static_mass_scale("nope", tool_speed=0.002)
    s.set_quasi_static_mass_scale("blk", 4000, tool_speed=0.002)
    assert alloy.Scene.from_dict(s.to_dict()).quasi_static == {"blk": {"scale": 4000.0, "tool_speed": 0.002}}
    assert "quasi_static" not in alloy.Scene(dt=0.01).to_dict()


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_without_the_setting_the_build_is_as_before(tmp_path, voxel3):
    """Physics' condition: the default path gets nothing of it (no plug-in, no share check, no route note, true
    masses); and the new flag only adds its own bit to the DEGRADED mask."""
    from alloy.precision.ladder import DEGRADED_MASK

    sim = alloy.build(_scene(tmp_path), device="cpu")
    assert sim.engine.health._plugins.shares == []
    assert "quasi_static_weight" not in (getattr(getattr(sim.engine, "plugins", None), "plugins", None) or {})
    assert not any("quasi-static mass scaling x" in r for r in sim.routes()["blk"]["reasons"])
    m = sim.engine.model.particle_mass.numpy()
    assert float(m.sum()) == pytest.approx(LX * LX * LZ * BRASS["density"], rel=1e-3)
    # the mask before the flag (step 1's tree), less FAR_FROM_ORIGIN: a notice since TRAIN-FIX 4c (physics 2026-11-11)
    assert DEGRADED_MASK & ~int(HealthFlags.QUASI_STATIC) == 0xFF1660 & ~int(HealthFlags.FAR_FROM_ORIGIN)


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_a_solid_that_does_not_rest_on_a_support_is_refused(tmp_path, voxel3):
    s = _scene(tmp_path, z=0.05)
    s.set_quasi_static_mass_scale("blk", tool_speed=0.002)
    with pytest.raises(SceneError, match="rests on the ground or on a static box"):
        alloy.build(s, device="cpu")


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_on_a_static_box_it_builds_and_names_its_scale(tmp_path, voxel3):
    s = _scene(tmp_path, table=True)
    s.set_quasi_static_mass_scale("blk", tool_speed=0.002)
    sim = alloy.build(s, device="cpu")
    assert any("quasi-static mass scaling x2,913 (auto" in r for r in sim.routes()["blk"]["reasons"])
    assert [n for n, *_ in sim.engine.health._plugins.shares] == ["quasi-static mass scaling of blk: motion",
                                                                  "quasi-static mass scaling of blk: contact"]
    from alloy.integration.graph_budget import eager_scene_reason  # noqa: PLC0415

    assert eager_scene_reason(sim.built.report.scene_effects), "a j2 solid on MPM must run without CUDA graphs"


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_an_object_that_is_not_a_j2_solid_on_mpm_is_refused(voxel3):
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("dough", size=(0.03, 0.03, 0.015), pos=(0, 0, 0.0075), material="dough_wheat")
    s.set_quasi_static_mass_scale("dough", tool_speed=0.002)
    with pytest.raises(SceneError, match="takes a j2 solid on the MPM route"):
        alloy.build(s, device="cpu")


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_a_share_over_5_percent_is_refused_and_over_1_percent_warns(tmp_path, voxel3, monkeypatch):
    said: list[str] = []
    monkeypatch.setattr("alloy.api._say.warn", lambda fmt, *a: said.append(fmt % a))
    monkeypatch.setattr("alloy.route.router._STIFF_WARNED", set())
    s = _scene(tmp_path)
    s.set_quasi_static_mass_scale("blk", 1.0e6, tool_speed=0.002)             # share 18 %
    with pytest.raises(SceneError, match="over 5 %"):
        alloy.build(s, device="cpu")
    assert not any("inertial share" in x for x in said), said                 # refused, not warned
    s.set_quasi_static_mass_scale("blk", 16000, tool_speed=0.002)             # share 2.3 %
    alloy.build(s, device="cpu")
    assert any("inertial share of 2.3 %" in x for x in said), said


def _floor_over_weight(sim, true_weight: float, periods: int = 40) -> float:
    eng = sim.engine
    solver = next(x for x in eng.solvers if hasattr(x, "collect_collider_impulses"))
    sim.step(1)
    step_dt = DT / sim.physics_steps
    sim.step(periods - 1)
    imp, _, cid = solver.collect_collider_impulses(eng.state)
    imp, cid, cbi = imp.numpy(), cid.numpy(), solver.collider_body_index.numpy()
    floor = -float(imp[np.where(cid >= 0, cbi[np.maximum(cid, 0)], -2) < 0][:, 2].sum()) / step_dt
    return floor / true_weight


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_the_weight_stays_true_at_rest(tmp_path, voxel3):
    """x2,913 ("auto"): the floor carries the block's true weight, not 2,913 times it (the plug-in's upward force)."""
    s = _scene(tmp_path)
    s.set_quasi_static_mass_scale("blk", tool_speed=0.002)
    sim = alloy.build(s, device="cpu")
    weight = LX * LX * LZ * BRASS["density"] * 9.81
    assert float(sim.engine.model.particle_mass.numpy().sum()) == pytest.approx(weight / 9.81 * 2913, rel=1e-3)
    assert _floor_over_weight(sim, weight) == pytest.approx(1.0, abs=0.01)
    assert sim.check()["ok"]


PRESS = """<mujoco><compiler angle="radian"/><worldbody><body name="carriage" pos="0 0 0.03">
  <joint name="z" type="slide" axis="0 0 1" range="-0.02 0.02" damping="5"/>
  <geom type="sphere" size="0.005" pos="0 0 0" mass="0.2"/>
  </body></worldbody><actuator><position joint="z" kp="2000" kv="50"/></actuator></mujoco>"""


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_the_weight_stays_true_beside_a_robot(tmp_path, voxel3):
    """The plug-in's force reaches the MPM step with a MuJoCo robot in the scene too (measured 1.0007)."""
    s = _scene(tmp_path)
    s.add_robot("press", mjcf=PRESS, pos=(0, 0, 0.0))
    s.set_quasi_static_mass_scale("blk", tool_speed=0.002)
    sim = alloy.build(s, device="cpu")
    sim.set_targets({"z": 0.0})
    assert _floor_over_weight(sim, LX * LX * LZ * BRASS["density"] * 9.81, periods=60) == pytest.approx(1.0, abs=0.05)
    assert sim.check()["ok"]


def _press(tmp_path, speed: float) -> dict:
    """The small rig with scale "auto" declared for 2 mm/s, pressed at ``speed`` to 1 mm: the force over the
    converged reference, the periods flagged QUASI_STATIC and the largest share."""
    s = _scene(tmp_path, tool=True)
    s.set_quasi_static_mass_scale("blk", tool_speed=0.002)
    sim = alloy.build(s, device="cpu")
    eng, solver = sim.engine, sim.engine.solvers[0]
    plug = eng.plugins.plugins["quasi_static_weight"]
    body = int(np.where(np.asarray(eng.model.body_label).astype(str) == "tool")[0][0])
    sim.step(10)
    step_dt = DT / (sim.physics_steps // 10)

    def depth() -> float:
        return LZ - (float(sim.state(["tool"], world=0)["tool.pos"][2]) - R)

    sim.drive("tool", velocity=(0, 0, -max(speed, 0.01)), duration=10)
    while depth() < -0.8e-3:
        sim.step(1)
    sim.stop_drives()
    sim.drive("tool", velocity=(0, 0, -speed), duration=100)
    flagged, motion, contact = 0, [], []
    while depth() < 1.0e-3 - 1e-9:
        sim.step(1)
        motion.append(float(plug.motion[0].numpy()[0]))
        contact.append(float(plug.contact[0].numpy()[0]))
        flagged += "QUASI_STATIC" in sim.health()["worlds"][0]["flags"]
    imp, _, cid = solver.collect_collider_impulses(eng.state)
    imp, cid, cbi = imp.numpy(), cid.numpy(), solver.collider_body_index.numpy()
    force = float(imp[np.where(cid >= 0, cbi[np.maximum(cid, 0)], -2) == body][:, 2].sum()) / step_dt
    return {"force": force / (TOY_N * YIELD_RATIO), "flagged": flagged, "periods": len(motion),
            "motion_max": max(motion), "contact_max": max(contact), "ok": sim.check()["ok"]}


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_auto_brass_reads_its_converged_force_and_is_not_flagged(tmp_path, voxel3):
    """Brass, scale "auto" (x2,913, share 1 % at 2 mm/s) pressed at its declared 2 mm/s: 0.991 of the converged force
    at 1 mm (unscaled: 0.007), never flagged: the block's mean velocity stays far under half the declared speed, and the
    tool touches it at 1x."""
    r = _press(tmp_path, 0.002)
    print("LAW auto", r)
    assert r["force"] >= 0.99 and r["ok"], r
    assert r["flagged"] == 0 and r["motion_max"] < qs.MOTION_MARGIN / 3 and r["contact_max"] < 2.0, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_a_tool_ten_times_faster_than_declared_is_not_flagged_and_reads_within_6_percent(tmp_path, voxel3):
    """Physics ruling 2026-11-11: a tool up to 12x the declared speed is the user's statement, not flagged; at 10x
    (20 mm/s) the force at 1 mm reads within 6 % of the declared-speed force (resolved rig, both voxels: 0 to
    +5.8 %)."""
    slow, fast = _press(tmp_path, 0.002), _press(tmp_path, 0.02)
    print("LAW 10x", fast, "against", slow)
    assert fast["flagged"] == 0 and fast["contact_max"] < qs.CONTACT_MARGIN and fast["ok"], fast
    assert fast["force"] == pytest.approx(slow["force"], rel=0.06), (fast, slow)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_a_tool_just_past_the_contact_bound_is_flagged(tmp_path, voxel3):
    """Physics' edge row: a tool at 15x the declared speed (30 mm/s), just over ``CONTACT_MARGIN`` (12x), flags."""
    r = _press(tmp_path, 0.03)
    print("LAW 15x", r)
    assert qs.CONTACT_MARGIN < r["contact_max"] < 20.0 and r["flagged"] >= 1, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_a_tool_striking_at_100x_the_declared_speed_is_flagged(tmp_path, voxel3):
    """A tool at 0.2 m/s (100x the declared 2 mm/s) touches the block faster than ``CONTACT_MARGIN`` (12x): DEGRADED
    with QUASI_STATIC from its first contact."""
    r = _press(tmp_path, 0.2)
    print("LAW 100x", r)
    assert r["contact_max"] > qs.CONTACT_MARGIN and r["flagged"] >= 1, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_a_block_pushed_along_is_flagged(tmp_path, voxel3):
    """Physics ruling 2026-11-11 (1): pushed at its declared 20 mm/s ("auto" x29), the block as a whole moves at 1x the
    declared speed, over ``MOTION_MARGIN`` (0.5): DEGRADED with QUASI_STATIC while it is carried (its push force read
    5x the unscaled one at x2,913, notes/wp7-dent/slice2/STEP2-CHECK.md); still at rest before the push, it is not."""
    s = _scene(tmp_path)
    s.add_box("pusher", size=(0.01, 0.04, 0.012), pos=(-LX / 2 - 0.0055, 0, 0.0075), kinematic=True)
    s.set_quasi_static_mass_scale("blk", tool_speed=0.02)
    sim = alloy.build(s, device="cpu")
    plug = sim.engine.plugins.plugins["quasi_static_weight"]
    sim.step(10)
    assert "QUASI_STATIC" not in sim.health()["worlds"][0]["flags"]
    sim.drive("pusher", velocity=(0.02, 0, 0), duration=10)
    flagged, motion = 0, []
    for _ in range(40):                                     # 0.2 s: 4 mm of push
        sim.step(1)
        motion.append(float(plug.motion[0].numpy()[0]))
        flagged += "QUASI_STATIC" in sim.health()["worlds"][0]["flags"]
    print("LAW pushed", {"flagged": flagged, "motion_end": motion[-1], "s": plug.solids[0].s})
    assert motion[-1] == pytest.approx(1.0, abs=0.1) and flagged >= 30, (flagged, motion)
    assert sim.check()["ok"]                                # DEGRADED: listed, not a failed run


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_law_a_block_pushed_off_its_table_is_flagged_as_it_falls(tmp_path, voxel3):
    """AUDIT-STACK's push-off (audit/stack/work/review/wp7qs/qs.py): "auto" x29 at 20 mm/s on a static table's edge,
    pushed off at its declared speed. It falls s times slowly (g / s) and is flagged all the way down: the mean velocity
    passes 1.4x, 3.6x and 6.2x the declared speed as it drops (probe_qs/pushoff.json)."""
    (tmp_path / "materials").mkdir(parents=True, exist_ok=True)
    (tmp_path / "materials" / "reg_law_j2.yaml").write_text(_card(BRASS))
    s = alloy.Scene(dt=DT, roots=[str(tmp_path)])
    s.add_ground(friction=1.0)
    s.add_box("table", size=(0.1, 0.1, 0.05), pos=(0, 0, 0.025), static=True)
    s.add_box("blk", size=(LX, LX, LZ), pos=(0.035, 0, 0.05 + LZ / 2), material="reg_law_j2",
              card_extra={"route": {"route": "paste_mpm", "reason": "step 2 test"}})
    s.add_box("pusher", size=(0.01, 0.04, 0.012), pos=(0.0145, 0, 0.0575), kinematic=True)
    s.set_quasi_static_mass_scale("blk", tool_speed=0.02)
    sim = alloy.build(s, device="cpu")
    plug = sim.engine.plugins.plugins["quasi_static_weight"]
    sim.step(10)
    z0 = float(sim.state(world=0)["blk.points"][:, 2].min())
    sim.drive("pusher", velocity=(0.02, 0, 0), duration=30.0)
    falling = []
    for _ in range(500):                                    # up to 2.5 s
        sim.step(1)
        drop = z0 - float(sim.state(world=0)["blk.points"][:, 2].min())
        if drop > 2e-3:
            falling.append(("QUASI_STATIC" in sim.health()["worlds"][0]["flags"], float(plug.motion[0].numpy()[0])))
        if drop > 10e-3:
            break
    print("LAW push-off", {"falling periods": len(falling), "motion_end": falling[-1][1] if falling else None})
    assert len(falling) >= 3 and all(f for f, _ in falling), falling
    assert falling[-1][1] > 1.0, falling
