# SPDX-License-Identifier: Apache-2.0
"""Holes as geometry (drilling v2): the union-of-stages volume, where a new stage starts, the sourced work table,
the hole records, and the drill's multi-hole table (CPU, light)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.drilling import Drill
from alloy.plugins.drilling.plugin import Layer
from alloy.plugins.drilling import laws as dl
from alloy.plugins.errors import PluginError
from alloy.plugins.holes import Hole, Stage, WORK_MATERIALS, work_material
from alloy.plugins.holes.work import PERCUSSIVE
from alloy.plugins.holes.geometry import stage_start_np, union_volume_np

T = math.tan(math.radians(59.0))          # a 118 deg point


def _numeric(radii, bottoms, span, n=4000):
    """Midpoint-rule volume of the union of drill-point solids (cone, then cylinder) clipped to [0, span]."""
    u = (np.arange(n) + 0.5) * span / n
    rho = np.zeros(n)
    for r, b in zip(radii, bottoms, strict=True):
        rho = np.maximum(rho, np.clip((b - u) * T, 0.0, r))
    return float(np.sum(math.pi * rho**2) * span / n)


@pytest.mark.parametrize("radii, bottoms, span", [
    ([0.003], [0.001], 0.01),                       # the point only, entering
    ([0.003], [0.006], 0.01),                       # blind, full diameter
    ([0.003], [0.02], 0.004),                       # through
    ([0.0015, 0.003], [0.02, 0.003], 0.003),        # a through pilot, the drill half way
    ([0.0015, 0.003], [0.002, 0.004], 0.01),        # a blind pilot overtaken by the drill
    ([0.003, 0.0015], [0.004, 0.009], 0.01),        # a narrow drill deeper than a wide one
])
def test_the_union_volume_is_the_drill_point_solids_union(radii, bottoms, span):
    exact = union_volume_np(radii, bottoms, span, T)
    assert exact == pytest.approx(_numeric(radii, bottoms, span), rel=2e-4)


def test_a_through_hole_and_an_enlarged_pilot_are_exact():
    assert union_volume_np([0.003], [0.02], 0.004, T) == pytest.approx(math.pi * 0.003**2 * 0.004, rel=1e-14)
    both = union_volume_np([0.0015, 0.003], [0.02, 0.02], 0.004, T)
    assert both == pytest.approx(math.pi * 0.003**2 * 0.004, rel=1e-14)       # the pilot's volume is not doubled


def test_a_new_stage_starts_where_its_lips_meet_material():
    assert stage_start_np([], [], 0.003, T) == 0.0                              # a fresh hole: at the face
    assert stage_start_np([0.0015], [0.02], 0.003, T) == pytest.approx(0.0015 / T)   # over a through pilot
    assert stage_start_np([0.0015], [0.0005], 0.003, T) == pytest.approx(0.0005)     # a pilot shallower than that
    assert stage_start_np([0.004], [0.01], 0.003, T) == pytest.approx(0.01)          # inside a wider hole


def test_the_work_table_is_sourced_and_names_its_rows():
    assert work_material("aluminium").kd_thrust == 7000.0
    assert work_material("steel_200bhn").kd_thrust == work_material("steel_200bhn").kd_torque == 24000.0
    assert work_material("stainless_316").kd_torque == 24000.0 < work_material("stainless_316").kd_thrust
    assert all(m.quote and ("Book-of-Spindles" in m.source or m.name == "pine_southern")
               for m in WORK_MATERIALS.values())
    assert all(m.quote and m.source.count("https://") == 1 for m in PERCUSSIVE.values())
    with pytest.raises(PluginError, match="steel_200bhn"):
        work_material("steel")


def test_a_layer_with_an_unknown_work_material_is_refused():
    with pytest.raises(PluginError):
        Drill(layers=({"workpiece": "plate", "work": "unobtainium"},))
    with pytest.raises(PluginError):
        Drill(layers=({"workpiece": "plate", "density": -1.0},))
    with pytest.raises(ValueError):
        Layer("plate", thickness=0.0)


def test_a_layer_uses_its_rows_kd():
    steel = Drill(layers=(Layer("plate", work="steel_200bhn", density=7850.0),))
    alu = Drill()
    assert float(steel.thrust_law(1e-4)) / float(alu.thrust_law(1e-4)) == pytest.approx(24000.0 / 7000.0)


def test_hole_records_read_depth_through_and_tilt():
    tilt = math.radians(2.0)
    h = Hole(0, np.zeros(3), np.array([0.0, math.sin(tilt), -math.cos(tilt)]), (Stage(0.003, 0.02),
                                                                               Stage(0.006, 0.004)), 0.01, T)
    assert h.tilt_deg == pytest.approx(2.0)
    assert h.span == pytest.approx(0.01 / math.cos(tilt), rel=1e-6)
    assert h.through_at(0.003) and not h.through_at(0.006)
    assert h.full_depth(0.006) == pytest.approx(0.004 - 0.003 / T)
    assert h.diameter == 0.006


def test_the_config_survives_dicts_and_copies():
    from alloy.plugins.harness.check import fresh

    d = Drill(layers=(Layer("cover", work="steel_200bhn", density=7850.0), Layer("backer")))
    copy = fresh(d, {"max_holes": 4})
    assert copy.config.layers == d.config.layers and copy.config.max_holes == 4


def test_two_holes_are_two_records():
    """Drill, retract, move 20 mm, drill again: two records in the table, each with its own entry and depth."""
    from tests.plugins.test_drilling import _feeding_sim, _move_head, _steps

    drill = Drill()
    sim = _feeding_sim(drill)
    _steps(sim, 30)
    _move_head(sim, 0.02, gap=dl.GAP)
    _steps(sim, 30)
    holes = drill.holes()
    assert len(holes) == 2
    assert np.allclose(holes[0].entry[:2], (0.0, 0.0), atol=1e-4)
    assert np.allclose(holes[1].entry[:2], (0.02, 0.0), atol=1e-4)
    assert holes[0].stages[0].depth == pytest.approx(holes[1].stages[0].depth, rel=0.02)
    assert all(h.tilt_deg < 0.05 and not h.through for h in holes)


def test_a_full_hole_table_holds_the_drill_and_cuts_nothing(caplog):
    """max_holes = 1: a second hole 20 mm away is not booked. The drill is held as by a face there (it pushes, it
    removes nothing), the table is flagged (the kernel prints one line per world when it fills) and read() warns."""
    from tests.plugins.test_drilling import _feeding_sim, _hole_mass, _move_head, _steps

    drill = Drill(max_holes=1)
    sim = _feeding_sim(drill)
    _steps(sim, 30)
    first = float(drill.read()["depth"][0])
    _move_head(sim, 0.02, gap=dl.GAP)
    _steps(sim, 30)
    r = drill.read()
    assert int(r["holes"][0]) == 1 and int(r["full"][0]) == 1
    assert float(r["removed"][0]) == pytest.approx(_hole_mass(drill, (first,)), rel=1e-6)
    assert float(r["thrust"][0]) > 100.0                                     # pressed on the face, not cutting
    assert float(r["depth"][0]) == pytest.approx(first, rel=1e-6)
    assert "hole table filled up in worlds [0]" in caplog.text


def _seed(drill, entry, stages, layer=0):
    """One hole at face-frame ``entry`` along -z with ``stages`` [(diameter, bottom)], in every world."""
    H, S = int(drill.config.max_holes), int(drill.config.max_stages)
    slot = layer * H
    arrays = {k: getattr(drill, k).numpy().copy() for k in ("hole_n", "hole_entry", "hole_axis", "stage_n",
                                                           "stage_d", "stage_b")}
    arrays["hole_n"][:, layer] = 1
    arrays["hole_entry"][:, slot] = entry
    arrays["hole_axis"][:, slot] = (0.0, 0.0, -1.0)
    arrays["stage_n"][:, slot] = len(stages)
    for k, (d, b) in enumerate(stages):
        arrays["stage_d"][:, slot * S + k], arrays["stage_b"][:, slot * S + k] = d, b
    for k, v in arrays.items():
        getattr(drill, k).assign(v)


def test_a_small_drill_inside_a_wider_hole_cuts_nothing_and_takes_no_stage():
    """Review probe: a 4 mm drill fed down a through hole of 6 and 8 mm stages (max_stages = 2) passes free: no stage
    of its own, no thrust, nothing removed (it was held at the face as a full table, with 1.4 kN)."""
    from tests.plugins.test_drilling import _feeding_sim, _steps

    drill = Drill(diameter=0.004, max_stages=2)
    sim = _feeding_sim(drill)
    _seed(drill, (0.0, 0.0), [(0.006, 0.05), (0.008, 0.05)])
    peak = 0.0
    for _ in range(20):
        _steps(sim, 1)
        peak = max(peak, float(drill.read()["thrust"][0]))
    r = drill.read()
    assert peak == 0.0 and float(r["removed"][0]) == 0.0
    assert int(r["stage_n"][0, 0]) == 2 and int(r["holes"][0]) == 1 and int(r["full"][0]) == 0


def test_a_small_drill_off_the_centre_of_a_wider_hole_stays_in_it():
    """Review probe: a 4 mm drill 2.5 mm off the axis of an 8 mm through hole (more than its own radius, inside the
    hole's) is in that hole: no new hole, nothing removed."""
    from tests.plugins.test_drilling import _feeding_sim, _steps

    drill = Drill(diameter=0.004)
    sim = _feeding_sim(drill, x0=0.0025)
    _seed(drill, (0.0, 0.0), [(0.008, 0.05)])
    _steps(sim, 20)
    r = drill.read()
    assert int(r["holes"][0]) == 1 and float(r["removed"][0]) == 0.0


def test_no_chisel_term_gives_finite_forces():
    """Review probe: chisel_j = 0 leaves no edge law at the first contact; the shares fall back to equal ones."""
    from tests.plugins.test_drilling import _feeding_sim, _steps

    drill = Drill(chisel_j=0.0)
    sim = _feeding_sim(drill)
    _steps(sim, 10)
    assert np.isfinite(sim.state.body_q.numpy()).all()
    assert float(drill.read()["removed"][0]) > 0.0


@pytest.mark.parametrize("bottom", [0.05, 0.005])
def test_a_wider_drill_in_a_full_hole_is_held_where_it_meets_material(bottom):
    """Re-review probe: a 10 mm drill into a hole of 3, 6 and 8 mm stages (max_stages = 3; through, or 5 mm blind) has
    no stage left. It is held where its point meets uncut material (4 mm / tan 59 deg = 2.4 mm in), not past the far
    face: it pushes, cuts nothing, and the table is flagged."""
    from tests.plugins.test_drilling import _feeding_sim, _steps

    drill = Drill(diameter=0.010)
    sim = _feeding_sim(drill)
    _seed(drill, (0.0, 0.0), [(0.003, bottom), (0.006, bottom), (0.008, bottom)])
    _steps(sim, 40)
    r = drill.read()
    tip = float(sim.state.body_q.numpy()[0, 2]) + float(drill.config.tip[2])
    depth = 0.03 - tip                                                     # the 30 mm block's face at z = 30 mm
    assert float(r["removed"][0]) == 0.0 and int(r["full"][0]) == 1 and int(r["stage_n"][0, 0]) == 3
    assert float(r["thrust"][0]) > 100.0
    assert 0.004 / T < depth < 0.004 / T + 1.0e-3, depth                     # held just past where it meets material


def test_a_stack_with_a_pilot_in_the_lower_layer_keeps_one_front():
    """Re-review check: the stack law's plate on block, with a 3 mm through pilot already in the block. The 6 mm
    drill's stage in the block is free over the pilot's void and then cuts; it follows the plate's front throughout,
    so the block's bottom is the plate's less 3 mm (1 um). The thrust does dip at the plate's breakthrough here, and
    should: the chisel edge leaves the plate into the pilot's void, where it cuts nothing. Once the point has left the
    plate, the thrust is steel's annulus law (lips outside the pilot, no chisel) at the realized feed (1 %)."""
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.run import run_scene

    d = Drill()
    law = next(law for law in d.laws() if law.name == "stack")
    seed_init = law.scene.init

    def init(sim):
        seed_init(sim)
        _seed(sim.plugins["drill"], (0.0, 0.0), [(0.003, 0.05)], layer=1)

    scene = dataclasses.replace(law.scene, init=init)
    run = run_scene(scene, [fresh(d, dict(law.config))], "cpu")
    assert run.error is None, run.error
    p = run.plugin
    H, S = int(p.config.max_holes), int(p.config.max_stages)
    b = np.asarray(run.trace.plugin(p.label, "stage_b"), np.float64)
    top, low = b[:, 0, 0], b[:, 0, H * S + 1]                         # the block's 6 mm stage (its pilot is stage 0)
    R = 0.5 * float(np.float32(p.config.diameter))
    span = (low != 0.0) & (top < dl.STACK_PLATE + R / T)
    assert span.any()
    assert np.abs(low[span] - (top[span] - dl.STACK_PLATE)).max() < 1.0e-6
    thrust = np.asarray(run.trace.plugin(p.label, "thrust"), np.float64)[:, 0]
    feed = np.asarray(run.trace.plugin(p.label, "feed"), np.float64)[:, 0]
    rc, rp = float(np.float32(R * p.config.chisel_ratio)), 0.0015
    block = (low > R / T + 2.0e-4) & (top > dl.STACK_PLATE + R / T)
    assert block.any()
    want = (R - rp) / (R - rc) * float(p.coefficients_of(1).lips) * feed[block] ** 0.8 * (2.0 * R) ** 0.8
    assert np.abs(thrust[block] / want - 1.0).max() < 0.01


def test_a_small_drill_off_the_axis_of_a_wide_hole_is_not_pulled_to_it():
    """Re-review minor: a 4 mm drill 1.5 mm off an 8 mm through hole's axis (inside its 2 mm clearance) meets no wall:
    the head stays where it holds (the walls pulled it 0.2 mm toward the axis against the head before)."""
    from tests.plugins.test_drilling import _feeding_sim, _steps

    drill = Drill(diameter=0.004)
    sim = _feeding_sim(drill, x0=0.0015)
    _seed(drill, (0.0, 0.0), [(0.008, 0.05)])
    _steps(sim, 30)
    assert abs(float(sim.state.body_q.numpy()[0, 0]) - 0.0015) < 2.0e-5
    assert float(drill.read()["removed"][0]) == 0.0


def test_pin_configs_are_checked_and_survive_copies():
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.holes.pins import Pin, PushHead

    d = Drill(pins=({"body": "bolt", "radius": 0.004, "head": 0.0065, "chamfer": 0.0005},))
    assert isinstance(d.config.pins[0], Pin) and fresh(d).config.pins == d.config.pins
    for bad in ({"body": "bolt", "radius": 0.0}, {"body": "bolt", "top": (0, 0, 0), "tip": (0, 0, 0)},
                {"body": "bolt", "friction": -0.1}, {"body": "bolt", "stiffness": 0.0},
                {"body": "bolt", "chamfer": 0.0005, "damping": 1.0e-4}):          # a chamfer needs damping
        with pytest.raises(PluginError):
            Drill(pins=(bad,))
    with pytest.raises(PluginError):
        PushHead(direction=(0.0, 0.0, -2.0))


def test_the_pin_laws_fail_a_bolt_that_tips():
    """The laws bound every world's tilt: a wedged bolt at the right depth but tipped 5 deg (as at 200 N) fails, one
    at 2.7 deg (150 N) passes, and so does an upright one."""
    from alloy.plugins.holes import pin_laws as pl

    case = pl.GUIDE.cases[5]                                                      # plate_a 8.4 mm, plate_b 0.75 mm off
    tip = pl.PLATE - case.stop - pl.GUIDE.force / pl.GUIDE.stiffness / case.axial   # exactly at its rest
    head = tip + pl.SHANK
    assert pl._error(case, head, tip, 0.0, pl.GUIDE) < 1e-6
    assert pl._error(case, head, tip, 2.7, pl.GUIDE) < 1.0
    assert pl._error(case, head, tip, 5.0, pl.GUIDE) > 1.0


@pytest.mark.compile_heavy
def test_a_pin_pushed_past_the_wall_stays_in_its_hole():
    """A seated bolt jerked 5 mm sideways (its axis past the 4.5 mm wall, as a large side load or a
    jump in one step would) keeps its hole in both plates, so the walls push it back in instead of letting it through
    the part sideways."""
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.holes import pin_laws as pl

    d = Drill()
    law = next(law for law in d.laws() if law.name == "pin_chamfer")              # world 0: aligned, 150 N
    sim = build_simulation(dataclasses.replace(law.scene, worlds=1), [fresh(d, dict(law.config))], "cpu")
    for _ in range(50):
        sim.step_period()
    bolt = 0                                                                         # the scene's first body
    q = sim.state.joint_q.numpy()
    seat = 2.0 * pl.PLATE - pl.GUIDE.force / pl.STIFFNESS
    assert abs(float(sim.state.body_q.numpy()[bolt, 2]) - seat) < 2.0e-4             # seated
    q[0] += 0.005                                                                    # the bolt's free joint comes first
    sim.state.joint_q.assign(q)
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    for _ in range(20):
        sim.step_period()
    drill = sim.plugins["drill"]
    assert (drill.read()["pin_hole"][0] == 1).all()                                  # still in both holes
    assert abs(float(sim.state.body_q.numpy()[bolt, 0])) < 0.0006                    # back within the clearance


@pytest.mark.compile_heavy
def test_the_stops_hold_deeper_than_the_plate():
    """Every stop holds at any depth. A soft pin (1.5e4 N/m) pushed at 150 N sinks F / k = 10 mm,
    past the 6 mm plate (plate_a, the only drilled layer here): on the face, on a 7.9 mm mouth, a 5 mm pilot, a
    blind hole's cone and a step, and with its head on the face and on a counterbore's floor, it rests at u* + 10 mm."""
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.run import run_scene
    from alloy.plugins.holes import pin_laws as pl

    cases = tuple(case for case in pl.STRONG.cases if case.want != "b")          # plate_b is not drilled here
    soft = dataclasses.replace(pl.STRONG, name="deep", cases=cases, stiffness=1.5e4, d_push=1250.0)   # 120 mm/s
    d = Drill()
    config = {"layers": pl.LAYERS[:1], "pins": soft.pins()}
    run = run_scene(dataclasses.replace(pl.pin_scene(d.config, soft), periods=60), [fresh(d, config)], "cpu")
    assert run.error is None, run.error
    res = pl.checker(soft)(run)
    assert res.passed, res.detail
    assert int(run.plugin.read()["pin_solid"].max()) == 0                       # landed, never embedded


@pytest.mark.compile_heavy
def test_a_fast_pin_lands_on_a_thin_sheet():
    """A pin at 12 m/s moves 1.9 mm per step, more than a 1 mm sheet: the landing is found from the step's crossing,
    so the face catches it (and it rebounds) instead of the pin skipping the sheet."""
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.run import run_scene
    from alloy.plugins.harness.scene import TestScene, body_xform
    from alloy.plugins.holes import pin_laws as pl

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.0005), hx=0.03, hy=0.03, hz=0.0005, label="sheet")
        bolt = b.add_body(xform=body_xform(0.0, 0.0, 0.002 + pl.SHANK), label="bolt")
        b.add_shape_box(bolt, xform=body_xform(0.0, 0.0, pl.HOLDER_HALF), hx=pl.HOLDER_HALF, hy=pl.HOLDER_HALF,
                        hz=pl.HOLDER_HALF, cfg=newton.ModelBuilder.ShapeConfig(
                            density=pl.HOLDER_MASS / (2.0 * pl.HOLDER_HALF) ** 3, has_shape_collision=False))
        spindle = b.add_body(xform=body_xform(0.2, 0.0, 0.2), label="spindle")
        b.add_shape_box(spindle, hx=0.03, hy=0.03, hz=0.03,
                        cfg=newton.ModelBuilder.ShapeConfig(density=3.0 / 0.06**3, has_shape_collision=False))

    def launch(sim) -> None:
        qd = sim.state.joint_qd.numpy()
        qd[2] = -12.0                                                     # the bolt's free joint: vz
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)

    scene = TestScene(build=build, init=launch, worlds=1, periods=10, dt=pl.STEP, ground=False,
                      gravity=(0.0, 0.0, 0.0))
    config = {"layers": ({"workpiece": "sheet", "work": "steel_200bhn", "density": 7850.0},),
              "pins": ({"body": "bolt", "tip": (0.0, 0.0, -pl.SHANK), "radius": pl.SHANK_R},)}
    run = run_scene(scene, [fresh(Drill(), config)], "cpu")
    assert run.error is None, run.error
    tip = run.trace.body_q[:, run.bodies("bolt")[0], 2].astype(np.float64) - pl.SHANK
    assert tip.min() > -0.03 and tip[-1] > 0.0                                   # caught and thrown back up


def _anchor_scene(force: float):
    """A 38 mm ledger on a 200 mm concrete wall (seeded: a 14.3 mm hole through the ledger, a 12.7 mm hole 108 mm
    deep in the wall) under a 1/2 in wedge anchor: 12.6 mm shank 140 mm long, 24 mm washer, on a 2 kg holder, its tip
    through the ledger 2 mm above the wall, pushed in at 0.2 m/s."""
    from alloy.plugins.harness.scene import TestScene, body_xform
    from alloy.plugins.holes import pin_laws as pl
    from alloy.plugins.holes.pins import PushHead

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.1), hx=0.1, hy=0.1, hz=0.1, label="wall")
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.219), hx=0.1, hy=0.1, hz=0.019, label="ledger")
        anchor = b.add_body(xform=body_xform(0.0, 0.0, 0.2 + 0.002 + 0.14), label="anchor")   # through the ledger
        b.add_shape_box(anchor, xform=body_xform(0.0, 0.0, 0.05), hx=0.05, hy=0.05, hz=0.05,
                        cfg=newton.ModelBuilder.ShapeConfig(density=2.0 / 0.1**3, has_shape_collision=False))
        spindle = b.add_body(xform=body_xform(0.5, 0.0, 0.5), label="spindle")
        b.add_shape_box(spindle, hx=0.03, hy=0.03, hz=0.03,
                        cfg=newton.ModelBuilder.ShapeConfig(density=3.0 / 0.06**3, has_shape_collision=False))

    seed = pl._seeder((pl.Case(((0.0143, 0.06),), ((0.0127, 0.108 + 0.00635 / pl.TAN),), 0.0, "pass"),
                       pl.Case(((0.0143, 0.06),), ((0.020, 0.108 + 0.010 / pl.TAN),), 0.0, "pass")))   # too wide

    def init(sim) -> None:
        seed(sim)
        sim.plugins["drill"].set_input("nut_torque", np.full((sim.model.world_count, 1), 68.0))

    return TestScene(build=build, init=init, worlds=2, periods=40, dt=pl.STEP, ground=False,
                     gravity=(0.0, 0.0, 0.0), companions=lambda: [PushHead(body="anchor", force=force, d_push=250.0)])


ANCHOR_CONFIG = {"layers": ({"workpiece": "ledger", "work": "pine_southern", "density": 538.0, "thickness": 0.038},
                            {"workpiece": "wall", "work": "concrete_medium_hard", "density": 2400.0}),
                 "pins": ({"body": "anchor", "tip": (0.0, 0.0, -0.14), "radius": 0.0063, "head": 0.012,
                           "anchor": 0.095, "anchor_hole": 0.01346},)}       # a 1/2 in carbide bit's upper size


@pytest.mark.compile_heavy
def test_a_wedge_anchor_engages_holds_and_reads_its_clamp_load():
    """The anchor's washer seats on the ledger with its tip 102 mm into the wall (past the 95 mm the expansion needs),
    torqued to 68 N m (ESR-4266's 1/2 in install torque) the clip engages and the clamp load reads T / (K d) with
    K 0.2; pulled back at 50 N the clip holds it within F / k of where it engaged."""
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.scene import build_simulation

    sim = build_simulation(_anchor_scene(50.0), [fresh(Drill(), ANCHOR_CONFIG)], "cpu")
    for _ in range(40):
        sim.step_period()
    drill = sim.plugins["drill"]
    r = drill.read()
    sink = 50.0 / 2.0e5
    assert int(r["pin_anchored"][0, 0]) == 1 and int(r["pin_anchored"][1, 0]) == 0       # a 20 mm hole: no grip
    over = float(r["pin_embed"][0, 0]) - (0.14 - 0.038 + sink)
    assert 0.0 <= over < 5.0e-4                                  # the one-way clip keeps the seating's overshoot
    assert float(r["anchor_clamp"][0, 0]) == pytest.approx(68.0 / (0.2 * 0.0126), rel=1e-6)
    z_set = float(sim.state.body_q.numpy()[0, 2])
    sim.plugins["push_head"].set_direction((0.0, 0.0, 1.0))                      # pull it back out
    for _ in range(10):
        sim.step_period()
    z = float(sim.state.body_q.numpy()[0, 2])
    assert 0.0 <= z - z_set < 2.0 * sink + 2.0e-4                       # off the seat; the clip holds
    assert int(drill.read()["pin_anchored"][0, 0]) == 1
