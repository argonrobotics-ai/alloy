# SPDX-License-Identifier: Apache-2.0
"""Drilling plug-in (engine item 10, material removal part 2): harness laws R-5, config, stability, reactions, nulls."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.drilling import Drill
from alloy.plugins.drilling import laws as dl
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import fresh
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.npgeom import quat_rotate

pytestmark = pytest.mark.law


LAWS = ("shaw_feed", "shaw_diameter", "through_hole", "no_cut_when_lifted", "hole_axis", "blind_depth", "stack",
        "pilot", "pin_through", "pin_chamfer", "hammer_rate", "rebar_stop", "rebar_through")


@pytest.mark.compile_heavy
def test_harness_quick_check_passes():
    """The harness's generic checks; the laws run one per test below."""
    report = ap.check(Drill(), quick=True, laws=False)
    assert report.passed, report.summary()
    assert tuple(law.name for law in Drill().laws()) == LAWS                 # a new law gets its own test


@pytest.mark.compile_heavy
@pytest.mark.parametrize("name", LAWS)
def test_law(name):
    drill = Drill()
    law = next(law for law in drill.laws() if law.name == name)
    run = run_scene(law.scene or drill.test_scene(), [fresh(drill, dict(law.config or {}))], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, f"{name}: {res.value:.4g} vs bound {res.bound:.4g}: {res.detail} {res.fix}"


@pytest.mark.parametrize("bad", [{"diameter": 0.0}, {"capture": 1.5}, {"point_angle": 180.0}, {"chisel_ratio": 1.0},
                                 {"axis": (0.0, 0.0, 0.0)}, {"work_factor": math.nan}, {"thickness": -1.0}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(PluginError):
        Drill(**bad)


def test_unstable_front_is_refused_with_the_knob_to_change():
    plugin = Drill(stiffness=1.0e10)
    sim = build_simulation(dl.drilling_scene(plugin.config), [plugin], "cpu", worlds=1)
    with pytest.raises(PluginError, match="stiffness"):
        sim.step_period()


def test_shaw_coefficients_reproduce_the_handbook_example():
    """Machinery's Handbook's worked example (7/8 in drill, 0.008 in/rev, Kd 24,000, W 1.30): T = 2313 lb and
    M = 559 in-lb. The SI constants are an exact unit conversion, so they give the same numbers."""
    d = Drill(work_factor=24000.0, diameter=0.875 * 0.0254)
    f = 0.008 * 0.0254
    assert float(d.thrust_law(f)) / 4.4482216152605 == pytest.approx(2313.0, abs=1.0)
    assert float(d.torque_law(f)) / (4.4482216152605 * 0.0254) == pytest.approx(559.0, abs=1.0)


def test_a_drill_that_does_not_spin_is_a_wall():
    """Fed into the block without spinning, the drill cuts nothing: the front holds the head like a rigid face."""
    plugin = Drill()
    c = plugin.config
    scene = TestScene(build=dl.spindle_and_block(c), init=dl.feeding((dl.rate_of(dl.F_REF),), (0.0,)), worlds=1,
                      periods=15, dt=dl.STEP, ground=False, companions=dl._head(c))
    run = run_scene(scene, [plugin], "cpu")
    assert run.error is None, run.error
    read = run.sim.plugins.read()[plugin.label]
    assert read["removed"][0] == 0.0 and read["depth"][0] == 0.0
    assert read["thrust"][0] > 100.0                         # the head pushes on the uncut face


WIDE = (0.05, 0.05, 0.015)           # m: a block wide enough to move the spindle 20 mm


def _feeding_sim(plugin: Drill, half=WIDE, x0: float = 0.0, gap: float = dl.GAP, rate: float = dl.rate_of(dl.F_REF)):
    c = plugin.config

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, half[2]), hx=half[0], hy=half[1], hz=half[2],
                        label=c.workpiece)
        head = b.add_body(xform=body_xform(x0, 0.0, 2.0 * half[2] + gap - float(c.tip[2])), label=c.tool)
        b.add_shape_box(head, hx=dl.HEAD_HALF, hy=dl.HEAD_HALF, hz=dl.HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=dl.HEAD_MASS / (2.0 * dl.HEAD_HALF) ** 3))
    scene = TestScene(build=build, init=dl.feeding((rate,), (dl.SPIN,)), worlds=1, periods=1, dt=dl.STEP,
                      ground=False, companions=dl._head(c))
    return build_simulation(scene, [plugin], "cpu")


def _steps(sim, periods: int) -> None:
    for _ in range(periods):
        sim.step_period()


def _move_head(sim, x: float, gap: float) -> None:
    """Teleport the spindle to x with its tip ``gap`` above the face (still spinning) and re-aim its feed ramp there."""
    head = sim.plugins["feed_head"]
    drill = sim.plugins["drill"]
    q = sim.state.joint_q.numpy()
    z = 2.0 * WIDE[2] + gap - float(drill.config.tip[2])
    q[0:3] = (x, 0.0, z)
    sim.state.joint_q.assign(q)
    qd = sim.state.joint_qd.numpy()
    qd[0:5] = 0.0
    sim.state.joint_qd.assign(qd)
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    t = float(head.ctx.time.numpy()[0])
    rate = float(head.feed_rate.numpy()[0])
    head.hold.assign(np.asarray([[x, 0.0, z + rate * t]], np.float32))


def _hole_mass(drill: Drill, depths) -> float:
    """density x the drill-point solid's volume of blind holes of the given depths (float64, as the kernel)."""
    r = 0.5 * float(np.float32(drill.config.diameter))
    h = r / float(np.float32(math.tan(math.radians(0.5 * drill.config.point_angle))))
    vol = lambda z: math.pi * r * r * z / 3.0 * (z / h) ** 2 if z < h else math.pi * r * r * (z - h + h / 3.0)  # noqa: E731
    return drill.config.density * sum(vol(z) for z in depths)


def test_a_re_entry_far_away_starts_a_new_hole_at_the_face():
    """Drill, retract, move 20 mm, drill again: the second hole cuts from the face (not from the first hole's depth),
    the walls do not pull the bit back to the first hole, and the ledger holds both holes (review D2)."""
    drill = Drill()
    sim = _feeding_sim(drill)
    _steps(sim, 30)
    first = float(drill.read()["depth"][0])
    _move_head(sim, 0.02, gap=dl.GAP)
    _steps(sim, 30)
    r = drill.read()
    x = float(sim.state.body_q.numpy()[0, 0])
    assert int(r["holes"][0]) == 2
    assert np.allclose(r["entry"][0], (0.02, 0.0), atol=1e-4), r["entry"][0]
    assert abs(x - 0.02) < 5e-5, x                                   # no 2 kN pull toward the first hole
    assert float(r["depth"][0]) == pytest.approx(first, rel=0.02)    # same feed, same time: the same depth again
    assert float(r["removed"][0]) == pytest.approx(_hole_mass(drill, (first, float(r["depth"][0]))), rel=1e-6)


def test_peck_drilling_continues_the_same_hole():
    """Retract 1 mm above the face and come back down at the same spot: one hole, deepened, exact ledger."""
    drill = Drill()
    sim = _feeding_sim(drill)
    _steps(sim, 30)
    first = float(drill.read()["depth"][0])
    _move_head(sim, 0.0, gap=1.0e-3)
    _steps(sim, 60)                                      # 0.6 s back down to the bottom, then 0.6 s of cutting
    r = drill.read()
    assert int(r["holes"][0]) == 1
    assert float(r["depth"][0]) > first + 1.0e-3
    assert float(r["removed"][0]) == pytest.approx(_hole_mass(drill, (float(r["depth"][0]),)), rel=1e-6)


def test_a_drill_beside_the_plate_cuts_nothing():
    """Below the face plane but beside the patch there is no material: no thrust, no hole, no mass (review D3)."""
    drill = Drill()
    sim = _feeding_sim(drill, x0=WIDE[0] + 0.005, gap=-2.0e-3)
    _steps(sim, 15)
    r = drill.read()
    assert float(r["removed"][0]) == 0.0 and float(r["depth"][0]) == 0.0 and int(r["holes"][0]) == 0
    assert float(np.abs(r["thrust"]).max()) == 0.0


def _inclined_plate(c, slope: float):
    """A static plate tilted by ``slope`` about x (its top face's normal leans toward -y) under a vertical spindle."""
    half = (0.05, 0.05, 0.015)
    q = (math.sin(0.5 * slope), 0.0, 0.0, math.cos(0.5 * slope))
    face_z = half[2] / math.cos(slope)                     # where the tilted top face crosses x = y = 0

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.0, q=q), hx=half[0], hy=half[1], hz=half[2],
                        label=c.workpiece)
        head = b.add_body(xform=body_xform(0.0, 0.0, face_z + dl.GAP - float(c.tip[2])), label=c.tool)
        b.add_shape_box(head, hx=dl.HEAD_HALF, hy=dl.HEAD_HALF, hz=dl.HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=dl.HEAD_MASS / (2.0 * dl.HEAD_HALF) ** 3))
    return build, face_z


def _below_face(sim, c, slope: float, face_z: float) -> float:
    """How far the drill tip is inside the tilted plate along its top face's normal (m; negative above it)."""
    q = sim.state.body_q.numpy()[0]
    tip = q[:3] + quat_rotate(q[3:], np.asarray(c.tip, np.float64))
    n = np.array([0.0, -math.sin(slope), math.cos(slope)])
    return float(-(n @ (tip - np.array([0.0, 0.0, face_z]))))


def test_a_drill_on_a_steep_face_is_held_and_cuts_nothing():
    """A vertical drill fed onto a face inclined 45 deg (past MIN_FACING) does not cut, opens no hole and does not pass
    through the plate: the front holds it like a rigid face (review D1). The feed ramp drives the tip 1.6 mm into the
    face (along its normal) in 25 periods; the tip stays within 0.5 mm of it. The front's push comes and goes inside a
    period as the tip slides on the tilted face, so the last substep's thrust readout is no measure of it (it read 0
    or 600 N depending on which way the head leaned), and the depth is taken along the face normal over the run."""
    drill = Drill()
    slope = math.radians(45.0)
    build, face_z = _inclined_plate(drill.config, slope)
    scene = TestScene(build=build, init=dl.feeding((dl.rate_of(dl.F_REF),), (dl.SPIN,)), worlds=1, periods=1,
                      dt=dl.STEP, ground=False, companions=dl._head(drill.config))
    sim = build_simulation(scene, [drill], "cpu")
    periods, deepest = 25, -1.0
    for _ in range(periods):
        _steps(sim, 1)
        deepest = max(deepest, _below_face(sim, drill.config, slope, face_z))
    ramp = (dl.rate_of(dl.F_REF) * periods * float(sim.built.control_dt) - dl.GAP) * math.cos(slope)
    r = drill.read()
    assert float(r["removed"][0]) == 0.0 and float(r["depth"][0]) == 0.0 and int(r["holes"][0]) == 0
    assert ramp > 1.5e-3                                                         # the feed pressed well past the face
    assert deepest < 0.5e-3, deepest                                             # held at the face, not through


FLOATING = (0.1, 0.1, 0.02)          # m: a 4.3 kg aluminium workpiece block on a free body


def _floating_block(c, half=FLOATING):
    def build(b: newton.ModelBuilder) -> None:
        block = b.add_body(xform=body_xform(0.0, 0.0, half[2]), label="block_body")
        b.add_shape_box(block, hx=half[0], hy=half[1], hz=half[2], label=c.workpiece,
                        cfg=newton.ModelBuilder.ShapeConfig(density=2700.0))
        tip_depth = 3.5e-3                                   # past a 3 mm hole: front, torque and both wall springs
        z = 2.0 * half[2] - tip_depth - float(c.tip[2])
        head = b.add_body(xform=body_xform(0.0012, -0.0004, z, q=(0.01, -0.02, 0.0, math.sqrt(1.0 - 5e-4))),
                          label=c.tool)
        b.add_shape_box(head, hx=dl.HEAD_HALF, hy=dl.HEAD_HALF, hz=dl.HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=dl.HEAD_MASS / (2.0 * dl.HEAD_HALF) ** 3))
    return build


def _spin_and_slide(sim) -> None:
    qd = sim.state.joint_qd.numpy().reshape(-1, 6)            # free joints in body order: block, head
    qd[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    qd[1] = (0.02, 0.01, -0.005, 0.0, 0.0, 300.0)
    sim.state.joint_qd.assign(qd.reshape(-1))
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    drill = sim.plugins["drill"]                               # a 3 mm hole at the origin, the bit in it
    drill.hole_n.assign(np.ones(drill.hole_n.shape, np.int32))
    axis = np.zeros(drill.hole_axis.shape, np.float32)
    axis[:, 0] = (0.0, 0.0, -1.0)
    drill.hole_axis.assign(axis)
    for name, value in (("stage_n", 1), ("stage_d", drill.config.diameter), ("stage_b", 3.0e-3), ("cur", 1)):
        a = getattr(drill, name)
        v = np.zeros(a.shape, a.numpy().dtype)
        v[:, 0] = value
        a.assign(v)


def test_reaction_on_a_moving_workpiece_balances_the_tool():
    """With the workpiece on a body, thrust, torque and wall forces on the tool come back on it: zero net wrench."""
    plugin = Drill()
    scene = TestScene(build=_floating_block(plugin.config), init=_spin_and_slide, ground=False,
                      gravity=(0.0, 0.0, 0.0), periods=1, worlds=1, dt=dl.STEP / 4.0)   # the block rotates too
    sim = build_simulation(scene, [plugin], "cpu")                    # runs _spin_and_slide
    host = sim.plugins                                                 # before() runs every check_dt first
    host.before(sim.state, None, float(sim.clock.dt))
    f = host.forces.body_f.numpy().astype(np.float64)
    host.restore_forces(sim.state)
    q = sim.state.body_q.numpy().astype(np.float64)
    com = q[:, :3] + np.stack([_rotate(q[i, 3:], c) for i, c in enumerate(sim.model.body_com.numpy())])
    net_f = f[:, :3].sum(axis=0)
    net_m = (f[:, 3:] + np.cross(com, f[:, :3])).sum(axis=0)
    scale = np.linalg.norm(f[:, :3], axis=1).sum()
    assert scale > 100.0                                     # the front is pressed in
    assert np.linalg.norm(f[1, 3:]) > 0.1                    # cutting torque and wall moments act
    assert np.linalg.norm(f[1, :2]) > 50.0                   # the walls pull the bit (1.2 mm off) back to the axis
    assert np.linalg.norm(net_f) < 1e-5 * scale
    assert np.linalg.norm(net_m) < 1e-5 * scale


def test_a_light_floating_workpiece_is_refused():
    """The front acts on the point mobility of tool and workpiece: a 0.29 kg floating block is too light for the
    default stiffness at this step, and check_dt names the knob."""
    plugin = Drill()
    scene = TestScene(build=_floating_block(plugin.config, half=dl.BLOCK_HALF), init=_spin_and_slide, ground=False,
                      gravity=(0.0, 0.0, 0.0), periods=1, worlds=1, dt=dl.STEP)
    sim = build_simulation(scene, [plugin], "cpu")
    with pytest.raises(PluginError, match="front (spring|damping)"):
        sim.step_period()


def _rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * np.asarray(v))


@pytest.mark.gpu
def test_harness_full_check_on_gpu():
    """Laws, hooks, determinism, worlds, graph replay and speed with real CUDA graphs (the one L4 smoke)."""
    report = ap.check(Drill(), device="cuda:0")
    assert report.passed, report.summary()
    print(report.summary())
