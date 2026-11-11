# SPDX-License-Identifier: Apache-2.0
"""Material removal plug-in (engine item 10 v1): harness laws R-1..R-3, deburring feedback, wiping, reaction forces."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.removal import Removal
from alloy.plugins.removal import laws as rl

pytestmark = pytest.mark.law

RIDGE = 2.0e-4          # m: burr ridge height in the deburring test
WIPE = 0.02             # m: wipe length in the wiping test


@pytest.mark.compile_heavy
def test_harness_quick_check_passes():
    report = ap.check(Removal(), quick=True)
    assert report.passed, report.summary()
    laws = {c.name for c in report.checks if c.group == "law"}
    assert {"archard", "preston", "mass_ledger", "pad_load", "no_removal_when_lifted"} <= laws


@pytest.mark.parametrize("bad", [{"pad_radius": 0.0}, {"capture": 1.5}, {"hardness": -1.0}, {"cells": (0, 4)},
                                 {"pad_normal": (0.0, 0.0, 0.0)}, {"friction": math.nan}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(PluginError):
        Removal(**bad)


def test_unstable_pad_is_refused_with_the_knob_to_change():
    plugin = Removal(pad_stiffness=1.0e11)
    sim = build_simulation(rl.sanding_scene(plugin.config), [plugin], "cpu", worlds=1)
    with pytest.raises(PluginError, match="pad_stiffness"):
        sim.step_period()


def _slide(plugin: Removal, setup=None, periods: int = 25):
    """The R-1 pin slides 0.1 m along x at 0.2 m/s; ``setup(plugin)`` edits the grid first. Returns world 0's read."""
    c = plugin.config
    def init(sim):
        rl.start_velocity((rl.PIN_SPEED, 0.0, 0.0))(sim)
        if setup is not None:
            setup(sim.plugins[plugin.label])
    scene = TestScene(build=rl.plate_and_tool(c, x0=-0.05), init=init, ground=False, periods=periods, worlds=1)
    run = run_scene(scene, [plugin], "cpu")
    assert run.error is None, run.error
    return run.sim.plugins.read()[plugin.label], run.sim.plugins[plugin.label]


def _columns(p: Removal) -> np.ndarray:
    nx = p.depth.shape[1]
    return (np.arange(nx) + 0.5) * float(p.grid.cell[0]) - float(p.grid.half[0])


def _ridge(p: Removal) -> np.ndarray:
    """A one-column burr ridge across the path at x ~ 0 (``RIDGE`` high)."""
    profile = np.zeros(p.profile.shape[1:])
    profile[int(np.argmin(np.abs(_columns(p))))] = RIDGE
    p.set_profile(profile)


@pytest.mark.parametrize("feedback", [True, False])
def test_burr_ridge_takes_the_load_only_with_feedback(feedback):
    """Deburring: a pad crossing a 0.1 mm ridge rides on it, so the ridge wears much faster than the flat."""
    plugin = Removal(**{**rl.PIN, "pad_radius": 0.02, "pad_stiffness": 1.0e7, "feedback": feedback})
    read, p = _slide(plugin, _ridge)
    depth = read["depth"][0]
    x = _columns(p)
    ridge = int(np.argmin(np.abs(x)))
    rows = np.abs((np.arange(depth.shape[1]) + 0.5) * float(p.grid.cell[1]) - float(p.grid.half[1])) < 0.01
    flat = [ridge - 4, ridge - 3, ridge + 3, ridge + 4]             # fully swept columns either side
    ratio = depth[ridge, rows].mean() / depth[flat][:, rows].mean()
    if feedback:
        assert ratio > 3.0, ratio
    else:
        assert abs(ratio - 1.0) < 0.1, ratio


def test_wiping_lifts_dirt_by_exp_of_slide_over_length():
    """A cloth pad (no wear) passing once over uniform dirt leaves exp(-2R / L) on the cells along its centre line."""
    dirt = 1.0e-6
    plugin = Removal(**rl.PIN, wear_coefficient=0.0, wipe_length=WIPE)
    read, p = _slide(plugin, lambda q: q.set_loose(np.full(q.loose.shape[1:], dirt)))
    left = read["loose"][0] / dirt
    x = _columns(p)
    ny = left.shape[1]
    centre = left[(np.abs(x) < 0.03)][:, ny // 2 - 1: ny // 2 + 1]
    want = math.exp(-2.0 * p.config.pad_radius / WIPE)
    assert np.abs(centre / want - 1.0).max() < 0.05, (centre.min(), centre.max(), want)
    assert read["removed"][0] == 0.0
    total = read["loose_total"][0] + read["sink"][0]
    assert total == pytest.approx(dirt * left.size, rel=1e-12)


def _floating_plate(c, density: float = 2700.0):
    def build(b: newton.ModelBuilder) -> None:
        plate = b.add_body(xform=body_xform(0.0, 0.0, rl.PLATE_HALF[2]), label="plate_body")
        b.add_shape_box(plate, hx=rl.PLATE_HALF[0], hy=rl.PLATE_HALF[1], hz=rl.PLATE_HALF[2], label=c.workpiece,
                        cfg=newton.ModelBuilder.ShapeConfig(density=density))
        tool = b.add_body(xform=body_xform(0.01, 0.005, rl.resting_height(c) - 2e-4,
                                           q=(0.02, -0.01, 0.0, math.sqrt(1.0 - 5e-4))), label=c.tool)
        hx, hy, hz = rl.tool_half(c)
        b.add_shape_box(tool, hx=hx, hy=hy, hz=hz, cfg=newton.ModelBuilder.ShapeConfig(density=rl.TOOL_DENSITY))
    return build


def _slide_tool(sim) -> None:
    qd = sim.state.joint_qd.numpy().reshape(-1, 6)        # free joints in body order: plate, tool
    qd[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    qd[1] = (0.3, 0.1, 0.0, 0.0, 0.0, 5.0)
    sim.state.joint_qd.assign(qd.reshape(-1))
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)


def test_reaction_on_a_moving_workpiece_balances_the_tool():
    """With the workpiece on a body, pad force and friction on the tool come back on the plate: zero net wrench."""
    plugin = Removal(pad_stiffness=1.0e7)          # the 1.08 kg plate's rotation (N1) needs a softer pad at 1.25 ms
    scene = TestScene(build=_floating_plate(plugin.config), init=_slide_tool, ground=False, gravity=(0.0, 0.0, 0.0),
                      periods=1, worlds=1)
    sim = build_simulation(scene, [plugin], "cpu")
    host = sim.plugins
    host.before(sim.state, None, float(sim.clock.dt))
    f = host.forces.body_f.numpy().astype(np.float64)
    host.restore_forces(sim.state)
    q = sim.state.body_q.numpy().astype(np.float64)
    com = q[:, :3] + np.stack([_rotate(q[i, 3:], c) for i, c in enumerate(sim.model.body_com.numpy())])
    net_f = f[:, :3].sum(axis=0)
    net_m = (f[:, 3:] + np.cross(com, f[:, :3])).sum(axis=0)
    scale = np.linalg.norm(f[:, :3], axis=1).sum()
    assert scale > 1.0                                     # the pad is pressed in
    assert np.linalg.norm(net_f) < 1e-5 * scale
    assert np.linalg.norm(net_m) < 1e-5 * scale


def _rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * np.asarray(v))


@pytest.mark.gpu
def test_harness_full_check_on_gpu():
    """Laws, hooks, determinism, worlds, graph replay and speed with real CUDA graphs (an L4 in the gate)."""
    report = ap.check(Removal(), device="cuda:0")
    assert report.passed, report.summary()
    print(report.summary())


# --- WP2 review: stability bounds (K1, K2, C1-C3) ----------------------------------------------------------------

def _small_tool(c, half: tuple[float, float, float]):
    """A static plate under a free tool of the given half extents, its pad at rest under its weight."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, rl.PLATE_HALF[2]), hx=rl.PLATE_HALF[0], hy=rl.PLATE_HALF[1],
                        hz=rl.PLATE_HALF[2], label=c.workpiece)
        mass = rl.TOOL_DENSITY * 8.0 * half[0] * half[1] * half[2]
        sink = mass * 9.81 / (c.pad_stiffness * math.pi * c.pad_radius**2)
        tool = b.add_body(xform=body_xform(0.0, 0.0, 2.0 * rl.PLATE_HALF[2] - c.pad_center[2] - sink), label=c.tool)
        b.add_shape_box(tool, hx=half[0], hy=half[1], hz=half[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=rl.TOOL_DENSITY))
    return build


def test_spinning_pad_on_a_slender_tool_comes_to_rest():
    """K2: a 40 mm rod with a 20 mm pad spins at 20 rad/s on the plate. Friction must stop it; a friction bound on the
    tool's mass alone left it flipping between +-7.4 rad/s every step (its spin damping was c·dt/I = 33)."""
    plugin = Removal(pad_radius=0.02, pad_center=(0.0, 0.0, -0.025), pad_stiffness=1.0e5, friction=0.5)
    scene = TestScene(build=_small_tool(plugin.config, (0.003, 0.003, 0.02)), ground=False, periods=25, worlds=1,
                      init=rl.start_velocity(angular=(0.0, 0.0, 20.0)))
    run = run_scene(scene, [plugin], "cpu")
    assert run.error is None, run.error
    spin = run.trace.body_qd[:, 0, 5]
    assert abs(spin[-1]) < 0.1 and abs(spin[-2]) < 0.1, spin[-4:]


def test_light_floating_workpiece_tightens_the_bound():
    """K1/C1: a 10 g floating plate under the 0.29 kg block: the pad spring acts on the reduced mass (refused), and
    the friction mobility includes the plate."""
    plugin = Removal()
    sim = build_simulation(TestScene(build=_floating_plate(plugin.config, density=25.0), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1), [plugin], "cpu")
    with pytest.raises(PluginError, match=r"pad spring \(normal \+ tilt\)"):
        sim.step_period()
    heavy = Removal(pad_stiffness=1.0e7)
    build_simulation(TestScene(build=_floating_plate(heavy.config), ground=False, gravity=(0.0, 0.0, 0.0), worlds=1),
                     [heavy], "cpu")
    static = Removal()
    build_simulation(rl.sanding_scene(static.config), [static], "cpu")
    assert plugin.response.mobility[0] > heavy.response.mobility[0] > static.response.mobility[0]


def test_tilt_term_is_in_the_damper_bound():
    """C2: a small dense tool under a wide pad: the normal damper alone is fine (0.2), with the tilt term it is not."""
    plugin = Removal(pad_radius=0.03, pad_center=(0.0, 0.0, -0.015), pad_stiffness=3.0e5, pad_damping=3.0e-3)
    build_simulation(TestScene(build=_small_tool(plugin.config, (0.01, 0.01, 0.01)), ground=False, worlds=1),
                     [plugin], "cpu")
    c, r = plugin.config, plugin.response
    stiffness = c.pad_stiffness * math.pi * c.pad_radius**2
    assert c.pad_damping * stiffness * r.normal_mobility * 1.25e-3 < 0.25         # normal alone would pass
    with pytest.raises(PluginError, match=r"pad damping \(normal \+ tilt\)"):
        plugin.check_dt(1.25e-3)
    plugin.config = dataclasses.replace(plugin.config, pad_damping=1.0e-3)
    plugin.check_dt(1.25e-3)                                                       # a third of the damping passes


def test_floating_face_rotation_is_in_the_normal_bound():
    """N1: a 0.28 kg floating plate passes a bound on its mass alone but not with its rotation under an off-centre
    load (corner of the plate: 1/m + r^2/I), so it is refused."""
    plugin = Removal(pad_damping=1.0e-3)
    sim = build_simulation(TestScene(build=_floating_plate(plugin.config, density=700.0), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1), [plugin], "cpu")
    c, r, dt = plugin.config, plugin.response, 1.25e-3
    m_tool, m_face = (float(sim.model.body_mass.numpy()[b]) for b in (1, 0))
    stiffness = c.pad_stiffness * math.pi * c.pad_radius**2
    mass_only = stiffness * (1.0 / m_tool + 1.0 / m_face) + 2.0 * stiffness * 0.25 * c.pad_radius**2 / r.tilt_inertia
    assert math.sqrt(mass_only) * dt < 1.0 and c.pad_damping * mass_only * dt < 1.0
    with pytest.raises(PluginError, match="normal"):
        plugin.check_dt(dt)


def test_bounds_take_the_worst_world():
    """C3: per-world masses (e.g. randomized): the check uses the lightest world, the friction bound each world's own."""
    from alloy.plugins.harness.scene import build_model  # noqa: PLC0415
    from alloy.plugins.removal.stability import pad_response  # noqa: PLC0415
    from alloy.plugins.removal.surface import patch_of  # noqa: PLC0415

    c = Removal().config
    model = build_model(TestScene(build=rl.plate_and_tool(c), ground=False), "cpu", worlds=2)
    tools = np.asarray([i for i, x in enumerate(model.body_label) if str(x).endswith(c.tool)])
    shapes = [i for i, x in enumerate(model.shape_label) if str(x).endswith(c.workpiece)]
    patches = [patch_of(model, s, c.workpiece) for s in shapes]
    before = pad_response(model, tools, patches, c.pad_center, c.pad_normal, c.pad_radius)
    mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
    mass[tools[1]] *= 0.25
    inertia[tools[1]] *= 0.25
    model.body_mass.assign(mass)
    model.body_inertia.assign(inertia)
    after = pad_response(model, tools, patches, c.pad_center, c.pad_normal, c.pad_radius)
    assert after.normal_mobility == pytest.approx(4.0 * before.normal_mobility, rel=1e-6)
    assert after.tilt_inertia == pytest.approx(0.25 * before.tilt_inertia, rel=1e-6)
    assert after.mobility[0] == pytest.approx(before.mobility[0], rel=1e-6)
    assert after.mobility[1] == pytest.approx(4.0 * before.mobility[1], rel=1e-5)
