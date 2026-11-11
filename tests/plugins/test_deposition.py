# SPDX-License-Identifier: Apache-2.0
"""Deposition plug-in (contact-driven ink, XB4-7 whiteboard gap): harness laws, saturation, reservoir, colours,
reactions, config and stability checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.deposition import Deposition
from alloy.plugins.deposition import laws as dl
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation

pytestmark = pytest.mark.law


@pytest.mark.compile_heavy
def test_harness_quick_check_passes():
    report = ap.check(Deposition(), quick=True)
    assert report.passed, report.summary()
    laws = {c.name for c in report.checks if c.group == "law"}
    assert {"line_density", "stroke_width", "ink_ledger", "wipe_lifts_ink", "no_ink_when_lifted"} <= laws


@pytest.mark.parametrize("bad", [{"kinds": ("pen", "wiper")}, {"colours": (2,)}, {"tip_radius": (0.003, 0.02, 0.01)},
                                 {"wipe_length": (0.0, 0.0)}, {"saturation": -1.0}, {"channels": ()},
                                 {"stiffness": (math.nan, 3e6)}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(PluginError):
        Deposition(**bad)


def test_unstable_nib_is_refused_naming_the_tool():
    plugin = Deposition(stiffness=(1.0e12, 3.0e6))
    sim = build_simulation(dl.marker_and_wiper_scene(plugin.config), [plugin], "cpu", worlds=1)
    with pytest.raises(PluginError, match="marker tip spring"):
        sim.step_period()


def _marker_run(overrides: dict, rigs: list, periods: int = 25, worlds: int = 1):
    plugin = Deposition(**{**dl.MARKER, **overrides})
    scene = dl.bench(plugin.config, rigs, dl.SMALL_BOARD, worlds=worlds, periods=periods)
    run = run_scene(scene, [plugin], "cpu")
    assert run.error is None, run.error
    return plugin, run.sim.plugins[plugin.label].read()


def test_saturation_caps_the_board():
    """A marker held still keeps flowing, but no cell takes more than saturation x cell area."""
    plugin, read = _marker_run({"saturation": 2.0e-4}, [dl.marker_rig(speed=(0.0,), press=(4.0,))])
    held = read["ink"][0].sum(axis=-1)
    capacity = 2.0e-4 * plugin.cell_area
    assert held.max() <= capacity * (1.0 + 1e-6)
    assert held.max() >= 0.9 * capacity                           # it did fill up
    assert read["ledger"][0] == pytest.approx(3.0e-3, rel=1e-12)


def test_reservoir_runs_dry_exactly():
    plugin, read = _marker_run({"reservoir": (1.0e-9,)}, [dl.marker_rig(speed=(0.1,), press=(1.0,))])
    laid, left = read["laid"][0, 0], read["reservoir"][0, 0]
    assert 0.0 <= left < 1.0e-15
    assert laid + left == pytest.approx(1.0e-9, abs=1e-20)
    assert read["ink_total"][0] == pytest.approx(laid, abs=1e-20)


def test_each_marker_inks_its_own_channel():
    two = {"tools": ("red", "blue"), "kinds": ("marker",), "colours": (0, 1), "channels": ("red", "blue")}
    rigs = [dl.Rig("red", (0.008, 0.008, 0.01), 0.1, -0.04, (0.1,), (1.0,), y0=-0.01),
            dl.Rig("blue", (0.008, 0.008, 0.01), 0.1, -0.04, (0.1,), (1.0,), y0=0.01)]
    plugin, read = _marker_run(two, rigs)
    ink = read["ink"][0]
    y = (np.arange(ink.shape[1]) + 0.5) * float(plugin.grid.cell[1]) - float(plugin.grid.half[1])
    assert ink[:, y < 0, 0].sum() > 0 and ink[:, y < 0, 1].sum() == 0
    assert ink[:, y > 0, 1].sum() > 0 and ink[:, y > 0, 0].sum() == 0
    assert read["laid"][0] == pytest.approx(read["by_channel"][0], rel=1e-12)


def _floating_board(c):
    def build(b: newton.ModelBuilder) -> None:
        board = b.add_body(xform=body_xform(0.0, 0.0, dl.BIG_BOARD[2]), label="board_body")
        b.add_shape_box(board, hx=dl.BIG_BOARD[0], hy=dl.BIG_BOARD[1], hz=dl.BIG_BOARD[2], label=c.board,
                        cfg=newton.ModelBuilder.ShapeConfig(density=2000.0))
        for label, x, half in (("marker", 0.02, 0.01), ("wiper", -0.05, 0.03)):
            tool = b.add_body(xform=body_xform(x, 0.01, 2.0 * dl.BIG_BOARD[2] + 0.015 - 3e-4,
                                               q=(0.01, -0.02, 0.0, math.sqrt(1.0 - 5e-4))), label=label)
            b.add_shape_box(tool, hx=half, hy=half, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=8000.0))
    return build


def test_reaction_on_a_floating_board_balances_the_tools():
    """With the board on a body, every tip's pressure and friction comes back on it: zero net wrench. (A 0.4 kg
    floating board rotates under an off-centre load, so the stiff nib is refused at 1.25 ms: use a softer one.)"""
    plugin = Deposition(stiffness=(1.0e8, 3.0e6))
    def init(sim) -> None:
        qd = sim.state.joint_qd.numpy().reshape(-1, 6)       # free joints in body order: board, marker, wiper
        qd[1] = (0.2, 0.05, 0.0, 0.0, 0.0, 1.0)
        qd[2] = (-0.1, 0.1, 0.0, 0.0, 0.0, 3.0)
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=_floating_board(plugin.config), init=init, ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1), [plugin], "cpu")
    host = sim.plugins
    host.before(sim.state, None, float(sim.clock.dt))
    f = host.forces.body_f.numpy().astype(np.float64)
    host.restore_forces(sim.state)
    q = sim.state.body_q.numpy().astype(np.float64)
    com = q[:, :3] + np.stack([_rotate(q[i, 3:], c) for i, c in enumerate(sim.model.body_com.numpy())])
    scale = np.linalg.norm(f[:, :3], axis=1).sum()
    assert scale > 1.0
    assert np.linalg.norm(f[:, :3].sum(axis=0)) < 1e-5 * scale
    assert np.linalg.norm((f[:, 3:] + np.cross(com, f[:, :3])).sum(axis=0)) < 1e-5 * scale


def test_cells_must_resolve_the_tips():
    """F1: a cell larger than a tip's radius is refused (its contact could fall between cell centres)."""
    plugin = Deposition(cells=(40, 20))                      # 5 mm cells on the 200 x 100 mm board, 3 mm nib
    with pytest.raises(PluginError, match="larger than tool 'marker'"):
        build_simulation(dl.marker_and_wiper_scene(plugin.config), [plugin], "cpu", worlds=1)


def test_tools_on_a_floating_board_share_its_bound():
    """F2: on a floating board, each tool's bound carries every tool's board terms, and the friction mobility
    T·w_board; on a static board the board terms vanish."""
    from alloy.plugins.deposition.stability import stability_numbers  # noqa: PLC0415

    both = Deposition(stiffness=(1.0e8, 3.0e6))
    build_simulation(TestScene(build=_floating_board(both.config), ground=False, gravity=(0.0, 0.0, 0.0), worlds=1),
                     [both], "cpu")
    r, dt = both.response, 1.25e-3
    k_marker, k_wiper = (s.contact_stiffness for s in both.specs)
    tilt = [s.contact_stiffness * 0.25 * s.radius**2 for s in both.specs]
    own = k_marker * r.w_tool[0, 0] + 2.0 * tilt[0] * r.inv_tilt_tool[0, 0]
    alone = own + k_marker * r.w_board[0] + 2.0 * tilt[0] * r.inv_tilt_board[0]
    shared = alone + k_wiper * r.w_board[0] + 2.0 * tilt[1] * r.inv_tilt_board[0]
    spring, _ = stability_numbers(r, both.specs, dt)["marker"]
    assert spring == pytest.approx(math.sqrt(shared) * dt, rel=1e-9)
    assert spring > math.sqrt(alone) * dt * 1.2                   # the wiper's load on the board matters here
    assert both.mobility.numpy()[0, 0] == pytest.approx(r.friction_tool[0, 0] + 2 * r.w_board[0], rel=1e-6)
    static = Deposition()
    build_simulation(dl.marker_and_wiper_scene(static.config), [static], "cpu", worlds=1)
    assert static.response.w_board[0] == 0.0
    assert static.mobility.numpy()[0, 0] == pytest.approx(static.response.friction_tool[0, 0], rel=1e-6)


def _rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * np.asarray(v))


@pytest.mark.gpu
def test_harness_full_check_on_gpu():
    """Laws, hooks, determinism, worlds, graph replay and speed with real CUDA graphs (3-4 worlds: a light smoke)."""
    report = ap.check(Deposition(), device="cuda:0")
    print(report.summary())
    assert report.passed, report.summary()
