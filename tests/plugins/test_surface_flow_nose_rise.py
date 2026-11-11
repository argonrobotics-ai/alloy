# SPDX-License-Identifier: Apache-2.0
"""Surface flow, a sealant nose that rises (stores_fill.py, ``pass_low``): a nose fills a cell no higher than the
lowest its underside has been over it in this pass. SOFT-2's tool-sealant: a nose lifting off its cut stroke grazed
the bead past the stroke's end, and GA's touch-then-fill laid its whole carry under its raised arc (CPU)."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store

SEALANT = FilmMaterial("latex_1Pas", density=1300.0, viscosity=1.0, surface_tension=0.0)
LENGTH, WIDTH, Z = 0.1, 0.04, 0.05         # plate [m]; the carriage's height
R, HALF_JOINT, BOTTOM, BACKER = 0.011, 0.006, -2.1e-3, -8.5e-3   # nose radius; joint; nose bottom; backer [m]
X0, HALF = -0.02, 0.009                    # the nose's start; half its sole along the joint


def carriage_scene(x0: float):
    """A static plate (top face at z = 0, the chart) and a heavy carriage at height Z on two slides, x then z."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01,
                        label="plate")
        xf = body_xform(x0, 0.0, Z)
        heavy = newton.ModelBuilder.ShapeConfig(density=1.0e7)
        slide = b.add_link(xform=xf, label="slide")
        b.add_shape_sphere(slide, radius=0.005, cfg=heavy)
        nose = b.add_link(xform=xf, label="nose")
        b.add_shape_sphere(nose, radius=0.005, cfg=heavy)
        b.add_articulation([b.add_joint_prismatic(-1, slide, parent_xform=xf, axis=newton.Axis.X),
                            b.add_joint_prismatic(slide, nose, axis=newton.Axis.Z)])
    return build


def arc(y: np.ndarray, up: float = 0.0) -> np.ndarray:
    """The nose's underside across the joint, ``up`` over its rest [m over the faces]."""
    return BOTTOM + up + R - np.sqrt(np.maximum(R * R - y * y, 0.0))


def nose_run(top, speed: float, rise: float, periods: int, initial: float = 0.5, load_at: int | None = None):
    """The nose (r 11 mm, its 18 mm sole along the joint) over a joint whose top is ``top(x, y)`` [m over the faces],
    sliding along x at ``speed`` while it rises at ``rise`` [m/s]. Its store starts ``initial`` full (of 30 ml); at
    period ``load_at`` it is given 5 ml."""
    store = Store(capacity=3.0e-5, retained=0.0, held=1.0, split=1.0, bins=1, initial=initial, drip=0.0, pickup=0.0,
                  scrape=True, fill="profile")
    nose = Applicator(body="nose", shape="roller", size=(R, HALF), axis=(1.0, 0.0, 0.0),
                      center=(0.0, 0.0, BOTTOM + R - Z), stiffness=1.0e3, friction=0.0, store=store)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(100, 60)),), material=SEALANT, inertia=False,
                       max_depth=1.6e-2, tools=(nose,))

    def init(sim) -> None:
        c = flow.cell_centres(0).reshape(-1, 3)
        joint = np.abs(c[:, 1]) < HALF_JOINT
        flow.set_input("base", np.where(joint, BACKER, 0.0).reshape(1, -1).astype(np.float32))
        flow.set_input("paste_mask", joint.astype(np.float32).reshape(1, -1))
        flow.set_depth(np.where(joint, top(c[:, 0], c[:, 1]) - BACKER, 0.0).reshape(1, -1))
        sim.state.joint_qd.assign(np.array([speed, rise], np.float32))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=carriage_scene(X0), init=init, ground=False, gravity=(0.0, 0.0, 0.0),
                                     worlds=1, dt=2.0e-3), [flow], "cpu")
    c = flow.cell_centres(0).reshape(-1, 3)
    h0, load0 = flow.h.numpy()[0].copy(), float(flow.tool_load.numpy().sum())
    for i in range(periods):
        if i == load_at:
            flow.tool_load.assign(np.full((1, 1), 5.0e-6))
        sim.step_period()
    return flow, sim, c, h0, load0


def test_a_lifting_nose_that_grazes_the_bead_past_its_end_lays_nothing_under_its_raised_arc():
    """SOFT-2's lift: the nose sits on a joint cut just under it, with a tall bead (6 mm proud) starting at its leading
    end, and lifts 3 mm while it creeps 3 mm along: its end cuts the bead all the way up, and the cells it covered
    from the start fill no higher than where it was (GA's rule filled every one up to its raised arc)."""
    bead_x = X0 + HALF

    def top(x, y):
        return np.where(x < bead_x, arc(y, up=2.0e-5), 6.0e-3)
    flow, sim, c, h0, load0 = nose_run(top, speed=0.01, rise=0.01, periods=15)
    h = flow.h.numpy()[0]
    joint = np.abs(c[:, 1]) < HALF_JOINT
    behind = joint & (c[:, 0] < bead_x)
    q = sim.state.body_q.numpy()[1]
    assert q[0] - X0 > 0.0025 and q[2] - Z > 0.0025                                     # it crept and lifted
    assert float((h0 - h)[joint & (c[:, 0] > bead_x)].max()) > 2.0e-3                  # its end cut the bead
    assert float(flow.tool_load.numpy().sum()) > load0
    assert float((h - h0)[behind].max()) <= 0.0, float((h - h0)[behind].max())        # and laid nothing behind
    r_ = flow.read()
    start = float((h0 * flow.tables.area_np).sum())
    assert abs(float(r_["ledger"][0]) - start) <= 1e-12 * start
    assert abs(float(r_["store_ledger"][0]) - 1.5e-5) <= 1e-12 * 1.5e-5


def test_a_lifting_nose_fills_the_lows_under_it_only_to_where_it_rested():
    """An empty nose on both arrises over a joint 1 mm low lifts at 1 mm/s (1.25 µm per exchange). Given 5 ml at its
    third period, 40 µm up and still on the arrises, it fills the lows under it to where it was at its first exchange
    (1.25 µm up), not to its raised arc (80 µm up at the end, as GA's rule would)."""
    flow, sim, c, h0, _ = nose_run(lambda x, y: arc(y) - 1.0e-3, speed=0.0, rise=1.0e-3, periods=4, initial=0.0,
                                   load_at=2)
    h, b = flow.h.numpy()[0], flow.base.numpy()[0]
    under = (np.abs(c[:, 1]) < HALF_JOINT) & (np.abs(c[:, 0] - X0) < HALF - 1.0e-3)
    assert under.sum() >= 200
    assert float(sim.state.body_q.numpy()[1][2]) - Z > 7.5e-5                          # it lifted 80 µm
    over = (b + h - arc(c[:, 1]))[under]
    assert float(over.min()) >= 0.0 and float(over.max()) <= 2.0e-6, (float(over.min()), float(over.max()))


def test_a_nose_sliding_at_its_height_still_fills_the_lows_it_passes():
    """Not rising, the nose fills a 1 mm low under it to its arc, as GA's does."""
    flow, sim, c, h0, _ = nose_run(lambda x, y: arc(y) - 1.0e-3, speed=0.05, rise=0.0, periods=4)
    h, b = flow.h.numpy()[0], flow.base.numpy()[0]
    y = c[:, 1]
    under = (np.abs(y) < HALF_JOINT) & (np.abs(c[:, 0] - float(sim.state.body_q.numpy()[1][0])) < HALF - 1.0e-3)
    assert under.sum() >= 100
    assert np.allclose((b + h)[under], arc(y)[under], rtol=0.0, atol=2.0e-8)
