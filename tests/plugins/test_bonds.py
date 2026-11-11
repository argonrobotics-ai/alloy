# SPDX-License-Identifier: Apache-2.0
"""Bonds (film-to-rigid cohesive bonds, route D): cards, layout helpers, wiring, the plug-in host's force carry in
coupled scenes, and the laws (CPU, on the coupled rig; ``alloy.plugins.bonds.laws``)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from alloy.plugins.base import ForcePlugin
from alloy.plugins.bonds import ROWS_7818, TAPE_7818, Bonds, adhesive
from alloy.plugins.bonds.adhesives import g_of_c
from alloy.plugins.bonds.layout import nearest_grip, vertex_areas, vertex_spacing, world_ranges
from alloy.plugins.errors import PluginError

SHEET = {"stainless_steel": (3.8, 4.7), "polycarbonate": (4.0, 5.1), "polypropylene": (1.8, 2.0),
         "glass": (3.7, 5.7), "hdpe": (1.8, 2.6), "ldpe": (1.3, 2.2)}      # N/cm, 3M 7818 May 2017, 10 min RT


@pytest.mark.parametrize("sub", sorted(SHEET))
def test_cards_reproduce_the_sheet_rows(sub):
    """G(theta) is calibrated so a 90 and a 180 deg peel give the sheet's rows (N/cm x 100 = N/m)."""
    card = TAPE_7818[sub]
    assert card.peel_force(90.0, 1.0) == pytest.approx(100.0 * SHEET[sub][0], rel=1e-12)
    assert card.peel_force(180.0, 1.0) == pytest.approx(100.0 * SHEET[sub][1], rel=1e-12)
    assert ROWS_7818[sub] == pytest.approx((100.0 * SHEET[sub][0], 100.0 * SHEET[sub][1]))


def test_g_is_held_below_90_and_linear_above():
    assert g_of_c(380.0, 940.0, 0.3) == 380.0
    assert g_of_c(380.0, 940.0, 1.5) == pytest.approx(660.0)
    assert g_of_c(380.0, 940.0, 2.0) == pytest.approx(940.0)


def test_the_371_kraft_liner_card_is_one_g_kendall():
    """SOFT-1's 3M 371 box-sealing tape on kraft liner: the liner tears in its plane, one G = 200 J/m² (ESTIMATE, the
    low end of kraftliner Scott bond), so a 48 mm strip peels at Kendall's G·b/(1 - cos theta): 9.6 N at 90 deg and
    4.8 N at 180 deg. An unknown name lists every card, both tapes."""
    card = adhesive("3m_371/kraft_liner")
    assert card.peel_force(90.0, 0.048) == pytest.approx(9.6, rel=1e-12)
    assert card.peel_force(180.0, 0.048) == pytest.approx(4.8, rel=1e-12)
    with pytest.raises(KeyError, match="3m_371/kraft_liner.*3m_7818/glass"):
        adhesive("3m_371/steel")


def test_unknown_card_and_config_mismatch_raise():
    with pytest.raises(KeyError):
        adhesive("3m_7818/cardboard")
    with pytest.raises(PluginError):
        Bonds(substrates=("a", "b"), adhesives=("3m_7818/glass",))
    with pytest.raises(PluginError):
        Bonds(stiffness=0.0)


def test_layout_helpers():
    from alloy.solvers.film.rigs import grid_mesh

    v, t = grid_mesh(4, 2, 0.01)
    assert vertex_areas(v, t).sum() == pytest.approx(4 * 2 * 1e-4)
    h = vertex_spacing(v, t)
    assert h.min() >= 0.01 - 1e-12 and h.max() <= math.sqrt(2) * 0.01 + 1e-12
    start = world_ranges(np.array([0, 0, 0, 1, 1, 1]), 2)
    assert start.tolist() == [0, 3, 6]
    x = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 0, 0], [2, 0, 0], [5, 0, 0]], float)
    d, g = nearest_grip(x, np.array([True, False, False, False, False, True]), start)
    assert d.tolist() == [0.0, 1.0, 3.0, 5.0, 3.0, 0.0] and g.tolist() == [0, 0, 0, 5, 5, 5]
    with pytest.raises(PluginError):
        world_ranges(np.array([1, 0]), 2)


def test_needs_a_film_entry():
    """In a MuJoCo-only scene the plug-in refuses to attach (it extends a film solver)."""
    from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation

    def build(b):
        b.add_shape_box(b.add_body(xform=body_xform(0, 0, 0.1), label="box"), hx=0.01, hy=0.01, hz=0.01)
        b.add_particle((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), 1e-3)

    with pytest.raises(PluginError, match="film"):
        build_simulation(TestScene(build=build, worlds=1), [Bonds(substrates=(), adhesives=())], "cpu")


@wp.kernel
def _push(body_f: wp.array[wp.spatial_vector], b: int, fz: float):
    body_f[b] = body_f[b] + wp.spatial_vector(0.0, 0.0, fz, 0.0, 0.0, 0.0)


class _Push(ForcePlugin):
    name = "bonds_test_push"

    @dataclass
    class Config:
        fz: float = -10.0

    def setup(self, ctx):
        self.b = int([i for i, lab in enumerate(ctx.model.body_label) if lab.endswith("gantry_z")][0])

    def apply(self, state, forces, dt):
        wp.launch(_push, dim=1, inputs=[forces.body_f, self.b, self.config.fz], device=self.ctx.model.device)


def test_plugin_force_does_not_pile_up_in_a_coupled_scene():
    """Newton's coupler copies a step's input body_f into its output state. The host must restore both, or a
    constant plug-in force grows by itself every exchange (before the fix a 10 N push sank a 2e4 N/m servo at
    0.8 m/s). The z drive sags (F + m·g)/k_e."""
    from alloy.plugins.bonds import rig as R

    rig = R.PeelRig()
    sim = R.simulation(rig, [_Push()], worlds=1)
    cols = R.gantry_columns(sim)
    for _ in range(12):
        R.set_gantry(sim, cols, 0.0, 5e-3)
        sim.step_period()
    z = float(sim.state.joint_q.numpy()[cols[0, 1]])
    sag = (10.0 + rig.link_mass * 9.81) / rig.drive_ke
    assert 5e-3 - z == pytest.approx(sag, rel=0.02)


def _law(name: str, film_substeps: int = 1):
    from alloy.plugins.bonds.laws import bonds_laws
    from alloy.plugins.harness.run import run_scene

    law = next(x for x in bonds_laws(film_substeps) if x.name == name)
    run = run_scene(law.scene, [Bonds(**dict(law.config or {}))], "cpu")
    assert run.error is None, run.error
    return law.check(run)


@pytest.mark.parametrize("name", ["null", "none_below_dwell"])
def test_law_quick(name):
    """The two shortest laws (about 20 s each on one CPU core)."""
    res = _law(name)
    assert res.passed, res


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["peel_90", "peel_180", "slack", "stall", "forms_above_p_form", "none_below_p_form"])
def test_law_slow(name):
    """Rows (CPU, 1 world, 35-100 s each):
    - peel_90: F/b 388.4 N/m vs 380 (+2.2 %), work / G·b·Δa 1.0039;
    - peel_180: 469.0 vs 470 (-0.2 %), 0.9973;
    - slack: 0 N and 0 mm² after the turn;
    - stall: time-averaged pull 2.038 N vs 2.038 N, nothing released;
    - formation: 1.000 bonded at 2x p_form, 0.000 at 0.4x."""
    res = _law(name)
    assert res.passed, res


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_harness_checks_without_laws():
    """Contract, crash, hook checks, determinism, identical and isolated worlds, graph replay (CPU, ~80 s). Shape ids
    in plug-in state are world-local: with global ids, world 0's state copied into world 1 pulled world 0's grip."""
    import alloy.plugins as ap
    from alloy.plugins.bonds.laws import PEEL

    report = ap.check(Bonds(**PEEL), laws=False, speed=False)
    assert report.passed, report.summary()


class _Zero(ForcePlugin):
    """A forcing plug-in that adds nothing (so the host saves and restores forces around each step)."""

    name = "bonds_test_zero"

    @dataclass
    class Config:
        body: str = "box"

    def setup(self, ctx):
        self.b = int(ctx.bodies(self.config.body).numpy()[0])

    def apply(self, state, forces, dt):
        wp.launch(_push, dim=1, inputs=[forces.body_f, self.b, 0.0], device=self.ctx.model.device)


def test_user_force_acts_every_exchange():
    """A force the user writes once on sim.state (uncoupled MuJoCo) acts every exchange: the host restores it in
    state_out as well (before: every other exchange, as the states ping-ponged). v = F·t/m after one period."""
    from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation

    def build(b):
        box = b.add_body(xform=body_xform(0.0, 0.0, 1.0), label="box")
        b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)

    sim = build_simulation(TestScene(build=build, worlds=1, ground=False, gravity=(0.0, 0.0, 0.0)), [_Zero()], "cpu")
    m = float(sim.model.body_mass.numpy()[0])
    f = sim.state.body_f.numpy()
    f[0, 0] = 2.0                                      # 2 N along x, written once
    sim.state.body_f.assign(f)
    sim.step_period()                                  # one control period, 0.02 s (TestScene default)
    v = float(sim.state.body_qd.numpy()[0, 0])
    assert v == pytest.approx(2.0 * 0.02 / m, rel=1e-4)


@pytest.mark.timeout(180)              # 56 s of the default 60 s on a loaded host-1 (WP13, 11-08)
def test_dwell_with_two_film_substeps():
    """The film entry at 2 substeps per exchange: substeps advance the dwell clock, coupling iterations do not
    (SOFT-1 review (c): restoring before every substep ran the clock at 0.5 s/s and nothing bonded)."""
    res = _law("none_below_dwell", film_substeps=2)
    assert res.passed, res


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["forms_above_p_form", "stall", "peel_90"])
def test_law_with_two_film_substeps(name):
    """forms_above_p_form failed at 2 substeps before the fix (bonded 0.000); stall and peel_90 pin the impulse and
    the arm cap on the exchange dt."""
    res = _law(name, film_substeps=2)
    assert res.passed, res
