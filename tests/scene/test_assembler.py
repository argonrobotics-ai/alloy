# SPDX-License-Identifier: Apache-2.0
"""SceneAssembler end to end on CPU: routing, template, replicate, pair table, entries, readback (CARD-3)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.testing import scene as S

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


def assembler(lib):
    return SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS)))


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def test_box_on_table_layout():
    card, lib = S.box_on_table(worlds=3)
    built = assembler(lib).assemble(card, "cpu")
    m = built.model
    assert built.world_count == 3 and m.body_count == 3 and m.shape_count == 6
    assert [e.name for e in built.entries] == ["mujoco"]
    assert built.entries[0].bodies == (0, 1, 2) and built.entries[0].target_dt == pytest.approx(1.25e-3)
    assert built.coupler.kind == "none"
    h = built.handles["box"]
    assert h.decision.route == Route.RIGID and len(h.world_stride.body) == 1
    sm = m.alloy.shape_material.numpy()
    assert sm.tolist() == [built.pair_table.material_index("wood"), built.pair_table.material_index("abs")] * 3
    np.testing.assert_allclose(m.body_mass.numpy(), 1050.0 * 0.06**3, rtol=1e-6)
    assert built.model_digest and built.report.decisions["table"].route == Route.STATIC


def test_pair_table_values():
    card, lib = S.box_on_table()
    built = assembler(lib).assemble(card, "cpu")
    t = built.pair_table
    i, j = t.material_index("abs"), t.material_index("wood")
    row = t.table.numpy()[i, j]
    hm = lambda a, b: 2 * a * b / (a + b)  # noqa: E731  Drake harmonic, unsourced pair
    np.testing.assert_allclose(row[:3], [hm(0.44, 0.55), hm(0.4, 0.5), hm(0.3, 0.4)], rtol=1e-6)
    assert ("abs", "wood") in built.report.unsourced_pairs
    assert t.stiffness.numpy()[i, j] == pytest.approx(2e6 * 2e6 / 4e6, rel=1e-6)   # wood default K 2e6


@pytest.mark.parametrize("name", [n for n, _, _ in S.mixed_scenes()])
def test_card3_readback_mixed(name):
    _, card, lib = next(s for s in S.mixed_scenes() if s[0] == name)
    built = assembler(lib).assemble(card, "cpu")     # readback raises ReadbackError on any silent drop
    assert built.report.decisions
    fields = {f for f, _ in built.report.unsupported}
    for obj in built.handles:
        assert not any(f.startswith(f"{obj}.material.surface.mu_dynamic") for f in fields
                       if built.handles[obj].decision.route in (Route.RIGID,))


def test_randomized_values_reach_model():
    _, card, lib = next(s for s in S.mixed_scenes() if s[0] == "randomized")
    built = assembler(lib).assemble(card, "cpu")
    dens = built.randomized[("box", "material.density")]
    assert dens.shape == (4,) and np.all((dens >= 900) & (dens <= 1200)) and len(set(dens.round(6))) == 4
    h = built.handles["box"]
    bodies = [h.template.body.start + w * len(h.world_stride.body) for w in range(4)]
    np.testing.assert_allclose(built.model.body_mass.numpy()[bodies], dens * 0.06**3, rtol=1e-6)
    mu = built.randomized[("box", "material.surface.mu_dynamic")]
    i = built.pair_table.material_index("abs@box")
    np.testing.assert_allclose(built.pair_table.world_table.numpy()[:, i, i, 1], mu, rtol=1e-6)



def _box_masses(built) -> np.ndarray:
    h = built.handles["box"]
    bodies = [h.template.body.start + w * len(h.world_stride.body) for w in range(built.world_count)]
    return built.model.body_mass.numpy()[bodies].astype(np.float64)


def test_a_redraw_does_not_compound():
    """SCR-DC3-8 (WP12): density/youngs scale the Model by new/previous draw, so redrawing at the same seed is a no-op
    and a different seed then the assembly seed gets the assembly values back (it multiplied mass by 1.11-1.14)."""
    from alloy.scene.randomize import apply_randomization, draw_seed
    from alloy.scene.readback import readback

    _, card, lib = next(s for s in S.mixed_scenes() if s[0] == "randomized")
    built = assembler(lib).assemble(card, "cpu")
    m0 = _box_masses(built)
    assert draw_seed(built) == 0
    apply_randomization(built, 0)
    np.testing.assert_allclose(_box_masses(built), m0, rtol=1e-6)
    assert readback(built) == []
    apply_randomization(built, 5)
    assert draw_seed(built) == 5 and not np.allclose(_box_masses(built), m0, rtol=1e-3)
    apply_randomization(built, 0)
    np.testing.assert_allclose(_box_masses(built), m0, rtol=1e-6)


def test_reset_redraws_at_the_assembly_seed():
    """Simulation.reset redraws at draw_seed(built) (the assembly draw), not at the simulation's own seed."""
    from alloy.sim import Simulation

    _, card, lib = next(s for s in S.mixed_scenes() if s[0] == "randomized")
    built = assembler(lib).assemble(card, "cpu")
    m0 = _box_masses(built)
    sim = Simulation(built, seed=7)
    sim.step_period()
    sim.reset()
    np.testing.assert_allclose(_box_masses(built), m0, rtol=1e-6)


def test_discrete_solref_below_two_dt_is_warned():
    import types

    from alloy.scene.finish import discrete_solref_warnings

    class _Arr:
        def __init__(self, v):
            self.v = np.asarray(v, np.float32)
            self.shape = self.v.shape

        def numpy(self):
            return self.v

    mj = types.SimpleNamespace(geom_solref=_Arr([[2.5e-4, 1.0], [0.02, 1.0], [-1e4, -10.0]]),
                               pair_solref=_Arr(np.zeros((0, 2))))
    model = types.SimpleNamespace(mujoco=mj)
    entry = types.SimpleNamespace(name="mujoco", capability="mujoco", solver_kwargs={"integrator": "discrete"},
                                  target_dt=6.25e-4, substeps=1)
    (msg,) = discrete_solref_warnings(model, [entry], 3)       # direct (negative) solref is not a time constant
    assert "1 geom_solref rows" in msg and "0.00025" in msg
    entry.solver_kwargs = {"integrator": "implicitfast"}      # MJW clamps τ ≥ 2·dt itself
    assert discrete_solref_warnings(model, [entry], 3) == []
