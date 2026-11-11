# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-1 (WP12 part): the topology's ``adjust_contacts`` (WP0's helper) and each contact adjuster beside it run
after collide and before the solver step; ``wire`` builds a ``Blade`` per world for each object card with a ``blade:``
block, and WP10's ``BladeContact`` over the controller's bonds when that module exists; with blades, the controller's
cut-front arrays reach WP5's health monitor (SCR-SC5-3) (CPU)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.cards.library import CardLibrary
from alloy.cards.types import BladeSpec, SceneCard
from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.integration import IntegratedSimulation
from alloy.integration.wiring import _blades, _contact_adjusters, _health_topology
from alloy.precision.types import PrecisionConfig
from alloy.scene.types import IndexRanges
from alloy.testing import integration as I
from alloy.testing import scene as S
from alloy.testing.fakes import FakeTopology


class _Hooked(FakeTopology):
    """Records the exchange order; ``adjust_contacts`` is a no-op."""

    def __init__(self, world_count: int, log: list[str]) -> None:
        super().__init__(world_count)
        self.log = log

    def pre_step(self, state, contacts, dt) -> None:
        super().pre_step(state, contacts, dt)
        self.log.append("pre_step")

    def adjust_contacts(self, state, contacts) -> None:
        super().adjust_contacts(state, contacts)
        self.log.append("adjust_contacts")


class _Adjuster:
    """A contact adjuster beside the topology (the role of WP10's BladeContact); a no-op."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def adjust_contacts(self, state, contacts) -> None:
        self.log.append("adjuster")


def _sim(topology, adjusters=()):
    card, lib = S.box_on_table(worlds=2)
    built = I.assemble(card, lib, "cpu", 2)
    return IntegratedSimulation(built, precision=PrecisionConfig(native_contacts="newton"), topology=topology,
                                use_graph=False, contact_adjusters=adjusters)


def _logged(sim, log: list[str]) -> None:
    collide, step = sim.contact_layer.collide, sim.stepper.step

    def logged_collide(*a, **k):
        log.append("collide")
        return collide(*a, **k)

    def logged_step(*a, **k):
        log.append("step")
        return step(*a, **k)

    sim.contact_layer.collide, sim.stepper.step = logged_collide, logged_step


@pytest.mark.compile_heavy
def test_adjust_contacts_after_collide_before_step_and_noop_is_bitwise():
    with isolated_registry(empty=True):
        log: list[str] = []
        topo = _Hooked(2, log)
        hooked = _sim(topo, [_Adjuster(log)])
        _logged(hooked, log)
        hooked.step_period()
        plain = _sim(None)
        plain.step_period()
    per = ["pre_step", "collide", "adjust_contacts", "adjuster", "step"]
    k = len(per)
    assert log[:k] == per and log == per * (len(log) // k) and len(log) // k == hooked.steps == topo.adjusts
    np.testing.assert_array_equal(I.digests(hooked), I.digests(plain))       # no-op hooks change nothing


def _built(blade, worlds: int = 3, bodies: range = range(1, 2), shapes: range = range(2, 3)):
    handle = SimpleNamespace(name="knife", blade=blade,
                             template=IndexRanges(bodies, shapes, *(range(0),) * 5),
                             world_stride=IndexRanges(range(4), range(6), *(range(0),) * 5))
    plain = SimpleNamespace(name="board", blade=None, template=IndexRanges.empty(), world_stride=IndexRanges.empty())
    return SimpleNamespace(handles={"knife": handle, "board": plain}, world_count=worlds, model="model")


def test_blade_per_world_on_the_card_body():
    spec = BladeSpec(edge_a=(-0.06, -0.0125, 0.0), edge_b=(0.06, -0.0125, 0.0), edge_radius=2e-4)
    blades = _blades(_built(spec))
    assert [(b.body, b.shape) for b in blades] == [(1, 2), (5, 8), (9, 14)]
    assert all(b.edge_a == spec.edge_a and b.edge_b == spec.edge_b and b.edge_radius == 2e-4 for b in blades)
    assert _blades(_built(None)) == []                      # no blade: block, no blades (controller unchanged)


def test_blade_without_a_body_is_refused():
    spec = BladeSpec(edge_a=(0.0, 0.0, 0.0), edge_b=(0.01, 0.0, 0.0))
    with pytest.raises(AssemblyError, match="'knife' has a blade: block but no body"):
        _blades(_built(spec, bodies=range(0)))


@pytest.mark.compile_heavy
def test_assembled_knife_card_gets_a_blade_per_world():
    """The library's knife_blade_steel (blade: block) through WP2's assembler: ObjectHandle.blade -> one Blade per
    world on the knife's own body and shape."""
    card = SceneCard(robots=(), objects=(S.place("knife_blade_steel", z=0.1),), pairs=(), worlds=2)
    with isolated_registry(empty=True):
        built = I.assemble(card, CardLibrary(), "cpu", 2)
    spec = built.handles["knife_blade_steel"].blade
    blades = _blades(built)
    assert spec is not None and len(blades) == 2
    shape_body = built.model.shape_body.numpy()
    body_world = built.model.body_world.numpy()
    assert [int(body_world[b.body]) for b in blades] == [0, 1]
    assert all(int(shape_body[b.shape]) == b.body and b.edge_a == spec.edge_a for b in blades)


def test_blade_contact_beside_the_controller(monkeypatch):
    """With blades and a controller, wiring builds WP10's ``BladeContact(model, controller.bonds, blades)``; without
    the module (not on main yet), no blades or no controller, there is no adjuster."""
    made = []
    fake = ModuleType("alloy.topology.blade_contact")
    fake.BladeContact = lambda model, bonds, blades: made.append((model, bonds, len(blades))) or "adjuster"
    spec = BladeSpec(edge_a=(0.0, 0.0, 0.0), edge_b=(0.01, 0.0, 0.0))
    topo, notes = SimpleNamespace(bonds="bonds"), []
    monkeypatch.setitem(sys.modules, "alloy.topology.blade_contact", fake)
    assert _contact_adjusters(_built(spec), topo, notes) == ["adjuster"] and made == [("model", "bonds", 3)]
    assert _contact_adjusters(_built(None), topo, notes) == [] and _contact_adjusters(_built(spec), None, notes) == []
    monkeypatch.setitem(sys.modules, "alloy.topology.blade_contact", None)      # import fails
    assert _contact_adjusters(_built(spec), topo, notes) == [] and "not on main" in notes[-1]


def test_cut_front_reaches_health_with_blades():
    """SCR-SC5-3: with blades, wiring hands the controller's cut-front arrays (WP10's predicate) to WP5's real
    HealthMonitor, which checks their dtype and length; without blades, or on a controller without the predicate
    (before WP10's branch merges), it sets nothing. ``body_active`` (SCR-WP10-2) is set either way."""
    from alloy.integration import build_simulation

    card, lib = S.scene([S.table(), S.box(), S.cloth()], [S.place("table"), S.place("box", z=0.051),
                                                         S.place("towel", z=0.12)], worlds=2)     # box on cloth
    with isolated_registry(empty=True):
        sim = build_simulation(I.assemble(card, lib, "cpu", 2), "standard", graph="off")
    health, model, calls = sim.health, sim.model, []
    cut_front, body_active = health.set_cut_front, health.set_body_active
    health.set_cut_front = lambda blade, half: calls.append(("cut_front", blade, half)) or cut_front(blade, half)
    health.set_body_active = lambda active: calls.append(("body_active", active)) or body_active(active)
    assert model.particle_count > 0 and model.body_count > 0
    for blades, predicate in ((0, True), (2, False)):
        notes = []
        _health_topology(health, _cut_topology(model, blades, predicate), notes)
        assert [c[0] for c in calls] == ["body_active"] and notes == []
        calls.clear()
    topo, notes = _cut_topology(model, 2, True), []
    _health_topology(health, topo, notes)
    assert calls == [("body_active", topo.body_active), ("cut_front", topo.cut_front_blade, topo.blade_half_thickness)]
    assert "SCR-SC5-3" in notes[-1]


def _cut_topology(model, blades: int, predicate: bool) -> SimpleNamespace:
    """The controller attributes the wiring reads (WP10's TopologyController, branch wp10-sc5-3)."""
    dev = model.device
    topo = SimpleNamespace(body_active=wp.ones(model.body_count, dtype=wp.int8, device=dev),
                           blades=SimpleNamespace(count=blades))
    if predicate:
        topo.cut_front_blade = wp.full(model.particle_count, -1, dtype=wp.int32, device=dev)
        topo.blade_half_thickness = wp.zeros(model.body_count, dtype=wp.float32, device=dev)
    return topo
