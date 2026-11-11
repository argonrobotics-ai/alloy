# SPDX-License-Identifier: Apache-2.0
"""Codec: strict decoding with dotted field paths, Q forms, tagged shapes, default-omitting encoding, round trips."""

from __future__ import annotations

import math

import pytest

from alloy.cards import types as ct
from alloy.cards.loader import dump_card, loads_card, to_data
from alloy.core.errors import CardError
from alloy.core.quantity import Q
from alloy.topology.types import TopologyBudget

HEAD = "schema: alloy.object/1\nname: o\nmaterial: m\n"


def _obj(extra: str, shape: str = "shape: {primitive: {kind: box, size: [1 cm, 1 cm, 1 cm]}}\n") -> ct.ObjectCard:
    return loads_card(HEAD + shape + extra, ct.ObjectCard)


def test_q_forms() -> None:
    o = _obj("loads:\n  grip_force: 12\n  tolerance: {value: 0.2 mm, source: measured, unc: 5%, note: caliper}\n"
             "  bend_force: {value: 1.0, source: default}\n")
    assert o.loads.grip_force == Q(12.0, "default")
    assert o.loads.tolerance.value == pytest.approx(2e-4) and o.loads.tolerance.unc == pytest.approx((1.9e-4, 2.1e-4))
    assert o.loads.bend_force == Q(1.0, "default")
    assert o.shape == ct.Primitive("box", (0.01, 0.01, 0.01))


@pytest.mark.parametrize("extra, field", [
    ("loads: {grip_force: {value: 1, source: catalog}}\n", "loads.grip_force"),          # note required
    ("loads: {grip_force: {value: 1}}\n", "loads.grip_force.source"),
    ("loads: {grip_force: {source: catalog, note: x}}\n", "loads.grip_force.value"),
    ("loads: {grip_force: {value: 1, source: hearsay, note: x}}\n", "loads.grip_force.source"),
    ("loads: {grip_force: {value: 1, source: catalog, note: x, extra: 1}}\n", "loads.grip_force.extra"),
    ("loads: {grip_force: {value: 1, source: catalog, note: x, unc: [1, 2, 3]}}\n", "loads.grip_force.unc"),
    ("loads: {grip_force: {value: 1, source: catalog, note: 7}}\n", "loads.grip_force.note"),
    ("loads: {grip: 1}\n", "loads.grip"),
    ("features: {thickness: 1 kg}\n", "features.thickness"),
    ("features: {self_contact: 1}\n", "features.self_contact"),
    ("features: {creases: [{a: [0, 0], b: [1, 1, 1], stiffness: 1}]}\n", "features.creases.0.a"),
    ("precision_partners: other\n", "precision_partners"),
    ("part_materials: [a]\n", "part_materials"),
    ("budget: {tets: 1.5}\n", "budget.tets"),
    ("motion: true\n", "motion"),
    ("topology: {spare_vertices: x}\n", "topology.spare_vertices"),
    ("route: {route: rigid}\n", "route.reason"),
])
def test_decode_errors_name_the_field(extra: str, field: str) -> None:
    with pytest.raises(CardError) as exc:
        _obj(extra)
    assert exc.value.field == field, exc.value


@pytest.mark.parametrize("shape, field", [
    ("shape: {primitive: {kind: blob, size: [1]}}\n", "shape.kind"),
    ("shape: {primitive: {kind: box}}\n", "shape.size"),
    ("shape: {mesh: {path: a.obj}}\n", "shape.sha256"),
    ("shape: {mesh: {path: a.obj, sha256: x}, curve: {}}\n", "shape"),
    ("shape: {box: {}}\n", "shape.box"),
    ("shape: box\n", "shape"),
    ("shape: {curve: {points: [[0, 0, 0], [1, 0]], radius: 1}}\n", "shape.points.1"),
])
def test_shape_variant_errors(shape: str, field: str) -> None:
    with pytest.raises(CardError) as exc:
        _obj("", shape)
    assert exc.value.field == field, exc.value


def test_every_shape_variant_round_trips() -> None:
    shapes = [ct.Primitive("capsule", (0.01, 0.02)), ct.MeshSource("meshes/a.obj", "ab" * 32, (1.0, 2.0, 1.0), "decompose", 8),
              ct.SDFSource(analytic_id="iso_thread:M10x1.5:6g", voxel=1.6e-5), ct.TetSource(from_mesh="meshes/a.obj", edge_length=2e-3),
              ct.CurveSource(((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 0.0, 0.0)), 1.75e-3),
              ct.ParticleSource(((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)), 2e-3, 3e-4), ct.USDSource("a.usd", "/World/obj"),
              ct.MJCFSource("robots/a.xml", "drawer")]
    for s in shapes:
        card = ct.ObjectCard("o", s, "m")
        assert loads_card(dump_card(card), ct.ObjectCard) == card


def _full_object() -> ct.ObjectCard:
    q = lambda v, n="x": Q(v, "model", (v * 0.9, v * 1.1), n)  # noqa: E731
    return ct.ObjectCard(
        name="knife", shape=ct.Primitive("box", (0.06, 0.0125, 6e-4)), material="steel", part_materials={"b": "x", "a": "y"},
        motion="kinematic", loads=ct.LoadEnvelope(grip_force=q(5.0), design_force=Q(20.0, "measured", None, "load cell")),
        features=ct.Features(thickness=q(1.2e-3), creases=(ct.Crease((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), q(0.05)),),
                             self_contact=True, passage_width=q(0.01), precision_anchor=(0.0, 0.0, 0.001), min_wall=q(1.2e-3)),
        precision_partners=("board:slot",),
        precision_features=(ct.PrecisionFeature("edge", ((-0.06, -0.013, -0.001), (0.06, -0.012, 0.001)), "hydroelastic", 5e-5,
                                                None, q(2e-4)),),
        budget=ct.Budget(4, 100, 1000), topology=TopologyBudget(1, 2, 3, 8),
        blade=ct.BladeSpec((-0.06, -0.0125, 0.0), (0.06, -0.0125, 0.0), 5e-5, 100.0, 250.0),
        route=ct.RouteOverride("rigid", "tool"))


def _full_scene() -> ct.SceneCard:
    return ct.SceneCard(
        robots=(ct.RobotRef("robots/arm.xml", "mjcf", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), "arm"),),
        objects=(ct.Placement("knife", (0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
                              (ct.Randomization("material.density", 7000.0, 8000.0), ct.Randomization("pose.0", -1e-3, 1e-3))),),
        pairs=(ct.PairCard("steel", "wood", Q(0.5, "catalog", None, "x"), Q(0.4, "catalog", None, "x"),
                           Q(0.3, "model", None, "y"), Q(1e-5, "model", None, "z"), "static"),),
        attachments=(ct.Attachment("knife", "arm:hand", "joint", Q(5.0, "model", None, "w"), Q(0.1, "model", None, "s"),
                                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),),
        emitters=(ct.EmitterSpec("emitter", "water", (0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0), (0.01, 0.01, 0.001), 2e-5,
                                 (0.0, 0.0, -0.5), 0.1, 2.0), ct.EmitterSpec("sink", "water", (1.0, 0, 0, 0, 0, 0, 1.0), (0.1, 0.1, 0.1))),
        worlds=64, control_dt=0.01, combine="max")


def test_full_featured_round_trip_and_idempotence() -> None:
    for card in (_full_object(), _full_scene()):
        text = dump_card(card)
        back = loads_card(text, type(card))
        assert back == card and dump_card(back) == text


def test_encoding_omits_defaults_and_writes_schema_first() -> None:
    data = to_data(ct.ObjectCard("o", ct.Primitive("sphere", (0.01,)), "m"))
    assert list(data) == ["schema", "name", "shape", "material"]
    assert data["shape"] == {"primitive": {"kind": "sphere", "size": [0.01]}}


def test_negative_zero_survives() -> None:
    card = ct.ObjectCard("o", ct.Primitive("box", (0.01, 0.01, 0.01)), "m", features=ct.Features(precision_anchor=(-0.0, 0.0, 0.0)))
    back = loads_card(dump_card(card), ct.ObjectCard)
    assert math.copysign(1.0, back.features.precision_anchor[0]) == -1.0


def test_ints_become_floats_on_float_fields() -> None:
    o = _obj("features: {precision_anchor: [0, 0, 1]}\n")
    assert all(isinstance(v, float) for v in o.features.precision_anchor)


def test_randomization_units_follow_the_target_leaf() -> None:
    s = loads_card("schema: alloy.scene/1\nrobots: []\nobjects:\n- object: o\n  pose: [0, 0, 0, 0, 0, 0, 1]\n  randomize:\n"
                   "  - {field: material.elastic.youngs, lo: 1 MPa, hi: 2 GPa}\n  - {field: pose.2, lo: -1 mm, hi: 1 cm}\n"
                   "  - {field: loads.contact_speed, lo: 10 mm/s, hi: 0.5}\n", ct.SceneCard)
    r = s.objects[0].randomize
    assert (r[0].lo, r[0].hi) == (1e6, 2e9) and (r[1].lo, r[1].hi) == pytest.approx((-1e-3, 1e-2))
    assert r[2].lo == pytest.approx(0.01)


def test_scr_units_rate_tear_energy_and_rod_stiffness() -> None:
    s = loads_card("schema: alloy.scene/1\nrobots: []\nobjects: []\nemitters:\n- {kind: emitter, material: water, "
                   "pose: [0, 0, 0, 0, 0, 0, 1], half_extents: [1 mm, 1 mm, 1 mm], rate: 1.2 L/min}\n", ct.SceneCard)
    assert s.emitters[0].rate == pytest.approx(2e-5)
    m = loads_card("schema: alloy.material/1\nname: m\ndensity: 1000\nsurface: {mu_static: 0.5, mu_dynamic: 0.4, "
                   "restitution: 0}\nfailure: {tear_energy: 2 kJ/m^2}\n", ct.MaterialCard)
    assert m.failure.tear_energy.value == 2000.0
    o = _obj("features: {bend_stiffness: 5.67e-5, twist_stiffness: {value: 9.29e-5, source: model, note: x}}\n")
    assert o.features.bend_stiffness == Q(5.67e-5, "default") and o.features.twist_stiffness.value == 9.29e-5
