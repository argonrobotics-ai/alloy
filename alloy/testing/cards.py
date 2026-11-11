# SPDX-License-Identifier: Apache-2.0
"""Card test helpers (WP1): the CARD-5 corpus of seeded invalid cards.

Each ``BadCase`` breaks one SPEC §3.2 rule on a library card (or a YAML snippet) and names the dotted field that
the rejection must point at. ``run_case`` returns the fields the loader/validator named; a case passes when the
expected field is among them. Cases that need broken assets write them into the library's first root, so they
must run on a copy of the library (``tests/conftest.py::tmp_library``), never on the repo's ``library/``.
"""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Callable
from dataclasses import dataclass

from alloy.cards import types as ct
from alloy.cards.library import CardLibrary, default_root
from alloy.cards.loader import loads_card
from alloy.cards.validate import errors, validate_material, validate_object, validate_pair, validate_scene
from alloy.core.errors import CardError
from alloy.core.quantity import Q
from alloy.topology.types import TopologyBudget

Runner = Callable[[CardLibrary], list[str]]


@dataclass(frozen=True)
class BadCase:
    id: str
    rule: str
    field: str
    run: Runner


def _fields(issues: list[ct.CardIssue]) -> list[str]:
    return [i.field for i in errors(issues)]


def _mat(name: str, **surface_or_top: object) -> Runner:
    def run(lib: CardLibrary) -> list[str]:
        m = lib.material(name)
        top = {k: v for k, v in surface_or_top.items() if not k.startswith(("s_", "e_", "p_", "f_"))}
        sub = {"s_": ("surface", m.surface), "e_": ("elastic", m.elastic), "p_": ("plastic", m.plastic or ct.Plastic()),
               "f_": ("fluid", m.fluid)}
        for prefix, (attr, base) in sub.items():
            kw = {k[2:]: v for k, v in surface_or_top.items() if k.startswith(prefix)}
            if kw:
                top[attr] = dc.replace(base, **kw)
        return _fields(validate_material(dc.replace(m, **top)))
    return run


def _yaml(cls: type, text: str) -> Runner:
    def run(lib: CardLibrary) -> list[str]:
        try:
            card = loads_card(text, cls, "seeded")
        except CardError as exc:
            return [exc.field]
        if cls is ct.ObjectCard:
            return _fields(validate_object(card, lib))
        if cls is ct.MaterialCard:
            return _fields(validate_material(card))
        return _fields(validate_scene(card, lib)) if cls is ct.SceneCard else _fields(validate_pair(card, lib))
    return run


def _obj(name: str, mutate: Callable[[ct.ObjectCard, CardLibrary], ct.ObjectCard]) -> Runner:
    def run(lib: CardLibrary) -> list[str]:
        return _fields(validate_object(mutate(lib.peek_object(name), lib), lib))
    return run


def _scene(mutate: Callable[[ct.SceneCard], ct.SceneCard], base: str = "demo_peg_d12") -> Runner:
    def run(lib: CardLibrary) -> list[str]:
        return _fields(validate_scene(mutate(lib.peek_scene(base)), lib))
    return run


def _pair(mat_a: str, mat_b: str, **kw: object) -> Runner:
    def run(lib: CardLibrary) -> list[str]:
        p = lib.pair(mat_a, mat_b)
        assert p is not None, (mat_a, mat_b)
        return _fields(validate_pair(dc.replace(p, **kw), lib))
    return run


def _q(v: float, note: str = "seeded") -> Q:
    return Q(v, "model", None, note)


def _feat(card: ct.ObjectCard, **kw: object) -> ct.ObjectCard:
    return dc.replace(card, features=dc.replace(card.features, **kw))


def _pf(card: ct.ObjectCard, i: int = 0, **kw: object) -> ct.ObjectCard:
    feats = list(card.precision_features)
    feats[i] = dc.replace(feats[i], **kw)
    return dc.replace(card, precision_features=tuple(feats))


def _write_open_mesh(lib: CardLibrary) -> ct.MeshSource:
    """A cube with one face missing, written into the (temporary) first root."""
    root = lib.roots[0]
    if root == default_root():
        raise RuntimeError("seeded asset cases must run on a copy of the library (tmp_library)")
    v = [(x, y, z) for x in (0.0, 0.02) for y in (0.0, 0.02) for z in (0.0, 0.02)]
    f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2)]
    path = root / "meshes" / "seeded_open_box.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in v) + "".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in f))
    from alloy.cards.geometry import sha256_file  # noqa: PLC0415

    return ct.MeshSource("meshes/seeded_open_box.obj", sha256_file(path))


def _write_inverted_mesh(lib: CardLibrary) -> ct.MeshSource:
    """A closed 50 x 50 x 5 mm plate wound inside out, written into the (temporary) first root."""
    root = lib.roots[0]
    if root == default_root():
        raise RuntimeError("seeded asset cases must run on a copy of the library (tmp_library)")
    from alloy.cards import meshgen as mg  # noqa: PLC0415
    from alloy.cards.geometry import MeshData, sha256_file  # noqa: PLC0415

    plate = mg.extrude(mg.rect(0.05, 0.05), 0.0, 0.005)
    path = root / "meshes" / "seeded_inverted_plate.obj"
    mg.write_obj(path, MeshData(plate.vertices, plate.faces[:, ::-1].copy()))
    return ct.MeshSource("meshes/seeded_inverted_plate.obj", sha256_file(path))


def _tearable(lib: CardLibrary) -> str:
    m = lib.material("cotton_woven")
    torn = dc.replace(m, name="seeded_cotton_tearable", failure=ct.Failure(tear_energy=_q(300.0)))
    lib.put(torn, reason="CARD-5 seeded case: tear energy without a topology budget")
    return torn.name


OBJ_HEAD = "schema: alloy.object/1\nname: seeded\nshape: {primitive: {kind: box, size: [0.01, 0.01, 0.01]}}\nmaterial: steel_carbon\n"
MAT_HEAD = ("schema: alloy.material/1\nname: seeded\nsurface: {mu_static: 0.5, mu_dynamic: 0.4, restitution: 0.5}\n")
SCENE_HEAD = "schema: alloy.scene/1\nrobots: []\n"


def _material_cases() -> list[BadCase]:
    m = "steel_carbon"
    return [
        BadCase("density_low", "density in [10, 25000]", "density", _mat(m, density=_q(5.0))),
        BadCase("density_high", "density in [10, 25000]", "density", _mat(m, density=_q(30000.0))),
        BadCase("youngs_low", "E in [1e3, 1e12]", "elastic.youngs", _mat(m, e_youngs=_q(100.0))),
        BadCase("youngs_high", "E in [1e3, 1e12]", "elastic.youngs", _mat(m, e_youngs=_q(2e12))),
        BadCase("poisson_half", "nu in (-1, 0.5)", "elastic.poisson", _mat(m, e_poisson=_q(0.5))),
        BadCase("poisson_low", "nu in (-1, 0.5)", "elastic.poisson", _mat(m, e_poisson=_q(-1.2))),
        BadCase("poisson_linear", "nu > 0.45 with linear/corotated", "elastic.model",
                _mat("nitrile_70a", e_model="linear")),
        BadCase("mu_order", "mu_d <= mu_s", "surface.mu_dynamic", _mat(m, s_mu_dynamic=_q(0.9))),
        BadCase("mu_negative", "mu >= 0", "surface.mu_static", _mat(m, s_mu_static=_q(-0.1), s_mu_dynamic=_q(-0.2))),
        BadCase("restitution_high", "e in [0, 1]", "surface.restitution", _mat(m, s_restitution=_q(1.2))),
        BadCase("damping_negative", "damping ratio >= 0", "surface.contact_damping_ratio",
                _mat(m, s_contact_damping_ratio=_q(-0.5))),
        BadCase("stiffness_zero", "contact stiffness > 0", "surface.contact_stiffness", _mat(m, s_contact_stiffness=_q(0.0))),
        BadCase("torsional_negative", "torsional lever >= 0", "surface.torsional_lever", _mat(m, s_torsional_lever=_q(-1e-4))),
        BadCase("friction_angle", "friction angle in (0, pi/2)", "plastic.friction_angle",
                _mat("sand_silica_dry", p_friction_angle=_q(2.0))),
        BadCase("viscosity_negative", "viscosity >= 0", "fluid.viscosity", _mat("water", f_viscosity=_q(-1.0))),
        BadCase("hb_half", "consistency and flow_index together", "fluid.flow_index",
                _mat("sauce_ketchup", f_consistency=_q(29.1))),
        BadCase("hb_flow_index", "flow index in (0, 1.5]", "fluid.flow_index",
                _mat("sauce_ketchup", f_consistency=_q(29.1), f_flow_index=_q(2.0))),
        BadCase("note_missing", "note required unless default", "density",
                _yaml(ct.MaterialCard, MAT_HEAD + "density: {value: 7850, source: catalog}\n")),
        BadCase("unc_inverted", "unc lo <= hi", "density",
                _yaml(ct.MaterialCard, MAT_HEAD + "density: {value: 7850, source: catalog, unc: [8000, 7000], note: x}\n")),
        BadCase("source_unknown", "source is one of five", "density.source",
                _yaml(ct.MaterialCard, MAT_HEAD + "density: {value: 7850, source: guess, note: x}\n")),
        BadCase("unit_kind", "unit of the right kind", "density.value",
                _yaml(ct.MaterialCard, MAT_HEAD + "density: {value: 7.8 m, source: catalog, note: x}\n")),
        BadCase("non_finite", "finite values", "density", _yaml(ct.MaterialCard, MAT_HEAD + "density: .inf\n")),
        BadCase("unknown_key_nested", "unknown keys are errors", "surface.grip",
                _yaml(ct.MaterialCard, MAT_HEAD.replace("restitution: 0.5}", "restitution: 0.5, grip: 1}") + "density: 1000\n")),
        BadCase("unknown_key_top", "unknown keys are errors", "colour",
                _yaml(ct.MaterialCard, MAT_HEAD + "density: 1000\ncolour: red\n")),
        BadCase("duplicate_key", "duplicate keys are errors", "density",
                _yaml(ct.MaterialCard, MAT_HEAD + "density: 1000\ndensity: 2000\n")),
        BadCase("schema_wrong", "schema matches the card type", "schema",
                _yaml(ct.MaterialCard, MAT_HEAD.replace("alloy.material/1", "alloy.object/1") + "density: 1000\n")),
    ]


def _pair_cases() -> list[BadCase]:
    return [
        BadCase("pair_unknown_material", "pair materials resolve", "b",
                _pair("pmma_cast", "stainless_303", b="unobtainium")),
        BadCase("pair_mu_order", "mu_d <= mu_s", "mu_dynamic", _pair("pmma_cast", "stainless_303", mu_dynamic=_q(0.9))),
        BadCase("pair_restitution", "e in [0, 1]", "restitution", _pair("pmma_cast", "stainless_303", restitution=_q(-0.1))),
    ]


def _object_cases() -> list[BadCase]:
    peg, plug = "peg_d12", "usb_a_plug"
    return [
        BadCase("unknown_material", "material resolves", "material", _obj(peg, lambda c, l: dc.replace(c, material="nope"))),
        BadCase("unknown_part_material", "part materials resolve", "part_materials.shell",
                _obj(plug, lambda c, l: dc.replace(c, part_materials={**c.part_materials, "shell": "nope"}))),
        BadCase("part_not_in_mesh", "part names are mesh groups", "part_materials.handle",
                _obj(plug, lambda c, l: dc.replace(c, part_materials={**c.part_materials, "handle": "abs"}))),
        BadCase("primitive_arity", "primitive size arity", "shape.size",
                _obj(peg, lambda c, l: dc.replace(c, shape=ct.Primitive("box", (0.01, 0.01))))),
        BadCase("primitive_negative", "primitive sizes > 0", "shape.size.0",
                _obj(peg, lambda c, l: dc.replace(c, shape=ct.Primitive("cylinder", (-0.006, 0.025))))),
        BadCase("shell_too_thick", "thickness <= 0.2 x in-plane extent", "features.thickness",
                _obj("towel_cotton", lambda c, l: _feat(c, thickness=_q(0.05)))),
        BadCase("thin_member", "member < 2 v dt needs min_wall", "features.min_wall",
                _obj("fish_turner_blade", lambda c, l: _feat(c, min_wall=None))),
        BadCase("parts_over_budget", "collision parts <= budget", "shape.max_parts",
                _obj(plug, lambda c, l: dc.replace(c, shape=dc.replace(c.shape, max_parts=64)))),
        BadCase("sha_mismatch", "mesh sha256 matches the file", "shape.sha256",
                _obj(plug, lambda c, l: dc.replace(c, shape=dc.replace(c.shape, sha256="0" * 64)))),
        BadCase("mesh_missing", "mesh file exists", "shape.path",
                _obj(plug, lambda c, l: dc.replace(c, shape=dc.replace(c.shape, path="meshes/nope.obj")))),
        BadCase("sdf_open_mesh", "SDF sources watertight", "shape.path",
                _obj(peg, lambda c, l: dc.replace(c, shape=dc.replace(_write_open_mesh(l), collision="sdf"),
                                                  precision_partners=(), precision_features=()))),
        BadCase("hydro_open_mesh", "hydroelastic sources watertight", "precision_features.0.mode",
                _obj(peg, lambda c, l: _pf(dc.replace(c, shape=_write_open_mesh(l)), mode="hydroelastic"))),
        BadCase("partner_unknown", "partner object exists", "precision_partners.0",
                _obj(peg, lambda c, l: dc.replace(c, precision_partners=("ghost:bore",)))),
        BadCase("partner_feature_missing", "partner feature with clearance", "precision_partners.0",
                _obj(peg, lambda c, l: dc.replace(c, precision_partners=("block_bore_d12_2:nope",)))),
        BadCase("own_feature_missing", "own feature with clearance", "precision_features",
                _obj(peg, lambda c, l: _pf(c, clearance=None))),
        BadCase("crop_inverted", "crop lo < hi", "precision_features.0.crop",
                _obj(peg, lambda c, l: _pf(c, crop=((0.01, 0.0, 0.0), (0.0, 0.01, 0.01))))),
        BadCase("feature_duplicate", "feature names unique", "precision_features.1.name",
                _obj(peg, lambda c, l: dc.replace(c, precision_features=c.precision_features * 2))),
        BadCase("analytic_format", "analytic id '<family>:<args>'", "precision_features.0.analytic_id",
                _obj("bolt_m10x60", lambda c, l: _pf(c, analytic_id="M10"))),
        BadCase("iso_thread_format", "iso_thread:M<d>x<p>:<class>", "precision_features.0.analytic_id",
                _obj("bolt_m10x60", lambda c, l: _pf(c, analytic_id="iso_thread:M10:6g"))),
    ]


def _object_cases_2() -> list[BadCase]:
    peg = "peg_d12"
    return [
        BadCase("blade_no_topology", "blade needs a topology budget", "topology",
                _obj("knife_blade_steel", lambda c, l: dc.replace(c, topology=None))),
        BadCase("tear_no_topology", "tear energy needs a topology budget", "topology",
                _obj("towel_cotton", lambda c, l: dc.replace(c, material=_tearable(l)))),
        BadCase("route_unknown", "override names a route", "route.route",
                _obj(peg, lambda c, l: dc.replace(c, route=ct.RouteOverride("teleport", "seeded")))),
        BadCase("route_no_reason", "override has a reason", "route.reason",
                _obj(peg, lambda c, l: dc.replace(c, route=ct.RouteOverride("rigid", " ")))),
        BadCase("chain_links_two", "CHAIN link count >= 3", "features.chain_links",
                _obj("usb_cable_1m", lambda c, l: _feat(c, chain_links=2))),
        BadCase("curve_one_point", "curve >= 2 points", "shape.points",
                _obj("usb_cable_1m", lambda c, l: dc.replace(c, shape=ct.CurveSource(((0.0, 0.0, 0.0),), 1e-3)))),
        BadCase("region_inverted", "particle region lo < hi", "shape.region",
                _obj("water_cup", lambda c, l: dc.replace(c, shape=ct.ParticleSource(((0.1, 0.0, 0.0), (0.0, 0.1, 0.1)), 4e-3)))),
        BadCase("sdf_both", "SDF path xor analytic id", "shape",
                _obj(peg, lambda c, l: dc.replace(c, shape=ct.SDFSource("meshes/x.obj", "iso_thread:M10x1.5:6g"),
                                                  precision_partners=(), precision_features=()))),
        BadCase("tolerance_zero", "tolerance > 0", "loads.tolerance",
                _obj(peg, lambda c, l: dc.replace(c, loads=dc.replace(c.loads, tolerance=_q(0.0))))),
        BadCase("negative_topology", "topology spares >= 0", "topology.spare_vertices",
                _obj("knife_blade_steel", lambda c, l: dc.replace(c, topology=TopologyBudget(spare_vertices=-1)))),
        BadCase("name_chars", "object names [A-Za-z0-9_-]", "name", _obj(peg, lambda c, l: dc.replace(c, name="peg:12"))),
        BadCase("mesh_inside_out", "meshes wound outward", "shape.path",
                _obj(peg, lambda c, l: dc.replace(c, shape=_write_inverted_mesh(l), precision_partners=(),
                                                  precision_features=()))),
        BadCase("non_finite_float", "plain floats finite", "precision_features.0.crop.0.0",
                _obj(peg, lambda c, l: _pf(c, crop=((float("nan"), -0.007, -0.0255), (0.007, 0.007, -0.01))))),
        BadCase("mjcf_unknown_body", "MJCF body exists", "shape.body",
                _obj("hinged_box_plywood", lambda c, l: dc.replace(c, shape=ct.MJCFSource("mjcf/hinged_box.xml", "ghost")))),
        BadCase("motion_literal", "Literal values", "motion", _yaml(ct.ObjectCard, OBJ_HEAD + "motion: floating\n")),
        BadCase("shape_variant", "known shape variant", "shape.cube",
                _yaml(ct.ObjectCard, OBJ_HEAD.replace("{primitive:", "{cube:"))),
        BadCase("int_type", "integer fields", "budget.particles", _yaml(ct.ObjectCard, OBJ_HEAD + "budget: {particles: lots}\n")),
        BadCase("unknown_key_feature", "unknown keys are errors", "precision_features.0.depth",
                _yaml(ct.ObjectCard, OBJ_HEAD + "precision_features: [{name: f, crop: [[0,0,0],[1,1,1]], depth: 3}]\n")),
        BadCase("missing_required", "required fields", "material",
                _yaml(ct.ObjectCard, OBJ_HEAD.replace("material: steel_carbon\n", ""))),
    ]


def _scene_cases() -> list[BadCase]:
    def pl(s: ct.SceneCard, i: int, **kw: object) -> ct.SceneCard:
        objs = list(s.objects)
        objs[i] = dc.replace(objs[i], **kw)
        return dc.replace(s, objects=tuple(objs))
    rnd = ct.Randomization
    return [
        BadCase("scene_unknown_object", "placed objects resolve", "objects.0.object", _scene(lambda s: pl(s, 0, object="ghost"))),
        BadCase("scene_quaternion", "unit quaternion", "objects.0.pose", _scene(lambda s: pl(s, 0, pose=(0, 0, 0, 0, 0, 0, 2.0)))),
        BadCase("scene_path", "randomized path is a leaf", "objects.2.randomize.0.field",
                _scene(lambda s: pl(s, 2, randomize=(rnd("material.colour", 0.0, 1.0),)))),
        BadCase("scene_range_order", "lo <= hi", "objects.2.randomize.0.lo",
                _scene(lambda s: pl(s, 2, randomize=(rnd("material.density", 8000.0, 7000.0),)))),
        BadCase("scene_range_nu", "both ends valid", "objects.2.randomize.0.hi",
                _scene(lambda s: pl(s, 2, randomize=(rnd("material.elastic.poisson", 0.2, 0.6),)))),
        BadCase("scene_range_density", "both ends valid", "objects.2.randomize.0.lo",
                _scene(lambda s: pl(s, 2, randomize=(rnd("material.density", 5.0, 8000.0),)))),
        BadCase("scene_attachment", "attachment names resolve", "attachments.0.b",
                _scene(lambda s: dc.replace(s, attachments=(ct.Attachment("peg_d12", "ghost"),)))),
        BadCase("scene_attach_timeconst_spring", "solref overrides only on weld/joint", "attachments.0.timeconst",
                _scene(lambda s: dc.replace(s, attachments=(ct.Attachment(
                    "peg_d12", "block_bore_d12_2", "spring", timeconst=Q(5e-3, "model", None, "seeded")),)))),
        BadCase("scene_attach_dampratio", "dampratio > 0", "attachments.0.dampratio",
                _scene(lambda s: dc.replace(s, attachments=(ct.Attachment(
                    "peg_d12", "block_bore_d12_2", dampratio=Q(0.0, "model", None, "seeded")),)))),
        BadCase("scene_emitter_material", "emitter material is a fluid", "emitters.0.material",
                _scene(lambda s: dc.replace(s, emitters=(ct.EmitterSpec("emitter", "steel_carbon", (0, 0, 0, 0, 0, 0, 1.0),
                                                                        (0.01, 0.01, 0.01), 1e-5),)))),
        BadCase("scene_emitter_time", "stop after start", "emitters.0.stop_s",
                _scene(lambda s: dc.replace(s, emitters=(ct.EmitterSpec("emitter", "water", (0, 0, 0, 0, 0, 0, 1.0),
                                                                        (0.01, 0.01, 0.01), 1e-5, start_s=1.0, stop_s=0.5),)))),
        BadCase("scene_robot_names", "robot names unique", "robots.1.name",
                _scene(lambda s: dc.replace(s, robots=(ct.RobotRef("meshes/usb_a_plug.obj", "mjcf", (0, 0, 0, 0, 0, 0, 1.0), "arm"),) * 2))),
        BadCase("scene_robot_file", "robot file exists", "robots.0.path",
                _scene(lambda s: dc.replace(s, robots=(ct.RobotRef("robots/ghost.xml", "mjcf", (0, 0, 0, 0, 0, 0, 1.0), "arm"),)))),
        BadCase("scene_worlds", "worlds >= 1", "worlds", _scene(lambda s: dc.replace(s, worlds=0))),
        BadCase("scene_pair", "inline pair materials resolve", "pairs.0.a",
                _scene(lambda s: dc.replace(s, pairs=(ct.PairCard("aaa_ghost", "steel_carbon", _q(0.5), _q(0.4)),)))),
        BadCase("scene_unit_kind", "randomization unit of the target kind", "objects.0.randomize.0.lo",
                _yaml(ct.SceneCard, SCENE_HEAD + "objects: [{object: peg_d12, pose: [0,0,0,0,0,0,1], "
                                    "randomize: [{field: pose.0, lo: 2 kg, hi: 0.01}]}]\n")),
    ]


def seeded_bad_cases() -> list[BadCase]:
    """The CARD-5 corpus (>= 50 cases, one rule each)."""
    return _material_cases() + _pair_cases() + _object_cases() + _object_cases_2() + _scene_cases()


def run_case(case: BadCase, lib: CardLibrary) -> tuple[bool, list[str]]:
    """(rejected with the expected field named, fields named)."""
    named = case.run(lib)
    return case.field in named, named


def card5_rate(lib: CardLibrary) -> tuple[int, int, list[tuple[str, str, list[str]]]]:
    """(passed, total, failures as (id, expected field, named fields))."""
    fails = []
    cases = seeded_bad_cases()
    for case in cases:
        ok, named = run_case(case, lib)
        if not ok:
            fails.append((case.id, case.field, named))
    return len(cases) - len(fails), len(cases), fails


__all__ = ["BadCase", "card5_rate", "run_case", "seeded_bad_cases"]


def card10_rows(lib: CardLibrary) -> list[tuple[str, float, float, float]]:
    """(object, card mass kg, catalogue kg, relative error) for every entry of ``library/catalog.yaml``."""
    from alloy.cards.codec import Ctx, decode  # noqa: PLC0415
    from alloy.cards.mass import card_mass  # noqa: PLC0415
    from alloy.cards.paths import CardView  # noqa: PLC0415
    from alloy.cards.units_yaml import yaml_load  # noqa: PLC0415

    doc = yaml_load((lib.roots[0] / "catalog.yaml").read_text(encoding="utf-8"), "catalog")
    rows = []
    for name, raw in doc["masses"].items():
        cat = decode(Q, raw, f"masses.{name}", Ctx("catalog"), "mass").value
        r = lib.resolve(lib.object(name))
        m = card_mass(CardView(r.card, r.material, r.part_materials), lib.asset_path)
        rows.append((name, m, cat, (m - cat) / cat))
    return rows


def main() -> int:
    """Print the CARD-5 and CARD-10 numbers for docs/notes/wp1-results.md (CARD-5 runs on a temporary copy)."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(default_root(), Path(tmp) / "library")
        ok, n, fails = card5_rate(CardLibrary([Path(tmp) / "library"]))
    print(f"CARD-5 {ok}/{n} = {100.0 * ok / n:.1f} % rejected with the field named; failures: {fails}")
    rows = card10_rows(CardLibrary(assets=None))
    for name, m, cat, rel in rows:
        print(f"CARD-10 {name}: card {m * 1e3:.3f} g, catalogue {cat * 1e3:.3f} g, {rel:+.2%} "
              f"{'pass' if abs(rel) <= 0.02 else 'fail'}")
    print(f"CARD-10 {sum(abs(r[3]) <= 0.02 for r in rows)}/{len(rows)} within 2 %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
