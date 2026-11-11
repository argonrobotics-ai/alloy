# SPDX-License-Identifier: Apache-2.0
"""Objaverse importer, card stage: cards written with the shared writers load and validate through CardLibrary, the
engine's emitted body mass equals the mass the importer chose, and thin walls are declared."""

from __future__ import annotations

import dataclasses
import doctest
import hashlib
import json

import numpy as np
import pytest

from alloy.cards import CardLibrary
from alloy.cards.geometry import MeshData, write_obj
from alloy.cards.mass import mesh_mass
from alloy.importers.common.cards import MIN_WALL_NOTE, min_wall_note
from alloy.importers.objaverse import assemble as A
from alloy.importers.objaverse import cards as C
from alloy.importers.objaverse.categories import card_name, load_categories
from alloy.importers.objaverse.errors import ImporterConfigError, ImportRejected
from alloy.importers.objaverse.recentre import recentre_dir
from alloy.route.types import Route, RouteDecision
from alloy.scene.emitters.rigid import emit_rigid

UID = "0f3a9c51d2e84b7a9e1c2d3f4a5b6c7d"
CAND = {"uid": UID, "category": "mug", "score": 1.0, "spdx": "CC-BY-4.0", "name": "Mug | big", "author": "Ann",
        "url": "https://sketchfab.com/3d-models/x"}


def _box(hx: float, hy: float, hz: float) -> MeshData:
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (0.0, 2 * hz)])
    f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                  (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return MeshData(v, f)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh_dir(tmp_path, mesh: MeshData):
    """A mesh-pipeline output folder for ``mesh`` (origin at the base, as uploads often are), re-centred."""
    d = tmp_path / "out" / card_name("mug", UID)
    d.mkdir(parents=True)
    write_obj(d / C.PHYSICS, mesh)
    body = "\n".join(ln for ln in (d / C.PHYSICS).read_text().splitlines() if not ln.startswith("#"))
    (d / C.HULLS).write_text(f"# source_sha256: {_sha(d / C.PHYSICS)}\no hull_0\n{body}\n")
    vol = float(np.prod(mesh.vertices.max(0) - mesh.vertices.min(0)))
    result = {"name": d.name, "ok": True, "watertight": True, "volume": vol, "n_hulls": 1,
              "files": {C.PHYSICS: _sha(d / C.PHYSICS), C.HULLS: _sha(d / C.HULLS), "visual.glb": "g",
                        "thumb.png": "t"},
              "sizes": {C.PHYSICS: 10}, "repair": {"method": "as_is"}, "mass_props": {"extents": [0.1, 0.08, 0.08]}}
    (d / "result.json").write_text(json.dumps(result))
    return d, recentre_dir(d)


def _write(tmp_path, mesh: MeshData, wall: A.Wall):
    d, result = _mesh_dir(tmp_path, mesh)
    root = tmp_path / "imports"
    out = A.write_root(root, [{"cand": CAND, "category": "mug", "result": result}], {d.name: d}, {d.name: wall})
    return root, d.name, out


def test_docstring_examples_run():
    assert doctest.testmod(C).failed == 0 and doctest.testmod(A).failed == 0


@pytest.mark.parametrize("hz,effective", [(0.008, False), (0.05, True)])
def test_written_card_validates_and_emits_the_chosen_mass(tmp_path, hz, effective):
    root, name, out = _write(tmp_path, _box(0.04, 0.04, hz), wall=0.01)
    assert out["written"] == 1 and out["rejected"] == [] and out["queue"] == 1
    lib = CardLibrary([CardLibrary().roots[0], root])
    card = lib.object(name)                                          # raises on validation errors
    mat = lib.material(card.material)
    assert card.shape.collision == "decompose" and card.material == f"{name}_eff"
    assert mat.density.note.startswith("ESTIMATE") and mat.surface.mu_static.note.startswith("library material")
    mug = load_categories()["mug"]
    expect = mug.mass.value if effective else 2300.0 * 0.08 * 0.08 * 2 * hz
    assert mesh_mass(lib.mesh(card.shape.path), mat.density.value) == pytest.approx(expect, rel=1e-5)
    obj = lib.resolve(card)
    obj = dataclasses.replace(obj, card=dataclasses.replace(
        card, shape=dataclasses.replace(card.shape, path=str(lib.asset_path(card.shape.path)))))
    from alloy._vendor import newton

    b = newton.ModelBuilder()
    emit_rigid(b, obj, RouteDecision(Route.RIGID, "mujoco", {}, 1.0, False, ()))
    assert b.finalize(device="cpu").body_mass.numpy()[0] == pytest.approx(expect, rel=1e-5)
    man = json.loads((root / "manifest.json").read_text())
    assert man["objects"][name]["files"][C.PHYSICS]["sha256"] == card.shape.sha256
    att = (root / "ATTRIBUTION.md").read_text()
    assert "Ann" in att and "Mug / big" in att and (root / ".gitignore").read_text() == "meshes/\n"


def test_thin_wall_is_declared_on_the_card(tmp_path):
    root, name, _ = _write(tmp_path, _box(0.04, 0.04, 0.008), wall=0.0004)
    card = CardLibrary([CardLibrary().roots[0], root]).peek_object(name)
    assert card.features.min_wall is not None and card.features.min_wall.value == pytest.approx(0.0004)
    assert card.features.min_wall.note == MIN_WALL_NOTE
    assert "min_wall_probe" not in json.loads((root / "manifest.json").read_text())["objects"][name]["physics"]


def test_re_measured_wall_keeps_the_probe_reading(tmp_path):
    """A probe reading on a remesh defect, re-measured by common.walls: the card says how, the manifest keeps both."""
    root, name, _ = _write(tmp_path, _box(0.04, 0.04, 0.008), wall=(1.17e-3, 2.3e-5))
    card = CardLibrary([CardLibrary().roots[0], root]).peek_object(name)
    assert card.features.min_wall.value == pytest.approx(1.17e-3, rel=0, abs=0)
    assert card.features.min_wall.note == min_wall_note(1.17e-3, 2.3e-5) != MIN_WALL_NOTE
    phys = json.loads((root / "manifest.json").read_text())["objects"][name]["physics"]
    assert (phys["min_wall"], phys["min_wall_probe"]) == (1.17e-3, 2.3e-5)


def test_pick_ranks_within_category_and_round_robins():
    cands = [dict(CAND, uid=f"{c}{i}" + "0" * 30, category=c, score=s)
             for c, i, s in [("mug", 1, 1.0), ("mug", 2, 3.0), ("apple", 1, 0.5)]]
    kept = [{"uid": c["uid"], "category": c["category"]} for c in cands]
    results = {card_name(c["category"], c["uid"]): {"ok": True, "watertight": True, "repair": {}} for c in cands}
    results[card_name("mug", cands[0]["uid"])]["watertight"] = False
    picked = A.pick(cands, kept, results, target=5)
    assert [p["cand"]["uid"][:4] for p in picked] == ["appl", "mug2"]


def test_incomplete_pipeline_output_is_rejected():
    cat = load_categories()["mug"]
    base = CardLibrary().material(cat.material)
    meta = A._meta(CAND)
    with pytest.raises(ImportRejected, match="physics.hulls.obj"):
        C.card_spec(meta, cat, C.Processed({C.PHYSICS: "x"}, 1e-4, 1, (0.1, 0.1, 0.1)), base, None)
    with pytest.raises(ImportRejected, match="40 hulls"):
        C.card_spec(meta, cat, C.Processed({C.PHYSICS: "x", C.HULLS: "y"}, 1e-4, 40, (0.1, 0.1, 0.1)), base, None)


def test_checks_pass_on_a_good_object_and_prune_removes_a_bad_one(tmp_path):
    from alloy.importers.objaverse import check

    root, name, _ = _write(tmp_path, _box(0.04, 0.04, 0.008), wall=0.01)
    checks = check.run_checks(root, workers=1)
    assert check.failures(checks) == [], checks[name]
    assert check.prune(root, [name]) == 1
    assert not (root / "objects" / "objaverse" / f"{name}.yaml").exists()
    assert json.loads((root / "manifest.json").read_text())["objects"] == {}
    assert (root / "queue.jsonl").read_text().strip() == "" and name not in (root / "ATTRIBUTION.md").read_text()


def test_pick_skips_excluded_names_and_backfills():
    cands = [dict(CAND, uid=f"mug{i}" + "0" * 30, category="mug", score=float(i)) for i in range(3)]
    kept = [{"uid": c["uid"], "category": "mug"} for c in cands]
    results = {card_name("mug", c["uid"]): {"ok": True, "watertight": True, "repair": {}} for c in cands}
    best = card_name("mug", cands[2]["uid"])
    picked = A.pick(cands, kept, results, target=2, exclude=frozenset({best}))
    assert [p["cand"]["uid"][:4] for p in picked] == ["mug1", "mug0"]


def test_rerun_keeps_a_rerendered_thumbnail_entry(tmp_path):
    root, name, _ = _write(tmp_path, _box(0.04, 0.04, 0.008), wall=0.01)
    man = json.loads((root / "manifest.json").read_text())
    man["objects"][name]["files"]["thumb.png"] = {"sha256": "new", "bytes": 7}
    (root / "manifest.json").write_text(json.dumps(man))
    d = tmp_path / "out" / name
    result = _mesh_dir(tmp_path / "again", _box(0.04, 0.04, 0.008))[1]
    A.write_root(root, [{"cand": CAND, "category": "mug", "result": result}], {name: d}, {name: 0.01})
    files = json.loads((root / "manifest.json").read_text())["objects"][name]["files"]
    assert files["thumb.png"] == {"sha256": "new", "bytes": 7} and files["visual.glb"]["sha256"] == "g"


def test_assemble_refuses_meshes_that_are_not_centred(tmp_path):
    d, result = _mesh_dir(tmp_path, _box(0.04, 0.04, 0.008))
    result["mass_props"]["bounds"] = [[0.0, 0.0, 0.0], [0.08, 0.08, 0.016]]
    with pytest.raises(ImporterConfigError, match="recentre"):
        A.write_root(tmp_path / "imports", [{"cand": CAND, "category": "mug", "result": result}], {d.name: d},
                     {d.name: 0.01})
