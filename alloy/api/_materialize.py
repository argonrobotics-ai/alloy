# SPDX-License-Identifier: Apache-2.0
"""Turn an ``alloy.Scene`` into engine cards on disk (materials/, pairs/, objects/, scenes/, meshes/, robots/).

Per-object overrides (mass, density, friction, restitution) become a derived material ``<base>_for_<object>`` so
they touch that object only; unless the object sets its own friction, the derived card names its base
(``pairs_from``), so the library's sourced pair cards of the base apply to it (``alloy.scene.pairs.pair_card``; one
card per object, not one per pair of objects), and it takes its base's look file (``derived_look``), so
the object renders and varies as its base does. Per-world start positions are returned for ``build``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api import _args as A
from alloy.api import _anchor, _hulls, catalog
from alloy.api import _card_objects as _CO
from alloy.api._items import mixed_winding_edges
from alloy.api.errors import SceneError
from alloy.api.fixtures import PRECISE_K, PRECISE_MIN_E, precise_substeps

SCENE_NAME = "scene"
THIN_FIELDS = {"box": "size", "sphere": "radius", "cylinder": "radius", "capsule": "radius"}   # add_* primitives


@dataclass
class Materialized:
    """What ``materialize`` wrote: the card root, the scene file and what the cards cannot hold."""

    root: Path
    scene_file: Path
    per_world_pos: dict[str, list[list[float]]] = field(default_factory=dict)
    cards: list[str] = field(default_factory=list)                 # object card names placed (fetch checks)
    pours: dict[str, dict[str, Any]] = field(default_factory=dict)  # pour name -> emitter parameters
    materials: dict[str, str] = field(default_factory=dict)      # object name -> material name used
    substeps: int = 1                                             # engine periods per Scene.dt (precise fixtures)
    scene: Any = None                                             # the scene written: anchors turned into origins
    names_of: dict[str, list[str]] = field(default_factory=dict)  # card name -> its objects' names (partners)


def materialize(scene: Any, root: Path) -> Materialized:
    """Write ``scene``'s cards under ``root`` (created). Example: ``materialize(scene, Path("cards")).scene_file``."""
    from alloy.cards.units_yaml import yaml_dump  # noqa: PLC0415 - the card layer's canonical YAML

    scene = _anchor.resolve(scene)                  # library objects placed by their anchor -> body origins
    for sub in ("materials", "pairs", "objects", "scenes", "meshes", "robots"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    out = Materialized(root, root / "scenes" / f"{SCENE_NAME}.yaml", substeps=precise_substeps(scene), scene=scene)
    roots, out.names_of = list(scene.roots), names_by_card(scene.items)
    for name, spec in scene.materials.items():
        _write(root / "materials" / f"{name}.yaml", _custom_material(name, spec, roots), yaml_dump)
    forced = {n for p in scene.pairs for n in (p["a"], p["b"])}
    placements, robots = [], []
    for name, item in scene.items.items():
        if item["kind"] == "robot":
            robots.append(_robot_ref(name, item, root))
            continue
        placements.append(_placement(name, item, root, roots, (forced, out.substeps > 1), out, yaml_dump))
    card = {"schema": "alloy.scene/1", "robots": robots, "objects": placements, **scene.extra,
            "worlds": scene.worlds, "control_dt": scene.dt / out.substeps}
    card["pairs"] = list(scene.extra.get("pairs", [])) + [_scene_pair(p, out.materials, scene) for p in scene.pairs]
    _write(out.scene_file, card, yaml_dump)
    return out


def _write(path: Path, data: dict[str, Any], dump: Any) -> None:
    path.write_text(dump(data, header="written by alloy.api (alloy.Scene)"), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    from alloy.cards.units_yaml import yaml_load  # noqa: PLC0415

    return yaml_load(path.read_text(encoding="utf-8"), path.stem)


def _base_material(name: str, roots: list[str]) -> dict[str, Any]:
    path = catalog.card_path("materials", name, roots)
    if path is None:
        raise SceneError(f"unknown material {name!r}", f"use one of {catalog.near('materials', name, roots)}",
                         field="material")
    return _load_yaml(path)


def _custom_material(name: str, spec: dict[str, Any], roots: list[str]) -> dict[str, Any]:
    note = "set through alloy.Scene.add_material"
    data = _base_material(spec["like"], roots) if spec.get("like") else {"schema": "alloy.material/1"}
    data["name"] = name
    data["density"] = A.user_q(spec["density"], note)
    mu = A.user_q(spec["friction"], note)
    data["surface"] = {**(data.get("surface") or {}), "mu_static": mu, "mu_dynamic": dict(mu),
                       "restitution": A.user_q(spec["restitution"], note)}
    return data


def _placement(name: str, item: dict[str, Any], root: Path, roots: list[str], forced: tuple[set[str], bool],
               out: Materialized, dump: Any) -> dict[str, Any]:
    """The scene card placement of one item (cards written as needed); ``forced`` = (names whose material needs
    its own card for a pair, precise-fit scene)."""
    if item.get("raw_placement") is not None:
        return _raw_placement(name, item, root, roots, forced, out, dump)
    if item["per_world_pos"] is not None:
        out.per_world_pos[name] = item["per_world_pos"]
    if item.get("pour"):
        out.pours[name] = dict(item["pour"])
    if item["kind"] == "card":
        card_name = _card_object(name, item, root, roots, dump, out, forced)
        out.cards.append(item["card_name"])
    else:
        if item["kind"] == "mesh":
            _refuse_mixed_winding(name, item, root)
        material = _object_material(name, item, root, roots, forced, dump)
        out.materials[name] = material
        _write(root / "objects" / f"{name}.yaml", _object_card(name, item, material, root), dump)
        card_name = name
    placement: dict[str, Any] = {"object": card_name, "pose": [float(x) for x in item["pose"]]}
    if item["randomize"]:
        placement["randomize"] = [dict(r) for r in item["randomize"]]
    return placement


def _object_material(name: str, item: dict[str, Any], root: Path, roots: list[str], forced: tuple[set[str], bool],
                     dump: Any) -> str:
    base, ov = item["material"], item["overrides"]
    cap = _friction_cap(item)
    own = root / "materials" / f"{base}.yaml"          # a Scene.add_material card written above
    data = copy.deepcopy(_load_yaml(own) if own.exists() else _base_material(base, roots))
    stiff = forced[1] and _stiffens(data, item)
    if not ov and name not in forced[0] and cap is None and not stiff:
        return base
    mat_name = f"{base}_for_{A.slug(name)}"
    data["name"] = mat_name
    note = f"set for object {name!r} through alloy.Scene"
    density = _density(name, item, ov, root)
    if density is not None:
        data["density"] = A.user_q(density, note)
    surface = dict(data.get("surface") or {})
    if "friction" in ov:
        surface["mu_static"] = A.user_q(ov["friction"], note)
        surface["mu_dynamic"] = A.user_q(ov["friction"], note)
    if "restitution" in ov:
        surface["restitution"] = A.user_q(ov["restitution"], note)
    if cap is not None and _q_value(surface.get("mu_dynamic")) > cap:
        surface["mu_dynamic"] = A.user_q(cap, f"capped at the lowest randomized static friction of {name!r}")
    if stiff:
        surface["contact_stiffness"] = A.user_q(PRECISE_K, "precise-fit scene (add_block_with_hole): a machined part's "
                                                "contact stiffness at its 0.25 ms rigid step (alloy.api.fixtures)")
    data["surface"] = surface
    if "friction" not in ov:
        data["pairs_from"] = base           # the base's sourced pair cards apply (a friction= of its own replaces them)
    _write(root / "materials" / f"{mat_name}.yaml", data, dump)
    derived_look(base, mat_name, root, roots)
    return mat_name


def derived_look(base: str, name: str, root: Path, roots: list[str]) -> Path | None:
    """Write ``looks/materials/<name>.json`` under ``root``: the look of material ``base`` for the per-object
    material ``name`` made from it, so the object renders, varies (``alloy.vary``) and saves with its base's look,
    not the grey fallback. Returns the file, or None when ``base`` has no look file (nothing written).

    Example: ``derived_look("wood_beech", "wood_beech_for_lid", root, [])`` -> ``root/looks/materials/
    wood_beech_for_lid.json``."""
    from alloy.api._save import _look_file  # noqa: PLC0415

    src = next((p for r in (root, *catalog.search_roots(roots)) if (p := _look_file(Path(r), "materials", base))),
               None)
    if src is None:
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    data.update(material=name, note=f"the look of {base}, which this per-object material is made from")
    out = root / "looks" / "materials" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return out


def _stiffens(data: dict[str, Any], item: dict[str, Any]) -> bool:
    """A rigid object whose material is stiff (Young's modulus >= PRECISE_MIN_E) and has no contact stiffness."""
    if item["kind"] in ("liquid", "cloth") or (data.get("surface") or {}).get("contact_stiffness") is not None:
        return False
    youngs = (data.get("elastic") or {}).get("youngs")
    value = _q_value(youngs) if youngs is not None else float("nan")
    return math.isfinite(value) and value >= PRECISE_MIN_E


def _friction_cap(item: dict[str, Any]) -> float | None:
    """The lowest static friction a randomization can draw (dynamic friction must not exceed it), or None."""
    lows = [r["lo"] for r in item.get("randomize", []) if r["field"] == "material.surface.mu_static"]
    return min(lows) if lows else None


def _q_value(node: Any) -> float:
    if isinstance(node, dict):
        node = node.get("value")
    try:
        return float(node)
    except (TypeError, ValueError):
        return float("inf")


def _density(name: str, item: dict[str, Any], ov: dict[str, float], root: Path) -> float | None:
    if "density" in ov:
        return ov["density"]
    if "mass" not in ov:
        return None
    volume = item["volume"] if item["volume"] is not None else _mesh_volume(item, root)
    if not volume or volume <= 0.0:
        raise SceneError(f"{name}: cannot derive a density from mass (volume {volume})",
                         "pass density= instead of mass=, or close the mesh (watertight, outward normals)",
                         field="mass")
    return ov["mass"] / volume


def _mesh_volume(item: dict[str, Any], root: Path) -> float:
    v, f = _mesh_arrays(item)
    scale = np.asarray(item["shape"]["mesh"]["scale"], float)
    v = v * scale
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _refuse_mixed_winding(name: str, item: dict[str, Any], root: Path) -> None:
    """A SceneError for a dynamic ``add_mesh`` object whose mass would come out 0 or less because its closed mesh has
    faces wound both ways (the body's mass is its density x the mesh's signed volume: half a cube's faces flipped
    give 0). The build's rigid emitter refuses any such body; this names the winding. Static and kinematic objects
    and an explicit mass= build as before (mass= over a volume of 0 raises in ``_density``). A file that cannot be
    read is left to the card check, which names it."""
    if item["static"] or item["card_extra"].get("motion") == "kinematic" or "mass" in item["overrides"]:
        return
    try:
        if _mesh_volume(item, root) > 0.0:
            return
        edges = mixed_winding_edges(*_mesh_arrays(item))
    except (ValueError, OSError):
        return
    if edges:
        where = "path" if "file" in item["mesh"] else "faces"
        raise SceneError(f"{name}: the mesh is closed but {edges} of its edges run the same way in both of their "
                         "faces, so some faces are wound inside out: it encloses no volume and the body would have "
                         "no mass", "wind every face counter-clockwise seen from outside (trimesh.repair."
                         "fix_normals(mesh) repairs a mesh), or pass static=True for a part that never moves",
                         field=where)


def _mesh_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mesh = item["mesh"]
    if "vertices" in mesh:
        return np.asarray(mesh["vertices"], float), np.asarray(mesh["faces"], int)
    from alloy.cards.geometry import read_mesh  # noqa: PLC0415

    data = read_mesh(Path(mesh["file"]))
    return np.asarray(data.vertices, float), np.asarray(data.faces, int).reshape(-1, 3)


def _object_card(name: str, item: dict[str, Any], material: str, root: Path) -> dict[str, Any]:
    shape = copy.deepcopy(item["shape"])
    if "mesh" in shape:
        shape["mesh"]["path"], shape["mesh"]["sha256"] = _write_mesh(name, item, root)
        if shape["mesh"]["collision"] == "decompose":
            source = Path(item["mesh"]["file"]) if "file" in item["mesh"] else None
            _hulls.write_sidecar(name, root / shape["mesh"]["path"], _mesh_arrays(item), source)
    card: dict[str, Any] = {"schema": "alloy.object/1", "name": name, "shape": shape, "material": material}
    if item["static"]:
        card["motion"] = "static"
    card.update(copy.deepcopy(item["card_extra"]))
    _refuse_thin(name, item["kind"], card)
    return card


def _thin_limit() -> tuple[float, float, float]:
    """(limit, contact speed, rigid step) of the card check's thin-member rule: limit = 2 x speed x step, 1.25 mm for
    ``alloy.Scene`` cards (they set no contact speed; the check runs at the library's rigid step, whatever the dt)."""
    from alloy.cards import types as ct  # noqa: PLC0415
    from alloy.cards.library import RIGID_DT  # noqa: PLC0415

    speed = ct.LoadEnvelope().contact_speed.value
    return 2.0 * speed * RIGID_DT, speed, RIGID_DT


def _refuse_thin(name: str, kind: str, card: dict[str, Any]) -> None:
    """A SceneError with the numbers for a primitive thinner than the thin-member rule allows (audit GAPS-7): the
    card check's own error names ``features.min_wall``, which ``add_box`` has no argument for, and a smaller dt does
    not lower the limit. Measured (CPU): a 0.5 mm plate dropped 5 mm onto a 0.5 mm shelf fell through it, so the
    rule stays. A card declaring ``features.min_wall`` itself (``add_cloth``) passes as before."""
    from alloy.cards import geometry as geo  # noqa: PLC0415
    from alloy.cards import types as ct  # noqa: PLC0415

    prim = card["shape"].get("primitive")
    if kind not in THIN_FIELDS or prim is None or "min_wall" in (card.get("features") or {}):
        return
    thin = geo.primitive_min_member(ct.Primitive(prim["kind"], tuple(float(v) for v in prim["size"])))
    limit, speed, step = _thin_limit()
    if thin >= limit:
        return
    raise SceneError(f"{name!r} is {_mm(thin)} at its thinnest; a solid shape needs at least {_mm(limit)}, what a "
                     f"part at the {speed:g} m/s design contact speed covers in two {step * 1e3:.3g} ms rigid steps: "
                     "a thinner one can pass through another thin part in a step",
                     f"make it at least {_mm(limit)} thick and keep a moving part's real weight with mass=; a sheet "
                     "meant to bend is add_cloth(..., thickness=...); a smaller Scene(dt=...) does not lower the "
                     "limit", field=_thin_field(kind, prim))


def _mm(x: float) -> str:
    return f"{x * 1e3:.3g} mm"


def _thin_field(kind: str, prim: dict[str, Any]) -> str:
    """The ``add_*`` argument that sets the thinnest member: a cylinder's height when it is thinner than across."""
    if kind == "cylinder" and float(prim["size"][1]) < float(prim["size"][0]):
        return "height"
    return THIN_FIELDS[kind]


def _write_mesh(name: str, item: dict[str, Any], root: Path) -> tuple[str, str]:
    """The card's mesh file: an OBJ copied as is; an STL (or arrays) written as OBJ, the format the engine's mesh
    emitter reads (it refused .stl, agent friction trial F6); STL vertices are merged exactly (``read_stl``)."""
    mesh, rel = item["mesh"], f"meshes/{name}.obj"
    if "file" in mesh and Path(mesh["file"]).suffix.lower() == ".obj":
        shutil.copyfile(mesh["file"], root / rel)
    else:
        v, f = _mesh_arrays(item)
        lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in v.tolist()]
        lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in f.tolist()]
        (root / rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel, hashlib.sha256((root / rel).read_bytes()).hexdigest()


def names_by_card(items: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """{card name: the names of the scene's objects placed from it}, for the cards some object renames."""
    out: dict[str, list[str]] = {}
    for name, item in items.items():
        if item.get("kind") == "card" and item.get("raw_placement") is None:
            out.setdefault(str(item["card_name"]), []).append(name)
    return out if any(names != [card] for card, names in out.items()) else {}


def _partners(entries: list[str], names_of: dict[str, list[str]]) -> list[str]:
    """A card's precision partners ("card" or "card:feature") by the names this scene gives that card's objects:
    the engine pairs objects by name, and an object placed under another name has a card of that name (PHYSA-4:
    ``add_object("peg", "peg_d12")`` beside ``add_object("blk", "block_bore_d12_2")`` made no pair, with no word)."""
    out = []
    for entry in entries:
        card, sep, feature = str(entry).partition(":")
        out += [f"{n}{sep}{feature}" for n in names_of.get(card, [card])]
    return out


def _card_object(name: str, item: dict[str, Any], root: Path, roots: list[str], dump: Any, out: Materialized,
                 forced: tuple[set[str], bool]) -> str:
    """A library object's card name in the scene card: the card itself, or a copy named ``name`` (written here) that
    carries its static / kinematic / rigid keywords, a derived material and its precision partners' new names."""
    card, names_of = item["card_name"], out.names_of
    plain = item["static"] is None and not _CO.changes(name, item, forced[0])
    if name == card and plain and not names_of:
        return card
    path = catalog.object_card(card, roots)                 # fetched from the Hub here on a fresh install
    data = copy.deepcopy(_load_yaml(path))
    declared = list(data.get("precision_partners") or [])
    partners = _partners(declared, names_of)
    if name == card and plain and partners == declared:
        return card
    data["name"] = name
    if declared:
        data["precision_partners"] = partners
    if item["static"] is not None:
        data["motion"] = "static" if item["static"] else "dynamic"
    data.update(copy.deepcopy(item.get("card_extra") or {}))   # kinematic=, rigid=
    own = _CO.needs_material(name, item, forced[0])
    material = _card_material(name, item, data, (root, roots, dump), forced) if own else None
    if material is not None:
        data["material"] = out.materials[name] = material
    _write(root / "objects" / f"{name}.yaml", data, dump)
    return name


def _card_material(name: str, item: dict[str, Any], data: dict[str, Any], where: tuple[Path, list[str], Any],
                   forced: tuple[set[str], bool]) -> str | None:
    """The material a library object's own card names when it needs one (``material=``, ``mass=`` and the other
    values, a ``set_friction`` pair, a randomized static friction below its dynamic one), else None. Derived as for
    primitives (``_object_material``), from ``material=`` or the card's own material; ``where`` = (root, roots,
    dump)."""
    root, roots, dump = where
    ov = dict(item.get("overrides") or {})
    _CO.refuse_parts(name, data, _CO.value_asks(name, ov, forced[0]))
    base = str(item.get("material") or data["material"])
    vol = _CO.volume(name, item["card_name"], roots) if "mass" in ov else None
    spec = {"kind": "card", "material": base, "overrides": ov, "randomize": item.get("randomize") or [], "volume": vol}
    material = _object_material(name, spec, root, roots, (forced[0], False), dump)    # no precise-fit stiffening
    return None if material == data["material"] else material


def _raw_placement(name: str, item: dict[str, Any], root: Path, roots: list[str], forced: tuple[set[str], bool],
                   out: Materialized, dump: Any) -> dict[str, Any]:
    """A loaded scene card's placement as it was; its card is written again (same name) with a derived material when
    ``set_friction`` names the object or its randomized static friction falls below its dynamic friction."""
    placement = copy.deepcopy(item["raw_placement"])
    card = str(placement["object"])
    out.cards.append(card)
    rows = {"card_name": card, "randomize": placement.get("randomize") or []}
    if not _CO.needs_material(name, rows, forced[0]):
        return placement
    data = copy.deepcopy(_load_yaml(catalog.object_card(card, roots)))
    material = _card_material(name, rows, data, (root, roots, dump), forced)
    if material is not None:
        _CO.refuse_shared(name, card, out.scene.items)
        data["material"] = out.materials[name] = material
        _write(root / "objects" / f"{card}.yaml", data, dump)
    return placement


def _robot_ref(name: str, item: dict[str, Any], root: Path) -> dict[str, Any]:
    """The scene card's robot entry. An ``mjcf=`` robot is written to ``robots/<name>.xml`` and referenced relative
    to the card root, as the library's scenes reference theirs, so a saved folder still loads after it is moved or
    copied (an absolute path pointed at the old place); a robot file keeps its own absolute path."""
    if item.get("raw") is not None:
        return copy.deepcopy(item["raw"])
    path = item["path"]
    if path is None:
        path = f"robots/{name}.xml"
        (root / path).write_text(item["mjcf"], encoding="utf-8")
    ref = {"path": path, "kind": item["robot_kind"], "base_pose": [float(x) for x in item["pose"]], "name": name}
    if item.get("contact_gap") is not None:
        ref["contact_gap"] = float(item["contact_gap"])
    return ref


def _scene_pair(pair: dict[str, Any], materials: dict[str, str], scene: Any) -> dict[str, Any]:
    names = []
    for n in (pair["a"], pair["b"]):
        if n not in materials:
            kind = scene.items[n]["kind"]
            fix = ("set a robot's friction in its file (MJCF <geom friction=...>); " if kind == "robot" else "")
            raise SceneError(f"set_friction: {n!r} is a {kind} with no material of its own", f"{fix}set_friction "
                             "works between objects placed with add_* or add_object, and those of a loaded scene",
                             field=n)
        names.append(materials[n])
    if names[0] == names[1]:
        raise SceneError(f"set_friction({pair['a']!r}, {pair['b']!r}): both use material {names[0]!r}",
                         "pass friction= on one of the two objects instead", field="mu")
    mu = A.user_q(pair["mu"], "set through alloy.Scene.set_friction")
    return {"a": names[0], "b": names[1], "mu_static": mu, "mu_dynamic": dict(mu)}
