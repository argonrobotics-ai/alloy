# SPDX-License-Identifier: Apache-2.0
"""Visual meshes for the arm embodiment packs, as a layer beside the physics: ``library/robots/<pack>/visuals/``.

The pack converter (:mod:`alloy.tasks.emb_scene`) drops visual-only geoms, so a pack alone rendered as its collision
hulls. The pack's ``scene.xml`` stays as it is, byte for byte; its source's visual meshes live next to it and are
attached when the pack is loaded:

- ``visuals/visuals.json``: the source (url, the pack's pinned commit, licence, sha256 of every model and mesh file
  used) and one entry per visual-only geom of the source (contype = conaffinity = 0): its pack body, mesh file, pos /
  quat in the body frame, rgba of its material; or ``same_as`` another pack's layer, once per ``prefixes`` entry (a
  contact variant: its verdict scene's; the bimanual pack: the FR3's, once per arm);
- ``visuals/<model dir>_<mesh>.msh``: each source OBJ / STL as MuJoCo ``.msh``, exactly coincident vertices merged and
  zero-area faces dropped (the surface is the source's).

Attaching (:mod:`alloy.scene.robot_visuals`: every scene load, the clip arms, ``scene_with_visuals``) adds display
geometry only: no collision, density 0, under the packs' explicit inertials (``inertiafromgeom="false"``), so no
mass, inertia or contact changes. Geoms are named as the clip kit names them (``<body>_vis<k>_<material>``, mesh
``vis_<body>_<mesh>``) so look rules match.

Build: ``python -m alloy.tasks.emb_visuals`` reads a mujoco_menagerie checkout at the packs' commit
(``ALLOY_MENAGERIE``), writes every layer and rewrites ``library/robots/SHA256SUMS.json``.
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.scene.robot_visuals import MANIFEST, clear_cache, layer_dir
from alloy.tasks.emb_scene import ROBOTS, scene_path
from alloy.tasks.emb_specs import EMBODIMENTS, Embodiment
from alloy.tasks.msh import MeshData, sha256_file, write_manifest, write_msh

log = logging.getLogger("alloy.tasks")

SCHEMA = "alloy.robot_visuals/1"
NOTE = ("display only: contype = conaffinity = 0, density 0, under the bodies' explicit inertials; meshes as .msh with "
        "exactly coincident vertices merged and zero-area faces dropped")


@dataclass(frozen=True)
class VisualSource:
    model: str          # source model xml, relative to the source root
    prefix: str = ""    # body-name prefix of that model's bodies in the pack


_FR3, _HAND = VisualSource("franka_fr3/fr3.xml"), VisualSource("franka_emika_panda/hand.xml")
#: verdict packs whose layer is built from their source
SOURCES: dict[str, tuple[VisualSource, ...]] = {
    "e1_fr3": (_FR3, _HAND),
    "e2_panda": (VisualSource("franka_emika_panda/panda.xml"),),
    "e3_ur10e_2f85": (VisualSource("universal_robots_ur10e/ur10e.xml"),
                      VisualSource("robotiq_2f85/2f85.xml", "gripper_")),
}
#: verdict packs that reuse another pack's layer: key -> (that key, one prefix per copy)
LINKS: dict[str, tuple[str, tuple[str, ...]]] = {"e4_fr3_bimanual": ("e1_fr3", ("left_", "right_"))}


# --- build ------------------------------------------------------------------------------------------------------------


def visual_mesh(path: Path) -> MeshData:
    """A source mesh file as float32 vertices with exact duplicates merged and zero-area (repeated-index) faces gone."""
    from alloy.cards.geometry import read_mesh

    src = read_mesh(path)
    verts, inv = np.unique(src.vertices.astype(np.float32), axis=0, return_inverse=True)
    faces = inv.reshape(-1)[src.faces].reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return MeshData(verts.astype(np.float32), faces[keep].astype(np.int32))


def _pack_bodies(e: Embodiment) -> set[str]:
    return {b.get("name") for b in ET.parse(scene_path(e)).getroot().iter("body")}


def _source_geoms(e: Embodiment, src: VisualSource, out: Path, rec: dict) -> list[dict]:
    """One source model's visual geoms as manifest entries; their meshes written to ``out``."""
    import mujoco

    root = e.source_root()
    spec = mujoco.MjSpec.from_file(str(root / src.model))     # parse only: resolves the source's default classes
    meshdir = Path(src.model).parent / (spec.meshdir or "")
    meshes = {m.name: m for m in spec.meshes}
    rgba = {m.name: list(m.rgba) for m in spec.materials}
    bodies, entries = _pack_bodies(e), []
    for b in spec.bodies:
        vis = [g for g in b.geoms if g.contype == 0 and g.conaffinity == 0 and g.meshname]
        body = src.prefix + b.name
        if vis and body not in bodies:
            raise KeyError(f"{e.name}: no body {body!r} for the {src.model} visuals")
        for k, g in enumerate(vis):
            if g.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
                raise ValueError(f"{src.model}: visual geom {k} of {b.name!r} is not oriented by a quat")
            mesh = meshes[g.meshname]
            rel = (meshdir / mesh.file).as_posix()
            fname = f"{Path(src.model).parent.name}_{mesh.name}.msh"
            if rel not in rec["meshes"]:
                write_msh(out / fname, visual_mesh(root / rel))
                rec["meshes"][rel] = sha256_file(root / rel)
            entry = {"name": f"{body}_vis{k}_{(g.material or 'plain').replace(' ', '_')}", "body": body,
                     "mesh": f"vis_{body}_{mesh.name}", "file": fname, "pos": [float(x) for x in g.pos],
                     "quat": [float(x) for x in g.quat], "rgba": [float(x) for x in rgba.get(g.material, list(g.rgba))],
                     "material": g.material}  # fmt: skip
            if list(mesh.scale) != [1.0, 1.0, 1.0]:
                entry["scale"] = [float(x) for x in mesh.scale]
            entries.append(entry)
    return entries


def _fresh(e: Embodiment) -> Path:
    out = layer_dir(scene_path(e))
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    return out


def build(key: str) -> dict:
    """Write the layer of verdict pack ``key`` from its source. Returns a summary."""
    e = EMBODIMENTS[key]
    out = _fresh(e)
    url, commit = e.source_ref()
    src = {"url": url, "commit": commit, "licence": e.licence,
           "models": {s.model: sha256_file(e.source_root() / s.model) for s in SOURCES[key]}, "meshes": {}}  # fmt: skip
    geoms = [g for s in SOURCES[key] for g in _source_geoms(e, s, out, src)]
    rec = {"schema": SCHEMA, "pack": e.name, "by": "alloy.tasks.emb_visuals", "source": src, "note": NOTE,
           "geoms": geoms}
    (out / MANIFEST).write_text(json.dumps(rec, indent=1) + "\n")
    return {"pack": e.name, "geoms": len(geoms), "meshes": len(src["meshes"])}


def link(key: str, other: str, prefixes: tuple[str, ...] = ("",)) -> dict:
    """Write the layer of pack ``key`` as ``other``'s, once per prefix."""
    e = EMBODIMENTS[key]
    out = _fresh(e)
    rec = {"schema": SCHEMA, "pack": e.name, "same_as": EMBODIMENTS[other].name, "prefixes": list(prefixes)}
    (out / MANIFEST).write_text(json.dumps(rec, indent=1) + "\n")
    return {"pack": e.name, "same_as": rec["same_as"], "prefixes": list(prefixes)}


def keys() -> list[str]:
    """Every pack with a layer: the verdict packs above and their contact variants (with a scene)."""
    return sorted(k for k, e in EMBODIMENTS.items() if e.builder in SOURCES.keys() | LINKS.keys()
                  and scene_path(e).is_file())  # fmt: skip


def build_all() -> list[dict]:
    done = [build(k) for k in sorted(SOURCES)]
    done += [link(k, other, prefixes) for k, (other, prefixes) in sorted(LINKS.items())]
    done += [link(k, EMBODIMENTS[k].base) for k in keys() if EMBODIMENTS[k].base]
    clear_cache()
    write_manifest(ROBOTS)
    return done


def main(argv: list[str] | None = None) -> int:
    """``python -m alloy.tasks.emb_visuals``: write every pack's visual layer and the library/robots manifest."""
    import argparse

    argparse.ArgumentParser(description=main.__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for row in build_all():
        log.info("%s", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
