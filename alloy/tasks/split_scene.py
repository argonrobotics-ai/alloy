# SPDX-License-Identifier: Apache-2.0
"""Replace the large collision hulls of selected bodies in a decompiled scene by error-bounded convex pieces.

Each collision mesh geom on a body whose name matches ``SceneSplit.bodies`` is split with
:func:`alloy.tasks.hull_split.split_hull`, if its hull has more than ``max_verts`` vertices. The geom becomes one geom
per piece (``<name>_p<i>``, same body, pose and contact attributes); each piece is a new content-addressed mesh
(``<mesh>_p<i>``, ``maxhullvert="-1"``, so Newton keeps it as written). Explicit ``<pair>`` elements that name a split
geom are refused rather than silently dropped. Inertia is unaffected: the converter writes every body's
``<inertial>``. The report goes next to the scene as ``hull_split.json``.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from alloy.core.errors import TraceError
from alloy.tasks.hull_split import SplitConfig, outward_faces, split_hull
from alloy.tasks.msh import MeshData, read_msh, store_msh

log = logging.getLogger("alloy.tasks")
REPORT_NAME = "hull_split.json"


@dataclass(frozen=True)
class SceneSplit:
    bodies: str                                   # regex on body names; T12 uses "." (every body)
    cfg: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True)
class MeshSplit:
    mesh: str
    hull_verts: int
    piece_verts: tuple[int, ...]
    bound_m: float                                # max distance of the recorded hull surface to the pieces


def _collides(geom: ET.Element) -> bool:
    return int(geom.get("contype", "1")) != 0 or int(geom.get("conaffinity", "1")) != 0


def _split_asset(asset: ET.Element, mesh: ET.Element, store: Path, cfg: SplitConfig) -> tuple[list[str], MeshSplit]:
    """Split one mesh asset; returns the piece mesh names (empty if the hull is already small) and the report."""
    name, file = mesh.get("name", ""), PurePosixPath(mesh.get("file", ""))
    verts = read_msh(store / file.name).vertices.astype(np.float64)
    res = split_hull(verts, cfg)
    report = MeshSplit(name, res.hull_verts, tuple(len(p) for p in res.pieces), res.deviation_m)
    if res.hull_verts <= cfg.max_verts:
        return [], report
    names, at = [], list(asset).index(mesh)
    for i, piece in enumerate(res.pieces):
        piece_name = f"{name}_p{i}"
        fname = store_msh(store, piece_name, MeshData(piece.astype(np.float32), outward_faces(piece).astype(np.int32)))
        elem = ET.Element("mesh", {"name": piece_name, "file": str(file.parent / fname), "maxhullvert": "-1"})
        asset.insert(at + 1 + i, elem)
        names.append(piece_name)
    return names, report


def _replace_geom(body: ET.Element, geom: ET.Element, pieces: list[str]) -> None:
    at = list(body).index(geom)
    body.remove(geom)
    for i, piece in enumerate(pieces):
        attrs = dict(geom.attrib) | {"mesh": piece}
        if "name" in attrs:
            attrs["name"] = f"{attrs['name']}_p{i}"
        body.insert(at + i, ET.Element("geom", attrs))


def _drop_unused_meshes(root: ET.Element, asset: ET.Element) -> None:
    used = {g.get("mesh") for g in root.iter("geom")}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used:
            asset.remove(mesh)


def _check_pairs(root: ET.Element, split_geoms: set[str]) -> None:
    named = {p.get(k) for p in root.iter("pair") for k in ("geom1", "geom2")}
    if named & split_geoms:
        raise TraceError(f"explicit <pair> on split geoms {sorted(named & split_geoms)}: not supported")


def split_scene_hulls(root: ET.Element, store: Path, split: SceneSplit) -> list[MeshSplit]:
    """Split, in place, the large collision hulls of the bodies matching ``split.bodies``; returns one report per
    distinct mesh examined."""
    asset = root.find("asset")
    if asset is None:
        return []
    meshes = {m.get("name"): m for m in asset.findall("mesh")}
    pattern, pieces, reports, split_geoms = re.compile(split.bodies), {}, [], set()
    for body in root.iter("body"):
        if not pattern.search(body.get("name", "")):
            continue
        for geom in [g for g in body.findall("geom") if g.get("mesh") in meshes and _collides(g)]:
            mesh = geom.get("mesh", "")
            if mesh not in pieces:
                pieces[mesh], report = _split_asset(asset, meshes[mesh], store, split.cfg)
                reports.append(report)
            if pieces[mesh]:
                split_geoms.add(geom.get("name", ""))
                _replace_geom(body, geom, pieces[mesh])
    _check_pairs(root, split_geoms)
    _drop_unused_meshes(root, asset)
    return reports


def write_report(scene_dir: Path, split: SceneSplit, reports: list[MeshSplit]) -> Path:
    """``hull_split.json``: the settings and, per mesh, the recorded hull size, the pieces and the deviation bound."""
    changed = [r for r in reports if r.hull_verts > split.cfg.max_verts]
    doc = {"bodies": split.bodies, "config": asdict(split.cfg),
           "max_bound_m": max((r.bound_m for r in changed), default=0.0),
           "max_piece_verts": max((max(r.piece_verts) for r in changed), default=0),
           "meshes": [asdict(r) for r in reports]}  # fmt: skip
    path = scene_dir / REPORT_NAME
    path.write_text(json.dumps(doc, indent=1) + "\n")
    log.info("hull split %s: %d of %d meshes split into %d pieces, max %d vertices, bound %.4g m", split.bodies,
             len(changed), len(reports), sum(len(r.piece_verts) for r in changed), doc["max_piece_verts"],
             doc["max_bound_m"])  # fmt: skip
    return path
