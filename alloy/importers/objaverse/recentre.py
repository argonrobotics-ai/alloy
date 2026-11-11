# SPDX-License-Identifier: Apache-2.0
"""Put every object's origin at the centre of its physics mesh's bounding box.

Objaverse uploads keep their author's origin, which can lie far from the object: after scaling to metres, 123 of the
first 1,990 objects sat more than ten half-sizes or 1 m from it, up to 750 m. A body whose geometry lies that far from
its frame loses float32 precision. Newton's mass of such a mesh came out up to 1,900 times the card's, and every
orientation error is multiplied by the offset.

This step runs on the shared mesh pipeline's output folders, after ``alloy.importers.common.batch`` and before
``assemble``, which refuses meshes that are not centred. It moves:

- ``physics.obj`` and the ``physics.hulls.obj`` sidecar, re-tagged with the new sha256;
- ``visual.glb``, by adding the shift to its root nodes' transforms in the glTF JSON (float64; vertices, textures and
  the binary chunk are untouched);
- ``result.json``: file hashes and sizes, ``mass_props`` bounds and centre of mass, and ``origin_shift``, the total
  shift in metres.

Running it again on a centred folder changes nothing.

Example:
    >>> bbox_centre(np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]])).tolist()
    [1.0, 2.0, 0.5]
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from alloy.importers.common import objio
from alloy.importers.common.errors import ImporterError
from alloy.importers.common.visual import Z_UP_TO_GLTF
from alloy.importers.objaverse.cards import HULLS, PHYSICS

VISUAL = "visual.glb"
RESULT = "result.json"
CENTRE_TOL = 1e-6                      # m, largest bounding-box centre offset that counts as centred
_GLB_MAGIC, _JSON_CHUNK = 0x46546C67, 0x4E4F534A


def bbox_centre(vertices: np.ndarray) -> np.ndarray:
    """Centre of the axis-aligned bounding box of ``vertices`` (N, 3)."""
    v = np.asarray(vertices, dtype=np.float64)
    return 0.5 * (v.min(axis=0) + v.max(axis=0))


def is_centred(result: Mapping[str, Any]) -> bool:
    """True when a pipeline ``result.json`` row records physics-mesh bounds centred on the origin."""
    bounds = result.get("mass_props", {}).get("bounds")
    return bounds is not None and float(np.abs(bbox_centre(np.asarray(bounds))).max()) <= CENTRE_TOL


def _read_mesh_obj(text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Header comment lines (without ``# ``), vertices and 0-based faces of an importer ``physics.obj``."""
    header, verts, faces = [], [], []
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "#" and not verts:
            header.append(line[2:])
        elif tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok[0] == "f":
            faces.append([int(t.split("/")[0]) - 1 for t in tok[1:4]])
    return header, np.asarray(verts, np.float64).reshape(-1, 3), np.asarray(faces, np.int64).reshape(-1, 3)


def translate_glb(data: bytes, shift_gltf: np.ndarray) -> bytes:
    """``data`` (a GLB) with ``shift_gltf`` (glTF frame, metres) added to the transform of every scene root node.

    Only the JSON chunk changes; the root nodes' matrices or translations stay in float64.
    """
    magic, version, _ = struct.unpack_from("<III", data, 0)
    js_len, js_type = struct.unpack_from("<II", data, 12)
    if magic != _GLB_MAGIC or version != 2 or js_type != _JSON_CHUNK:
        raise ImporterError(VISUAL, "recentre", "not a glTF 2.0 binary with a leading JSON chunk; re-export it")
    gltf = json.loads(data[20:20 + js_len])
    if gltf.get("animations"):
        raise ImporterError(VISUAL, "recentre", "animated glTF: root transforms are animated, cannot shift them")
    t = [float(x) for x in shift_gltf]
    roots = sorted({n for s in gltf.get("scenes", []) for n in s.get("nodes", [])})
    for k in roots:
        node = gltf["nodes"][k]
        if "matrix" in node:                           # column-major: the translation is elements 12..14
            node["matrix"][12:15] = [m + d for m, d in zip(node["matrix"][12:15], t)]
        else:                                          # T * R * S: a world shift adds to T
            node["translation"] = [m + d for m, d in zip(node.get("translation", [0.0, 0.0, 0.0]), t)]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    rest = data[20 + js_len:]
    return struct.pack("<III", _GLB_MAGIC, 2, 12 + 8 + len(js) + len(rest)) + struct.pack("<II", len(js),
                                                                                          _JSON_CHUNK) + js + rest


def _check_unchanged(d: Path, result: Mapping[str, Any]) -> None:
    """The folder's files must still be the ones ``result.json`` describes (an interrupted run leaves a mix)."""
    for f in (PHYSICS, HULLS, VISUAL):
        want = result["files"].get(f)
        if (d / f).is_file() and want is not None and objio.sha256_path(d / f) != want:
            raise ImporterError(d.name, "recentre", f"{f} differs from result.json (an interrupted re-centring?); "
                                                    "fetch the folder again or re-run the mesh pipeline")


def _new_files(d: Path, shift: np.ndarray) -> dict[str, bytes]:
    """New bytes of the physics mesh, the hull sidecar and (when present) the visual, moved by ``shift``."""
    header, v, f = _read_mesh_obj((d / PHYSICS).read_text(encoding="utf-8"))
    header = [h for h in header if not h.startswith("origin:")]
    header.append("origin: physics-mesh bounding-box centre")
    phys = objio.mesh_obj_text(v + shift, f, header="\n".join(header)).encode("utf-8")
    old_sha, hulls = objio.read_hulls_obj(d / HULLS)
    if old_sha != objio.sha256_path(d / PHYSICS):
        raise ImporterError(d.name, "recentre", f"{HULLS} was not made from this {PHYSICS}; re-run the mesh pipeline")
    hull_header = [ln[2:] for ln in (d / HULLS).read_text(encoding="utf-8").splitlines()[1:] if ln.startswith("# ")]
    out = {PHYSICS: phys, HULLS: objio.hulls_obj_text([(hv + shift, hf) for hv, hf in hulls], objio.sha256_bytes(phys),
                                                      header="\n".join(hull_header)).encode("utf-8")}
    if (d / VISUAL).is_file():
        out[VISUAL] = translate_glb((d / VISUAL).read_bytes(), Z_UP_TO_GLTF[:3, :3] @ shift)
    return out


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def recentre_dir(d: Path) -> dict[str, Any]:
    """Move one pipeline output folder's origin to its physics mesh's bounding-box centre; returns its new
    ``result.json`` row (unchanged when already centred)."""
    d = Path(d)
    result = json.loads((d / RESULT).read_text(encoding="utf-8"))
    _, v, _ = _read_mesh_obj((d / PHYSICS).read_text(encoding="utf-8"))
    shift = -bbox_centre(v)
    if float(np.abs(shift).max()) <= CENTRE_TOL:
        return result
    _check_unchanged(d, result)
    files = _new_files(d, shift)
    for f, data in files.items():
        _write_atomic(d / f, data)
        result["files"][f] = objio.sha256_bytes(data)
        result.setdefault("sizes", {})[f] = len(data)
    mp = result.setdefault("mass_props", {})
    if "center_mass" in mp:
        mp["center_mass"] = (np.asarray(mp["center_mass"], np.float64) + shift).tolist()
    mp["bounds"] = np.stack([v.min(0) + shift, v.max(0) + shift]).round(7).tolist()
    result["origin_shift"] = (np.asarray(result.get("origin_shift", [0.0, 0.0, 0.0])) + shift).round(9).tolist()
    _write_atomic(d / RESULT, json.dumps(result, indent=1, sort_keys=True).encode("utf-8"))
    return result
