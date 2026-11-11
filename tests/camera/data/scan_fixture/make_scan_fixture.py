# SPDX-License-Identifier: Apache-2.0
"""Rebuild the scan-texture law's fixture: the front panel of the YCB 004 sugar box scan, its own photo texture.

Run from a checkout with the fetched scan on disk (``alloy fetch ycb_004_sugar_box``):

    PYTHONPATH=. python tests/camera/data/scan_fixture/make_scan_fixture.py

It keeps the scan's triangles that face +x in the body frame (the printed front of the box, with the scan's own UV
charts), shrinks the 4,096 px texture 8x (box filter, JPEG quality 90) and writes a card whose physics mesh is the
panel's bounding box (collision "decompose" with its one-hull sidecar, as imported cards have), so the camera's
bounds check between the glb and the display mesh holds. The glb keeps the
source's glTF layout (+Y up, one primitive, POSITION + TEXCOORD_0, the embedded JPEG and its baseColorFactor 0.4,
trimesh's default that the camera reads as 1 beside a texture). Source and licence: ATTRIBUTION.md.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from alloy.importers.common.objio import hulls_obj_text

NAME = "ycb_004_sugar_box_front"
SOURCE = "ycb_004_sugar_box"
FACING = 0.3       # keep triangles whose body-frame normal has x above this
SHRINK = 8         # 4,096 px -> 512 px
HERE = Path(__file__).resolve().parent


def read_glb(path: Path) -> tuple[dict, bytes]:
    """(JSON chunk, BIN chunk) of a glTF binary file."""
    b = path.read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    body = b[20 + n:]
    return json.loads(b[20:20 + n]), body[8:8 + struct.unpack("<I", body[:4])[0]]


def accessor(j: dict, binary: bytes, i: int) -> np.ndarray:
    """Accessor ``i`` as a numpy array (tightly packed views only, as the importer writes them)."""
    a = j["accessors"][i]
    view = j["bufferViews"][a["bufferView"]]
    dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    out = np.frombuffer(binary, dtype, a["count"] * width, start)
    return out.reshape(a["count"], width) if width > 1 else out


def image_bytes(j: dict, binary: bytes) -> bytes:
    view = j["bufferViews"][j["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def to_body(p: np.ndarray) -> np.ndarray:
    """glTF (+Y up) -> body frame (+Z up): (x, y, z) -> (x, -z, y), the inverse of the importer's turn."""
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def front_panel(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Triangles of the box's printed front (body +x)."""
    b = to_body(points.astype(np.float64))
    n = np.cross(b[tris[:, 1]] - b[tris[:, 0]], b[tris[:, 2]] - b[tris[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return tris[n[:, 0] > FACING]


def compact(points: np.ndarray, uv: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    used, inverse = np.unique(tris.ravel(), return_inverse=True)
    return points[used], uv[used], inverse.reshape(-1, 3).astype(np.uint16)


def write_glb(path: Path, points: np.ndarray, uv: np.ndarray, tris: np.ndarray, jpeg: bytes, material: dict) -> None:
    """One primitive (indices, POSITION, TEXCOORD_0) and one material with the embedded JPEG."""
    blobs = [tris.astype(np.uint16).tobytes(), points.astype(np.float32).tobytes(), uv.astype(np.float32).tobytes(),
             jpeg]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\0" * (-len(blob) % 4)
    views[0]["target"], views[1]["target"], views[2]["target"] = 34963, 34962, 34962
    j = {"asset": {"version": "2.0", "generator": "alloy tests/camera/data/scan_fixture/make_scan_fixture.py"},
         "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": NAME, "mesh": 0}],
         "meshes": [{"name": NAME, "primitives": [{"attributes": {"POSITION": 1, "TEXCOORD_0": 2}, "indices": 0,
                                                    "mode": 4, "material": 0}]}],
         "materials": [material], "textures": [{"source": 0}],
         "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
         "accessors": [
             {"bufferView": 0, "componentType": 5123, "count": int(tris.size), "type": "SCALAR"},
             {"bufferView": 1, "componentType": 5126, "count": len(points), "type": "VEC3",
              "min": points.min(0).tolist(), "max": points.max(0).tolist()},
             {"bufferView": 2, "componentType": 5126, "count": len(uv), "type": "VEC2"}],
         "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    text = json.dumps(j, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(binary))
    out += struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(out)


def box_mesh(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The panel's bounding box: 8 corners and 12 triangles, counter-clockwise from outside."""
    corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return corners, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int64)


def box_obj(corners: np.ndarray, tris: np.ndarray) -> str:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in tris]
    return "\n".join(lines) + "\n"


def card(physics_sha: str) -> str:
    return (f"# the scan-texture law's fixture: the front panel of the YCB 004 sugar box scan (see ATTRIBUTION.md);\n"
            f"# rebuilt by make_scan_fixture.py, do not edit\n"
            f"schema: alloy.object/1\nname: {NAME}\nshape:\n  mesh:\n    path: meshes/{NAME}/physics.obj\n"
            f"    sha256: {physics_sha}\n    collision: decompose\n    max_parts: 1\nmaterial: {NAME}_eff\n")


def material_card(cache: Path) -> str:
    """The scan's own material card (density, friction, elasticity and their sources) under the fixture's name. Like
    every imported card's material it has no look, so the camera's look fallback does not cover the texture."""
    text = (cache / "materials" / "ycb" / f"{SOURCE}_eff.yaml").read_text()
    return text.replace(f"name: {SOURCE}_eff", f"name: {NAME}_eff", 1)


def main() -> None:
    cache = Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy")) / "imports" / "ycb_gso"
    j, binary = read_glb(cache / "meshes" / SOURCE / "visual.glb")
    prim = j["meshes"][0]["primitives"][0]
    tris = accessor(j, binary, prim["indices"]).reshape(-1, 3)
    points = accessor(j, binary, prim["attributes"]["POSITION"])
    uv = accessor(j, binary, prim["attributes"]["TEXCOORD_0"])
    points, uv, tris = compact(points, uv, front_panel(points, tris))
    img = Image.open(io.BytesIO(image_bytes(j, binary))).convert("RGB")
    img = img.reduce(SHRINK)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    out = HERE / "meshes" / NAME
    out.mkdir(parents=True, exist_ok=True)
    write_glb(out / "visual.glb", points, uv, tris, buf.getvalue(), j["materials"][0])
    body = to_body(points.astype(np.float64))
    corners, box_tris = box_mesh(body.min(0), body.max(0))
    obj = box_obj(corners, box_tris)
    sha = hashlib.sha256(obj.encode()).hexdigest()
    (out / "physics.obj").write_text(obj)
    # as an imported card: collision "decompose" with its hull sidecar shipped (one hull, the box), so a build needs
    # no CoACD and the camera draws the glb in place of the card's display mesh ("<item>/visual")
    hulls = hulls_obj_text([(corners, box_tris)], sha, header=f"{NAME}: 1 hull, the box")
    (out / "physics.hulls.obj").write_text(hulls)
    (HERE / "objects").mkdir(exist_ok=True)
    (HERE / "objects" / f"{NAME}.yaml").write_text(card(sha))
    (HERE / "materials").mkdir(exist_ok=True)
    (HERE / "materials" / f"{NAME}_eff.yaml").write_text(material_card(cache))
    print(f"{NAME}: {len(tris)} triangles, {len(points)} vertices, texture {img.size}, "
          f"glb {(out / 'visual.glb').stat().st_size} bytes, bounds {body.min(0).round(4)} .. {body.max(0).round(4)}")


if __name__ == "__main__":
    main()
