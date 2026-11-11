# SPDX-License-Identifier: Apache-2.0
"""A small glTF 2.0 binary (.glb) reader for the camera: triangle meshes with UVs and base-colour images.

It reads what the scan importers write (``alloy.importers.common.visual.export_glb``: trimesh's glb): one or more
meshes in a node tree, triangle primitives with POSITION and TEXCOORD_0, materials with ``baseColorFactor``,
``baseColorTexture``, ``roughnessFactor`` and ``metallicFactor``, images embedded in the binary chunk (or as data /
relative URIs). Only numpy and json are needed; images stay encoded (``GlbImage.data``) until a caller decodes them.

Example::

    parts = read_glb("meshes/ycb_025_mug/visual.glb")      # one GlbPart per primitive, in the glb's own frame
    parts[0].points.shape, parts[0].uv.shape, parts[0].image.mime   # (V, 3), (V, 2), "image/jpeg"
"""
from __future__ import annotations

import base64
import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GLB_MAGIC, CHUNK_JSON, CHUNK_BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_NORM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


class GlbError(ValueError):
    """A glb file this reader cannot use (the message says what)."""


@dataclass(frozen=True)
class GlbImage:
    """An encoded image: its bytes and MIME type ("image/jpeg" or "image/png")."""

    data: bytes
    mime: str


@dataclass
class GlbPart:
    """One triangle primitive, its node transform applied: points (V, 3) float32, tris (T, 3) int32, uv (V, 2)
    float32 or None (glTF convention: v down from the image top), and its material."""

    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray | None
    factor: tuple = (1.0, 1.0, 1.0, 1.0)  # baseColorFactor (linear RGBA)
    image: GlbImage | None = None  # baseColorTexture's image
    roughness: float = 1.0
    metallic: float = 1.0
    name: str = ""
    notes: list = field(default_factory=list)  # what was not read (texCoord > 0, texture transforms, ...)


def _chunks(raw: bytes) -> tuple[dict, bytes]:
    if len(raw) < 20:
        raise GlbError("too short for a glb")
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC or version != 2:
        raise GlbError("not a glTF 2.0 binary file")
    doc, binary, pos = None, b"", 12
    while pos + 8 <= min(length, len(raw)):
        size, kind = struct.unpack_from("<II", raw, pos)
        body = raw[pos + 8: pos + 8 + size]
        if kind == CHUNK_JSON:
            doc = json.loads(body.decode("utf-8"))
        elif kind == CHUNK_BIN and not binary:
            binary = body
        pos += 8 + size
    if doc is None:
        raise GlbError("no JSON chunk")
    return doc, binary


class _Doc:
    """The parsed JSON with its buffers, for accessor and image reads."""

    def __init__(self, doc: dict, binary: bytes, base: Path):
        self.doc, self.base = doc, base
        self.buffers = [self._buffer(b, binary) for b in doc.get("buffers", [])]

    def _buffer(self, spec: dict, binary: bytes) -> bytes:
        uri = spec.get("uri")
        return binary if uri is None else self._uri(uri)

    def _uri(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        path = (self.base / uri).resolve()
        if self.base.resolve() not in path.parents:
            raise GlbError(f"uri {uri!r} points outside the glb's folder")
        return path.read_bytes()

    def view(self, index: int) -> tuple[bytes, int, int]:
        """(buffer bytes, byte offset, byte stride or 0) of a bufferView."""
        bv = self.doc["bufferViews"][index]
        return self.buffers[bv["buffer"]], int(bv.get("byteOffset", 0)), int(bv.get("byteStride", 0))

    def accessor(self, index: int) -> np.ndarray:
        """An accessor as a float32 / integer array (count, width); normalized integers come back as float32."""
        acc = self.doc["accessors"][index]
        if "sparse" in acc or "bufferView" not in acc:
            raise GlbError(f"accessor {index}: sparse or empty accessors are not read")
        dtype, width, count = np.dtype(_COMPONENTS[acc["componentType"]]), _WIDTH[acc["type"]], int(acc["count"])
        buf, off, stride = self.view(acc["bufferView"])
        off += int(acc.get("byteOffset", 0))
        item = dtype.itemsize * width
        if stride and stride != item:
            rows = np.lib.stride_tricks.as_strided(np.frombuffer(buf, np.uint8, offset=off),
                                                   (count, item), (stride, 1))
            arr = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
        else:
            arr = np.frombuffer(buf, dtype, count * width, off).reshape(count, width)
        if acc.get("normalized") and acc["componentType"] in _NORM:
            return np.clip(arr.astype(np.float32) / _NORM[acc["componentType"]], -1.0, 1.0)
        return arr

    def image(self, texture: int) -> GlbImage | None:
        tex = self.doc.get("textures", [])[texture]
        src = tex.get("source")
        if src is None:
            return None
        img = self.doc["images"][src]
        if "bufferView" in img:
            buf, off, _ = self.view(img["bufferView"])
            size = int(self.doc["bufferViews"][img["bufferView"]]["byteLength"])
            return GlbImage(bytes(buf[off: off + size]), img.get("mimeType", "image/jpeg"))
        uri = img.get("uri", "")
        mime = uri[5:].split(";", 1)[0] if uri.startswith("data:") else img.get("mimeType", _mime_of(uri))
        return GlbImage(self._uri(uri), mime)


def _mime_of(uri: str) -> str:
    return "image/png" if uri.lower().endswith(".png") else "image/jpeg"


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T  # glTF stores column-major
    t = np.eye(4)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t[:3, :3] = rot * np.asarray(node.get("scale", (1.0, 1.0, 1.0)))[None, :]
    t[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return t


def _mesh_nodes(doc: dict) -> list[tuple[int, np.ndarray]]:
    """(mesh index, world-from-mesh 4x4) of every node with a mesh, walking the default scene."""
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    roots = scenes[doc.get("scene", 0)]["nodes"] if scenes else list(range(len(nodes)))
    out, stack = [], [(r, np.eye(4), 0) for r in roots]
    while stack:
        k, parent, depth = stack.pop()
        if depth > 64:
            raise GlbError("node tree deeper than 64 (a cycle?)")
        m = parent @ _node_matrix(nodes[k])
        if "mesh" in nodes[k]:
            out.append((int(nodes[k]["mesh"]), m))
        stack.extend((c, m, depth + 1) for c in nodes[k].get("children", []))
    if not nodes and doc.get("meshes"):
        out = [(i, np.eye(4)) for i in range(len(doc["meshes"]))]
    return out


def _material(gd: _Doc, index: int | None, part: GlbPart) -> None:
    if index is None:
        return
    mat = gd.doc.get("materials", [])[index]
    pbr = mat.get("pbrMetallicRoughness", {})
    part.factor = tuple(float(v) for v in pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)))
    part.roughness = float(pbr.get("roughnessFactor", 1.0))
    part.metallic = float(pbr.get("metallicFactor", 1.0))
    part.name = str(mat.get("name", ""))
    tex = pbr.get("baseColorTexture")
    if tex is None:
        return
    if int(tex.get("texCoord", 0)) != 0:
        part.notes.append(f"texture uses TEXCOORD_{tex['texCoord']} (only TEXCOORD_0 is read)")
    if tex.get("extensions"):
        part.notes.append(f"texture extensions not applied: {sorted(tex['extensions'])}")
    part.image = gd.image(int(tex["index"]))


def _primitive(gd: _Doc, prim: dict, xf: np.ndarray) -> GlbPart | None:
    if int(prim.get("mode", 4)) != 4:
        return None  # points / lines / strips are not drawn
    attrs = prim.get("attributes", {})
    if "POSITION" not in attrs:
        return None
    pts = gd.accessor(attrs["POSITION"]).astype(np.float64)
    pts = pts @ xf[:3, :3].T + xf[:3, 3]
    if "indices" in prim:
        tris = gd.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
    else:
        tris = np.arange(len(pts) - len(pts) % 3).reshape(-1, 3)
    if np.linalg.det(xf[:3, :3]) < 0:
        tris = tris[:, [0, 2, 1]]  # a mirroring node flips the winding
    uv = gd.accessor(attrs["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in attrs else None
    part = GlbPart(pts.astype(np.float32), tris.astype(np.int32), uv)
    _material(gd, prim.get("material"), part)
    return part


def read_glb(path) -> list[GlbPart]:
    """Every triangle primitive of a .glb file, in its scene's frame (node transforms applied). Raises GlbError
    (or OSError) for a file it cannot read. Example: ``read_glb(path)[0].uv``."""
    path = Path(path)
    doc, binary = _chunks(path.read_bytes())
    gd = _Doc(doc, binary, path.parent)
    parts = []
    for mesh, xf in _mesh_nodes(doc):
        for prim in doc["meshes"][mesh].get("primitives", []):
            part = _primitive(gd, prim, xf)
            if part is not None and len(part.tris):
                parts.append(part)
    if not parts:
        raise GlbError("no triangle mesh in the file")
    return parts


def write_glb(path, parts: list[dict]) -> Path:
    """Write a .glb of triangle meshes (one node and primitive each): every dict has ``points`` (V, 3), ``tris``
    (T, 3), optional ``uv`` (V, 2, glTF convention), ``image`` (encoded bytes, PNG or JPEG), ``mime``, ``factor``
    (RGBA), ``roughness``, ``metallic``. For tests and fixtures.

    Example: ``write_glb(p, [{"points": v, "tris": f}])``."""
    blob, views, accs, meshes, mats, imgs, texs = bytearray(), [], [], [], [], [], []

    def add(data: bytes) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        return len(views) - 1

    def accessor(arr: np.ndarray, kind: str, ctype: int) -> int:
        spec = {"bufferView": add(arr.tobytes()), "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            spec |= {"min": arr.min(0).tolist(), "max": arr.max(0).tolist()}
        accs.append(spec)
        return len(accs) - 1

    for p in parts:
        attrs = {"POSITION": accessor(np.asarray(p["points"], np.float32).reshape(-1, 3), "VEC3", 5126)}
        if p.get("uv") is not None:
            attrs["TEXCOORD_0"] = accessor(np.asarray(p["uv"], np.float32).reshape(-1, 2), "VEC2", 5126)
        pbr = {"baseColorFactor": list(p.get("factor", (1.0, 1.0, 1.0, 1.0))),
               "roughnessFactor": float(p.get("roughness", 0.5)), "metallicFactor": float(p.get("metallic", 0.0))}
        if p.get("image") is not None:
            imgs.append({"bufferView": add(p["image"]), "mimeType": p.get("mime", "image/png")})
            texs.append({"source": len(imgs) - 1})
            pbr["baseColorTexture"] = {"index": len(texs) - 1}
        mats.append({"pbrMetallicRoughness": pbr})
        meshes.append({"primitives": [{"attributes": attrs, "mode": 4, "material": len(mats) - 1,
                                       "indices": accessor(np.asarray(p["tris"], np.uint32).reshape(-1), "SCALAR",
                                                           5125)}]})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(meshes)))}],
           "nodes": [{"mesh": i} for i in range(len(meshes))], "meshes": meshes, "materials": mats,
           "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    if imgs:
        doc |= {"images": imgs, "textures": texs}
    js = json.dumps(doc).encode("utf-8")
    js += b" " * (-len(js) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    out = struct.pack("<III", GLB_MAGIC, 2, 28 + len(js) + len(blob))
    out += struct.pack("<II", len(js), CHUNK_JSON) + js + struct.pack("<II", len(blob), CHUNK_BIN) + bytes(blob)
    Path(path).write_bytes(out)
    return Path(path)
