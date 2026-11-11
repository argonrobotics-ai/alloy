# SPDX-License-Identifier: Apache-2.0
"""Thumbnails of imported models: MuJoCo's offscreen renderer (OSMesa on machines without a GPU), PNG out.

The camera looks at the centre of the drawn geometry from a three-quarter view, far enough to frame its bounding
sphere. Only visual geoms are drawn (collision hulls are group 3, regions group 4). Set ``MUJOCO_GL`` before the
first ``import mujoco`` in the process; ``render_thumbnail`` sets it to ``osmesa`` if unset.

Example::

    >>> from alloy.importers.robocasa.thumbs import render_thumbnail
    >>> render_thumbnail(Path("out/mjcf/glass_cup/robocasa_glass_cup_003/model.xml"), Path("thumb.png"))  # doctest: +SKIP
    PosixPath('thumb.png')
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

SIZE = 256


def write_png(path: Path, rgb: np.ndarray) -> Path:
    """Write an (H, W, 3) uint8 image as an 8-bit RGB PNG (no imaging library needed)."""
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _scene_xml(model_path: Path, size: int) -> str:
    root = ET.parse(model_path).getroot()
    comp = root.find("compiler")
    comp = comp if comp is not None else ET.SubElement(root, "compiler")
    comp.set("meshdir", str(model_path.parent))
    comp.set("texturedir", str(model_path.parent))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(size), offheight=str(size))
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6", specular="0.1 0.1 0.1")
    asset = root.find("asset")
    asset = asset if asset is not None else ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="flat", rgb1="0.96 0.96 0.96", rgb2="0.96 0.96 0.96",
                  width="16", height="16")
    world = root.find("worldbody")
    world.insert(0, ET.Element("light", pos="1 -1 2", dir="-0.4 0.4 -1", diffuse="0.5 0.5 0.5", directional="true"))
    return ET.tostring(root, encoding="unicode")


def _visual_bounds(m, d) -> tuple[np.ndarray, float]:
    """Centre and radius of the drawn (group 2) mesh vertices in world coordinates."""
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_dataid[g] < 0:
            continue
        mesh = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mesh]: m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    if not pts:
        raise ValueError("model draws nothing (no group-2 visual meshes)")
    arr = np.concatenate(pts)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    return 0.5 * (lo + hi), float(0.5 * np.linalg.norm(hi - lo))


def _require_offscreen(mujoco) -> None:
    """Raise (instead of letting GLFW abort the process) when MuJoCo has no usable offscreen GL backend."""
    backend = getattr(getattr(mujoco, "GLContext", None), "__module__", "")
    if not backend:
        raise RuntimeError("MuJoCo was imported with rendering disabled (MUJOCO_GL); render with the post step")
    if backend.endswith("glfw") and not os.environ.get("DISPLAY"):
        raise RuntimeError("MuJoCo picked GLFW but there is no display: set MUJOCO_GL=osmesa or egl before the "
                           "first `import mujoco`")


def render_thumbnail(model_path: Path, out: Path, size: int = SIZE) -> Path:
    """Render ``model_path`` to ``out`` (PNG, ``size`` x ``size``); raises if no GL backend is available."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco  # noqa: PLC0415 - heavy import; MUJOCO_GL must be set first

    _require_offscreen(mujoco)
    m = mujoco.MjModel.from_xml_string(_scene_xml(model_path, size))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    centre, radius = _visual_bounds(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = centre, 1.1 * radius / np.sin(np.radians(22.5)), 135, -25
    with mujoco.Renderer(m, size, size) as r:
        r.update_scene(d, camera=cam)
        img = r.render()
    return write_png(out, img)


__all__ = ["SIZE", "render_thumbnail", "write_png"]
