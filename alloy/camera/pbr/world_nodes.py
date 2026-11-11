# SPDX-License-Identifier: Apache-2.0
"""World node graphs beyond a plain HDRI: the Mapping node that rotates the Environment Texture, and the common
"camera sees a plain colour, everything else sees the HDRI" mix (Light Path Is Camera Ray -> Mix Shader).

Example::

    rot = mapping_rotation(tree)        # 3x3: world direction -> texture direction (identity without Mapping)
    cam = camera_background(tree)       # linear RGB camera rays see, or None
"""
from __future__ import annotations

import numpy as np

from ..errors import CameraError


def _euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Blender's XYZ Euler rotation matrix (applied X first): Rz @ Ry @ Rx."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _link_into(tree: dict, node: str, socket: str) -> dict | None:
    return next((lk for lk in tree.get("links", []) if lk["to"] == node and lk["to_id"] == socket
                 and not lk.get("muted", False)), None)


def mapping_rotation(tree: dict) -> np.ndarray:
    """The rotation a Mapping node between Texture Coordinate and the Environment Texture applies to the lookup
    direction (identity without one). Location must be 0 and scale uniform: the lookup normalises the direction,
    so only rotation (and a uniform scale) keep the map a rotation."""
    nodes = {n["name"]: n for n in tree.get("nodes", [])}
    env = next((n for n in tree.get("nodes", []) if n["type"] == "ShaderNodeTexEnvironment"), None)
    link = _link_into(tree, env["name"], "Vector") if env else None
    if link is None or nodes[link["from"]]["type"] != "ShaderNodeMapping":
        return np.eye(3)
    m = nodes[link["from"]]
    vals = {i["name"]: i for i in m.get("inputs", [])}
    if any(vals.get(k, {}).get("linked") for k in ("Location", "Rotation", "Scale")):
        raise CameraError("the world's Mapping node has linked inputs", "give the world Mapping constant values")
    loc = np.asarray(vals.get("Location", {}).get("value", [0, 0, 0]), np.float64)
    scale = np.asarray(vals.get("Scale", {}).get("value", [1, 1, 1]), np.float64)
    kind = m.get("props", {}).get("vector_type", "POINT")
    if np.abs(loc).max() > 1e-9 and kind == "POINT" or not np.allclose(scale, scale[0]) or scale[0] <= 0:
        raise CameraError(f"world Mapping location {loc.tolist()} / scale {scale.tolist()}",
                          "use only rotation (and a uniform positive scale) on the world's Mapping node")
    rot = _euler_xyz(*np.asarray(vals.get("Rotation", {}).get("value", [0, 0, 0]), np.float64))
    return rot.T if kind == "TEXTURE" else rot


def camera_background(tree: dict) -> np.ndarray | None:
    """Linear RGB that camera rays see when the world output is Mix Shader(Fac = Light Path Is Camera Ray, A, B)
    with B a Background of constant colour; None for a plain world."""
    nodes = {n["name"]: n for n in tree.get("nodes", [])}
    out = next((n for n in tree.get("nodes", []) if n["type"] == "ShaderNodeOutputWorld"), None)
    link = _link_into(tree, out["name"], "Surface") if out else None
    if link is None or nodes[link["from"]]["type"] != "ShaderNodeMixShader":
        return None
    mix = link["from"]
    fac = _link_into(tree, mix, "Fac")
    if fac is None or nodes[fac["from"]]["type"] != "ShaderNodeLightPath" or fac["from_id"] != "Is Camera Ray":
        raise CameraError("the world mixes shaders by something other than Light Path Is Camera Ray",
                          "use a plain Background, or mix by Is Camera Ray only")
    cam = _link_into(tree, mix, "Shader_001")
    bg = nodes.get(cam["from"]) if cam else None
    if bg is None or bg["type"] != "ShaderNodeBackground":
        raise CameraError("the world's camera-ray shader is not a Background node", "use a Background node there")
    vals = {i["name"]: i for i in bg.get("inputs", [])}
    if vals.get("Color", {}).get("linked") or vals.get("Strength", {}).get("linked"):
        raise CameraError("the world's camera-ray Background is linked", "give it a constant Color and Strength")
    color = np.asarray(vals["Color"]["value"][:3], np.float64)
    return color * float(vals.get("Strength", {}).get("value", 1.0))


def lighting_background(tree: dict) -> dict | None:
    """The Background node non-camera rays see (the Mix Shader's first input), or None without the camera mix."""
    nodes = {n["name"]: n for n in tree.get("nodes", [])}
    out = next((n for n in tree.get("nodes", []) if n["type"] == "ShaderNodeOutputWorld"), None)
    link = _link_into(tree, out["name"], "Surface") if out else None
    if link is None or nodes[link["from"]]["type"] != "ShaderNodeMixShader":
        return None
    first = _link_into(tree, link["from"], "Shader")
    return nodes.get(first["from"]) if first else None
