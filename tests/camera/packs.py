# SPDX-License-Identifier: Apache-2.0
"""Hand-made scene packs for tests (no Blender): a v2 pack with one Principled material, an area light and a
uniform HDRI world, and a v1 pack with two coloured meshes."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def write_hdr(path: Path, rgb: np.ndarray) -> None:
    """Uncompressed Radiance RGBE, rows top first."""
    h, w, _ = rgb.shape
    m = rgb.max(axis=2)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, -128).astype(np.int32)
    scale = np.where(m > 1e-32, 256.0 / np.exp2(e), 0.0)
    px = np.zeros((h, w, 4), np.uint8)
    px[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    px[..., 3] = np.clip(e + 128, 0, 255)
    head = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode()
    path.write_bytes(head + px.tobytes())


def _socket(sid: str, name: str, value, linked: bool = False, kind: str = "NodeSocketFloat") -> dict:
    return {"id": sid, "name": name, "type": kind, "linked": linked, "enabled": True, "value": value}


def principled_material(base=(0.5, 0.5, 0.5), roughness=1.0, specular=0.0, metallic=0.0,
                        noise_roughness: bool = False) -> dict:
    """A Principled BSDF material as a pack v2 node graph (optionally roughness from a Noise Texture)."""
    vals = {"Base Color": list(base), "Metallic": metallic, "Roughness": roughness, "IOR": 1.5,
            "Specular IOR Level": specular, "Coat Weight": 0.0, "Sheen Weight": 0.0, "Transmission Weight": 0.0,
            "Subsurface Weight": 0.0, "Emission Strength": 0.0}
    linked = {"Roughness"} if noise_roughness else set()
    inputs = [_socket(k, k, v, k in linked, "NodeSocketColor" if k == "Base Color" else "NodeSocketFloat")
              for k, v in vals.items()]
    nodes = [{"name": "Principled BSDF", "type": "ShaderNodeBsdfPrincipled", "props": {"distribution": "MULTI_GGX"},
              "inputs": inputs}]
    links = []
    if noise_roughness:
        nodes.append({"name": "Coord", "type": "ShaderNodeTexCoord", "props": {}, "inputs": []})
        nodes.append({"name": "Noise", "type": "ShaderNodeTexNoise",
                      "props": {"noise_dimensions": "3D", "noise_type": "FBM", "normalize": True},
                      "inputs": [_socket("Vector", "Vector", [0, 0, 0], True, "NodeSocketVector"),
                                 _socket("Scale", "Scale", 5.0), _socket("Detail", "Detail", 2.0),
                                 _socket("Roughness", "Roughness", 0.5), _socket("Lacunarity", "Lacunarity", 2.0),
                                 _socket("Distortion", "Distortion", 0.0)]})
        links = [{"from": "Coord", "from_id": "Object", "to": "Noise", "to_id": "Vector"},
                 {"from": "Noise", "from_id": "Fac", "to": "Principled BSDF", "to_id": "Roughness"}]
    return {"tree": {"nodes": nodes, "links": links},
            "principled": {"distribution": "MULTI_GGX",
                           "inputs": {k: {"value": v, "linked": k in linked} for k, v in vals.items()}}}


def _world(strength: float) -> dict:
    nodes = [{"name": "Env", "type": "ShaderNodeTexEnvironment", "image": "sky", "props": {}, "inputs": []},
             {"name": "Background", "type": "ShaderNodeBackground", "props": {},
              "inputs": [_socket("Color", "Color", [1, 1, 1, 1], True, "NodeSocketColor"),
                         _socket("Strength", "Strength", strength)]}]
    return {"tree": {"nodes": nodes, "links": [{"from": "Env", "from_id": "Color", "to": "Background",
                                                "to_id": "Color"}]}, "cycles": {"sampling_method": "AUTOMATIC"}}


def _settings() -> dict:
    cyc = {"max_bounces": 12, "diffuse_bounces": 4, "glossy_bounces": 4, "transmission_bounces": 12,
           "sample_clamp_direct": 0.0, "sample_clamp_indirect": 10.0, "seed": 0, "blur_glossy": 1.0,
           "caustics_reflective": True}
    return {"cycles": cyc, "render": {}, "view_settings": {"view_transform": "Standard"}}


def _camera(eye, target) -> dict:
    from alloy.camera.pose import look_at_matrix
    m = np.eye(4)
    m[:3, :3] = look_at_matrix(eye, target, (0.0, 1.0, 0.0))
    m[:3, 3] = eye
    return {"matrix_world": m.tolist(), "props": {"type": "PERSP", "lens": 50.0, "sensor_width": 36.0,
                                                  "sensor_fit": "AUTO", "clip_start": 0.01, "clip_end": 100.0},
            "dof": {"use_dof": False}}


def write_pack_v2(root: Path, material: dict, sky: float = 0.5, light_energy: float = 0.0, half: float = 10.0,
                  uv: np.ndarray | None = None, eye=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0),
                  textures: dict | None = None, attrs: dict | None = None) -> Path:
    """A plane (side ``2 * half``, default 20 m) at z = 0 with ``material``, a uniform sky of radiance ``sky`` and a
    camera looking down. ``uv``: optional per-corner UVs (2, 3, 2) of the two triangles; ``textures``: extra
    pack textures {key: {file, colorspace}} (the files are written by the caller under ``root``); ``attrs``: per-corner
    attributes {name: (2, 3, k)}."""
    (root / "frames/0/meshes").mkdir(parents=True, exist_ok=True)
    (root / "textures").mkdir(exist_ok=True)
    write_hdr(root / "textures/sky.hdr", np.full((8, 16, 3), sky, np.float32))
    v = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], np.float32)
    tri = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    extra = {} if uv is None else {"uv:UVMap": np.asarray(uv, np.float32)}
    extra.update({f"attr:{k}": np.asarray(a, np.float32) for k, a in (attrs or {}).items()})
    np.savez(root / "frames/0/meshes/0000.npz", v=v, tri=tri, n=np.tile([0, 0, 1.0], (2, 3, 1)).astype(np.float32),
             mat=np.zeros(2, np.int32), **extra)
    mesh = {"name": "plane", "file": "frames/0/meshes/0000.npz", "materials": ["mat0"],
            "object_to_world": np.eye(4).tolist(), "object_name": "plane"}
    lights = []
    if light_energy > 0:
        m = np.eye(4)
        m[:3, 3] = (0.0, 0.0, 2.0)
        lights.append({"name": "key", "type": "AREA", "matrix_world": m.tolist(),
                       "props": {"shape": "SQUARE", "size": 0.5, "color": [1, 1, 1], "energy": light_energy}})
    frame = {"camera": _camera(eye, target), "lights": lights, "meshes": [mesh]}
    (root / "frames/0/frame.json").write_text(json.dumps(frame))
    static = {"pack_version": 2, "frames": [0], "settings": _settings(), "world": _world(1.0),
              "materials": {"mat0": material}, "materials_used_by_meshes": ["mat0"],
              "textures": {"sky": {"file": "textures/sky.hdr", "colorspace": "Linear Rec.709"}, **(textures or {})}}
    (root / "pack.json").write_text(json.dumps(static))
    return root


def _ply(path: Path, v: np.ndarray, tri: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(tri)}\nproperty list uchar int vertex_indices\nend_header\n")
    faces = b"".join(struct.pack("<B3i", 3, *t) for t in tri)
    path.write_bytes(head.encode() + v.astype("<f4").tobytes() + faces)


def write_pack_v1(root: Path) -> Path:
    """Two quads (a grey floor and a red wall), a sun, a sky and a camera, as a v1 pack."""
    (root / "meshes").mkdir(parents=True, exist_ok=True)
    floor = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], np.float32)
    wall = np.array([[-1, 1, 0], [1, 1, 0], [1, 1, 1], [-1, 1, 1]], np.float32)
    tri = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    _ply(root / "meshes/floor.ply", floor, tri)
    _ply(root / "meshes/wall.ply", wall, tri[:, ::-1])
    from alloy.camera.pose import look_at_matrix
    doc = {"meshes": [{"name": "floor", "file": "meshes/floor.ply", "color": [0.5, 0.5, 0.5]},
                      {"name": "wall", "file": "meshes/wall.ply", "color": [0.8, 0.1, 0.1]}],
           "camera": {"eye": [0.0, -3.0, 1.5], "rot": look_at_matrix((0, -3, 1.5), (0, 0.5, 0.3)).tolist(),
                      "fov": 0.9},
           "light": {"sun_euler_xyz": [0.6, 0.3, 0.8], "sun_energy": 3.0, "sky_rgb": [0.6, 0.65, 0.7]}}
    (root / "scene.json").write_text(json.dumps(doc))
    return root


UV_SOURCES = ("texcoord", "uvmap", "attribute", "unlinked")


def textured_material(source: str, image: str = "checker") -> dict:
    """A rough, non-specular Principled material whose Base Color is an Image Texture of pack texture ``image``; the
    texture coordinate comes from ``source``: a Texture Coordinate UV output, a UV Map node, an Attribute node named
    UVMap, or nothing (an unlinked Vector input, which reads the UV)."""
    mat = principled_material(roughness=1.0)
    mat["tree"]["nodes"][0]["inputs"][0]["linked"] = True
    mat["principled"]["inputs"]["Base Color"]["linked"] = True
    tex = {"name": "Tex", "type": "ShaderNodeTexImage", "image": image,
           "props": {"projection": "FLAT", "interpolation": "Cubic", "extension": "REPEAT"},
           "inputs": [_socket("Vector", "Vector", [0, 0, 0], source != "unlinked", "NodeSocketVector")]}
    links = [{"from": "Tex", "from_id": "Color", "to": "Principled BSDF", "to_id": "Base Color"}]
    src = {"texcoord": ({"name": "Src", "type": "ShaderNodeTexCoord", "props": {}, "inputs": []}, "UV"),
           "uvmap": ({"name": "Src", "type": "ShaderNodeUVMap", "props": {"uv_map": "UVMap"}, "inputs": []}, "UV"),
           "attribute": ({"name": "Src", "type": "ShaderNodeAttribute",
                          "props": {"attribute_name": "UVMap", "attribute_type": "GEOMETRY"}, "inputs": []}, "Vector"),
           "unlinked": (None, None)}[source]
    mat["tree"]["nodes"].append(tex)
    if src[0] is not None:
        mat["tree"]["nodes"].append(src[0])
        links.append({"from": "Src", "from_id": src[1], "to": "Tex", "to_id": "Vector"})
    mat["tree"]["links"] = links
    return mat


def checker_image(block: int = 8) -> np.ndarray:
    """A 2 x 2 checker of ``block``-pixel squares as stored in an image file (row 0 = top): red, green / blue, white."""
    quad = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    return np.kron(quad, np.ones((block, block, 1), np.uint8))


def write_textured_pack(root: Path, source: str, sky: float = 1.0) -> Path:
    """A unit quad [0, 1]^2 at z = 0 textured with ``checker_image`` through ``source`` (see ``textured_material``),
    UV (0, 0) at its bottom-left corner and (1, 1) at the top-right, seen from above (+y up) under a uniform sky."""
    from PIL import Image
    (root / "textures").mkdir(parents=True, exist_ok=True)
    Image.fromarray(checker_image()).save(root / "textures/checker.png")
    corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    uv = corner[np.array([[0, 1, 2], [0, 2, 3]])]
    write_pack_v2(root, textured_material(source), sky=sky, half=0.5, uv=uv, eye=(0.0, 0.0, 2.0),
                  textures={"checker": {"file": "textures/checker.png", "colorspace": "sRGB"}})
    return root


def attribute_colour_material(name: str = "Col") -> dict:
    """A rough Principled material whose Base Color is the Attribute node ``name`` (Color output)."""
    mat = principled_material(roughness=1.0)
    mat["tree"]["nodes"][0]["inputs"][0]["linked"] = True
    mat["principled"]["inputs"]["Base Color"]["linked"] = True
    mat["tree"]["nodes"].append({"name": "Src", "type": "ShaderNodeAttribute", "inputs": [],
                                 "props": {"attribute_name": name, "attribute_type": "GEOMETRY"}})
    mat["tree"]["links"] = [{"from": "Src", "from_id": "Color", "to": "Principled BSDF", "to_id": "Base Color"}]
    return mat
