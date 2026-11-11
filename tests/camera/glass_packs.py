# SPDX-License-Identifier: Apache-2.0
"""Hand-made scene packs with see-through materials (no Blender): a glass box under a uniform sky.

A closed box of Principled glass (transmission 1) with an optional Volume Absorption node; the camera looks straight
down at the box's top face. In a uniform sky a lossless glass object is invisible (every path ends in the sky), and a
smooth absorbing slab seen at normal incidence has the closed-form brightness ``slab_brightness``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.camera.packs import _camera, _settings, _socket, _world, principled_material, write_hdr

BOX_TRIS = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                     [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)


def box_mesh(half=(1.0, 1.0, 0.05)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices (8, 3), triangles (12, 3) wound outward, flat corner normals (12, 3, 3)) of a box at the origin."""
    hx, hy, hz = half
    v = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]], np.float32)
    p = v[BOX_TRIS]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return v, BOX_TRIS, np.repeat(n[:, None, :], 3, axis=1).astype(np.float32)


def glass_material(roughness: float = 0.0, ior: float = 1.5, absorb_color=None, density: float = 0.0,
                   transmission: float = 1.0) -> dict:
    """Principled glass (white base; ``transmission`` 1: all glass, below 1: glass over a white diffuse layer) with
    an optional Volume Absorption on the Volume output: sigma_a = (1 - absorb_color) * density per metre."""
    mat = principled_material(base=(1.0, 1.0, 1.0, 1.0), roughness=roughness, specular=0.5)
    mat["principled"]["inputs"]["Transmission Weight"]["value"] = transmission
    mat["principled"]["inputs"]["IOR"]["value"] = ior
    for sock in mat["tree"]["nodes"][0]["inputs"]:
        if sock["name"] == "Transmission Weight":
            sock["value"] = transmission
        if sock["name"] == "IOR":
            sock["value"] = ior
    out = {"name": "Material Output", "type": "ShaderNodeOutputMaterial", "props": {"is_active_output": True},
           "inputs": [_socket("Surface", "Surface", None, True, "NodeSocketShader"),
                      _socket("Volume", "Volume", None, absorb_color is not None, "NodeSocketShader")]}
    mat["tree"]["nodes"].append(out)
    mat["tree"]["links"].append({"from": "Principled BSDF", "from_id": "BSDF", "to": "Material Output",
                                 "to_id": "Surface"})
    if absorb_color is not None:
        mat["tree"]["nodes"].append({"name": "Absorb", "type": "ShaderNodeVolumeAbsorption", "props": {},
                                     "inputs": [_socket("Color", "Color", list(absorb_color) + [1.0], False,
                                                        "NodeSocketColor"),
                                                _socket("Density", "Density", density)]})
        mat["tree"]["links"].append({"from": "Absorb", "from_id": "Volume", "to": "Material Output",
                                     "to_id": "Volume"})
    return mat


def write_glass_pack(root: Path, material: dict, half=(1.0, 1.0, 0.05), sky: float = 0.5, eye_z: float = 0.6,
                     cycles: dict | None = None, world_sampling: str = "AUTOMATIC") -> Path:
    """A glass box (``half`` extents, centred at the origin) in a uniform sky, the camera ``eye_z`` above it looking
    down; ``transmission_bounces`` is raised to 32 so long internal paths are not cut. ``cycles`` updates the render
    settings; ``world_sampling`` is the world's sampling method. Blender's render_pack_cycles.py renders it too."""
    (root / "frames/0/meshes").mkdir(parents=True, exist_ok=True)
    (root / "textures").mkdir(exist_ok=True)
    write_hdr(root / "textures/sky.hdr", np.full((8, 16, 3), sky, np.float32))
    v, tri, n = box_mesh(half)
    np.savez(root / "frames/0/meshes/0000.npz", v=v, tri=tri, n=n, mat=np.zeros(len(tri), np.int32))
    mesh = {"name": "box", "file": "frames/0/meshes/0000.npz", "materials": ["glass"],
            "object_to_world": np.eye(4).tolist(), "object_name": "box"}
    frame = {"camera": _camera((0.0, 0.0, eye_z), (0.0, 0.0, 0.0)), "lights": [], "meshes": [mesh]}
    (root / "frames/0/frame.json").write_text(json.dumps(frame))
    settings = _settings()
    settings["cycles"].update({"max_bounces": 64, "transmission_bounces": 32, "glossy_bounces": 32, **(cycles or {})})
    world = _world(1.0)
    world["cycles"]["sampling_method"] = world_sampling
    world["tree"]["nodes"].append({"name": "World Output", "type": "ShaderNodeOutputWorld",
                                   "props": {"is_active_output": True}, "inputs": []})
    world["tree"]["links"].append({"from": "Background", "from_id": "Background", "to": "World Output",
                                   "to_id": "Surface"})
    static = {"pack_version": 2, "frames": [0], "settings": settings, "world": world,
              "materials": {"glass": material}, "materials_used_by_meshes": ["glass"],
              "textures": {"sky": {"file": "textures/sky.hdr", "colorspace": "Linear Rec.709"}}}
    (root / "pack.json").write_text(json.dumps(static))
    return root


def slab_brightness(ior: float, sigma: float, thickness: float) -> float:
    """Reflectance + transmittance of a smooth absorbing slab at normal incidence (all internal reflections): the
    brightness of the slab in a uniform sky of radiance 1."""
    r = ((ior - 1.0) / (ior + 1.0)) ** 2
    a = np.exp(-sigma * thickness)
    t_total = (1.0 - r) ** 2 * a / (1.0 - r * r * a * a)
    r_total = r + (1.0 - r) ** 2 * r * a * a / (1.0 - r * r * a * a)
    return float(t_total + r_total)


def tie_hits(tracer_kind: str, gap: float, device: str = "cpu", base: float = 0.0, n_side: int = 5) -> np.ndarray:
    """Closest-hit instance of ``n_side``^2 slanted rays shot down at a glass sheet (instance 1, marked transmissive)
    lying ``gap`` metres above a floor (instance 0, at height ``base``) of the same size: at gap 0 the two are exactly
    coplanar. ``tracer_kind``: "wavefront-warp" or "optix"."""
    import warp as wp

    from alloy.camera.scene import SceneData
    from alloy.camera.tracer.gpu_scene import GpuScene
    from alloy.camera.wavefront.tracers import make_tracer
    from tests.camera.helpers import quad
    floor, glass = quad(2.0, base), quad(2.0, float(np.float32(base) + np.float32(gap)))
    sc = GpuScene(SceneData.single_world([floor, glass], [(0.8,) * 3] * 2), device=device)
    sc.set_transmissive([False, True])
    xy = np.stack(np.meshgrid(np.linspace(-0.37, 0.41, n_side), np.linspace(-0.43, 0.39, n_side)), -1).reshape(-1, 2)
    o = np.concatenate([xy, np.full((len(xy), 1), 0.7)], 1).astype(np.float32)
    d = np.array([0.31, -0.17, -1.0], np.float32) / np.linalg.norm([0.31, -0.17, -1.0])
    n = len(o)
    arr = lambda a, t=wp.vec3: wp.array(a, dtype=t, device=device)  # noqa: E731
    hits = [wp.zeros(n, dtype=float, device=device) for _ in range(3)] + [wp.zeros(n, dtype=wp.int32, device=device)
                                                                         for _ in range(2)]
    make_tracer(sc, tracer_kind).trace_closest(n, arr(o), arr(np.tile(d, (n, 1))), arr(np.full(n, 1e30), float),
                                               arr(np.zeros(n, np.int32), wp.int32), *hits)
    return hits[4].numpy()


def sss_material(weight: float, scale: float, base=(0.8, 0.8, 0.8)) -> dict:
    """A Principled material with random-walk subsurface of ``weight`` and ``scale`` (radius 1, 1, 1)."""
    m = principled_material(base=base, roughness=0.5, specular=0.5)
    for k, v in (("Subsurface Weight", weight), ("Subsurface Scale", scale), ("Subsurface Radius", [1.0, 1.0, 1.0]),
                 ("Subsurface Anisotropy", 0.0)):
        m["principled"]["inputs"][k] = {"value": v, "linked": False}
    return m
