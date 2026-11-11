# SPDX-License-Identifier: Apache-2.0
"""Write a sim frame as a scene pack v2 on disk (``PACK_V2.md``): the full-material scene the camera renders (model
shapes, deformables at their current shape, look materials, lights and world), placed in world space for one world,
so that Blender's pack importer renders the same scene in Cycles and ``load_pack`` reads it back.

The pack's settings are the model scene's (Blender defaults otherwise: Blackman-Harris 1.5 px filter, AgX view, no
white balance); lights are invisible to camera rays, as the camera draws them; the world image is written as a
Radiance HDR behind an Environment Texture.

Example::

    from alloy.camera.pack_write import write_sim_pack
    write_sim_pack(sim, "towel_pack", look_at=((0.5, -0.5, 0.5), (0, 0, 0.2)), look="towel.look.json")
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from .scene import quat_to_matrix
from .smooth import corner_normals

log = logging.getLogger(__name__)
VIS_ALL = {"visible_camera": True, "visible_diffuse": True, "visible_glossy": True, "visible_transmission": True,
           "visible_volume_scatter": True, "visible_shadow": True, "is_holdout": False, "is_shadow_catcher": False}


def write_hdr(path: Path, rgb: np.ndarray) -> None:
    """Uncompressed Radiance RGBE, rows top first (what Blender's and the camera's HDR readers take)."""
    rgb = np.asarray(rgb, np.float32)[..., :3]
    h, w, _ = rgb.shape
    m = rgb.max(axis=2)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, -128).astype(np.int32)
    scale = np.where(m > 1e-32, 256.0 / np.exp2(e), 0.0)
    px = np.zeros((h, w, 4), np.uint8)
    px[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    px[..., 3] = np.clip(e + 128, 0, 255)
    path.write_bytes(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode() + px.tobytes())


def _socket(sid: str, value, linked: bool = False, kind: str = "NodeSocketFloat") -> dict:
    return {"id": sid, "name": sid, "type": kind, "linked": linked, "enabled": True, "value": value}


def _world(world: dict) -> dict:
    """Environment Texture (key "world_sky") -> Background -> World Output."""
    nodes = [{"name": "World Output", "type": "ShaderNodeOutputWorld", "props": {"is_active_output": True},
              "inputs": [_socket("Surface", None, True, "NodeSocketShader")]},
             {"name": "Environment Texture", "type": "ShaderNodeTexEnvironment", "image": "world_sky",
              "props": {"projection": "EQUIRECTANGULAR", "interpolation": "Linear"}, "inputs": []},
             {"name": "Background", "type": "ShaderNodeBackground", "props": {},
              "inputs": [_socket("Color", [1.0, 1.0, 1.0, 1.0], True, "NodeSocketColor"),
                         _socket("Strength", float(world.get("strength", 1.0)))]}]
    links = [{"from": "Environment Texture", "from_id": "Color", "to": "Background", "to_id": "Color"},
             {"from": "Background", "from_id": "Background", "to": "World Output", "to_id": "Surface"}]
    return {"tree": {"nodes": nodes, "links": links}, "light_settings": {},
            "cycles": {"sampling_method": "AUTOMATIC" if world.get("use_mis", True) else "NONE"}}


def _settings(settings: dict) -> dict:
    view = {"gamma": 1.0, "use_curve_mapping": False, "use_white_balance": False, **settings["view_settings"]}
    cyc = {"pixel_filter_type": "BLACKMAN_HARRIS", "filter_width": 1.5, **settings["cycles"]}
    return {"render": {}, "cycles": cyc, "view_settings": view, "display_settings": {"display_device": "sRGB"},
            "sequencer_colorspace": {"name": "sRGB"}, "view_layer": {}, "fps": 24.0}


def _camera(eye, target, lens: float = 50.0) -> dict:
    from .pose import look_at_matrix
    m = np.eye(4)
    m[:3, :3] = look_at_matrix(np.asarray(eye, np.float64), np.asarray(target, np.float64))
    m[:3, 3] = eye
    return {"name": "camera", "matrix_world": m.tolist(), "pixel_aspect": [1.0, 1.0],
            "props": {"type": "PERSP", "lens": lens, "sensor_width": 36.0, "sensor_fit": "AUTO", "clip_start": 0.01,
                      "clip_end": 100.0, "shift_x": 0.0, "shift_y": 0.0}, "dof": {"use_dof": False}}


def _xf(q7) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(np.asarray(q7, np.float64)[3:7])
    m[:3, 3] = np.asarray(q7, np.float64)[:3]
    return m


def _placed(mesh, xf: np.ndarray):
    """(v, n, object_to_world) of a pack mesh moved by the 4x4 ``xf`` (body pose or fixed copy transform)."""
    v = (np.c_[mesh.v.astype(np.float64), np.ones(len(mesh.v))] @ xf.T)[:, :3].astype(np.float32)
    n = (mesh.n.reshape(-1, 3).astype(np.float64) @ xf[:3, :3].T).reshape(mesh.n.shape).astype(np.float32)
    o2w = xf @ np.asarray(mesh.meta["object_to_world"], np.float64).reshape(4, 4)
    return v, n, o2w


def _liquid_corners(v: np.ndarray, tri: np.ndarray, vn: np.ndarray) -> np.ndarray:
    """(T, 3, 3) corner normals of a liquid surface as the camera draws them (``liquid_kernels.corner_normals``)."""
    n = vn[tri].astype(np.float64)
    p = v[tri].astype(np.float64)
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, None, :]
    small = np.linalg.norm(n, axis=-1, keepdims=True) <= 1e-30
    n = np.where(small, np.broadcast_to(fn, n.shape), n)
    n = np.where(np.linalg.norm(n, axis=-1, keepdims=True) <= 1e-30, [0.0, 0.0, 1.0], n)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def _frame_meshes(frame, binding, state, deform_points: dict, world: int) -> list[tuple]:
    """[(meta, v, tri, n, mat)] of world ``world``: body meshes at ``state.body_q``, deformables at their points,
    liquids as their surfaces (``deform_points[name]`` = (verts, tris, vertex normals); none when empty)."""
    body, local, present = binding
    bq = state.body_q.numpy() if state is not None and getattr(state, "body_q", None) is not None else None
    out = []
    for k, m in enumerate(frame.meshes):
        if not present[k, world]:
            continue
        if m.meta.get("kind") == "liquid":
            v, tri, vn = deform_points[m.meta["object_name"]]
            if len(tri):
                out.append(({**m.meta, "object_to_world": np.eye(4).tolist(), "dynamic": False}, v, tri,
                            _liquid_corners(v, tri, vn), np.zeros(len(tri), np.int32)))
            continue
        if m.meta.get("dynamic"):
            v = np.asarray(deform_points[m.meta["object_name"]], np.float32)
            if len(v) != len(m.v):
                raise ValueError(f"{m.meta['name']}: {len(v)} points for {len(m.v)} vertices")
            n = m.n if m.meta.get("kind") == "grains" else corner_normals(v, m.tri, 180.0)  # grains: fixed normals
            out.append(({**m.meta, "object_to_world": np.eye(4).tolist()}, v, m.tri, n, m.mat))
            continue
        b = int(body[k, world])
        xf = _xf(bq[b]) if b >= 0 and bq is not None else _xf(local[k, world])
        v, n, o2w = _placed(m, xf)
        out.append(({**m.meta, "object_to_world": o2w.tolist()}, v, m.tri, n, m.mat))
    return out


def write_pack(root, pack, frame, binding, state=None, deform_points: dict | None = None, camera: dict | None = None,
               world: int = 0) -> Path:
    """Write ``(pack, frame, binding)`` (``model_pack.pack_from_model``) at ``state`` to ``root``; ``deform_points``
    maps each deformable's name to its drawn vertices of ``world``. Returns ``root``."""
    root = Path(root)
    (root / "frames/0/meshes").mkdir(parents=True, exist_ok=True)
    (root / "textures").mkdir(exist_ok=True)
    st = pack.static
    textures = {}
    for key, t in st.get("textures", {}).items():
        src = Path(t["file"])
        dst = root / "textures" / f"{len(textures):02d}_{src.name}"
        shutil.copyfile(src if src.is_absolute() else pack.root / src, dst)
        textures[key] = {**t, "file": f"textures/{dst.name}", "alpha_mode": t.get("alpha_mode", "STRAIGHT")}
    write_hdr(root / "textures/world_sky.hdr", np.asarray(st["world"]["image"], np.float32))
    textures["world_sky"] = {"file": "textures/world_sky.hdr", "colorspace": "Linear Rec.709", "alpha_mode": "STRAIGHT"}
    entries = []
    for i, (meta, v, tri, n, mat) in enumerate(_frame_meshes(frame, binding, state, deform_points or {}, world)):
        f = f"frames/0/meshes/{i:04d}.npz"
        np.savez(root / f, v=v, tri=np.asarray(tri, np.int32), n=np.asarray(n, np.float32), mat=np.asarray(mat, np.int32))
        entries.append({"name": meta["name"], "object_name": meta.get("object_name", meta["name"]), "file": f,
                        "tris": int(len(tri)), "verts": int(len(v)), "object_to_world": meta["object_to_world"],
                        "mirrored": False, "materials": meta["materials"], "visibility": VIS_ALL,
                        "is_instance": False, "modifiers": [], "uv_layers": []})
    lights = [{**lt, "cycles": lt.get("cycles", {}), "visibility": {**VIS_ALL, "visible_camera": False}}
              for lt in frame.lights]
    cam = camera or {**frame.camera, "name": "camera", "pixel_aspect": [1.0, 1.0]}
    (root / "frames/0/frame.json").write_text(json.dumps({"camera": cam, "lights": lights, "meshes": entries}))
    mats = {k: {"props": {}, "cycles": {}, **v} for k, v in st["materials"].items()}
    static = {"pack_version": 2, "frames": [0], "settings": _settings(st["settings"]), "world": _world(st["world"]),
              "materials": mats, "materials_used_by_meshes": sorted({e["materials"][0] for e in entries}),
              "textures": textures, "unsupported": []}
    (root / "pack.json").write_text(json.dumps(static))
    log.info("pack: %d meshes, %d materials, %d textures -> %s", len(entries), len(mats), len(textures), root)
    return root


def write_sim_pack(sim, root, look_at, look=None, world: int = 0, lens: float = 50.0) -> Path:
    """Write world ``world`` of an ``alloy.Sim`` at its current state as a pack v2 with a camera at
    ``look_at = (eye, target)``. Example: ``write_sim_pack(sim, "pack", ((0.5, -0.5, 0.5), (0, 0, 0.2)))``."""
    from .deform import deform_objects, grain_mesh
    from .item_looks import items_look
    from .liquid_scene import liquid_objects, liquid_surfaces
    from .model_pack import pack_from_model
    layout, liquid = deform_objects(sim), liquid_objects(sim)
    model, state = sim.engine.model, sim.engine.state
    pk, frame, binding = pack_from_model(model, look, None, items_look(sim), deform=layout, liquid=liquid)
    pts = liquid_surfaces(sim, liquid, world) if liquid is not None else {}
    for o in (layout.objects if layout else ()):
        p = sim.state([o.name], world=world)[f"{o.name}.points"]
        if o.kind == "grains":
            p = grain_mesh(p, model.particle_radius.numpy()[o.start:o.start + o.count])[0]
        pts[o.name] = p
    return write_pack(root, pk, frame, binding, state, pts, _camera(*look_at, lens=lens), world)
