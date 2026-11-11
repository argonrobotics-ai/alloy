# SPDX-License-Identifier: Apache-2.0
"""Scene pack v2 frame -> device shading data: corner normals, per-triangle materials, object-space transforms
(Texture Coordinate Object), Object Info Random, the light scene (area lights + HDRI world) and textures.

Example::

    from alloy.camera.pbr.pack_io import load_pack
    fr = load_pack("c3_pack").frame(420)
    sd = build_shade_data(fr, names, gpu_scene)         # names = pack.static["materials_used_by_meshes"]
"""
from __future__ import annotations

import logging
import math

import numpy as np
import warp as wp

from ..errors import CameraError

log = logging.getLogger(__name__)
_KEEP: list = []  # wp.Mesh objects whose ids live in ShadeData structs


@wp.struct
class ShadeData:
    tri_n: wp.array(dtype=wp.vec3)  # corner shading normals, world space of the frame, (sum T) * 3
    tri_mat: wp.array(dtype=wp.int32)  # global material index per triangle
    mesh_tri_off: wp.array(dtype=wp.int32)  # first global triangle of each mesh
    mesh_obj_inv: wp.array(dtype=wp.mat44)  # frame world -> object space (Texture Coordinate Object)
    mesh_n_obj: wp.array(dtype=wp.mat33)  # world normal -> object normal (M3^T, normalise after)
    mesh_rnd: wp.array(dtype=float)  # Object Info Random
    mesh_texloc: wp.array(dtype=wp.vec3)  # auto texture space centre (object-space bbox centre)
    mesh_texinv: wp.array(dtype=wp.vec3)  # 0.5 / texture space half size (Generated = (co - loc) * inv + 0.5)
    mesh_n_world: wp.array(dtype=wp.mat33)  # object normal -> world: transpose(inverse(M3))
    inst_src: wp.array(dtype=wp.int32)  # instance -> mesh
    bevel_mesh: wp.uint64  # all objects of the frame in one wp.Mesh (global triangle order) for the Bevel node
    tri_obj: wp.array(dtype=int)  # global triangle -> mesh index (Bevel stays on its own object)
    uv: wp.array(dtype=wp.vec2)  # corner UVs of the meshes that have them, 3 per triangle (corner_attrs.py)
    uv_off: wp.array(dtype=wp.int32)  # per mesh: first triangle in uv, -1 without UVs
    col: wp.array(dtype=wp.vec3)  # corner colours of the colour attribute, same layout
    col_off: wp.array(dtype=wp.int32)


def _mesh_frames(m) -> tuple:
    """(obj_inv, n_obj, n_world, texloc, texinv) of one pack mesh (Blender auto texture space)."""
    o2w = np.asarray(m.meta["object_to_world"], np.float64).reshape(4, 4)
    inv = np.linalg.inv(o2w)
    vo = (np.c_[np.asarray(m.v, np.float64), np.ones(len(m.v))] @ inv.T)[:, :3]
    lo, hi = vo.min(0), vo.max(0)
    half = 0.5 * (hi - lo)  # BKE_mesh_texspace_calc: 0 -> 1, tiny -> 1e-5
    half = np.where(half == 0.0, 1.0, np.maximum(half, 1e-5))
    return (inv.astype(np.float32), o2w[:3, :3].T.astype(np.float32), np.linalg.inv(o2w[:3, :3]).T.astype(np.float32),
            (0.5 * (lo + hi)).astype(np.float32), (0.5 / half).astype(np.float32))


def build_shade_data(frame, mat_names: list[str], sc, col_name: str | None = None) -> ShadeData:
    """Per-mesh / per-triangle shading arrays for the frame's meshes (mesh k = SceneData mesh k); ``col_name``: the
    colour attribute the materials read (corner_attrs.select_attributes)."""
    from .corner_attrs import fill_corner_attrs
    from . import nodes as nd
    dev = sc.device
    tri_n, tri_mat, off, frames, rnd = [], [], [], [], []
    k = 0
    default_mid = len(mat_names)
    for m in frame.meshes:
        off.append(k)
        k += len(m.tri)
        tri_n.append(np.asarray(m.n, np.float32).reshape(-1, 3))
        slots = [default_mid if s is None else mat_names.index(s) for s in m.meta["materials"]]
        tri_mat.append(np.asarray(slots, np.int32)[np.asarray(m.mat, np.int64)])
        frames.append(_mesh_frames(m))
        rnd.append(float(nd.object_random(m.meta.get("object_name", m.meta["name"]))))
    inv, nobj, nwor, tloc, tinv = (np.stack(x) for x in zip(*frames))
    sd = ShadeData()
    sd.tri_n = wp.array(np.concatenate(tri_n), dtype=wp.vec3, device=dev)
    sd.tri_mat = wp.array(np.concatenate(tri_mat), dtype=wp.int32, device=dev)
    sd.mesh_tri_off = wp.array(np.asarray(off, np.int32), dtype=wp.int32, device=dev)
    sd.mesh_obj_inv = wp.array(inv, dtype=wp.mat44, device=dev)
    sd.mesh_n_obj = wp.array(nobj, dtype=wp.mat33, device=dev)
    sd.mesh_rnd = wp.array(np.asarray(rnd, np.float32), dtype=float, device=dev)
    sd.mesh_texloc = wp.array(tloc, dtype=wp.vec3, device=dev)
    sd.mesh_texinv = wp.array(tinv, dtype=wp.vec3, device=dev)
    sd.mesh_n_world = wp.array(nwor, dtype=wp.mat33, device=dev)
    sd.inst_src = sc.inst_src
    fill_corner_attrs(sd, frame, col_name, dev)
    _bevel_mesh(sd, frame, dev)
    return sd


def _bevel_mesh(sd: ShadeData, frame, dev: str) -> None:
    """One frame-space wp.Mesh of every object (global triangle order) for the Bevel node's ray march."""
    verts, faces, objs, vo = [], [], [], 0
    for k, m in enumerate(frame.meshes):
        if m.meta.get("dynamic"):  # deformables (always last): their points move, Bevel does not reach them
            break
        verts.append(np.asarray(m.v, np.float32))
        faces.append(np.asarray(m.tri, np.int32) + vo)
        objs.append(np.full(len(m.tri), k, np.int32))
        vo += len(m.v)
    mesh = wp.Mesh(points=wp.array(np.concatenate(verts), dtype=wp.vec3, device=dev),
                   indices=wp.array(np.concatenate(faces).reshape(-1), dtype=wp.int32, device=dev),
                   bvh_constructor="sah")
    sd.bevel_mesh = mesh.id
    sd.tri_obj = wp.array(np.concatenate(objs), dtype=int, device=dev)
    _KEEP.append(mesh)


def build_lights(pack, frame, num_worlds: int, device: str):
    """The frame's area lights and suns (same in every world) + the pack's world, as an L.LightScene."""
    from . import lights as L
    areas, suns = [], []
    for lt in frame.lights:
        if lt["type"] == "SUN":
            suns.append(_sun(lt))
        else:
            areas.append(_area(lt))
    world = _world_map(pack, device)
    return build_light_scene([areas] * num_worlds, world=world, suns=[suns] * num_worlds, device=device,
                             world_energy_scale=_world_emission_estimate(pack))


def _area(lt: dict):
    from . import lights as L
    p = lt["props"]
    if lt["type"] != "AREA" or p.get("shape", "SQUARE") not in ("SQUARE", "RECTANGLE"):
        raise CameraError(f"light {lt['name']} is {lt['type']} {p.get('shape')}",
                          "use square or rectangle area lights, or suns (other light types are not supported yet)")
    sy = p["size"] if p.get("shape") == "SQUARE" else p.get("size_y", p["size"])
    return L.make_area_light(lt["matrix_world"], p["size"], sy, p["color"], p["energy"],
                             spread=p.get("spread", math.pi), normalize=p.get("normalize", True),
                             exposure=p.get("exposure", 0.0),
                             use_mis=lt.get("cycles", {}).get("use_multiple_importance_sampling", True))


def _sun(lt: dict):
    from . import lights as L
    p = lt["props"]
    return L.make_distant_light(lt["matrix_world"], p["color"], p["energy"], p.get("angle", math.radians(0.526)),
                                exposure=p.get("exposure", 0.0),
                                use_mis=lt.get("cycles", {}).get("use_multiple_importance_sampling", True))


def build_light_scene(areas, world=None, suns=None, device=None, world_energy_scale: float = 1.0):
    """L.build_light_scene plus the background emitter's tree energy as Cycles computes it: average_radiance * pi *
    the shader's emission estimate (scene/light_tree.cpp:229-241); only selection probabilities change."""
    from . import lights as L
    sc = L.build_light_scene(areas, world=world, suns=suns, device=device)
    nodes = sc.nodes.numpy()
    ems = sc.emitters.numpy()
    inner = nodes["type"] == 0
    nodes["num_emitters"][inner] = -1  # scene/light.cpp:554 (lights_build writes -1 already; kept as a guard)
    if world is not None and world_energy_scale != 1.0:
        ems["energy"][ems["kind"] == 1] *= np.float32(world_energy_scale)
        for w in range(nodes.shape[0]):
            for i in np.flatnonzero(nodes["type"][w] == 2):
                a, n = int(nodes["first_emitter"][w, i]), int(nodes["num_emitters"][w, i])
                nodes["energy"][w, i] = ems["energy"][w, a:a + n].sum()
            if inner[w, 0]:
                nodes["energy"][w, 0] = nodes["energy"][w, nodes["left"][w, 0]] + nodes["energy"][w, nodes["right"][w, 0]]
        sc.emitters = wp.array(ems, dtype=sc.emitters.dtype, device=sc.emitters.device)
    sc.nodes = wp.array(nodes, dtype=sc.nodes.dtype, device=sc.nodes.device)
    return sc


def _world_emission_estimate(pack) -> float:
    """Cycles Shader::estimate_emission of the world's Background node: colour (if unlinked) * strength."""
    if "image" in pack.static["world"]:
        return float(pack.static["world"].get("emission_estimate", 1.0))
    from .world_nodes import lighting_background
    tree = pack.static["world"]["tree"]
    bg = lighting_background(tree) or next(n for n in tree["nodes"] if n["type"] == "ShaderNodeBackground")
    est = 1.0
    for i in bg["inputs"]:
        if i["name"] == "Strength" and not i.get("linked"):
            est *= float(i["value"])
        if i["name"] == "Color" and not i.get("linked") and i.get("value") is not None:
            est *= float(np.mean(np.abs(np.asarray(i["value"][:3], np.float64))))
    return est


def _world_map(pack, device: str):
    """The world: an equirect HDRI (Environment Texture -> Background), or an in-memory image
    (``static["world"]["image"]``, rows top first, with ``strength`` and ``use_mis``) for scenes built in Python."""
    from . import lights as L
    w = pack.static["world"]
    if "image" in w:
        return L.make_world(np.asarray(w["image"], np.float32), strength=float(w.get("strength", 1.0)),
                            use_mis=bool(w.get("use_mis", True)), device=device)
    from .world_nodes import lighting_background, mapping_rotation
    tree = w["tree"]
    env = next((n for n in tree["nodes"] if n["type"] == "ShaderNodeTexEnvironment"), None)
    bg = lighting_background(tree) or next(n for n in tree["nodes"] if n["type"] == "ShaderNodeBackground")
    strength = float(next(i["value"] for i in bg["inputs"] if i["name"] == "Strength"))
    if env is None:
        raise CameraError("the pack's world has no environment texture",
                          "give the world an HDRI (Environment Texture -> Background)")
    tex = pack.static["textures"][env["image"]]
    img = L.load_hdr(str(pack.root / tex["file"]))
    use_mis = w["cycles"].get("sampling_method", "AUTOMATIC") != "NONE"
    return L.make_world(img, strength=strength, rotation=mapping_rotation(tree), use_mis=use_mis, device=device)


def read_textures(root, files: list[str], arrays: dict | None = None) -> list[np.ndarray]:
    """Images as host float32 (H, W, C<=3), row 0 = BOTTOM, /255 or /65535, no sRGB decode (the generated functions
    convert after interpolation). A file name that is a key of ``arrays`` (``static["texture_arrays"]``: float or uint8
    (H, W, 1|3|4), row 0 = TOP, sRGB-encoded like a file) is taken from there; Pillow is needed only for real files."""
    arrays = arrays or {}
    out = []
    for f in files:
        if f in arrays:
            a = np.asarray(arrays[f])
            a = a.astype(np.float32) / np.float32(255.0 if a.dtype == np.uint8 else 1.0)
        else:
            a = _read_file(root / f)
        if a.ndim == 2:
            a = a[..., None]
        out.append(np.ascontiguousarray(a[::-1, :, :3]))
    return out


def _read_file(path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CameraError("this scene has image textures and Pillow is not installed", "pip install pillow") from exc
    im = Image.open(path)
    a = np.asarray(im)
    scale = 65535.0 if a.dtype == np.uint16 or im.mode.startswith("I;16") else 255.0
    return a.astype(np.float32) / np.float32(scale)


def load_textures(root, files: list[str], device: str, arrays: dict | None = None) -> list[wp.array]:
    """Image textures as the material functions expect, as float32 (H, W, C) device arrays (see read_textures)."""
    return [wp.array(a, dtype=float, device=device) for a in read_textures(root, files, arrays)]


def camera_params(frame, width: int, height: int) -> dict:
    """The frame's Blender camera -> eye, rotation, image extent (sensor fit), clip, thin-lens aperture and focus."""
    c = frame.camera
    p, dof = c["props"], c.get("dof", {})
    m = np.asarray(c["matrix_world"], np.float64).reshape(4, 4)
    s = p["sensor_width"] / p["lens"]
    fit = p.get("sensor_fit", "AUTO")
    if fit == "AUTO":
        fit = "HORIZONTAL" if width >= height else "VERTICAL"
    if fit == "HORIZONTAL":
        hx, hy = 0.5 * s, 0.5 * s * height / width
    else:
        hx, hy = 0.5 * s * width / height, 0.5 * s
    use_dof = bool(dof.get("use_dof", False))
    aperture = p["lens"] * 1e-3 / (2.0 * max(dof.get("aperture_fstop", 8.0), 1e-5)) if use_dof else 0.0
    return {"eye": m[:3, 3], "rot": m[:3, :3], "extent": (hx, hy), "clip": (p["clip_start"], p["clip_end"]),
            "aperture": aperture, "focus": max(float(dof.get("focus_distance", 1.0)), 1e-5)}
