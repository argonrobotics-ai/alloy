# SPDX-License-Identifier: Apache-2.0
"""Textured meshes straight from Python to the full-material path tracer: no pack directory, no Blender, no codegen.

Every object becomes a *generic* material (``alloy.camera.pbr.generic``): colour, roughness, metallic and an optional
base-colour texture are data, so the generated shader module is the same text for every ``mesh_scene`` and a new
scene costs a render, never a compile. UVs follow Blender (origin bottom-left, v up); an image array is stored the
usual way (row 0 = top) and wraps (REPEAT) outside 0..1.

Example::

    from alloy.camera import Camera, TexturedMesh, mesh_scene
    quad = TexturedMesh("quad", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]],
                        uv=[[0, 0], [1, 0], [1, 1], [0, 1]], texture="albedo.png")
    img = Camera("c", look_at=((0.5, -1.5, 1.0), (0.5, 0.5, 0.0)), width=256, height=256).render(mesh_scene([quad]))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import CameraError
from .scene import IDENTITY_XF, Lighting

log = logging.getLogger(__name__)
ZERO_POSE = tuple(float(x) for x in IDENTITY_XF)


@dataclass
class TexturedMesh:
    """One object: ``points`` (V, 3), ``tris`` (T, 3), ``uv`` (V, 2) per vertex or (T, 3, 2) per corner (Blender
    convention, origin bottom-left), ``texture`` an image path or a float / uint8 (H, W, 1|3|4) array (row 0 = top,
    sRGB-encoded), ``color`` the sRGB base colour (used without a texture, or multiplied into it with
    ``texture_tint``: a glTF baseColorFactor), ``pose`` (7,) = x y z qx qy qz qw.
    ``body`` (:func:`model_scene` only): the model body it rides on, by label or the label's last ``/`` part
    ("" = fixed in the world); ``pose`` is then in that body's frame. ``transmission`` 1 with ``ior`` makes it glass or
    water (a closed mesh, triangles wound outward), ``absorption`` its Beer-Lambert colour (sigma per metre, rgb).

    Example::

        TexturedMesh("floor", pts, tris, color=(0.6, 0.6, 0.65), roughness=0.8)
        TexturedMesh("label", quad_pts, quad_tris, uv=quad_uv, texture=label_rgb, body="item")   # rides on "item"
        TexturedMesh("water", pts, tris, color=(1, 1, 1), roughness=0.0, transmission=1.0, ior=1.33)
    """

    name: str
    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray | None = None
    texture: str | Path | np.ndarray | None = None
    color: tuple = (0.8, 0.8, 0.8)
    roughness: float = 0.5
    metallic: float = 0.0
    pose: tuple = field(default=ZERO_POSE)
    body: str = ""
    transmission: float = 0.0
    ior: float = 1.5
    absorption: tuple | None = None
    texture_tint: bool = False


def _checked_arrays(obj: TexturedMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pts = np.asarray(obj.points, np.float32)
    tris = np.asarray(obj.tris, np.int32)
    if pts.ndim != 2 or pts.shape[1] != 3 or tris.ndim != 2 or tris.shape[1] != 3:
        raise CameraError(f"object {obj.name!r}: points {pts.shape}, tris {tris.shape}",
                          "give points as (V, 3) and tris as (T, 3)")
    if tris.size and (tris.min() < 0 or tris.max() >= len(pts)):
        raise CameraError(f"object {obj.name!r}: a triangle index is outside 0..{len(pts) - 1}", "fix tris")
    if obj.uv is None:
        if obj.texture is not None:
            raise CameraError(f"object {obj.name!r} has a texture but no uv", "pass uv=(V, 2) or (T, 3, 2)")
        return pts, tris, None
    uv = np.asarray(obj.uv, np.float32)
    if uv.shape == (len(pts), 2):
        uv = uv[tris]
    if uv.shape != (len(tris), 3, 2):
        raise CameraError(f"object {obj.name!r}: uv {uv.shape}",
                          f"use ({len(pts)}, 2) per vertex or ({len(tris)}, 3, 2) per corner")
    return pts, tris, uv


def _world_points(pts: np.ndarray, pose) -> tuple[np.ndarray, np.ndarray]:
    """(points moved by ``pose``, object-to-world matrix)."""
    from .model_pack import _xf_matrix
    pose = np.asarray(pose, np.float64)
    if pose.shape != (7,):
        raise CameraError(f"pose {pose.shape}", "use [x, y, z, qx, qy, qz, qw]")
    m = _xf_matrix(pose)
    return (pts.astype(np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32), m


def _texture_entry(obj: TexturedMesh, index: int, static: dict) -> str | None:
    """Register the object's texture in the pack (an in-memory array or a file path); returns its key."""
    if obj.texture is None:
        return None
    if isinstance(obj.texture, (str, Path)):
        path = Path(obj.texture).expanduser().resolve()
        if not path.exists():
            raise CameraError(f"object {obj.name!r}: texture {path} does not exist", "pass an existing image file")
        key, file = path.name, str(path)
        if static["textures"].get(key, {"file": file})["file"] != file:
            key = f"{index}:{path.name}"
    else:
        arr = np.asarray(obj.texture)
        if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
            raise CameraError(f"object {obj.name!r}: texture array {arr.shape}", "use (H, W), (H, W, 3) or (H, W, 4)")
        key = file = f"array:{index}"
        static["texture_arrays"][file] = arr
    static["textures"][key] = {"file": file, "colorspace": "sRGB"}
    return key


def _material(obj: TexturedMesh, tex_key: str | None) -> dict:
    from .looks import rgba
    g = {"base_color": list(rgba(obj.color)[:3]), "roughness": float(obj.roughness), "metallic": float(obj.metallic),
         "texture": tex_key, "transmission": float(obj.transmission), "ior": float(obj.ior)}
    if obj.texture_tint and tex_key is not None:
        g["texture_tint"] = True
    if obj.absorption is not None:
        g["absorption"] = [float(x) for x in obj.absorption]
    return {"generic": g}


def _pack_meshes(static: dict, objects: list[TexturedMesh]) -> list:
    """Pack meshes of ``objects`` (moved by their poses), registering their generic materials and textures in
    ``static``."""
    from .pbr.pack_io import Mesh
    from .smooth import corner_normals
    meshes = []
    for i, obj in enumerate(objects):
        pts, tris, uv = _checked_arrays(obj)
        moved, o2w = _world_points(pts, obj.pose)
        mat = f"mesh{i}:{obj.name}"
        static["materials"][mat] = _material(obj, _texture_entry(obj, i, static))
        meta = {"name": obj.name, "object_name": obj.name, "materials": [mat], "object_to_world": o2w.tolist()}
        meshes.append(Mesh(meta, moved, tris, corner_normals(moved, tris), np.zeros(len(tris), np.int32),
                           {} if uv is None else {"UVMap": uv}))
    return meshes


def _studio(points: np.ndarray, lighting: Lighting | None) -> tuple[dict, list[dict]]:
    """(world, lights): a bright gradient sky and two soft area lights sized to the scene, or ``lighting``'s suns."""
    from .model_pack import _lighting_lights, _look_lights, _world, studio_sky
    if lighting is not None:
        return _world({}, lighting), _lighting_lights(lighting)
    sky = studio_sky()
    world = {"image": sky, "strength": 1.0, "use_mis": True, "emission_estimate": float(sky.mean())}
    lo, hi = points.min(0), points.max(0)
    centre, radius = 0.5 * (lo + hi), max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)
    specs = []
    for k, (offset, kelvin) in enumerate((((1.6, -1.8, 2.2), 5200.0), ((-2.2, 1.2, 1.4), 6500.0))):
        eye = centre + radius * np.asarray(offset)
        d = float(np.linalg.norm(eye - centre))
        specs.append({"type": "area", "eye": eye.tolist(), "target": centre.tolist(), "size_m": 1.2 * radius,
                      "watts": (30.0 if k == 0 else 12.0) * d * d, "kelvin": kelvin})
    return world, _look_lights({"lights": specs})


def mesh_scene(objects: list[TexturedMesh], lighting: Lighting | None = None, num_worlds: int = 1,
               device: str | None = None):
    """A :class:`~alloy.camera.pbr.PbrScene` of the objects (every world shows the same objects), lit by a soft
    studio (bright sky and two area lights fitted to the objects) unless ``lighting`` gives suns and a sky. A camera
    without a pose auto-frames the objects, like model scenes; ``Camera(look_at=...)`` places it.

    Example::

        scene = mesh_scene([TexturedMesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])], num_worlds=2)
        scene.num_worlds     # 2
    """
    from .camera import default_device
    from .model_pack import _default_camera, _settings
    from .pbr import PbrScene
    from .pbr.pack_io import Frame, Pack
    if not objects:
        raise CameraError("mesh_scene got no objects", "pass at least one TexturedMesh")
    on_body = [o.name for o in objects if o.body]
    if on_body:
        raise CameraError(f"objects {on_body} name a body, and mesh_scene has no model",
                          "use model_scene(model, objects) to put meshes on a model's bodies")
    static = {"pack_version": 2, "frames": [0], "settings": _settings({}), "materials": {}, "textures": {},
              "texture_arrays": {}}
    meshes = _pack_meshes(static, objects)
    static["materials_used_by_meshes"] = sorted(static["materials"])
    all_pts = np.concatenate([m.v for m in meshes])
    static["world"], lights = _studio(all_pts, lighting)
    frame = Frame(0, _default_camera(meshes), lights, meshes)
    n = len(meshes)
    body = np.full((n, num_worlds), -1, np.int32)
    local = np.tile(IDENTITY_XF, (n, num_worlds, 1)).astype(np.float32)
    log.debug("mesh_scene: %d objects, %d worlds", n, num_worlds)
    return PbrScene(Pack(Path("/"), static, 2), frame, num_worlds, str(device or default_device()),
                    binding=(body, local))



def _body_row(labels: list[str], worlds: np.ndarray, name: str, num_worlds: int) -> np.ndarray:
    """(worlds,) int32: the body labelled ``name`` (else whose label ends in ``/name``) in each world, -1 where none.
    A body outside any world (world -1) serves every world."""
    hits = [i for i, lb in enumerate(labels) if lb == name] or \
        [i for i, lb in enumerate(labels) if lb.split("/")[-1] == name]
    row = np.full(num_worlds, -1, np.int32)
    for i in hits:
        w = int(worlds[i])
        for ww in (range(num_worlds) if w < 0 else [w] if w < num_worlds else []):
            if row[ww] >= 0:
                raise CameraError(f"body {name!r} matches both {labels[row[ww]]!r} and {labels[i]!r} in world {ww}",
                                  "give the body's full label")
            row[ww] = i
    return row


def _body_rows(model, objects: list[TexturedMesh], num_worlds: int) -> np.ndarray:
    """(objects, worlds) int32: the body each object rides on in each world (-1: fixed in the world)."""
    from .model_pack import _labels
    labels = [str(lb) for lb in _labels(model)[1]]
    bw = getattr(model, "body_world", None)
    worlds = bw.numpy() if bw is not None else np.zeros(len(labels), np.int32)
    out = np.full((len(objects), num_worlds), -1, np.int32)
    for k, obj in enumerate(objects):
        if not obj.body:
            continue
        out[k] = _body_row(labels, worlds, obj.body, num_worlds)
        missing = np.flatnonzero(out[k] < 0)
        if missing.size:
            raise CameraError(f"object {obj.name!r}: no body {obj.body!r} in world {int(missing[0])}",
                              f"use a body label from the model, e.g. {labels[:6]}")
    return out


def model_scene(model, objects: list[TexturedMesh] = (), look=None, lighting: Lighting | None = None,
                device: str | None = None, items: dict | None = None):
    """A Newton model's full-material scene (``look`` / ``lighting`` as the camera uses them) plus textured meshes
    riding on its bodies in every world: printed labels, decals, markers (``items``: the scene items' look layer,
    ``item_looks.items_look``). Each object's ``body`` names its body
    (``pose`` in that body's frame); objects without one are fixed at ``pose`` in every world. Render with
    ``Camera.render(scene, state)``; the meshes move with their bodies.

    Example::

        label = TexturedMesh("label", quad_pts, quad_tris, uv=quad_uv, texture=label_rgb, body="item")
        scene = model_scene(model, [label])
        rgb = Camera("c", look_at=((0.3, 0, 0.2), (0, 0, 0.1)), width=320, height=240).render(scene, state)
    """
    from .model_pack import pack_from_model
    from .pbr import PbrScene
    pk, frame, (body, local, present) = pack_from_model(model, look, lighting, items)
    objects = list(objects)
    if not objects:
        return PbrScene(pk, frame, body.shape[1], str(device or model.device), binding=(body, local, present))
    rows = _body_rows(model, objects, body.shape[1])
    st = pk.static
    st.setdefault("texture_arrays", {})
    st.setdefault("textures", {})
    extra = _pack_meshes(st, objects)
    st["materials_used_by_meshes"] = sorted(set(st["materials_used_by_meshes"]) | {m.meta["materials"][0]
                                                                                    for m in extra})
    frame.meshes.extend(extra)
    ident = np.tile(np.asarray(IDENTITY_XF, np.float32), (len(objects), body.shape[1], 1))
    binding = (np.concatenate([body, rows]).astype(np.int32), np.concatenate([local, ident]).astype(np.float32),
               np.concatenate([present, np.ones(rows.shape, bool)]))
    log.debug("model_scene: %d model meshes + %d objects, %d worlds", len(body), len(objects), body.shape[1])
    return PbrScene(pk, frame, body.shape[1], str(device or model.device), binding=binding)
