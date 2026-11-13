# SPDX-License-Identifier: Apache-2.0
"""Newton model + look file -> an in-memory scene pack for the full-material renderer (no Blender, no files).

Every drawn shape of world 0 (and the shared ones; ``from_newton.drawn_shapes``) becomes one mesh in its body's
frame, with smooth corner normals (30 degree auto-smooth) and a look material: the look's ``assign`` rules (regex on
the shape and body labels, first match wins) and ``materials`` entries, else a matte plastic of the shape's colour.
Lights come from the look's ``lights`` and ``world`` (HDRI from the local asset cache, or a studio gradient), else
from a :class:`~alloy.camera.scene.Lighting`. Worlds 1..N-1 reuse world 0's meshes on their own bodies.

Example::

    from alloy.camera.model_pack import pack_from_model
    pk, frame, (body, local, present) = pack_from_model(model, look={"assign": [{"match": "box", "material": "aluminum_bare"}]})
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .card_looks import assigned, fallback_rules
from .errors import CameraError
from .from_newton import _model_colors, _shape_mesh, default_rgb, drawn_shapes, palette_index
from .pbr.pack_io import Frame, Mesh, Pack
from .scene import Lighting, quat_to_matrix
from .smooth import corner_normals

DEFAULT_PRESET = "plastic_matte"


def _xf_matrix(xf) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(np.asarray(xf, np.float64)[3:7])
    m[:3, 3] = np.asarray(xf, np.float64)[:3]
    return m


def _load_look(look) -> dict:
    if look is None:
        return {}
    if isinstance(look, dict):
        return look
    try:
        return json.loads(Path(look).expanduser().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraError(f"cannot read look file {look}: {exc}", "pass a valid look JSON file or a dict") from exc


def _template_shapes(model) -> list[int]:
    """Drawn shapes (``from_newton.drawn_shapes``) of world 0 and shared ones, in model order."""
    world = model.shape_world.numpy()
    vis = drawn_shapes(model)
    return [s for s in range(int(model.shape_count)) if vis[s] and world[s] in (-1, 0)]


def _binding(model, shapes: list[int], num_worlds: int) -> tuple[np.ndarray, np.ndarray]:
    """(body (meshes, worlds), local (meshes, worlds, 7)): the body moving template shape k in world w (-1: fixed),
    and the extra transform of a fixed shape's copy (world-w pose x world-0 pose^-1; identity on bodies)."""
    from .scene import IDENTITY_XF, matrix_to_quat
    world, body = model.shape_world.numpy(), model.shape_body.numpy()
    xf = model.shape_transform.numpy()
    vis = drawn_shapes(model)
    out_b = np.full((len(shapes), num_worlds), -1, np.int32)
    out_l = np.tile(IDENTITY_XF, (len(shapes), num_worlds, 1)).astype(np.float32)
    t0 = [s for s in shapes if world[s] == 0]
    for w in range(num_worlds):
        own = [s for s in range(int(model.shape_count)) if vis[s] and world[s] == w]
        if len(own) != len(t0):
            raise CameraError(f"world {w} has {len(own)} visible shapes, world 0 has {len(t0)}",
                              "use identical worlds (ModelBuilder.replicate) for the full-material camera")
        for k, s in enumerate(shapes):
            c = own[t0.index(s)] if world[s] == 0 else s
            out_b[k, w] = body[c]
            if body[c] < 0 and c != s:
                m = _xf_matrix(xf[c]) @ np.linalg.inv(_xf_matrix(xf[s]))
                out_l[k, w] = np.concatenate([m[:3, 3], matrix_to_quat(m[:3, :3])])
    return out_b, out_l


def _mesh(model, s: int, label: str, material: str) -> Mesh | None:
    """One shape as a pack mesh in its body's frame (shape transform applied), with corner normals."""
    from alloy._vendor import newton
    geom = _shape_mesh(newton, int(model.shape_type.numpy()[s]), model.shape_scale.numpy()[s], model.shape_source[s])
    if geom is None:
        return None
    o2w = _xf_matrix(model.shape_transform.numpy()[s])  # shape -> body frame (world frame for fixed shapes)
    v = (geom[0].astype(np.float64) @ o2w[:3, :3].T + o2w[:3, 3]).astype(np.float32)
    meta = {"name": label, "object_name": label, "materials": [material], "object_to_world": o2w.tolist()}
    return Mesh(meta, v, geom[1], corner_normals(v, geom[1]), np.zeros(len(geom[1]), np.int32))


def _materials(model, shapes: list[int], look: dict, items: dict | None = None) -> tuple[list[str], dict, dict]:
    """(material name per shape, {name: pack material}, {texture key: entry}): the look's rules, then the scene items'
    looks (``items``, see item_looks), then the cards' library looks, else shape colours."""
    from alloy._vendor import newton
    from .looks import material_tree
    colors, pal = _model_colors(model, drawn_shapes(model)), palette_index(model)  # the fast tracer's colours
    flags, ty = model.shape_flags.numpy(), model.shape_type.numpy()
    labels, blabels = _labels(model)
    body = model.shape_body.numpy()
    names, specs, fb = [], {}, None
    for s in shapes:
        bl = blabels[body[s]] if body[s] >= 0 and body[s] < len(blabels) else ""
        name = assigned(look, [labels[s], bl])
        if name is not None:
            specs.setdefault(name, look.get("materials", {}).get(name, name))
            names.append(name)
            continue
        name = assigned(items, [labels[s], bl]) if items else None
        if name is not None:
            specs.setdefault(name, items["materials"][name])
            names.append(name)
            continue
        if fb is None:  # the scene's look does not cover this shape: its card's library look, if any
            fb = fallback_rules([labels[i] for i in shapes] + list(blabels))
        name = assigned(fb, [labels[s], bl]) if fb else None
        if name is not None:
            specs.setdefault(name, fb["materials"][name])
        else:
            rgb = default_rgb(newton, s, colors, int(ty[s]), int(flags[s]), model.shape_source[s], int(pal[s]))
            name = f"shape_{'%02x%02x%02x' % tuple(int(round(255 * c)) for c in rgb)}"
            specs.setdefault(name, {"preset": DEFAULT_PRESET, "base_color": list(rgb)})
        names.append(name)
    mats, tex = {}, {}
    for name, spec in specs.items():
        mats[name] = material_tree(name, spec)
        tex.update(mats[name].pop("textures", {}))
    return names, mats, tex


def _labels(model) -> tuple[list[str], list[str]]:
    n = int(model.shape_count)
    sl = list(getattr(model, "shape_label", None) or getattr(model, "shape_key", None) or [""] * n)
    bl = list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])
    return [lb or f"shape_{i}" for i, lb in enumerate(sl)], bl


def _look_lights(look: dict) -> list[dict]:
    """The look's area lights and suns in pack-frame form (aimed eye -> target, colour from kelvin)."""
    from alloy.render.blender import motion
    out = []
    for i, lt in enumerate(look.get("lights", [])):
        kind = lt.get("type", "area").lower()
        m = np.asarray(motion.look_at(lt["eye"], lt.get("target", [0, 0, 0])), np.float64).tolist()
        color = [float(c) for c in motion.kelvin_to_rgb(float(lt.get("kelvin", 5500.0)))]
        if kind == "sun":
            props = {"color": color, "energy": float(lt.get("strength", lt.get("watts", 3.0))),
                     "angle": math.radians(float(lt.get("angle_deg", 0.526)))}
            out.append({"name": f"light_{i}", "type": "SUN", "matrix_world": m, "props": props})
            continue
        if kind != "area" or lt.get("shape") == "disk":
            raise CameraError(f"look light {i} is a {lt.get('shape', kind)} light",
                              "use rectangle area lights or suns with the full-material camera")
        size = lt.get("size_m", 0.5)
        sx, sy = (size, size) if not isinstance(size, list) else (size[0], size[1])
        props = {"shape": "RECTANGLE", "size": float(sx), "size_y": float(sy), "color": color,
                 "energy": float(lt.get("watts", 50.0)), "spread": math.radians(float(lt.get("spread_deg", 180.0)))}
        out.append({"name": f"light_{i}", "type": "AREA", "matrix_world": m, "props": props})
    return out


def _lighting_lights(lighting: Lighting) -> list[dict]:
    """Suns of a Lighting as pack-frame lights (a sun shines along its local -Z)."""
    from .pose import look_at_matrix
    out = []
    for i, sun in enumerate(lighting.suns):
        m = np.eye(4)
        m[:3, :3] = look_at_matrix(np.zeros(3), -np.asarray(sun.direction_to_sun, np.float64))
        out.append({"name": f"sun_{i}", "type": "SUN", "matrix_world": m.tolist(),
                    "props": {"color": [float(c) for c in sun.strength], "energy": 1.0, "angle": float(sun.angle)}})
    return out


def studio_sky(height: int = 64, width: int = 128) -> np.ndarray:
    """Alloy's studio background as an equirect image (rows top first): dark floor, grey horizon, bright top
    (colour ramp 0.35 -> 0.03, 0.55 -> 0.25, 1.0 -> 1.0 over 0.5 z + 0.5), as the Blender build's default world."""
    z = np.cos(np.pi * (np.arange(height) + 0.5) / height)  # row 0 = straight up
    x = 0.5 * z + 0.5
    v = np.interp(x, [0.35, 0.55, 1.0], [0.03, 0.25, 1.0])
    return np.repeat(np.repeat(v[:, None, None], width, axis=1), 3, axis=2).astype(np.float32)


def _world(look: dict, lighting: Lighting | None) -> dict:
    """The pack world: the look's HDRI (local asset cache), the studio gradient, or the Lighting's constant sky."""
    cfg = look.get("world") or {}
    if cfg.get("hdri"):
        from alloy.render.blender.assets import AssetCache
        from .pbr.lights import load_hdr
        path = AssetCache().hdri(cfg["hdri"], cfg.get("res", "4k"))
        if path is None:
            raise CameraError(f"HDRI {cfg['hdri']!r} is not available", "check the reference or the network")
        return {"image": load_hdr(str(path)), "strength": float(cfg.get("strength", 1.0)), "use_mis": True}
    if look:
        return {"image": studio_sky(), "strength": float(cfg.get("strength", 1.0)), "use_mis": True}
    lt = lighting or Lighting.studio()
    c = np.asarray(lt.world_color, np.float32) * np.float32(lt.world_strength)
    return {"image": np.tile(c, (8, 16, 1)), "strength": 1.0, "use_mis": False,
            "emission_estimate": float(np.mean(np.abs(c)))}


# the look file's render view defaults (alloy.render.blender.look.RENDER_DEFAULTS, which imports Blender-side modules)
VIEW_DEFAULTS = {"view": "AgX", "look": "AgX - Medium High Contrast", "exposure": 0.0}


def _settings(look: dict) -> dict:
    """Cycles settings with Blender's defaults (12 bounces, 4 diffuse / glossy, clamp indirect 10) and the look's view
    (``render.view`` / ``look`` / ``exposure``; AgX "Medium High Contrast" at exposure 0 by default, as Blender)."""
    render = {**VIEW_DEFAULTS, **(look.get("render") or {})}
    caustics = (look.get("render") or {}).get("caustics") or {}
    view = {"view_transform": render["view"], "look": render["look"], "exposure": float(render["exposure"])}
    return {"view_settings": view, "cycles": {"max_bounces": 12, "diffuse_bounces": 4, "glossy_bounces": 4, "transmission_bounces": 12,
                       "sample_clamp_direct": 0.0, "sample_clamp_indirect": 10.0, "seed": 0, "blur_glossy": 1.0,
                       "caustics_reflective": bool(caustics.get("reflective", True))}}


def _add_dynamic(parts: tuple, meshes: list, mats: dict, tex: dict, binding: tuple) -> tuple:
    """Append per-world meshes ``parts`` = (meshes, materials, textures, present) of deformables or liquids to the
    pack; returns the extended binding."""
    from .scene import IDENTITY_XF
    extra, dmats, dtex, present = parts
    meshes.extend(extra)
    mats.update({k: v for k, v in dmats.items() if k not in mats})
    tex.update(dtex)
    n, w = present.shape
    return (np.concatenate([binding[0], np.full((n, w), -1, np.int32)]),
            np.concatenate([binding[1], np.tile(IDENTITY_XF, (n, w, 1)).astype(np.float32)]),
            np.concatenate([binding[2], present]))


def _default_camera(frame_meshes: list[Mesh]) -> dict:
    """A 50 mm three-quarter view of the meshes (the Camera API replaces it with the requested pose)."""
    from .pose import auto_frame
    pts = np.concatenate([m.v for m in frame_meshes]) if frame_meshes else np.zeros((1, 3))
    xf = auto_frame(pts.min(0), pts.max(0))
    return {"matrix_world": _xf_matrix(xf).tolist(), "dof": {"use_dof": False},
            "props": {"lens": 50.0, "sensor_width": 36.0, "sensor_fit": "AUTO", "clip_start": 0.01, "clip_end": 1000.0}}


def pack_from_model(model, look=None, lighting: Lighting | None = None, items: dict | None = None,
                    deform=None, liquid=None) -> tuple[Pack, Frame, tuple]:
    """(pack, frame, (body, local, present)): the model's world-0 shapes as a full-material scene; ``body[k, w]``
    moves mesh k in world w (-1: fixed, placed by ``local[k, w]``), drawn where ``present[k, w]``. ``look``: a look
    dict or JSON path; else ``lighting``. ``items``: the scene items' look layer (materials and rules only, see
    item_looks). ``deform``: a :class:`~alloy.camera.deform.DeformLayout`, whose per-world meshes come last, then
    those of ``liquid`` (a :class:`~alloy.camera.liquid_scene.LiquidLayout`: capacity placeholders, filled live)."""
    look = _load_look(look)
    shapes = _template_shapes(model)
    if not shapes:
        raise CameraError("the model has no visible shapes in world 0", "add shapes before creating the camera")
    names, mats, tex = _materials(model, shapes, look, items)
    labels, _ = _labels(model)
    meshes, keep = [], []
    for k, (s, name) in enumerate(zip(shapes, names)):
        m = _mesh(model, s, labels[s], name)
        if m is not None:
            meshes.append(m)
            keep.append(k)
    num_worlds = max(1, int(getattr(model, "world_count", 0) or 1))
    body, local = _binding(model, shapes, num_worlds)
    binding = (body[keep], local[keep], np.ones((len(keep), num_worlds), bool))
    if deform is not None:
        from .deform_pack import deform_meshes
        binding = _add_dynamic(deform_meshes(deform, look, items), meshes, mats, tex, binding)
    if liquid is not None:
        from .liquid_scene import liquid_pack_meshes
        binding = _add_dynamic(liquid_pack_meshes(liquid), meshes, mats, tex, binding)
    lights = _look_lights(look) if look else _lighting_lights(lighting or Lighting.studio())
    static = {"pack_version": 2, "frames": [0], "settings": _settings(look), "world": _world(look, lighting),
              "materials": mats, "materials_used_by_meshes": sorted({m.meta["materials"][0] for m in meshes}),
              "textures": tex}
    frame = Frame(0, _default_camera([m for m in meshes if m.meta.get("kind") != "liquid"]), lights, meshes)
    return Pack(Path("/"), static, 2), frame, binding
