# SPDX-License-Identifier: Apache-2.0
"""White balance (look.render.white_balance): "auto", {"kelvin": K, "tint": t} or "off".

"auto" works like a photographer's grey-ball reading at the subject: before the scene is built, a tiny probe render
(32x32 px) of a grey diffuse ball at the first camera target, seen from the camera's side, under the look lights and
the environment only, gives the colour of the light. That colour becomes Blender 4.5's view white point (chromatic
adaptation before the view transform), so a neutral grey renders neutral. If the probe fails, an analytic estimate
(lights by power and distance, environment by its mean colour) is used. Without view white balance (older Blender) a
compositor gain does the same in linear RGB.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

import meshdata
import motion
import primitives
from look import Look, LookError

log = logging.getLogger(__name__)
DEFAULT_TINT = 10.0  # Blender's tint for the daylight locus (its UI default)
PROBE_PX, PROBE_SPP, PROBE_R, PROBE_DIST = 32, 64, 0.01, 0.12
PROBE_SETTINGS = (("render", "resolution_x"), ("render", "resolution_y"), ("render", "resolution_percentage"),
                  ("render", "use_motion_blur"), ("render", "filepath"), ("cycles", "samples"),
                  ("cycles", "use_denoising"), ("cycles", "use_adaptive_sampling"))


def _probe_objects(scene: bpy.types.Scene, target: np.ndarray, eye: np.ndarray) -> list:
    """Grey diffuse ball at target and a camera looking at it from the main camera's side."""
    mat = bpy.data.materials.new("awb_probe")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    mat.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.0
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    me = meshdata.make_mesh("awb_probe", *primitives.sphere(PROBE_R, PROBE_R, PROBE_R, 32, 16), None)
    me.materials.append(mat)
    ball = meshdata.new_object("awb_probe", me, scene.collection)
    ball.location = target
    view = (target - eye) / max(float(np.linalg.norm(target - eye)), 1e-9)
    cam = bpy.data.objects.new("awb_probe_cam", bpy.data.cameras.new("awb_probe_cam"))
    cam.data.lens, cam.data.sensor_width = 100.0, 36.0
    scene.collection.objects.link(cam)
    cam.matrix_world = Matrix(motion.look_at(target - view * PROBE_DIST, target).tolist())
    return [ball, cam, me, mat, cam.data]


def probe_illuminant(scene: bpy.types.Scene, target, eye) -> np.ndarray:
    """Mean linear RGB of the lit side of a grey ball at target (lights and environment only)."""
    saved = {k: getattr(getattr(scene, k[0]), k[1]) for k in PROBE_SETTINGS}
    fmt = (scene.render.image_settings.file_format, scene.render.image_settings.color_depth, scene.camera)
    objs = _probe_objects(scene, np.asarray(target, float), np.asarray(eye, float))
    tmp = Path(tempfile.mkdtemp(prefix="awb_")) / "probe.exr"
    try:
        scene.camera = objs[1]
        for (grp, attr), val in zip(PROBE_SETTINGS, (PROBE_PX, PROBE_PX, 100, False, str(tmp), PROBE_SPP, False,
                                                      False)):
            setattr(getattr(scene, grp), attr, val)
        scene.render.image_settings.file_format, scene.render.image_settings.color_depth = "OPEN_EXR", "32"
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp))
        px = np.empty(PROBE_PX * PROBE_PX * 4, np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        c = PROBE_PX // 2
        return px.reshape(PROBE_PX, PROBE_PX, 4)[c - 4:c + 4, c - 4:c + 4, :3].reshape(-1, 3).mean(axis=0)
    finally:
        for k, v in saved.items():
            setattr(getattr(scene, k[0]), k[1], v)
        scene.render.image_settings.file_format, scene.render.image_settings.color_depth, scene.camera = fmt
        bpy.data.objects.remove(objs[0])
        bpy.data.objects.remove(objs[1])
        bpy.data.meshes.remove(objs[2])
        bpy.data.materials.remove(objs[3])
        bpy.data.cameras.remove(objs[4])


def _auto_white(scene: bpy.types.Scene, look: Look, env_mean: np.ndarray, key: dict) -> tuple[np.ndarray, str]:
    try:
        light = probe_illuminant(scene, key["target"], key["eye"])
        if float(light.max()) > 0.0:
            return light, "probe"
    except Exception as exc:  # noqa: BLE001 - fall back to the analytic estimate
        log.warning("white balance probe failed (%s); using the analytic estimate", exc)
    return motion.scene_illuminant(look.lights, env_mean, float(look.world.get("strength", 1.0)), key["target"]), \
        "estimate"


def setup_white_balance(scene: bpy.types.Scene, look: Look, env_mean: np.ndarray, key: dict) -> dict:
    """Apply the look's white balance; returns what was applied (for timing.jsonl)."""
    mode = look.render.get("white_balance", "auto")
    vs = scene.view_settings
    has_view_wb = hasattr(vs, "use_white_balance")
    if mode == "off" or mode is False or mode is None:
        if has_view_wb:
            vs.use_white_balance = False
        return {"mode": "off"}
    if isinstance(mode, dict):
        kelvin, tint = float(mode.get("kelvin", 6500.0)), float(mode.get("tint", DEFAULT_TINT))
        if has_view_wb:
            vs.use_white_balance = True
            vs.white_balance_temperature, vs.white_balance_tint = kelvin, tint
        else:
            _compositor_gain(scene, motion.kelvin_to_rgb(kelvin))
        return {"mode": "kelvin", "temperature": kelvin, "tint": tint}
    if mode != "auto":
        raise LookError(f"render.white_balance is {mode!r}, expected 'auto', 'off' or {{'kelvin': K, 'tint': t}}")
    light, source = _auto_white(scene, look, env_mean, key)
    white = light / max(float(light.max()), 1e-12)
    if not has_view_wb:
        _compositor_gain(scene, white)
        return {"mode": "auto", "whitepoint": np.round(white, 4).tolist(), "via": "compositor"}
    vs.use_white_balance = True
    vs.white_balance_whitepoint = tuple(float(x) for x in white)
    info = {"mode": "auto", "source": source, "whitepoint": np.round(white, 4).tolist(),
            "temperature": round(vs.white_balance_temperature, 1), "tint": round(vs.white_balance_tint, 2)}
    log.info("auto white balance: scene light %s -> %.0f K, tint %.1f", info["whitepoint"], info["temperature"],
             info["tint"])
    return info


def _compositor(scene: bpy.types.Scene) -> tuple:
    """(tree, socket feeding the Composite node, Composite node), creating Render Layers -> Composite once."""
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    comp = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"), None)
    if comp is None:
        tree.nodes.clear()
        rl, comp = tree.nodes.new("CompositorNodeRLayers"), tree.nodes.new("CompositorNodeComposite")
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    return tree, comp.inputs["Image"].links[0].from_socket, comp


def _insert(scene: bpy.types.Scene, node_type: str):
    """Add a compositor node between the current image and the Composite output; returns the node."""
    tree, src, comp = _compositor(scene)
    node = tree.nodes.new(node_type)
    tree.links.new(src, node.inputs["Image"])
    tree.links.new(node.outputs["Image"], comp.inputs["Image"])
    return node


def _compositor_gain(scene: bpy.types.Scene, white: np.ndarray) -> None:
    """Divide the render by `white` (linear, per channel) in the compositor, before the view transform."""
    gain = 1.0 / np.maximum(np.asarray(white, float), 1e-6)
    gain /= float(np.dot(gain, (0.2126, 0.7152, 0.0722)) / np.dot(np.ones(3), (0.2126, 0.7152, 0.0722)))
    bal = _insert(scene, "CompositorNodeColorBalance")
    bal.correction_method = "OFFSET_POWER_SLOPE"
    bal.slope = tuple(float(x) for x in gain)


def setup_glare(scene: bpy.types.Scene, cfg: dict | None) -> None:
    """render.glare {"type": "fog_glow"|"bloom", "threshold": 1.0, "size": 6, "strength": 1.0}: lens bloom on
    highlights brighter than threshold (scene-linear), so LEDs and specular glints glow like through a lens.

    `size` follows Blender's classic 6..9 scale (glare reach 2^size pixels at 1080p)."""
    if not cfg:
        return
    kind = str(cfg.get("type", "fog_glow")).upper()
    if kind not in ("FOG_GLOW", "BLOOM", "STREAKS", "GHOSTS", "SIMPLE_STAR"):
        raise LookError(f"render.glare.type is {cfg.get('type')!r}, expected fog_glow or bloom")
    g = _insert(scene, "CompositorNodeGlare")
    g.glare_type = kind
    g.quality = "HIGH"
    size = float(cfg.get("size", 6))
    for name, value in (("Threshold", float(cfg.get("threshold", 1.0))), ("Strength", float(cfg.get("strength", 1.0))),
                        ("Size", min(max((size - 5.0) / 4.0, 0.0), 1.0))):
        if name in g.inputs:
            g.inputs[name].default_value = value
    log.info("glare %s threshold %.2f size %.1f", kind.lower(), float(cfg.get("threshold", 1.0)), size)
