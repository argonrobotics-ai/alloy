# SPDX-License-Identifier: Apache-2.0
"""Stage: Cycles device and render settings, world (HDRI or studio fallback) and lights."""

from __future__ import annotations

import logging
import math

import bpy
import numpy as np
from mathutils import Matrix

import motion
import nodes as nd
from assets import AssetCache
from look import Look

log = logging.getLogger(__name__)

LIGHT_KINDS = {"area": "AREA", "spot": "SPOT", "point": "POINT", "sun": "SUN"}
FORMATS = {"png": ("PNG", "8", "png"), "png16": ("PNG", "16", "png"), "exr": ("OPEN_EXR", "16", "exr")}


class DeviceError(Exception):
    """The requested Cycles device is not available (never fall back silently)."""


def setup_device(scene: bpy.types.Scene, device: str) -> list[str]:
    """Enable only devices of the requested type; return their names. Raises DeviceError if there are none."""
    scene.render.engine = "CYCLES"
    if device == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = device
    except TypeError as exc:
        raise DeviceError(f"--device {device}: Cycles does not support this backend here ({exc})") from exc
    prefs.get_devices()
    enabled = []
    for d in prefs.devices:
        d.use = d.type == device
        if d.use:
            enabled.append(f"{d.type}:{d.name}")
    if not enabled:
        have = sorted({f"{d.type}:{d.name}" for d in prefs.devices})
        raise DeviceError(f"--device {device} requested but Cycles lists no {device} device (found {have}); "
                          "refusing to fall back to CPU")
    scene.cycles.device = "GPU"
    return enabled


def _set_if(obj, attr: str, value) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def setup_render(scene: bpy.types.Scene, look: Look, samples: int, fmt: str, fps: float) -> None:
    """Sampling, denoising, colour management, bounces, motion blur and output format."""
    r, c = look.render, scene.cycles
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = float(r["noise_threshold"])
    c.use_denoising = r.get("denoiser") not in (None, "none", False)
    c.denoiser = "OPTIX" if r.get("denoiser") == "OPTIX" else "OPENIMAGEDENOISE"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.denoising_prefilter = "ACCURATE"
    _set_if(c, "denoising_use_gpu", True)
    b = r["bounces"]
    total = int(b.get("total", 12))
    c.max_bounces = total
    c.diffuse_bounces = min(int(b.get("diffuse", 4)), total)
    c.glossy_bounces = min(int(b.get("glossy", 8)), total)
    c.transmission_bounces = int(b.get("transmission", 16))
    c.volume_bounces = int(b.get("volume", 2))
    c.transparent_max_bounces = max(16, total)
    c.caustics_reflective = bool(r["caustics"].get("reflective", False))
    c.caustics_refractive = bool(r["caustics"].get("refractive", True))
    c.blur_glossy = float(r.get("filter_glossy", 0.3))
    c.seed = 0
    scene.render.film_transparent = bool(r.get("film_transparent", False))
    scene.render.use_persistent_data = True
    _colour(scene, r)
    _motion_blur(scene, r, fps)
    _output(scene, fmt)


def _colour(scene: bpy.types.Scene, r: dict) -> None:
    vs = scene.view_settings
    vs.view_transform = r.get("view", "AgX")
    try:
        vs.look = r.get("look", "None")
    except TypeError:
        log.warning("render.look %r unknown for view %r; using None", r.get("look"), vs.view_transform)
        vs.look = "None"
    vs.exposure = float(r.get("exposure", 0.0))
    vs.gamma = 1.0


def _motion_blur(scene: bpy.types.Scene, r: dict, fps: float) -> None:
    """Shutter in frames = shutter_s * bundle fps; scene fps = bundle fps so velocity blur uses bundle seconds."""
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = scene.render.fps / fps
    mb = r.get("motion_blur")
    shutter = None if not mb else mb.get("shutter_s")
    scene.render.use_motion_blur = bool(shutter)
    if shutter:
        scene.render.motion_blur_shutter = float(shutter) * fps
        _set_if(scene.render, "motion_blur_position", "CENTER")


def _output(scene: bpy.types.Scene, fmt: str) -> None:
    ftype, depth, _ = FORMATS[fmt]
    s = scene.render.image_settings
    s.file_format = ftype
    s.color_mode = "RGBA" if scene.render.film_transparent else "RGB"
    s.color_depth = depth
    if ftype == "PNG":
        s.compression = 15
    else:
        s.exr_codec = "ZIP"
    scene.render.use_file_extension = False


STUDIO_MEAN = (0.3, 0.3, 0.3)  # neutral grey: the fallback studio gradient has no colour cast


def setup_world(scene: bpy.types.Scene, cfg: dict, assets: AssetCache) -> np.ndarray:
    """HDRI lighting (rotated, scaled); camera rays see background_color unless show_hdri.

    Returns the environment's mean linear RGB radiance before `strength` (for auto white balance)."""
    w = bpy.data.worlds.new("alloy_world")
    w.use_nodes = True
    t = w.node_tree
    t.nodes.clear()
    out = nd.node(t, "ShaderNodeOutputWorld")
    path = assets.hdri(cfg["hdri"], cfg.get("res", "4k")) if cfg.get("hdri") else None
    if path:
        light, mean = _hdri_background(t, path, cfg)
    else:
        light, mean = _studio_background(t, cfg), np.asarray(STUDIO_MEAN)
    if cfg.get("show_hdri", False):
        nd.link(t, light, out.inputs["Surface"])
    else:
        bg = nd.node(t, "ShaderNodeBackground", {"Strength": 1.0})
        bg.inputs["Color"].default_value = (*background_linear(cfg, mean), 1.0)
        mix = nd.node(t, "ShaderNodeMixShader")
        nd.link(t, nd.node(t, "ShaderNodeLightPath").outputs["Is Camera Ray"], mix.inputs[0])
        nd.link(t, light, mix.inputs[1])
        nd.link(t, bg.outputs[0], mix.inputs[2])
        nd.link(t, mix.outputs[0], out.inputs["Surface"])
    scene.world = w
    return mean


def background_linear(cfg: dict, mean) -> tuple:
    """Linear RGB that camera rays see when show_hdri is off: background_color (sRGB), or with
    background_from_hdri the environment's mean radiance x strength (a ground plane's horizon fades into it)."""
    if cfg.get("background_from_hdri", False):
        lin = np.asarray(mean, dtype=float)[:3] * float(cfg.get("strength", 1.0))
        return tuple(float(v) for v in lin)
    return _rgba(cfg["background_color"])[:3]


def _rgba(c) -> tuple:
    lin = motion.srgb_to_linear(c)
    return (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)


def _hdri_background(t, path, cfg: dict):
    coord = nd.node(t, "ShaderNodeTexCoord")
    mapping = nd.node(t, "ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(float(cfg.get("rotation_deg", 0.0))))
    img = bpy.data.images.load(str(path), check_existing=True)
    env = nd.node(t, "ShaderNodeTexEnvironment", image=img)
    bg = nd.node(t, "ShaderNodeBackground", {"Strength": float(cfg.get("strength", 1.0))})
    nd.link(t, coord.outputs["Generated"], mapping.inputs["Vector"])
    nd.link(t, mapping.outputs[0], env.inputs["Vector"])
    nd.link(t, env.outputs["Color"], bg.inputs["Color"])
    px = np.empty(img.size[0] * img.size[1] * img.channels, np.float32)
    img.pixels.foreach_get(px)
    mean = motion.equirect_mean(px if img.channels == 4 else _rgba_pixels(px, img.channels), img.size[1], img.size[0])
    log.info("HDRI %s: mean linear RGB %s", path.name, np.round(mean, 4).tolist())
    return bg.outputs[0], mean


def _rgba_pixels(px: np.ndarray, channels: int) -> np.ndarray:
    out = np.ones((len(px) // channels, 4), np.float32)
    out[:, :channels] = px.reshape(-1, channels)[:, :4]
    return out.ravel()


def _studio_background(t, cfg: dict):
    """Neutral studio: a bright soft top, mid-grey horizon and dark floor, from the view direction's Z."""
    coord = nd.node(t, "ShaderNodeTexCoord")
    sep = nd.node(t, "ShaderNodeSeparateXYZ")
    nd.link(t, coord.outputs["Generated"], sep.inputs[0])
    ramp = nd.node(t, "ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.35, (0.03, 0.03, 0.03, 1.0)
    els[1].position, els[1].color = 1.0, (1.0, 1.0, 1.0, 1.0)
    mid = els.new(0.55)
    mid.color = (0.25, 0.25, 0.25, 1.0)
    nd.link(t, nd.math(t, "MULTIPLY_ADD", sep.outputs[2], 0.5, 0.5), ramp.inputs[0])
    bg = nd.node(t, "ShaderNodeBackground", {"Strength": float(cfg.get("strength", 1.0))})
    nd.link(t, ramp.outputs[0], bg.inputs["Color"])
    return bg.outputs[0]


def setup_lights(lights: list, coll: bpy.types.Collection) -> None:
    """Lights aimed from eye to target; colour from kelvin (black body); watts as Blender power."""
    for i, lt in enumerate(lights):
        kind = LIGHT_KINDS[lt.get("type", "area")]
        data = bpy.data.lights.new(f"light_{i}", kind)
        data.energy = float(lt.get("watts", 50.0))
        data.color = tuple(float(x) for x in motion.kelvin_to_rgb(float(lt.get("kelvin", 5500.0))))
        if kind == "AREA":
            data.shape = "DISK" if lt.get("shape") == "disk" else "RECTANGLE"
            size = lt.get("size_m", 0.5)
            data.size, data.size_y = (size, size) if not isinstance(size, list) else (size[0], size[1])
            if "spread_deg" in lt:
                data.spread = math.radians(float(lt["spread_deg"]))
        elif kind in ("POINT", "SPOT"):
            data.shadow_soft_size = float(lt.get("size_m", 0.05))
        obj = bpy.data.objects.new(f"light_{i}", data)
        coll.objects.link(obj)
        obj.matrix_world = Matrix(motion.look_at(lt["eye"], lt.get("target", [0, 0, 0])).tolist())


def setup_horizon_fade(world: dict, mats) -> None:
    """world.fade {"material": "floor", "start_m": 110, "depth_m": 500}: to camera rays, that material turns
    transparent between start_m and start_m + depth_m from the camera, so a large ground plane dissolves into
    exactly the background camera rays see (no horizon line, whatever the exposure or white balance). Light,
    shadow and reflection rays still see the full material."""
    cfg = world.get("fade")
    if not cfg:
        return
    start, depth = float(cfg.get("start_m", 110.0)), float(cfg.get("depth_m", 500.0))
    tree = mats.get(cfg["material"]).node_tree
    out = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    surface = out.inputs["Surface"].links[0].from_socket
    rng = nd.node(tree, "ShaderNodeMapRange", {"From Min": start, "From Max": start + depth, "To Min": 0.0,
                                               "To Max": 1.0}, interpolation_type="SMOOTHSTEP", clamp=True)
    nd.link(tree, nd.node(tree, "ShaderNodeCameraData").outputs["View Distance"], rng.inputs["Value"])
    fac = nd.math(tree, "MULTIPLY", rng.outputs["Result"],
                  nd.node(tree, "ShaderNodeLightPath").outputs["Is Camera Ray"])
    mix = nd.node(tree, "ShaderNodeMixShader")
    nd.link(tree, fac, mix.inputs[0])
    nd.link(tree, surface, mix.inputs[1])
    nd.link(tree, nd.node(tree, "ShaderNodeBsdfTransparent").outputs[0], mix.inputs[2])
    nd.link(tree, mix.outputs[0], out.inputs["Surface"])
    log.info("horizon fade: %s turns transparent to camera rays between %.0f and %.0f m", cfg["material"], start,
             start + depth)
