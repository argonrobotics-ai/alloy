# SPDX-License-Identifier: Apache-2.0
"""Physically based Cycles materials from presets plus look overrides."""

from __future__ import annotations

import logging
from typing import Callable

import bpy

import matkeys
import motion
import nodes as nd
import shading
from assets import AssetCache
from look import Look

log = logging.getLogger(__name__)

SSS_RADIUS_RGB = (1.0, 0.55, 0.3)  # red scatters furthest, as in most organic and plastic media
CRUST_DARK = (0.09, 0.035, 0.012, 1.0)  # linear; deep-baked crust
CRUST_PALE = (0.62, 0.36, 0.13, 1.0)  # linear; golden blisters
CRUMB_SHADOW = (0.42, 0.33, 0.2, 1.0)  # linear; inside pores


def _rgba(c) -> tuple:
    lin = motion.srgb_to_linear(list(c)[:3])
    return (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)


class MaterialLibrary:
    """Builds each named material once and hands out the same datablock afterwards."""

    def __init__(self, look: Look, assets: AssetCache, time_to_frame: Callable[[float], float] | None = None):
        self.look = look
        self.assets = assets
        self.time_to_frame = time_to_frame or (lambda t: t)
        self._cache: dict[str, bpy.types.Material] = {}

    def get(self, name: str) -> bpy.types.Material:
        """Material for a look material name or a preset name (with its time keys, if any)."""
        if name not in self._cache:
            spec = self.look.material_spec(name)
            self._cache[name] = build_material(name, spec, self.assets)
            matkeys.apply_keys(self._cache[name], spec, self.time_to_frame)
        return self._cache[name]

    def tinted(self, color: list | None, opacity: float = 1.0) -> bpy.types.Material:
        """The fallback plastic_matte tinted with a shape's colour (sRGB) and opacity."""
        rgb = [round(float(c), 3) for c in (color or [0.6, 0.6, 0.6])[:3]]
        key = f"plastic_matte_{rgb[0]:.3f}_{rgb[1]:.3f}_{rgb[2]:.3f}_{opacity:.2f}"
        if key not in self._cache:
            spec = self.look.material_spec("plastic_matte")
            spec.update(base_color=rgb, alpha=float(opacity), _name=key)
            self._cache[key] = build_material(key, spec, self.assets)
        return self._cache[key]

    def for_labels(self, labels: list[str], color=None, opacity: float = 1.0) -> bpy.types.Material:
        name = self.look.assigned(labels)
        return self.get(name) if name else self.tinted(color, opacity)


def _textures(spec: dict, assets: AssetCache) -> dict:
    if not spec.get("texture"):
        return {}
    files = assets.texture(spec["texture"], spec.get("res", "2k")) or {}
    out = {}
    for role, path in files.items():
        out[role] = shading.load_image(path, "sRGB" if role == "color" else "Non-Color")
    return out


def _mapping(spec: dict) -> str:
    """Own art (file: textures) is mapped by the mesh UVs unless the look says otherwise; CC0 sets are triplanar."""
    if "mapping" in spec.get("_overrides", []):
        return spec["mapping"]
    return "uv" if str(spec.get("texture") or "").startswith("file:") else spec.get("mapping", "triplanar")


def _planar(spec: dict) -> dict:
    """mapping "planar": project along planar_axis onto the plane through planar_origin, v along planar_up."""
    return {"axis": spec.get("planar_axis", (0, 0, 1)), "origin": spec.get("planar_origin", (0, 0, 0)),
            "up": spec.get("planar_up")}


def _decal(tree, coords: shading.Coords, image, color: tuple, spec: dict):
    """Texture over the base colour by the texture's alpha (a printed label or jacket print)."""
    if _mapping(spec) == "planar":
        col, alpha = shading.planar_image(coords, image, spec["scale_m"], _planar(spec), extension="CLIP")
    else:
        col, alpha = shading.uv_image(coords, image)
    mix = nd.node(tree, "ShaderNodeMix", data_type="RGBA", blend_type="MIX", clamp_result=True)
    nd.link(tree, alpha, mix.inputs[0])
    mix.inputs[6].default_value = color
    nd.link(tree, col, mix.inputs[7])
    return mix.outputs[2]


def _base_color(tree, spec: dict, coords: shading.Coords, tex: dict):
    color = _rgba(spec["base_color"])
    sock = None
    if "color" in tex and spec.get("texture_alpha"):
        sock = _decal(tree, coords, tex["color"], color, spec)
    elif "color" in tex:
        sock = shading.triplanar(coords, tex["color"], spec["scale_m"], _mapping(spec), _planar(spec))
        if spec.get("tex_tint"):
            lum = nd.vmath(tree, "DOT_PRODUCT", sock, shading.LUMA)
            sock = nd.vmath(tree, "SCALE", color[:3], scale=nd.math(tree, "DIVIDE", lum,
                                                                     shading.mean_luminance(tex["color"])))
        # Otherwise the texture replaces the base colour; every other preset value stays.
    detail = spec.get("detail")
    if detail == "crust":
        # Maillard browning: dark patches and pale golden blisters at two scales.
        burn = shading.noise(tree, coords.vector, 0.008, detail=6.0, roughness=0.6, distortion=0.4)
        sock = _ramp_mix(tree, sock if sock is not None else color, burn, 0.48, 0.72, CRUST_DARK)
        blister = shading.noise(tree, coords.vector, 0.003, detail=3.0, roughness=0.5)
        sock = _ramp_mix(tree, sock, blister, 0.6, 0.8, CRUST_PALE)
    elif detail == "crumb":
        pores = nd.math(tree, "SUBTRACT", 1.0, shading.crumb(tree, coords.vector))
        sock = _ramp_mix(tree, sock if sock is not None else color, pores, 0.2, 0.9, CRUMB_SHADOW)
    tint = float(spec.get("tint_var", 0.0))
    if tint > 0.0:
        fac = shading.noise(tree, coords.vector, 0.03, detail=3.0)
        per_object = nd.node(tree, "ShaderNodeObjectInfo").outputs["Random"]  # differs per instance (grains)
        spread = nd.math(tree, "ADD", shading.signed(tree, fac, tint), shading.signed(tree, per_object, 0.6 * tint))
        gain = nd.math(tree, "ADD", spread, 1.0)
        sock = nd.vmath(tree, "SCALE", sock if sock is not None else color[:3], scale=gain)
    gain = float(spec.get("color_gain", 1.0))  # exposure-style trim of the albedo: 0.707 = -0.5 stop
    if gain != 1.0:
        sock = nd.vmath(tree, "SCALE", sock if sock is not None else color[:3], scale=gain)
    return sock if sock is not None else color


def _ramp_mix(tree, sock, fac, lo: float, hi: float, color: tuple):
    """Blend toward `color` as fac goes from lo to hi (clamped)."""
    ramp = nd.node(tree, "ShaderNodeMapRange", {"From Min": lo, "From Max": hi}, clamp=True)
    nd.link(tree, fac, ramp.inputs["Value"])
    mix = nd.node(tree, "ShaderNodeMix", data_type="RGBA", blend_type="MIX", clamp_result=True)
    nd.link(tree, ramp.outputs[0], mix.inputs[0])
    nd.feed(tree, mix.inputs[6], sock)
    mix.inputs[7].default_value = color
    return mix.outputs[2]


def _roughness(tree, spec: dict, coords: shading.Coords, tex: dict):
    explicit = "roughness" in spec.get("_overrides", [])
    if "roughness" in tex and not explicit:
        img = shading.triplanar(coords, tex["roughness"], spec["scale_m"], _mapping(spec), _planar(spec))
        rough = nd.math(tree, "MULTIPLY", nd.vmath(tree, "DOT_PRODUCT", img, shading.THIRDS),
                        float(spec.get("roughness_gain", 1.0)))
    else:
        rough = float(spec["roughness"])
    if spec.get("detail") == "brushed":
        streak = shading.brushed(tree, coords.vector)
        rough = shading.vary(tree, rough, streak, 0.35)
    elif spec.get("detail") == "scratches":
        scr = shading.scratches(tree, coords.vector)
        rough = nd.math(tree, "MINIMUM", nd.math(tree, "MULTIPLY_ADD", scr, 0.3, rough), 1.0)
    fac = shading.noise(tree, coords.vector, 0.02, detail=5.0, roughness=0.55)
    rough = shading.vary(tree, rough, fac, float(spec.get("rough_var", 0.0)))
    return _imperfection(tree, spec, coords, rough, assets=spec.get("_assets"))


def _imperfection(tree, spec: dict, coords: shading.Coords, rough, assets: AssetCache | None):
    """Roughness pulled toward a smudge / fingerprint / scratch map (materials.<m>.imperfection) at its scale."""
    if not spec.get("imperfection") or assets is None:
        return rough
    files = assets.texture(spec["imperfection"], spec.get("res", "2k")) or {}
    path = files.get("roughness") or files.get("color")
    if path is None:
        return rough
    img = shading.load_image(path, "Non-Color")
    lum = nd.vmath(tree, "DOT_PRODUCT", shading.triplanar(coords, img, float(spec.get("imperfection_scale_m", 0.1))),
                   shading.THIRDS)
    mix = nd.node(tree, "ShaderNodeMix", data_type="FLOAT", clamp_factor=True)
    mix.inputs[0].default_value = float(spec.get("imperfection_strength", 0.3))
    nd.feed(tree, mix.inputs[2], rough)
    nd.link(tree, lum, mix.inputs[3])
    return mix.outputs[0]


def _height(tree, spec: dict, coords: shading.Coords, tex: dict):
    """Height [m] for the Bump node: texture displacement plus procedural micro detail (None if flat)."""
    terms = []
    if "height" in tex and spec.get("bump_m"):
        img = shading.triplanar(coords, tex["height"], spec["scale_m"], _mapping(spec), _planar(spec))
        h = nd.vmath(tree, "DOT_PRODUCT", img, shading.THIRDS)
        terms.append(nd.math(tree, "MULTIPLY", h, float(spec["bump_m"])))
    micro = float(spec.get("micro_bump_m") or 0.0)
    detail = spec.get("detail")
    if micro > 0.0:
        if detail == "crumb":
            fac = shading.crumb(tree, coords.vector)
        elif detail == "brushed":
            fac = shading.brushed(tree, coords.vector)
        elif detail == "crust":
            fac = shading.noise(tree, coords.vector, 0.004, detail=8.0, roughness=0.7, distortion=0.3)
        else:
            fac = shading.noise(tree, coords.vector, max(micro * 40.0, 0.0004), detail=6.0, roughness=0.6)
        terms.append(nd.math(tree, "MULTIPLY", fac, micro))
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else nd.math(tree, "ADD", terms[0], terms[1])


def _set_bsdf(tree, bsdf, spec: dict) -> None:
    inp = bsdf.inputs
    inp["Metallic"].default_value = float(spec["metallic"])
    inp["IOR"].default_value = float(spec["ior"])
    inp["Alpha"].default_value = float(spec.get("alpha", 1.0))
    inp["Transmission Weight"].default_value = float(spec["transmission"])
    inp["Specular IOR Level"].default_value = float(spec.get("specular", 0.5))
    inp["Subsurface Weight"].default_value = float(spec["subsurface"])
    inp["Subsurface Radius"].default_value = tuple(float(x) for x in spec.get("subsurface_rgb", SSS_RADIUS_RGB))
    inp["Subsurface Scale"].default_value = float(spec["subsurface_radius_m"])
    inp["Sheen Weight"].default_value = float(spec["sheen"])
    inp["Sheen Roughness"].default_value = float(spec.get("sheen_roughness", 0.5))
    inp["Coat Weight"].default_value = float(spec["coat"])
    inp["Coat Roughness"].default_value = float(spec.get("coat_roughness", 0.05))
    inp["Anisotropic"].default_value = float(spec["anisotropic"])
    inp["Anisotropic Rotation"].default_value = float(spec.get("anisotropic_rotation", 0.0))
    _set_emission(bsdf, spec)


def _set_emission(bsdf, spec: dict) -> None:
    em = spec.get("emission")
    if em is None or em == 0:
        return
    if isinstance(em, (list, tuple)):
        bsdf.inputs["Emission Color"].default_value = _rgba(em)
        bsdf.inputs["Emission Strength"].default_value = float(spec.get("emission_strength") or 1.0)
    else:
        bsdf.inputs["Emission Color"].default_value = _rgba(spec["base_color"])
        bsdf.inputs["Emission Strength"].default_value = float(em)


def _volume(tree, out, spec: dict) -> None:
    """Volume Absorption inside a transmissive surface: the liquid's colour deepens with thickness."""
    if not spec.get("absorption_color") or not spec.get("absorption_m"):
        return
    color, density = motion.absorption_density(spec["absorption_color"], float(spec["absorption_m"]))
    vol = nd.node(tree, "ShaderNodeVolumeAbsorption", {"Density": density})
    vol.inputs["Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    nd.link(tree, vol.outputs[0], out.inputs["Volume"])


def build_material(name: str, spec: dict, assets: AssetCache) -> bpy.types.Material:
    """A Principled BSDF material with variation, edge bevel, bump and (for liquids/glass) volume absorption."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = nd.node(tree, "ShaderNodeOutputMaterial")
    bsdf = nd.node(tree, "ShaderNodeBsdfPrincipled")
    nd.link(tree, bsdf.outputs[0], out.inputs["Surface"])
    coords = shading.Coords(tree)
    spec = {**spec, "_assets": assets}
    tex = _textures(spec, assets)
    nd.feed(tree, bsdf.inputs["Base Color"], _base_color(tree, spec, coords, tex))
    nd.feed(tree, bsdf.inputs["Roughness"], _roughness(tree, spec, coords, tex))
    _set_bsdf(tree, bsdf, spec)
    normal = shading.bump_chain(tree, _height(tree, spec, coords, tex), spec["bevel_m"] if spec.get("hard") else None)
    if normal is not None:
        nd.link(tree, normal, bsdf.inputs["Normal"])
        nd.link(tree, normal, bsdf.inputs["Coat Normal"])
    _volume(tree, out, spec)
    mat.diffuse_color = _rgba(spec["base_color"])
    if hasattr(mat.cycles, "homogeneous_volume"):
        mat.cycles.homogeneous_volume = True
    log.debug("material %s from preset values %s", name, {k: spec[k] for k in ("roughness", "metallic")})
    return mat

