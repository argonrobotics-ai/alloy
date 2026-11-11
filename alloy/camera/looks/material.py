# SPDX-License-Identifier: Apache-2.0
"""Pack-v2 material trees for look materials, built node by node like ``alloy/render/blender/materials.py``.

Example::

    mat = material_tree("housing", {"preset": "abs_black", "roughness": 0.5})
    mat["tree"]["nodes"][1]["type"]            # 'ShaderNodeBsdfPrincipled'
    mat["principled"]["inputs"]["IOR"]         # {'value': 1.5299999713897705, 'linked': False}
"""
from __future__ import annotations

import logging
from pathlib import Path

from alloy.render.blender import motion

from . import shading as sh
from .nodes import Node, NodeTree, f32
from .spec import check_supported, resolve_spec
from .textures import image_entry, local_texture_files

log = logging.getLogger(__name__)

SSS_RADIUS_RGB = (1.0, 0.55, 0.3)  # red scatters furthest, as in most organic and plastic media
UNLINKED_NORMAL_SOCKETS = ("Normal", "Coat Normal")


def rgba(c) -> tuple:
    """Linear RGBA (alpha 1) of an sRGB colour: ``rgba([0.5, 0.5, 0.5])[0]`` is 0.2140..."""
    lin = motion.srgb_to_linear(list(c)[:3])
    return (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)


def _mapping(spec: dict) -> str:
    """Own art (file: textures) is mapped by the mesh UVs unless the look says otherwise; CC0 sets are triplanar."""
    if "mapping" in spec.get("_overrides", []):
        return spec["mapping"]
    return "uv" if str(spec.get("texture") or "").startswith("file:") else spec.get("mapping", "triplanar")


def _texture_images(spec: dict, files: dict | None) -> tuple[dict, dict]:
    """({role: texture key}, {key: {file, colorspace}}) for the spec's texture set (empty without a texture)."""
    if not spec.get("texture"):
        return {}, {}
    files = (files or {}).get(spec["texture"]) or local_texture_files(spec["texture"], spec.get("res", "2k"))
    roles, entries = {}, {}
    for role, path in files.items():
        key, entry = image_entry(role, Path(path))
        roles[role] = key
        entries[key] = entry
    return roles, entries


def _base_color(tree: NodeTree, spec: dict, coords: sh.Coords, tex: dict):
    color = rgba(spec["base_color"])
    sock = None
    if "color" in tex:  # the texture replaces the base colour; every other preset value stays
        sock = sh.triplanar(coords, tex["color"], spec["scale_m"], _mapping(spec))
    tint = float(spec.get("tint_var", 0.0))
    if tint > 0.0:
        fac = sh.noise(tree, coords.vector, 0.03, detail=3.0)
        per_object = tree.node("ShaderNodeObjectInfo").outputs["Random"]  # differs per instance (grains)
        spread = tree.math("ADD", sh.signed(tree, fac, tint), sh.signed(tree, per_object, 0.6 * tint))
        gain = tree.math("ADD", spread, 1.0)
        sock = tree.vmath("SCALE", sock if sock is not None else color[:3], scale=gain)
    gain = float(spec.get("color_gain", 1.0))  # exposure-style trim of the albedo
    if gain != 1.0:
        sock = tree.vmath("SCALE", sock if sock is not None else color[:3], scale=gain)
    return sock if sock is not None else color


def _roughness(tree: NodeTree, spec: dict, coords: sh.Coords, tex: dict):
    explicit = "roughness" in spec.get("_overrides", [])
    if "roughness" in tex and not explicit:
        img = sh.triplanar(coords, tex["roughness"], spec["scale_m"], _mapping(spec))
        rough = tree.math("MULTIPLY", tree.vmath("DOT_PRODUCT", img, sh.THIRDS), float(spec.get("roughness_gain", 1.0)))
    else:
        rough = float(spec["roughness"])
    if spec.get("detail") == "brushed":
        rough = sh.vary(tree, rough, sh.brushed(tree, coords.vector), 0.35)
    fac = sh.noise(tree, coords.vector, 0.02, detail=5.0, roughness=0.55)
    return sh.vary(tree, rough, fac, float(spec.get("rough_var", 0.0)))


def _height(tree: NodeTree, spec: dict, coords: sh.Coords, tex: dict):
    """Height [m] for the Bump node: texture displacement plus procedural micro detail (None if flat)."""
    terms = []
    if "height" in tex and spec.get("bump_m"):
        img = sh.triplanar(coords, tex["height"], spec["scale_m"], _mapping(spec))
        terms.append(tree.math("MULTIPLY", tree.vmath("DOT_PRODUCT", img, sh.THIRDS), float(spec["bump_m"])))
    micro = float(spec.get("micro_bump_m") or 0.0)
    if micro > 0.0:
        if spec.get("detail") == "brushed":
            fac = sh.brushed(tree, coords.vector)
        else:
            fac = sh.noise(tree, coords.vector, max(micro * 40.0, 0.0004), detail=6.0, roughness=0.6)
        terms.append(tree.math("MULTIPLY", fac, micro))
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else tree.math("ADD", terms[0], terms[1])


def _set_emission(tree: NodeTree, bsdf: Node, spec: dict) -> None:
    em = spec.get("emission")
    if em is None or em == 0:
        return
    if isinstance(em, (list, tuple)):
        tree.set_default(bsdf.inputs["Emission Color"], rgba(em))
        tree.set_default(bsdf.inputs["Emission Strength"], float(spec.get("emission_strength") or 1.0))
    else:
        tree.set_default(bsdf.inputs["Emission Color"], rgba(spec["base_color"]))
        tree.set_default(bsdf.inputs["Emission Strength"], float(em))


def _set_bsdf(tree: NodeTree, bsdf: Node, spec: dict) -> None:
    values = {
        "Metallic": float(spec["metallic"]), "IOR": float(spec["ior"]), "Alpha": float(spec.get("alpha", 1.0)),
        "Transmission Weight": float(spec["transmission"]), "Specular IOR Level": float(spec.get("specular", 0.5)),
        "Subsurface Weight": float(spec["subsurface"]),
        "Subsurface Radius": tuple(float(x) for x in spec.get("subsurface_rgb", SSS_RADIUS_RGB)),
        "Subsurface Scale": float(spec["subsurface_radius_m"]), "Sheen Weight": float(spec["sheen"]),
        "Sheen Roughness": float(spec.get("sheen_roughness", 0.5)), "Coat Weight": float(spec["coat"]),
        "Coat Roughness": float(spec.get("coat_roughness", 0.05)), "Anisotropic": float(spec["anisotropic"]),
        "Anisotropic Rotation": float(spec.get("anisotropic_rotation", 0.0)),
    }
    for name, value in values.items():
        tree.set_default(bsdf.inputs[name], value)
    _set_emission(tree, bsdf, spec)


def principled_digest(bsdf: Node) -> dict:
    """The ``principled`` summary of a pack material: distribution, subsurface method and {input: {value, linked}} of the enabled inputs."""
    inputs = {i["name"]: {"value": i["value"], "linked": i["linked"]}
              for i in bsdf.data["inputs"] if "value" in i and i["enabled"]}
    return {"distribution": bsdf.data["props"]["distribution"],
            "subsurface_method": bsdf.data["props"]["subsurface_method"], "inputs": inputs}


def _build_tree(spec: dict, tex: dict) -> tuple[NodeTree, Node]:
    tree = NodeTree()
    out = tree.node("ShaderNodeOutputMaterial")
    bsdf = tree.node("ShaderNodeBsdfPrincipled")
    tree.link(bsdf.outputs[0], out.inputs["Surface"])
    coords = sh.Coords(tree)
    tree.feed(bsdf.inputs["Base Color"], _base_color(tree, spec, coords, tex))
    tree.feed(bsdf.inputs["Roughness"], _roughness(tree, spec, coords, tex))
    _set_bsdf(tree, bsdf, spec)
    normal = sh.bump_chain(tree, _height(tree, spec, coords, tex), spec["bevel_m"] if spec.get("hard") else None)
    if normal is not None:
        for sid in UNLINKED_NORMAL_SOCKETS:
            tree.link(normal, bsdf.inputs[sid])
    return tree, bsdf


def material_tree(name: str, spec: dict | str | None = None, textures: dict | None = None) -> dict:
    """Pack-v2 material ``{name, use_nodes, diffuse_color, tree, principled, textures}`` for a look material.

    ``spec`` is a look file's ``materials`` entry (``{"preset": "abs_black", "roughness": 0.5, ...}``), a bare preset
    name, or None (``name`` is then a preset). ``textures`` maps a texture reference ("polyhaven:plywood") to
    ``{role: file}`` (roles color / roughness / height); without it the sets come from the local asset cache and a
    missing set raises CameraError. ``textures`` of the result is {key: {file, colorspace}} for the images used.

    Not mirrored (they raise CameraError): Voronoi / Map Range details (crust, crumb, scratches), decals, tex_tint,
    imperfection maps, planar mapping. Volume absorption and time keys are left out (the generator never reads them).

    Example::

        mat = material_tree("pad", "rubber_black")
        [n["type"] for n in mat["tree"]["nodes"][:2]]   # ['ShaderNodeOutputMaterial', 'ShaderNodeBsdfPrincipled']
    """
    spec = resolve_spec(name, spec)
    check_supported(name, spec)
    roles, entries = _texture_images(spec, textures)
    tree, bsdf = _build_tree(spec, roles)
    used = {n.data["image"] for n in tree.nodes.values() if "image" in n.data}
    log.debug("material %s: %d nodes, %d images", name, len(tree.nodes), len(used))
    return {"name": name, "use_nodes": True, "diffuse_color": _f32_list(rgba(spec["base_color"])),
            "tree": tree.to_dict(), "principled": principled_digest(bsdf),
            "textures": {k: v for k, v in entries.items() if k in used}}


def _f32_list(values) -> list:
    return [f32(v) for v in values]
