# SPDX-License-Identifier: Apache-2.0
"""Shader-network building blocks: texture coordinates, triplanar images, noise variation and surface detail."""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

import nodes as nd

TRIPLANAR_SHARPNESS = 4  # weight = |n|^4, a soft blend between projections
LUMA = (0.2126, 0.7152, 0.0722)
THIRDS = (0.3333, 0.3333, 0.3334)


class Coords:
    """Metre-scale texture coordinates and the matching normal, shared by every map of one material.

    Rigid objects use object space. Deforming surfaces carry `rest_p` / `rest_n` attributes (their first frame)
    and a `has_rest` flag, so textures stick to cloth instead of swimming through it.
    """

    def __init__(self, tree: bpy.types.NodeTree):
        self.tree = tree
        tc = nd.node(tree, "ShaderNodeTexCoord")
        rest_p = nd.node(tree, "ShaderNodeAttribute", attribute_name="rest_p", attribute_type="GEOMETRY")
        rest_n = nd.node(tree, "ShaderNodeAttribute", attribute_name="rest_n", attribute_type="GEOMETRY")
        flag = nd.node(tree, "ShaderNodeAttribute", attribute_name="has_rest", attribute_type="GEOMETRY")
        self.vector = self._mix(flag.outputs["Fac"], tc.outputs["Object"], rest_p.outputs["Vector"])
        self.normal = self._mix(flag.outputs["Fac"], tc.outputs["Normal"], rest_n.outputs["Vector"])
        uv = nd.node(tree, "ShaderNodeAttribute", attribute_name="UVMap", attribute_type="GEOMETRY")
        self.uv = uv.outputs["Vector"]  # mesh UV layers and GN-stored corner UVs alike
        self._weights = None

    def _mix(self, fac, a, b) -> bpy.types.NodeSocket:
        n = nd.node(self.tree, "ShaderNodeMix", data_type="VECTOR")
        nd.link(self.tree, fac, n.inputs[0])
        nd.link(self.tree, a, n.inputs[4])
        nd.link(self.tree, b, n.inputs[5])
        return n.outputs[1]

    def weights(self) -> tuple:
        """Triplanar blend weights (wx, wy, wz) from the coordinate normal, summing to 1."""
        if self._weights is None:
            t = self.tree
            n = nd.vmath(t, "ABSOLUTE", self.normal)
            n2 = nd.vmath(t, "MULTIPLY", n, n)
            n4 = nd.vmath(t, "MULTIPLY", n2, n2) if TRIPLANAR_SHARPNESS == 4 else n2
            sep = nd.node(t, "ShaderNodeSeparateXYZ")
            nd.link(t, n4, sep.inputs[0])
            total = nd.math(t, "ADD", nd.math(t, "ADD", sep.outputs[0], sep.outputs[1]), sep.outputs[2])
            total = nd.math(t, "MAXIMUM", total, 1e-6)
            self._weights = tuple(nd.math(t, "DIVIDE", sep.outputs[i], total) for i in range(3))
        return self._weights


def load_image(path: Path, colorspace: str) -> bpy.types.Image:
    """Image with the given colour space; colour EXRs are scene-linear."""
    img = bpy.data.images.load(str(path), check_existing=True)
    linear = colorspace == "sRGB" and path.suffix.lower() == ".exr"
    img.colorspace_settings.name = "Linear Rec.709" if linear else colorspace
    img.alpha_mode = "STRAIGHT"
    return img


def mean_luminance(image: bpy.types.Image, stride: int = 16) -> float:
    """Mean linear luminance of an image (every `stride`-th pixel), for normalising tex_tint textures."""
    px = np.empty(image.size[0] * image.size[1] * image.channels, np.float32)
    image.pixels.foreach_get(px)
    rgb = px.reshape(-1, image.channels)[::stride, :3]
    return max(float((rgb @ np.asarray(LUMA, np.float32)).mean()), 1e-4)


def uv_image(coords: Coords, image: bpy.types.Image) -> tuple:
    """(colour, alpha) sockets of `image` on the mesh UVs."""
    tex = nd.node(coords.tree, "ShaderNodeTexImage", image=image, interpolation="Cubic")
    nd.link(coords.tree, coords.uv, tex.inputs["Vector"])
    return tex.outputs["Color"], tex.outputs["Alpha"]


def planar_basis(axis, up=None) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) unit vectors spanning the plane perpendicular to `axis`; v follows `up` projected into the plane."""
    n = np.asarray(axis, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    up = np.asarray(up if up is not None else ([0, 0, 1] if abs(n[2]) < 0.9 else [0, 1, 0]), float)
    v = up - up.dot(n) * n
    if np.linalg.norm(v) < 1e-9:
        raise ValueError(f"planar_up {up.tolist()} is parallel to planar_axis {n.tolist()}")
    v = v / np.linalg.norm(v)
    return np.cross(v, n), v


def planar_image(coords: Coords, image: bpy.types.Image, scale_m: float, planar: dict, extension: str = "EXTEND"):
    """(colour, alpha) sockets of `image` projected along planar['axis'] onto the plane through planar['origin']
    (metres, rest/object space): centred on the origin, scale_m across, v along planar['up']. EXTEND repeats the edge
    pixels beyond the image; CLIP leaves them transparent (decals)."""
    t = coords.tree
    u_dir, v_dir = planar_basis(planar.get("axis") or (0, 0, 1), planar.get("up"))
    origin = np.asarray(planar.get("origin") or (0, 0, 0), float)
    inv = 1.0 / max(scale_m, 1e-6)
    comb = nd.node(t, "ShaderNodeCombineXYZ")
    for i, d in enumerate((u_dir, v_dir)):
        dot = nd.vmath(t, "DOT_PRODUCT", coords.vector, tuple(float(x) for x in d))
        nd.link(t, nd.math(t, "MULTIPLY_ADD", dot, inv, 0.5 - float(origin.dot(d)) * inv), comb.inputs[i])
    tex = nd.node(t, "ShaderNodeTexImage", image=image, interpolation="Cubic", extension=extension)
    nd.link(t, comb.outputs[0], tex.inputs["Vector"])
    return tex.outputs["Color"], tex.outputs["Alpha"]


def triplanar(coords: Coords, image: bpy.types.Image, scale_m: float, mapping: str = "triplanar",
              planar: dict | None = None):
    """Colour socket of `image` tiled every scale_m metres by triplanar projection (or the mesh UVs, or one planar
    projection with `planar`)."""
    t = coords.tree
    if mapping == "uv":
        return uv_image(coords, image)[0]
    if mapping == "planar":
        return planar_image(coords, image, scale_m, planar or {})[0]
    p = nd.vmath(t, "SCALE", coords.vector, scale=1.0 / max(scale_m, 1e-6))
    sep = nd.node(t, "ShaderNodeSeparateXYZ")
    nd.link(t, p, sep.inputs[0])
    x, y, z = sep.outputs
    acc = None
    for (u, v), w in zip(((y, z), (x, z), (x, y)), coords.weights()):
        comb = nd.node(t, "ShaderNodeCombineXYZ")
        nd.link(t, u, comb.inputs[0])
        nd.link(t, v, comb.inputs[1])
        tex = nd.node(t, "ShaderNodeTexImage", image=image, interpolation="Cubic")
        nd.link(t, comb, tex.inputs["Vector"])
        term = nd.vmath(t, "SCALE", tex.outputs["Color"], scale=w)
        acc = term if acc is None else nd.vmath(t, "ADD", acc, term)
    return acc


def noise(tree, vector, feature_m: float, detail: float = 4.0, roughness: float = 0.6, distortion: float = 0.0):
    """Fac (about 0..1, mean 0.5) of a 3D noise with features about feature_m metres across."""
    n = nd.node(tree, "ShaderNodeTexNoise", {"Scale": 1.0 / feature_m, "Detail": detail, "Roughness": roughness,
                                             "Distortion": distortion}, noise_dimensions="3D")
    nd.link(tree, vector, n.inputs["Vector"])
    return n.outputs["Fac"]


def signed(tree, fac, amount: float):
    """amount * (2 * fac - 1): a zero-mean variation of the given amplitude."""
    return nd.math(tree, "MULTIPLY", nd.math(tree, "MULTIPLY_ADD", fac, 2.0, -1.0), amount)


def vary(tree, value, fac, rel: float):
    """value * (1 + rel * (2 * fac - 1)), clamped at 0."""
    if rel <= 0.0:
        return value
    gain = nd.math(tree, "ADD", signed(tree, fac, rel), 1.0)
    return nd.math(tree, "MAXIMUM", nd.math(tree, "MULTIPLY", value, gain), 0.0)


def brushed(tree, vector, along_m: float = 0.02, across_m: float = 0.00004):
    """Fac of streaks running along object X, the classic brushed-metal grain."""
    stretch = nd.vmath(tree, "MULTIPLY", vector, (1.0 / along_m, 1.0 / across_m, 1.0 / across_m))
    n = nd.node(tree, "ShaderNodeTexNoise", {"Scale": 1.0, "Detail": 2.0, "Roughness": 0.5}, noise_dimensions="3D")
    nd.link(tree, stretch, n.inputs["Vector"])
    return n.outputs["Fac"]


def scratches(tree, vector, length_m: float = 0.01, width_m: float = 0.00003):
    """Fac near 1 on sparse fine scratches running along X and along Y (moulded plastic handling marks)."""
    out = None
    for stretch in ((1.0 / length_m, 1.0 / width_m, 1.0 / width_m), (1.0 / width_m, 1.0 / length_m, 1.0 / width_m)):
        n = nd.node(tree, "ShaderNodeTexNoise", {"Scale": 1.0, "Detail": 1.0, "Roughness": 0.5}, noise_dimensions="3D")
        nd.link(tree, nd.vmath(tree, "MULTIPLY", vector, stretch), n.inputs["Vector"])
        ramp = nd.node(tree, "ShaderNodeMapRange", {"From Min": 0.66, "From Max": 0.72}, clamp=True)
        nd.link(tree, n.outputs["Fac"], ramp.inputs["Value"])
        out = ramp.outputs[0] if out is None else nd.math(tree, "MAXIMUM", out, ramp.outputs[0])
    return out


def _voronoi_distance(tree, vector, cell_m: float):
    v = nd.node(tree, "ShaderNodeTexVoronoi", {"Scale": 1.0 / cell_m, "Randomness": 1.0}, feature="F1",
                voronoi_dimensions="3D")
    nd.link(tree, vector, v.inputs["Vector"])
    return v.outputs["Distance"]


def crumb(tree, vector, cell_m: float = 0.0014):
    """Fac near 0 inside crumb pores and 1 on the cell walls: two pore sizes on a domain-warped Voronoi."""
    warp = nd.math(tree, "SUBTRACT", noise(tree, vector, cell_m * 4.0, detail=3.0), 0.5)
    offset = nd.vmath(tree, "SCALE", (1.0, 0.8, 0.6), scale=nd.math(tree, "MULTIPLY", warp, 3.0 * cell_m))
    p = nd.vmath(tree, "ADD", vector, offset)
    small = nd.math(tree, "MULTIPLY", _voronoi_distance(tree, p, cell_m), 2.2)
    large = nd.math(tree, "MULTIPLY", _voronoi_distance(tree, p, 2.6 * cell_m), 2.0)
    return nd.math(tree, "SMOOTH_MIN", nd.math(tree, "MINIMUM", small, large), 1.0, 0.3)


def bump_chain(tree, height, bevel_m: float | None):
    """Normal socket from a Bevel node (hard edges) followed by a Bump of `height` metres (None if neither)."""
    normal = None
    if bevel_m:
        bev = nd.node(tree, "ShaderNodeBevel", {"Radius": bevel_m}, samples=8)
        normal = bev.outputs["Normal"]
    if height is None:
        return normal
    bump = nd.node(tree, "ShaderNodeBump", {"Strength": 1.0, "Distance": 1.0})
    nd.link(tree, height, bump.inputs["Height"])
    if normal is not None:
        nd.link(tree, normal, bump.inputs["Normal"])
    return bump.outputs["Normal"]
