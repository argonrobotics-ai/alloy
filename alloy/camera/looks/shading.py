# SPDX-License-Identifier: Apache-2.0
"""Shader-network building blocks (mirror of ``alloy/render/blender/shading.py``): coordinates, triplanar images,
noise variation, brushed and scratch detail, bevel and bump. Only what the Warp generator can evaluate.

Example::

    tree = NodeTree()
    coords = Coords(tree)
    rough = vary(tree, 0.3, noise(tree, coords.vector, 0.02), 0.2)   # value * (1 + 0.2 * (2 * noise - 1))
"""
from __future__ import annotations

from .nodes import NodeTree, Socket

TRIPLANAR_SHARPNESS = 4  # weight = |n|^4, a soft blend between projections
LUMA = (0.2126, 0.7152, 0.0722)
THIRDS = (0.3333, 0.3333, 0.3334)


class Coords:
    """Metre-scale texture coordinates and the matching normal, shared by every map of one material.

    Rigid objects use object space; deforming surfaces carry ``rest_p`` / ``rest_n`` / ``has_rest`` attributes.
    """

    def __init__(self, tree: NodeTree):
        self.tree = tree
        tc = tree.node("ShaderNodeTexCoord")
        attr = {name: tree.node("ShaderNodeAttribute", attribute_name=name, attribute_type="GEOMETRY")
                for name in ("rest_p", "rest_n", "has_rest")}
        flag = attr["has_rest"].outputs["Fac"]
        self.vector = self._mix(flag, tc.outputs["Object"], attr["rest_p"].outputs["Vector"])
        self.normal = self._mix(flag, tc.outputs["Normal"], attr["rest_n"].outputs["Vector"])
        self.uv = tree.node("ShaderNodeAttribute", attribute_name="UVMap", attribute_type="GEOMETRY").outputs["Vector"]
        self._weights = None

    def _mix(self, fac: Socket, a: Socket, b: Socket) -> Socket:
        n = self.tree.node("ShaderNodeMix", data_type="VECTOR")
        self.tree.link(fac, n.inputs[0])
        self.tree.link(a, n.inputs[4])
        self.tree.link(b, n.inputs[5])
        return n.outputs[1]

    def weights(self) -> tuple:
        """Triplanar blend weights (wx, wy, wz) from the coordinate normal, summing to 1."""
        if self._weights is None:
            t = self.tree
            n = t.vmath("ABSOLUTE", self.normal)
            n2 = t.vmath("MULTIPLY", n, n)
            n4 = t.vmath("MULTIPLY", n2, n2) if TRIPLANAR_SHARPNESS == 4 else n2
            sep = t.node("ShaderNodeSeparateXYZ")
            t.link(n4, sep.inputs[0])
            total = t.math("ADD", t.math("ADD", sep.outputs[0], sep.outputs[1]), sep.outputs[2])
            total = t.math("MAXIMUM", total, 1e-6)
            self._weights = tuple(t.math("DIVIDE", sep.outputs[i], total) for i in range(3))
        return self._weights


def uv_image(coords: Coords, image: str) -> tuple:
    """(colour, alpha) sockets of texture ``image`` (a texture key) on the mesh UVs."""
    tex = coords.tree.node("ShaderNodeTexImage", image=image, interpolation="Cubic")
    coords.tree.link(coords.uv, tex.inputs["Vector"])
    return tex.outputs["Color"], tex.outputs["Alpha"]


def triplanar(coords: Coords, image: str, scale_m: float, mapping: str = "triplanar") -> Socket:
    """Colour socket of ``image`` tiled every ``scale_m`` metres by triplanar projection (or the mesh UVs)."""
    t = coords.tree
    if mapping == "uv":
        return uv_image(coords, image)[0]
    p = t.vmath("SCALE", coords.vector, scale=1.0 / max(scale_m, 1e-6))
    sep = t.node("ShaderNodeSeparateXYZ")
    t.link(p, sep.inputs[0])
    x, y, z = sep.outputs
    acc = None
    for (u, v), w in zip(((y, z), (x, z), (x, y)), coords.weights()):
        comb = t.node("ShaderNodeCombineXYZ")
        t.link(u, comb.inputs[0])
        t.link(v, comb.inputs[1])
        tex = t.node("ShaderNodeTexImage", image=image, interpolation="Cubic")
        t.link(comb, tex.inputs["Vector"])
        term = t.vmath("SCALE", tex.outputs["Color"], scale=w)
        acc = term if acc is None else t.vmath("ADD", acc, term)
    return acc


def noise(tree: NodeTree, vector, feature_m: float, detail: float = 4.0, roughness: float = 0.6,
          distortion: float = 0.0) -> Socket:
    """Fac (about 0..1, mean 0.5) of a 3D noise with features about ``feature_m`` metres across."""
    n = tree.node("ShaderNodeTexNoise", {"Scale": 1.0 / feature_m, "Detail": detail, "Roughness": roughness,
                                         "Distortion": distortion}, noise_dimensions="3D")
    tree.link(vector, n.inputs["Vector"])
    return n.outputs["Fac"]


def signed(tree: NodeTree, fac, amount: float) -> Socket:
    """``amount * (2 * fac - 1)``: a zero-mean variation of the given amplitude."""
    return tree.math("MULTIPLY", tree.math("MULTIPLY_ADD", fac, 2.0, -1.0), amount)


def vary(tree: NodeTree, value, fac, rel: float):
    """``value * (1 + rel * (2 * fac - 1))``, clamped at 0; ``value`` unchanged when ``rel <= 0``."""
    if rel <= 0.0:
        return value
    gain = tree.math("ADD", signed(tree, fac, rel), 1.0)
    return tree.math("MAXIMUM", tree.math("MULTIPLY", value, gain), 0.0)


def brushed(tree: NodeTree, vector, along_m: float = 0.02, across_m: float = 0.00004) -> Socket:
    """Fac of streaks running along object X, the classic brushed-metal grain."""
    stretch = tree.vmath("MULTIPLY", vector, (1.0 / along_m, 1.0 / across_m, 1.0 / across_m))
    n = tree.node("ShaderNodeTexNoise", {"Scale": 1.0, "Detail": 2.0, "Roughness": 0.5}, noise_dimensions="3D")
    tree.link(stretch, n.inputs["Vector"])
    return n.outputs["Fac"]


def bump_chain(tree: NodeTree, height, bevel_m: float | None):
    """Normal socket from a Bevel node (hard edges) followed by a Bump of ``height`` metres (None if neither)."""
    normal = None
    if bevel_m:
        normal = tree.node("ShaderNodeBevel", {"Radius": bevel_m}, samples=8).outputs["Normal"]
    if height is None:
        return normal
    bump = tree.node("ShaderNodeBump", {"Strength": 1.0, "Distance": 1.0})
    tree.link(height, bump.inputs["Height"])
    if normal is not None:
        tree.link(normal, bump.inputs["Normal"])
    return bump.outputs["Normal"]
