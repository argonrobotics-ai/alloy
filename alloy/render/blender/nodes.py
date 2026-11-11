# SPDX-License-Identifier: Apache-2.0
"""Small helpers to build shader and Geometry Nodes trees from Python."""

from __future__ import annotations

import bpy


def node(tree: bpy.types.NodeTree, typ: str, inputs: dict | None = None, **props) -> bpy.types.Node:
    """Add a node of bl_idname `typ`, set RNA properties, then input default values by socket name or index."""
    n = tree.nodes.new(typ)
    for key, value in props.items():
        setattr(n, key, value)
    for key, value in (inputs or {}).items():
        n.inputs[key].default_value = value
    return n


def link(tree: bpy.types.NodeTree, src, dst) -> None:
    """Link an output socket (or a node's first output) to an input socket."""
    if isinstance(src, bpy.types.Node):
        src = src.outputs[0]
    tree.links.new(src, dst)


def feed(tree: bpy.types.NodeTree, dst, value) -> None:
    """Connect a socket to dst, or set dst's default value when `value` is a constant."""
    if isinstance(value, (bpy.types.NodeSocket, bpy.types.Node)):
        link(tree, value, dst)
    else:
        dst.default_value = value


def math(tree: bpy.types.NodeTree, op: str, a, b=0.0, c=None) -> bpy.types.NodeSocket:
    """Scalar Math node `op` on sockets or constants; returns its output socket."""
    n = node(tree, "ShaderNodeMath", operation=op)
    feed(tree, n.inputs[0], a)
    feed(tree, n.inputs[1], b)
    if c is not None:
        feed(tree, n.inputs[2], c)
    return n.outputs[0]


def vmath(tree: bpy.types.NodeTree, op: str, a, b=None, scale=None) -> bpy.types.NodeSocket:
    """Vector Math node `op`; `scale` feeds the Scale input of SCALE; returns the vector (or value) output."""
    n = node(tree, "ShaderNodeVectorMath", operation=op)
    feed(tree, n.inputs[0], a)
    if b is not None:
        feed(tree, n.inputs[1], b)
    if scale is not None:
        feed(tree, n.inputs["Scale"], scale)
    return n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]


def gn_group(name: str, extra_inputs: tuple = ()) -> tuple[bpy.types.NodeTree, bpy.types.Node, bpy.types.Node]:
    """A Geometry Nodes group with Geometry in/out plus (name, socket_type, default) inputs."""
    g = bpy.data.node_groups.new(name, "GeometryNodeTree")
    g.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for sock_name, sock_type, default in extra_inputs:
        s = g.interface.new_socket(sock_name, in_out="INPUT", socket_type=sock_type)
        if default is not None:
            s.default_value = default
    g.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = g.nodes.new("NodeGroupInput")
    gout = g.nodes.new("NodeGroupOutput")
    return g, gin, gout


def gn_modifier(obj: bpy.types.Object, group: bpy.types.NodeTree, name: str = "alloy") -> bpy.types.Modifier:
    mod = obj.modifiers.new(name, "NODES")
    mod.node_group = group
    return mod


def named_attr(tree: bpy.types.NodeTree, name: str, data_type: str = "FLOAT_VECTOR") -> bpy.types.NodeSocket:
    n = node(tree, "GeometryNodeInputNamedAttribute", {"Name": name}, data_type=data_type)
    return n.outputs["Attribute"]


def sample_nearest_point(tree: bpy.types.NodeTree, points, value, data_type: str = "FLOAT_VECTOR"):
    """`value` (a field on `points`) at the point nearest each evaluated element."""
    near = node(tree, "GeometryNodeSampleNearest", domain="POINT")
    link(tree, points, near.inputs["Geometry"])
    link(tree, node(tree, "GeometryNodeInputPosition").outputs[0], near.inputs["Sample Position"])
    sample = node(tree, "GeometryNodeSampleIndex", data_type=data_type, domain="POINT")
    link(tree, points, sample.inputs["Geometry"])
    link(tree, value, sample.inputs["Value"])
    link(tree, near.outputs["Index"], sample.inputs["Index"])
    return sample.outputs[0]


def set_material_node(tree: bpy.types.NodeTree, geo, material: bpy.types.Material | None) -> bpy.types.NodeSocket:
    """Append a Set Material node after `geo` (no-op without a material)."""
    if material is None:
        return geo
    n = node(tree, "GeometryNodeSetMaterial")
    n.inputs["Material"].default_value = material
    link(tree, geo, n.inputs["Geometry"])
    return n.outputs["Geometry"]


def smooth_node(tree: bpy.types.NodeTree, geo) -> bpy.types.NodeSocket:
    n = node(tree, "GeometryNodeSetShadeSmooth")
    link(tree, geo, n.inputs["Geometry"])
    return n.outputs["Geometry"]


def solidify(obj, thickness: float, cfg: dict, even_default: bool) -> None:
    """Shell thickness without spikes at sharp folds.

    The offset is clamped to the local edge length (``thickness_clamp``, default 1). Even offset is opt-in for
    cloth and card (``even_offset``), because at a fold flattened to ~180 deg it divides by ~0 and throws spikes.
    """
    mod = obj.modifiers.new("solidify", "SOLIDIFY")
    mod.thickness, mod.offset = thickness, 0.0
    mod.use_even_offset = bool(cfg.get("even_offset", even_default))
    mod.use_quality_normals = True
    mod.thickness_clamp = float(cfg.get("thickness_clamp", 1.0))
    mod.use_thickness_angle_clamp = True

