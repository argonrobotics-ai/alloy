# SPDX-License-Identifier: Apache-2.0
"""A Blender-free node tree builder with the call shapes of Alloy's ``nodes.py`` (node, link, feed, math, vmath).

The tree is the pack-v2 dict ``{"nodes": [...], "links": [...]}``. Node names follow Blender's rule ("Math",
"Math.001", ...), socket values are rounded to float32 like Blender's stored defaults, and a node starts from the
templates in ``templates.py``.

Example::

    tree = NodeTree()
    out = tree.node("ShaderNodeOutputMaterial")
    bsdf = tree.node("ShaderNodeBsdfPrincipled")
    tree.link(bsdf.outputs[0], out.inputs["Surface"])
    tree.math("MULTIPLY", 2.0, 3.0)        # output Socket of a Math node with both operands set
    tree.to_dict()["links"][0]["to_id"]    # 'Surface'
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..errors import CameraError
from .templates import NODE_TEMPLATES

BASE_NAMES = {
    "ShaderNodeOutputMaterial": "Material Output", "ShaderNodeBsdfPrincipled": "Principled BSDF",
    "ShaderNodeTexCoord": "Texture Coordinate", "ShaderNodeAttribute": "Attribute", "ShaderNodeMix": "Mix",
    "ShaderNodeTexNoise": "Noise Texture", "ShaderNodeObjectInfo": "Object Info", "ShaderNodeMath": "Math",
    "ShaderNodeVectorMath": "Vector Math", "ShaderNodeBevel": "Bevel", "ShaderNodeBump": "Bump",
    "ShaderNodeSeparateXYZ": "Separate XYZ", "ShaderNodeCombineXYZ": "Combine XYZ", "ShaderNodeTexImage": "Image Texture",
}
THREE_OPERAND_MATH = frozenset({"MULTIPLY_ADD", "SMOOTH_MIN", "SMOOTH_MAX", "COMPARE", "WRAP", "PINGPONG"})
ONE_VECTOR_OPS = frozenset({"ABSOLUTE", "NORMALIZE", "LENGTH", "SCALE", "FLOOR", "CEIL", "FRACTION", "SINE", "COSINE"})


def f32(x) -> float:
    """``x`` rounded to float32 and back (what Blender stores): ``f32(0.03)`` is 0.029999999329447746."""
    return float(np.float32(x))


def _round_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [f32(c) for c in value]
    return f32(value)


@dataclass(frozen=True)
class Socket:
    """One socket of a node: ``node`` name, socket ``sid`` and the direction."""

    node: str
    sid: str
    is_output: bool = False


class SocketList:
    """The inputs or outputs of a node, indexable by position, identifier or name like ``node.inputs``."""

    def __init__(self, node: str, ids: list[str], names: list[str], is_output: bool):
        self._items = [Socket(node, sid, is_output) for sid in ids]
        self._names = names

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._items[key]
        for item, name in zip(self._items, self._names):
            if key in (item.sid, name):
                return item
        raise KeyError(f"no socket {key!r}; sockets: {[i.sid for i in self._items]}")

    def __iter__(self):
        return iter(self._items)


class Node:
    """A node of a ``NodeTree`` (its dict is ``self.data``)."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        ins = data["inputs"]
        self.inputs = SocketList(self.name, [i["id"] for i in ins], [i["name"] for i in ins], False)
        outs = data["_outputs"]
        self.outputs = SocketList(self.name, outs, outs, True)

    def socket_data(self, sid: str) -> dict:
        """The input dict with identifier ``sid``."""
        return next(i for i in self.data["inputs"] if i["id"] == sid)


class NodeTree:
    """The material node tree under construction."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.links: list[dict] = []
        self._counts: dict[str, int] = {}

    def _name(self, typ: str) -> str:
        base = BASE_NAMES[typ]
        n = self._counts.get(base, 0)
        self._counts[base] = n + 1
        return base if n == 0 else f"{base}.{n:03d}"

    def node(self, typ: str, inputs: dict | None = None, image: str | None = None, **props) -> Node:
        """Add a node of bl_idname ``typ``, set ``props``, then input defaults by identifier/name; ``image`` is a texture key."""
        if typ not in NODE_TEMPLATES:
            raise CameraError(f"node type {typ} has no template", "add it to alloy/camera/looks/templates.py")
        tpl = copy.deepcopy(NODE_TEMPLATES[typ])
        data = {"name": self._name(typ), "type": typ, "props": {**tpl["props"], **props},
                "inputs": [self._input(*i) for i in tpl["inputs"]], "_outputs": tpl["outputs"]}
        if "texture_mapping" in tpl:
            data["texture_mapping"] = tpl["texture_mapping"]
        if image is not None:
            data["image"] = image
        n = Node(data)
        self.nodes[n.name] = n
        _refresh_enabled(data)
        for key, value in (inputs or {}).items():
            self.set_default(n.inputs[key], value)
        return n

    @staticmethod
    def _input(sid, name, typ, enabled, value) -> dict:
        d = {"id": sid, "name": name, "type": typ, "linked": False, "enabled": enabled}
        if value is not None:
            d["value"] = _round_value(value)
        return d

    def set_default(self, sock: Socket, value) -> None:
        """Set an unlinked input's constant (float32-rounded; a vector keeps its component count)."""
        inp = self.nodes[sock.node].socket_data(sock.sid)
        inp["value"] = _round_value(value)

    def link(self, src, dst: Socket) -> None:
        """Link an output socket (or a node's first output) to an input socket."""
        if isinstance(src, Node):
            src = src.outputs[0]
        self.nodes[dst.node].socket_data(dst.sid)["linked"] = True
        self.links.append({"from": src.node, "from_id": src.sid, "to": dst.node, "to_id": dst.sid, "muted": False})

    def feed(self, dst: Socket, value) -> None:
        """Connect ``dst`` to a socket/node, or set its default when ``value`` is a constant."""
        if isinstance(value, (Socket, Node)):
            self.link(value, dst)
        else:
            self.set_default(dst, value)

    def math(self, op: str, a, b=0.0, c=None) -> Socket:
        """Scalar Math node ``op`` on sockets or constants; returns its output socket."""
        n = self.node("ShaderNodeMath", operation=op)
        self.feed(n.inputs[0], a)
        self.feed(n.inputs[1], b)
        if c is not None:
            self.feed(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op: str, a, b=None, scale=None) -> Socket:
        """Vector Math node ``op``; ``scale`` feeds the Scale input; returns the vector (or value) output."""
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(n.inputs[0], a)
        if b is not None:
            self.feed(n.inputs[1], b)
        if scale is not None:
            self.feed(n.inputs["Scale"], scale)
        return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]

    def to_dict(self) -> dict:
        """The pack-v2 tree ``{"nodes": [...], "links": [...]}`` (a copy)."""
        nodes = [{k: copy.deepcopy(v) for k, v in n.data.items() if k != "_outputs"} for n in self.nodes.values()]
        return {"nodes": nodes, "links": copy.deepcopy(self.links)}


def _refresh_enabled(data: dict) -> None:
    """Enabled flags that depend on the operation (Cycles greys out unused operands)."""
    op = data["props"].get("operation")
    ins = {i["id"]: i for i in data["inputs"]}
    if data["type"] == "ShaderNodeMath":
        ins["Value_002"]["enabled"] = op in THREE_OPERAND_MATH
    elif data["type"] == "ShaderNodeVectorMath":
        ins["Vector_001"]["enabled"] = op not in ONE_VECTOR_OPS
        ins["Scale"]["enabled"] = op == "SCALE"
