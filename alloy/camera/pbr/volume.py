# SPDX-License-Identifier: Apache-2.0
"""Material volumes the path tracer supports: a homogeneous Volume Absorption node on the Material Output's Volume
socket (Beer-Lambert colour through glass, water and clear plastic).

Cycles (kernel/svm/closure.h, svm_node_closure_volume) turns the node into an extinction coefficient
``sigma_a = (1 - Color) * Density`` per metre of world space (mesh objects have a volume density scale of 1). A ray
inside the object (entered through a transmission bounce) keeps ``exp(-sigma_a * distance)`` of its throughput.

Example::

    absorption({"tree": tree})     # {"sigma_a": [2.9, 0.93, 1.65]} or None without a volume shader
"""
from __future__ import annotations

from ..errors import CameraError


def _output_volume_link(tree: dict) -> dict | None:
    """The link feeding the active Material Output's Volume socket, if any."""
    outs = [n for n in tree.get("nodes", []) if n["type"] == "ShaderNodeOutputMaterial"]
    active = [n for n in outs if n.get("props", {}).get("is_active_output", True)] or outs
    if not active:
        return None
    name = active[0]["name"]
    return next((lk for lk in tree.get("links", []) if lk["to"] == name and lk["to_id"] == "Volume"
                 and not lk.get("muted", False)), None)


def absorption(mat: dict, name: str = "?") -> dict | None:
    """{"sigma_a": [r, g, b]} of the material's homogeneous absorption volume, None without a volume shader.

    Raises CameraError for any other volume shader (scatter, Principled Volume, emission) or a linked input."""
    tree = mat.get("tree") or {}
    link = _output_volume_link(tree)
    if link is None:
        return None
    node = next(n for n in tree["nodes"] if n["name"] == link["from"])
    if node["type"] != "ShaderNodeVolumeAbsorption":
        raise CameraError(f"material {name}: volume shader {node['type']}",
                          "use a Volume Absorption node (scattering and emitting volumes are not supported)")
    ins = {i["name"]: i for i in node["inputs"]}
    for key in ("Color", "Density", "Weight"):
        if ins.get(key, {}).get("linked"):
            raise CameraError(f"material {name}: Volume Absorption {key} is linked",
                              f"give the Volume Absorption node a constant {key} (homogeneous volumes only)")
    color = [float(c) for c in ins["Color"]["value"][:3]]
    density = max(float(ins["Density"]["value"]), 0.0)
    return {"sigma_a": [(1.0 - c) * density for c in color]}
