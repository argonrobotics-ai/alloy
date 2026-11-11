# SPDX-License-Identifier: Apache-2.0
"""Time-keyed material values (materials.<m>.keys): Principled inputs keyframed at bundle time.

    "keys": [{"t": 0.0, "emission_strength": 0.0}, {"t": 0.731, "emission_strength": 12.0}], "interp": "constant"

Constant interpolation by default (an LED switches at the click), "linear" on request. Times are bundle seconds,
mapped to bundle frames through frames/t.npy, so they match the physics at any look.frames step.
"""

from __future__ import annotations

import logging
from typing import Callable

import bpy

import motion
from look import LookError

log = logging.getLogger(__name__)
SOCKETS = {"emission": "Emission Color", "emission_strength": "Emission Strength", "alpha": "Alpha",
           "transmission": "Transmission Weight", "metallic": "Metallic", "coat": "Coat Weight",
           "sheen": "Sheen Weight", "subsurface": "Subsurface Weight", "ior": "IOR", "specular": "Specular IOR Level",
           "anisotropic": "Anisotropic"}
INTERP = {"constant": "CONSTANT", "linear": "LINEAR"}


def _value(key: str, value):
    if key == "emission":
        lin = motion.srgb_to_linear(list(value)[:3])
        return (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)
    return float(value)


def _fcurves(action) -> list:
    """F-curves of an action (legacy list, or the layered actions of Blender 4.4+)."""
    try:
        return list(action.fcurves)
    except AttributeError:
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out += list(bag.fcurves)
        return out


def apply_keys(mat: bpy.types.Material, spec: dict, time_to_frame: Callable[[float], float]) -> None:
    """Keyframe the material's Principled BSDF inputs at each key's bundle time."""
    keys = spec.get("keys") or []
    if not keys:
        return
    bsdf = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    where = f"materials.{spec.get('_name')}.keys"
    for i, key in enumerate(sorted(keys, key=lambda k: float(k.get("t", 0.0)))):
        frame = time_to_frame(float(key.get("t", 0.0)))
        for name, value in key.items():
            if name in ("t", "interp"):
                continue
            if name not in SOCKETS:
                raise LookError(f"{where}[{i}].{name} cannot be keyed; keyable: {sorted(SOCKETS)}")
            sock = bsdf.inputs[SOCKETS[name]]
            sock.default_value = _value(name, value)
            sock.keyframe_insert("default_value", frame=frame)
    mode = INTERP.get(str(spec.get("interp", "constant")), "CONSTANT")
    for fc in _fcurves(mat.node_tree.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = mode
    log.info("material %s: %d keys (%s)", spec.get("_name"), len(keys), mode.lower())
