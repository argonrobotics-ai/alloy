# SPDX-License-Identifier: Apache-2.0
"""Shape descriptions for a frame's topology part (SCR-WP13-EXPORT): every body's shapes in its body frame, and the
static shapes (ground, tables, bins) with their world pose, from Newton's shape arrays.

Sizes are full sizes, as in ``alloy.Scene`` (OSS-3): box ``size`` (full edges); sphere ``radius``; capsule, cylinder
and cone ``radius`` and ``height`` (a capsule's straight part between its caps, as ``add_capsule``); ellipsoid ``size``
(full axes); plane ``size`` (full width and length, 0 = infinite); mesh and convex mesh ``vertices`` (scaled) and
``faces``. Poses are ``pos`` and ``quat_xyzw``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

KINDS = {1: "plane", 2: "heightfield", 3: "sphere", 4: "capsule", 5: "ellipsoid", 6: "cylinder", 7: "box", 8: "mesh",
         9: "cone", 10: "convex_mesh", 11: "gaussian"}
VISIBLE = 1                                             # newton ShapeFlags.VISIBLE


def describe(kind: int, scale: np.ndarray, source: Any) -> dict[str, Any]:
    """One shape's kind and size fields (module docstring)."""
    name = KINDS.get(int(kind), f"type_{int(kind)}")
    s = [float(x) for x in scale]
    out: dict[str, Any] = {"kind": name}
    if name == "box":
        out["size"] = [2.0 * x for x in s]
    elif name == "sphere":
        out["radius"] = s[0]
    elif name in ("capsule", "cylinder", "cone"):
        out["radius"], out["height"] = s[0], 2.0 * s[1]
    elif name == "ellipsoid":
        out["size"] = [2.0 * x for x in s]
    elif name == "plane":
        out["size"] = s[:2]
    elif name in ("mesh", "convex_mesh") and source is not None:
        out["vertices"] = (np.asarray(source.vertices, np.float32) * np.asarray(s, np.float32)).astype(np.float32)
        out["faces"] = np.asarray(source.indices, np.int32).reshape(-1, 3)
    return out


def _posed(desc: dict[str, Any], xf: np.ndarray) -> dict[str, Any]:
    desc["pos"] = [float(x) for x in xf[:3]]
    desc["quat_xyzw"] = [float(x) for x in xf[3:7]]
    return desc


def model_shapes(model: Any) -> dict[str, np.ndarray]:
    """Host copies of the shape arrays the descriptions read (once per plan)."""
    return {"body": model.shape_body.numpy(), "type": model.shape_type.numpy(), "scale": model.shape_scale.numpy(),
            "xf": model.shape_transform.numpy(), "flags": model.shape_flags.numpy(),
            "world": model.shape_world.numpy() if getattr(model, "shape_world", None) is not None
            else np.full(model.shape_count, -1, np.int32)}


def body_shapes(model: Any, arrays: dict[str, np.ndarray], bodies: np.ndarray) -> list[list[dict[str, Any]]]:
    """Per body of ``bodies`` (global rows of one world), its visible shapes in the body frame."""
    by_body: dict[int, list[int]] = {}
    for s, b in enumerate(arrays["body"]):
        if b >= 0 and arrays["flags"][s] & VISIBLE:
            by_body.setdefault(int(b), []).append(s)
    return [[_posed(describe(arrays["type"][s], arrays["scale"][s], model.shape_source[s]), arrays["xf"][s])
             for s in by_body.get(int(b), [])] for b in bodies]


def statics(model: Any, arrays: dict[str, np.ndarray], world: int) -> list[dict[str, Any]]:
    """The visible static shapes world ``world`` sees (its own and the shared ones), each with its world pose."""
    out = []
    for s, b in enumerate(arrays["body"]):
        if b >= 0 or not arrays["flags"][s] & VISIBLE or int(arrays["world"][s]) not in (world, -1):
            continue
        label = str(model.shape_label[s]) if getattr(model, "shape_label", None) else f"shape_{s}"
        desc = describe(arrays["type"][s], arrays["scale"][s], model.shape_source[s])
        desc["name"] = label.rsplit("/", 1)[-1]
        out.append(_posed(desc, arrays["xf"][s]))
    return out


__all__ = ["KINDS", "body_shapes", "describe", "model_shapes", "statics"]
