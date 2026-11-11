# SPDX-License-Identifier: Apache-2.0
"""``compose`` for the ``alloy`` command line and the MCP tool server (an ``alloy.cli.spec.Op``; JSON in and out).

Registered in ``alloy.cli.registry.OPS`` by its provider list (SCR-OSS19-3). Example::

    alloy compose "a mug, a spoon and three wooden cubes on a table" --variations 2 --out my_scenes
"""

from __future__ import annotations

from typing import Any

from alloy.cli.spec import Op, Param


def op_compose(session: Any, a: dict[str, Any]) -> dict[str, Any]:
    """Compose scenes; returns ``ComposeResult.to_dict()`` (+ ``saved`` paths with ``out``)."""
    from alloy.compose.build import compose  # noqa: PLC0415

    request = a.get("spec") if a.get("spec") is not None else a.get("sentence")
    result = compose(request, variations=a.get("variations") or 0, seed=a.get("seed") or 0,
                     check=a.get("check") or "full", device=a.get("device"), roots=a.get("library") or (),
                     robots=a.get("robots") or (), area=a.get("area"), jitter=a.get("jitter"),
                     scene_worlds=a.get("worlds"))
    out = result.to_dict()
    if a.get("out"):
        out["saved"] = [str(p) for p in result.save(a["out"])]
    return out


OPS: tuple[Op, ...] = (Op(
    "compose", "Build physics-checked scenes from a sentence or a JSON spec: objects, liquids, grains, cloth, cables.",
    (Param("sentence", "string", "the scene in words, e.g. 'three nuts in a tray on a cluttered table'",
           positional=True),
     Param("spec", "object", "a SceneSpec as JSON instead of a sentence (alloy.compose.parse gives one)"),
     Param("variations", "integer", "extra scenes with new poses and objects", default=0, minimum=0),
     Param("seed", "integer", "seed (same seed, same scene)", default=0),
     Param("check", "string", "full: load checks per placement + a full check; load; none", default="full",
           choices=("full", "load", "none")),
     Param("device", "string", "'cpu' or 'cuda:0' (default: the GPU if there is one)"),
     Param("library", "strings", "your card folders, searched first (e.g. from alloy create)"),
     Param("robots", "strings", "robot files (MJCF) to stand at the table edge"),
     Param("area", "numbers", "x0 y0 x1 y1 (m): where objects may go"),
     Param("jitter", "boolean", "each world moves the free objects a little"),
     Param("worlds", "integer", "parallel worlds in the saved scene cards", minimum=1),
     Param("out", "string", "folder to save scenes/<name>.yaml and compose.json")),
    op_compose,
    "Pick objects from the card library (Alloy's, alloy-assets, every imported library and your own folders), place "
    "them with support, clearance and clutter rules, run alloy.check after every placement and a full check at the "
    "end, and fix what the checks flag. Liquids and grains fill containers by volume or mass ('a mug with 200 ml of "
    "water'), pour as a stream, or pile at their angle of repose; cloth lies flat, drapes or hangs over the table "
    "edge; cables have a length, a gauge and ends plugged into objects. Returns every scene with its objects, soft "
    "items (amounts, routes, estimates), check report and metrics.",
    {"sentence": "a mug and two cubes on a table", "variations": 1, "out": "my_scenes"},
    read_only=False, engine=True),)

__all__ = ["OPS", "op_compose"]
