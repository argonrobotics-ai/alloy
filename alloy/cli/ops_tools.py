# SPDX-License-Identifier: Apache-2.0
"""Handlers of the tool operations: check, render, estimate, import, export, list.

They wrap ``alloy.check``, ``alloy.camera``, ``alloy.estimate`` and ``alloy.io`` (each imported on first use), and
take the same ``scene`` forms as every operation (``alloy.cli.scenes``). Example::

    from alloy.cli.session import Session
    from alloy.cli.ops_tools import op_list
    op_list(Session(), {"kind": "scenes", "match": "peg", "library": None, "limit": 50})
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from alloy.cli.errors import RequestError
from alloy.cli.ops_sim import check_view, now
from alloy.cli.scenes import check_per_world, to_scene
from alloy.cli.session import Session

LIST_KINDS = ("scenes", "objects", "materials", "robots", "adds", "ops")
#: scene files the card tools (check, export) cannot read directly: converted to cards first
CONVERT_SUFFIXES = (".xml", ".mjcf", ".urdf", ".usd", ".usda", ".usdc", ".json")
_PNG_COUNT = [0]


@contextlib.contextmanager
def scene_card(value: Any, library: list[str] | None) -> Iterator[tuple[str, list[str] | None]]:
    """``(scene for the card tools, library roots)``: library names and .yaml cards pass through; other forms
    (robot files, Scene JSON, recipes) are written as cards to a temporary folder for the call.

    Example: ``with scene_card({"objects": [...]}, None) as (path, lib): alloy.check.check(path, library=lib)``."""
    if isinstance(value, str) and Path(value).suffix.lower() not in CONVERT_SUFFIXES:
        yield value, library
        return
    scene = to_scene(value, library=library or ())
    with tempfile.TemporaryDirectory(prefix="alloy-cli-") as tmp:
        path = scene.save(Path(tmp))
        yield str(path), [tmp, *scene.roots]


def op_check(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Physics check: a scene (cards, overlap, a short run) or, with ``sim``, an open simulation as it is now."""
    if a["sim"] is not None or (a["scene"] is None and session.last is not None):
        _live_only(a)
        sid, sim = session.get(a)
        return {"sim": sid, "mode": "live", "t": now(sim), **sim.check()}
    if a["scene"] is None:
        raise RequestError("check needs a scene or an open simulation", 'pass scene="demo_peg_d12" (or a path or '
                           'recipe), or sim="<id>" from build', field="scene")
    from alloy.check.api import check  # noqa: PLC0415

    if _liquid_setting(a) != "default":                         # the check reads saved cards, which do not carry it
        raise RequestError("check runs the scene from its saved cards, which keep no liquid_accuracy",
                           "build it with --liquid-accuracy super, then check the open simulation (sim=<id>)",
                           field="liquid_accuracy")
    with scene_card(a["scene"], a["library"]) as (target, library):
        report = check(target, a["worlds"], a["seconds"], device=a["device"], run=not a["quick"],
                       expect_rest=not a["moving"], profile=a["profile"] or "standard", seed=a["seed"] or 0,
                       library=library, budget_s=a["budget"] if a["budget"] and a["budget"] > 0 else None,
                       contacts=a["contacts"])
    return {"mode": "quick" if a["quick"] else "run", **report.to_dict()}


#: check arguments that shape a check run (a scene), and their defaults: an open simulation is checked as it is
RUN_ONLY = {"scene": None, "worlds": None, "device": None, "seed": None, "profile": None, "contacts": None,
            "library": None, "liquid_accuracy": None, "seconds": None, "quick": False, "moving": False, "budget": 25.0}


def _live_only(a: dict[str, Any]) -> None:
    """Refuse run arguments on a check of an open simulation (it checks the run so far, at its own settings)."""
    given = sorted(k for k, default in RUN_ONLY.items() if a.get(k) not in (None, default))
    if given:
        raise RequestError(f"check of an open simulation takes no {given}: it checks that simulation as it is now",
                           "drop them (step or run the simulation first to check later states), or check the scene "
                           'itself without sim: {"op": "check", "scene": ..., "seconds": 1}', field=given[0])


def _liquid_setting(a: dict[str, Any]) -> str:
    """The setting a check would run: the flag, else the scene's own (Scene JSON, a .json file or a recipe carry it;
    library names and .yaml cards are always "default")."""
    if a["liquid_accuracy"] is not None:
        return a["liquid_accuracy"]
    scene = a["scene"]
    if isinstance(scene, dict) or (isinstance(scene, str) and scene.lower().endswith(".json")):
        return to_scene(scene, library=a["library"] or ()).liquid_accuracy
    return "default"


def op_render(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """A PNG of one world (after ``steps`` control periods), framed automatically unless ``look_at`` is given."""
    kw: dict[str, Any] = {k: a[k] for k in ("spp", "fov") if a[k] is not None}
    if a["look_at"] is not None:
        kw["look_at"] = _look_at(a["look_at"])            # checked before a scene is built or a step is taken
    sid, sim = session.get(a)
    check_view(sim, None, a["world"])
    if a["steps"]:
        sim.step(a["steps"])
    path = Path(a["out"]).expanduser() if a["out"] else _temp_png()
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sim.render(None, world=a["world"], width=a["width"], height=a["height"], path=str(path), **kw)
    return {"sim": sid, "path": str(path), "t": now(sim), "world": a["world"], "width": int(img.shape[1]),
            "height": int(img.shape[0]), "mean_rgb": img.reshape(-1, img.shape[-1])[:, :3].mean(axis=0).round(1)}


def _look_at(values: list[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(values) != 6:
        raise RequestError(f"look_at needs 6 numbers (eye x y z, target x y z), got {len(values)}",
                           "pass look_at=[1.2, -1.2, 0.9, 0, 0, 0.2]", field="look_at")
    return tuple(values[:3]), tuple(values[3:])


def _temp_png() -> Path:
    _PNG_COUNT[0] += 1
    return Path(tempfile.gettempdir()) / f"alloy-render-{os.getpid()}-{_PNG_COUNT[0]}.png"


def op_estimate(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Memory, step time, wall time and compile time of a run, before running it."""
    from alloy.api._estimate import estimate  # noqa: PLC0415

    scene = to_scene(a["scene"], library=a["library"] or (), liquid_accuracy=a["liquid_accuracy"])
    if a["worlds"] is not None:
        check_per_world(scene, a["worlds"], more=True)   # what a build of that many worlds would refuse
    return estimate(scene, a["worlds"], a["seconds"], a["device"], measure=a["measure"])


def op_import(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """MJCF / URDF / USD file -> object, material and scene cards in a library folder."""
    from alloy.io.importer import import_file  # noqa: PLC0415

    return import_file(a["file"], a["out"], name=a["name"]).to_json()


def op_export(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """A scene (any scene form) or a library object -> MJCF, URDF and USD files, each checked by loading it."""
    from alloy.io.api import FORMATS, export_object, export_scene  # noqa: PLC0415

    formats = tuple(a["formats"] or FORMATS)
    if (a["scene"] is None) == (a["object"] is None):
        raise RequestError("export needs exactly one of scene / object", 'pass scene="demo_peg_d12" or '
                           'object="cube_beech_30"', field="scene")
    if a["object"] is not None:                       # every --library folder is searched, in order
        return export_object(a["object"], a["out"], formats, library=a["library"] or None).to_json()
    with scene_card(a["scene"], a["library"]) as (target, library):
        return export_scene(target, a["out"], formats, library=library or None).to_json()


def op_list(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Names in the card library (scenes, objects, materials, robots), recipe add kinds, or the operations."""
    kind, match = a["kind"], (a["match"] or "").lower()
    items = _list_items(kind, a["library"] or ())
    if match:
        items = [it for it in items if match in (it if isinstance(it, str) else it["name"]).lower()]
    return {"kind": kind, "count": len(items), "items": items[: a["limit"]],
            "truncated": len(items) > a["limit"]}


def _list_items(kind: str, library: Any) -> list[Any]:
    from alloy.api import catalog  # noqa: PLC0415

    if kind == "materials":
        return catalog.materials(library)
    if kind in ("scenes", "objects"):
        return catalog.names(kind, library)
    if kind == "robots":                      # name, title and file: pass the name as a recipe robot's path
        from alloy.cli.robots import robots  # noqa: PLC0415

        return robots(library)
    if kind == "adds":
        from alloy.cli.scenes import describe_adders  # noqa: PLC0415

        return [{"name": k, "arguments": v} for k, v in describe_adders().items()]
    from alloy.cli.registry import OPS  # noqa: PLC0415

    return [{"name": op.name, "summary": op.summary} for op in OPS.values()]
