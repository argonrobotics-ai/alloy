# SPDX-License-Identifier: Apache-2.0
"""The operations, their parameters and the one entry point every front end uses: ``respond(session, request)``.

``alloy <op> ...``, ``alloy batch`` and the MCP tool server all call ``respond``, so they take the same JSON and
give the same JSON::

    {"ok": true, "op": "step", "result": {...}}
    {"ok": false, "op": "step", "error": {"type": "SceneError", "message": "...", "fix": "...", "field": "size"}}

``ok`` says the call ran; a scene that fails its check, or a run with INVALID worlds, still has ``ok: true`` and says
so in ``result`` (``result.ok`` for check, ``result.health.ok`` for step and run). Example::

    from alloy.cli.registry import respond
    from alloy.cli.session import Session
    respond(Session(), {"op": "list", "kind": "scenes", "match": "peg"})
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from alloy.cli import ops_sim as S
from alloy.cli import ops_tools as T
from alloy.cli import ops_vary as V
from alloy.cli.errors import RequestError, error_payload
from alloy.cli.jsonio import plain
from alloy.cli.scenes import ADDERS
from alloy.cli.session import Session
from alloy.cli.spec import Op, Param, check_request

log = logging.getLogger("alloy.cli")

_RECIPE_SCHEMA = {
    "type": "object", "description": "a recipe: Scene.add_* calls as JSON",
    "properties": {"base": {"type": "string", "description": "start from this library scene or file"},
                   "worlds": {"type": "integer", "minimum": 1},
                   "dt": {"type": "number", "description": "control period, s"},
                   "liquid_accuracy": {"enum": ["default", "super"],
                                       "description": "liquid particles 4 mm apart (default) or 2 mm (super)"},
                   "materials": {"type": "array", "items": {"type": "object"}},
                   "objects": {"type": "array", "items": {
                       "type": "object", "required": ["add"],
                       "properties": {"add": {"enum": sorted(ADDERS)}, "name": {"type": "string"}},
                       "description": "keyword arguments of Scene.add_<add>: sizes in m (full edges), mass kg, "
                                      "pos [x, y, z], rpy radians, static true/false (alloy list adds)"}},
                   "friction": {"type": "array", "items": {"type": "object"}},
                   "randomize": {"type": "array", "items": {"type": "object"}}},
    "additionalProperties": False}

SCENE = Param("scene", "scene", positional=True,
              help='a library scene name (alloy list scenes), a .yaml/.json/.xml/.urdf/.usd path, Scene JSON,'
              ' or a recipe {"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": '
              '[0.1, 0.1, 0.1], "pos": [0, 0, 0.5], "mass": 1.0}]}',
              schema={"anyOf": [{"type": "string"}, _RECIPE_SCHEMA,
                                {"type": "object", "description": "Scene JSON (Scene.to_dict())"}]})
SIM = Param("sim", "string", "id of an open simulation (from build); default: build `scene`, else the last one built")
WORLDS = Param("worlds", "integer", "parallel worlds (default: the scene's)", minimum=1)
DEVICE = Param("device", "string", "'cpu', 'cuda' or 'cuda:<n>' (default: the GPU if there is one)")
SEED = Param("seed", "integer", "seed of randomized values (same seed, same worlds)")
PROFILE = Param("profile", "string", "integration profile", choices=("standard", "precision", "fast", "debug"))
CONTACTS = Param("contacts", "string", "contact pipeline", choices=("auto", "mujoco", "newton"))
LIBRARY = Param("library", "strings", "extra card library folders, searched first")
LIQUID = Param("liquid_accuracy", "string", "liquid particles: default 4 mm apart, super 2 mm (finer openings and "
               "streams, about 16x the cost); sets the scene's liquid_accuracy", choices=("default", "super"))
BUILD = (SCENE, WORLDS, DEVICE, SEED, PROFILE, CONTACTS, LIBRARY, LIQUID)
ON_SIM = (SIM, *BUILD)
NAMES = Param("names", "strings", "bodies, joints or particle objects to report (default: all)")
WORLD = Param("world", "world", "which world to report: an index or \"all\"", default=0)
POINTS = Param("points", "string", "cloth/liquid points: a summary or every point", default="summary",
               choices=("summary", "full"))
TARGETS = Param("targets", "object", 'joint position targets {"joint": value} (m or rad; a list gives one per world)')
KIND = Param("kind", "string", "target kind", default="position", choices=("position", "velocity"))


def _op(name: str, summary: str, params: tuple[Param, ...], handler: Any, example: dict[str, Any],
        description: str = "", *, read_only: bool = False, engine: bool = True, image: str = "") -> Op:
    return Op(name, summary, params, handler, description or summary, example, read_only, engine, image)


OPS: dict[str, Op] = {op.name: op for op in (
    _op("list", "List library scenes, objects or materials, recipe add kinds, or these operations.",
        (Param("kind", "string", "what to list", default="scenes", positional=True, choices=T.LIST_KINDS),
         Param("match", "string", "keep names containing this text"), LIBRARY,
         Param("limit", "integer", "at most this many items", default=200, minimum=1)),
        T.op_list, {"kind": "objects", "match": "peg"}, read_only=True, engine=False),
    _op("scene", "Check a scene description without building it (fast); optionally save it as JSON or cards.",
        (SCENE.req(), WORLDS, LIBRARY, Param("save", "string", "write the Scene JSON to this path"),
         Param("cards", "string", "write the engine's YAML cards to this folder")),
        S.op_scene, {"scene": {"objects": [{"add": "ground"}, {"add": "sphere", "name": "ball", "radius": 0.05,
                                                              "pos": [0, 0, 0.3]}]}}, read_only=True,
        engine=False),
    _op("build", "Build a scene into a simulation and keep it open; returns its id, names and solver routes.",
        (*(p.req() if p is SCENE else p for p in BUILD), Param("id", "string", "id for the simulation "
                                                                                 "(default sim<k>)")),
        S.op_build, {"scene": "demo_peg_d12", "worlds": 4},
        "Build a scene (first build of a new scene compiles kernels: about a minute). Later calls pass the "
        "returned id as sim. Names list every body, joint and particle object."),
    _op("step", "Advance a simulation some control periods; returns state and health.",
        (*ON_SIM, Param("steps", "integer", "control periods to advance", default=1, minimum=0), TARGETS, KIND,
         NAMES, WORLD, POINTS),
        S.op_step, {"sim": "sim1", "steps": 50, "names": ["cube"]}),
    _op("run", "Run some simulated seconds, recording frames; saves them to an .npz and samples a few.",
        (*ON_SIM, Param("seconds", "number", "simulated seconds", required=True, minimum=0),
         Param("targets", "object", 'joint targets held ({"j": 0.1}) or ramped ({"j": {"from": 0, "to": 0.1}})'),
         NAMES, Param("every", "integer", "record every n-th control period", default=1, minimum=1),
         Param("out", "string", "save the frames to this .npz path"),
         Param("samples", "integer", "frames of world `world` to include in the reply", default=5, minimum=0),
         WORLD, POINTS),
        S.op_run, {"scene": "demo_peg_d12", "seconds": 1.0, "names": ["peg_d12"], "out": "traj.npz"}),
    _op("state", "Read the current state of bodies, joints and particle objects.",
        (*ON_SIM, NAMES, WORLD, POINTS), S.op_state, {"sim": "sim1", "names": ["cube"], "world": 0}),
    _op("targets", "Set joint position or velocity targets (kept until changed).",
        (*ON_SIM, TARGETS.req(), KIND), S.op_targets, {"sim": "sim1", "targets": {"arm/lift": 0.1}}),
    _op("reset", "Put every world of a simulation back to its start state.", ON_SIM, S.op_reset, {"sim": "sim1"}),
    _op("close", "Close an open simulation.", (Param("sim", "string", "id to close (default: the last)"),),
        S.op_close, {"sim": "sim1"}, engine=False),
    _op("check", "Physics check of a scene (cards, overlaps, a short run) or of an open simulation now.",
        (*ON_SIM, Param("seconds", "number", "simulated seconds of the check run (default 2 GPU / 0.5 CPU)"),
         Param("quick", "boolean", "load checks only (cards, overlap), no run", default=False),
         Param("moving", "boolean", "the scene moves by design (drops, pours)", default=False),
         Param("budget", "number", "wall seconds the check run may take (0: no limit)", default=25.0)),
        T.op_check, {"scene": "demo_peg_d12", "quick": True},
        "Check a scene: card validation, overlap at load, rest drift, penetration, health events and energy "
        "drift, each finding with what to change. With sim (or after build) it checks the open simulation now.",
        read_only=True),
    _op("render", "Render a PNG of one world (auto-framed unless look_at is given).",
        (*ON_SIM, Param("steps", "integer", "control periods to advance first", default=0, minimum=0),
         Param("out", "string", "PNG path (default: a temporary file)"),
         Param("width", "integer", "pixels", default=640, minimum=16),
         Param("height", "integer", "pixels", default=480, minimum=16),
         Param("spp", "integer", "samples per pixel (more: less noise, slower)", minimum=1),
         Param("fov", "number", "vertical field of view, degrees"),
         Param("look_at", "numbers", "camera eye x, y, z then target x, y, z (m)"),
         Param("world", "integer", "world to render", default=0, minimum=0)),
        T.op_render, {"scene": "demo_peg_d12", "out": "peg.png", "width": 320, "height": 240}, image="path"),
    _op("estimate", "Estimate memory, step time and wall time of a run before running it.",
        (SCENE.req(), WORLDS, DEVICE, LIBRARY, LIQUID,
         Param("seconds", "number", "simulated seconds", default=1.0, minimum=0),
         Param("measure", "boolean", "time a short real run instead (compiles)", default=False)),
        T.op_estimate, {"scene": "demo_peg_d12", "worlds": 1024, "seconds": 2.0}, read_only=True),
    _op("import", "Import an MJCF, URDF or USD file into object, material and scene cards.",
        (Param("file", "string", "MJCF (.xml), URDF (.urdf) or USD file", required=True, positional=True),
         Param("out", "string", "library folder to write the cards to", required=True),
         Param("name", "string", "scene name (default: the file name)")),
        T.op_import, {"file": "robot.urdf", "out": "my_library"}),
    _op("export", "Export a scene or a library object to MJCF, URDF and USD (each checked by loading it).",
        (SCENE, Param("object", "string", "a library object name instead of a scene"),
         Param("out", "string", "output folder", required=True),
         Param("formats", "strings", "any of mjcf, urdf, usd (default: all)"), LIBRARY),
        T.op_export, {"scene": "demo_peg_d12", "out": "out/peg", "formats": ["mjcf", "usd"]}),
    _op("vary", "Make n variants of a scene (free objects moved, textures swapped) as Scene JSON files.",
        (SCENE.req(), Param("n", "integer", "variants to make", default=4, minimum=1),
         Param("pos", "number", "start jitter of each free object in x and y, +- m", default=0.005, minimum=0),
         Param("yaw", "number", "yaw jitter of each free object, +- degrees", default=0.0, minimum=0),
         Param("textures", "string", "texture swaps: any look with a family, textured looks only, or none",
               default="same-family", choices=V.TEXTURE_MODES),
         Param("tint", "number", "relative colour jitter (0: none)", default=0.05, minimum=0),
         Param("seed", "integer", "same seed, same variants", default=0, minimum=0),
         Param("settle", "number", "also build each variant on the CPU and run alloy.check this many seconds; "
               "redraw it on an error (about 2 s per variant)", minimum=0),
         Param("max_tries", "integer", "draws per variant before giving up", default=200, minimum=1),
         Param("out", "string", "folder for variant_000.json ... and variants.json (default: a new temporary "
               "folder)"),
         Param("sheet", "boolean", "also render contact_sheet.png with alloy.camera, one tile per variant",
               default=False),
         Param("sheet_options", "object", 'contact-sheet tiles {"width": 240, "height": 180, "spp": 4, "quality": '
               '"full" | "fast"}', schema=V.SHEET_SCHEMA), LIBRARY),
        V.op_vary, {"scene": "demo_peg_d12", "n": 4, "pos": 0.01, "yaw": 10, "seed": 3, "out": "variants",
                    "sheet": True},
        "Make n edited copies of one scene for visual and placement diversity, the same for the same seed: each free "
        "object (not robots, static, kinematic or held ones) moves within +-pos in x, y and +-yaw degrees, objects "
        "that touch move together, and a draw that overlaps or loses its support is redrawn; each look takes "
        "another texture of its family plus a small tint. Writes variant_000.json ... (pass one as scene to build, "
        "run or render) and variants.json (moves and textures); sheet adds contact_sheet.png.", image="sheet"),
)}


#: modules of other packages that add operations (SCR-OSS19-3): each defines ``OPS``, a tuple of ``alloy.cli.spec.Op``,
#: and stays light at import (stdlib + alloy.cli.spec; the engine is imported inside its handlers). One line each.
PROVIDERS = ("alloy.compose.ops", "alloy.create.ops", "alloy.audit.ops", "alloy.assets.hub.ops")


def add_providers(ops: dict[str, Op], modules: tuple[str, ...] = PROVIDERS) -> list[str]:
    """Add each provider module's ops to ``ops``; returns the names added. A provider this install lacks is skipped
    with a warning (the other commands keep working); a name already taken raises. Example: ``add_providers(OPS)``."""
    added = []
    for mod in modules:
        try:
            provided = importlib.import_module(mod).OPS
        except ImportError as exc:
            log.warning("operations of %s are not available: %s", mod, exc)
            continue
        for op in provided:
            if op.name in ops:
                raise RuntimeError(f"{mod}: operation {op.name!r} is already defined; rename it")
            ops[op.name] = op
            added.append(op.name)
    return added


add_providers(OPS)


def run(session: Session, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Run operation ``name`` with ``args`` (checked first); raises on errors. Example: ``run(s, "list", {})``."""
    op = OPS.get(name)
    if op is None:
        raise RequestError(f"unknown operation {name!r}", f"use one of {sorted(OPS)}", field="op")
    checked = check_request(op, args)
    if op.engine:
        _quiet_engine()
    return op.handler(session, checked)


def _quiet_engine() -> None:
    """Warp's banner and per-module load lines off stderr (agents read it too), unless ALLOY_VERBOSE=1."""
    try:
        from alloy.api._build import quiet_warp  # noqa: PLC0415

        quiet_warp()
    except ImportError:                       # no engine installed: the operation itself says what to install
        log.debug("warp not importable; logs stay as they are")


def respond(session: Session, request: dict[str, Any]) -> dict[str, Any]:
    """One request ``{"op": name, ...arguments}`` -> the reply envelope as JSON-ready data (never raises for a bad
    request).

    Example: ``respond(s, {"op": "scene", "scene": "demo_peg_d12"})["ok"]``."""
    name = request.get("op") if isinstance(request, dict) else None
    if not isinstance(request, dict) or not isinstance(name, str):
        exc = RequestError("a request is a JSON object with an \"op\"", '{"op": "list", "kind": "scenes"}', field="op")
        return {"ok": False, "op": name, "error": error_payload(exc)}
    return respond_op(session, name, {k: v for k, v in request.items() if k != "op"})


def respond_op(session: Session, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """``respond`` for an operation named apart from its arguments (the MCP tool name). Example:
    ``respond_op(s, "list", {"kind": "scenes"})``."""
    try:
        return {"ok": True, "op": name, "result": plain(run(session, name, args))}
    except Exception as exc:  # noqa: BLE001 - every failure becomes a reply with a fix
        return {"ok": False, "op": name, "error": error_payload(exc)}


def passed(reply: dict[str, Any]) -> bool:
    """True when the call ran and its result reports no failure (check ok, health ok). Example: exit codes."""
    if not reply.get("ok"):
        return False
    result = reply.get("result") or {}
    if "health" in result and not result["health"].get("ok", True):
        return False
    return bool(result.get("ok", True))
