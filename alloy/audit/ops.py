# SPDX-License-Identifier: Apache-2.0
"""``audit`` for the ``alloy`` command line and the MCP tool server (an ``alloy.cli.spec.Op``; JSON in and out).

Registered in ``alloy.cli.registry.OPS`` by its provider list. Example::

    alloy audit my_scenes/scenes/scene.yaml --rounds 3 --episodes 32 --library my_scenes
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from alloy.cli.spec import Op, Param


def op_audit(session: Any, a: dict[str, Any]) -> dict[str, Any]:
    """Audit a scene; returns ``AuditReport.to_dict()`` (findings by severity with replays and fixes), plus
    ``cards``: the folder the scene's cards were written to when the scene was not a card already."""
    from alloy.audit.run import audit  # noqa: PLC0415

    target, roots, kept = scene_files(a["scene"], a.get("library"), a.get("cards"))
    report = audit(target, rounds=a.get("rounds") or 3, episodes=a.get("episodes") or 16, seed=a.get("seed") or 0,
                   device=a.get("device"), worlds=a.get("worlds"), seconds=a.get("seconds"),
                   library=list(roots or ()), bully=a.get("bully", True) is not False)
    out = report.to_dict()
    if kept is not None:
        out["cards"] = kept
    return out


def scene_files(value: Any, library: list[str] | None, cards: str | None = None
                ) -> tuple[str, list[str] | None, str | None]:
    """``(scene for the audit, card folders, folder written or None)``. Library names and .yaml cards pass through.
    A recipe, Scene JSON, a .json file or a robot file is written as cards to a folder that stays (``cards``, else
    one per scene under ``cache_dir()``), because every replay in the report reads those cards afterwards.

    Example: ``scene_files("demo_peg_d12", None) == ("demo_peg_d12", None, None)``."""
    from alloy.cli.ops_tools import CONVERT_SUFFIXES  # noqa: PLC0415
    from alloy.cli.scenes import to_scene  # noqa: PLC0415

    if isinstance(value, str) and Path(value).suffix.lower() not in CONVERT_SUFFIXES:
        return value, library, None
    scene = to_scene(value, library=library or ())
    if cards:
        folder = Path(cards).expanduser().resolve()
    else:
        text = json.dumps(scene.to_dict(), sort_keys=True, default=str)
        folder = cache_dir() / hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    folder.mkdir(parents=True, exist_ok=True)
    path = scene.save(folder)
    return str(path), [str(folder), *scene.roots], str(folder)


def cache_dir() -> Path:
    """Where audited scenes' cards are kept: ``$ALLOY_CACHE`` or ``~/.cache/alloy``, sub-folder ``audit``."""
    return Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy")).expanduser().resolve() / "audit"


OPS: tuple[Op, ...] = (Op(
    "audit", "Stress-test a scene: alloy.check over several seeds plus the bully agent's random legal episodes.",
    (Param("scene", "scene", "scene: a library name, a .yaml path, Scene JSON or a recipe", required=True,
           positional=True),
     Param("rounds", "integer", "alloy.check rounds, one seed each", default=3, minimum=1),
     Param("episodes", "integer", "bully agent episodes (random legal actuator commands or pushes)", default=16,
           minimum=1),
     Param("seed", "integer", "first seed", default=0),
     Param("worlds", "integer", "worlds per check round (default 64 GPU / 2 CPU)", minimum=1),
     Param("seconds", "number", "simulated seconds per check round and per bully episode (default: theirs)"),
     Param("device", "string", "'cpu' or 'cuda:0' (default: the GPU if there is one)"),
     Param("library", "strings", "extra card folders, searched first (e.g. the folder alloy create wrote to)"),
     Param("cards", "string", "folder to keep the scene's cards in when the scene is a recipe, Scene JSON, a .json "
           "file or a robot file (default: a folder per scene under ~/.cache/alloy/audit); the replays read it"),
     Param("bully", "boolean", "run the bully agent too", default=True)),
    op_audit,
    "Audit a scene the way a reviewer would, but automatically and with numbers: alloy.check over `rounds` seeds "
    "(each finding with the share of rounds it showed in and the command that replays it) and the bully agent over "
    "`episodes` random legal episodes (class, first tick, body, replay handle). Findings come sorted by severity, "
    "each with a suggested fix; ok is false when any finding is an error. A scene given as a recipe, Scene JSON, a "
    ".json file or a robot file is first written as cards to a folder that stays (`cards` in the reply), so the "
    "replays keep working after the call.",
    {"scene": "demo_peg_d12", "rounds": 2, "episodes": 8}, read_only=True, engine=True),)

__all__ = ["OPS", "cache_dir", "op_audit", "scene_files"]
