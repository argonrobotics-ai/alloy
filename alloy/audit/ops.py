# SPDX-License-Identifier: Apache-2.0
"""``audit`` for the ``alloy`` command line and the MCP tool server (an ``alloy.cli.spec.Op``; JSON in and out).

Registered in ``alloy.cli.registry.OPS`` by its provider list. Example::

    alloy audit my_scenes/scenes/scene.yaml --rounds 3 --episodes 32 --library my_scenes
"""

from __future__ import annotations

from typing import Any

from alloy.cli.spec import Op, Param


def op_audit(session: Any, a: dict[str, Any]) -> dict[str, Any]:
    """Audit a scene; returns ``AuditReport.to_dict()`` (findings by severity with replays and fixes)."""
    from alloy.audit.run import audit  # noqa: PLC0415
    from alloy.cli.ops_tools import scene_card  # noqa: PLC0415

    with scene_card(a["scene"], a.get("library")) as (target, roots):
        report = audit(target, rounds=a.get("rounds") or 3, episodes=a.get("episodes") or 16, seed=a.get("seed") or 0,
                       device=a.get("device"), worlds=a.get("worlds"), seconds=a.get("seconds"),
                       library=list(roots or ()), bully=a.get("bully", True) is not False)
    return report.to_dict()


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
     Param("bully", "boolean", "run the bully agent too", default=True)),
    op_audit,
    "Audit a scene the way a reviewer would, but automatically and with numbers: alloy.check over `rounds` seeds "
    "(each finding with the share of rounds it showed in and the command that replays it) and the bully agent over "
    "`episodes` random legal episodes (class, first tick, body, replay handle). Findings come sorted by severity, "
    "each with a suggested fix; ok is false when any finding is an error.",
    {"scene": "demo_peg_d12", "rounds": 2, "episodes": 8}, read_only=True, engine=True),)

__all__ = ["OPS", "op_audit"]
