# SPDX-License-Identifier: Apache-2.0
"""A worker process of ``alloy.build(..., device="cpu", processes=N)`` (CPU mode A).

The parent starts ``python -P -m alloy.parallel.cpu_worker`` with one end of a socket pair (fd in
``ALLOY_WORKER_FD``) and the mode A environment (``alloy.parallel.multigpu``: rank, world offset and count). The
worker builds its block of the scene's worlds with global world ids, replies with the facts the parent shows
(names, routes, dt), then serves ``Sim`` calls until the parent sends "close" or goes away. Replies are
``("ok", result)`` or ``("err", details)``; an Alloy API error travels as its ``to_dict()`` and is raised again,
same class, in the parent.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Any

log = logging.getLogger("alloy.parallel")

ENV_FD = "ALLOY_WORKER_FD"
# Sim methods the parent may call, and the facts it may read
CALLS = frozenset({"step", "run", "settle", "set_pose", "drive", "stop_drives", "reset", "state", "set_targets",
                   "set_controls", "faces", "particle_volume", "liquid_volume", "temperature", "heat", "set_heater",
                   "health", "check", "render", "report", "names", "routes"})
FACTS = frozenset({"t", "dt", "worlds", "physics_steps"})
STEPPERS = frozenset({"step", "settle"})   # these reply with the worker's physics step count (its own clock)


def describe(exc: BaseException) -> dict[str, Any]:
    """A picklable account of ``exc``: an Alloy API error's ``to_dict()``, else its type, text and traceback."""
    from alloy.api.errors import AlloyAPIError  # noqa: PLC0415

    if isinstance(exc, AlloyAPIError):
        return {"kind": "api", **exc.to_dict()}
    return {"kind": "other", "type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc))}


def default_pivot(sim: Any, name: str) -> Any:
    """The pivot ``Sim.drive`` takes for a kinematic fixture of several parts: their mean position in (global)
    world 0, which only rank 0 holds; None for a single part (it turns about its own origin in every world)."""
    import numpy as np  # noqa: PLC0415

    from alloy.api import _drive  # noqa: PLC0415

    parts = _drive.parts_of(sim, name)
    if len(parts) <= 1:
        return None
    st = sim.state(parts, world=0)
    return np.mean([st[f"{p}.pos"] for p in parts], axis=0)


def mpm_solvers(sim: Any) -> list[Any]:
    """The worker's implicit MPM solvers, in the stepper's order (the same in every worker)."""
    return [s for s in sim.engine.solvers if hasattr(s, "_mpm_model")]   # every entry, same order in every worker


def mpm_boxes(sim: Any) -> list[tuple[Any, Any, int] | None]:
    """Per MPM solver: its dense grid's start box over this block (``mpm_worlds.start_box``) and its particle count."""
    from alloy.integration.mpm_worlds import start_box  # noqa: PLC0415

    out: list[tuple[Any, Any, int] | None] = []
    for s in mpm_solvers(sim):
        box = start_box(s)
        out.append(None if box is None else (box[0], box[1], int(s.model.particle_count)))
    return out


def mpm_hold(sim: Any, boxes: list[tuple[Any, Any, int] | None], worlds: int) -> None:
    """Hold every dense MPM grid at the whole scene's box (``boxes``: union over workers, total particles)."""
    from alloy.integration.mpm_worlds import hold_box  # noqa: PLC0415

    for s, box in zip(mpm_solvers(sim), boxes):
        if box is not None:
            hold_box(s, box[0], box[1], worlds, box[2])


def build(job: dict[str, Any]) -> Any:
    """The worker's ``Sim``: global worlds ``job["block"]`` of ``job["scene"]`` (with ``job["worlds"]`` in all)."""
    from alloy.api._build import _contacts, build_block, quiet_warp  # noqa: PLC0415
    from alloy.api.scene import Scene  # noqa: PLC0415

    quiet_warp()
    scene = Scene.from_dict(job["scene"])
    native = _contacts(scene, job["contacts"])
    return build_block(scene, int(job["worlds"]), "cpu", int(job["seed"]), profile=(job["profile"], native),
                       plugins=list(job["plugins"]), block=tuple(job["block"]))


def handle(sim: Any, msg: tuple[Any, ...]) -> Any:
    """One request: ("call", method, args, kwargs), ("get", fact) or ("pivot", name)."""
    op = msg[0]
    if op == "call" and msg[1] in CALLS:
        out = getattr(sim, msg[1])(*msg[2], **msg[3])
        if msg[1] in STEPPERS:
            return int(sim.physics_steps)
        return None if out is sim else out
    if op == "get" and msg[1] in FACTS:
        return getattr(sim, msg[1])
    if op == "pivot":
        return default_pivot(sim, msg[1])
    if op == "mpm_boxes":
        return mpm_boxes(sim)
    if op == "mpm_hold":
        return mpm_hold(sim, *msg[1:])
    raise ValueError(f"unknown worker request {msg[:2]!r}")


def serve(conn: Connection) -> int:
    """Build, then answer requests until "close" (0) or the parent goes away (0); 1 when the build fails."""
    job = conn.recv()
    rank = int(job["rank"])
    if rank > 0:                                    # build-time warnings are the same in every worker: rank 0 says them
        logging.getLogger("alloy").setLevel(logging.ERROR)
    try:
        sim = build(job)
    except BaseException as exc:  # noqa: BLE001 - every failure goes back to the parent
        conn.send(("err", describe(exc)))
        return 1
    logging.getLogger("alloy.api").setLevel(logging.ERROR)  # run-time warnings name local worlds: the parent says them
    conn.send(("ok", {"worlds": sim.worlds, "dt": sim.dt, "names": sim.names(), "routes": sim.routes()}))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            log.debug("worker %d: the parent went away", rank)
            break
        if msg[0] == "close":
            sim.close()
            conn.send(("ok", None))
            break
        try:
            conn.send(("ok", handle(sim, msg)))
        except BaseException as exc:  # noqa: BLE001
            conn.send(("err", describe(exc)))
    return 0


def main() -> int:
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl-C goes to the parent, which stops the workers
    return serve(Connection(int(os.environ[ENV_FD])))


if __name__ == "__main__":
    sys.exit(main())
