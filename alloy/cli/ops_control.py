# SPDX-License-Identifier: Apache-2.0
"""Handlers of the commands that act on an open simulation while it runs: drive, pose, wrench, settle, heater and
controls, one per ``alloy.Sim`` method (``drive`` / ``stop_drives``, ``set_pose``, ``apply_wrench``, ``settle``,
``set_heater``, ``set_controls``). Each is ``handler(session, args) -> dict`` with ``args`` checked against its spec
(``alloy.cli.registry``); arguments are checked before the simulation changes. Example::

    from alloy.cli.registry import respond
    from alloy.cli.session import Session
    s = Session()
    respond(s, {"op": "build", "scene": {"objects": [{"add": "ground"}, {"add": "box", "name": "belt", "size":
            [0.4, 0.2, 0.04], "pos": [0, 0, 0.02], "kinematic": True}]}, "device": "cpu"})
    respond(s, {"op": "drive", "name": "belt", "velocity": [0.25, 0, 0], "duration": 0.2})
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.cli.errors import RequestError
from alloy.cli.ops_sim import check_view, health_brief, now, state_view
from alloy.cli.session import Session


def _array(value: Any) -> Any:
    """JSON value -> API value: a list becomes a float array (one per world, or a vector), a number stays one."""
    if value is None or not isinstance(value, list):
        return value
    return np.asarray(value, dtype=float)


def op_drive(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Move a kinematic object (or fixture) at a constant velocity from now (``Sim.drive``); ``stop`` ends every
    drive (``Sim.stop_drives``)."""
    if a["stop"] == (a["name"] is not None):
        raise RequestError("drive needs a name (a kinematic object) or stop=true, not both",
                           'pass {"name": "belt", "velocity": [0.25, 0, 0]} or {"stop": true}', field="name")
    sid, sim = session.get(a)
    if a["stop"]:
        sim.stop_drives()
        return {"sim": sid, "t": now(sim), "stopped": True}
    kw = {k: _array(a[k]) for k in ("velocity", "angular_velocity", "pivot") if a[k] is not None}
    if a["duration"] is not None:
        kw["duration"] = a["duration"]
    sim.drive(a["name"], **kw)
    return {"sim": sid, "t": now(sim), "driving": a["name"], **kw}


def op_pose(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Teleport a free or kinematic body now, at rest (``Sim.set_pose``); returns its new pose."""
    if a["pos"] is None and a["quat_wxyz"] is None:
        raise RequestError("pose needs pos and/or quat_wxyz", 'pass {"name": "cube", "pos": [0, 0, 0.5]}',
                           field="pos")
    sid, sim = session.get(a)
    world = a["world"]
    check_view(sim, [a["name"]], 0 if world is None else world)
    sim.set_pose(a["name"], _array(a["pos"]), _array(a["quat_wxyz"]), world=world)
    return {"sim": sid, "t": now(sim), "posed": a["name"],
            "state": state_view(sim, [a["name"]], 0 if world is None else world)}


def op_wrench(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Hold a force (N) and torque (N m) on a body until changed; zeros remove it (``Sim.apply_wrench``)."""
    sid, sim = session.get(a)
    force, torque = _array(a["force"] or [0.0, 0.0, 0.0]), _array(a["torque"] or [0.0, 0.0, 0.0])
    sim.apply_wrench(a["name"], force, torque, at=a["at"])
    return {"sim": sid, "t": now(sim), "name": a["name"], "force": force, "torque": torque}


def op_settle(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Run ``seconds`` with no new commands and make the settled state the one reset returns to (``Sim.settle``)."""
    sid, sim = session.get(a)
    sim.settle(a["seconds"])
    return {"sim": sid, "t": now(sim), "health": health_brief(sim.health())}


def op_heater(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Change a heater's temperature (K) or power (W) while running (``Sim.set_heater``)."""
    if (a["temperature"] is None) == (a["power"] is None):
        raise RequestError("heater needs exactly one of temperature / power",
                           'pass {"name": "burner", "temperature": 370} (K) or {"name": "burner", "power": 500} (W)',
                           field="temperature")
    value = a["temperature"] if a["temperature"] is not None else a["power"]
    if not _numeric(value):
        raise RequestError(f"heater: {value!r} is not a number or a list of numbers"[:300],
                           'pass "temperature": 370 (K) or "power": [400, 500] (W, one per world)',
                           field="temperature" if a["temperature"] is not None else "power")
    sid, sim = session.get(a)
    sim.set_heater(a["name"], temperature=_array(a["temperature"]), power=_array(a["power"]))
    value = {k: a[k] for k in ("temperature", "power") if a[k] is not None}
    return {"sim": sid, "t": now(sim), "heater": a["name"], **value}


def op_controls(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Raw actuator inputs by actuator name (``Sim.set_controls``; names in the build reply's names.actuators)."""
    bad = [k for k, v in a["controls"].items() if not _numeric(v)]
    if bad:
        raise RequestError(f"controls {bad[:5]}: each value is a number or a list (one per world)",
                           'pass {"controls": {"arm/actuator1": 0.5}}', field=f"controls.{bad[0]}")
    sid, sim = session.get(a)
    sim.set_controls({k: _array(v) for k, v in a["controls"].items()})
    return {"sim": sid, "t": now(sim), "set": sorted(a["controls"])}


def _numeric(value: Any) -> bool:
    if isinstance(value, list):
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)
