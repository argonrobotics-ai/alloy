# SPDX-License-Identifier: Apache-2.0
"""Handlers of the simulation operations: scene, build, step, run, state, targets, reset, close.

Each is ``handler(session, args) -> dict`` with ``args`` already checked against its spec (``alloy.cli.registry``).
Arrays leave as NumPy; ``alloy.cli.jsonio.plain`` makes the reply JSON. Example::

    from alloy.cli.session import Session
    from alloy.cli.ops_sim import op_step
    op_step(Session(), {"scene": "demo_peg_d12", "steps": 10, ...})     # builds, steps, returns state + health
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api.errors import SimulationError
from alloy.cli.errors import RequestError
from alloy.cli.scenes import to_scene
from alloy.cli.session import Session

MAX_PROBLEMS = 3


# --- views of a simulation -------------------------------------------------------------------------------------

def now(sim: Any) -> float:
    """Simulated time without float summation noise (0.7, not 0.7000000000000001). Example: ``now(sim)``."""
    return round(float(sim.t), 9)


def health_brief(health: dict[str, Any], limit: int = MAX_PROBLEMS) -> dict[str, Any]:
    """Health without the per-world rows: counts, the first invalid world ids and up to ``limit`` problems with
    their fixes. Example: ``health_brief(sim.health())["ok"]``."""
    rows = health.get("worlds") or []
    bad = [(w, r) for w, r in enumerate(rows) if r.get("problem")]
    return {"ok": bool(health.get("ok", True)), "monitored": health.get("monitored", False),
            "invalid_worlds": len(health.get("invalid_worlds", [])),
            "degraded_worlds": len(health.get("degraded_worlds", [])),
            "first_invalid": list(health.get("invalid_worlds", []))[:10],
            "problems": [{"world": w, "severity": r["severity"], "problem": r["problem"], "fixes": r.get("fixes", [])}
                         for w, r in bad[:limit]]}


def state_view(sim: Any, names: list[str] | None, world: Any, points: str = "summary") -> dict[str, Any]:
    """``sim.state`` for one world (``world`` an int) or all ("all"); cloth and liquid points are summarised as
    count, centroid, lo and hi unless ``points="full"``. Example: ``state_view(sim, ["cube"], 0)["cube.pos"]``."""
    one = None if world == "all" else int(world)
    st = sim.state(names, world=one)
    out: dict[str, Any] = {}
    for key, value in st.items():
        if key == "t":
            continue                                  # the reply carries t once, beside the state
        if key.endswith(".points") and points != "full":
            base = key[: -len(".points")]
            arr = np.asarray(value)
            out[f"{base}.count"] = int(arr.shape[-2])
            out[f"{base}.centroid"] = arr.mean(axis=-2)
            out[f"{base}.lo"], out[f"{base}.hi"] = arr.min(axis=-2), arr.max(axis=-2)
        else:
            out[key] = value
    return out


def check_view(sim: Any, names: list[str] | None, world: Any) -> None:
    """Raise before anything changes when ``world`` is not a world of ``sim`` or ``names`` holds an unknown name, so
    a failed call leaves the simulation where it was. Example: ``check_view(sim, ["cube"], 0)``."""
    if world != "all" and not 0 <= int(world) < sim.worlds:
        raise SimulationError(f"world {world} is outside 0..{sim.worlds - 1}",
                              f'pass world 0 to {sim.worlds - 1}, or "all"', field="world")
    if names:
        sim.state(names)                              # an unknown name raises UnknownNameError listing the valid ones


def _targets(spec: dict[str, Any], ramps: bool) -> dict[str, Any]:
    """JSON targets -> API targets: a number or per-world list holds; ``{"from": a, "to": b}`` ramps (run only)."""
    out = {}
    for name, value in spec.items():
        if isinstance(value, dict):
            if not ramps or set(value) != {"from", "to"}:
                raise RequestError(f"target {name!r}: {value!r} is not a value", 'pass a number, a list (one per '
                                   'world), or {"from": a, "to": b} to ramp during run', field=f"targets.{name}")
            out[name] = (value["from"], value["to"])     # the API checks each value (a SimulationError naming it)
        elif isinstance(value, (int, float, list)) and not isinstance(value, bool):
            out[name] = value if isinstance(value, list) else float(value)
        else:
            raise RequestError(f"target {name!r} has a {type(value).__name__}", "pass a number (m or rad) or a "
                               "list with one per world", field=f"targets.{name}")
    return out


# --- handlers --------------------------------------------------------------------------------------------------

def op_scene(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Resolve a scene without building it (fast): names and kinds; optionally save it as JSON or cards."""
    scene = to_scene(a["scene"], worlds=a["worlds"], library=a["library"] or ())
    out: dict[str, Any] = {"summary": scene.summary(), "names": scene.names()}
    if a["save"]:
        path = Path(a["save"]).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(scene.to_json(), encoding="utf-8")
        out["saved"] = str(path)
    if a["cards"]:
        out["scene_card"] = str(scene.save(Path(a["cards"]).expanduser()))
    return out


def op_build(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Build a scene and keep it open under an id."""
    t0 = time.perf_counter()
    sid, sim = session.open(a, a["id"])
    out = sim_info(sid, sim)
    out["build_s"] = round(time.perf_counter() - t0, 3)
    return out


def sim_info(sid: str, sim: Any) -> dict[str, Any]:
    """Facts about an open simulation: id, worlds, dt, device, names and where each object was routed."""
    routes = {n: f"{r['route']} ({r['solver']})" for n, r in sim.routes().items()}
    return {"sim": sid, "worlds": sim.worlds, "dt": sim.dt, "device": sim.device, "profile": sim.profile,
            "t": now(sim), "names": sim.names(), "routes": routes, **liquid_setting(sim)}


def liquid_setting(sim: Any) -> dict[str, Any]:
    """``{"liquid_accuracy": ...}`` when the simulation holds a liquid (as ``traj.meta``), else ``{}``: a reply or a
    saved run names the particle spacing it ran at. Example: ``liquid_setting(sim)``."""
    scene = getattr(sim, "scene", None)
    if scene is None:
        return {}
    from alloy.api._liquid_accuracy import has_liquid  # noqa: PLC0415

    return {"liquid_accuracy": scene.liquid_accuracy} if has_liquid(scene.items) else {}


def op_step(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Advance ``steps`` control periods (after setting ``targets``) and report state and health."""
    sid, sim = session.get(a)
    check_view(sim, a["names"], a["world"])
    targets = _targets(a["targets"], ramps=False) if a["targets"] else None
    if targets:
        sim.set_targets(targets, a["kind"])
    t0 = time.perf_counter()
    sim.step(a["steps"])
    return {"sim": sid, "t": now(sim), "steps": a["steps"], "physics_steps": sim.physics_steps,
            "wall_s": round(time.perf_counter() - t0, 4), "health": health_brief(sim.health()),
            "state": state_view(sim, a["names"], a["world"], a["points"])}


def op_run(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Run ``seconds`` of simulated time, recording frames; save them to ``out`` (.npz) and sample a few."""
    sid, sim = session.get(a)
    check_view(sim, a["names"], a["world"])
    targets = _targets(a["targets"], ramps=True) if a["targets"] else None
    traj = sim.run(a["seconds"], record=a["names"], every=a["every"], targets=targets)
    out = {"sim": sid, "t": now(sim), "frames": traj.frames, "wall_s": round(traj.meta["wall_s"], 4),
           "physics_steps": traj.meta["physics_steps"], "health": health_brief(traj.meta["health"]),
           "keys": {k: list(np.shape(v)) for k, v in traj.items()},
           "samples": _samples(traj, a["samples"], 0 if a["world"] == "all" else int(a["world"])),
           "final": state_view(sim, a["names"], a["world"], a["points"]), **liquid_setting(sim)}
    if a["out"]:
        extra = {k: np.array(v) for k, v in liquid_setting(sim).items()}      # the .npz names its spacing too
        out["saved"] = str(traj.save(Path(a["out"]).expanduser(), **extra))
    return out


def _samples(traj: Any, k: int, world: int) -> list[dict[str, Any]]:
    """``k`` evenly spaced frames of world ``world``: time, positions and joint values."""
    if k <= 0 or traj.frames == 0:
        return []
    idx = np.unique(np.linspace(0, traj.frames - 1, min(k, traj.frames)).round().astype(int))
    keys = [key for key in traj if key.endswith((".pos", ".q"))]
    return [{"t": round(float(traj["t"][i]), 9), **{key: traj[key][i, world] for key in keys}} for i in idx]


def op_state(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Current state of named bodies, joints and particle objects."""
    sid, sim = session.get(a)
    check_view(sim, None, a["world"])
    return {"sim": sid, "t": now(sim), "state": state_view(sim, a["names"], a["world"], a["points"])}


def op_targets(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Set joint targets (kept until changed)."""
    sid, sim = session.get(a)
    sim.set_targets(_targets(a["targets"], ramps=False), a["kind"])
    return {"sim": sid, "set": sorted(a["targets"]), "kind": a["kind"]}


def op_reset(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Back to the start state of every world."""
    sid, sim = session.get(a)
    sim.reset()
    return {"sim": sid, "t": now(sim)}


def op_close(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Close an open simulation and free its memory."""
    sid = session.close(a["sim"])
    return {"closed": sid, "open": sorted(session.sims)}
