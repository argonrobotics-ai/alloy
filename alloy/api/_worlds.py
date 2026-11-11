# SPDX-License-Identifier: Apache-2.0
"""Per-world starts in the Scene API: what can start somewhere else in every world, and how it is moved there.

``pos=`` as a ``(worlds, 3)`` array (and ``scene.randomize(name, "x" | "y" | "z", lo, hi)``) works for:

- free and kinematic bodies (``add_box`` ..., ``add_mesh``, library objects, kinematic fixtures part by part): the
  body's free joint per world (``place_bodies``);
- free-base robots (``alloy.api._robots``);
- cloth, liquids and soft bodies: their particles shift by the world's offset (``shift_particles``).

Static things are one shape shared by every world (the ground, static shapes and fixtures, a pour's nozzle): they
take one ``pos``, and an ``(worlds, 3)`` array is refused when they are added (``refuse_static``), with the fix:
``kinematic=True`` makes an object that holds still unless driven and can start per world. All of it runs on the
Model before the simulation is made, so ``sim.reset()`` returns to these starts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alloy.api.errors import BuildError, SceneError

log = logging.getLogger("alloy.api")
AXES = {"x": 0, "y": 1, "z": 2, "pose.0": 0, "pose.1": 1, "pose.2": 2}
PARTICLE_KINDS = ("cloth", "liquid")
STATIC_FIX = ("pass kinematic=True: the object then holds still unless driven (sim.drive / set_pose) and can start "
              "per world; or give one pos=(x, y, z)")


def per_world(pos: Any) -> bool:
    """True when ``pos`` is one row per world. Example: ``per_world([[0, 0, 1], [0.1, 0, 1]])``."""
    return np.asarray(pos, dtype=object).ndim == 2


def refuse_static(name: str, what: str = "a static object", fix: str = STATIC_FIX) -> None:
    """The error for per-world positions of something every world shares. Example: ``refuse_static("table")``."""
    raise SceneError(f"{name!r} is {what}, the same in every world, so it takes one pos (x, y, z), not one per world",
                     fix, field="pos")


def add_start_random(item: dict[str, Any], name: str, field: str, lo: float, hi: float) -> None:
    """Record a randomized start coordinate (absolute, m) of a cloth or liquid, drawn at build as an object's is.

    Example: ``add_start_random(scene.items["towel"], "towel", "x", -0.01, 0.01)``."""
    if field not in AXES:
        raise SceneError(f"{name!r}: cannot randomize {field!r} of a {item['kind']}", "randomize its start: 'x', "
                         "'y' or 'z' (its material sets the rest)", field="field")
    item.setdefault("start_random", []).append({"field": f"pose.{AXES[field]}", "lo": float(lo), "hi": float(hi)})


def rows(name: str, base: Any, per_world_pos: Any, randoms: list[dict[str, Any]], worlds: int,
         seed: int) -> np.ndarray:
    """(worlds, 3) start positions: the per-world rows (else ``base`` in every world), then each randomized axis
    drawn as the engine draws an object's (``alloy.scene.randomize.draw``). Example: ``rows("a", (0, 0, 1), None, [],
    2, 0)``."""
    from alloy.scene.randomize import draw  # noqa: PLC0415

    out = np.array(per_world_pos or [list(base)[:3]] * worlds, dtype=np.float64)
    for r in randoms:
        out[:, AXES[r["field"]]] = [draw(seed, w, f"{name}.{r['field']}", r["lo"], r["hi"]) for w in range(worlds)]
    return out


def place(built: Any, scene: Any, per_world_pos: dict[str, Any], seed: int,
          block: tuple[int, int] | None = None) -> None:
    """Every per-world start of ``scene`` (built with ``scene.worlds`` worlds) written into ``built``'s Model.
    ``block`` = (offset, count): ``built`` holds only global worlds offset..offset+count-1 of the scene's (a worker
    process of ``alloy.build(..., processes=N)``); every row is drawn for the whole scene, by global world id, then
    cut to the block, so a world starts where it would in one process.

    Example: ``place(built, scene, materialized.per_world_pos, seed=0)``."""
    from alloy.api import _robots  # noqa: PLC0415
    from alloy.api.names import NameIndex  # noqa: PLC0415

    cut = slice(None) if block is None else slice(block[0], block[0] + block[1])
    idx = NameIndex.from_built(built)
    particles = {n: (scene.items[n]["pose"][:3], r[cut]) for n, r in _particle_rows(scene, seed).items()}
    bodies = {}
    for name, r in per_world_pos.items():
        if name in idx.free_joint:
            bodies[name] = np.asarray(r, np.float64)[cut]
        elif name in idx.particles:
            particles.setdefault(name, (scene.items[name]["pose"][:3], np.asarray(r, np.float64)[cut]))
        else:
            refuse_static(name, "static (its card says so)" if name in idx.static else "not a free body")
    if bodies:
        place_bodies(built, idx, bodies)
    if particles:
        shift_particles(built, idx, particles)
    robot_rows = {n: r[cut] for n, r in _robots.starts(scene, seed).items()}
    if robot_rows:
        _robots.place(built, robot_rows)


def _particle_rows(scene: Any, seed: int) -> dict[str, np.ndarray]:
    return {n: rows(n, it["pose"], it.get("per_world_pos"), it["start_random"], scene.worlds, seed)
            for n, it in scene.items.items() if it.get("start_random")}


def place_bodies(built: Any, idx: Any, per_world_pos: dict[str, Any]) -> None:
    """Move each named free (or kinematic) body's start to its per-world position: its free joint's translation.

    Example: ``place_bodies(built, NameIndex.from_built(built), {"cube": [[0, 0, 1], [0.1, 0, 1]]})``."""
    from alloy._vendor import newton  # noqa: PLC0415

    m = built.model
    state = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, state)
    body_q, joint_q = state.body_q.numpy(), m.joint_q.numpy()
    q_start, x_p = m.joint_q_start.numpy(), m.joint_X_p.numpy()
    for name, r in per_world_pos.items():
        bodies = idx.per_world(idx.bodies[name], "body")
        joints = idx.per_world(idx.free_joint[name], "joint")
        for w, (b, j) in enumerate(zip(bodies, joints)):
            delta = np.asarray(r[w], float) - body_q[b, :3]
            q0 = int(q_start[j])
            joint_q[q0:q0 + 3] += _rotate_inv(x_p[j, 3:7], delta)
    m.joint_q.assign(joint_q)
    newton.eval_fk(m, m.joint_q, m.joint_qd, m)


def shift_particles(built: Any, idx: Any, starts: dict[str, tuple[Any, np.ndarray]]) -> None:
    """Move each particle object (cloth, liquid, soft body) in world w by ``rows[w] - base``; ``starts`` maps a name
    to (base position it was built at, (worlds, 3) rows).

    Example: ``shift_particles(built, idx, {"towel": (p0, rows)})``."""
    m = built.model
    q = m.particle_q.numpy()
    stride = idx.stride["particle"]
    for name, (base, r) in starts.items():
        rng = idx.particle_range(name)
        offsets = np.asarray(r, np.float64) - np.asarray(base, np.float64)[None, :]
        for w, d in enumerate(offsets):
            q[rng.start + w * stride:rng.stop + w * stride] += d.astype(q.dtype)
        log.info("%s: per-world start offsets up to %.3g m", name, float(np.abs(offsets).max()))
    m.particle_q.assign(q)


def _rotate_inv(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)^T v for a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(c) for c in q_xyzw)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r.T @ v
