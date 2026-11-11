# SPDX-License-Identifier: Apache-2.0
"""Timed pushes: ``scene.add_push("go2/base", force=(80, 80, 0), every=(4.0, 8.0), duration=0.1)``.

Each world waits a time drawn uniformly in ``every`` (s), then a force drawn uniformly in +-``force`` per world axis
(N) is held on the body's centre of mass for ``duration`` seconds, then the world waits again. Every world has its own
clock and its own draws. A world that resets drops its push and starts a new wait.

``Pusher.tick`` runs before every control period of ``sim.step`` (so in ``sim.run`` and in ``alloy.envs`` too). The
force goes through ``sim.apply_wrench``: it is held in the state's body forces, the health check books its work as
work from outside, and nothing runs inside a captured graph. Draws are counter-based
(``alloy.scene.randomize.draw``): a function of the seed, the global world id, the body's name and that world's push
number, so they are the same on every device and for every world count.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.api.errors import BuildError, SceneError


def add(scene: Any, name: str, force: Any, every: Any, duration: float) -> None:
    """Check and record one push row on ``scene.pushes``. Example: ``add(scene, "go2/base", (80, 80, 0), (4, 8),
    0.1)``."""
    f = np.asarray(force, np.float64)
    if f.shape != (3,) or not np.isfinite(f).all() or (f < 0).any() or not (f > 0).any():
        raise SceneError(f"push {name!r}: force must be 3 values >= 0, one of them above 0, got {force!r}",
                         "pass the largest |force| per world axis in N, e.g. force=(80, 80, 0)", field="force")
    lo, hi = (float(v) for v in np.broadcast_to(np.asarray(every, np.float64), (2,)))
    duration = float(duration)
    if not (0.0 < duration and 0.0 < lo <= hi):
        raise SceneError(f"push {name!r}: needs duration > 0 and 0 < every[0] <= every[1], got every={every!r}, "
                         f"duration={duration!r}", "pass the wait between pushes in seconds, e.g. every=(4.0, 8.0), "
                         "duration=0.1", field="every")
    if any(p["target"] == name for p in scene.pushes):
        raise SceneError(f"push {name!r}: this body already has a push", "keep one add_push per body (name another "
                         "link for a second one)", field="name")
    scene.pushes.append({"target": str(name), "force": [float(v) for v in f], "every": [lo, hi],
                         "duration": duration})


def refuse(scene: Any, rerun_worlds: Any, processes: Any) -> None:
    """Pushes are applied by the one process that steps the worlds, on its main simulation.

    Example: ``refuse(scene, 0, 4)`` raises when the scene has pushes."""
    if not getattr(scene, "pushes", None) or not (rerun_worlds or processes is not None):
        return
    what = "rerun_worlds" if rerun_worlds else "processes"
    raise BuildError(f"{what} cannot be used with scene.add_push (the pushes are held on the one simulation that "
                     "steps every world)", f"build without {what}, or drop the pushes", field=what)


class Pusher:
    """The pushes of one ``Sim``. ``force[name]`` (worlds, 3) is the force held now (zeros while a world waits)."""

    def __init__(self, sim: Any, rows: list[dict[str, Any]], seed: int) -> None:
        self.sim, self.rows, self.seed = sim, [dict(r) for r in rows], int(seed)
        w, dt = int(sim.worlds), float(sim.dt) / int(sim._sub)
        for r in self.rows:
            if r["target"] not in sim.index.bodies:
                raise BuildError(f"push: {r['target']!r} is not a body that moves in the built scene",
                                 f"use one of {sorted(sim.index.bodies)}", field="name")
        self.dt = dt
        self.gids = np.asarray([sim.built.global_world_id(i) for i in range(w)], np.int64)
        self.length = [max(1, round(r["duration"] / dt)) for r in self.rows]
        self.force = {r["target"]: np.zeros((w, 3)) for r in self.rows}
        self.number = np.zeros((len(self.rows), w), np.int64)      # pushes started so far, per row and world
        self.left = np.zeros((len(self.rows), w), np.int64)        # control periods until the next change
        self.pushing = np.zeros((len(self.rows), w), bool)
        for i in range(len(self.rows)):
            self._wait(i, np.arange(w))

    def tick(self) -> None:
        """Start and end the pushes that are due, before a control period. Example: ``pusher.tick()``."""
        for i, r in enumerate(self.rows):
            due = np.flatnonzero(self.left[i] <= 0)
            if len(due):
                ending = due[self.pushing[i, due]]
                starting = due[~self.pushing[i, due]]
                self._wait(i, ending)
                self._start(i, starting)
                self.sim._io.apply_wrench(r["target"], self.force[r["target"]], (0.0, 0.0, 0.0), None)
            self.left[i] -= 1

    def on_reset(self, world_mask: Any = None) -> None:
        """The worlds of ``world_mask`` (all when None) drop their push and start a new wait (an engine reset hook).

        Example: ``pusher.on_reset()``."""
        w = self.number.shape[1]
        worlds = np.arange(w) if world_mask is None else np.flatnonzero(world_mask.numpy()[:w])
        for i, r in enumerate(self.rows):
            held = bool(self.pushing[i, worlds].any())
            self._wait(i, worlds)
            if held:
                self.sim._io.apply_wrench(r["target"], self.force[r["target"]], (0.0, 0.0, 0.0), None)

    def _draw(self, i: int, world: int, what: str, lo: float, hi: float) -> float:
        from alloy.scene.randomize import draw  # noqa: PLC0415

        key = f"{self.rows[i]['target']}.push.{what}#{int(self.number[i, world])}"
        return draw(self.seed, int(self.gids[world]), key, lo, hi)

    def _wait(self, i: int, worlds: np.ndarray) -> None:
        lo, hi = self.rows[i]["every"]
        for w in worlds:
            self.left[i, w] = max(1, round(self._draw(i, int(w), "wait", lo, hi) / self.dt))
        self.pushing[i, worlds] = False
        self.force[self.rows[i]["target"]][worlds] = 0.0

    def _start(self, i: int, worlds: np.ndarray) -> None:
        half = self.rows[i]["force"]
        for w in worlds:
            self.force[self.rows[i]["target"]][w] = [self._draw(i, int(w), axis, -h, h) for axis, h in zip("xyz", half)]
            self.number[i, w] += 1
        self.left[i, worlds] = self.length[i]
        self.pushing[i, worlds] = True


def attach(sim: Any, scene: Any, seed: int) -> None:
    """Give ``sim`` the scene's pushes (nothing when it has none). Example: ``attach(sim, scene, 0)`` at build."""
    rows = getattr(scene, "pushes", None)
    if not rows:
        return
    sim._pushes = Pusher(sim, rows, seed)
    sim.engine.reset_hooks.append(sim._pushes.on_reset)


def forces(sim: Any) -> dict[str, np.ndarray]:
    """{body: (worlds, 3) push force held now, N}. Example: ``forces(sim)["go2/base"]``."""
    pusher = getattr(sim, "_pushes", None)
    return {} if pusher is None else {k: v.copy() for k, v in pusher.force.items()}


__all__ = ["Pusher", "add", "attach", "forces", "refuse"]
