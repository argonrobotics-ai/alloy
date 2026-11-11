# SPDX-License-Identifier: Apache-2.0
"""Applying steering commands to a running ``VectorEnv`` (at step boundaries, in the training thread).

``Steering.apply(cmd)`` returns ``(ok, result, changed)``: ``changed`` tells the worker that some world's state
changed under the policy (reset, fork, restore), so its observation is re-read before the next action.

- pause / resume: the whole run (``world`` None) or one world (frozen in place, :class:`WorldOps`).
- set: a whitelisted setting (:class:`alloy.live.settings.Settings`).
- push: a force (N, world frame) on a body of one world for ``steps`` control steps, through ``Sim.apply_wrench``.
- fork: copy a world into another slot (``into`` an int) or into a snapshot (``into`` "snapshot:<name>").
- restore: a snapshot into a world slot. reset: one world to a fresh episode start.

Example::

    steer = Steering(env, settings)
    ok, result, changed = steer.apply({"op": "push", "args": {"world": 3, "body": "cube", "force": [0, 2, 0],
                                                              "steps": 10}})
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alloy.live.settings import SettingError, Settings
from alloy.live.worldops import WorldOpError, WorldOps

log = logging.getLogger(__name__)
MAX_FORCE_N = 50.0
MAX_PUSH_STEPS = 200


class Steering:
    def __init__(self, env: Any, settings: Settings) -> None:
        self.env, self.settings = env, settings
        self.ops = WorldOps(env)
        self.run_paused = False
        self._pushes: dict[tuple[str, int], tuple[np.ndarray, int]] = {}    # (body, world) -> (force, steps left)
        self._bodies = set(env.sim.index.bodies)

    def apply(self, cmd: dict[str, Any]) -> tuple[bool, dict[str, Any], bool]:
        op, args = cmd.get("op"), dict(cmd.get("args") or {})
        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            return False, {"error": f"unknown op {op!r}"}, False
        try:
            result, changed = handler(args)
            return True, result, changed
        except (SettingError, WorldOpError, ValueError, KeyError, TypeError) as e:
            return False, {"error": str(e)}, False

    def after_step(self) -> None:
        """Hold paused worlds and count down pushes (call after every env.step)."""
        self.ops.hold_frozen()
        if not self._pushes:
            return
        ended = [k for k, (_f, n) in self._pushes.items() if n <= 1]
        self._pushes = {k: (f, n - 1) for k, (f, n) in self._pushes.items() if n > 1}
        if ended:
            self._write_wrenches({k[0] for k in ended})

    # --- ops -------------------------------------------------------------------------------------------------------

    def _op_pause(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if a.get("world") is None:
            self.run_paused = True
            return {"run": "paused"}, False
        return self.ops.pause(int(a["world"])), False

    def _op_resume(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if a.get("world") is None:
            self.run_paused = False
            return {"run": "running"}, False
        return self.ops.resume(int(a["world"])), False

    def _op_set(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        return self.settings.set(str(a["name"]), a["value"]), False

    def _op_push(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        body, world = str(a["body"]), int(a["world"])
        if body not in self._bodies:
            raise ValueError(f"no body {body!r}; bodies are {sorted(self._bodies)}")
        if not 0 <= world < self.env.num_envs:
            raise ValueError(f"world {world} is outside 0..{self.env.num_envs - 1}")
        force = np.asarray(a["force"], np.float32).reshape(3)
        norm = float(np.linalg.norm(force))
        if norm > MAX_FORCE_N:
            force *= MAX_FORCE_N / norm
        steps = int(np.clip(int(a.get("steps", 10)), 1, MAX_PUSH_STEPS))
        self._pushes[(body, world)] = (force, steps)
        self._write_wrenches({body})
        return {"world": world, "body": body, "force": [round(float(x), 3) for x in force], "steps": steps}, False

    def _op_fork(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        src, into = int(a["world"]), a["into"]
        if isinstance(into, str) and into.startswith("snapshot:"):
            return self.ops.save(src, into.split(":", 1)[1]), False
        return self.ops.fork(src, int(into)), True

    def _op_restore(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        return self.ops.restore(str(a["snapshot"]), int(a["world"])), True

    def _op_reset(self, a: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        return self.ops.reset(int(a["world"])), True

    # --- wrenches --------------------------------------------------------------------------------------------------

    def _write_wrenches(self, bodies: set[str]) -> None:
        """Rewrite the (worlds, 3) force of each named body from the active pushes (zero where none)."""
        for body in bodies:
            f = np.zeros((self.env.num_envs, 3), np.float32)
            for (b, w), (force, _n) in self._pushes.items():
                if b == body:
                    f[w] = force
            self.env.sim.apply_wrench(body, force=f)
