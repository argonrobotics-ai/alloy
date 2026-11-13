# SPDX-License-Identifier: Apache-2.0
"""Per-world simulator state of a ``VectorEnv``: read it every step, write it back to start a replay.

The state of one world is its rows of the engine's joint coordinates and velocities (free bodies are joints there,
so rigid objects are included), its rows of the control targets (joint targets and the controls of MuJoCo-native
actuators), and the env's goal points. Cloth, liquid and other
particle objects are NOT part of it: an episode of such a scene records and replays by actions only from the env's own
reset, not from a restored start. Example::

    snap = WorldSnapshot(env)
    start = snap.capture()                     # {"joint_q": (W, nq), "joint_qd": (W, nv), "target.position": ...}
    snap.restore(mask, start)                  # the masked worlds go back to it (after ``env.reset_worlds(mask)``)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.envs.actions import control_array
from alloy.envs.errors import EnvError

#: snapshot key -> (engine State array, ``Layout.world_of`` key)
STATE_ARRAYS = {"joint_q": ("joint_q", "q"), "joint_qd": ("joint_qd", "qd")}
TARGET, GOAL, RANDOM = "target.", "goal.", "random."


class WorldSnapshot:
    """Reads and writes the per-world state of ``env`` as NumPy arrays with a leading world axis."""

    def __init__(self, env: Any) -> None:
        self.env, self.xp, self.engine = env, env.xp, env.sim.engine
        worlds, world_of = env.num_envs, env.layout.world_of
        self.rows: dict[str, np.ndarray] = {}
        for key, (array, wkey) in STATE_ARRAYS.items():
            if getattr(self.engine.state, array, None) is not None and len(world_of.get(wkey, ())):
                self.rows[key] = _rows_per_world(world_of[wkey], worlds, key)
        self.targets: dict[str, np.ndarray] = {}
        for kind, (rows, world) in env.writer._world.items():      # control rows that belong to one world
            ids = np.full(int(control_array(self.engine.control, kind).shape[0]), -1, np.int64)
            ids[self.xp.to_numpy(rows)] = self.xp.to_numpy(world)
            if kind == "ctrl" and not np.any(ids >= 0):            # the env's layout names no world for these rows
                ids = _actuator_worlds(self.engine.model, ids)
            if np.any(ids >= 0):
                self.targets[kind] = _rows_per_world(ids, worlds, TARGET + kind)
        self.has_particles = bool(len(world_of.get("particle", ())))

    # --- reading ---------------------------------------------------------------------------------------------------

    def randomized(self) -> dict[str, np.ndarray]:
        """What ``Scene.randomize`` drew for robots and gravity, per world: ``{"random.<target>.<field>": (W, n)}``.
        A replay cannot set these; it checks them. Example: ``snap.randomized()``."""
        read = getattr(self.env.sim, "randomized", None)
        values = read() if callable(read) else {}
        worlds = int(self.env.num_envs)
        return {RANDOM + name: np.asarray(v, np.float64).reshape(worlds, -1) for name, v in values.items()
                if np.size(v) and np.size(v) % worlds == 0}

    def capture(self, randomized: bool = False) -> dict[str, np.ndarray]:
        """Every world's state now: ``{key: (W, n) float32}``; with ``randomized`` also ``randomized()`` (what an
        episode's ``start`` keeps). Example: ``snap.capture()["joint_q"].shape``."""
        xp, out = self.xp, dict(self.randomized()) if randomized else {}
        for key, rows in self.rows.items():
            out[key] = xp.to_numpy(xp.view(getattr(self.engine.state, STATE_ARRAYS[key][0])))[rows].copy()
        for kind, rows in self.targets.items():
            out[TARGET + kind] = xp.to_numpy(xp.view(control_array(self.engine.control, kind)))[rows].copy()
        for name, value in self.env.goals.values.items():
            out[GOAL + name] = xp.to_numpy(value).copy()
        return out

    def flat(self, snapshot: dict[str, np.ndarray]) -> np.ndarray:
        """Joint coordinates then velocities as one (W, S) array: an episode's ``states`` row per step.

        Example: ``snap.flat(snap.capture()).shape``."""
        parts = [snapshot[k] for k in STATE_ARRAYS if k in snapshot]
        if not parts:
            return np.zeros((self.env.num_envs, 0), np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def flat_names(self) -> list[str]:
        """One label per column of ``flat``. Example: ``snap.flat_names()[:2]`` -> ``["joint_q[0]", "joint_q[1]"]``."""
        return [f"{k}[{i}]" for k in STATE_ARRAYS if k in self.rows for i in range(self.rows[k].shape[1])]

    # --- writing ---------------------------------------------------------------------------------------------------

    def restore(self, mask: np.ndarray, values: dict[str, np.ndarray]) -> None:
        """Put the worlds in the bool (W,) ``mask`` into ``values`` (arrays with a leading world axis, as ``capture``
        gives). Call it right after ``env.reset_worlds(mask)``: it then also makes this the episode's start state.

        Example: ``env.reset_worlds(m); snap.restore(m, start)``."""
        mask = np.asarray(mask, bool)
        self._check(values)
        sel = np.flatnonzero(mask)
        for key, rows in self.rows.items():
            self._put(getattr(self.engine.state, STATE_ARRAYS[key][0]), rows[sel], values[key][sel])
        for kind, rows in self.targets.items():
            self._put(control_array(self.engine.control, kind), rows[sel], values[TARGET + kind][sel])
        goals = self.env.goals
        for name in goals.values:
            new = self.xp.asarray(values[GOAL + name], "float32")
            goals.values[name] = self.xp.where(self.xp.asarray(mask, "bool")[:, None], new, goals.values[name])
        self._forward_kinematics(mask)
        self._new_start(mask)

    def _put(self, array: Any, rows: np.ndarray, values: np.ndarray) -> None:
        idx, vals = self.xp.asarray(rows, "int64"), self.xp.asarray(values, "float32")
        self.xp.update(array, lambda a: a.__setitem__(idx, vals))

    def _check(self, values: dict[str, np.ndarray]) -> None:
        need = {**self.rows, **{TARGET + k: r for k, r in self.targets.items()}}
        for key, rows in need.items():
            shape = tuple(np.shape(values.get(key, ())))
            if shape != rows.shape:
                raise EnvError(f"the start state has {key} of shape {shape}, this env needs {rows.shape}",
                               "replay in an env built like the one that recorded (same scene, same number of "
                               "worlds or pad the arrays to it)", field=key)
        missing = [GOAL + n for n in self.env.goals.values if GOAL + n not in values]
        if missing:
            raise EnvError(f"the start state lacks {missing}", "replay with the task that recorded it", field="goals")

    def _forward_kinematics(self, mask: np.ndarray) -> None:
        """Body poses of the masked worlds from their restored joint coordinates."""
        import warp as wp  # noqa: PLC0415

        from alloy._vendor import newton  # noqa: PLC0415

        eng, joint_world = self.engine, self.env.layout.world_of["joint"]
        if not len(joint_world):
            return
        sel = mask[np.maximum(joint_world, 0)] & (joint_world >= 0)
        with wp.ScopedDevice(eng.model.device):
            newton.eval_fk(eng.model, eng.state.joint_q, eng.state.joint_qd, eng.state,
                           joint_mask=self.xp.to_warp_bool(self.xp.asarray(sel, "bool")))

    def _new_start(self, mask: np.ndarray) -> None:
        """The restored state is the episode's start: start-relative checks and the energy reference follow it."""
        import warp as wp  # noqa: PLC0415

        env, m = self.env, self.xp.asarray(mask, "bool")
        health = self.engine.health
        if hasattr(health, "reset_reference"):
            with wp.ScopedDevice(env.device):
                health.reset_reference(self.engine.state, self.xp.to_warp_mask(m))
        env.start.capture(m)


def _actuator_worlds(model: Any, fallback: np.ndarray) -> np.ndarray:
    """World id of every row of ``control.mujoco.ctrl`` (MuJoCo-native actuators), from the model."""
    world = getattr(getattr(model, "mujoco", None), "actuator_world", None)
    if world is None or len(world) != len(fallback):
        return fallback
    return np.asarray(world.numpy(), np.int64)


def _rows_per_world(world_ids: np.ndarray, worlds: int, what: str) -> np.ndarray:
    """(W, n) row indices from a world id per row (-1: shared rows, left out); every world needs the same count."""
    ids = np.asarray(world_ids)
    rows = [np.flatnonzero(ids == w) for w in range(worlds)]
    if len({len(r) for r in rows}) != 1:
        raise EnvError(f"worlds have different numbers of {what} rows", "record with states=False (episodes then "
                       "replay by actions from the env's own reset)", field="states")
    return np.stack(rows)


__all__ = ["WorldSnapshot"]
