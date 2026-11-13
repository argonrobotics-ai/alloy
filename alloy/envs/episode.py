# SPDX-License-Identifier: Apache-2.0
"""Per-world episode bookkeeping: engine arrays as they were at each world's last reset (``state.start``), and the
step count, return and success of the running episode.

Example::

    start = StartState(engine, layout, xp)
    start.capture(mask)                    # after engine.reset(mask): remember the masked worlds' start state
    s0 = start.view                        # arrays accessor for a BatchState (``s.start["cube.pos"]``)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.envs.layout import Layout
from alloy.envs.state import ENGINE_ARRAYS, MODEL_ARRAYS

#: engine array -> key of ``Layout.world_of`` that gives the world of each row
WORLD_KEY = {"body_q": "body", "body_qd": "body", "joint_q": "q", "joint_qd": "qd", "particle_q": "particle"}


class LiveArrays:
    """Accessor for the engine's current State arrays (one backend view per array, made on first use)."""

    def __init__(self, engine: Any, xp: Any) -> None:
        self.engine, self.xp = engine, xp
        self._views: dict[str, Any] = {}

    def __call__(self, name: str) -> Any:
        if name == "particle_active":
            return self.active()
        if name not in self._views:
            arr = getattr(self.engine.state, name, None)
            if arr is None and name in MODEL_ARRAYS:
                arr = getattr(self.engine.model, name, None)
            if arr is None:
                raise KeyError(f"the engine state has no {name}")
            self._views[name] = self.xp.view(arr)
        return self._views[name]

    def active(self) -> Any:
        """(N,) bool, now: particles taking part (False: a pour's drop still at the nozzle, or a retired particle), from
        the model's flags and each liquid solver's own (``sim.state()``'s ``"<name>.active"``). Example: internal."""
        from alloy._vendor import newton  # noqa: PLC0415

        if "_liquid" not in self._views:
            from alloy.api._liquid import _local_to_global, liquid_solvers  # noqa: PLC0415

            pairs = []
            for solver in liquid_solvers(self.engine):
                ids = _local_to_global(self.engine, solver)
                if ids is not None and len(ids):
                    pairs.append((self.xp.asarray(ids, "int64"), self.xp.view(solver.model.particle_flags)))
            self._views["_liquid"] = pairs
        flags = self.xp.copy(self.xp.view(self.engine.model.particle_flags))
        for ids, view in self._views["_liquid"]:
            flags[ids] = view[: ids.shape[0]]
        return (flags & int(newton.ParticleFlags.ACTIVE)) != 0


class StartState:
    """Copies of the engine's State arrays taken at each world's last reset."""

    def __init__(self, engine: Any, layout: Layout, xp: Any) -> None:
        self.engine, self.xp = engine, xp
        self.arrays: dict[str, Any] = {}
        self._world: dict[str, tuple[Any, Any]] = {}
        live = LiveArrays(engine, xp)
        for name in ENGINE_ARRAYS:
            if getattr(engine.state, name, None) is None:
                continue
            self.arrays[name] = xp.copy(live(name))
            ids = layout.world_of.get(WORLD_KEY[name], np.zeros(0, np.int32))
            if len(ids) != int(self.arrays[name].shape[0]):
                ids = np.full(int(self.arrays[name].shape[0]), -1, np.int32)
            self._world[name] = (xp.asarray(np.maximum(ids, 0), "int64"), xp.asarray(ids >= 0, "bool"))

    def view(self, name: str) -> Any:
        """The start copy of engine array ``name``. Example: ``start.view("body_q")``."""
        if name not in self.arrays:
            if name in MODEL_ARRAYS:                             # constant: the model's own array
                return LiveArrays(self.engine, self.xp)(name)
            raise KeyError(f"the engine state has no {name}")
        return self.arrays[name]

    def capture(self, mask: Any) -> None:
        """Copy the current state of the worlds in the bool (W,) ``mask``. Example: ``start.capture(done)``."""
        live = LiveArrays(self.engine, self.xp)
        for name, saved in self.arrays.items():
            ids, valid = self._world[name]
            sel = mask[ids] & valid
            if saved.ndim > 1:
                sel = sel.reshape(-1, *([1] * (saved.ndim - 1)))
            self.arrays[name] = self.xp.where(sel, live(name), saved)


class Goals:
    """Per-world goal points (a reach target, a place spot), drawn uniformly in their box at every reset.

    They read like bodies: ``state["target.pos"]`` (W, 3), ``checks.distance("arm/hand", "target")``."""

    def __init__(self, boxes: Mapping[str, tuple[Sequence[float], Sequence[float]]], worlds: int, xp: Any) -> None:
        self.xp, self.worlds = xp, worlds
        self.boxes, self.values = {}, {}
        for name, (lo, hi) in boxes.items():
            lo_a, hi_a = np.asarray(lo, np.float32).reshape(-1), np.asarray(hi, np.float32).reshape(-1)
            if lo_a.shape != (3,) or hi_a.shape != (3,) or np.any(hi_a < lo_a):
                raise EnvError(f"goal {name!r} needs a box ((x, y, z) low, (x, y, z) high) with high >= low",
                               "pass goals={'target': ((0.3, -0.2, 0.1), (0.6, 0.2, 0.4))}", field="goals")
            self.boxes[name] = (xp.asarray(lo_a), xp.asarray(hi_a))
            self.values[name] = xp.zeros((worlds, 3)) + self.boxes[name][0]

    def resample(self, mask: Any) -> None:
        """New goal points for the worlds in the bool (W,) ``mask``. Example: ``goals.resample(done)``."""
        xp = self.xp
        for name, (lo, hi) in self.boxes.items():
            new = xp.uniform((self.worlds, 3), 0.0, 1.0) * (hi - lo) + lo
            self.values[name] = xp.where(mask[:, None], new, self.values[name])

    def extra(self) -> dict[str, Any]:
        """State keys: ``{"<goal>.pos": (W, 3)}``. Example: ``goals.extra()["target.pos"]``."""
        return {f"{name}.pos": v for name, v in self.values.items()}


class EpisodeStats:
    """Steps, return and last action of every world's running episode."""

    def __init__(self, worlds: int, actions: int, xp: Any) -> None:
        self.xp = xp
        self.steps = xp.zeros(worlds, "int64")
        self.returns = xp.zeros(worlds)
        self.action = xp.zeros((worlds, actions))

    def act(self, action: Any) -> None:
        """One step taken with ``action``: count it and keep the action. Example: ``stats.act(a)``."""
        self.steps = self.steps + 1
        self.action = action

    def add(self, reward: Any) -> None:
        """Add this step's reward to the running return. Example: ``stats.add(r)``."""
        self.returns = self.returns + reward

    def summary(self, success: Any) -> dict[str, Any]:
        """Gymnasium's ``info["episode"]``: return ``r``, length ``l`` and ``success`` per world."""
        return {"r": self.xp.copy(self.returns), "l": self.xp.copy(self.steps), "success": self.xp.copy(success)}

    def reset(self, mask: Any) -> None:
        """Zero the counters of the masked worlds. Example: ``stats.reset(done)``."""
        xp = self.xp
        self.steps = xp.where(mask, 0, self.steps)
        self.returns = xp.where(mask, 0.0, self.returns)
        self.action = xp.where(mask[:, None], 0.0, self.action)


__all__ = ["EpisodeStats", "Goals", "LiveArrays", "StartState", "WORLD_KEY"]
