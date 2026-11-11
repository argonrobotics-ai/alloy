# SPDX-License-Identifier: Apache-2.0
"""World-slot operations for steering a ``VectorEnv``: fork one world into another slot, save a world to a named
snapshot and restore it into any slot, freeze (pause) single worlds, and reset single worlds.

Physics state goes through the engine's own per-world snapshot (``WorldSnapshot``, the copy ``reset`` restores from:
``State`` arrays plus the stepper's solver-state arrays), so a forked world continues exactly as its source would
have. The env's per-world bookkeeping (episode steps and return, start pose, action targets, goals) is copied row by
row. Everything here runs at a step boundary, between two ``env.step`` calls.

Example::

    ops = WorldOps(env)
    ops.fork(3, 17)                 # world 17 becomes a copy of world 3
    ops.save(3, "before_push")      # host copy of world 3
    ops.restore("before_push", 5)   # world 5 continues from that moment
"""
from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


class WorldOpError(ValueError):
    """A world index out of range, a missing snapshot, or a scene whose state has no per-world copy."""


class WorldOps:
    """Fork / snapshot / pause / reset for the worlds of one ``VectorEnv``."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.engine = env.sim.engine
        self.worlds = int(env.num_envs)
        with wp.ScopedDevice(env.device):
            self._snap = self.engine._snapshot()        # scratch: one world in, one world out
            self._frozen_snap = self.engine._snapshot()  # paused worlds, restored after every step
        whole = self._snap.whole_arrays
        self.forkable = not whole
        self._maps = {s.name: (s.item_world.numpy() if s.item_world is not None else None) for s in self._snap.slots}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.frozen: dict[int, dict[str, Any]] = {}

    # --- public operations -----------------------------------------------------------------------------------------

    def fork(self, src: int, dst: int) -> dict[str, Any]:
        """World ``dst`` becomes an exact copy of world ``src`` (physics, episode clock, targets)."""
        self._check(src), self._check(dst)
        if src == dst:
            raise WorldOpError("fork needs two different worlds")
        rows = self._save_rows(src)
        self._load_rows(rows, dst)
        return {"from": src, "to": dst}

    def save(self, world: int, name: str) -> dict[str, Any]:
        """Keep a host copy of one world under ``name`` (overwrites a snapshot of that name)."""
        self._check(world)
        self.snapshots[name] = self._save_rows(world)
        return {"snapshot": name, "world": world, "bytes": _nbytes(self.snapshots[name])}

    def restore(self, name: str, world: int) -> dict[str, Any]:
        """Load snapshot ``name`` into world slot ``world``."""
        self._check(world)
        if name not in self.snapshots:
            raise WorldOpError(f"no snapshot {name!r}; snapshots are {sorted(self.snapshots)}")
        self._load_rows(self.snapshots[name], world)
        return {"snapshot": name, "world": world}

    def reset(self, world: int) -> dict[str, Any]:
        """Reset one world to a fresh episode start (the env's own per-world reset)."""
        self._check(world)
        mask = self.env.xp.zeros(self.worlds, "bool")
        mask[world] = True
        self.env.reset_worlds(mask)
        self.frozen.pop(world, None)
        return {"world": world}

    def pause(self, world: int) -> dict[str, Any]:
        """Freeze one world: after every step it is put back to the state it had now."""
        self._check(world)
        self.frozen[world] = self._save_rows(world)
        return {"world": world, "paused": sorted(self.frozen)}

    def resume(self, world: int) -> dict[str, Any]:
        self.frozen.pop(world, None)
        return {"world": world, "paused": sorted(self.frozen)}

    def hold_frozen(self) -> None:
        """Call after each ``env.step``: puts every paused world back."""
        for world, rows in self.frozen.items():
            self._load_rows(rows, world)

    # --- rows ------------------------------------------------------------------------------------------------------

    def _check(self, world: int) -> None:
        if not 0 <= int(world) < self.worlds:
            raise WorldOpError(f"world {world} is outside 0..{self.worlds - 1}")
        if not self.forkable:
            raise WorldOpError(f"this scene keeps state without a per-world axis ({self._snap.whole_arrays}); "
                               "world copies need per-world state (no SPH liquid)")

    def _save_rows(self, world: int) -> dict[str, Any]:
        """One world's physics rows (per snapshot slot) plus the env's per-world bookkeeping rows, on the host."""
        with wp.ScopedDevice(self.env.device):
            self._snap.save(self.engine.state, _mask(world, self.worlds, self.env.device))
        physics = {}
        for slot in self._snap.slots:
            idx = np.nonzero(self._maps[slot.name] == world)[0]
            physics[slot.name] = _slot_view(slot)[idx].copy()
        return {"physics": physics, "env": _env_rows(self.env, world)}

    def _load_rows(self, rows: dict[str, Any], world: int) -> None:
        for slot in self._snap.slots:
            idx = np.nonzero(self._maps[slot.name] == world)[0]
            words = _slot_view(slot)
            words[idx] = rows["physics"][slot.name]
            slot.store.assign(wp.array(words.ravel(), dtype=slot.store.dtype, device="cpu"))
        mask = _mask(world, self.worlds, self.env.device)
        with wp.ScopedDevice(self.env.device):
            self._snap.restore(self.engine.state, mask)
            latch = getattr(self.engine, "_latch_snap", None)
            if latch is not None:
                latch.save(self.engine.state, mask)
            if hasattr(self.engine.stepper, "reset_contact_matching"):
                self.engine.stepper.reset_contact_matching(mask)
        self.env.health.reset(mask)
        _set_env_rows(self.env, rows["env"], world)


def _mask(world: int, worlds: int, device: str) -> wp.array:
    m = np.zeros(worlds, np.int32)
    m[int(world)] = 1
    return wp.array(m, dtype=wp.int32, device=device)


def _slot_view(slot: Any) -> np.ndarray:
    """The snapshot slot's stored words as a host (items, words) array."""
    return slot.store.numpy().reshape(slot.n_items, slot.words) if slot.item_world is not None else \
        slot.store.numpy().reshape(1, -1)


def _nbytes(rows: dict[str, Any]) -> int:
    return int(sum(a.nbytes for a in rows["physics"].values()))


# --- env bookkeeping rows ---------------------------------------------------------------------------------------------

def _holders(env: Any) -> list[tuple[str, Any]]:
    return [("stats", env.stats), ("start", env.start), ("goals", env.goals)]


def _per_world_fields(obj: Any, worlds: int) -> list[tuple[Any, str]]:
    """(container, key) of every array field with a leading world axis, including dict values one level down."""
    out = []
    for key, val in vars(obj).items():
        if _is_world_array(val, worlds):
            out.append((obj, key))
        elif isinstance(val, dict):
            out += [(val, k) for k, v in val.items() if _is_world_array(v, worlds)]
    return out


def _is_world_array(val: Any, worlds: int) -> bool:
    shape = getattr(val, "shape", None)
    return shape is not None and len(shape) >= 1 and int(shape[0]) == worlds and not isinstance(val, wp.array)


def _get(container: Any, key: str) -> Any:
    return container[key] if isinstance(container, dict) else getattr(container, key)


def _env_rows(env: Any, world: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, holder in _holders(env):
        for i, (container, key) in enumerate(_per_world_fields(holder, env.num_envs)):
            rows[f"{name}.{key}.{i}"] = _host(_get(container, key)[world])
    idx, target = _target_rows(env, world)
    if idx is not None:
        rows["writer.target"] = _host(target[idx])
    return rows


def _set_env_rows(env: Any, rows: dict[str, Any], world: int) -> None:
    for name, holder in _holders(env):
        for i, (container, key) in enumerate(_per_world_fields(holder, env.num_envs)):
            arr = _get(container, key)
            val = rows.get(f"{name}.{key}.{i}")
            if val is not None:
                arr[world] = _like(arr, val)
    idx, target = _target_rows(env, world)
    if idx is not None and "writer.target" in rows:
        target[idx] = _like(target, rows["writer.target"])


def _target_rows(env: Any, world: int) -> tuple[Any, Any]:
    """Rows of the action writer's control array that belong to ``world`` (its joint targets), and the array as a
    torch/NumPy view. (None, None) when the writer keeps no per-world map."""
    w = env.writer
    maps = getattr(w, "_world", None)
    if not isinstance(maps, dict) or w.kind not in maps or w.target is None:
        return None, None
    rows, ids = (_host(x) for x in maps[w.kind])
    target = wp.to_torch(w.target) if isinstance(w.target, wp.array) and "cuda" in str(w.target.device) else (
        w.target.numpy() if isinstance(w.target, wp.array) else w.target)
    return rows[ids == world], target


def _host(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().copy()
    if isinstance(x, wp.array):
        return x.numpy().copy()
    return np.array(x, copy=True)


def _like(ref: Any, val: np.ndarray) -> Any:
    if hasattr(ref, "detach"):
        import torch  # noqa: PLC0415
        return torch.as_tensor(val, device=ref.device, dtype=ref.dtype)
    return val
