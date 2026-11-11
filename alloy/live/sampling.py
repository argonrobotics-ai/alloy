# SPDX-License-Identifier: Apache-2.0
"""Sampling what the live view shows, without stalling the training thread.

- :class:`WorldNumbers`: per-world reward, episode return, success, done and health for ALL worlds. Accumulated on
  the device every step (a few elementwise ops), copied to the host about once a second.
- The picked worlds' 3D frames come from the engine's frame export (:mod:`alloy.live.frames`).

Device -> host copies are asynchronous (pinned buffers + a CUDA event); the returned packer waits for the event and
packs the frame in the link's sender thread, so the training thread never waits for a copy.

Example::

    nums = WorldNumbers(env)
    nums.add(reward, done, info)                # every step
    link.offer(Kind.NUMBERS, {"step": k}, nums.sample())    # once a second
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from alloy.live.gate import GATE
from alloy.live.wire import pack_arrays

Packer = Callable[[], tuple[dict[str, Any], bytes] | None]
FIELDS = ("reward", "ret", "success", "done", "health", "ep_ret", "ep_success", "paused")


def _to_host_async(t: Any) -> Callable[[], np.ndarray]:
    """Start a device->host copy; returns a function that waits for it (call it in another thread)."""
    if not getattr(t, "is_cuda", False):
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        arr = arr.copy()
        return lambda: arr
    import torch  # noqa: PLC0415

    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    host.copy_(t, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()

    def wait() -> np.ndarray:
        with GATE.background():              # no CUDA call may overlap the engine's graph capture
            ev.synchronize()
        return host.numpy()
    return wait


class WorldNumbers:
    """Per-world numbers of every world, accumulated on the device between samples."""

    def __init__(self, env: Any) -> None:
        self.env, self.xp, self.W = env, env.xp, int(env.num_envs)
        z = self._zeros
        self.reward, self.success, self.done = z(), z(), z()
        self.ep_ret, self.ep_success = z(), z()
        self.paused = z()
        self._ep_sum_t = self._ep_n_t = self._ep_succ_t = None
        self._flags = _FlagReader(env)

    def _zeros(self) -> Any:
        return self.xp.zeros(self.W)

    def add(self, reward: Any, done: Any, info: dict[str, Any]) -> None:
        """Fold one step in (device ops only)."""
        xp = self.xp
        self.reward = reward
        self.success = xp.float(info["success"]) if "success" in info else self.success
        d = xp.float(done)
        self.done = xp.where(done, 1.0, self.done)
        if "episode" in info and "_episode" in info:
            m = info["_episode"]
            r, s = info["episode"]["r"], xp.float(info["episode"]["success"])
            self.ep_ret = xp.where(m, r, self.ep_ret)
            self.ep_success = xp.where(m, s, self.ep_success)
            self._ep_sum_t = _acc(self._ep_sum_t, (r * d).sum())
            self._ep_n_t = _acc(self._ep_n_t, d.sum())
            self._ep_succ_t = _acc(self._ep_succ_t, (s * d).sum())

    def sample(self, paused: set[int]) -> Packer:
        """Start the copy of every world's numbers; the packer (sender thread) returns ``(header, payload)``."""
        xp, env = self.xp, self.env
        health = self._flags.health() if env.health.monitored else self._zeros()
        p = np.zeros(self.W, np.float32)
        p[list(paused)] = 1.0
        stack = xp.stack([xp.float(self.reward), env.stats.returns, self.success, self.done, health, self.ep_ret,
                          self.ep_success, xp.asarray(p)])
        totals = [t if t is not None else xp.zeros(()) for t in (self._ep_sum_t, self._ep_n_t, self._ep_succ_t)]
        wait_rows, wait_tot = _to_host_async(stack), _to_host_async(xp.stack(totals))
        self.done = self._zeros()
        self._ep_sum_t = self._ep_n_t = self._ep_succ_t = None

        def pack() -> tuple[dict[str, Any], bytes]:
            rows, tot = wait_rows(), wait_tot()
            meta, blob = pack_arrays({name: rows[i].astype(np.float32) for i, name in enumerate(FIELDS)})
            return {"episodes": float(tot[1]), "ep_return_sum": float(tot[0]), "ep_success_sum": float(tot[2]),
                    **meta}, blob
        return pack


class _FlagReader:
    """Per-world alarm level from the engine's health record: 2 INVALID, 1 any health flag other than the run-wide
    DET_DOWNGRADED notice (bitwise determinism unavailable, set in every world on GPU routes), 0 otherwise."""

    def __init__(self, env: Any) -> None:
        import warp as wp  # noqa: PLC0415

        self.env, self.off = env, None
        rec = getattr(env.health.monitor, "record", None)
        if env.health.monitored and rec is not None:
            d = rec.dtype.numpy_dtype()
            if "flags" in d["names"]:
                self.off = int(d["offsets"][d["names"].index("flags")])
                self.out = wp.zeros(int(env.num_envs), dtype=wp.int32, device=env.device)

    def health(self) -> Any:
        import warp as wp  # noqa: PLC0415
        from alloy.envs.health import _severity_kernel  # noqa: PLC0415 - reads 4 bytes per world at an offset
        from alloy.precision.types import HealthFlags  # noqa: PLC0415

        env, xp = self.env, self.env.xp
        invalid = env.health.invalid()
        if self.off is None:
            return xp.float(invalid) * 2.0
        wp.launch(_severity_kernel, dim=int(env.num_envs), inputs=[env.health._record, self.off], outputs=[self.out],
                  device=self.out.device)
        flags = xp.view(self.out) & ~int(HealthFlags.DET_DOWNGRADED)
        return xp.where(invalid, 2.0, xp.float(flags != 0))


def _acc(total: Any, x: Any) -> Any:
    return x if total is None else total + x
