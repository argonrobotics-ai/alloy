# SPDX-License-Identifier: Apache-2.0
"""Twin-side recording: a preallocated device log and per-world recording kernels (SPEC §3.10 `TwinLog`).

`DeviceTwinLog` holds `[W, T]` Warp buffers that twins fill from inside the stepping loop with kernels, so the loop
never syncs to the host; `series(name)` copies a channel to numpy once, after the run. Column `c` holds the state
after outer step `c * stride` (column 0 is the initial state). Per-world reductions run one thread per world over
that world's contiguous body range: no float atomics (SPEC D9), so results are bitwise repeatable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

log = logging.getLogger("alloy.bench")


class DeviceTwinLog:
    """Preallocated per-world sample buffers on one device; implements the `TwinLog` protocol."""

    def __init__(self, n_worlds: int, n_steps: int, dt: float, device: str, stride: int = 1) -> None:
        if n_worlds <= 0 or n_steps < 0 or stride <= 0:
            raise ValueError(f"bad log shape: worlds={n_worlds} steps={n_steps} stride={stride}")
        self.n_worlds = n_worlds
        self.n_steps = n_steps
        self.stride = stride
        self.dt = dt
        self.device = device
        self.n_samples = n_steps // stride + 1
        self._channels: dict[str, wp.array] = {}
        self._host: dict[str, np.ndarray] = {}

    def column(self, step: int) -> int:
        """Column for outer step `step`, or -1 when this step is not sampled."""
        if step % self.stride != 0 or step < 0 or step > self.n_steps:
            return -1
        return step // self.stride

    def times(self) -> np.ndarray:
        """Simulated time [s] of each column (computed from integers, never accumulated)."""
        return np.arange(self.n_samples, dtype=np.float64) * (self.stride * self.dt)

    def channel(self, name: str, dtype: type = wp.float32, fill: float = np.nan) -> wp.array:
        """The `[W, T]` device buffer `name`, allocated (filled with `fill`) on first use."""
        arr = self._channels.get(name)
        if arr is None:
            arr = wp.array(_filled((self.n_worlds, self.n_samples), dtype, fill), dtype=dtype, device=self.device)
            self._channels[name] = arr
        return arr

    def accumulator(self, name: str, dtype: type = wp.float64, fill: float = 0.0) -> wp.array:
        """A per-world `[W]` device buffer for running reductions (max gain, last value, ...)."""
        key = f"acc:{name}"
        arr = self._channels.get(key)
        if arr is None:
            arr = wp.array(_filled((self.n_worlds,), dtype, fill), dtype=dtype, device=self.device)
            self._channels[key] = arr
        return arr

    def names(self) -> list[str]:
        return sorted(self._channels)

    def series(self, name: str) -> np.ndarray:
        """Host copy of channel `name` (`[W, T]`) or accumulator `acc:<name>` (`[W]`); cached after the first read."""
        if name not in self._host:
            if name not in self._channels:
                raise KeyError(f"no channel {name!r}; recorded: {self.names()}")
            self._host[name] = self._channels[name].numpy()
        return self._host[name]


def _filled(shape: tuple[int, ...], dtype: type, fill: float) -> np.ndarray:
    """Host array for a Warp scalar or vector dtype (vectors get a trailing component axis)."""
    length = int(getattr(dtype, "_length_", 1))
    scalar = getattr(dtype, "_wp_scalar_type_", dtype)
    full = shape if length == 1 else (*shape, length)
    return np.full(full, fill, dtype=wp.dtype_to_numpy(scalar))


@dataclass(frozen=True)
class WorldRanges:
    """Contiguous `[start, end)` body ranges per world, as `begin_world`/`end_world` lays them out."""

    start: wp.array
    end: wp.array
    n_worlds: int

    @staticmethod
    def bodies(model) -> "WorldRanges":
        bw = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
        return WorldRanges._from_item_world(bw, model.world_count, model.device)

    @staticmethod
    def _from_item_world(item_world: np.ndarray, n_worlds: int, device) -> "WorldRanges":
        start = np.zeros(n_worlds, np.int32)
        end = np.zeros(n_worlds, np.int32)
        for w in range(n_worlds):
            idx = np.flatnonzero(item_world == w)
            if idx.size and not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
                raise ValueError(f"world {w} items are not contiguous; build each world inside begin_world/end_world")
            start[w], end[w] = (idx[0], idx[-1] + 1) if idx.size else (0, 0)
        return WorldRanges(wp.array(start, dtype=wp.int32, device=device),
                           wp.array(end, dtype=wp.int32, device=device), n_worlds)


@wp.kernel
def _rec_body_pos(body_q: wp.array(dtype=wp.transform), bodies: wp.array(dtype=wp.int32), col: int,
                  out_x: wp.array2d(dtype=wp.float32), out_y: wp.array2d(dtype=wp.float32),
                  out_z: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    p = wp.transform_get_translation(body_q[bodies[w]])
    out_x[w, col] = p[0]
    out_y[w, col] = p[1]
    out_z[w, col] = p[2]


@wp.kernel
def _rec_body_vel(body_qd: wp.array(dtype=wp.spatial_vector), bodies: wp.array(dtype=wp.int32), col: int,
                  out_x: wp.array2d(dtype=wp.float32), out_y: wp.array2d(dtype=wp.float32),
                  out_z: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    qd = body_qd[bodies[w]]
    out_x[w, col] = qd[0]  # Newton public spatial vectors are (linear, angular), world frame, COM velocity
    out_y[w, col] = qd[1]
    out_z[w, col] = qd[2]


def record_body_position(log: DeviceTwinLog, prefix: str, body_q: wp.array, bodies: wp.array, step: int) -> None:
    """Write the world position of one body per world into channels `<prefix>x|y|z` at this step's column."""
    col = log.column(step)
    if col < 0:
        return
    outs = [log.channel(prefix + a) for a in "xyz"]
    wp.launch(_rec_body_pos, dim=log.n_worlds, inputs=[body_q, bodies, col], outputs=outs, device=log.device)


def record_body_velocity(log: DeviceTwinLog, prefix: str, body_qd: wp.array, bodies: wp.array, step: int) -> None:
    """Write the COM linear velocity of one body per world into channels `<prefix>x|y|z`."""
    col = log.column(step)
    if col < 0:
        return
    outs = [log.channel(prefix + a) for a in "xyz"]
    wp.launch(_rec_body_vel, dim=log.n_worlds, inputs=[body_qd, bodies, col], outputs=outs, device=log.device)


__all__ = ["DeviceTwinLog", "WorldRanges", "record_body_position", "record_body_velocity"]
