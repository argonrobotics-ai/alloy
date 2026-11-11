# SPDX-License-Identifier: Apache-2.0
"""World health for environments: which worlds are INVALID (crashed and frozen by the engine), read on the device,
and clearing a world's health record when the environment resets it.

The engine freezes a world that breaks (NaN, explosion, deep overlap) at its last good state. The environment ends
that world's episode (``info["invalid"]``) and resets it. Example::

    hw = WorldHealth(sim.engine, xp)
    bad = hw.invalid()          # (W,) bool on the simulation device, no host copy
    hw.reset(world_mask)        # before engine.reset(world_mask)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.envs")

INVALID = 2     # alloy.precision.types.Severity.INVALID


@wp.kernel
def _severity_kernel(rec: wp.array2d(dtype=wp.uint8), offset: int, out: wp.array(dtype=wp.int32)):
    w = wp.tid()
    b0 = wp.int32(rec[w, offset])
    b1 = wp.int32(rec[w, offset + 1])
    b2 = wp.int32(rec[w, offset + 2])
    b3 = wp.int32(rec[w, offset + 3])
    out[w] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


@wp.kernel
def _masked_bytes_kernel(src: wp.array2d(dtype=wp.uint8), mask: wp.array(dtype=wp.int32),
                         dst: wp.array2d(dtype=wp.uint8)):
    w, b = wp.tid()
    if mask[w] != 0:
        dst[w, b] = src[w, b]


def byte_rows(arr: wp.array, worlds: int) -> wp.array | None:
    """A (worlds, bytes per world) uint8 view of a world-major Warp array, or None if it is not world-major."""
    if arr is None or arr.ndim < 1 or arr.shape[0] != worlds or not arr.is_contiguous:
        return None
    per_world = int(arr.strides[0])          # bytes per world (contiguous, world-major)
    if per_world <= 0:
        return None
    return wp.array(ptr=arr.ptr, dtype=wp.uint8, shape=(worlds, per_world), device=arr.device)


class WorldHealth:
    """Device-side INVALID mask from the engine's health monitor, and a masked clear of its records."""

    def __init__(self, engine: Any, xp: Any) -> None:
        self.engine, self.xp = engine, xp
        self.monitor = getattr(engine, "health", None)
        self.worlds = int(engine.built.world_count)
        self._out = wp.zeros(self.worlds, dtype=wp.int32, device=engine.model.device)
        self._record, self._offset = self._severity_view()
        self._saved = self._save_records()

    @property
    def monitored(self) -> bool:
        """True when the engine checks health (otherwise ``invalid()`` is all False). Example: ``hw.monitored``."""
        return self._record is not None

    def _severity_view(self) -> tuple[wp.array | None, int]:
        rec = getattr(self.monitor, "record", None)
        if rec is None or not hasattr(rec.dtype, "numpy_dtype"):
            return None, 0
        d = rec.dtype.numpy_dtype()
        if "severity" not in d["names"]:
            return None, 0
        k = d["names"].index("severity")
        if np.dtype(d["formats"][k]).itemsize != 4:
            return None, 0
        return byte_rows(rec, self.worlds), int(d["offsets"][k])

    def invalid(self) -> Any:
        """(W,) bool: worlds the engine marked INVALID (frozen). Example: ``hw.invalid()``."""
        if self._record is None:
            return self.xp.zeros(self.worlds, "bool")
        wp.launch(_severity_kernel, dim=self.worlds, inputs=[self._record, self._offset], outputs=[self._out],
                  device=self._out.device)
        return self.xp.view(self._out) >= INVALID

    # --- reset ----------------------------------------------------------------------------------------------------

    def _save_records(self) -> list[tuple[wp.array, wp.array, wp.array]]:
        """Build-time copies of the monitor's per-world records (used until the engine clears them itself on reset)."""
        mon = self.monitor
        if mon is None or hasattr(mon, "reset_worlds") or not hasattr(mon, "state_arrays"):
            return []
        out = []
        for name, arr in mon.state_arrays().items():
            rows = byte_rows(arr, self.worlds)
            if rows is None:
                log.debug("health array %s is not world-major; left as is on reset", name)
                continue
            out.append((arr, rows, wp.clone(rows)))
        return out

    def reset(self, world_mask: wp.array) -> None:
        """Clear the health records of the masked worlds (int32 (W,) device mask). Example: ``hw.reset(mask)``."""
        mon = self.monitor
        if mon is not None and hasattr(mon, "reset_worlds"):
            mon.reset_worlds(world_mask)
            return
        for _arr, rows, saved in self._saved:
            wp.launch(_masked_bytes_kernel, dim=rows.shape, inputs=[saved, world_mask], outputs=[rows],
                      device=rows.device)


__all__ = ["INVALID", "WorldHealth", "byte_rows"]
