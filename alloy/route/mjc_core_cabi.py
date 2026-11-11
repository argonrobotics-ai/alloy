# SPDX-License-Identifier: Apache-2.0
"""The CPU route's calls into WP2's C extension (``alloy.route._mjc``, ABI ``alloy_mjc.h``; SCR-WP8-6).

:func:`load_extension` returns the ctypes library or None with the reason (the extension is not merged, cannot be
built here, or has another ABI / MuJoCo version). :class:`CWorld` holds one world's config and output structs (WP2's
``MjcConfig`` / ``MjcOut``, which keep their numpy buffers alive) and exposes the outputs under the names of
:class:`alloy.route.mjc_core_tick.WorldOut`, so the route reads both the same way.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any

import mujoco
import numpy as np

from alloy.core.errors import SolverOptionError

EXTENSION = "liballoy_mjc, compiled on first use"

log = logging.getLogger("alloy.route")
_FAILURE = ""                         # why a load() raised (kept: WP2's load() tries once per process)


def load_extension() -> tuple[Any, str]:
    """(library, "") or (None, why it is unavailable), whatever the failure: WP2's ``load()`` returns None on its
    BuildError, and anything else it raises (an unwritable cache, a broken compiler, a subprocess error) is caught
    here, so the route always falls back to its numpy step (F18)."""
    global _FAILURE
    try:
        from alloy.route import _mjc
    except ImportError as e:
        return None, f"not installed ({e})"
    if _FAILURE:
        return None, _FAILURE
    try:
        lib = _mjc.load()
    except Exception as e:  # noqa: BLE001 - any build or load failure leaves the route on numpy, never fails it
        _FAILURE = f"cannot build or load liballoy_mjc ({type(e).__name__}: {e})"
        log.debug("liballoy_mjc unavailable: %s", _FAILURE, exc_info=True)
        return None, _FAILURE
    return lib, ("" if lib is not None else str(_mjc.reason or "load() returned None"))


class CWorld:
    """One world's C config and outputs."""

    def __init__(self, lib: Any, *, noslip: Any, scope: Any, line: bool, body_map: np.ndarray, ncon_max: int,
                 traj_cap: int = 0) -> None:
        from alloy.route import _mjc

        self.lib = lib
        self.cfg = _mjc.MjcConfig(noslip=noslip, scope=scope, line=line, export_contacts=ncon_max > 0,
                                  pull_state=True, body_map=body_map)
        self.out = _mjc.MjcOut(ncon_max=ncon_max, nbody=int(body_map.size), traj_cap=traj_cap)
        self.errors, self.e_nomem = _mjc.ERRORS, _mjc.abi.E_NOMEM

    def __getattr__(self, name: str) -> Any:        # numpy buffers: geom, pos, frame, dist, force, body_q, ...
        return getattr(self.__dict__["out"], name)

    @property
    def ncon(self) -> int:
        return int(self.out.struct.ncon)

    @property
    def traj_cap(self) -> int:
        return int(self.out.struct.traj_cap)

    @property
    def ncon_max(self) -> int:
        return int(self.out.struct.ncon_max)

    def counters(self) -> dict[str, float]:
        return self.out.counters()

    def zero(self) -> None:
        s = self.out.struct
        for name in ("steps", "nonfinite", "contact_overflow", "noslip_slots", "noslip_sweeps", "noslip_overflow",
                     "line_fired"):
            setattr(s, name, 0)
        s.pen_max = s.qvel_max = 0.0

    def tick(self, m: mujoco.MjModel, d: mujoco.MjData, nstep: int) -> None:
        rc = self.lib.alloy_mjc_tick(ctypes.byref(self.cfg.struct), ctypes.c_void_p(m._address),
                                     ctypes.c_void_p(d._address), int(nstep), ctypes.byref(self.out.struct))
        if rc == self.e_nomem:   # the only code returned after stepping: the state advanced
            raise SolverOptionError(f"CPU route: alloy_mjc_tick returned {rc} ({self.errors.get(rc, 'unknown')}) after "
                                    f"stepping: the world's state advanced by {nstep} steps, one of them without "
                                    "the scoped noslip")
        if rc != 0:      # checked before the first step: nothing ran
            raise SolverOptionError(f"CPU route: alloy_mjc_tick returned {rc} ({self.errors.get(rc, 'unknown')}); "
                                    "nothing was stepped")


__all__ = ["EXTENSION", "CWorld", "load_extension"]
