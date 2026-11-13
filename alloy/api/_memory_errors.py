# SPDX-License-Identifier: Apache-2.0
"""Warp's device-memory failures after the build as API errors.

Two places allocate past the build and could leave the API with a bare ``RuntimeError: Failed to allocate ... bytes``
from Warp: ``sim.check()``, whose contact probe collides every world at once beside the simulation's own buffers (512
worlds of 16 decomposed bowls on an L4), and ``sim.step()``, whose first periods run eagerly and allocate MuJoCo
Warp's collision scratch (2,048 worlds of 4 decomposed bowls, 6 boxes and a clamped hose: 4.09 GB of EPA scratch
on an L4 already holding 13.7 GB). A failed CUDA graph is not one of them: stepping already falls back to smaller
graphs, then to none.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from alloy.api import errors as _errors

ESTIMATE = 'alloy.estimate(scene, worlds)["memory_bytes"] gives the memory before you build'


def oom_error(sim: Any, exc: BaseException, what: str, fix: str) -> _errors.SimulationError | None:
    """A ``SimulationError`` naming ``what`` ran out of device memory at the sim's world count, with ``fix``, when
    ``exc`` is Warp's or CUDA's out-of-memory ``RuntimeError`` (``errors.limit_error``'s rule); else None: any other
    error, Alloy's own included, keeps its type.
    Example: ``err = oom_error(sim, exc, "sim.step()", "build fewer worlds")``."""
    worlds = int(getattr(sim, "worlds", 0) or 0)
    if not isinstance(exc, RuntimeError) or _errors.limit_error(exc, worlds) is None:
        return None
    err = _errors.SimulationError(f"{what} ran out of device memory at {worlds:,} worlds ({type(exc).__name__}: "
                                  f"{exc})", fix, field="worlds")
    err.__cause__ = exc                               # as ``raise err from exc``, for ``raise step_oom(...) or exc``
    return err


@contextlib.contextmanager
def probe_memory(sim: Any) -> Iterator[None]:
    """``sim.check()``'s contact probe: inside the block, Warp's out-of-memory failure raises ``oom_error``'s
    ``SimulationError``; every other exception passes unchanged. Example: ``with probe_memory(sim): check_sim(sim)``."""
    try:
        yield
    except RuntimeError as exc:
        err = oom_error(sim, exc, "sim.check() (its contact probe collides every world at once, beside the "
                        "simulation)", f"sim.health() gives every world's health flags without the probe; or build "
                        f"fewer worlds ({ESTIMATE})")
        if err is None:
            raise
        raise err from exc


def step_oom(sim: Any, exc: BaseException) -> _errors.SimulationError | None:
    """``oom_error`` for ``sim.step()``, whose hot path keeps a plain try.
    Example: ``except RuntimeError as exc: raise step_oom(sim, exc) or exc``."""
    return oom_error(sim, exc, "sim.step() (the step's collision and solver scratch)",
                     f"build fewer worlds, or pass a smaller alloy.build(scene, contact_capacity=n) (its scratch grows "
                     f"with the contact buffer); {ESTIMATE}")
