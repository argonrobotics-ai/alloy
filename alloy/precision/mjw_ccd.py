# SPDX-License-Identifier: Apache-2.0
"""MuJoCo Warp's CCD scratch pool, sized per world (SCR-WP8-7 M1; coordinator ruling 2026-11-06 13:32Z).

Every MuJoCo Warp collision call allocates its EPA / multi-CCD scratch for ``Data.naccdmax`` slots
(``collision_convex.py:1314-1436``, ``wp.empty``): transient in eager mode, held by the CUDA graph pool when
captured. Newton builds the Data with the default ``naccdmax = naconmax`` (``io.py:1908``), i.e. ``nconmax`` slots per
world: 256 for T12, 4,996 B each, 1.22 MiB per world. That made the 16,384-world T12 run fail allocating ``epa_pr``
(9.1 GB; WP8, [internal note] §3). T12 uses at most 6 CCD slots per world; at 32 slots per
world WP8 measured 0 overflow and the same speed at 64 worlds (peak 146.6 -> 78.5 MiB, graph pool 96 -> 32 MiB).

A MuJoCo entry asks for it with the kwarg ``CCD_KEY`` (slots per world; ``with_ccd_slots`` puts it on a built scene's
MuJoCo entries, e.g. T12's ``T12_CCD_SLOTS``), popped by ``alloy.scene.builtin_plugins``,
which calls ``set_ccd_slots`` right after construction, before any step or graph capture (the scratch is sized at
each collision call, so a captured graph keeps the size it was captured with). No kwarg leaves MuJoCo Warp's default.
Newton calls ``put_data`` once, at construction (``solver_mujoco.py:9182``), so nothing rebuilds the Data later.

Overflow: a CCD pair past the pool gets no contact and sets ``OverflowType.CCD`` in that world's ``Data.overflow``.
Health maps that bit to CONTACT_OVERFLOW, so the world is INVALID (``alloy.precision.health_adapters._mjw_bits``).
The CCD slot counter is shared by all worlds, so which world overflows first depends on the order the narrowphase
threads run. The pool has to hold every world's CCD pairs, and the overflow fails closed.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("alloy.precision")

CCD_KEY = "alloy:ccd_slots"   # MuJoCo entry kwarg: CCD scratch slots per world
T12_CCD_SLOTS = 32            # T12 uses <= 6 per world; WP8's 64-world probe: 0 overflow, same speed (SCR-WP8-7)


def with_ccd_slots(built: Any, slots_per_world: int) -> Any:
    """``built`` (a BuiltScene) with ``CCD_KEY`` on each MuJoCo entry that does not set it already."""
    import dataclasses

    from alloy.integration.built_copy import replace_built

    return replace_built(built, entries=[
        dataclasses.replace(e, solver_kwargs={CCD_KEY: int(slots_per_world), **dict(e.solver_kwargs)})
        if e.capability == "mujoco" else e for e in built.entries])


def set_ccd_slots(solver: Any, slots_per_world: int) -> int | None:
    """Size ``solver.mjw_data.naccdmax`` to ``slots_per_world`` x worlds (at most ``naconmax``, MuJoCo Warp's bound).

    Call before the solver's first step or graph capture. Returns the new ``naccdmax``, or None for a solver with no
    MuJoCo Warp Data (``use_mujoco_cpu``: MuJoCo C sizes its own)."""
    slots = int(slots_per_world)
    if slots < 1:
        raise ValueError(f"{CCD_KEY} must be >= 1 slot per world, got {slots_per_world!r}")
    data = getattr(solver, "mjw_data", None)
    if getattr(solver, "use_mujoco_cpu", False) or data is None or not hasattr(data, "naccdmax"):
        log.info("mjw_ccd: %s has no MuJoCo Warp Data; %s=%d ignored", type(solver).__name__, CCD_KEY, slots)
        return None
    old, total = int(data.naccdmax), min(slots * int(data.nworld), int(data.naconmax))
    data.naccdmax = total
    log.info("mjw_ccd: naccdmax %d -> %d (%d slots x %d worlds, naconmax %d)", old, total, slots, int(data.nworld),
             int(data.naconmax))
    return total
