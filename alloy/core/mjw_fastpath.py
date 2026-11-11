# SPDX-License-Identifier: Apache-2.0
"""Host-side fast paths for MuJoCo Warp's Python glue (SCR-WP8-5, coordinator 2026-11-06 10:33Z: the Python-side call
cache is WP0's pin matter).

``collision_convex.collision`` (its ``_pair_count``, 68 calls a step on T12) and ``collision_primitive``'s narrowphase
(13 calls) call ``math.upper_trid_index``, a ``@wp.func``, from Python. Each such call resolves a Warp overload (~40 us
on host-1), about 3.3 ms of a T12 step's 5.6 ms of collision on warp-cpu, and the same host time per step on GPU
([internal note]). Here the two modules' own name ``upper_trid_index`` is rebound
to a plain-Python twin with the same integer result. The ``@wp.func`` itself is untouched, so kernels that use it (in
``math`` and elsewhere) compile as before; in the two rebound modules the name appears in Python code only, which
``tests/core/test_mjw_fastpath.py`` pins for each MuJoCo Warp pin.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("alloy.core")

#: MuJoCo Warp modules whose Python code calls ``upper_trid_index`` (never inside a kernel at the pin)
TARGETS: tuple[str, ...] = ("alloy._vendor.mujoco_warp._src.collision_convex", "alloy._vendor.mujoco_warp._src.collision_primitive")


def upper_trid_index(n: int, i: int, j: int) -> int:
    """``mujoco_warp._src.math.upper_trid_index`` in plain Python: the index of a_ij = a_ji in an upper triangular
    matrix, diagonal included. Example: ``upper_trid_index(3, 2, 0) == 2``."""
    if j < i:
        i, j = j, i
    return (i * (2 * n - i - 1)) // 2 + j


def install(module_name: str) -> None:
    """Rebind ``upper_trid_index`` in the loaded MuJoCo Warp module ``module_name`` (idempotent; a module without the
    name is left alone and logged, so a MuJoCo Warp bump that drops it does not break the import)."""
    module = sys.modules.get(module_name)
    if module is None:
        return
    current = getattr(module, "upper_trid_index", None)
    if current is None:
        log.warning("mjw_fastpath: %s has no upper_trid_index (MuJoCo Warp changed); nothing rebound", module_name)
        return
    if current is not upper_trid_index:
        module.upper_trid_index = upper_trid_index


__all__ = ["TARGETS", "install", "upper_trid_index"]
