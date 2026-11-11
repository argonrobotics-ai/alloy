# SPDX-License-Identifier: Apache-2.0
"""Keep Warp's kernel-load lines and Newton's converter messages off stdout while alloy.io works.

Agents read stdout; library chatter there costs context. Inside ``quiet_engine()`` Warp's log level is at least
WARNING (its kernel-load timings print at INFO) and Newton's MuJoCo solver logger only passes warnings; both are
restored afterwards.

Example::

    from alloy.io.quiet import quiet_engine
    with quiet_engine():
        ...   # assemble, convert
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

NEWTON_LOGGERS = ("alloy._vendor.newton._src.solvers.mujoco.solver_mujoco", "alloy._vendor.newton._src.solvers.mujoco")


@contextlib.contextmanager
def quiet_engine() -> Iterator[None]:
    """Warp at log level WARNING and Newton's MuJoCo logger at WARNING for the block; previous settings restored."""
    import warp as wp  # noqa: PLC0415

    old_level = getattr(wp.config, "log_level", None)
    loggers = [logging.getLogger(n) for n in NEWTON_LOGGERS]
    old_levels = [lg.level for lg in loggers]
    try:
        if old_level is not None:
            wp.config.log_level = max(old_level, wp.LOG_WARNING)
        for lg in loggers:
            lg.setLevel(max(lg.level, logging.WARNING))
        yield
    finally:
        if old_level is not None:
            wp.config.log_level = old_level
        for lg, level in zip(loggers, old_levels):
            lg.setLevel(level)
