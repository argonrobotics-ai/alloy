# SPDX-License-Identifier: Apache-2.0
"""What an env's reset removes besides the engine state, as ``sim.reset()`` does.

A reset world starts with no ``sim.apply_wrench`` load (the other worlds keep theirs); a reset of every world also
stops every ``sim.drive`` (a drive moves its object in every world, so it keeps running through a reset of some
worlds); and the engine's second state takes the restored one, so a row no solver writes does not come back from
before the reset. Example::

    before_reset(sim, every)          # ahead of engine.reset
    after_reset(sim, body_world, mask, every)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("alloy.envs")


def before_reset(sim: Any, every: bool) -> None:
    """A reset of every world: stop every drive, as ``sim.reset()`` does. Example: ``before_reset(sim, True)``."""
    if every:
        sim.stop_drives()


def after_reset(sim: Any, body_world: np.ndarray, mask: np.ndarray, every: bool) -> None:
    """The reset worlds' wrenches off, and the engine's second state matched to the restored one.

    ``body_world``: the world of each body row; ``mask``: (W,) bool, the worlds reset.
    Example: ``after_reset(sim, layout.world_of["body"], mask, False)``."""
    io = getattr(sim, "_io", None)
    if io is None:
        return
    if every:
        io.clear_wrenches()
    elif io.wrench_rows:
        rows = [r for r in io.wrench_rows if 0 <= body_world[r] < len(mask) and mask[body_world[r]]]
        _clear_rows(sim.engine, io, np.asarray(rows, np.int64))
    io.match_states()


def _clear_rows(engine: Any, io: Any, rows: np.ndarray) -> None:
    """Zero the held wrench of body ``rows``: in both of the engine's states and in health's booking of its work."""
    if not len(rows):
        return
    from alloy.precision.health_work import WRENCH_ATTR  # noqa: PLC0415

    for st in (getattr(engine, "state", None), getattr(engine, "_next", None)):
        if st is not None and getattr(st, "body_f", None) is not None:
            host = st.body_f.numpy()
            host[rows] = 0.0
            st.body_f.assign(host)
    ww = getattr(engine, WRENCH_ATTR, None)
    if ww is not None:
        ww.hold(rows, np.zeros(3, np.float32), np.zeros(3, np.float32))
    io.wrench_rows.difference_update(int(r) for r in rows)
    log.debug("reset: removed the held wrench of %d body rows", len(rows))


__all__ = ["after_reset", "before_reset"]
