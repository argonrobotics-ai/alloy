# SPDX-License-Identifier: Apache-2.0
"""State keys an env reads through the Sim: the values ``sim.state()`` gives that a solver keeps outside the engine's
state arrays (a heated liquid's ``"<name>.temperature"``, a yielding sheet's, wire's or joint's
``"<name>.plastic_strain"``, a yielding joint's ``"<name>.rest"``), as backend arrays with the leading world axis, and
their values at each world's last reset for ``state.start``. Each read copies from the device and waits for it, so an
observation that holds one costs a copy per step. Example::

    keys = SimKeys(sim, xp, state_keys(layout))
    keys.capture(mask)                 # after a reset of the worlds in mask
    BatchState(..., derived=keys.live) # s["water.temperature"]; derived=keys.start for s.start[...]
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable
from typing import Any

log = logging.getLogger("alloy.envs")


class SimKeys:
    """The keys ``sim.state()`` has beyond ``known``, read through the Sim (module docstring)."""

    def __init__(self, sim: Any, xp: Any, known: Iterable[str]) -> None:
        self.sim, self.xp = sim, xp
        known = set(known) | {"t"}
        self.names = {k: k.rpartition(".")[0] for k in sim.state() if k not in known}
        self._start: dict[str, Any] = {}
        self.live: dict[str, Callable[[], Any]] = {k: functools.partial(self.read, k) for k in self.names}
        self.start: dict[str, Callable[[], Any]] = {k: functools.partial(self._at_start, k) for k in self.names}
        if self.names:
            log.info("env state reads %s through the Sim (a device copy per read)", sorted(self.names))

    def read(self, key: str) -> Any:
        """``key``'s value now, (W, ...). Example: ``keys.read("water.temperature")``."""
        return self.xp.asarray(self.sim.state([self.names[key]])[key])

    def capture(self, mask: Any) -> None:
        """Keep the current values of the worlds in the bool (W,) ``mask`` as their start values (after a reset)."""
        if not self.names:
            return
        now = self.sim.state(sorted(set(self.names.values())))
        for key in self.names:
            value = self.xp.asarray(now[key])
            old = self._start.get(key)
            if old is None:
                self._start[key] = value
                continue
            sel = mask.reshape(-1, *([1] * (value.ndim - 1))) if value.ndim > 1 else mask
            self._start[key] = self.xp.where(sel, value, old)

    def _at_start(self, key: str) -> Any:
        if key not in self._start:                       # not captured yet: the start is now
            self._start[key] = self.read(key)
        return self._start[key]


__all__ = ["SimKeys"]
