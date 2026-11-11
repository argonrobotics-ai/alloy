# SPDX-License-Identifier: Apache-2.0
"""Open simulations by id, for the tool server and ``alloy batch`` (one process, many calls).

An operation that needs a simulation takes ``sim`` (an id from ``build``) or ``scene`` (it builds one first). With
neither, it uses the simulation built last. Example::

    from alloy.cli.session import Session
    s = Session()
    sid, sim = s.open({"scene": "demo_peg_d12", "worlds": 2, "device": "cpu"})
    s.get({"sim": sid}) is sim        # True
    s.close_all()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from alloy.cli.errors import SessionError

log = logging.getLogger("alloy.cli")

MAX_SIMS = 8
#: build arguments every simulation-taking operation accepts, passed to ``alloy.build``
BUILD_KEYS = ("worlds", "device", "seed", "profile", "contacts")


def default_builder(scene: Any, **kwargs: Any) -> Any:
    """``alloy.build`` (imported on first use). Example: ``default_builder(scene, worlds=1, device="cpu")``."""
    from alloy.api._build import build  # noqa: PLC0415

    return build(scene, **kwargs)


class Session:
    """Simulations kept open between calls. ``builder`` builds a Scene into a Sim (injectable for tests).

    Example: ``Session(builder=my_fake_build)``."""

    def __init__(self, builder: Callable[..., Any] = default_builder, max_sims: int = MAX_SIMS) -> None:
        self.builder, self.max_sims = builder, max_sims
        self.sims: dict[str, Any] = {}
        self.last: str | None = None
        self._count = 0

    def open(self, args: dict[str, Any], sim_id: str | None = None) -> tuple[str, Any]:
        """Build ``args["scene"]`` (with the build keys in ``args``) and keep it under ``sim_id`` (or "sim<k>").

        Example: ``sid, sim = session.open({"scene": "demo_peg_d12", "worlds": 4})``."""
        from alloy.cli.scenes import to_scene  # noqa: PLC0415

        if sim_id is not None and sim_id in self.sims:
            raise SessionError(f"a simulation called {sim_id!r} is already open", f"close it first ({{\"op\": "
                               f"\"close\", \"sim\": \"{sim_id}\"}}) or pick another id", field="id")
        if len(self.sims) >= self.max_sims:
            raise SessionError(f"{len(self.sims)} simulations are open (the limit is {self.max_sims})",
                               f"close one first: open ones are {sorted(self.sims)}", field="sim")
        scene = to_scene(args["scene"], library=args.get("library") or (), liquid_accuracy=args.get("liquid_accuracy"))
        kwargs = {k: args[k] for k in BUILD_KEYS if args.get(k) is not None}
        sim = self.builder(scene, **kwargs)
        self._count += 1
        sid = sim_id or f"sim{self._count}"
        self.sims[sid], self.last = sim, sid
        log.info("opened %s: %s", sid, sim)
        return sid, sim

    def get(self, args: dict[str, Any]) -> tuple[str, Any]:
        """The simulation an operation runs on: ``args["sim"]``, else a new one from ``args["scene"]``, else the
        last one built. Example: ``sid, sim = session.get({"sim": "sim1"})``."""
        sid = args.get("sim")
        if sid is not None:
            if sid not in self.sims:
                raise SessionError(f"no open simulation {sid!r}", self._open_hint(), field="sim")
            return sid, self.sims[sid]
        if args.get("scene") is not None:
            return self.open(args)
        if self.last is not None and self.last in self.sims:
            return self.last, self.sims[self.last]
        raise SessionError("no simulation to run on", 'pass scene="..." (it is built first) or build one with '
                           '{"op": "build", "scene": "..."} and pass its id as sim', field="sim")

    def close(self, sid: str | None) -> str:
        """Close one simulation (``sid`` None: the last one). Example: ``session.close("sim1")``."""
        sid = sid if sid is not None else self.last
        if sid is None or sid not in self.sims:
            raise SessionError(f"no open simulation {sid!r}", self._open_hint(), field="sim")
        sim = self.sims.pop(sid)
        _close(sim)
        if self.last == sid:
            self.last = next(reversed(self.sims), None) if self.sims else None
        return sid

    def close_all(self) -> None:
        """Close every open simulation (end of a batch or server). Example: ``session.close_all()``."""
        for sid in list(self.sims):
            _close(self.sims.pop(sid))
        self.last = None

    def _open_hint(self) -> str:
        if self.sims:
            return f"use one of the open ids {sorted(self.sims)}"
        return 'build one first with {"op": "build", "scene": "..."}, or pass scene="..." to this call'


def _close(sim: Any) -> None:
    try:
        sim.close()
    except Exception as exc:  # noqa: BLE001 - closing must not fail a batch
        log.warning("closing a simulation failed: %s", exc)
