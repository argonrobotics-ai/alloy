# SPDX-License-Identifier: Apache-2.0
"""One task port's episode as a coroutine: the port's own run function steps, one control period at a time, under
the environment's control.

A task port (``alloy.tasks``) is one function that builds its simulation, scripts its tool or robot, scores the
result and returns the verdict. :class:`PortRun` runs that function in a worker thread and pauses it at every
``step_period()`` of every simulation it builds (through the plug-in harness or ``alloy.build``; other threads are
not affected). At a pause the script has written its command for the period, so the
environment can read it (the expert's action), write the policy's own over it, and resume: the period steps and the
script goes on to the next pause or returns its verdict. Only one of the two threads runs at a time.

Example::

    run = PortRun(lambda: paint_roll_strip.run_roller(device="cpu"))
    while run.start() if run.steps == 0 else run.resume(write=None):   # write=None keeps the script's command
        pass
    print(run.result["succeeded"])
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger("alloy.envs.ports")

_ACTIVE = threading.local()          # the PortRun whose worker is this thread
_LOCK = threading.Lock()
_ORIGINAL: list[Callable[..., Any]] = []


class PortAborted(BaseException):  # noqa: N818 - a signal, not an error: it unwinds the port's run on close/reset
    """Raised inside a paused port run to stop it (``PortRun.abort``)."""


def _install() -> None:
    """Wrap, once, the two places a port's simulation comes from: ``alloy.plugins.host.simulation`` (the plug-in
    harness) and ``alloy.api.sim.Sim`` (``alloy.build``; its engine). Simulations built on a port's worker thread are
    adopted by its run; other threads see no change."""
    from alloy.api import sim as api_sim  # noqa: PLC0415
    from alloy.plugins import host  # noqa: PLC0415

    with _LOCK:
        if _ORIGINAL:
            return
        original, init = host.simulation, api_sim.Sim.__init__
        _ORIGINAL.extend([original, init])

        def simulation(*args: Any, **kwargs: Any) -> Any:
            sim = original(*args, **kwargs)
            run = getattr(_ACTIVE, "run", None)
            if run is not None:
                run.adopt(sim)
            return sim

        def sim_init(self: Any, *args: Any, **kwargs: Any) -> None:
            init(self, *args, **kwargs)
            run = getattr(_ACTIVE, "run", None)
            if run is not None:
                self.engine._port_api_sim = self        # the camera draws its deformables (ports/display.py)
                run.adopt(self.engine)

        simulation.__doc__, sim_init.__doc__ = original.__doc__, init.__doc__
        simulation.__wrapped__, sim_init.__wrapped__ = original, init  # type: ignore[attr-defined]
        host.simulation, api_sim.Sim.__init__ = simulation, sim_init


class PortRun:
    """``fn()`` (a port's run function, no arguments) as a coroutine paused at each control period. ``start()`` runs
    it to its first pause; ``resume(write)`` applies ``write(sim)`` (the policy's command), steps the period and runs
    to the next pause. Both return True while paused, False once ``fn`` returned (``result``) or raised (re-raised).
    """

    def __init__(self, fn: Callable[[], Any], name: str = "port") -> None:
        _install()
        self.fn, self.name = fn, name
        self.sim: Any = None                 # the simulation that paused last
        self.result: Any = None
        self.steps = 0                       # control periods stepped
        self.finished = False
        self._error: BaseException | None = None
        self._write: Callable[[Any], None] | None = None
        self._abort = False
        self._to_env, self._to_worker = threading.Semaphore(0), threading.Semaphore(0)
        self._thread: threading.Thread | None = None

    # --- environment side -------------------------------------------------------------------------------------------

    def start(self) -> bool:
        if self._thread is not None:
            raise RuntimeError(f"port run {self.name} already started")
        self._thread = threading.Thread(target=self._work, name=f"alloy-port-{self.name}", daemon=True)
        self._thread.start()
        return self._wait()

    def resume(self, write: Callable[[Any], None] | None = None) -> bool:
        if self.finished:
            raise RuntimeError(f"port run {self.name} has finished: start a new one")
        self._write = write
        self._to_worker.release()
        return self._wait()

    def abort(self) -> None:
        """Stop a paused run (its thread unwinds with :class:`PortAborted`) and wait for it."""
        if self._thread is None or self.finished:
            return
        self._abort = True
        self._to_worker.release()
        self._thread.join(timeout=60.0)
        self.finished = True

    def _wait(self) -> bool:
        self._to_env.acquire()
        if self._error is not None:
            err, self._error = self._error, None
            raise err
        return not self.finished

    # --- worker side ------------------------------------------------------------------------------------------------

    def _work(self) -> None:
        _ACTIVE.run = self
        try:
            self.result = self.fn()
        except PortAborted:
            log.debug("port run %s aborted after %d periods", self.name, self.steps)
        except BaseException as e:  # noqa: BLE001 - handed to the environment's thread, raised there
            self._error = e
        finally:
            _ACTIVE.run = None
            self.finished = True
            self._to_env.release()

    def adopt(self, sim: Any) -> None:
        """Pause before each of ``sim``'s control periods (called by the wrapped factory on the worker thread)."""
        if getattr(sim, "_port_run", None) is self:
            return
        sim._port_run = self
        real = sim.step_period

        def step_period() -> None:
            self.sim = sim
            self._to_env.release()                # paused: the script's command is written
            self._to_worker.acquire()
            if self._abort:
                raise PortAborted
            if self._write is not None:
                self._write(sim)
            real()
            self.steps += 1

        sim.step_period = step_period


__all__ = ["PortAborted", "PortRun"]
