# SPDX-License-Identifier: Apache-2.0
"""Alloy solvers for the xbench scenes: each stock Newton solver class, subclassed with Alloy's per-step parts.

An xbench scene owns its loop (it calls ``solver.step`` itself, captures its own CUDA graphs and collides with its
own pipeline), so the Alloy column swaps the solver, not the loop. ``alloy_solver_class(stock, capability)`` builds a
subclass of the stock class that:

- takes exactly the stock constructor arguments (``super().__init__``, nothing added or changed: the protocol's
  settings stay the protocol's);
- keeps the stock class ``__name__`` (``SolverMuJoCo`` ...), because Newton and Alloy dispatch on it
  (``alloy.precision.health_adapters.solver_sources`` keys on ``type(solver).__name__``), and keeps every stock
  attribute (``mjw_model``, ``mjw_data``, ``mj_model``, ``_update_newton_state``, ``register_custom_attributes``,
  nested ``Config`` classes) by inheritance;
- defines ``reset`` only when the stock class does (``newton_newton.py`` probes ``"reset" in vars(cls)``);
- after every ``step`` runs WP5's ``HealthMonitor.update`` for the model, as ``IntegratedSimulation._exchange``
  does after ``stepper.step`` (WP12): NaN/overflow/penetration/joint checks per the integration profile's
  ``CheckPolicy``, kernels only (graph-safe: the scenes capture the very first step).

Not wired, because the scene owns the part: the latch (needs the caller's last-good snapshot, so records are
kept but no world is restored or frozen), the WP3 contact layer and pair table (no Alloy cards), the energy check
(its reference needs the scene's initial state before any capture). Under a captured graph the check schedule is
the host step count seen at capture (``IntegratedSimulation`` known limit): a scene that captures one step (N2,
GPUSimBench) checks on every replay even under ``every = 10``, and MJW's two alternating graphs (steps 0 and 1)
check on every other step.

Each solver leaves a ``HealthLedger`` (its record array and counters, no solver reference, so a scene can free old
models) for the exit report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.registry import SolverCapability
from alloy.integration.profiles import IntegrationProfile
from alloy.precision.types import HealthFlags, Severity

log = logging.getLogger("alloy.xbench")

__all__ = ["HealthLedger", "XbenchSettings", "alloy_solver_class", "is_alloy_solver", "ledgers"]


@dataclass(frozen=True)
class XbenchSettings:
    """What every Alloy solver of one install wires: the integration profile (precision preset + parts)."""

    profile: IntegrationProfile


@dataclass
class HealthLedger:
    """What the exit report reads for one solver: the WP5 record array and host counters."""

    solver: str
    capability: str
    worlds: int
    checks: str
    every: int
    determinism: str
    record: wp.array
    host_steps: int = 0   # ``step()`` calls on the host; replays of a captured step run the checks but are not counted

    def summary(self) -> dict[str, Any]:
        """Worlds per severity and the flags seen (copies the record to the host)."""
        h = self.record.numpy()
        sev = h["severity"].astype(np.int32)
        flags = int(np.bitwise_or.reduce(h["flags"].astype(np.uint32))) if len(h) else 0
        return {"solver": self.solver, "capability": self.capability, "worlds": self.worlds,
                "host_steps": self.host_steps,
                "checks": self.checks, "every": self.every, "determinism": self.determinism,
                "ok": int(np.count_nonzero(sev == int(Severity.OK))),
                "degraded": int(np.count_nonzero(sev == int(Severity.DEGRADED))),
                "invalid": int(np.count_nonzero(sev >= int(Severity.INVALID))),
                "flags": [f.name for f in HealthFlags if flags & int(f)]}


_LEDGERS: list[HealthLedger] = []


def ledgers() -> list[HealthLedger]:
    """Every ledger since import (oldest first)."""
    return list(_LEDGERS)


class _SolverHealth:
    """WP5 ``HealthMonitor`` for one solver's model, updated after each step."""

    def __init__(self, solver: Any, model: newton.Model, settings: XbenchSettings) -> None:
        from alloy.precision.health import HealthMonitor  # noqa: PLC0415 - WP5

        cfg = settings.profile.precision_config(str(model.device))
        self.monitor = HealthMonitor(model, cfg.checks, solvers=[solver], config=cfg, energy="off")
        self.ledger = HealthLedger(type(solver).__name__, solver.alloy_capability, int(model.world_count),
                                   cfg.checks.level, int(cfg.checks.every), self.monitor.det_mode.name,
                                   self.monitor.record)
        _LEDGERS.append(self.ledger)

    def after_step(self, state_in: newton.State, state_out: newton.State, contacts: newton.Contacts | None) -> None:
        self.monitor.update(self.ledger.host_steps, state_in, state_out, contacts)
        self.ledger.host_steps += 1


class _AlloySolverMixin:
    """Stock constructor + Alloy health; see the module docstring. ``alloy_capability`` names the route entry."""

    alloy_capability: str = ""
    alloy_settings: XbenchSettings | None = None
    alloy_stock: type | None = None

    def __init__(self, model: newton.Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(model, *args, **kwargs)
        settings = type(self).alloy_settings
        if settings is None:
            raise RuntimeError(f"{type(self).__module__}.{type(self).__name__} built outside alloy.xbench.install")
        self._alloy_health = _SolverHealth(self, model, settings)
        led = self._alloy_health.ledger
        log.info("alloy %s (%s): %d worlds, profile %s, checks %s every %d, determinism %s", led.solver,
                 led.capability, led.worlds, settings.profile.name, led.checks, led.every, led.determinism)

    def step(self, state_in: newton.State, state_out: newton.State, control: Any, contacts: Any, dt: float,
             **kwargs: Any) -> None:
        super().step(state_in, state_out, control, contacts, dt, **kwargs)
        health = self.__dict__.get("_alloy_health")
        if health is not None:
            health.after_step(state_in, state_out, contacts)

    @property
    def alloy_ledger(self) -> HealthLedger | None:
        health = self.__dict__.get("_alloy_health")
        return None if health is None else health.ledger


def alloy_solver_class(stock: type, cap: SolverCapability, settings: XbenchSettings) -> type:
    """``stock`` subclassed with ``_AlloySolverMixin``, same ``__name__``/``__qualname__``, module alloy.xbench."""
    ns: dict[str, Any] = {"__module__": __name__, "__qualname__": stock.__qualname__, "__doc__": stock.__doc__,
                          "alloy_capability": cap.name, "alloy_settings": settings, "alloy_stock": stock}
    if "reset" in vars(stock):
        def reset(self, *args: Any, **kwargs: Any) -> Any:
            """Stock ``reset``; health records stay sticky (the report counts worlds that were ever INVALID)."""
            return stock.reset(self, *args, **kwargs)

        ns["reset"] = reset
    return type(stock.__name__, (_AlloySolverMixin, stock), ns)


def is_alloy_solver(obj: object) -> bool:
    return isinstance(obj, _AlloySolverMixin)
