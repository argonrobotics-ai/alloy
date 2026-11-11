# SPDX-License-Identifier: Apache-2.0
"""Configuration of the CPU route on the MuJoCo C core (SCR-WP8-6): how an entry asks for it and what it runs.

**Choosing it.** The ``mujoco`` entry factory (``alloy.scene.builtin_plugins._mujoco``) builds the route for an entry
on a CPU model when the entry kwarg ``alloy:cpu_route`` (:data:`CPU_ROUTE_KEY`) is ``"mjc"``, or when it is unset and
the process setting ``ALLOY_CPU_ROUTE`` (:data:`CPU_ROUTE_ENV`) is ``"mjc"`` or unset: :data:`DEFAULT_CPU_ROUTE` is
``"mjc"`` for users (SCR-WP8-8, coordinator ruling 2026-11-06 14:18Z). ``"warp"`` keeps MuJoCo Warp on CPU (the
warp-cpu route). The test suite runs on ``"warp"`` (``tests/conftest.py`` sets ``ALLOY_CPU_ROUTE=warp``): the CPU
tests stand in for the GPU kernels; the route's own tests ask for ``"mjc"`` explicitly.

**What it runs** (:class:`RouteConfig`): the CPU-route solver settings, the scoped noslip spec (the scene's noslip
request, scope "robot", or the static hold, scope "all"; None = none), the box-box line contacts, the per-world
contact export capacity, an optional per-step trajectory buffer, and the plug-in backend ("auto": WP2's C extension
when it loads, else numpy; "numpy": numpy always, for parity tests and A/B rows).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any

from alloy.contact.noslip import ALL, NoslipScope, NoslipSpec, build_scope
from alloy.core.errors import SolverOptionError
from alloy.route.mjc_core_models import RouteSettings

CPU_ROUTE_KEY = "alloy:cpu_route"
CPU_ROUTE_ENV = "ALLOY_CPU_ROUTE"
MJC, WARP = "mjc", "warp"
DEFAULT_CPU_ROUTE = MJC     # SCR-WP8-8: users get the MuJoCo C route on CPU; tests/conftest.py pins the suite to warp
NCON_MAX = 256              # exported contacts per world (T12's nconmax; MuJoCo C itself has no cap)
BACKENDS = ("auto", "numpy")


def requested_route(kw: dict[str, Any]) -> str:
    """Pop the entry's ``alloy:cpu_route`` and resolve it against ``ALLOY_CPU_ROUTE`` (module docstring)."""
    value = kw.pop(CPU_ROUTE_KEY, None)
    value = os.environ.get(CPU_ROUTE_ENV, DEFAULT_CPU_ROUTE) if value is None else value
    value = str(value).strip().lower()
    if value not in (MJC, WARP):
        raise SolverOptionError(f"{CPU_ROUTE_KEY} / {CPU_ROUTE_ENV} must be {MJC!r} or {WARP!r}, got {value!r}")
    return value


@dataclass(frozen=True)
class RouteConfig:
    """The route's plug-in choices (module docstring)."""

    settings: RouteSettings = field(default_factory=RouteSettings)
    noslip: NoslipSpec | None = None
    line: bool = True
    ncon_max: int = NCON_MAX
    traj_cap: int = 0
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise SolverOptionError(f"CPU route backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.noslip is not None and self.noslip.iterations < 1:
            raise SolverOptionError(f"CPU route noslip needs iterations >= 1, got {self.noslip}")

    def replace(self, **kw: Any) -> RouteConfig:
        return dataclasses.replace(self, **kw)

    def scope(self, mj_model: Any) -> NoslipScope:
        spec = self.noslip
        return build_scope(mj_model, spec.scope, spec.pairs, spec.surfaces)

    def max_slots(self) -> int:
        """Slots per world, as ScopedNoslip and the C config: rows_per_world, else 64 for scope "robot" without
        surfaces, else unbounded."""
        spec = self.noslip
        if spec.rows_per_world is not None:
            return int(spec.rows_per_world)
        return 64 if spec.scope != ALL and not spec.surfaces else 2**31 - 1


__all__ = ["BACKENDS", "CPU_ROUTE_ENV", "CPU_ROUTE_KEY", "DEFAULT_CPU_ROUTE", "MJC", "NCON_MAX", "WARP", "RouteConfig",
           "requested_route"]
