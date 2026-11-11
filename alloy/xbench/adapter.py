# SPDX-License-Identifier: Apache-2.0
"""``install(protocol)``: put Alloy's solvers in ``newton.solvers`` for one xbench scene process (bench/xbench).

``bench/xbench/scenes/engine_shim.py --engine alloy`` calls ``install`` before it runs the unchanged scene script.
Every stock Newton solver that Alloy routes to (a registered ``SolverCapability`` whose ``solver_cls`` is the class
in ``newton.solvers``: SolverMuJoCo, SolverXPBD, SolverFeatherstone, SolverVBD, SolverKamino, ...) is replaced by
its Alloy subclass (``alloy.xbench.solvers``). The capabilities and WP4/WP5 adapters are registered first, so the
registry keeps the stock classes and the capability factories never build an Alloy subclass inside Alloy itself.

The integration profile comes from ``install(profile=...)``, else ``$ALLOY_XBENCH_PROFILE``, else ``standard``
(Alloy's default: full checks every step). At exit one ``XB_ALLOY_HEALTH:{json}`` line goes to stdout (the shim's
marker channel, read from the job log) with each solver's health summary.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy._vendor import alias_top_level_names, newton

from alloy.integration.profiles import profile as integration_profile
from alloy.xbench.solvers import XbenchSettings, alloy_solver_class, ledgers

log = logging.getLogger("alloy.xbench")

__all__ = ["HEALTH_MARKER", "PROFILE_ENV", "install", "installed", "uninstall"]

PROFILE_ENV = "ALLOY_XBENCH_PROFILE"
HEALTH_MARKER = "XB_ALLOY_HEALTH:"
REPO = Path(__file__).resolve().parents[2]
NOT_WIRED = ("latch (records kept, no world restored or frozen: the scene owns its states)",
             "WP3 contact layer and pair table (no Alloy cards in xbench scenes)",
             "energy check (its reference needs the scene's initial state before capture)")


@dataclass
class _Install:
    protocol: str
    settings: XbenchSettings
    stock: dict[str, type] = field(default_factory=dict)       # name -> stock class put back by ``uninstall``
    alloy: dict[str, type] = field(default_factory=dict)       # name -> Alloy subclass now in newton.solvers
    capabilities: dict[str, str] = field(default_factory=dict)  # name -> capability


_ACTIVE: _Install | None = None
_PROTOCOL: list[str] = []   # protocols installed in this process (the exit report names the last)
_EXIT_HOOK = False


def install(protocol: str, profile: str | None = None) -> dict[str, Any]:
    """Swap the routed solver classes in ``newton.solvers``; returns JSON-serialisable provenance."""
    global _ACTIVE
    if _ACTIVE is not None:
        uninstall()
    name = profile or os.environ.get(PROFILE_ENV) or "standard"
    settings = XbenchSettings(integration_profile(name))
    _register_alloy()
    alias_top_level_names()      # the unchanged scene script's `import newton` gets Alloy's copy
    inst = _Install(protocol, settings)
    for cap in _routed_capabilities():
        cls_name = cap.solver_cls.__name__
        inst.stock[cls_name] = cap.solver_cls
        inst.alloy[cls_name] = alloy_solver_class(cap.solver_cls, cap, settings)
        inst.capabilities[cls_name] = cap.name
        setattr(newton.solvers, cls_name, inst.alloy[cls_name])
    _ACTIVE = inst
    _PROTOCOL.append(protocol)
    _hook_exit()
    log.info("alloy.xbench installed for %s: profile %s, solvers %s", protocol, name, sorted(inst.alloy))
    return provenance(inst)


def uninstall() -> None:
    """Put the stock classes back (tests; a second ``install``)."""
    global _ACTIVE
    if _ACTIVE is None:
        return
    for cls_name, stock in _ACTIVE.stock.items():
        setattr(newton.solvers, cls_name, stock)
    _ACTIVE = None


def installed() -> dict[str, type]:
    """Solver name -> Alloy class currently in ``newton.solvers`` (empty when not installed)."""
    return dict(_ACTIVE.alloy) if _ACTIVE is not None else {}


def provenance(inst: _Install) -> dict[str, Any]:
    from alloy.precision.config import preset  # noqa: PLC0415 - WP5; no device query (no wp.init before the scene)

    p = inst.settings.profile
    cfg = preset(p.preset)
    return {"engine": "alloy", "adapter": "alloy.xbench", "protocol": inst.protocol, **_git(),
            "profile": p.name, "preset": p.preset,
            "health": {"level": cfg.checks.level, "every": int(cfg.checks.every),
                       "full_every": int(cfg.checks.full_every), "energy": "off", "latch": "off"},
            "solver_map": {n: f"{c.__module__}.{c.__qualname__} (capability {inst.capabilities[n]}, "
                              f"subclass of {inst.stock[n].__module__}.{inst.stock[n].__qualname__})"
                           for n, c in sorted(inst.alloy.items())},
            "constructor_args": "unchanged (stock signature, the scene's own arguments)",
            "not_wired": list(NOT_WIRED)}


def _register_alloy() -> None:
    """Capabilities (with the stock classes) and WP4/WP5 adapters, before any class is swapped."""
    from alloy.coupling.adapters import register_builtin_adapters as state_adapters  # noqa: PLC0415 - WP4
    from alloy.precision.health_adapters import register_builtin_adapters as health_adapters  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415 - WP2

    register_builtin_solvers()
    state_adapters()
    health_adapters()


def _routed_capabilities():
    """Registered capabilities whose ``solver_cls`` is the stock class in ``newton.solvers`` now, by name."""
    from alloy.core.registry import registered_solvers, solver  # noqa: PLC0415

    caps = [solver(n) for n in registered_solvers()]
    out = [c for c in caps if c.solver_cls is not None
           and getattr(newton.solvers, c.solver_cls.__name__, None) is c.solver_cls]
    return sorted(out, key=lambda c: c.solver_cls.__name__)


def _git() -> dict[str, Any]:
    """Alloy commit and whether the tree has local changes (None outside a git checkout)."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no", "--", "alloy")
    return {"alloy_commit": run("rev-parse", "HEAD"), "alloy_dirty": None if status is None else bool(status)}


def health_report() -> dict[str, Any]:
    """Every Alloy solver's health summary since import (device sync)."""
    out = []
    for led in ledgers():
        try:
            out.append(led.summary())
        except Exception as exc:  # noqa: BLE001 - a dead context at exit must not hide the other rows
            out.append({"solver": led.solver, "error": f"{type(exc).__name__}: {exc}"})
    return {"protocol": _PROTOCOL[-1] if _PROTOCOL else None, "solvers": out}


def _emit_health() -> None:
    """Exit hook: the health summary as one marker line on stdout (the harness's channel, not a log). Only while
    installed: the shim never uninstalls; tests do."""
    if _ACTIVE is None or not ledgers():
        return
    try:
        line = HEALTH_MARKER + json.dumps(health_report())
    except Exception as exc:  # noqa: BLE001
        log.warning("alloy.xbench health report failed: %s", exc)
        return
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _hook_exit() -> None:
    global _EXIT_HOOK
    if not _EXIT_HOOK:
        atexit.register(_emit_health)
        _EXIT_HOOK = True
