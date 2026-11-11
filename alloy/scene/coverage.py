# SPDX-License-Identifier: Apache-2.0
"""CARD-1 / SB-9 v1 coverage: library objects that run from their card alone (no override, no hand tuning) and pass
their analytic law or self-test, over the object count (SPEC §6 SB-9, §7 CARD-1; target ≥ 80 %).

Per object, from the real CardLibrary and the real Router (rod self-test on, cached by card hash):
1. route: a RouteError or an `overridden` decision fails the object;
2. runs: the object alone on `table_top_plywood` assembles and `WORLDS` worlds step SETTLE_S + WINDOW_S on the device
   with a finite state. Dynamic objects sit DROP_M above the table in their most stable axis-aligned pose (largest
   tipping angle, `coverage_pose`); static fixtures sit on the table as authored, with a 30 mm beech cube (the
   probe) dropped onto their top;
3. the check of the route family (`coverage_checks`):
   - rigid / rigid_compliant / rigid_pile / panels / static: rest law. After SETTLE_S no body of the object (the probe
     for fixtures) moves more than τ, the router's tolerance for the object (default 0.1 mm), over the next WINDOW_S;
   - chain: the router's rod self-test (cantilever sag vs the elastica and release tip drop, ≤ 5 % each; CARD-7, ROD-2);
   - cloth_vbd / cloth_style3d: sheet check, a simply supported coupon of the card's material and thickness sags
     under self-weight like a Navier plate (≤ 15 %, CARD-8);
   - soft_vbd: A15 uniaxial Neo-Hookean at the card's Lamé constants (≤ 5 %);
   - liquid_sph: A13 hydrostatic wall force at the card's density, viscosity and SPH radius (≤ 3 %);
   - grain_mpm: R8 column collapse runout at aspect 2 vs Lube et al. (≤ 10 %);
   - paste_mpm: SPEC §5.3 has no paste law, so a paste object is not covered.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from alloy.cards import types as ct
from alloy.core.errors import AlloyError
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.scene.assembler import SceneAssembler
from alloy.scene.coverage_pose import UP, rest_pose, top_of
from alloy.scene.types import BuiltScene, ObjectHandle
from alloy.sim import Simulation

log = logging.getLogger("alloy.scene")

TABLE = "table_top_plywood"
PROBE = "cube_beech_30"
DROP_M = 1.0e-3                 # gap under a dynamic object at release [m]
SETTLE_S, WINDOW_S = 1.0, 0.25  # settle, then the rest window [s]
WORLDS = 4
TAU_DEFAULT = 1.0e-4            # the router's default tolerance [m] (router.py `card.loads.tolerance`)
TARGET = 0.80
REST_ROUTES = frozenset({"rigid", "rigid_compliant", "rigid_pile", "panels", "static"})


@dataclass(frozen=True)
class CoverageRow:
    object: str
    route: str                  # route value, or "error:<Type>"
    check: str                  # rest | rod_self_test | sheet_navier | neohookean | hydrostatic | runout | none
    runs: bool
    passed: bool
    value: float | None = None  # the check's error (rest: displacement [m]; the others relative)
    bound: float | None = None
    detail: str = ""
    wall_s: float = 0.0

    @property
    def covered(self) -> bool:
        return self.runs and self.passed


@dataclass(frozen=True)
class CoverageReport:
    rows: tuple[CoverageRow, ...]
    device: str
    worlds: int

    @property
    def covered(self) -> int:
        return sum(r.covered for r in self.rows)

    @property
    def rate(self) -> float:
        return self.covered / len(self.rows) if self.rows else math.nan

    def by_route(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.rows:
            c = out.setdefault(r.route, [0, 0])
            c[0] += int(r.covered)
            c[1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}


@dataclass
class _Run:
    """One object's assembled scene and what the checks read from it."""
    built: BuiltScene
    handle: ObjectHandle
    tracked: str                                # the object whose bodies the rest law watches
    displacement: float = math.nan
    finite: bool = False
    extra: dict = field(default_factory=dict)


# --- scenes --------------------------------------------------------------------------------------------------------

def scene_for(name: str, lib, worlds: int = WORLDS) -> tuple[ct.SceneCard, str]:
    """The object alone on the table (fixtures: with the probe on top); returns the card and the tracked object."""
    origin = (0.0, 0.0, 0.0, *UP)
    t_top = top_of(lib.object(TABLE), lib, origin)
    placements = [ct.Placement(TABLE, origin)]
    card = lib.object(name)
    tracked = name
    if card.motion == "static":
        pose = rest_pose(card, lib, t_top, 0.0, choose=False) if name != TABLE else origin
        if name != TABLE:
            placements.append(ct.Placement(name, pose))
        probe = lib.object(PROBE)
        placements.append(ct.Placement(PROBE, rest_pose(probe, lib, top_of(card, lib, pose), DROP_M)))
        tracked = PROBE
    else:
        placements.append(ct.Placement(name, rest_pose(card, lib, t_top, DROP_M)))
    return ct.SceneCard(robots=(), objects=tuple(placements), worlds=worlds), tracked


def _body_ids(built: BuiltScene, name: str) -> np.ndarray:
    h = built.handles[name]
    per = len(h.world_stride.body)
    tmpl = np.arange(h.template.body.start, h.template.body.stop)
    return np.concatenate([tmpl + w * per for w in range(built.world_count)]) if len(tmpl) else tmpl


def _finite(sim: Simulation) -> bool:
    ok = bool(np.isfinite(sim.state.body_q.numpy()).all()) if sim.model.body_count else True
    if sim.model.particle_count:
        ok = ok and bool(np.isfinite(sim.state.particle_q.numpy()).all())
    return ok


def run_scene(name: str, lib, router: Router, device: str, worlds: int = WORLDS) -> _Run:
    """Assemble and step SETTLE_S + WINDOW_S; the rest displacement is the largest body-origin travel of the tracked
    object over the window (nan when it has no bodies)."""
    card, tracked = scene_for(name, lib, worlds)
    built = SceneAssembler(lib, router).assemble(card, device, world_count=worlds)
    sim = Simulation(built)
    periods = round(SETTLE_S / built.control_dt), round(WINDOW_S / built.control_dt)
    for _ in range(periods[0]):
        sim.step_period()
    ids = _body_ids(built, tracked)
    q0 = sim.state.body_q.numpy()[ids, :3].astype(np.float64) if len(ids) else np.zeros((0, 3))
    travel = np.zeros(len(ids))
    for _ in range(periods[1]):
        sim.step_period()
        if len(ids):
            travel = np.maximum(travel, np.linalg.norm(sim.state.body_q.numpy()[ids, :3] - q0, axis=1))
    run = _Run(built, built.handles[name], tracked)
    run.finite = _finite(sim)
    run.displacement = float(travel.max()) if len(ids) else math.nan
    return run


# --- driver --------------------------------------------------------------------------------------------------------

def _route_name(handle: ObjectHandle) -> str:
    return str(handle.decision.route)


def measure_object(name: str, lib, device: str, checks: dict[str, Callable], worlds: int = WORLDS,
                   quick: bool = False) -> CoverageRow:
    """One object's row: route, run, check. Errors become a failed row with the error named; `quick` runs the bench
    laws behind the soft/liquid/granular checks at their quick settings (CPU development only)."""
    from alloy.scene.coverage_checks import check_for

    t0 = time.perf_counter()
    router = Router()
    try:
        with isolated_registry(empty=True):
            from alloy.scene.builtin_plugins import register_builtin_solvers

            register_builtin_solvers()
            run = run_scene(name, lib, router, device, worlds)
    except Exception as exc:  # noqa: BLE001 - an object that does not assemble or step is a failed row, never hidden
        if not isinstance(exc, AlloyError):
            log.exception("CARD-1 %s: crashed", name)
        return CoverageRow(name, f"error:{type(exc).__name__}", "none", False, False, detail=str(exc)[:300],
                           wall_s=time.perf_counter() - t0)
    run.extra["quick"] = quick
    route = _route_name(run.handle)
    if run.handle.decision.overridden:
        return CoverageRow(name, route, "none", False, False, detail="route override",
                           wall_s=time.perf_counter() - t0)
    kind, fn = check_for(route, checks)
    if not run.finite:
        return CoverageRow(name, route, kind, False, False, detail="non-finite state",
                           wall_s=time.perf_counter() - t0)
    try:
        value, bound, passed, detail = fn(name, lib, run, device)
    except Exception as exc:  # noqa: BLE001 - a check that cannot run fails the object, with the error named
        log.exception("CARD-1 %s: %s check crashed", name, kind)
        value, bound, passed, detail = None, None, False, f"{type(exc).__name__}: {str(exc)[:300]}"
    log.info("CARD-1 %s: %s %s value=%s bound=%s %s", name, route, kind, value, bound, detail)
    value, bound = (None if v is None else float(v) for v in (value, bound))   # plain floats/bools for JSON rows
    return CoverageRow(name, route, kind, True, bool(passed), value, bound, detail, time.perf_counter() - t0)


def measure(names: Iterable[str], lib, device: str, *, worlds: int = WORLDS, quick: bool = False,
            checks: dict[str, Callable] | None = None,
            on_row: Callable[[CoverageRow], None] | None = None) -> CoverageReport:
    from alloy.scene.coverage_checks import CHECKS

    rows = []
    for name in names:
        row = measure_object(name, lib, device, dict(CHECKS if checks is None else checks), worlds, quick)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return CoverageReport(tuple(rows), device, worlds)


__all__ = ["CoverageRow", "CoverageReport", "measure", "measure_object", "run_scene", "scene_for", "TARGET", "TABLE",
           "PROBE", "REST_ROUTES"]
