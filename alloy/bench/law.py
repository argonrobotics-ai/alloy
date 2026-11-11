# SPDX-License-Identifier: Apache-2.0
"""Law registry and discovery (SPEC §3.10, §5.3, D23).

Producers register their laws from their own `laws.py` with `register_law`. `discover()` imports every producer
module named in `CATALOGUE`; a module that does not exist yet (its package has not merged) leaves its laws
`not_run` with the reason, so the scoreboard and the baseline see the gap instead of silently skipping it.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from alloy.bench.compare import placeholder_verdict
from alloy.bench.provenance import provenance
from alloy.bench.types import LawResult, LawSpec
from alloy.core.errors import LawError

log = logging.getLogger("alloy.bench")


@dataclass(frozen=True)
class CatalogueEntry:
    """A law family the spec promises: id prefix (`A7`), suite, producing package and its registering module."""

    prefix: str
    suite: str
    producer: str
    module: str
    title: str


def _entries(prefixes: str, suite: str, producer: str, module: str, title: str) -> list[CatalogueEntry]:
    return [CatalogueEntry(p, suite, producer, module, title) for p in prefixes.split()]


CATALOGUE: tuple[CatalogueEntry, ...] = tuple(
    _entries("A1 A2 A3 A4 A5 A6 A15 A16", "analytic", "WP6", "alloy.bench.analytic", "WP6 analytic laws")
    + _entries("A7 A8 A9", "analytic", "WP3", "alloy.contact.laws", "contact laws")
    + _entries("A10 A11 A12", "analytic", "WP2", "alloy.scene.laws", "scene/rod laws")
    + _entries("A13 A14", "analytic", "WP9", "alloy.solvers.sph.laws", "liquid laws")
    + _entries("A17", "analytic", "WP11", "alloy.fit.momentum_law", "VBD linear momentum")  # SCR-WP11-2
    + _entries("D1 D2 D3", "determinism", "WP6", "alloy.bench.determinism", "determinism suite")
    + _entries("R1 R2", "rig", "WP3", "alloy.contact.laws", "contact rig twins")
    + _entries("R3 R4", "rig", "WP6", "alloy.bench.twins", "incline and bounce twins")
    + _entries("R5 R6 R8 R9", "rig", "WP2", "alloy.scene.laws", "squeeze/drape/granular/rope twins")
    + _entries("R7", "rig", "WP9", "alloy.solvers.sph.laws", "pour twin")
    + _entries("R10a R10b R10c", "rig", "WP14", "alloy.realrig", "YAM arm twins (SCR-WP14-1)")
)

TwinFactory = Callable[[], Any]
_REGISTRY: dict[str, tuple[LawSpec, TwinFactory]] = {}
_IMPORT_ERRORS: dict[str, str] = {}


# Laws reported but not gated (SPEC §5.3 "known gap"): never in SB-12, shown with this reason on their rows.
_SPIN_UP = ("an off-axis multi-contact slider spins up about its vertical axis and feels less down-slope friction "
            "(WP3 FRIC-cause; MuJoCo C too); azimuth 0 is gated in A5.slide")
KNOWN_GAPS: dict[str, str] = {
    "A5.slide_spread": f"SPEC §5.3 A5 (SCR-WP3-7): MuJoCo sliding spread over 16 azimuths at the defaults: {_SPIN_UP}",
    "A5.slide_recipe": f"SPEC §5.3 A5 (SCR-WP3-7): the same spread under the static-hold recipe (flat imp 0.99, "
                       f"impratio 300): {_SPIN_UP}",
}


@dataclass(frozen=True)
class Trade:
    """A known gap a ruling let grow to fix something else: shown beside SB-12 with both numbers, never absorbed."""

    metric: str      # the metric the trade moved
    before: float    # its value before the change
    after: float     # its value after
    why: str         # the change, what it fixed, and the ruling


TRADES: dict[str, Trade] = {
    "A5.slide_spread": Trade(
        "accel_spread", 0.034, 0.103,
        "SCR-WP3-10 300 fill-in (on main since 27da8424): A5.speed fixed, slide spread 0.034 -> 0.103 (coordinator "
        "ruling 2026-11-03: a stated trade, speed-independent friction matters for slow manipulation; WP3 measures "
        "an isotropic option after SCR-DC3-8)"),
}



@dataclass(frozen=True)
class RouteGap:
    """A known gap of the default route that no law result carries (a ruling's measured row): shown beside SB-12
    with its number, never counted in it."""

    name: str                   # row id `SB-12:known gap <name>`
    value: float                # the measured gap, in `unit`
    unit: str
    target: str                 # what closing it means (not gated)
    bound: float                # the target as a number, in `unit`
    text: str                   # the ruling's statement of the gap
    detail: str                 # the ruling and the producer's rows behind the number
    sources: tuple[str, ...]    # the producer's raw rows


ROUTE_GAPS: tuple[RouteGap, ...] = (
    RouteGap(
        "static hold (soft friction creep)", 0.109, "mm/s", "<= 0.001 mm/s at <= 0.5 mu*N, real: 0 (not gated)", 0.001,
        "soft friction creeps below the cone on the default route (0.11 mm/s at 0.25 mu*N push, real: 0); held with "
        "opt-in noslip scope all (alloy:noslip_all, T12 SB-6 cost 1.12x -> 0.21x); rk4 route entries keep the creep",
        "coordinator ruling 2026-11-04 01:36Z (c), [internal note]: no default switch, SB-6 N=1 would fail "
        ">= 1.03x. WP3 rows ([internal note] 2026-11-04 02:00Z and 03:40Z): DC3 QSFP check scene, latch push "
        "~0.25 mu*N, impratio 300 (the route at SCR-WP3-10): 109 um/s; scope all 0.13 um/s, period +21 %. A5 box rig "
        "at 0.25 / 0.5 mu*N: impratio 300 1.06 / 2.07 um/s, main's impratio 1 317 / 619 um/s, scope all 0 / 0. GPU "
        "(L4, scope all on every non-rk4 MuJoCo entry vs the default): T12 SB-6 N=1 1.118x -> 0.207x RT, T12 SB-5 "
        "4,096 176x -> 49.7x, T09 SB-5 3.32x -> 2.46x at 4,096; SB-7 smoke 0 trips; SB-12 suite 0 verdict flips",
        ("[internal note]", "[internal note]",
         "[internal note]", "[internal note]")),
)


# A gated law's standing note, printed in SB-12's reason while the law is in its population (coordinator rulings;
# A17: SCR-WP11-2, 2026-11-04 05:53Z; A8: BUL-0014, 2026-11-04 18:14Z (a) and 2026-11-05 21:04Z).
SB12_DETAIL: dict[str, str] = {
    "A17": ("A17: WP11's coarse step is the vbd default since main 8423239b (stock VBD failed flat 0.40 / curved "
            "2.29); scope: free components, contacting components keep stock VBD"),
    "A8": ("A8.squeeze_wedge: peak block-sheet penetration <= 1 mm, BUL-0014's functional bar, gated and red as a "
           "known gap (coordinator 2026-11-05 21:04Z, [internal note]): 1.84 mm on main with "
           "T12's measured 0.3 kg tip (WP13, a0eb9cf6; 2.02 mm with the 1 kg stand-in, about 2.8x the note's "
           "expected depth δ_exp = sqrt(2E/K_pair); pair stiffness D22's default 1e6 N/m, marked default; the "
           "law's detail adds δ_exp and measured / δ_exp, WP3). A derived bound min(1 mm, δ_exp (1 + tol)), tol "
           "from a converged reference, never the route, replaces 1 mm only once the cards carry a sourced "
           "contact_stiffness (the tip-mass condition is met since a0eb9cf6). The path to green is the engine, "
           "not the bound: the route's soft contact (τ = 2.083 ms) gives a depth of about v·τ/e"),
    "A13": ("A13 seal laws (WP9, SCR-SC10-1): A13.sealed_column fails full on GPU on csf's meniscus shedding "
            "(piston 0.855 of rho g H, drift 1.14 spacing, void 1.9 %; SCR-WP9-7); A13.piston_displacement's quick "
            "fail was its 2-layer push, full passes at +0.18 %"),
}


# Laws reported for the record but not gated: the measurement, not the solver, sets the number.
INFORMATIONAL: dict[str, str] = {
    "A4.collide_small_lever": "the A4 collide world before the 2026-11-01 test-conditioning fix (coordinator): L0 "
                              "from a 2 cm lever and 10 s of free flight over ~5 m, so |dL|/|L| reads float32 "
                              "position rounding (MuJoCo full: 1.18e-3 at impratio 1; 2.90e-3 on CUDA at the "
                              "SCR-WP3-10 fill-in, impratio 300), not the solver; A4.momentum gates the "
                              "re-conditioned world (0.2 m lever, window ends at 1 m separation)",
}


@dataclass(frozen=True)
class Conditional:
    """A gated law whose pass holds under a condition the default route does not meet yet (coordinator ruling)."""

    caveat: str      # what the measured pass depends on
    ruling: str      # what must hold before the pass counts
    met: bool = False  # set when the producer has shown the ruling holds (until then a pass counts as fail)


# Printed on the law row, beside SB-12 and under the board's rulings.
CONDITIONAL: dict[str, Conditional] = {
    "A10.capstan": Conditional(
        caveat="the route's CHAIN emitter floors every rotational dof of every CHAIN rope at armature 1e-6 kg·m² "
               "(emitters.chain.ARMATURE_FLOOR, WP2 main aeea819; the twin sets none); without a floor MuJoCo Warp's "
               "float32 Newton goes non-finite on this rope (16 of 16 worlds at 1e-9). Inertia criterion fails (1e-6 "
               "is 42x an A10 link's bend inertia); dynamics check passes: hanging-chain swing omega1, floor off -> "
               "on, A10 rope -0.105 %, A11 0.5 m +0.004 %, 0.8 m -0.277 % (< 1 %; rerun by WP6 at aeea819)",
        ruling="coordinator 2026-11-02: A10 counts as pass only when (1) WP2 makes the chain armature a default-route "
               "rule for every CHAIN rope (set by the route, not inside the test) and (2) shows it does not change "
               "rope physics: armature <= 10 % of the link rotational inertia, or a dynamics check (swing period or "
               "wave speed) moving < 1 %. Until both hold A10 counts as fail and SB-12 stays fail",
        met=True),  # 2026-11-02: route rule in aeea819 (code 1ca2e29), dynamics check < 1 % (verified by WP6)
}
CAVEATS: dict[str, str] = {law_id: c.caveat for law_id, c in CONDITIONAL.items()}


def caveat(law_id: str) -> str:
    """The condition a gated law's result holds under, and the ruling while it is unmet ("" for none)."""
    c = CONDITIONAL.get(law_id)
    if c is None:
        return ""
    return f"caveat: {c.caveat}" + ("" if c.met else f"; counted as fail ({c.ruling})")


def counted_verdict(law_id: str, verdict: str) -> str:
    """The verdict SB-12 counts: a pass under an unmet ruling counts as fail."""
    c = CONDITIONAL.get(law_id)
    return "fail" if verdict == "pass" and c is not None and not c.met else verdict


def not_gated(law_id: str) -> str:
    """Why a law is shown but never counted in SB-12 ("" for a gated law)."""
    if law_id in KNOWN_GAPS:
        return f"known gap, not gated: {KNOWN_GAPS[law_id]}"
    if law_id in INFORMATIONAL:
        return f"informational, not gated: {INFORMATIONAL[law_id]}"
    return ""


def law_prefix(law_id: str) -> str:
    """`A5.onset` -> `A5`; `LIQ-4.hydrostatic` -> `LIQ-4`."""
    return law_id.split(".", 1)[0]


def register_law(spec: LawSpec, twin_factory: TwinFactory) -> None:
    """Register a law; re-registering the same id with an equal spec is a no-op, a different spec is an error."""
    if not spec.id or not spec.metrics:
        raise LawError(f"law {spec.id!r} needs an id and at least one metric")
    missing_src = [m for m, tol in spec.metrics.items() if not tol.source]
    if missing_src:
        raise LawError(f"law {spec.id}: tolerances without a source: {missing_src}")
    if set(spec.stat) - set(spec.metrics):
        raise LawError(f"law {spec.id}: stat keys {sorted(set(spec.stat) - set(spec.metrics))} are not metrics")
    old = _REGISTRY.get(spec.id)
    if old is not None and old[0] != spec:
        raise LawError(f"law {spec.id} registered twice with different specs")
    _REGISTRY[spec.id] = (spec, twin_factory)


def unregister_law(law_id: str) -> None:
    """Remove a law (tests only)."""
    _REGISTRY.pop(law_id, None)


def registered_laws(suite: str | None = None) -> list[tuple[LawSpec, TwinFactory]]:
    """Registered laws, sorted by id, optionally filtered by suite."""
    items = sorted(_REGISTRY.values(), key=lambda it: _sort_key(it[0].id))
    return [it for it in items if suite is None or it[0].suite == suite]


def _sort_key(law_id: str) -> tuple[str, float, str]:
    head = law_prefix(law_id)
    digits = "".join(ch for ch in head if ch.isdigit())
    return head.rstrip("0123456789"), float(digits) if digits else math.inf, law_id


def discover(modules: tuple[str, ...] | None = None) -> dict[str, str | None]:
    """Import every producer module; returns {module: None if imported else the reason it was not}."""
    wanted = modules or tuple(dict.fromkeys(e.module for e in CATALOGUE))
    status: dict[str, str | None] = {}
    for mod in wanted:
        try:
            importlib.import_module(mod)
            status[mod] = None
            _IMPORT_ERRORS.pop(mod, None)
        except ModuleNotFoundError as exc:
            status[mod] = f"producer module {mod} not merged ({exc.name} missing)"
        except Exception as exc:  # noqa: BLE001 - a broken producer must show up as not_run with the reason
            status[mod] = f"producer module {mod} failed to import: {type(exc).__name__}: {exc}"
            log.error("law discovery: %s", status[mod])
        if status[mod]:
            _IMPORT_ERRORS[mod] = status[mod]  # type: ignore[assignment]
    return status


def missing_laws(suite: str | None = None) -> list[tuple[CatalogueEntry, str]]:
    """Catalogue entries with no registered law, each with the reason (unmerged producer or law not registered)."""
    have = {law_prefix(spec.id) for spec, _ in registered_laws()}
    out = []
    for e in CATALOGUE:
        if (suite is None or e.suite == suite) and e.prefix not in have:
            reason = _IMPORT_ERRORS.get(e.module) or f"{e.producer} ({e.module}) registered no {e.prefix} law"
            out.append((e, reason))
    return out


def not_run_result(law_id: str, solver: str, device: str, reason: str) -> LawResult:
    """A LawResult that records a law which could not run; it never counts as a pass."""
    verdict = placeholder_verdict("law", "1", "not_run")
    return LawResult(law_id, solver, device, (verdict,), {"not_run_reason": reason},
                     provenance(device, {"law": law_id, "solver": solver}), 0.0, False)


__all__ = ["CATALOGUE", "CAVEATS", "CONDITIONAL", "INFORMATIONAL", "KNOWN_GAPS", "ROUTE_GAPS", "SB12_DETAIL", "TRADES",
           "CatalogueEntry", "Conditional", "RouteGap", "Trade", "caveat", "counted_verdict", "discover", "law_prefix",
           "missing_laws", "not_gated", "not_run_result", "register_law", "registered_laws", "unregister_law"]
