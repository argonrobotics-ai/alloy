# SPDX-License-Identifier: Apache-2.0
"""Coupler selection (D5, SPEC §3.4 `coupler_select`, §3.8 fluid pairs).

One dynamic entry: no coupler. Two: `proxy` (Scheduler / stock SolverCoupledProxy, lagged). Three or more: `admm`.
A pair is strong if m_a/m_s > `added_mass_ratio_max` (0.5), a force-saturating actuator clamps through it, or it must
hold static friction below CREEP_GENERAL; for a body in a liquid m_a = C_a·ρ_f·V_sub (C_a = 1, V_sub = the body's
volume, fully submerged worst case) and m_s = the body's mass. Strong pairs across entries run ADMM (k ≤ 20) and are
reported "strong, unrouted". A floating container (V_sub includes the air inside) routes like a floating solid (the
liquid's solver integrates its proxy inside every step at the source's effective mass, SCR-WP9-2) up to D5's ratio
``FLOATING_CONTAINER_RATIO_MAX`` (the largest WP9 measured holding: docs/notes/wp9-results-float.md); above it it is
a ``FloatingContainerError`` (Gissler interlinked pressure, §8.3, is not built). Light unflagged solids are not
bounded (docs/known_gaps/liquid-floating-container-ratio.md).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from alloy.core.errors import AssemblyError, RouteError
from alloy.coupling.types import PairPolicy
from alloy.route.types import Route
from alloy.scene.types import CouplerChoice

log = logging.getLogger("alloy.scene")

ADDED_MASS_COEFF = 1.0
FLOATING_CONTAINER_RATIO_MAX = 50.0       # D5 m_a/m_s routed up to: WP9 (a1) held 3-50 on the L4 (wp9-results-float.md)
FLUID_ROUTES = frozenset({Route.LIQUID_SPH, Route.LIQUID_MPM})
RIGID_ENTRIES = frozenset({"mujoco", "featherstone", "semi_implicit", "kamino", "xpbd"})
# Destinations whose strong pairs stay lagged (still reported strong-unrouted): SolverImplicitMPM takes rigid bodies as
# grid colliders (the proxy path) and never reads the particle forces that carry ADMM's contact rows (WP12, engine
# item 2; notes/wp12-sand).
LAGGED_STRONG_DESTINATIONS = frozenset({"mpm"})
# Partner routes whose surface SPH takes as boundary faces (SPH-VBD P1, WP9): a hub, such a partner and SPH form the
# P3 chain hub -> partner -> SPH (engine item 3; notes/wp4-p3-composition.md)
SPH_FACE_ROUTES = frozenset({Route.SOFT_VBD, Route.CLOTH_VBD, Route.CLOTH_FILM})
SPH_WALL_ROUTES = frozenset({Route.PASTE_MPM, Route.GRAIN_MPM})   # SPH particle walls (SCR-WP12-MIX)


class FloatingContainerError(RouteError):
    """A floating container too light for its liquid: D5's m_a/m_s above ``FLOATING_CONTAINER_RATIO_MAX``. ``fix``
    names the lightest mass that routes (the API shows it as the error's fix)."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.fix = fix


class StrongPairBesideLiquidError(AssemblyError):
    """A rigid / deformable pair that needs the strong (ADMM) coupler in a scene with a liquid, which ADMM cannot
    step yet. ``fix`` names what to change (the API shows it as the error's fix)."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.fix = fix


class UnsupportedSceneError(AssemblyError):
    """Solvers Alloy cannot couple in one scene yet (a liquid beside two other deformable solvers): the message
    names each solver's objects; ``fix`` the nearest scene that steps (the API shows it
    as the error's fix)."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True)
class EntryMass:
    """What D5 needs about one entry: its routes, total dynamic mass and the objects it carries."""

    name: str
    mass: float                                       # kg per world
    routes: frozenset[Route]
    objects: tuple[str, ...]
    fluid_density: float | None = None                # kg/m^3 if the entry carries a liquid


@dataclass(frozen=True)
class BodyInfo:
    name: str
    mass: float                                       # kg
    volume: float                                     # m^3 (solid volume)
    container: bool = False                           # Features.passage_width set (has an opening)
    dynamic: bool = True


@dataclass(frozen=True)
class CouplerInputs:
    entries: Sequence[EntryMass]
    bodies: Mapping[str, BodyInfo] = field(default_factory=dict)   # rigid objects by name
    saturating: frozenset[tuple[str, str]] = frozenset()
    static_hold: frozenset[tuple[str, str]] = frozenset()
    precision: frozenset[tuple[str, str]] = frozenset()


def strong_pair(m_a: float, m_s: float, policy: PairPolicy) -> bool:
    """D5 predicate (same rule WP4's watchdog promotes with): m_a/m_s > ratio_max, saturating actuator, static hold."""
    ratio = m_a / m_s if m_s > 0.0 else float("inf")
    return ratio > policy.added_mass_ratio_max or policy.saturating_actuator or policy.static_hold


def _round_up(x: float, digits: int = 3) -> float:
    """``x`` > 0 rounded up at ``digits`` significant figures (a printed bound a user can type and pass)."""
    step = 10.0 ** (math.floor(math.log10(x)) - digits + 1)
    return math.ceil(x / step * (1.0 - 1.0e-12)) * step


def fluid_added_mass(body: BodyInfo, rho_fluid: float) -> float:
    """m_a = C_a·ρ_f·V_sub for a fully submerged body (worst case), C_a = 1 (D5)."""
    return ADDED_MASS_COEFF * rho_fluid * body.volume


def _ordered(a: EntryMass, b: EntryMass) -> tuple[EntryMass, EntryMass]:
    """(source, destination): a liquid is always the destination (its solver sees the other entry's proxies as
    boundary, SPH-VBD P1, WP9), else the rigid entry drives the deformable one (proxy bodies live in the
    destination)."""
    if (a.fluid_density is None) != (b.fluid_density is None):
        return (a, b) if b.fluid_density is not None else (b, a)
    if a.name in RIGID_ENTRIES and b.name not in RIGID_ENTRIES:
        return a, b
    if b.name in RIGID_ENTRIES and a.name not in RIGID_ENTRIES:
        return b, a
    return (a, b) if a.name <= b.name else (b, a)


def _pair_policy(src: EntryMass, dst: EntryMass, inp: CouplerInputs) -> PairPolicy:
    key = (src.name, dst.name)
    if dst.fluid_density is not None:
        return _fluid_policy(src, dst, inp)
    m_s = src.mass
    return PairPolicy(source=src.name, destination=dst.name, mode="lagged", precision=key in inp.precision,
                      saturating_actuator=key in inp.saturating, static_hold=key in inp.static_hold,
                      added_mass=dst.mass if m_s > 0 else None)


def _fluid_policy(src: EntryMass, dst: EntryMass, inp: CouplerInputs) -> PairPolicy:
    """Body-liquid pair: the lightest dynamic body decides (largest m_a/m_s). A rigid source runs lagged, and a
    floating rigid container raises. A soft source (VBD surface as SPH boundary faces, SPH-VBD P1) always runs
    staggered with Aitken: lagged soft walls diverged or settled at 0.08 of the draft (WP9), so SolverDFSPH refuses
    them. Its added mass comes from the soft objects' volumes in ``inp.bodies``; without them it is C_a·m_s (neutral
    buoyancy, strong)."""
    soft = src.name not in RIGID_ENTRIES
    worst_ratio, worst_ma = 0.0, 0.0
    for name in src.objects:
        body = inp.bodies.get(name)
        if body is None or not body.dynamic:
            continue
        m_a = fluid_added_mass(body, dst.fluid_density)
        if not soft and body.container and m_a > FLOATING_CONTAINER_RATIO_MAX * body.mass * (1.0 + 1.0e-9):
            need = _round_up(m_a / FLOATING_CONTAINER_RATIO_MAX)
            raise FloatingContainerError(
                f"{name}: a floating container {m_a / body.mass:.3g}x lighter than the liquid filling its outline "
                f"(D5 m_a/m_s = ρ_f·V/m) is beyond the {FLOATING_CONTAINER_RATIO_MAX:g}x measured to hold "
                f"(Gissler interlinked pressure, §8.3, is not built); make it heavier (mass >= {need:.3g} kg), "
                "smaller, or static",
                fix=f"make {name!r} at least {need:.3g} kg (it is {body.mass:.3g} kg), smaller, or static")
        if body.mass > 0 and m_a / body.mass > worst_ratio:
            worst_ratio, worst_ma = m_a / body.mass, m_a
    if soft and not worst_ma:
        # volumes unknown (the assembler lists rigid objects only): assume neutral buoyancy, m_a = C_a·m_s, which is
        # strong (Aitken ω0 0.1, as every P1 law ran; ω0 0.5 on a light soft body was never run, WP9)
        worst_ma = ADDED_MASS_COEFF * src.mass
        log.info("%s -> %s: soft volumes unknown, added mass taken as C_a·m_s = %.3g kg (strong)", src.name,
                 dst.name, worst_ma)
    return PairPolicy(source=src.name, destination=dst.name, mode="staggered" if soft else "lagged",
                      added_mass=worst_ma or None, static_hold=(src.name, dst.name) in inp.static_hold)


def _ratio(policy: PairPolicy, src: EntryMass, inp: CouplerInputs) -> tuple[float, float]:
    """(m_a, m_s) for the strong test."""
    if policy.added_mass is None:
        return 0.0, 1.0
    if inp.entries and any(e.fluid_density for e in inp.entries if e.name == policy.destination):
        bodies = [inp.bodies[n] for n in src.objects if n in inp.bodies and inp.bodies[n].dynamic]
        lightest = min((b.mass for b in bodies), default=src.mass)
        return policy.added_mass, lightest
    return policy.added_mass, src.mass


def _hub_partner_sph(dyn: Sequence[EntryMass], inp: CouplerInputs) -> CouplerChoice:
    """SCR-DC5-7: a rigid hub with one two-way partner (lagged, Scheduler) plus SPH as a one-way follower of the hub
    (hub shapes are its boundary only; the liquid loads nothing, so it does no work on the hub: alloy.coupling.one_way
    ledgers hub_work = 0). A partner with SPH boundary faces (``SPH_FACE_ROUTES``: a soft bag, a cloth) also couples
    two-way to the liquid, partner -> SPH staggered (``_fluid_policy``): the P3 chain, one Scheduler over the three
    entries (``alloy.coupling.compose``). So does an MPM paste or powder partner (``SPH_WALL_ROUTES``: its
    particles are the liquid's particle walls, SCR-WP12-MIX). Refused: not exactly one hub and one partner, or a
    strong hub pair (ADMM has no one-way stage: ``StrongPairBesideLiquidError`` with the mass that routes)."""
    sph = next(e for e in dyn if e.name == "sph")
    hubs = [e for e in dyn if e.name in RIGID_ENTRIES]
    partners = [e for e in dyn if e is not sph and e.name not in RIGID_ENTRIES]
    _check_beside_liquid(dyn)
    src, dst = _ordered(hubs[0], partners[0])
    pol = _pair_policy(src, dst, inp)
    m_a, m_s = _ratio(pol, src, inp)
    if strong_pair(m_a, m_s, pol):
        raise _strong_beside_liquid(src, dst, pol, m_a, m_s)
    log.info("SPH follows %s one-way (no load and no work on the hub); %s/%s lagged", hubs[0].name, src.name,
             dst.name)
    follower = PairPolicy(source=hubs[0].name, destination=sph.name, mode="lagged", one_way=True)
    partner = partners[0]
    if not partner.routes & (SPH_FACE_ROUTES | SPH_WALL_ROUTES):
        return CouplerChoice(kind="proxy", pairs=(pol, follower))
    wet = _fluid_policy(partner, sph, inp)
    m_a, m_s = _ratio(wet, partner, inp)
    unrouted = ((wet.source, wet.destination),) if strong_pair(m_a, m_s, wet) else ()
    log.info("P3 chain: %s -> %s lagged, %s -> %s %s, %s -> %s one-way", src.name, dst.name, wet.source,
             wet.destination, wet.mode, hubs[0].name, sph.name)
    return CouplerChoice(kind="proxy", pairs=(pol, wet, follower), strong_unrouted=unrouted)


def _side(e: EntryMass) -> str:
    """``towel, sheet (cloth)``: an entry's objects and what they are, in a user's words."""
    from alloy.coupling.unsupported import ROUTE_WORDS  # noqa: PLC0415

    what = "rigid bodies" if e.name in RIGID_ENTRIES else " and ".join(
        sorted({ROUTE_WORDS.get(r, str(r)) for r in e.routes}))
    names = list(e.objects) or [e.name]
    return ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "") + f" ({what})"


def _beside_liquid(sph: EntryMass, hubs: Sequence[EntryMass], partners: Sequence[EntryMass]) -> UnsupportedSceneError:
    """The refusal of a liquid beside two other solvers, in the scene's terms: a liquid steps beside one solver of rigid
    bodies and one more deformable solver (cloth, soft bodies, cables and film sheets; or sand, paste and powder)."""
    msg = (f"the liquid {_side(sph)} beside {'; '.join(_side(e) for e in (*hubs, *partners))} is not supported in "
           "one scene yet: a scene with a liquid steps its rigid bodies and robots and one more kind of deformable "
           "object (cloth, soft bodies, cables or film sheets; or sand, paste or powder), not two")
    if len(partners) > 1:
        fix = (f"keep only one of {' or '.join(_side(e) for e in partners)} beside the liquid: take the others out, "
               "make the sand, paste or soft bodies among them rigid (rigid=True), or build them in a separate scene")
    else:
        fix = "keep the scene's rigid bodies and robots on one rigid solver (MuJoCo, the default) beside the liquid"
    return UnsupportedSceneError(msg, fix)


def _check_beside_liquid(dyn: Sequence[EntryMass]) -> None:
    """The supported shape beside a liquid: a liquid (entry "sph") beside more than one other solver steps only as
    one rigid hub with one deformable partner; else ``UnsupportedSceneError`` naming every solver's objects."""
    sph = next((e for e in dyn if e.name == "sph"), None)
    if sph is None or len(dyn) <= 2:
        return
    hubs = [e for e in dyn if e.name in RIGID_ENTRIES]
    partners = [e for e in dyn if e is not sph and e.name not in RIGID_ENTRIES]
    if len(hubs) != 1 or len(partners) != 1:
        raise _beside_liquid(sph, hubs, partners)


def _strong_beside_liquid(src: EntryMass, dst: EntryMass, pol: PairPolicy, m_a: float,
                          m_s: float) -> StrongPairBesideLiquidError:
    """The refusal of a strong hub pair beside a liquid (SCR-DC5-7), in the scene's own terms: which objects, why
    the pair is strong, and what makes it route (docs/known_gaps/liquid-cloth-light-rigid.md). The message names no
    engine internals (audit COUPLE-1 re-test): the pair needs the strong (ADMM) coupler, D5 m_a/m_s > the ratio max,
    and ADMM has no one-way stage for SPH."""
    rigid, soft = list(src.objects) or [src.name], list(dst.objects) or [dst.name]
    gap = "docs/known_gaps/liquid-cloth-light-rigid.md"
    if pol.saturating_actuator or pol.static_hold:
        why = "an actuator pressed to its force limit" if pol.saturating_actuator else "a grip that must not creep"
        return StrongPairBesideLiquidError(
            f"{soft} is held by {why} on {rigid}, beside a liquid: such a hold needs a tighter coupling of the two "
            "solvers than a scene with a liquid can run yet",
            fix=f"build the liquid and {soft} in separate scenes ({gap})")
    need = _round_up(m_a / pol.added_mass_ratio_max)
    most = m_s * pol.added_mass_ratio_max
    return StrongPairBesideLiquidError(
        f"the free rigid objects {rigid} weigh {m_s:.3g} kg in all, less than {1.0 / pol.added_mass_ratio_max:g}x "
        f"{soft} ({m_a:.3g} kg), beside a liquid: rigid bodies that light against cloth or a soft body need a tighter "
        "coupling of the two solvers than a scene with a liquid can run yet",
        fix=f"make the free rigid objects weigh at least {need:.3g} kg in all (mass=; they weigh {m_s:.3g} kg), or "
            f"make them static=True or kinematic=True, or make {soft} lighter than {most:.3g} kg, or build the "
            f"liquid in its own scene ({gap})")


def _strong_mode(dst: EntryMass) -> str:
    """Pair mode for a strong pair (D5) by destination. SCR-WP9-2: body-fluid pairs never use ADMM (Newton's ADMM
    carries forces only through its own contact rows); they run staggered + Aitken ω0 0.1. MPM destinations stay
    lagged (``LAGGED_STRONG_DESTINATIONS``). Everything else goes to ADMM."""
    if dst.fluid_density is not None:
        return "staggered"
    if dst.name in LAGGED_STRONG_DESTINATIONS:
        return "lagged"
    return "admm"


def select_coupler(inp: CouplerInputs) -> CouplerChoice:
    """D5 coupler choice and per-pair policies (strong pairs → ``_strong_mode``, reported strong-unrouted)."""
    from alloy.coupling.unsupported import check_entries  # noqa: PLC0415

    dyn = [e for e in inp.entries if e.name]
    check_entries(dyn)              # route pairs no solver steps together: refused here, so estimate refuses them too
    if any(e.fluid_density is not None and e.name == "sph" for e in dyn) and len(dyn) > 2:
        return _hub_partner_sph(dyn, inp)
    if len(dyn) <= 1:
        return CouplerChoice(kind="none", pairs=())
    pairs, unrouted = [], []
    by_name = {e.name: e for e in dyn}
    for i in range(len(dyn)):
        for j in range(i + 1, len(dyn)):
            src, dst = _ordered(dyn[i], dyn[j])
            pol = _pair_policy(src, dst, inp)
            m_a, m_s = _ratio(pol, by_name[pol.source], inp)
            if strong_pair(m_a, m_s, pol):
                pol = replace(pol, mode=_strong_mode(by_name[pol.destination]))
                unrouted.append((pol.source, pol.destination))
                log.info("strong pair %s/%s (m_a/m_s = %.3g): %s, strong unrouted", pol.source, pol.destination,
                         m_a / m_s if m_s else float("inf"), pol.mode)
            pairs.append(pol)
    strong_admm = any(p.mode == "admm" for p in pairs)
    kind = "admm" if len(dyn) >= 3 or strong_admm else "proxy"
    return CouplerChoice(kind=kind, pairs=tuple(pairs), strong_unrouted=tuple(unrouted))
