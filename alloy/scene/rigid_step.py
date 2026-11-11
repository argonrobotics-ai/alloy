# SPDX-License-Identifier: Apache-2.0
"""The rigid step of a scene's MuJoCo entry (SCR-WP0-RIGIDDT; coordinator ruling 2026-11-08 11:25Z, the plan and its
WP2 / WP3 / WP13 reviews in [internal note]).

The public API asks for ``mode="auto"`` under the standard and fast profiles: a rigid-only scene of free bodies then
steps at the largest grid step ``control_dt / 2**j`` that every cap below allows, and never below the fixed step
(``today_step``: snap_to_grid(1/600, control_dt), the step every scene took before this rule). Every other scene, the
precision and debug profiles, and the engine layers by default (``mode="fixed"``) keep the fixed step, bit for bit. A
number pins the step (snapped to the grid).

Eligible (phase 1): every dynamic object is a single free rigid body on the MuJoCo entry, routed RIGID,
RIGID_COMPLIANT or RIGID_PILE (pile_backend "mujoco"); no robots, attachments, kinematic objects or precision
features. Robots (``alloy.scene.rigid_step_robots``) are opt-in: a scene with robots on the MuJoCo entry is eligible
only when the user asked, with a number or an explicit "auto" (``RigidStepRequest.explicit``); its robots then add
the caps "servo", "damping", "solref", "link_gap" and "loose_body" to the scan; its free bodies keep the caps below.

Caps, per dynamic body b at speed v_b (``BodyFacts.speed``: its card's contact speed or its start drop's impact):
1. tunnel: 2 v_b dt <= its thinnest member (the card validator's thin-member rule at the chosen step);
2. gap: v_b dt <= RIGID_GAP_M (5 mm): the travel-per-step limit; MuJoCo applies no contact force until the surfaces
   overlap, so a body meets a part about v dt deep (docs/known_gaps/rigid-tunnelling.md);
3. spin: (v_b / lever) dt <= SPIN_SHARE x the SPIN limit (0.5 rad per step), lever the middle half extent;
4. floor: a slender body steps at most R0_DT x max(1, r_gyr / R0_MIN), so the inertia floor raises no body more than
   it does at the fixed step (with the floor off too), and at most FLOOR_SWEPT_DT until WP3's boundary law passes;
5. sag: contacts never get softer than at the fixed step: the step's native gains are at least those at the fixed
   step under the integrator the scene had before this rule, so every rest-sag law holds as it did. Where the
   stability cap sets a body's stiffness (light bodies), that keeps the fixed step under the same integrator; where
   its material does (heavy ones), the step can grow until the cap reaches it. (A 0.1 kg cube on a table rests 4 um
   in at 1.25 ms, 16 um at 2.5 ms under rk4 and sinks 55 um on the way, past test_box_diagonal's 50 um: an absolute
   0.5 mm bound let that through.) The report gives the sag under the column load (``steady_state_violation``: d0
   at rest, WP3's form);
6. creep: the friction saturation speed, ~ kd / (ke impratio) (``alloy.contact.law_speed``), at the step and
   FILL_IMPRATIO stays at or below its value at A5.speed's pass point: the fixed step, implicitfast and
   SLIDING_MIN_IMPRATIO;
7. control_dt, the grid top;
8. fixed: dt <= FIXED_TAU / 2: the friction rows' solref (0.02, 1) (``mujoco_pairs``) and Newton's kd = 0 fallback
   meet MuJoCo's refsafe clamp (WP2);
9. compliant: dt <= tau / 2 for the card's own contact time constant (WP2): a pad is never softer than its card.

Fast mode (SPEED-NOW S1; coordinator ruling 2026-11-09 09:18Z, its design change accepted): ``mode="fast"``, asked by
``scene.rigid_step = "fast"`` or by ``profile="fast"`` on a scene that never set it. Eligibility and caps are "auto"'s
with one change: cap 5 also passes when the predicted rest sag under the column load is at most SAG_SHARE of the
body's thinnest member, so contacts may be softer than at the fixed step (``FAST_COST``, on the route report). Cap 6
stays, so slow sliding friction keeps A5.speed's model as under "auto" (impratio 300). Free-body entries keep
implicitfast in fast mode, where "auto" runs rk4 (ruling 2026-11-09: R1 and R2 are off by default because a pushed
tall body rocks more and more; ``integrator_route._fast_implicit``), and step 2c takes the own gains at R1's step
too. XB4-9's 1 m stacks step at 10 ms; 5-10 cm cubes at 5 ms, 2 cm at 2.5 ms. The entry carries FAST_EFFECT
(``entries.RIGID_FAST_EFFECT``): on MuJoCo Warp its box-box face pairs keep 4 contact points instead of up to 8
(``alloy.contact.mjw_boxbox_reduce``, WP3), +37-40 % on XB4-9's stacks.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.contact.inertia_floor import R0_MIN, floor_radius
from alloy.core.defaults import GRAVITY, RIGID_GAP_M
from alloy.core.timebase import TICK_EXP, snap_to_grid
from alloy.precision.health_setup import steady_state_violation
from alloy.precision.types import CheckPolicy
from alloy.route.types import Route
from alloy.scene import pairs as pt
from alloy.scene import rigid_step_bodies as RB
from alloy.scene import rigid_step_robots as RR
from alloy.scene import rigid_step_warn as RW
from alloy.scene.entries import FILL_IMPRATIO, RIGID_FAST_EFFECT, RIGID_TARGET, SLIDING_MIN_IMPRATIO
from alloy.scene.integrator_route import concave_names
from alloy.scene.native_gains import DEFAULT_IMP, native_gains, resting_floor
from alloy.scene.template import is_compliant

log = logging.getLogger("alloy.scene")

PHASE1_ROUTES = (Route.RIGID, Route.RIGID_COMPLIANT, Route.RIGID_PILE)
SPIN_SHARE = 0.5                # cap 3: half the SPIN limit
SOLIMP_D0 = 0.9                 # MuJoCo's default solimp d0, the impedance at rest under the solimp width (WP3)
FLOOR_SWEPT_DT = 1.5625e-3      # s: the inertia floor's sweep reaches this step (alloy.contact.inertia_floor)
FIXED_MODES = ("fixed", "today")     # "today": the name before 0.1.0, kept for old Scene JSON
FIXED_TAU = 0.02                # s: friction rows' solref and Newton's kd = 0 fallback (cap 8)
CREEP_REF_INTEGRATOR = "implicitfast"   # cap 6's reference: A5.speed passes at the fixed step, impratio 50
FAST_MODE = "fast"              # SPEED-NOW S1: the opt-in rigid fast mode (module docstring, "Fast mode")
FAST_EFFECT = RIGID_FAST_EFFECT         # on the fast mode's MuJoCo entry (WP3's 4-point box-box faces: entries)
SAG_SHARE = 0.01                # fast mode's cap 5: predicted rest sag <= 1 % of the body's thinnest member
FAST_COST = (f"rigid fast mode (profile 'fast' or scene.rigid_step = 'fast'): contacts can be softer than at the fixed "
             f"step, a resting body sinking up to {SAG_SHARE:.0%} of its thinnest member; on MuJoCo Warp (GPU, CPU "
             "warp route) free bodies step on implicitfast with the friction pull capped, and a 30 cm cube sliding at "
             "5 m/s rocks under 1 deg and lifts 1 mm; on the CPU's MuJoCo C route (no cap) they stay on rk4, and that "
             "cube rocks 6 deg and hops 8 cm (2 deg and 1 cm at 2 m/s); under profile 'fast' (checks every 10 steps) a "
             "world latched INVALID freezes at its last saved checked state, up to 9 steps before the failure at one "
             "step per period (at 2, 4 or 8 steps per period, at the start of its failing period); "
             "scene.rigid_step = 'auto' keeps contacts as stiff as at the fixed step")


@dataclass(frozen=True)
class RigidStepRequest:
    """What the builder of a scene asks for: ``mode`` "fixed" (the engine layers' default), "auto" (the API under
    the standard and fast profiles), "fast" or a step [s] to pin; ``rows``: per-world start positions {name: (W, 3)};
    ``from_profile``: the API's profile asked for ``mode`` (a scene that is not eligible keeps the fixed step without
    a warning); ``announce``: called once with the step when the fast mode steps the scene (the API's warning);
    ``newton_contacts``: the build collides the scene on Newton's pipeline (``alloy.build``'s ``contacts``), which
    keeps the fixed step and takes rk4 off (PHYSA-1; ``integrator_route.choose_integrator``);
    ``explicit``: the user set ``mode`` (an "auto" that is not the never-set default: robots may then step coarser);
    ``warn``: called with each gap warning (``rigid_step_warn``; else it is logged)."""

    mode: str | float = "fixed"
    rows: Mapping[str, Any] | None = None
    from_profile: bool = False
    announce: Callable[[Any], None] | None = field(default=None, compare=False)
    newton_contacts: bool = False
    explicit: bool = False
    warn: Callable[[str], None] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class RigidStep:
    """The MuJoCo entry's step [s], the fixed step (``today``), and why: the cap and body that bound it, or why the
    scene keeps the fixed step. ``caps``: (cap, the largest grid step it allows) for an eligible scene."""

    dt: float
    today: float
    reason: str
    caps: tuple[tuple[str, float], ...] = ()
    fast: bool = False              # the fast mode steps this scene (its MuJoCo entry gets FAST_EFFECT)
    eligible: bool = False          # "auto" / "fast" could step it coarser (``why_not`` is None; "fixed" checks too)

    @property
    def raised(self) -> bool:
        return self.dt > self.today

    def effects(self) -> tuple[str, ...]:
        return (FAST_EFFECT,) if self.fast else ()


@dataclass(frozen=True)
class Gains:
    """A body's native-gain inputs (``native_gains.apply_native_gains``)."""

    K: float
    e: float
    zeta_min: float
    compliant: bool


def today_step(control_dt: float) -> float:
    """The fixed step, the one every scene took before this rule: snap_to_grid(1/600, control_dt)."""
    return snap_to_grid(RIGID_TARGET, control_dt)


def candidates(control_dt: float, today: float) -> list[float]:
    """Grid steps from control_dt down to the fixed step ``today``, largest first."""
    return [control_dt / 2.0**j for j in range(TICK_EXP + 1) if control_dt / 2.0**j >= today * (1 - 1e-9)]


def why_not(b: newton.ModelBuilder, objs: Sequence[Any], scene: Any,
            request: RigidStepRequest | None = None) -> str | None:
    """None when the scene is eligible (module docstring), else why it keeps the fixed step. Robots are eligible
    only under a ``request`` the user made (``rigid_step_robots.refusal``)."""
    if request is not None and request.newton_contacts:
        # PHYSA-1: rows on Newton's pipeline take the pair table's gains, not the native gains the coarser steps are
        # sized for (a 5 kg sdf cube: 1.03 mm of vibration at the 5 ms step, 0.12 mm at the fixed one;
        # docs/known_gaps/rigid-sdf-mesh-contact-limits.md). A card's sdf / mesh collision alone does not put a
        # scene there: add_object keeps MuJoCo's own contacts. Robots too: their rows would not be the file's.
        return f"Newton's contact pipeline{concave_names(objs)} keeps the fixed step"
    if getattr(scene, "robots", None) and (why := RR.refusal(request, scene)) is not None:
        return why
    if getattr(scene, "attachments", None):
        return "attachments keep the fixed step"
    for t in objs:
        if t.obj is None:
            if (why := RR.robot_why_not(t, request)) is not None:
                return f"{t.name}: {why}"
            continue
        card, d = t.obj.card, t.decision
        if card.motion == "kinematic":                  # routed STATIC, but moved at run time at speeds set then
            return f"{t.name}: kinematic objects keep the fixed step"
        if card.motion == "static" or d.route == Route.STATIC:
            continue
        if d.entry != "mujoco" or d.route not in PHASE1_ROUTES:
            return f"{t.name}: route {d.route.value} on {d.entry} keeps the fixed step"
        if card.precision_partners or card.precision_features:
            return f"{t.name}: precision pairs keep the fixed step"
        if len(t.ranges.body) != 1 or not _free(b, t.ranges.body[0]):
            return f"{t.name}: not a single free body"
    return None


def _free(b: newton.ModelBuilder, body: int) -> bool:
    return any(int(b.joint_child[j]) == body and b.joint_type[j] == newton.JointType.FREE
               for j in range(len(b.joint_type)))


def gains_of(t: Any) -> Gains:
    K = t.decision.numbers["k_compliant"] if is_compliant(t.decision) else pt.material_k(
        pt.MaterialEntry(t.material_key, t.obj.material))
    return Gains(float(K), float(t.obj.material.surface.restitution.value), resting_floor(t), is_compliant(t.decision))


# --- facts ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Collected:
    """``collect_all``'s result: facts and gains per dynamic colliding body, their shapes, and the start data."""

    facts: list[RB.BodyFacts]
    gains: list[Gains]
    boxes: list[list[RB.ShapeBox]]
    starts: RW.Starts | None


def collect_all(b: newton.ModelBuilder, objs: Sequence[Any], rows: Mapping[str, Any] | None) -> Collected:
    """``BodyFacts`` (with start drops and loads) and ``Gains`` of every dynamic body that collides, with the start
    data that gave the drops (no wall probe: ``collect`` adds it)."""
    facts, gains, boxes, offset, half = [], [], [], [], []
    for t in objs:
        if t.obj is None or t.obj.card.motion != "dynamic":
            continue
        body = int(t.ranges.body[0])
        shapes = RB.body_boxes(b, body)
        if not shapes:
            continue
        mass = float(b.body_mass[body])
        c, h = RB.union(shapes)
        r_gyr, slender = RB.inertia_facts(b, body, mass)
        facts.append(RB.BodyFacts(t.name, body, mass, min(x.thin for x in shapes), float(np.sort(h)[1]), r_gyr, slender,
                                  float(t.obj.card.loads.contact_speed.value)))
        gains.append(gains_of(t))
        boxes.append(shapes)
        pose = b.body_q[body]
        wc, wh = RB._rotated(pose, c, h)
        offset.append(wc - np.asarray(pose[:3], np.float64))
        half.append(wh)
    if not facts:
        return Collected([], [], [], None)
    template = np.array([np.asarray(b.body_q[f.body][:3], np.float64) for f in facts])
    placements = {t.name: t.placement for t in objs}
    P, U = RB.start_positions([f.name for f in facts], template, rows, placements)
    mass, fixed = np.array([f.mass for f in facts]), RB.statics(b)
    starts = RW.Starts(P, U, np.asarray(offset), np.asarray(half), mass, fixed)
    drop, load = RB.drops_and_loads(P + starts.offset, U, starts.half, mass, fixed)
    return Collected(RB.with_starts(facts, drop, load), gains, boxes, starts)


def collect(b: newton.ModelBuilder, objs: Sequence[Any], rows: Mapping[str, Any] | None,
            control_dt: float) -> tuple[list[RB.BodyFacts], list[Gains]]:
    """``collect_all``'s facts with each closed non-convex mesh's thinnest wall, and its gains."""
    c = collect_all(b, objs, rows)
    return [_with_walls(f, x, control_dt) for f, x in zip(c.facts, c.boxes, strict=True)], c.gains


def _with_walls(f: RB.BodyFacts, shapes: Sequence[RB.ShapeBox], control_dt: float) -> RB.BodyFacts:
    """A closed non-convex mesh's thinnest wall, probed only below what can bind (2 v control_dt)."""
    wall = RB.mesh_thin(shapes, within=2.0 * f.speed * control_dt)
    return replace(f, thin=min(f.thin, wall)) if wall < f.thin else f


# --- caps -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ctx:
    today: float
    integrator: str                     # the gains mode (IntegratorChoice.gains: an integrator, or "own" under step 2c)
    today_integrator: str = ""          # the gains mode before SCR-WP0-RIGIDDT (cap 5's baseline); "" = integrator
    fast: bool = False                  # the fast mode: cap 5 also passes on the absolute sag bound (cap 6 stays)

    @property
    def baseline(self) -> str:
        return self.today_integrator or self.integrator


def _ms(x: float) -> str:
    return f"{x * 1e3:.3g} mm"


def over(x: float, limit: float) -> bool:
    """``x`` above ``limit`` beyond rounding (grid steps are exact; a bound met exactly passes)."""
    return x > limit * (1.0 + 1e-9)


def cap_tunnel(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    step = 2.0 * f.speed * dt
    if not over(step, f.thin):
        return None
    return f"2 v dt {_ms(step)} > thinnest member {_ms(f.thin)} (v {f.speed:.3g} m/s)"


def cap_gap(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    step = f.speed * dt
    return None if not over(step, RIGID_GAP_M) else f"v dt {_ms(step)} > contact gap {_ms(RIGID_GAP_M)}"


def cap_spin(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    turn = f.speed / max(f.lever, 1e-9) * dt
    limit = SPIN_SHARE * CheckPolicy.spin_rad_per_step
    return None if not over(turn, limit) else f"{turn:.3g} rad per step > {limit:.3g} (lever {_ms(f.lever)})"


def cap_floor(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    if not f.slender or dt <= c.today:
        return None
    if over(dt, max(c.today, FLOOR_SWEPT_DT)):
        return f"slender (r_gyr {_ms(f.r_gyr)}): the inertia floor is swept to {FLOOR_SWEPT_DT * 1e3:g} ms"
    r0 = floor_radius(dt)
    return None if not over(r0, max(R0_MIN, f.r_gyr)) else f"inertia floor r0 {_ms(r0)} > r_gyr {_ms(f.r_gyr)}"


def sag(ke: float, kd: float, load: float) -> float:
    """Rest sag [m] of a contact with acceleration-space gains (ke, kd) under ``load`` x g (WP3's form, health's)."""
    return steady_state_violation((2.0 / kd, 0.5 * kd / math.sqrt(ke)), (SOLIMP_D0, DEFAULT_IMP), load * GRAVITY)


def cap_sag(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    ke, kd = native_gains(g.K, f.mass, g.e, dt=dt, zeta_min=g.zeta_min, integrator=c.integrator)
    if c.fast and not over(sag(ke, kd, f.load), SAG_SHARE * f.thin):    # else as "auto": never softer
        return None
    ke0, kd0 = native_gains(g.K, f.mass, g.e, dt=c.today, zeta_min=g.zeta_min, integrator=c.baseline)
    if not over(ke0, ke):
        return None
    fast = f"; fast mode: over {SAG_SHARE:.0%} of the thinnest member {_ms(f.thin)}" if c.fast else ""
    return (f"contacts {ke / ke0:.3g} as stiff as at the fixed step (rest sag {_ms(sag(ke, kd, f.load))} against "
            f"{_ms(sag(ke0, kd0, f.load))} under load x{f.load:.3g}{fast})")


def cap_creep(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    ke, kd = native_gains(g.K, f.mass, g.e, dt=dt, zeta_min=g.zeta_min, integrator=c.integrator)
    ke0, kd0 = native_gains(g.K, f.mass, g.e, dt=c.today, zeta_min=g.zeta_min, integrator=CREEP_REF_INTEGRATOR)
    ratio = (kd / ke / FILL_IMPRATIO) / (kd0 / ke0 / SLIDING_MIN_IMPRATIO)
    return None if not over(ratio, 1.0) else f"friction saturation speed x{ratio:.3g} of A5.speed's pass point"


def cap_fixed(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    return None if not over(dt, 0.5 * FIXED_TAU) else f"friction rows' solref {FIXED_TAU * 1e3:g} ms < 2 dt"


def cap_compliant(f: RB.BodyFacts, g: Gains, dt: float, c: Ctx) -> str | None:
    if not g.compliant:
        return None
    _, kd = native_gains(g.K, f.mass, g.e, zeta_min=g.zeta_min)
    tau = 2.0 / kd
    return None if not over(dt, 0.5 * tau) else f"compliant card's time constant {tau * 1e3:.3g} ms < 2 dt"


CAPS: dict[str, Callable[[RB.BodyFacts, Gains, float, Ctx], str | None]] = {
    "tunnel": cap_tunnel, "gap": cap_gap, "spin": cap_spin, "floor": cap_floor, "sag": cap_sag, "creep": cap_creep,
    "fixed": cap_fixed, "compliant": cap_compliant}


# --- choice -----------------------------------------------------------------------------------------------------

def failures(facts: Sequence[RB.BodyFacts], gains: Sequence[Gains], dt: float, ctx: Ctx,
             cap: Callable[..., str | None]) -> str | None:
    """The first body ``cap`` refuses at ``dt`` ("name: why"), else None."""
    for f, g in zip(facts, gains, strict=True):
        msg = cap(f, g, dt, ctx)
        if msg is not None:
            return f"{f.name}: {msg}"
    return None


Checks = Mapping[str, Callable[[float], "str | None"]]      # {cap: dt -> None or why}: the robots' caps


def broken_caps(facts: Sequence[RB.BodyFacts], gains: Sequence[Gains], dt: float, ctx: Ctx,
                robots: Checks) -> list[str]:
    """"cap: why" of every cap ``dt`` breaks: the bodies' CAPS, then the robots' checks."""
    bodies = {n: failures(facts, gains, dt, ctx, cap) for n, cap in CAPS.items()}
    return [f"{n}: {m}" for n, m in {**bodies, **{n: check(dt) for n, check in robots.items()}}.items() if m]


def scan(facts: Sequence[RB.BodyFacts], gains: Sequence[Gains], control_dt: float, ctx: Ctx,
         robots: Checks | None = None) -> tuple[float, str, tuple[tuple[str, float], ...]]:
    """(step, reason, caps): the largest candidate every cap allows, the cap and body (or robot joint or shape:
    ``robots``, the checks of ``rigid_step_robots``) that bound it."""
    steps = candidates(control_dt, ctx.today)
    allowed: dict[str, float] = {}
    why: dict[str, str] = {}
    checks = {n: (lambda dt, cap=cap: failures(facts, gains, dt, ctx, cap)) for n, cap in CAPS.items()}
    for name, check in {**checks, **(robots or {})}.items():
        allowed[name] = ctx.today
        for dt in steps:
            fail = check(dt)
            if fail is None:
                allowed[name] = dt
                break
            why[name] = f"{name} cap, {fail} at {dt * 1e3:g} ms"
    dt = min([control_dt, *allowed.values()])
    binding = min(allowed, key=lambda k: (allowed[k], k))
    reason = why[binding] if allowed[binding] < control_dt else f"the control period ({control_dt * 1e3:g} ms)"
    return dt, reason, tuple(sorted(allowed.items()))


def choose_rigid_step(b: newton.ModelBuilder, objs: Sequence[Any], scene: Any, integrator: str,
                      request: RigidStepRequest | None, report: Any = None) -> RigidStep:
    """The MuJoCo entry's step for the emitted template (module docstring); recorded in ``report`` when given."""
    control_dt = float(scene.control_dt)
    today = today_step(control_dt)
    req = request or RigidStepRequest()
    if req.mode in FIXED_MODES:
        step = RigidStep(today, today, "the fixed step (rigid_step 'fixed')",
                         eligible=why_not(b, objs, scene, req) is None)
    elif req.mode not in ("auto", FAST_MODE) and not isinstance(req.mode, (int, float)):
        raise ValueError(f"rigid_step must be 'auto', 'fast', 'fixed' or a step in seconds, got {req.mode!r}")
    elif (why := why_not(b, objs, scene, req)) is not None:
        if req.mode != "auto" and not req.from_profile:
            log.warning("rigid_step=%r is ignored: %s", req.mode, why)
        step = RigidStep(today, today, why)
    else:
        step = replace(_eligible(b, objs, scene, req, Ctx(today, integrator, fast=req.mode == FAST_MODE)),
                       eligible=True)
    first = report is not None and step.fast and FAST_COST not in report.warnings    # step 2c's trials record too
    if report is not None:
        record(report, step)
    if first and req.announce is not None:
        req.announce(step)
    warn_gap(b, objs, scene, step, req, report)
    return step


def warn_gap(b: newton.ModelBuilder, objs: Sequence[Any], scene: Any, step: RigidStep, req: RigidStepRequest,
             report: Any = None) -> None:
    """Warn once per body whose start drop breaks the gap cap at the chosen step (``rigid_step_warn``), whatever the
    mode and eligibility: the free MuJoCo bodies ``collect_all`` can describe are checked; the rest are skipped.
    Never raises, never changes the step."""
    try:
        free = [t for t in objs if _describable(b, t)]
        got = collect_all(b, free, req.rows)
        texts = RW.gap_warnings(got.facts, got.starts, step.dt, float(scene.control_dt), step.eligible)
    except Exception as exc:  # noqa: BLE001
        log.info("gap warning skipped (%s: %s)", type(exc).__name__, exc)
        return
    for text in texts:
        if report is not None and text in report.warnings:
            continue
        if report is not None:
            report.warnings.append(text)
        (req.warn or log.warning)(text)


def _describable(b: newton.ModelBuilder, t: Any) -> bool:
    """A single free dynamic body on the MuJoCo entry, routed as phase 1 does: what ``collect_all`` can describe."""
    return (t.obj is not None and t.obj.card.motion == "dynamic" and t.decision.entry == "mujoco"
            and t.decision.route in PHASE1_ROUTES and len(t.ranges.body) == 1 and _free(b, t.ranges.body[0]))


def _eligible(b: newton.ModelBuilder, objs: Sequence[Any], scene: Any, req: RigidStepRequest,
              ctx: Ctx) -> RigidStep:
    control_dt = float(scene.control_dt)
    facts, gains = collect(b, objs, req.rows, control_dt)
    robots = RR.checks(b, objs, scene, ctx.today)       # the robots' caps (none without robots)
    if not facts and not robots:
        return RigidStep(ctx.today, ctx.today, "no dynamic body collides")
    if req.mode in ("auto", FAST_MODE):
        return _auto(facts, gains, control_dt, ctx, robots)
    dt = snap_to_grid(float(req.mode), control_dt)
    broken = broken_caps(facts, gains, dt, ctx, robots)
    if broken:
        log.warning("rigid_step pinned at %g ms breaks %d cap(s): %s", dt * 1e3, len(broken), "; ".join(broken))
    return RigidStep(dt, ctx.today, f"pinned at {dt * 1e3:g} ms (rigid_step={req.mode!r})")


def _auto(facts: Sequence[RB.BodyFacts], gains: Sequence[Gains], control_dt: float, ctx: Ctx,
          robots: Checks | None = None) -> RigidStep:
    dt, reason, caps = scan(facts, gains, control_dt, ctx, robots)
    return RigidStep(dt, ctx.today, f"fast mode: {reason}" if ctx.fast else reason, caps, fast=ctx.fast)


def record(report: Any, step: RigidStep) -> None:
    """The step on the route report, one INFO line, and a reason on every dynamic object's decision when the step is
    not the fixed step (the router's decisions themselves are untouched)."""
    report.rigid_step = step
    log.info("rigid step %g ms (the fixed step: %g ms): %s", step.dt * 1e3, step.today * 1e3, step.reason)
    if step.fast and FAST_COST not in report.warnings:     # step 2c's trials record on the same report
        report.warnings.append(FAST_COST)
    if step.dt == step.today:
        return
    why = (f"rigid step {step.dt * 1e3:g} ms (the fixed step: {step.today * 1e3:g} ms; scene.rigid_step = 'fixed' "
           f"keeps it): {step.reason}" + (f"; {FAST_COST}" if step.fast else ""))
    for name, d in list(report.decisions.items()):
        if d.entry == "mujoco" and d.route in PHASE1_ROUTES:
            report.decisions[name] = replace(d, reasons=(*d.reasons, why))


__all__ = ["CAPS", "FAST_COST", "FAST_EFFECT", "FAST_MODE", "RigidStep", "RigidStepRequest", "choose_rigid_step",
           "today_step", "why_not"]
