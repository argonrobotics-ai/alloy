# SPDX-License-Identifier: Apache-2.0
"""Contact gains for SolverMuJoCo's own (native) contacts, written into the template's shape ke/kd.

With `use_mujoco_contacts=True` Newton sets geom_solref = convert_solref(shape_ke, shape_kd, 1, 1)
(newton@ac7f7830:newton/_src/solvers/mujoco/kernels.py:2523-2534, 190-204): acceleration-space gains, and **kd = 0
falls back to the default solref (0.02 s, 1)**, which makes a 0.3 g chain link's contact ~15 N/m. So every shape on
a MuJoCo-owned body gets gains by its entry's gains mode and its resting floor (`impact_applies`):
- balls (one-point contacts: no resting floor, `resting_floor` 0) in rk4 entries and in implicitfast entries on rk4's
  or their own gains (free-body clutter and the fast mode) whose dynamic bodies are all balls get impact-matched gains
  (`alloy.contact.impact_gains`): ke = K/m, so a single-body contact is a spring of the material's force-space
  stiffness K during an impact (m the body mass), and kd = 2ζ·√ke with ζ the zero-force-release damping for
  max(e, 0.02), both up to the stability cap below and never under `static_gains` (`impact_applies`,
  `static_entry`). One floored dynamic body (a free box) keeps the whole entry static: a geom carries one solref, and
  a ball's contact with a free box mixes the two geoms' (or takes the box's under a solmix weight, sized for the
  box's mass: a 4.1 kg ball dropped on a free 212 kg steel box then sank past 1 cm), while a pair row for every
  ball-box pair could reach millions of rows in clutter;
- everything else keeps `static_gains`: ke = K·(1−imp)/m (K at rest, about 5 % of K on an impact) and
  ζ = −ln e/√(π² + ln²e) (`contact_zeta`, which realises e 0.135 at e = 0 and reads every e a little high).
  Multi-point and CCD contacts (boxes, capsules, cylinders, meshes) do: MuJoCo's friction rows take the geom's solref
  uncapped, and impact gains' larger kd pulled a sliding 27 kg cube up its contacts (on rk4 at 4.6 m/s it rocked
  8.6 deg against 3.9; at a 2.5 ms implicitfast step it hopped 4-15 mm against 0.6-2.0), while a ball rolls or skids
  on one point (no rocking: a 4.1 kg ball skidding at 4.6 m/s rose 0.93 mm against 0.90). So do robots, kinematic
  objects, chains and free bodies beside any of them (their entries' gains mode is not in IMPACT_MODES).
The rigid-step planner sizes steps on `static_gains` in every mode, so its step choices do not change. MuJoCo mixes the
two geoms' solref by solmix, so body–body pairs get an average, not the series value: exact pair values need
native = "newton" (the pair-table per-contact path overrides these shape gains, kernels.py:663-685).
Static and kinematic shapes are left alone.

Stability cap (`stable_ke`): MuJoCo's soft contact follows r'' = (1 − d)·a₀ − d·(b·r' + k·r) with k = ke/dmax²-scaled
gains (convert_solref: τ = 2/kd, ζ = kd/2·√(1/ke)), so stepped at dt it is stable iff dt²·ke/dmax + 2·dt·kd < 4
(semi-implicit Euler, d ≤ dmax). MJW's own clamp τ ≥ 2·dt is this bound with margin S = (0.25/dmax + 2)/4 at ζ = 1
only; with ζ(e) < 1 a light, stiff body passes the τ clamp and still diverges (a 19 g beech cube on plywood at e = 0.5,
dt = 1.25 ms: τ = 2.9 ms, ζ = 0.22, dt·√ke = 2.0, bounced off the table within 0.4 s). `native_gains(..., dt=)`
caps ke at the same margin S for the pair's ζ; the realised static stiffness is then below K, as under MJW's clamp.

rk4 accuracy cap (RK4_KE_FRACTION): rk4 entries (SCR-WP6-4, `integrator_route`) are not symplectic: at the semi-implicit
cap (ωh ≈ 1.3) each step keeps |R(iωh)| ≈ 0.85 of the contact oscillation and a 0.1 m drop at e = 0.8 rebounded to
51.7 mm vs e²h0 = 64 mm (A6 fails, 12.3 vs 7.3 mm). A quarter of the cap (ωh ≈ 0.65, π(ωh)⁵/144 ≈ 0.25 % amplitude
loss per contact) gives 64.0 mm at e 0.8, 29.1 mm at e 0.5 (bound 5.3 mm) and a 0.03 mm hop at e 0 (CPU, assembler
path, `[internal note]`); the static sink grows 4x (0.007 → 0.027 mm for a 35 g ball).

Resting floor on CCD contacts (ZETA_CCD): contacts MuJoCo generates through its general convex path (meshes, MJCF
assemblies, cylinders, cones, ellipsoids; `entries.ccd_backed`) change their point set from step to step, and at
ζ(e) < 0.5 light bodies never come to rest: on an L4 a USB plug (e 0.6, ζ 0.16) walked 2.9 mm and a beech cube on a
drilled PMMA block 0.2-1.1 mm per 0.25 s, nuts, an oak spoon and a hinged box 0.3-0.8 mm (CARD-1). At ζ = 1 all rest
at 0.000 mm; the hinged box needs 0.5 on the L4 (1.7 mm at 0.4, 0.000 at 0.5 and 0.7), the plug 0.4 (CPU); more
solver iterations or a pyramidal cone do not help (plug 5.4 / 1.05 mm). These contacts take ζ = max(ζ(e), 0.5), which
realises e <= 0.30 (`restitution_cap`, zero-force release), the price of resting on MuJoCo's native path.

The same floor on multi-point primitives (`resting_floor`, 2026-11-07): a box face or a capsule on its side rests on
2-4 contact points, and at ζ(e) < 0.5 it never comes to rest at the route's 1.25 ms step either. A 0.1 kg box (e 0.5,
ζ 0.215) landed from 0.1-1 mm keeps a 10-19 µm normal oscillation for good, on flat ground and in plain MuJoCo C
(frictionless too), and on a slope it walks down at a pair μ well above tan θ: 1.7 mm/s at 10°, 4-5 mm/s at 30°,
6 mm/s at 40° (μ 1.09), where a box placed in contact holds at 0.01 mm/s. A sphere (one point) rests at 0.215. The
walk stops from ζ 0.5 at the same stiffness (0.4 still walks at 4 mm/s), or at dt 0.2 ms. Dynamic boxes and capsules
take ζ ≥ ZETA_MULTIPOINT = 0.7: at 0.5 2 of 16 landings (0.05-2 mm) still walked at 40°, at 0.6 and 0.7 none over
5 g-1 kg and 20-40° (CPU); on an L4, 64 landings: no floor 63 walk at 40° and 17 at 10°, 0.7 none. They realise
e <= 0.21 (`restitution_cap`); spheres and planes keep the card's e. In an explicit pair row the multi-point floor
comes from the dynamic sides only (a sphere bouncing on a static box table keeps its e) and the CCD floor from either
side. CCD shapes landed the same way rest at 0.5 (CPU, 16 landings at 40°: a cylinder on its face and a convex-mesh
box 0 walking; without the floor 16 and 13 walk).

Entries on implicitfast's own gains (robots, kinematic objects, chains: gains mode "implicitfast") take
ZETA_MULTIPOINT_IMPLICIT = 1.0 instead (2026-11-14). There a resting stack of light boxes never comes to rest at
0.7: an 18-level tower of 12.6 g crossed blocks beside a kinematic body bobbed at 3.9 mm/s and turned at 0.05-0.18
rad/s RMS for good, every level's contacts at 66-100 % of their friction limit, so a push on one block sheared the
whole tower (32 mm at the top). At 0.85 it still bobbed at 2.7 mm/s; at 0.95 and 1.0, 0.001 mm/s. Under rk4 (MuJoCo
C, or free bodies on MuJoCo Warp) the same tower rests at 0.7. On an L4 (CUDA) it bobbed at 3.3 mm/s at 0.7 and
rests at 0.03 mm/s at 1.0; a 10-level stack there still bobbed at 1.7-1.8 mm/s after a 0.5 s settle at 0.7 (at rest
after 2 s) and is quiet within 0.5 s at 1.0, though CUDA contact varies from run to run (0.0004-0.565 mm/s in eight
runs). The price is a little rebound: a 5 cm aluminium or glass box dropped 10 cm onto steel beside a kinematic body
rebounds at e 0.22 (0.26 and 0.28 at 0.7); the stability cap then sets the contact's time constant, so e reads above
`restitution_cap` (0.135). rk4 and free-body clutter (gains modes "rk4" and "own", whose rigid step the planner
sizes) keep 0.7.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator, Sequence

from alloy._vendor import newton

from alloy.contact import chain_grip
from alloy.contact.inertia_floor import floor_template  # noqa: F401 - re-exported for the assembler (SCR-WP3 floor)
from alloy.scene import derive
from alloy.contact import impact_gains as IG
from alloy.scene import pairs as pt
from alloy.scene.template import TemplateObject, is_compliant

log = logging.getLogger("alloy.scene")
DEFAULT_IMP = 0.95  # Newton's default mujoco:geom_solimp dmax


STABILITY_MARGIN = (0.25 / DEFAULT_IMP + 2.0) / 4.0   # MJW's τ ≥ 2·dt at ζ = 1, in units of the bound 4


RK4_KE_FRACTION = 0.25   # rk4 entries: ke at 1/4 of the cap, ωh about 0.65 (module docstring, rk4 accuracy cap)
OWN_KE_FRACTION = 0.6    # gains mode "own" (step 2c): implicitfast stacks at 0.6 of the cap (WP3 + WP6 rows, R2)
KE_FRACTION = {"rk4": RK4_KE_FRACTION, "own": OWN_KE_FRACTION}   # gains mode -> fraction of the cap (else 1)


def stable_ke(zeta: float, dt: float, imp: float = DEFAULT_IMP, integrator: str = "implicitfast") -> float:
    """Largest ke with dt²·ke/imp + 4ζ·dt·√ke ≤ 4·STABILITY_MARGIN (module docstring); under rk4 a quarter of it.
    ``integrator`` is the gains mode: an integrator, or "own" (``integrator_route.own_gains``: implicitfast at
    OWN_KE_FRACTION of the cap, chosen only where every contact then equals today's)."""
    u = 2.0 * (math.sqrt(zeta * zeta * imp * imp + STABILITY_MARGIN * imp) - zeta * imp)   # u = dt·√ke
    return (u / dt) ** 2 * KE_FRACTION.get(integrator, 1.0)


ZETA_CCD = 0.5   # damping-ratio floor on contacts MuJoCo collides through its CCD path (module docstring)
MULTIPOINT_PRIMITIVES = ("box", "capsule")   # rest on 2-4 contact points (module docstring)
ZETA_MULTIPOINT = 0.7   # their damping-ratio floor under rk4's and free-body clutter's gains (module docstring)
ZETA_MULTIPOINT_IMPLICIT = 1.0   # the same floor under implicitfast's own gains (module docstring)


def resting_floor(t: TemplateObject, gains: str = "rk4") -> float:
    """The damping-ratio floor an object's contacts need to come to rest at the route's step: ZETA_CCD for shapes on
    MuJoCo's CCD path, ZETA_MULTIPOINT (ZETA_MULTIPOINT_IMPLICIT under the gains mode "implicitfast") for multi-point
    primitives (box, capsule), 0 for spheres and planes (module docstring)."""
    from alloy.cards import types as ct
    from alloy.scene.entries import ccd_backed

    if t.obj is None:
        return 0.0
    s = t.obj.card.shape
    multipoint = isinstance(s, ct.Primitive) and s.kind in MULTIPOINT_PRIMITIVES
    zeta = ZETA_MULTIPOINT_IMPLICIT if gains == "implicitfast" else ZETA_MULTIPOINT
    return max(zeta if multipoint else 0.0, ZETA_CCD if ccd_backed(t) else 0.0)


def contact_zeta(e: float, zeta_min: float = 0.0) -> float:
    """The damping ratio of `static_gains` (zero-overlap release, ``derive.zeta_from_restitution``), at least
    ``zeta_min`` (a resting floor)."""
    return max(derive.zeta_from_restitution(e), zeta_min)


def impact_zeta(e: float, zeta_min: float = 0.0) -> float:
    """The damping ratio of impact-matched gains: zero-force release for max(e, E_FLOOR)
    (``alloy.contact.impact_gains``), at least ``zeta_min`` (a resting floor)."""
    return max(IG.zeta_for_restitution(e), zeta_min)


IMPACT_MODES = ("rk4", "own")   # gains modes of free-body entries: rk4, or implicitfast on rk4's / own gains + the cap


_PLANNING = [0]


@contextlib.contextmanager
def planning_rule() -> Iterator[None]:
    """Inside, every gains mode writes ``static_gains``: step 2c's trials (``integrator_route._trial``) decide on the
    contacts they decided on before impact-matched gains, so its step choices do not change."""
    _PLANNING[0] += 1
    try:
        yield
    finally:
        _PLANNING[0] -= 1


def impact_mode(integrator: str) -> bool:
    """Whether gains mode `integrator` is a free-body entry's (``IMPACT_MODES``), outside ``planning_rule``."""
    return integrator in IMPACT_MODES and not _PLANNING[0]


def impact_applies(integrator: str, zeta_min: float = 0.0) -> bool:
    """Whether a contact gets impact-matched gains: a free-body entry's gains mode and no resting floor (a ball)."""
    return impact_mode(integrator) and zeta_min <= 0.0


def static_gains(K: float, mass: float, e: float, imp: float = DEFAULT_IMP, dt: float | None = None,
                 zeta_min: float = 0.0, integrator: str = "implicitfast") -> tuple[float, float]:
    """(ke, kd) of the static rule, written outside ``IMPACT_MODES`` and kept as the rigid-step planner's sizing rule:
    ke = K (1 - imp) / m (K at rest), capped at `stable_ke` with `dt`; kd = 2 contact_zeta sqrt(ke)."""
    zeta = contact_zeta(e, zeta_min)
    ke = K * (1.0 - imp) / mass
    if dt is not None:
        ke = min(ke, stable_ke(zeta, dt, imp, integrator))
    return ke, max(2.0 * zeta * math.sqrt(ke), derive.TINY_DAMPING)


def impact_gains(K: float, mass: float, e: float, imp: float = DEFAULT_IMP, dt: float | None = None,
                 zeta_min: float = 0.0, integrator: str = "implicitfast") -> tuple[float, float, float]:
    """(ke, kd, zeta) for gains mode `integrator`. Unless ``impact_applies`` (a free-body entry and no resting floor):
    ``static_gains`` and ``contact_zeta``. Else matched to an impact (``alloy.contact.impact_gains``): ke = K / m,
    zeta = impact_zeta; with `dt`, ke is capped at `stable_ke` for that zeta, and never under `static_gains`: where the
    cap is lower the row keeps the static ke at the largest zeta that ke allows (``impact_gains.zeta_at``), never less
    than the static zeta."""
    if not impact_applies(integrator, zeta_min):
        return (*static_gains(K, mass, e, imp, dt, zeta_min, integrator), contact_zeta(e, zeta_min))
    zeta = impact_zeta(e, zeta_min)
    ke = K / mass
    if dt is not None:
        ke = min(ke, stable_ke(zeta, dt, imp, integrator))
        ke0, _ = static_gains(K, mass, e, imp, dt, zeta_min, integrator)
        if ke < ke0:
            frac = KE_FRACTION.get(integrator, 1.0)
            ke = ke0
            zeta = max(min(zeta, IG.zeta_at(ke0, dt, imp, STABILITY_MARGIN, frac)), contact_zeta(e, zeta_min))
    return ke, max(2.0 * zeta * math.sqrt(ke), derive.TINY_DAMPING), zeta


def native_gains(K: float, mass: float, e: float, imp: float = DEFAULT_IMP, dt: float | None = None,
                 zeta_min: float = 0.0, integrator: str = "implicitfast") -> tuple[float, float]:
    """(ke, kd) acceleration-space shape gains written for a body of `mass` on a force-space K under gains mode
    `integrator` (``impact_gains``: impact-matched where ``impact_applies``, else ``static_gains``)."""
    ke, kd, _ = impact_gains(K, mass, e, imp, dt, zeta_min, integrator)
    return ke, kd


def rigid_dt_max(control_dt: float) -> float:
    """Today's rigid step for ``control_dt``: snap_to_grid(1/600, control_dt). The API may step a rigid-only scene of
    free bodies coarser (``BuiltScene.rigid_dt``, ``alloy.scene.rigid_step``, SCR-WP0-RIGIDDT); every other scene
    takes this one."""
    from alloy.core.timebase import snap_to_grid
    from alloy.scene.entries import RIGID_TARGET

    return snap_to_grid(RIGID_TARGET, control_dt)


def capped_ratio(K: float, mass: float, e: float, dt: float, imp: float = DEFAULT_IMP, zeta_min: float = 0.0,
                 integrator: str = "implicitfast") -> float:
    """Realised / card stiffness under the stability cap (1.0 when uncapped): ke m / K where ``impact_applies`` (the
    impact spring), else ke m / (K (1 - imp)) (the static stiffness)."""
    if not impact_applies(integrator, zeta_min):
        return min(1.0, stable_ke(contact_zeta(e, zeta_min), dt, imp, integrator) / (K * (1.0 - imp) / mass))
    ke, _ = native_gains(K, mass, e, imp, dt, zeta_min, integrator)
    return min(1.0, ke * mass / K)


def restitution_cap(zeta_min: float) -> float:
    """The largest restitution a damping ratio of `zeta_min` still realises (zero-force release)."""
    return IG.release_speed(zeta_min)


def apply_native_gains(b: newton.ModelBuilder, objs: Sequence[TemplateObject], dt: float | None = None,
                       integrator: str = "implicitfast", report=None) -> int:
    """Write native gains on the template shapes of MuJoCo-owned dynamic objects (capped for stability at `dt`, the
    largest step the MuJoCo entry can take, under its `integrator`); returns the shapes written. Then the chain-grip
    rule (`alloy.contact.chain_grip`), which names itself in `report`'s route decisions."""
    with planning_rule() if static_entry(objs) else contextlib.nullcontext():
        written = _write_gains(b, objs, dt, integrator)
    log.debug("native MuJoCo contact gains on %d shapes", written)
    chain_grip.apply_solmix(b, objs, report)   # a CHAIN beside a moving body: its links' solref governs (CHAINGRIP)
    return written


def static_entry(objs: Sequence[TemplateObject]) -> bool:
    """Whether a MuJoCo entry keeps the static rule for every contact although its gains mode is in IMPACT_MODES: it
    has a dynamic body with a resting floor (a box, capsule, cylinder or mesh beside the balls; module docstring)."""
    return any(t.obj is not None and t.decision.entry == "mujoco" and t.obj.card.motion == "dynamic"
               and resting_floor(t) > 0.0 for t in objs)


def _write_gains(b: newton.ModelBuilder, objs: Sequence[TemplateObject], dt: float | None, integrator: str) -> int:
    written = 0
    for t in objs:
        if t.obj is None or t.decision.entry != "mujoco" or t.obj.card.motion != "dynamic":
            continue
        K = t.decision.numbers["k_compliant"] if is_compliant(t.decision) else pt.material_k(
            pt.MaterialEntry(t.material_key, t.obj.material))
        e = float(t.obj.material.surface.restitution.value)
        zmin = resting_floor(t, integrator)
        if e > restitution_cap(zmin):
            log.info("%s: CCD / multi-point contacts realise e <= %.2f (card %.2f): ζ floor %.2f for resting "
                     "stability", t.obj.card.name, restitution_cap(zmin), e, zmin)
        for s in t.ranges.shape:
            body = b.shape_body[s]
            mass = float(b.body_mass[body]) if body >= 0 else 0.0
            if mass <= 0.0:
                continue
            b.shape_material_ke[s], b.shape_material_kd[s] = native_gains(K, mass, e, dt=dt, zeta_min=zmin,
                                                                          integrator=integrator)
            written += 1
            ratio = capped_ratio(K, mass, e, dt, zeta_min=zmin, integrator=integrator) if dt is not None else 1.0
            if ratio < 1.0:
                log.info("%s shape %d: contact stiffness capped for stability at dt %.3g s, %s (%.3g of K)",
                         t.obj.card.name, s, dt, integrator, ratio)
    return written
