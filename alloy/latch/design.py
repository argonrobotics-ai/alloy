# SPDX-License-Identifier: Apache-2.0
"""Closed-form design checks for sprung latches (run before any simulation; the sim tests compare against them).

- Wedge law ([internal reference]): pushing a member up a flank at angle ``a`` from the push
  axis needs ``F_push / F_lift = (tan a + mu) / (1 - mu·tan a)`` (F_lift: the member's resisting force normal to
  the push), infinite at ``mu = 1/tan a``: every ramp must stay below ``atan(1/mu_max)`` with margin.
- Cantilever tab as an equivalent root hinge: tip rate ``3EI/L³``, hinge rate ``k_theta = 3EI/L`` (I = b t³/12),
  root strain at tip deflection d: ``eps = 3 t d / (2 L²)`` (Euler-Bernoulli; Roark 7th ed. Table 8.1 case 1a).
- Catch normal flip (CON-48-type): box-box contact keeps the axial normal while the axial penetration
  ``F/K`` stays below the lift overlap; require ``F_hold/K <= overlap/2``.
- Slide regularisation weight: added mass x g must stay under 10 % of the preload.
- Snap per step: preload·dt²/I from rest; over 5 % of the travel, set ``Member.snap_step`` (DC4).
- Thin touched boxes: thinner than twice the contact depth, the box normal can flip to the back face (CON-19).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alloy.latch.mass import axis_inertia, member_armature, member_props, mode_omega, snap_per_step
from alloy.latch.spec import ROOT, TOUCHED, LatchSpec

G = 9.80665
SNAP_FRACTION = 0.05      # warn when one step under the preload alone covers more than this share of the travel


def wedge_ratio(angle: float, mu: float) -> float:
    """F_push / F_n to drive a member up a flank at ``angle`` (rad, from the push axis) with friction ``mu``;
    inf when self-locking."""
    t = math.tan(angle)
    den = 1.0 - mu * t
    return math.inf if den <= 0.0 else (t + mu) / den


def lock_angle(mu_max: float) -> float:
    """Flank angle at which the wedge self-locks: atan(1/mu_max)."""
    return math.atan(1.0 / mu_max)


def cantilever_hinge(E: float, width: float, thickness: float, length: float) -> float:
    """Equivalent root-hinge rate k_theta = 3EI/L (N·m/rad) of a cantilever tab (I = b t³/12)."""
    return 3.0 * E * (width * thickness**3 / 12.0) / length


def cantilever_strain(thickness: float, length: float, tip_deflection: float) -> float:
    """Root bending strain of a cantilever deflected ``tip_deflection`` at its tip: 3 t d / (2 L²)."""
    return 1.5 * thickness * tip_deflection / length**2


@dataclass(frozen=True)
class Issue:
    level: str               # "error" | "warn"
    where: str
    message: str


def check_ramp(name: str, angle: float, mu_max: float, margin: float = math.radians(10.0)) -> list[Issue]:
    lim = lock_angle(mu_max)
    if angle >= lim:
        return [Issue("error", name, f"flank {math.degrees(angle):.1f} deg self-locks at mu {mu_max} "
                                     f"(limit {math.degrees(lim):.1f} deg)")]
    if angle > lim - margin:
        return [Issue("warn", name, f"flank {math.degrees(angle):.1f} deg within {math.degrees(margin):.0f} deg "
                                    f"of self-locking at mu {mu_max} (CON-61)")]
    return []


def check_flip(name: str, hold_force: float, contact_stiffness: float, overlap: float) -> list[Issue]:
    pen = hold_force / contact_stiffness
    if pen > 0.5 * overlap:
        return [Issue("error", name, f"axial penetration {pen * 1e3:.3f} mm at {hold_force:.0f} N exceeds half the "
                                     f"catch overlap {overlap * 1e3:.3f} mm: the box normal can flip")]
    return []


def check_thin_boxes(spec: LatchSpec, contact_depth: float) -> list[Issue]:
    """Touched boxes thinner than twice the expected contact depth: past half the thickness the box-box normal
    turns to the back face and the contact pushes the wrong way (CON-19; DC4's thin hook ramp)."""
    out: list[Issue] = []
    for part in (ROOT, *[m.name for m in spec.members]):
        for b in spec.part_shapes(part):
            t = 2.0 * min(b.half)
            if b.role in TOUCHED and b.collide and t < 2.0 * contact_depth:
                out.append(Issue("error", b.name, f"{b.role} box {t * 1e3:.2f} mm thick < 2 x contact depth "
                                                  f"{contact_depth * 1e3:.2f} mm: its normal can flip to the back face"))
    return out


def check_spec(spec: LatchSpec, contact_depth: float | None = None) -> list[Issue]:
    """Regularisation report, slide weight and snap-per-step checks for every member; with ``contact_depth`` (m,
    e.g. hold force / pair stiffness) also ``check_thin_boxes``."""
    out: list[Issue] = check_thin_boxes(spec, contact_depth) if contact_depth is not None else []
    for m in spec.members:
        p, arm = member_props(m, spec.design_dt), member_armature(m, spec.design_dt)
        wh = mode_omega(m, p, arm) * spec.design_dt
        snap = snap_per_step(m, spec.design_dt, axis_inertia(m, p) + arm)
        if snap > SNAP_FRACTION * m.travel:
            out.append(Issue("warn", m.name, f"preload moves it {snap:.4g} per step from rest ({snap / m.travel:.0%} "
                                             f"of its travel): set Member.snap_step (DC4)"))
        if m.regularise == "armature" and spec.static and arm > 0.0:
            out.append(Issue("warn", m.name, "armature-regularised under a welded root: MuJoCo's contact metric "
                                             "ignores the armature here; give touched faces precision pairs"))
        if wh > 0.351:
            out.append(Issue("error", m.name, f"omega·h {wh:.3f} > 0.35 after regularisation"))
        if m.kind == "slide" and p.added_mass * G > 0.1 * max(m.spring.preload, 1e-12):
            out.append(Issue("warn", m.name, f"regularisation adds {p.added_mass * 1e3:.2f} g "
                                             f"({p.added_mass * G:.3f} N) vs preload {m.spring.preload:.3f} N"))
        if not p.principal_ok():
            out.append(Issue("error", m.name, "inertia not positive definite / triangle inequality"))
    return out


def regularisation_table(spec: LatchSpec) -> list[dict]:
    """One row per member: physical mass, added inertia/mass, omega·h at design_dt (for results notes)."""
    rows = []
    for m in spec.members:
        p, arm = member_props(m, spec.design_dt), member_armature(m, spec.design_dt)
        rows.append({"member": m.name, "kind": m.kind, "rate": m.spring.rate, "preload": m.spring.preload,
                     "mass_kg": m.mass, "added_inertia": p.added_inertia, "added_mass": p.added_mass,
                     "armature": arm, "omega_h": mode_omega(m, p, arm) * spec.design_dt,
                     "design_dt": spec.design_dt})
    return rows
