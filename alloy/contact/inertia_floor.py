# SPDX-License-Identifier: Apache-2.0
"""Inertia floor for light slender free bodies (coordinator ruling 2026-11-08 05:50Z, [internal note]).

A slender free MuJoCo body whose smallest principal inertia is below m·r0² gets armature on its free joint's three
rotational dofs, so that every principal inertia is at least m·r0². r0 = R0_MIN · max(1, dt / R0_DT), dt being the
MuJoCo entry's largest rigid step: 9 mm at the default 1.25 ms. Slender: largest / smallest principal inertia >=
SLENDER (4). Compact parts (spheres, cubes, coins) and parts in precision pairs keep their inertia: the failure needs a
thin axis, and the precision ladder sizes its own pairs.

Why: a rim contact spins a light slender part about its thin axis. YCB's large marker (15.8 g, axial radius of
gyration 6.2 mm, inertia ratio 29) reaches 300-560 rad/s, 0.4-0.7 rad per 1.25 ms step, then thousands. Contacts
linearised over that much rotation inject energy. In WP13's clutter pile (100 objects, 8 worlds, CPU) 3 of 8 worlds
went INVALID: a marker at 240 m/s, +705 J against the pile's 70 J. Re-cutting the hulls or the integrator does not
stop it; the floor does.

Sweep ([internal note]; clean there = 0 of 8 INVALID and no body above 4.3 m/s, the pile's 3.3 m/s free
fall + 1 m/s):
- 0.625 ms: clean from 8 mm (6.5 mm still 1 of 8 INVALID);
- 1.25 ms: clean from 9 mm (7 mm 3 of 8 INVALID; 8 mm 0 INVALID but flings to 9.7 m/s);
- 1.5625 ms: clean at the rule's 11.25 mm (no floor: 0 INVALID, one world at 5.0 m/s).
The threshold does not fall below 1.25 ms, so r0 is flat there and grows with dt above it.

The pile law's "clean" (ruling 2026-11-08 13:36Z, tests/contact/test_inertia_floor_pile.py): each world's mechanical
energy stays at or below its start + ENERGY_TOL, no body is faster than CLEAN_SPEED (2 x the pile's free fall: a light
body struck by a heavier one moving at V leaves at up to about (1 + e) V; WP6's seed 1 hands a marker 4.47 m/s from a
2.4 m/s box while the neighbourhood loses energy), no world is INVALID and none raises SPIN. The flings the floor stops
gain energy and reach 8-240 m/s.

What changes, and only for the raised bodies (every other body is bitwise unchanged):
- spin and roll about the thin axis are slower. Rolling on a round face of radius R accelerates
  (1 + r_g²/R²) / (1 + r0²/R²) times the true value (r_g: the body's own smallest radius of gyration). The marker
  at 9 mm: 0.76x (I x2.09); tumbling +3 %.
- the build logs one warning naming every raised body and its increase. The route report carries the same
  (``RouteReport.inertia_floor``, a reason on each body's decision in ``sim.routes()``).
- opt out per scene: ``inertia_floor: false`` on the scene card (``alloy.Scene.set_inertia_floor(False)``), for
  exact rolling at a smaller rigid dt.

Known gap: docs/known_gaps/rigid-slender-spin-floor.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.core.log import get_logger

log = get_logger("contact.inertia_floor")

R0_MIN = 9.0e-3          # m: the floor's radius of gyration at and below R0_DT (sweep above)
R0_DT = 1.25e-3          # s: above this rigid step the floor grows in proportion
SLENDER = 4.0            # largest / smallest principal inertia from which a body is slender
CLEAN_SPEED = 6.6        # m/s: 2 x the pile's free fall (3.3 m/s), the pile law's hand-off limit (ruling 13:36Z)
ENERGY_TOL = 0.01        # of a world's start energy: the pile law's bound on energy gained (float32 noise)
OPT_OUT = "inertia_floor: false on the scene card (alloy.Scene.set_inertia_floor(False))"


@dataclass(frozen=True)
class Floored:
    """One body the floor raised."""

    name: str
    mass: float          # kg
    inertia: float       # smallest principal inertia before, kg·m²
    floor: float         # m·r0², kg·m²

    @property
    def ratio(self) -> float:
        return self.floor / self.inertia if self.inertia > 0.0 else float("inf")


def floor_radius(dt: float) -> float:
    """The floor's radius of gyration r0 [m] for a largest rigid step ``dt`` [s]. Example: ``floor_radius(1.25e-3)``
    -> 0.009."""
    return R0_MIN * max(1.0, float(dt) / R0_DT)


def _free_joint(b: newton.ModelBuilder, t) -> int | None:
    """The free joint of a single-body dynamic MuJoCo object outside precision pairs, else None."""
    if t.obj is None or t.decision.entry != "mujoco" or t.obj.card.motion != "dynamic" or len(t.ranges.body) != 1:
        return None
    if t.obj.card.precision_partners or t.obj.card.precision_features:
        return None
    for j in t.ranges.joint:
        if b.joint_type[j] == newton.JointType.FREE and b.joint_child[j] == t.ranges.body[0]:
            return j
    return None


def floor_candidates(b: newton.ModelBuilder, objs: Sequence, dt: float) -> list[tuple[Any, int, Floored]]:
    """The bodies ``apply_inertia_floor`` would raise, as (object, free joint, Floored), without touching the template.
    The SCR-WP6-4 integrator rule reads it before it chooses (F6 R1: an entry with a floored body keeps rk4)."""
    r0 = floor_radius(dt)
    out = []
    for t in objs:
        j = _free_joint(b, t)
        if j is None:
            continue
        body = int(b.joint_child[j])
        mass = float(b.body_mass[body])
        principal = np.linalg.eigvalsh(np.asarray(b.body_inertia[body], np.float64).reshape(3, 3))
        smallest = float(principal[0])
        add = mass * r0 * r0 - smallest
        if mass <= 0.0 or add <= 0.0 or float(principal[-1]) < SLENDER * smallest:
            continue
        out.append((t, j, Floored(t.name, mass, smallest, mass * r0 * r0)))
    return out


def apply_inertia_floor(b: newton.ModelBuilder, objs: Sequence, dt: float) -> list[Floored]:
    """Raise every slender free single-body MuJoCo object of the template below m·r0(dt)² to it, by armature on its
    free joint's rotational dofs (Newton's free-joint dofs: 3 linear, then 3 angular). Returns the raised bodies;
    others are not touched."""
    out: list[Floored] = []
    for t, j, f in floor_candidates(b, objs, dt):
        add = f.floor - f.inertia
        start = int(b.joint_qd_start[j])
        for d in range(start + 3, start + 6):
            b.joint_armature[d] = float(b.joint_armature[d]) + add
        out.append(f)
    return out


def disclose(floored: Sequence[Floored], report, dt: float) -> None:
    """One build warning naming every raised body and its increase; the same in the route report."""
    if not floored:
        return
    r0 = floor_radius(dt)
    items = ", ".join(f"{f.name} x{f.ratio:.2f}" for f in floored)
    log.warning("inertia floor: %d light slender bodies spin and roll slower about their thin axis (smallest inertia "
                "raised to m*(%.1f mm)^2 for the %.3g ms rigid step: %s); turn it off with %s (exact rolling needs a "
                "smaller rigid dt)", len(floored), r0 * 1e3, dt * 1e3, items, OPT_OUT)
    for f in floored:
        report.inertia_floor[f.name] = (f.inertia, f.floor)
        d = report.decisions.get(f.name)
        if d is not None:
            why = (f"inertia floor: smallest inertia {f.inertia:.3g} -> {f.floor:.3g} kg m^2 (x{f.ratio:.2f}, "
                   f"r0 {r0 * 1e3:.1f} mm at dt {dt * 1e3:.3g} ms; off: {OPT_OUT})")
            report.decisions[f.name] = replace(d, reasons=(*d.reasons, why))


def floor_template(b: newton.ModelBuilder, objs: Sequence, dt: float, scene, report) -> list[Floored]:
    """The floor on the emitted template, unless ``scene.inertia_floor`` is False; disclosed (``disclose``). The
    assembler calls it after the integrator choice: armature would make ``alloy.scene.integrator_route`` drop rk4,
    and the pile was measured under rk4."""
    if getattr(scene, "inertia_floor", None) is False:
        return []
    floored = apply_inertia_floor(b, objs, dt)
    disclose(floored, report, dt)
    return floored


__all__ = ["CLEAN_SPEED", "ENERGY_TOL", "OPT_OUT", "R0_DT", "R0_MIN", "SLENDER", "Floored", "apply_inertia_floor",
           "disclose", "floor_candidates", "floor_radius", "floor_template"]
