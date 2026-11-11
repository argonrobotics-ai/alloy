# SPDX-License-Identifier: Apache-2.0
"""CLOTH_VBD hinge map and the opt-in "accurate cloth" profile (SCR-WP11-3 (3); CARD-8, ENGINE item 1).

Newton's hinge energy is ½·edge_ke·L·θ² per edge, so writing the plate rigidity D into every hinge (Newton's USD
importer mapping, `derive.cloth_stiffness`) gives a bending energy per area of about D·cell instead of D: ~60x too
soft on CARD-8's 12.5 mm cell. Every CLOTH_VBD sheet therefore gets the plate-consistent map edge_ke = c·D·L/(A1+A2)
per hinge (`derive.plate_hinge_ke`, c = `HINGE_C`), edge_kd = β·edge_ke, 0 on boundary edges (VBD has no hinge
there); the CLOTH_VBD readback expects the same values from the model's rest positions. CLOTH_STYLE3D keeps D.

The map alone settled stiff sheets far past their sag at 10 iterations (CARD-8 30x Navier): Newton starts each step at
the inertial target x + h·v + h²·g, and a finite iteration count leaves part of h²·g in every step. The scene's VBD
entry (`make_vbd`) passes particle_pinned_piece_start, so a pinned piece starts from x + h·v and its static
equilibrium is a fixed point (CARD-8 1.08x at the test's 0.8 s; CARD-1 coupons 0.95-1.02x at 10 iterations;
measurements in docs/notes/wp11-results.md). A sheet resting on or draping over a body keeps the inertial start.

`SceneAssembler(library, router, cloth_profile="accurate")` tags every CLOTH_VBD decision with `ACCURATE_EFFECT`, and
the scene's VBD entry runs `ACCURATE_ITERATIONS` iterations (every VBD object of the scene, soft ones too). N was
measured on the inertial start, where it removed the bias by brute force; every report of the profile prints
`ACCURATE_CAVEAT`. A scene without a CLOTH_VBD object is unchanged under either profile.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from alloy.core.errors import AssemblyError
from alloy.route.types import Route, RouteDecision
from alloy.scene import derive

log = logging.getLogger("alloy.scene")

CLOTH_PROFILES = ("default", "accurate")
ACCURATE_EFFECT = "vbd:profile=accurate_cloth"
HINGE_C = 0.75                    # converged CARD-1 coupons 0.988-1.016 x Navier at 3,000 iterations (wp11-results)
ACCURATE_ITERATIONS = 6_000       # smallest measured N, all seven CARD-1 coupons within 10 % of Navier (pin 8f3466f9)
ACCURATE_CAVEAT = (f"accurate cloth profile: VBD at {ACCURATE_ITERATIONS} iterations per step (default 10; the "
                   "plate hinge map c·D·L/(A1+A2) is the CLOTH_VBD default); exact for free cloth, slower to converge "
                   "for cloth in contact; cost per step in docs/notes/wp11-results.md; runs without CUDA graphs "
                   "(build_simulation skips capture: one exchange is far over the graph node budget)")
HINGE_READBACK_RTOL = 1.0e-4      # the emitter uses the builder's float64 rest positions, the model float32 ones


def is_accurate(d: RouteDecision) -> bool:
    return d.route == Route.CLOTH_VBD and ACCURATE_EFFECT in d.scene_effects


def tag_accurate(objects: Sequence) -> list[str]:
    """Add `ACCURATE_EFFECT` to every CLOTH_VBD template object's decision (in place); the names tagged."""
    names = []
    for t in objects:
        d = t.decision
        if d is not None and d.route == Route.CLOTH_VBD and ACCURATE_EFFECT not in d.scene_effects:
            t.decision = replace(d, scene_effects=(*d.scene_effects, ACCURATE_EFFECT))
            names.append(t.name)
    return names


def checked_profile(name: str) -> str:
    if name not in CLOTH_PROFILES:
        raise AssemblyError(f"unknown cloth profile {name!r}; expected one of {CLOTH_PROFILES}")
    return name


def apply_to_plan(plan, profile: str) -> None:
    """Under "accurate", tag the assembler plan's CLOTH_VBD decisions with `ACCURATE_EFFECT` (emitter, readback, VBD
    entry), add the scene effect and the cost caveat to its route report; "default" leaves the plan unchanged."""
    if profile != "accurate":
        return
    names = tag_accurate(plan.objects)
    if not names:
        log.info("accurate cloth profile: no CLOTH_VBD object; the scene is unchanged")
        return
    for t in plan.objects:
        if t.name in names:
            plan.report.decisions[t.name] = t.decision
    if ACCURATE_EFFECT not in plan.report.scene_effects:
        plan.report.scene_effects.append(ACCURATE_EFFECT)
    plan.report.warnings.append(f"{ACCURATE_CAVEAT} ({', '.join(names)})")
    log.info("accurate cloth profile on %s", ", ".join(names))


def hinge_geometry(rest: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(L, A1 + A2) per hinge; `edges` rows are Newton's [o0, o1, v0, v1] (o = -1 on a boundary edge: area sum 0)."""
    rest = np.asarray(rest, np.float64)
    v0, v1 = rest[edges[:, 2]], rest[edges[:, 3]]
    inner = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    area = np.zeros(len(edges))
    for k in (0, 1):
        p = rest[edges[inner, k]]
        area[inner] += 0.5 * np.linalg.norm(np.cross(v0[inner] - p, v1[inner] - p), axis=1)
    return np.linalg.norm(v1 - v0, axis=1), area


def hinge_ke(rest: np.ndarray, edges: np.ndarray, D: float) -> np.ndarray:
    """Per-hinge edge_ke = HINGE_C·D·L/(A1+A2) [N] for edges indexing `rest`."""
    length, area_sum = hinge_geometry(rest, edges)
    return derive.plate_hinge_ke(D, length, area_sum, HINGE_C)


def rewrite_hinges(b, e0: int, p0: int, D: float, beta: float) -> np.ndarray:
    """Write the plate map into the builder's edges e0.. (one cloth added at particles p0..); returns edge_ke. The
    emitter writes it for every CLOTH_VBD sheet (CARD-8, ENGINE item 1); the accurate profile only adds iterations."""
    rest = np.asarray(b.particle_q[p0:], np.float64)
    local = np.asarray(b.edge_indices[e0:], np.int64)
    local = np.where(local >= 0, local - p0, local)
    ke = hinge_ke(rest, local, D)
    b.edge_bending_properties[e0:] = [(float(k), float(beta * k)) for k in ke]
    return ke


def expected_edge_ke(model, edges: np.ndarray, D: np.ndarray, plate: bool) -> tuple[np.ndarray, float]:
    """Readback: [W, E] expected edge_ke of one sheet (`edges` its [W, E] global edge ids, `D` per world) and the
    atol: HINGE_C·D_w·L/(A1+A2) from the model's rest positions (`plate`: CLOTH_VBD, every profile), else D.

    The accurate map reads the geometry from world 0's `model.particle_q`, which holds while sheets are emitted at
    rest and identical per world; a pre-draped start would need the builder's rest positions."""
    if not plate:
        return D[:, None] * np.ones(edges.shape[1]), 0.0
    unit = hinge_ke(model.particle_q.numpy(), model.edge_indices.numpy()[edges[0]], 1.0)
    exp = D[:, None] * unit[None, :]
    hb = getattr(getattr(model, "cloth", None), "bend_hb", None)      # hysteretic hinges carry (1 + LOCK) k
    if hb is not None:
        from alloy.solvers.vbd.hysteresis import LOCK  # noqa: PLC0415

        exp = np.where(hb.numpy()[edges] > 0.0, exp * (1.0 + LOCK), exp)
    return exp, HINGE_READBACK_RTOL * float(np.abs(exp).max(initial=0.0))


__all__ = ["ACCURATE_CAVEAT", "ACCURATE_EFFECT", "ACCURATE_ITERATIONS", "CLOTH_PROFILES", "HINGE_C",
           "expected_edge_ke", "hinge_geometry", "hinge_ke", "is_accurate", "rewrite_hinges", "tag_accurate"]
