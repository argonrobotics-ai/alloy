# SPDX-License-Identifier: Apache-2.0
"""Readback of a ``cloth_film`` object (SCR-SC8-1: alloy/scene/readback_checks.py ``CHECKERS[CLOTH_FILM]``).

Checks what the film emitter (emit.py) writes against the card: scaled membrane ke (membrane_scale·E·h/(1−ν²)),
tri_ka = 0, tri_kd = β·ke; per-hinge bending ke from D (topology.bending_ke on the rest mesh; a card mesh's crease
hinges, its OBJ lines, D/(crease_ratio·h) when the material sets crease_ratio; a ``plastic.model: j2`` card's interior
hinges D/l_h with their section E', σ_y', H', t and l_h, emit._j2_hinges); mass = ρ·h·area;
particle radius = the film contact radius; ``film:strain_cap`` = yield_stress/E, ``film:air``, ``film:yield_angle``.
"""

from __future__ import annotations

import numpy as np

from .emit import crease_mask, hinge_length, obj_creases, spec_from_card
from .topology import bending_ke


def check_film(built, plan, t):
    from alloy.scene.readback_checks import Check, _cloth_area, gids, per_world

    m, obj = built.model, t.obj
    spec = spec_from_card(obj)
    ex = spec.extras
    E = per_world(built, t, "material.elastic.youngs", spec.youngs)
    nu, h = spec.poisson, spec.thickness
    tri_ke = ex.membrane_scale * E * h / (1.0 - nu * nu)
    tris, edges, parts = gids(t, built, "tri"), gids(t, built, "edge"), gids(t, built, "particle")
    tm, eb = m.tri_materials.numpy(), m.edge_bending_properties.numpy()
    x = m.particle_q.numpy().astype(np.float64)
    ei = m.edge_indices.numpy()[edges[0]].astype(np.int64)
    D = E * h**3 / (12.0 * (1.0 - nu * nu))
    ke0 = bending_ke(x, ei, 1.0)                             # per unit D, rest mesh of world 0
    want_ke = D[:, None] * ke0[None, :]
    creases = obj_creases(obj)
    if creases is not None and ex.crease_ratio > 0.0:           # fold lines: κ = D/(L*/h · h) (emit.add_film_mesh)
        local = np.full(int(parts.max()) + 1, -1, np.int64)
        local[parts[0]] = np.arange(parts.shape[1])
        rows = np.where(ei >= 0, local[np.maximum(ei, 0)], -1)
        want_ke[:, crease_mask(rows, creases)] = (D / (ex.crease_ratio * h))[:, None]
    checks = []
    if spec.j2_yield > 0.0:
        lh = hinge_length(x, ei)
        want_ke = np.where(lh > 0.0, D[:, None] / np.maximum(lh, 1e-30)[None, :], want_ke)
        checks += _j2_checks(Check, m, spec, E, edges, lh)
    checks += [
        Check("material.elastic.youngs", tri_ke[:, None] * np.ones(tris.shape[1]), tm[tris, 0]),
        Check("cloth.tri_ka", np.zeros(tris.shape), tm[tris, 1]),
        Check("material.elastic.rayleigh_beta", spec.rayleigh_beta * tri_ke[:, None] * np.ones(tris.shape[1]),
              tm[tris, 2], atol=1e-12),
        Check("material.elastic.poisson", want_ke, eb[edges, 0],
              atol=1e-5 * float(np.max(want_ke, initial=0.0))),            # rest areas re-read in float32
        Check("material.plastic.yield_stress", np.full(tris.shape, spec.strain_cap, np.float32),
              m.film.strain_cap.numpy()[tris]),
        Check("card.features.thickness", np.full(parts.shape, ex.contact_radius), m.particle_radius.numpy()[parts]),
    ]
    area = _cloth_area(m, tris[0])
    rho = per_world(built, t, "material.density", spec.density)
    checks.append(Check("material.density", rho * h * area, m.particle_mass.numpy()[parts].sum(axis=1)))
    unsupported = {"material.surface.mu_dynamic": "film soft contacts use the dominant deformable's μ (§3.4)"}
    return checks, unsupported


def _j2_checks(Check, m, spec, E, edges, lh) -> list:
    """The J2 section on the interior hinges: E' = E/(1-ν²), σ_y' = 2σ_y0/√3, H' = 4H/3, t and l_h (0 elsewhere)."""
    from alloy.solvers.plastic.j2 import plane_strain

    ns, on = m.film, (lh > 0.0)[None, :]
    nu = spec.poisson
    _, syp, hp = plane_strain(spec.youngs, nu, spec.j2_yield, spec.j2_hardening)
    full = lambda v: np.where(on, v, 0.0) * np.ones(edges.shape)                # noqa: E731
    return [Check("material.plastic.yield_stress", full(syp), ns.j2_yield.numpy()[edges]),
            Check("material.plastic.hardening_modulus", full(hp), ns.j2_hardening.numpy()[edges], atol=1e-3),
            Check("material.elastic.youngs", full((E / (1.0 - nu * nu))[:, None]), ns.j2_modulus.numpy()[edges]),
            Check("card.features.thickness", full(spec.thickness), ns.j2_thickness.numpy()[edges], atol=1e-12),
            Check("film.hinge_length", full(lh[None, :]), ns.hinge_length.numpy()[edges], atol=1e-9)]
