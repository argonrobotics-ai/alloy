# SPDX-License-Identifier: Apache-2.0
"""Elastoplastic bending hinge (crease memory).

Each hinge edge keeps SolverVBD's elastic energy ½·k·L·(θ - θ̄)² (newton vbd/particle_vbd_kernels.py:696-752, the
same angle convention: x0, x1 opposite vertices, x2→x3 the edge, θ = atan2((n̂1×n̂2)·ê, n̂1·n̂2)). After a step, when
|θ - θ̄| exceeds the hinge's yield angle θ_y the rest angle follows the hinge so that |θ - θ̄| = θ_y (perfect
plasticity, no hardening, as Narain, Pfaff and O'Brien 2013 use for bending: "Sp ← Sp + Se/‖Se‖(‖Se‖ − κ)").
Released from a flat fold (θ = π) a hinge springs back by θ_y: the yield angle is the fold-release return angle
of the dead-fold test (US10562684, sources/clips/sc8/film_bending.md). The angle difference is wrapped to (-π, π] so
a fold through ±π does not read as a 2π jump. The updated rest angles live in ``model.edge_rest_angle`` (read by
VBD every iteration); the solver restores the authored angles on reset and lists them in its state arrays.
"""

from __future__ import annotations

import warp as wp

PI = wp.constant(3.14159265358979)


@wp.func
def wrap_pi(a: float) -> float:
    return wp.atan2(wp.sin(a), wp.cos(a))


@wp.func
def hinge_angle(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> float:
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    l1 = wp.length(n1)
    l2 = wp.length(n2)
    le = wp.length(e)
    if l1 < 1.0e-12 or l2 < 1.0e-12 or le < 1.0e-12:
        return 0.0
    n1 = n1 / l1
    n2 = n2 / l2
    return wp.atan2(wp.dot(wp.cross(n1, n2), e / le), wp.dot(n1, n2))


@wp.kernel(enable_backward=False)
def plastic_hinge(
    x: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    yield_angle: wp.array[float],
    rest_angle: wp.array[float],
    plastic_work: wp.array[float],
):
    """rest_angle[e] moves to θ ∓ θ_y when |θ - θ̄| > θ_y; plastic_work[e] accumulates |Δθ̄| (rad, diagnostics)."""
    e = wp.tid()
    ty = yield_angle[e]
    if ty <= 0.0:
        return
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    if i0 < 0 or i1 < 0:
        return
    th = hinge_angle(x[i0], x[i1], x[edge_indices[e, 2]], x[edge_indices[e, 3]])
    rest = rest_angle[e]
    d = wrap_pi(th - rest)
    if wp.abs(d) <= ty:
        return
    new_rest = wrap_pi(th - wp.sign(d) * ty)
    plastic_work[e] = plastic_work[e] + wp.abs(wrap_pi(new_rest - rest))
    rest_angle[e] = new_rest


@wp.kernel(enable_backward=False)
def hinge_angles(x: wp.array[wp.vec3], edge_indices: wp.array2d[wp.int32], out: wp.array[float]):
    """Current hinge angle per edge (0 for boundary edges); for laws and diagnostics."""
    e = wp.tid()
    out[e] = 0.0
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    if i0 < 0 or i1 < 0:
        return
    out[e] = hinge_angle(x[i0], x[i1], x[edge_indices[e, 2]], x[edge_indices[e, 3]])
