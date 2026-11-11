# SPDX-License-Identifier: Apache-2.0
"""Orthotropic woven-cloth membrane on the VBD cloth route (gap #3, SCR-WP2-15; coordinator 2026-11-08 19:58Z).

Newton's VBD cloth membrane is isotropic: one E h and one Poisson ratio, so its in-plane shear stiffness is tied to
its stretch stiffness. A woven fabric is stiff along its yarns (KES EM secant 5-18 kN/m for Petrak's fabrics) and
100-400x softer in shear between them (KES G 23-66 N/m). A card with ``elastic.cloth`` (warp / weft stretch, shear,
warp / weft bending; ``alloy.cards.types.ClothAniso``) on CLOTH_VBD gets this membrane instead of Newton's:

- Energy per rest area (Baraff and Witkin 1998, the stretch and shear conditions), with w_u, w_v the columns of the
  triangle's deformation gradient in the panel frame (u = the mesh's x = warp, v = its y = weft):
  psi = 1/2 k_u (|w_u| - 1)^2 + 1/2 k_v (|w_v| - 1)^2 + 1/2 k_s (w_u . w_v)^2,
  so at small strain k_u = E_warp h, k_v = E_weft h (N/m: the cards' stretch_warp / stretch_weft) and k_s = G h (N/m
  per radian: the card's shear). There is no Poisson coupling (KES-F does not measure one).
- Per-vertex Hessian blocks, PSD: the stretch blocks are k A c^2 (n n^T + max(C, 0) / |w| (I - n n^T)), the shear
  block k_s A (q q^T + max(0, 2 a b s) I). Damping is Rayleigh on each condition's rate (rotation-free), beta from the
  card's ``rayleigh_beta`` as for the isotropic membrane.
- Near contact, a normal regulariser in the Hessian only (forces stay exact, so the rest state does not change):
  k_s (a^2 + b^2) A along the triangle normal, on every piece Newton marks near a body or self contact. It is the
  out-of-plane term Newton's isotropic membrane keeps there, mu (a^2 + b^2) A with mu its shear modulus ("it
  regularises the local solves against the stiff, switching contact terms", particle_vbd_kernels.py), so here the
  card's shear k_s (WP11's review: a tensile-scale term would slow the normal corrections ~1000x per iteration).
- Bending stays Newton's hinge (the plate map, ``cloth_profile.rewrite_hinges``), each edge scaled to the orthotropic
  plate's rigidity across it (Huber's twist term): D(phi) = (sqrt(B_warp) cos^2 phi + sqrt(B_weft) sin^2 phi)^2, phi
  the angle of the curvature (perpendicular to the edge) from the warp. KES B_warp is the rigidity of a strip whose
  warp yarns bend, so an edge along the weft gets B_warp.

The emitter writes Newton's membrane off on these triangles (tri_ke = tri_ka = tri_kd = 0, so its kernels skip them)
and the per-triangle data into ``alloy:tri_ortho_*`` model attributes (one row per triangle; zero where isotropic).
The VBD entry's factory (``builtin_plugins._vbd``) calls ``install``, which adds the membrane through
SolverVBDAlloy's public ``after_accumulate`` colour callback: per colour, one thread per particle gathers its adjacent
triangles (deterministic, no atomics) and adds force and Hessian before Newton's solve reads them.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.newton._src.solvers.vbd.particle_vbd_kernels import _cloth_piece_is_free
from alloy._vendor.newton._src.utils.mesh import (MeshAdjacencyData, get_vertex_adjacent_face_id_order,
                                                  get_vertex_num_adjacent_faces)

log = logging.getLogger("alloy.scene.ortho_membrane")

NS = "alloy"
K_ATTR, DM_ATTR = "tri_ortho_k", "tri_ortho_dm"     # vec4 (k_u, k_v, k_s, beta), vec4 Dm^-1 (00, 01, 10, 11)
_EPS = wp.constant(1.0e-12)
_EPS_NP = 1.0e-12


# --- emit side (builder) -------------------------------------------------------------------------------------------
def _register(b: Any) -> None:
    import alloy._vendor.newton as newton  # noqa: PLC0415

    CA, F = newton.ModelBuilder.CustomAttribute, newton.Model.AttributeFrequency
    for name in (K_ATTR, DM_ATTR):
        if f"{NS}:{name}" not in b.custom_attributes:
            b.add_custom_attribute(CA(name=name, dtype=wp.vec4, frequency=F.TRIANGLE, namespace=NS,
                                      default=wp.vec4(0.0, 0.0, 0.0, 0.0)))


def panel_dm_inv(uv: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """[T, 4] inverse rest matrices (00, 01, 10, 11) of triangles `tris` (local vertex ids) in panel coordinates `uv`;
    Dm = [[u1-u0, u2-u0], [v1-v0, v2-v0]], so F = [x1-x0, x2-x0] Dm^-1. Raises on a degenerate triangle."""
    d1, d2 = uv[tris[:, 1]] - uv[tris[:, 0]], uv[tris[:, 2]] - uv[tris[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    if np.any(np.abs(det) < _EPS):
        raise ValueError(f"orthotropic membrane: {int(np.sum(np.abs(det) < _EPS))} degenerate panel triangles")
    return np.column_stack([d2[:, 1], -d2[:, 0], -d1[:, 1], d1[:, 0]]) / det[:, None]


def edge_bending_scale(uv: np.ndarray, edges: np.ndarray, b_warp: float, b_weft: float, b_card: float) -> np.ndarray:
    """Per hinge edge (rows [o0, o1, v0, v1], local ids; -1 = boundary): D(phi) / b_card (module docstring)."""
    t = uv[edges[:, 3]] - uv[edges[:, 2]]
    t = t / np.maximum(np.linalg.norm(t, axis=1), _EPS)[:, None]
    d = (math.sqrt(b_warp) * t[:, 1] ** 2 + math.sqrt(b_weft) * t[:, 0] ** 2) ** 2   # cos phi = |t_v|, sin phi = |t_u|
    return d / b_card


ALIGNED_MIN = 0.9          # share of triangles with a warp leg and a weft leg below which the build warns


def aligned_share(uv: np.ndarray, tris: np.ndarray, tol: float = 1.0e-3) -> float:
    """The share of triangles with one edge along the warp (u) and one along the weft (v) in the panel: a grid of any
    cell aspect gives 1 (the third edge is free under the shear term), D03M's grid disc with its rim on the circle 0.92,
    the R6 jittered-hex disc 0.005."""
    def along(axis: int, a: int, b: int) -> np.ndarray:
        t = np.abs(uv[tris[:, b]] - uv[tris[:, a]])
        return t[:, axis] > (1.0 - tol) * np.maximum(np.linalg.norm(t, axis=1), _EPS_NP)

    pairs = ((0, 1), (1, 2), (2, 0))
    u = np.any([along(0, a, b) for a, b in pairs], axis=0)
    v = np.any([along(1, a, b) for a, b in pairs], axis=0)
    return float(np.mean(u & v)) if len(tris) else 1.0


def emit(b: Any, aniso: Any, beta: float, rest: np.ndarray, starts: tuple[int, int, int], b_card: float) -> None:
    """Write the orthotropic rows for the cloth just added (`starts` = its first triangle, edge, particle) and scale
    its hinges to the orthotropic bending (after ``cloth_profile.rewrite_hinges``); `rest` its flat mesh vertices in
    the panel frame (x = warp, y = weft, z = 0)."""
    tri0, e0, p0 = starts
    rest = np.asarray(rest, np.float64)
    if rest.shape[1] == 3 and np.ptp(rest[:, 2]) > 1.0e-9:
        raise ValueError("orthotropic membrane: the cloth mesh must be flat in its x-y panel plane (warp along x)")
    uv = rest[:, :2]
    _register(b)
    tris = np.asarray(b.tri_indices[tri0:], np.int64) - p0
    dm = panel_dm_inv(uv, tris)
    k = (float(aniso.stretch_warp.value), float(aniso.stretch_weft.value), float(aniso.shear.value), float(beta))
    kv, dv = b.custom_attributes[f"{NS}:{K_ATTR}"].values, b.custom_attributes[f"{NS}:{DM_ATTR}"].values
    for i, row in enumerate(dm):
        kv[tri0 + i], dv[tri0 + i] = wp.vec4(*k), wp.vec4(*(float(x) for x in row))
    edges = np.asarray(b.edge_indices[e0:], np.int64)
    if len(edges):
        scale = edge_bending_scale(uv, np.where(edges >= 0, edges - p0, edges), float(aniso.bend_warp.value),
                                   float(aniso.bend_weft.value), b_card)
        b.edge_bending_properties[e0:] = [(float(ke) * s, float(kd) * s)
                                          for (ke, kd), s in zip(b.edge_bending_properties[e0:], scale)]
    log.debug("orthotropic membrane: %d triangles, k_u %.4g, k_v %.4g, k_s %.4g N/m", len(dm), *k[:3])
    share = aligned_share(uv, tris)
    if share < ALIGNED_MIN:
        log.warning("orthotropic membrane: only %.0f %% of the sheet's edges run along its warp, weft or a cell "
                    "diagonal; its stiff yarns lock on such a mesh and it drapes far too stiff (Petrak F1 on an "
                    "irregular disc: DC 0.500 against 0.252 on a warp-aligned grid). Use a grid sheet (add_cloth's "
                    "box) or a warp / weft grid mesh (docs/known_gaps/cloth-woven-drape-isotropic.md)", 100.0 * share)


def colour(b: Any) -> None:
    """``b.color()`` with the orthotropic triangles counted. Newton's colouring leaves out triangles whose Lame pair is 0
    (builder.color), which these have, so a woven sheet had one colour and its Gauss-Seidel sweep became a Jacobi one:
    a disc of F1 reached 87 % strain in one step. Their rows read (1, 1, ...) while it colours, then go back to 0."""
    attr = b.custom_attributes.get(f"{NS}:{K_ATTR}")
    rows = [t for t, k in (attr.values.items() if attr is not None else ()) if any(float(k[i]) != 0.0 for i in range(3))]
    saved = [b.tri_materials[t] for t in rows]
    for t, m in zip(rows, saved):
        b.tri_materials[t] = (1.0, 1.0, *tuple(m)[2:])
    try:
        b.color()
    finally:
        for t, m in zip(rows, saved):
            b.tri_materials[t] = m


# --- solve side (VBD colour callback) -------------------------------------------------------------------------------
@wp.func
def _stretch(w: wp.vec3, w_prev: wp.vec3, c: float, k: float, area: float, beta: float, inv_dt: float,
             f: wp.vec3, h: wp.mat33):
    """Add vertex coefficient c's share of 1/2 k A (|w| - 1)^2 (and its damping) to (f, h)."""
    length = wp.max(wp.length(w), _EPS)
    n = w / length
    C = length - 1.0
    g = n * c
    nn = wp.outer(n, n)
    eye = wp.identity(n=3, dtype=float)
    f = f - (k * area * C) * g
    h = h + (k * area * c * c) * (nn + (wp.max(C, 0.0) / length) * (eye - nn))
    if beta > 0.0:
        rate = (C - (wp.length(w_prev) - 1.0)) * inv_dt
        f = f - (beta * k * area * rate) * g
        h = h + (beta * k * area * inv_dt) * wp.outer(g, g)
    return f, h


@wp.func
def _shear(wu: wp.vec3, wv: wp.vec3, wu_prev: wp.vec3, wv_prev: wp.vec3, a: float, b: float, k: float, area: float,
           beta: float, inv_dt: float, f: wp.vec3, h: wp.mat33):
    """Add the vertex's share of 1/2 k A (w_u . w_v)^2 (and its damping) to (f, h)."""
    s = wp.dot(wu, wv)
    q = a * wv + b * wu
    eye = wp.identity(n=3, dtype=float)
    f = f - (k * area * s) * q
    h = h + (k * area) * (wp.outer(q, q) + wp.max(0.0, 2.0 * a * b * s) * eye)
    if beta > 0.0:
        rate = (s - wp.dot(wu_prev, wv_prev)) * inv_dt
        f = f - (beta * k * area * rate) * q
        h = h + (beta * k * area * inv_dt) * wp.outer(q, q)
    return f, h


@wp.kernel(enable_backward=False)
def accumulate_ortho_membrane(dt: float, particles: wp.array[wp.int32], pos: wp.array[wp.vec3],
                              pos_prev: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
                              tri_areas: wp.array[float], tri_k: wp.array[wp.vec4], tri_dm: wp.array[wp.vec4],
                              adjacency: MeshAdjacencyData, piece: wp.array[wp.int32], near_self: wp.array[wp.int32],
                              near_body: wp.array[wp.int32], forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    p = particles[wp.tid()]
    inv_dt = 1.0 / dt
    regularise = not _cloth_piece_is_free(p, piece, near_self, near_body)
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    for i in range(get_vertex_num_adjacent_faces(adjacency, p)):
        t, order = get_vertex_adjacent_face_id_order(adjacency, p, i)
        k = tri_k[t]
        if k[0] == 0.0 and k[1] == 0.0 and k[2] == 0.0:
            continue
        dm = tri_dm[t]
        i0 = tri_indices[t, 0]
        i1 = tri_indices[t, 1]
        i2 = tri_indices[t, 2]
        d1 = pos[i1] - pos[i0]
        d2 = pos[i2] - pos[i0]
        e1 = pos_prev[i1] - pos_prev[i0]
        e2 = pos_prev[i2] - pos_prev[i0]
        wu = d1 * dm[0] + d2 * dm[2]
        wv = d1 * dm[1] + d2 * dm[3]
        wu_prev = e1 * dm[0] + e2 * dm[2]
        wv_prev = e1 * dm[1] + e2 * dm[3]
        m0 = float(order == 0)
        m1 = float(order == 1)
        m2 = float(order == 2)
        a = dm[0] * (m1 - m0) + dm[2] * (m2 - m0)
        b = dm[1] * (m1 - m0) + dm[3] * (m2 - m0)
        area = tri_areas[t]
        f, h = _stretch(wu, wu_prev, a, k[0], area, k[3], inv_dt, f, h)
        f, h = _stretch(wv, wv_prev, b, k[1], area, k[3], inv_dt, f, h)
        f, h = _shear(wu, wv, wu_prev, wv_prev, a, b, k[2], area, k[3], inv_dt, f, h)
        if regularise:
            nrm = wp.cross(d1, d2)
            nrm = nrm / wp.max(wp.length(nrm), _EPS)
            h = h + (k[2] * (a * a + b * b) * area) * wp.outer(nrm, nrm)
    forces[p] = forces[p] + f
    hessians[p] = hessians[p] + h


class OrthoMembrane:
    """The colour callback (``install``): adds the orthotropic membrane for each colour's particles."""

    def __init__(self, solver: Any, k: wp.array, dm: wp.array) -> None:
        self.solver, self.k, self.dm, self.dt = solver, k, dm, None

    def wrap_step(self) -> None:
        """Record each step's dt (the callback signature carries none); a captured graph records it per launch."""
        step = self.solver.step

        def step_with_dt(state_in, state_out, control, contacts, dt, *args, **kwargs):
            self.dt = float(dt)
            return step(state_in, state_out, control, contacts, dt, *args, **kwargs)

        self.solver.step = step_with_dt

    def __call__(self, solver: Any, colour: int, state: Any, contacts: Any) -> None:
        if self.dt is None:
            raise RuntimeError("orthotropic membrane: the colour callback ran before a step set dt")
        m = solver.model
        group = m.particle_color_groups[colour]
        wp.launch(accumulate_ortho_membrane, dim=group.size,
                  inputs=[self.dt, group, state.particle_q, solver.particle_q_prev, m.tri_indices, m.tri_areas,
                          self.k, self.dm, solver.particle_adjacency, solver.particle_cloth_piece,
                          solver.piece_near_self_contact, solver.piece_near_body_contact],
                  outputs=[solver.particle_forces, solver.particle_hessians], device=solver.device)


def install(solver: Any, model: Any) -> Any:
    """Add the orthotropic membrane to a VBD entry's solver when its model has orthotropic triangles; returns it."""
    ns = getattr(model, NS, None)
    k, dm = getattr(ns, K_ATTR, None), getattr(ns, DM_ATTR, None)
    if k is None or dm is None or int(getattr(model, "tri_count", 0)) == 0:
        return solver
    if not np.any(k.numpy()[:, :3] != 0.0):
        return solver
    if len(model.particle_color_groups) < 2:
        raise RuntimeError("orthotropic membrane: the model has one colour group, so VBD would sweep these sheets as one "
                           "Jacobi group (it diverges); colour the builder with ortho_membrane.colour(b), not b.color()")
    if k.shape[0] != model.tri_count or not hasattr(solver, "add_colour_callback"):
        raise RuntimeError(f"orthotropic membrane: {k.shape[0]} rows for {model.tri_count} triangles, or a VBD solver "
                           f"without colour callbacks ({type(solver).__name__}); orthotropic cloth needs the full "
                           "model on SolverVBDAlloy")
    cb = OrthoMembrane(solver, k, dm)
    cb.wrap_step()
    solver.add_colour_callback(cb, external=False, when="after_accumulate")
    return solver


__all__ = ["ALIGNED_MIN", "DM_ATTR", "K_ATTR", "OrthoMembrane", "accumulate_ortho_membrane", "aligned_share", "colour",
           "edge_bending_scale", "emit", "install", "panel_dm_inv"]
