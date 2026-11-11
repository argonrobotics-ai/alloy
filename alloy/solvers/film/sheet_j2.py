# SPDX-License-Identifier: Apache-2.0
"""J2 sheet hinges: a metal sheet that bends past yield keeps a set (SCR-WP7-15 P3; the model is
``alloy/solvers/plastic/j2.py``).

Each hinge edge keeps SolverVBD's elastic energy ½·ke·L·(θ - θ̄)². A J2 sheet's interior hinges (the
``film:j2_yield`` attribute, σ_y' > 0) have ke = D/l_h, D = E' t³/12, with l_h = (A1 + A2)/|e| the mean of the two
triangles' heights (``emit.hinge_length``): a bend of curvature κ normal to the edge turns the hinge by θ = κ l_h on
any mesh, and the energy of an axis-aligned bend of a grid mesh is the plate's to 1e-6. (Paper's hinges keep
``topology.bending_ke``'s centroid distance, 2/3 of l_h.) Each also carries a fibre section: once per step, after the
strain projection, on the converged positions,

- κ = wrap(θ - θ0) / l_h, its curvature normal to the edge (θ0 the authored rest angle, so a curved sheet such as a
  can wall starts unstressed);
- ``j2.sheet_section`` returns the moment per unit width M(κ) of 21 Gauss-Lobatto layers in plane strain (E', σ_y',
  H' from ``j2.plane_strain``), each layer returning on its strain z κ;
- the rest angle moves to θ̄ = θ - M / ke, so the elastic hinge carries the section's moment, and unloading has the
  elastic slope (spring-back).

While the section is elastic M = D κ and θ̄ = θ0: nothing moves. The emitter writes E' with E' t³/12 = the hinge's D,
so the elastic range itself never shifts the rest (``SheetJ2`` checks it at build). Each hinge yields on its own
normal curvature: a one-dimensional section per hinge, not plate plasticity with a curvature tensor
(docs/known_gaps/plasticity-metal.md).

State per hinge: the layers' plastic strain ε_p and ε̄p [E, 21] and the largest layer ε̄p (``"<obj>.plastic_strain"``).
The solver lists them with the rest angles in its state arrays (snapshot, rollback, a coupler's restart pass) and
restores the authored state on reset.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.solvers.plastic import j2

from .hinge import hinge_angle, wrap_pi


@wp.kernel(enable_backward=False)
def j2_hinge(
    x: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    bending: wp.array2d[float],
    rest0: wp.array[float],
    sy: wp.array[float],
    hp: wp.array[float],
    ep: wp.array[float],
    t: wp.array[float],
    lh: wp.array[float],
    quad: j2.Quad,
    eps_p: wp.array2d[float],
    ebar: wp.array2d[float],
    ebar_max: wp.array[float],
    rest_angle: wp.array[float],
):
    """Per hinge with σ_y' > 0: the section's moment at κ = wrap(θ - θ0)/l_h sets θ̄ = θ - M/ke (module doc)."""
    e = wp.tid()
    if sy[e] <= 0.0:
        return
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    ke = bending[e, 0]
    if i0 < 0 or i1 < 0 or ke <= 0.0 or lh[e] <= 0.0:
        return
    th = hinge_angle(x[i0], x[i1], x[edge_indices[e, 2]], x[edge_indices[e, 3]])
    kappa = wrap_pi(th - rest0[e]) / lh[e]
    nm = j2.sheet_section(e, 0.0, kappa, t[e], ep[e], sy[e], hp[e], quad, eps_p, ebar)
    rest_angle[e] = wrap_pi(th - nm[1] / ke)
    top = float(0.0)
    for k in range(eps_p.shape[1]):
        top = wp.max(top, ebar[e, k])
    ebar_max[e] = top


@wp.kernel(enable_backward=False)
def reset_rows(edge_world: wp.array[wp.int32], world_mask: wp.array[wp.bool], eps_p: wp.array2d[float],
               ebar: wp.array2d[float], ebar_max: wp.array[float]):
    """Zero the J2 state of the hinges in the selected worlds."""
    e = wp.tid()
    w = edge_world[e]
    if w < 0:
        w = world_mask.shape[0] - 1        # Newton's reset mask: the last entry selects the global world (-1)
    if not world_mask[w]:
        return
    for k in range(eps_p.shape[1]):
        eps_p[e, k] = 0.0
        ebar[e, k] = 0.0
    ebar_max[e] = 0.0


class SheetJ2:
    """The J2 hinge state of a film model (built by SolverFilm only when some hinge has ``film:j2_yield`` > 0)."""

    def __init__(self, model, device) -> None:
        ns = model.film
        self.sy, self.hp, self.ep, self.t, self.lh = (ns.j2_yield, ns.j2_hardening, ns.j2_modulus, ns.j2_thickness,
                                                     ns.hinge_length)
        self.count = int((self.sy.numpy() > 0.0).sum())
        self.quad = j2.sheet_quad(device)
        E, K = int(model.edge_count), j2.fibres(self.quad)
        self.eps_p = wp.zeros((E, K), dtype=float, device=device)
        self.ebar = wp.zeros((E, K), dtype=float, device=device)
        self.ebar_max = wp.zeros(E, dtype=float, device=device)
        check_stiffness(model)

    def step(self, model, particle_q, rest0) -> None:
        wp.launch(j2_hinge, dim=model.edge_count,
                  inputs=[particle_q, model.edge_indices, model.edge_bending_properties, rest0, self.sy, self.hp,
                          self.ep, self.t, self.lh, self.quad, self.eps_p, self.ebar, self.ebar_max,
                          model.edge_rest_angle], device=model.device)

    def reset(self, model, edge_world=None, world_mask=None) -> None:
        if world_mask is None:
            for a in (self.eps_p, self.ebar, self.ebar_max):
                a.zero_()
            return
        wp.launch(reset_rows, dim=model.edge_count, inputs=[edge_world, world_mask, self.eps_p, self.ebar,
                                                            self.ebar_max], device=model.device)

    def state_arrays(self) -> dict[str, wp.array]:
        return {"film_j2_eps_p": self.eps_p, "film_j2_ebar": self.ebar, "film_j2_ebar_max": self.ebar_max}

    def vertex_strain(self, model) -> tuple[np.ndarray, np.ndarray]:
        """Host read, per particle: (the largest layer ε̄p of the J2 hinges along it, whether any J2 hinge runs
        along it). A hinge's line is its edge (v0, v1); its two wings' far vertices do not count."""
        on = self.sy.numpy() > 0.0
        ends = model.edge_indices.numpy()[on][:, 2:4].ravel()
        strain = np.zeros(model.particle_count)
        np.maximum.at(strain, ends, np.repeat(self.ebar_max.numpy()[on].astype(np.float64), 2))
        has = np.zeros(model.particle_count, dtype=bool)
        has[ends] = True
        return strain, has


def check_stiffness(model, rtol: float = 1.0e-4) -> None:
    """Every J2 hinge's ke must equal E' t³ / (12 l_h), else the elastic range moves its rest angle (module doc)."""
    ns = model.film
    on = ns.j2_yield.numpy() > 0.0
    ke = model.edge_bending_properties.numpy()[:, 0]
    want = ns.j2_modulus.numpy() * ns.j2_thickness.numpy() ** 3 / (12.0 * np.maximum(ns.hinge_length.numpy(), 1e-30))
    bad = on & (np.abs(ke - want) > rtol * np.abs(want))
    if bad.any():
        e = int(np.flatnonzero(bad)[0])
        raise ValueError(f"SolverFilm: J2 hinge {e} has ke {ke[e]:.6g}, but E' t^3/(12 l_h) = {want[e]:.6g}: the "
                         "section's elastic range would move its rest angle (write E' = E/(1 - nu^2) of the hinge's D)")
