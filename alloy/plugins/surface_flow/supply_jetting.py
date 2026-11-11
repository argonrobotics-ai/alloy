# SPDX-License-Identifier: Apache-2.0
"""The supply's solid jets on charts (``supply_jump``: the jump, one jet at a time, row-parallel), jets plunging into
pools (their wall shear on the flooded cells under them), and the scratch the deposits share (a mixin of ``Supply``: it
uses the supply's tables, hits, network and film).

Submerged jets (WP15's SCR with WP9: soil under a pool never eroded). A solid jet (not a spray, not a drip) that hits
a pool's surface carries on straight through the water (no gravity: it is water in water) from its hit along its
velocity there; the first chart cell it meets within ``SUB_REACH`` is under the pool, z down the jet from the surface.
Past its core the free jet's centreline speed falls as u_c = u0·min(1, 6.2·d0/z) (d0 the jet's diameter at the
surface, sqrt(4q/(π·u0)); Rajaratnam, *Turbulent Jets*, 1976), and the wall shear where it strikes peaks at
τ0 = 0.16·ρ·u0²·(d0/z)² past the core (Beltaos & Rajaratnam 1974, impinging round jets), i.e. ``TAU_K``·ρ·u_c² with
TAU_K = 0.16/6.2² = 0.00416, the jet erosion test's C_f with its core held inside 6.2·d0 (Hanson & Cook 2004). Over
the cells round the strike, flooded ones only, τ = τ0·exp(-(r/b)²) with b = d0/2 + 0.1·z (the jet's spread), the
largest over the jets, into the film's ``pool_jet_tau`` [W, C] Pa (registered by the film: WP9's soil law reads it
as wall shear on flooded cells), inside the box written this exchange (cleared by the next exchange's clear rule, as
``jet_tau``). ESTIMATEs: the profile's shape and the strike's angle. Beltaos & Rajaratnam's measured shear is zero on
the axis and peaks at r ≈ 0.14·z; here, as the jet erosion test applies its peak at the scour's centre, the peak sits
on the axis, so the patch it cleans is a disc where their profile gives a ring (7 mm out at 50 mm deep, 42 mm at
300 mm); the strike is taken square-on (an oblique jet's skewed footprint is not modelled). One thread per world
walks each strike's window: fine on a CPU, slow on a GPU for many deep jets (a row-parallel gather is the fix).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy._vendor import newton

from alloy.plugins.surface_flow import supply_deposit as D
from alloy.plugins.surface_flow import supply_jump as J
from alloy.plugins.surface_flow import supply_spray as S
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.geometry import cell_window, chart_xform, ray_hit
from alloy.plugins.surface_flow.supply_deposit import cell_of, ring_step, row_span, surf_dist, union_box
from alloy.plugins.surface_flow.supply_kernels import JET, POOL, NozzleSpec, weight_sum
from alloy.plugins.surface_flow.supply_spec import STAGE

SUB_REACH = wp.constant(1.0)          # m: how far a plunging jet is followed under the surface
TAU_K = wp.constant(0.16 / 6.2 ** 2)  # τ0 = TAU_K·ρ·u_c²: the peak wall shear past the core (Hanson & Cook's C_f)
PI = wp.constant(3.14159265358979)


@wp.kernel
def submerged_hits(body_q: wp.array[wp.transform], charts: wp.array[Chart], cbody: wp.array2d[wp.int32],
                   clocal: wp.array2d[wp.transform], prof: wp.array[wp.vec3], nozzles: wp.array[NozzleSpec],
                   ray_noz: wp.array[wp.int32], discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32],
                   hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3], sub_cell: wp.array2d[wp.int32],
                   sub_k: wp.array2d[wp.int32], sub_p: wp.array2d[wp.vec3], sub_z: wp.array2d[wp.float32]):
    """One ray: a solid jet into a pool carries on straight from its hit (world) along its velocity; the first chart
    cell within ``SUB_REACH`` (-1: none), its chart, the strike (chart frame) and the depth z down the jet."""
    w, r = wp.tid()
    sub_cell[w, r] = -1
    noz = nozzles[ray_noz[r]]
    if hit_kind[w, r] != POOL or noz.pattern != JET or noz.drip != 0 or discharge[w, noz.edge] >= wp.float64(0.0):
        return
    v = hit_w[w, r]
    if wp.length(v) <= 0.0:
        return
    u = wp.normalize(v)
    best = float(SUB_REACH)
    for k in range(charts.shape[0]):
        h = ray_hit(charts[k], prof, chart_xform(body_q, cbody[w, k], clocal[w, k]), hit_p[w, r], u)
        if h.cell >= 0 and h.t <= best:
            best = h.t
            sub_cell[w, r] = h.cell
            sub_k[w, r] = k
            sub_p[w, r] = h.point
            sub_z[w, r] = h.t


@wp.kernel
def pool_shear(charts: wp.array[Chart], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3],
               nozzles: wp.array[NozzleSpec], ray_noz: wp.array[wp.int32], ray_wt: wp.array[wp.float32],
               discharge: wp.array2d[wp.float64], hit_w: wp.array2d[wp.vec3], sub_cell: wp.array2d[wp.int32],
               sub_k: wp.array2d[wp.int32], sub_p: wp.array2d[wp.vec3], sub_z: wp.array2d[wp.float32],
               flooded: wp.array2d[wp.int32], pool_tau: wp.array2d[wp.float32], written: wp.array2d[wp.vec4i]):
    """Per world, the plunging jets in ray order: their wall shear on the flooded cells round each strike (module
    docstring), the largest over the jets, and the cells' window into the written box."""
    w = wp.tid()
    for r in range(ray_noz.shape[0]):
        c0 = sub_cell[w, r]
        if c0 < 0:
            continue
        noz = nozzles[ray_noz[r]]
        q = float(-discharge[w, noz.edge]) * noz.efficiency * float(ray_wt[r]) / float(weight_sum(noz, ray_wt))
        u0 = wp.length(hit_w[w, r])
        d0 = wp.sqrt(4.0 * q / (PI * u0))
        z = float(sub_z[w, r])
        uc = u0 * wp.min(1.0, 6.2 * d0 / wp.max(z, 1.0e-9))
        tau0 = TAU_K * noz.rho * uc * uc
        b = 0.5 * d0 + 0.1 * z
        k = sub_k[w, r]
        chart = charts[k]
        loc = c0 - chart.cell_start
        i_hit = loc // chart.ny
        span = row_span(chart, prof, i_hit)
        win = cell_window(chart, prof, i_hit, loc % chart.ny, 2.0 * b + ring_step(chart))
        win = wp.vec4i(wp.max(win[0], span[0]), wp.min(win[1], span[1]), win[2], win[3])
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_of(chart, i, j)
                if flooded[w, c] < 0:
                    continue
                d = surf_dist(chart, prof, cpos, c, i, j, sub_p[w, r], i_hit)
                t = tau0 * wp.exp(-(d / b) * (d / b))
                if t > pool_tau[w, c]:
                    pool_tau[w, c] = t
        written[w, k] = union_box(chart, written[w, k], win)


class Jetting:
    """``Supply``'s jets on charts and the deposits' scratch."""

    def _setup_jets(self, ctx: Any) -> None:
        """Scratch for the parallel deposits: the jump (``supply_jump``, one jet at a time) and the sprays
        (``supply_spray``); rows up to the longest chart."""
        W = ctx.world_count
        self.jet_rays = [r for r, n in enumerate(self.ray_noz.numpy()) if self.nozzles and
                         self.nozzles[int(n)].pattern == "jet" and self.nozzles[int(n)].drip_of != STAGE]
        self.rows = max(nx for nx, _ in self.charts.shapes)
        self.jet = wp.zeros(W, dtype=J.Jet, device=ctx.device)
        self.part_h = ctx.zeros((W, self.rows, D.N_BANDS), wp.float64)
        self.part_a = ctx.zeros((W, self.rows, D.N_BANDS), wp.float64)
        self.part_far = ctx.zeros((W, self.rows), wp.float32)
        self.part_w = ctx.zeros((W, self.rows), wp.float64)
        self.part_swept = ctx.zeros((W, self.rows), wp.float64)
        self.part_wh = ctx.zeros((W, self.rows), wp.float64)
        self.part_sm = ctx.zeros((W, self.rows), wp.vec4d)
        self.part_fa = ctx.zeros((W, self.rows), wp.float64)
        sp = getattr(self.film, "sp", None)                  # the film's species (none on a film without them)
        self.n_sp = int(sp.count) if sp is not None else 0
        dummy = ctx.zeros((1, 1, 1), wp.float64)
        self.sp_mass, self.sp_src = (sp.mass, sp.src) if self.n_sp else (dummy, dummy)
        self.swept_sp = self.film.add_state("supply_swept_species", ctx.zeros((W, max(self.n_sp, 1)), wp.float64))
        self.sprays = any(n.pattern != "jet" or n.drip_of == STAGE for n in self.nozzles)   # stage drips: whole
        self.spray = wp.zeros((W, max(self.ray_count, 1)), dtype=S.Spray, device=ctx.device)
        self.box = ctx.zeros((W, len(self.charts.shapes)), wp.vec4i)
        self._setup_shadows(ctx)                              # the spray stencil's tables (supply_shadow)
        # the box of cells written per world and chart, cleared at the next before_film (the interface's clear rule)
        self.written = self.film.add_state("supply_written", ctx.full((W, len(self.charts.shapes)),
                                                                      wp.vec4i(1 << 30, -1, 0, -1), wp.vec4i))
        self._setup_submerged(ctx)

    def _setup_submerged(self, ctx: Any) -> None:
        """The plunging jets' strikes (scratch) and the film's ``pool_jet_tau`` (off when the film has none, or no
        pool or solid jet could plunge)."""
        W, R = ctx.world_count, max(self.ray_count, 1)
        tau = getattr(self.film, "pool_jet_tau", None)
        self.has_pool_tau = int(tau is not None)
        self.pool_tau = tau if tau is not None else ctx.zeros((1, 1), wp.float32)
        self.submerged = bool(tau is not None and self.pools and self.jet_rays)
        self.sub_cell, self.sub_k = ctx.full((W, R), -1, wp.int32), ctx.zeros((W, R), wp.int32)
        self.sub_p, self.sub_z = ctx.zeros((W, R), wp.vec3), ctx.zeros((W, R), wp.float32)

    def _submerged(self, state: newton.State) -> None:
        """In ``before_film``, after the clear and the flood: the plunging jets' strikes and their wall shear."""
        if not self.submerged:
            return
        W, dev, c, h = self.ctx.world_count, self.ctx.device, self.charts, self.hit
        wp.launch(submerged_hits, dim=(W, self.ray_count),
                  inputs=[state.body_q, c.charts, c.cbody, c.clocal, c.prof, self.nozzle_spec, self.ray_noz,
                          self.nozzle_q, h["kind"], h["p"], h["w"]],
                  outputs=[self.sub_cell, self.sub_k, self.sub_p, self.sub_z], device=dev)
        wp.launch(pool_shear, dim=W,
                  inputs=[c.charts, c.prof, c.cpos, self.nozzle_spec, self.ray_noz, self.ray_wt, self.nozzle_q,
                          h["w"], self.sub_cell, self.sub_k, self.sub_p, self.sub_z, self.film.flooded],
                  outputs=[self.pool_tau, self.written], device=dev)

    def _jets(self, state: newton.State, dt: float) -> None:
        """Each solid jet on a chart, in ray order: the eight row-parallel launches of ``supply_jump``."""
        W, dev, f, film, c, h = self.ctx.world_count, self.ctx.device, self.fluid, self.film, self.charts, self.hit
        rows = (W, self.rows)
        common = [c.charts, c.prof, c.cpos]
        for r in self.jet_rays:
            wp.launch(J.jet_setup, dim=W,
                      inputs=[r, state.body_q, c.charts, c.prof, c.cbody, c.clocal, c.cnrm, self.nozzle_spec,
                              self.ray_noz, self.ray_wt, self.nozzle_q, h["kind"], h["idx"], h["cell"], h["p"], h["w"],
                              wp.vec3(*self.gravity)], outputs=[self.jet, self.written], device=dev)
            wp.launch(J.jet_bands, dim=rows, inputs=[*common, c.area, film.h, film.jet_cover, self.jet],
                      outputs=[self.part_h, self.part_a], device=dev)
            wp.launch(J.jet_radius, dim=W, inputs=[self.jet, self.part_h, self.part_a], device=dev)
            wp.launch(J.jet_ring, dim=rows, inputs=[*common, c.area, self.jet], outputs=[self.part_far, self.part_w],
                      device=dev)
            wp.launch(J.jet_fallback, dim=W, inputs=[self.jet, self.part_far, self.part_w], device=dev)
            wp.launch(J.jet_sweep, dim=rows,
                      inputs=[*common, c.area, self.nozzle_spec, self.ray_noz, r, self.jet],
                      outputs=[film.h, film.jet_cover, film.jet_tau, film.cwet, self.n_sp, self.sp_mass,
                               self.part_swept, self.part_w, self.part_wh, self.part_sm, self.part_fa],
                      device=dev)
            wp.launch(J.jet_finish, dim=W,
                      inputs=[r, wp.float64(dt), self.nozzle_spec, self.ray_noz, c.area, self.jet, self.part_swept,
                              self.part_w, self.part_wh, self.part_sm, self.part_fa, self.n_sp],
                      outputs=[film.src_acc, self.sp_src, film.impact_p, film.impact_share, film.cwet, self.ledger,
                               self.swept_sp, h["on"], self.ns, self.nconc, self.nnode], device=dev)
            wp.launch(J.jet_enter, dim=rows,
                      inputs=[*common, c.cnrm, c.area, self.nozzle_spec, self.ray_noz, r, self.jet, self.n_sp],
                      outputs=[film.src_acc, self.sp_src, film.src_mom, film.cwet, film.impact_p,
                               film.impact_share], device=dev)


__all__ = ["Jetting", "PI", "SUB_REACH", "TAU_K", "pool_shear", "submerged_hits"]
