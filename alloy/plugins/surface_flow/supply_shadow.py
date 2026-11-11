# SPDX-License-Identifier: Apache-2.0
"""Occlusion per footprint cell for sprays (SOFT-3's ask A, the spray stencil): a spray ray's footprint spans 1.5 ray
spacings, so near a stencil's edge it would paint cells the stencil hides from the nozzle, and a ray whose centre the
stencil hides would lose the part of its footprint the nozzle sees. A spray ray therefore passes occluders on its way
to a chart (``supply_kernels.cast_rays``: it is caught by the first one only when no chart lies behind), and each
exchange, in ``apply`` right after the cast (steps 1-3 and 5: the hidden share's momentum is this exchange's):

1. ``shadow_cull``: per world, chart and occluder slot, whether the occluder can lie between an active spray nozzle and
   the chart: its bounding sphere (an infinite plane: its half-space) against the box bounding the nozzles' exits and
   the chart's cells (their local bounds, posed); the chart's own shape never shades it (``keep [W, K, S]``);
2. ``spray_shadow``: every cell of the sprays' boxes tests the straight segment from each spray nozzle's exit to its
   centre against the kept occluders; ``shade [W, C]`` holds a bit per spray nozzle (``sbit``: the first 32 spray
   nozzles, jets do not count) that cannot see the cell;
3. ``spray_visible``: each ray's footprint keeps its visible cells, its flow re-weighted over them; its hidden share
   ``q_shadow`` is caught by the occluder nearest its own nozzle on the segment to its most-weighted hidden cell (a ray
   hidden whole lands nothing);
4. ``spray_caught`` (in ``before_film``): per world, in ray order, the hidden shares into the ``caught`` ledger,
   ``caught_by`` and the species ledger (from ``sent`` to ``caught``);
5. ``shadow_forces`` (after ``jet_forces``): each hidden share's momentum ρ·q_shadow·w moves from the chart's body to
   its occluder's, all of it (as a jet on a shape), at ``bp``: where the segments to the ray's hidden cells meet that
   occluder, weighted as the footprint weights them (step 3). w is the ray's velocity at its chart hit, absolute
   (ESTIMATE at the occluder: they differ by gravity over the gap between them, a few cm for SC9's scrap on the
   floor; and, as ``jet_forces`` does, not relative to the occluder's own motion).

The gather then skips a ray's hidden cells: the paint stops at the stencil's shadow. A cell decides by its centre,
within ``SHADOW_TOL`` of the segment's end (the chart's own face). Cells are written only inside this exchange's boxes,
where every read happens. Limits: a cell facing away from the nozzle is not culled (a revolution chart's far side is
shaded only through a shape).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.funcs import add_wrench, com_world, wrench_at
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.geometry import chart_xform
from alloy.plugins.surface_flow.supply_deposit import cell_of, surf_dist, union_box
from alloy.plugins.surface_flow.supply_kernels import (CAUGHT, CHART, JET, NozzleSpec, body_xform, hit_world,
                                                       pool_frame, shape_t)
from alloy.plugins.surface_flow.supply_species import SP_CAUGHT, SP_SENT, NetSpecies, refresh
from alloy.plugins.surface_flow import supply_spray as SPR
from alloy.plugins.surface_flow.supply_spray import Spray

log = logging.getLogger("alloy.plugins.surface_flow")

SHADOW_TOL = wp.constant(5.0e-5)   # m: an occluder this close to the cell (its own chart's face) does not hide it
MAX_SHADE = 32                     # spray nozzles with a shade bit
PLANE = wp.constant(int(newton.GeoType.PLANE))


@wp.func
def shaded(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], shape_transform: wp.array[wp.transform],
           shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3], shape_ptr: wp.array[wp.uint64],
           occ: wp.array2d[wp.int32], keep: wp.array3d[wp.int32], w: int, k: int, p0: wp.vec3, x: wp.vec3) -> int:
    """1 when an occluder kept for chart k lies on the segment p0 -> x short of x."""
    seg = x - p0
    length = wp.length(seg)
    if length <= SHADOW_TOL:
        return 0
    d = seg / length
    for si in range(occ.shape[1]):
        s = occ[w, si]
        if s < 0:
            break
        if keep[w, k, si] != 0:
            t = shape_t(body_q, shape_body, shape_transform, shape_type, shape_scale, shape_ptr, s, p0, d, length)
            if t >= 0.0 and t < length - SHADOW_TOL:
                return 1
    return 0


@wp.func
def nearest_blocker(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                    shape_transform: wp.array[wp.transform], shape_type: wp.array[wp.int32],
                    shape_scale: wp.array[wp.vec3], shape_ptr: wp.array[wp.uint64], occ: wp.array2d[wp.int32],
                    keep: wp.array3d[wp.int32], w: int, k: int, p0: wp.vec3, x: wp.vec3) -> int:
    """The slot of the kept occluder nearest p0 on the segment p0 -> x short of x, or -1."""
    seg = x - p0
    length = wp.length(seg)
    best = int(-1)
    if length <= SHADOW_TOL:
        return best
    d = seg / length
    best_t = length - SHADOW_TOL
    for si in range(occ.shape[1]):
        s = occ[w, si]
        if s < 0:
            break
        if keep[w, k, si] != 0:
            t = shape_t(body_q, shape_body, shape_transform, shape_type, shape_scale, shape_ptr, s, p0, d, length)
            if t >= 0.0 and t < best_t:
                best_t = t
                best = si
    return best


@wp.func
def spray_exit(body_q: wp.array[wp.transform], nozzles: wp.array[NozzleSpec], nbody: wp.array2d[wp.int32],
               npos: wp.array2d[wp.vec3], discharge: wp.array2d[wp.float64], w: int, n: int) -> wp.vec4:
    """A spray nozzle's world exit point and 1 while it sprays (0: a jet, or no flow)."""
    noz = nozzles[n]
    p = wp.transform_point(pool_frame(body_q, nbody[w, n]), npos[w, n])
    on = float(0.0)
    if noz.pattern != JET and discharge[w, noz.edge] < wp.float64(0.0):
        on = 1.0
    return wp.vec4(p[0], p[1], p[2], on)


@wp.kernel
def shadow_cull(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                shape_transform: wp.array[wp.transform], shape_type: wp.array[wp.int32], aabb_lo: wp.array[wp.vec3],
                aabb_hi: wp.array[wp.vec3], occ: wp.array2d[wp.int32], cbody: wp.array2d[wp.int32],
                clocal: wp.array2d[wp.transform], clo: wp.array[wp.vec3], chi: wp.array[wp.vec3],
                cshape: wp.array2d[wp.int32], nozzles: wp.array[NozzleSpec], nbody: wp.array2d[wp.int32],
                npos: wp.array2d[wp.vec3], discharge: wp.array2d[wp.float64], keep: wp.array3d[wp.int32]):
    """Occluder slot si of world w can shade chart k (module docstring, step 1)."""
    w, k, si = wp.tid()
    s = occ[w, si]
    if s < 0 or s == cshape[w, k]:
        keep[w, k, si] = 0
        return
    X = chart_xform(body_q, cbody[w, k], clocal[w, k])
    lo = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    hi = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    for c in range(8):
        corner = clo[k]
        if (c & 1) != 0:
            corner = wp.vec3(chi[k][0], corner[1], corner[2])
        if (c & 2) != 0:
            corner = wp.vec3(corner[0], chi[k][1], corner[2])
        if (c & 4) != 0:
            corner = wp.vec3(corner[0], corner[1], chi[k][2])
        p = wp.transform_point(X, corner)
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)
    for n in range(nozzles.shape[0]):
        e = spray_exit(body_q, nozzles, nbody, npos, discharge, w, n)
        if e[3] > 0.0:
            p = wp.vec3(e[0], e[1], e[2])
            lo = wp.min(lo, p)
            hi = wp.max(hi, p)
    Xs = body_xform(body_q, shape_body[s], shape_transform[s])
    kept = int(1)
    if shape_type[s] == PLANE:                                    # culled when the hull lies wholly above it
        nrm = wp.transform_vector(Xs, wp.vec3(0.0, 0.0, 1.0))
        o = wp.transform_get_translation(Xs)
        low = float(1.0e30)
        for c in range(8):
            corner = lo
            if (c & 1) != 0:
                corner = wp.vec3(hi[0], corner[1], corner[2])
            if (c & 2) != 0:
                corner = wp.vec3(corner[0], hi[1], corner[2])
            if (c & 4) != 0:
                corner = wp.vec3(corner[0], corner[1], hi[2])
            low = wp.min(low, wp.dot(nrm, corner - o))
        if low > SHADOW_TOL:
            kept = 0
    else:                                                         # its bounding sphere misses the hull's box
        a = aabb_lo[s]
        b = aabb_hi[s]
        ctr = wp.transform_point(Xs, 0.5 * (a + b))
        q = wp.min(wp.max(ctr, lo), hi)
        if wp.length(ctr - q) > 0.5 * wp.length(b - a) + SHADOW_TOL:
            kept = 0
    keep[w, k, si] = kept


@wp.kernel
def spray_shadow(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                 shape_transform: wp.array[wp.transform], shape_type: wp.array[wp.int32],
                 shape_scale: wp.array[wp.vec3], shape_ptr: wp.array[wp.uint64], occ: wp.array2d[wp.int32],
                 keep: wp.array3d[wp.int32], charts: wp.array[Chart], cbody: wp.array2d[wp.int32],
                 clocal: wp.array2d[wp.transform], cpos: wp.array[wp.vec3], nozzles: wp.array[NozzleSpec],
                 nbody: wp.array2d[wp.int32], npos: wp.array2d[wp.vec3], discharge: wp.array2d[wp.float64],
                 sbit: wp.array[wp.int32], box: wp.array2d[wp.vec4i], shade: wp.array2d[wp.int32]):
    """Row ``ii`` of chart k's box: which spray nozzles each cell cannot see (module docstring, step 2)."""
    w, k, ii = wp.tid()
    b = box[w, k]
    i = b[0] + ii
    if i > b[1]:
        return
    chart = charts[k]
    X = chart_xform(body_q, cbody[w, k], clocal[w, k])
    for j in range(b[2], b[3] + 1):
        c = cell_of(chart, i, j)
        x = wp.transform_point(X, cpos[c])
        mask = int(0)
        for n in range(nozzles.shape[0]):
            bit = sbit[n]
            if bit < 0:
                continue
            e = spray_exit(body_q, nozzles, nbody, npos, discharge, w, n)
            if e[3] > 0.0 and shaded(body_q, shape_body, shape_transform, shape_type, shape_scale, shape_ptr, occ,
                                     keep, w, k, wp.vec3(e[0], e[1], e[2]), x) == 1:
                mask = mask | (1 << bit)
        shade[w, c] = mask


@wp.func
def is_hidden(shade: wp.array2d[wp.int32], sbit: wp.array[wp.int32], w: int, c: int, n: int) -> int:
    """1 when cell c is hidden from spray nozzle n."""
    bit = sbit[n]
    if bit < 0:
        return 0
    return (shade[w, c] >> bit) & 1


@wp.kernel
def spray_visible(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                  shape_transform: wp.array[wp.transform], shape_type: wp.array[wp.int32],
                  shape_scale: wp.array[wp.vec3], shape_ptr: wp.array[wp.uint64], occ: wp.array2d[wp.int32],
                  keep: wp.array3d[wp.int32], charts: wp.array[Chart], cbody: wp.array2d[wp.int32],
                  clocal: wp.array2d[wp.transform], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3],
                  area: wp.array[wp.float64], nbody: wp.array2d[wp.int32], npos: wp.array2d[wp.vec3],
                  sbit: wp.array[wp.int32], shade: wp.array2d[wp.int32], spray: wp.array2d[Spray]):
    """One ray: its footprint over the visible cells only, its hidden share, the occluder nearest its nozzle that
    hides its most-weighted hidden cell and where the hidden cells' segments meet it (module docstring, step 3)."""
    w, r = wp.tid()
    s = spray[w, r]
    if s.on == 0:
        return
    chart = charts[s.chart]
    sw = wp.float64(0.0)
    bw = wp.float64(0.0)
    best = wp.float64(0.0)
    bc = int(-1)
    if s.on == 2:                                         # all at its hit cell
        if is_hidden(shade, sbit, w, s.cell, s.noz) == 1:
            bw = wp.float64(1.0)
            bc = s.cell
    else:
        for i in range(s.win[0], s.win[1] + 1):
            for j in range(s.win[2], s.win[3] + 1):
                c = cell_of(chart, i, j)
                d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
                wt = wp.float64(wp.max(1.0 - d / s.radius, 0.0)) * area[c]
                if wt > wp.float64(0.0):
                    if is_hidden(shade, sbit, w, c, s.noz) == 1:
                        bw = bw + wt
                        if wt > best:
                            best = wt
                            bc = c
                    else:
                        sw = sw + wt
    if bw <= wp.float64(0.0):
        return
    p0 = wp.transform_point(pool_frame(body_q, nbody[w, s.noz]), npos[w, s.noz])
    x = wp.transform_point(chart_xform(body_q, cbody[w, s.chart], clocal[w, s.chart]), cpos[bc])
    s.blocker = nearest_blocker(body_q, shape_body, shape_transform, shape_type, shape_scale, shape_ptr, occ, keep, w,
                                s.chart, p0, x)
    if s.blocker < 0:                                     # the shade saw one: keep the paint rather than lose it
        return
    X = chart_xform(body_q, cbody[w, s.chart], clocal[w, s.chart])
    s.bp = x                                              # the blocker's point, weighted over the hidden cells
    acc = wp.vec3()
    wsum = float(0.0)
    for i in range(s.win[0], s.win[1] + 1):
        for j in range(s.win[2], s.win[3] + 1):
            c = cell_of(chart, i, j)
            cw = float(1.0)
            if s.on == 2:
                if c != s.cell:
                    cw = 0.0
            else:
                cw = wp.max(1.0 - float(surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)) / s.radius, 0.0) * \
                    float(area[c])
            if cw > 0.0 and is_hidden(shade, sbit, w, c, s.noz) == 1:
                seg = wp.transform_point(X, cpos[c]) - p0
                u = seg / wp.length(seg)
                t = shape_t(body_q, shape_body, shape_transform, shape_type, shape_scale, shape_ptr,
                            occ[w, s.blocker], p0, u, wp.length(seg))
                if t >= 0.0:
                    acc = acc + cw * (p0 + u * t)
                    wsum = wsum + cw
    if wsum > 0.0:
        s.bp = acc / wsum
    if sw <= wp.float64(0.0) or s.on == 2:                # hidden whole: all of it, exactly
        s.q_shadow = s.q
        s.q = wp.float64(0.0)
        s.on = 0
    else:
        s.q_shadow = s.q * bw / (sw + bw)
        s.q = s.q - s.q_shadow
        s.sum_w = sw                                      # in-chart only: stage drips (q_off) have no shade bit
    spray[w, r] = s


@wp.kernel
def spray_caught(dt: wp.float64, spray: wp.array2d[Spray], ns: NetSpecies, ledger: wp.array2d[wp.float64],
                 caught_by: wp.array2d[wp.float64]):
    """Per world, in ray order: the rays' hidden shares caught by their occluders (module docstring, step 4)."""
    w = wp.tid()
    for r in range(spray.shape[1]):
        s = spray[w, r]
        if s.q_shadow <= wp.float64(0.0):
            continue
        v = s.q_shadow * dt
        ledger[w, CAUGHT] = ledger[w, CAUGHT] + v
        caught_by[w, s.blocker] = caught_by[w, s.blocker] + v
        for k in range(ns.n_film):
            m = v * s.conc[k]
            ns.led[w, SP_SENT, k] = ns.led[w, SP_SENT, k] - m
            ns.led[w, SP_CAUGHT, k] = ns.led[w, SP_CAUGHT, k] + m


@wp.kernel
def shadow_forces(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], shape_body: wp.array[wp.int32],
                  cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], cnrm: wp.array[wp.vec3],
                  occ: wp.array2d[wp.int32], hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3],
                  hit_on: wp.array2d[wp.float32], spray: wp.array2d[Spray], body_f: wp.array[wp.spatial_vector]):
    """Per world, in ray order: each hidden share's momentum off the chart's body (as ``jet_forces`` put it there)
    and onto its occluder's at ``bp`` (module docstring, step 5)."""
    w = wp.tid()
    for r in range(spray.shape[1]):
        s = spray[w, r]
        if s.q_shadow <= wp.float64(0.0):
            continue
        m = float(s.rho) * float(s.q_shadow) * hit_w[w, r]
        cb = cbody[w, s.chart]
        if cb >= 0:
            X = chart_xform(body_q, cb, clocal[w, s.chart])
            nrm = wp.transform_vector(X, cnrm[s.cell])
            mn = wp.dot(m, nrm) * nrm
            pw = hit_world(body_q, cbody, clocal, w, CHART, s.chart, hit_p[w, r])
            add_wrench(body_f, cb, wrench_at(-(mn + hit_on[w, r] * (m - mn)), pw, com_world(body_q, body_com, cb)))
        b = shape_body[occ[w, s.blocker]]
        if b >= 0:
            add_wrench(body_f, b, wrench_at(m, s.bp, com_world(body_q, body_com, b)))


@wp.kernel
def boxes_written(charts: wp.array[Chart], box: wp.array2d[wp.vec4i], written: wp.array2d[wp.vec4i]):
    """In ``before_film``, after the clear: apply's boxes into the box of cells the supply writes this exchange."""
    w, k = wp.tid()
    written[w, k] = union_box(charts[k], written[w, k], box[w, k])


def shade_bits(nozzles: tuple[Any, ...]) -> np.ndarray:
    """Each nozzle's shade bit: spray nozzles in order up to ``MAX_SHADE`` (a warning past it), jets -1."""
    bits, n = [], 0
    for noz in nozzles:
        if noz.pattern == "jet" or n >= MAX_SHADE:
            bits.append(-1)
            n += int(noz.pattern != "jet")
            continue
        bits.append(n)
        n += 1
    if n > MAX_SHADE:
        log.warning("surface_flow supply: %d spray nozzles; occlusion per footprint cell covers the first %d only",
                    n, MAX_SHADE)
    return np.asarray(bits or [-1], np.int32)


class Spraying:
    """``Supply``'s spray launches (a mixin: it uses the supply's tables, hits and arrays)."""

    def _setup_shadows(self, ctx: Any) -> None:
        """The shade (read by the gather whenever there are sprays), and with occluders the culling tables."""
        W, c = ctx.world_count, self.charts
        K, S = len(c.shapes), self.occ.shape[1]
        self.shadows = any(n.pattern != "jet" for n in self.nozzles) and bool((self.occ.numpy() >= 0).any())
        self.shade = ctx.zeros((W, c.cell_count if self.sprays else 1), wp.int32)
        self.sbit = ctx.array(shade_bits(self.nozzles), wp.int32)
        self.keep = ctx.zeros((W, K, S) if self.shadows else (1, 1, 1), wp.int32)
        cpos = c.cpos.numpy()
        bounds = [(cpos[s:s + nx * ny].min(0), cpos[s:s + nx * ny].max(0)) for s, (nx, ny) in zip(c.starts, c.shapes)]
        self.clo = ctx.array(np.asarray([b[0] for b in bounds], np.float32), wp.vec3)
        self.chi = ctx.array(np.asarray([b[1] for b in bounds], np.float32), wp.vec3)

    def _spray_shadows(self, state: Any, dt: float) -> None:
        """In ``apply`` after the cast: each spray ray's footprint, the boxes, and with occluders the hidden shares
        (module docstring, steps 1-3), so ``shadow_forces`` moves this exchange's."""
        W, dev, c, h, m = self.ctx.world_count, self.ctx.device, self.charts, self.hit, self.ctx.model
        wp.launch(SPR.spray_rays, dim=(W, self.ray_count),
                  inputs=[wp.float64(dt), state.body_q, c.charts, c.prof, c.cbody, c.clocal, c.cpos, c.cnrm, c.area, self.nozzle_spec,
                          self.ray_noz, self.ray_wt, self.ray_fp, self.nozzle_q, h["kind"], h["idx"], h["cell"],
                          h["p"], h["w"], h["s"], refresh(self.ns, self.fluid), self.nconc, self.nnode],
                  outputs=[self.spray], device=dev)
        wp.launch(SPR.spray_boxes, dim=self.box.shape, inputs=[c.charts, self.spray], outputs=[self.box], device=dev)
        if self.shadows:                                  # footprint cells hidden from their nozzles (SOFT-3 A)
            shapes = self._model_shapes()
            noz = [self.nozzle_spec, self.nbody, self.npos, self.nozzle_q]
            wp.launch(shadow_cull, dim=self.keep.shape,
                      inputs=[state.body_q, *shapes[:3], m.shape_collision_aabb_lower, m.shape_collision_aabb_upper,
                              self.occ, c.cbody, c.clocal, self.clo, self.chi, c.cshape, *noz],
                      outputs=[self.keep], device=dev)
            wp.launch(spray_shadow, dim=(*self.box.shape, self.rows),
                      inputs=[state.body_q, *shapes, self.occ, self.keep, c.charts, c.cbody, c.clocal, c.cpos, *noz,
                              self.sbit, self.box], outputs=[self.shade], device=dev)
            wp.launch(spray_visible, dim=(W, self.ray_count),
                      inputs=[state.body_q, *shapes, self.occ, self.keep, c.charts, c.cbody, c.clocal, c.prof, c.cpos,
                              c.area, self.nbody, self.npos, self.sbit, self.shade], outputs=[self.spray], device=dev)

    def _shadow_forces(self, state: Any, forces: Any) -> None:
        """In ``apply`` after ``jet_forces``: the hidden shares' momentum onto their occluders (step 5)."""
        if self.shadows:
            c, h, m = self.charts, self.hit, self.ctx.model
            wp.launch(shadow_forces, dim=self.ctx.world_count,
                      inputs=[state.body_q, m.body_com, m.shape_body, c.cbody, c.clocal, c.cnrm, self.occ, h["p"],
                              h["w"], h["on"], self.spray], outputs=[forces.body_f], device=self.ctx.device)

    def _sprays(self, state: Any, dt: float) -> None:
        """In ``before_film``: the boxes into ``written`` (after its clear), the hidden shares caught, the stage drips'
        cones past their charts' edges lost, then a gather per row of each box (apply's footprints and shade)."""
        W, dev, c = self.ctx.world_count, self.ctx.device, self.charts
        wp.launch(boxes_written, dim=self.box.shape, inputs=[c.charts, self.box], outputs=[self.written], device=dev)
        if self.shadows:
            wp.launch(spray_caught, dim=W, inputs=[wp.float64(dt), self.spray, self.ns],
                      outputs=[self.ledger, self.caught_by], device=dev)
        if self.n_stage:
            wp.launch(SPR.spray_off, dim=W, inputs=[wp.float64(dt), self.spray, self.ns], outputs=[self.ledger],
                      device=dev)
        wp.launch(SPR.spray_gather, dim=(*self.box.shape, self.rows),
                  inputs=[wp.float64(dt), c.charts, c.prof, c.cpos, c.area, self.spray, self.box, self.sbit],
                  outputs=[self.film.src_acc, self.film.impact_p, self.film.impact_share, self.film.cwet, self.ns,
                           self.shade], device=dev)


__all__ = ["MAX_SHADE", "SHADOW_TOL", "Spraying", "boxes_written", "is_hidden", "nearest_blocker", "shade_bits",
           "shaded", "shadow_cull", "shadow_forces", "spray_caught", "spray_shadow", "spray_visible"]
