# SPDX-License-Identifier: Apache-2.0
"""Film forces on bodies (design §4.5 and §4.7; WP9): the film's weight, the impulse of liquid it takes in, and the
film's viscous drag on listed debris bodies.

- **Weight.** ``weight_kernel`` (world, tile) sums ``ρ·h·A`` and its first moment in world over the active tiles
  only; the others exit at once (no dense pass). Then ``apply_kernel`` (one thread per world, charts in order) adds
  ``M·g`` at the centroid to each chart's body. A static chart has no body, and its film's weight goes to the
  ground.
- **Absorbed impulse.** ``drops.absorb`` adds each absorbed particle's or drop's normal momentum to
  ``impulse [W, K]`` and its moment ``Σ p × J`` to ``moment [W, K]`` (world frame). ``apply`` adds ``J/dt`` with its
  torque to the chart's body, then zeroes both.
- **Debris drag.** ``Debris(body, radius)`` is a body lying on the film. Over the cells within ``radius`` of the
  point under its centre of mass, on the nearest chart, the force on the body is
  ``-film_shear(μ, τ_y, h, v)·A·v``, with ``v`` the body's tangential velocity relative to the film surface
  (``1.5 x`` the depth-mean film velocity, plus the chart body's own motion). ``film_shear`` is the tools' ``μ/h``
  (D-8). The reaction goes on the chart's body. As for tools, the drag is scaled so that one step cannot reverse
  the relative velocity: ``scale = min(1, 0.5 / (Σ coefficient · inverse mass · dt))``. The film's own momentum
  loss is a follow-up (§4.7).

``add_wrench`` writes without atomics, so every write to ``body_f`` is made by its world's one thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.geometry import cell_window, chart_xform, point_cell
from alloy.plugins.surface_flow.kernels import WALL, Cells, Faces, Fluid
from alloy.plugins.surface_flow.links import cell_dir
from alloy.plugins.surface_flow.tiles import Tiles
from alloy.plugins.surface_flow.tool_kernels import film_shear

SURFACE_SPEED = wp.constant(1.5)   # film surface speed / depth-mean speed (parabolic profile, design §3.2)


@dataclass(frozen=True)
class Debris:
    """A body lying on a film (grit on a floor being flushed): ``radius`` [m] of its footprint disc."""

    body: str
    radius: float = 0.01


@wp.kernel
def weight_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], cells: Cells, charts: wp.array[Chart],
                  cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                  h: wp.array2d[wp.float64], rho: float, mass: wp.array2d[wp.float64],
                  moment: wp.array2d[wp.vec3d]):
    """Per (world, tile): the film mass [kg] and its first moment [kg·m] (world) over the tile's cells. Only the
    tiles the last film substep stepped (``tile_on``): film moves at most one cell per substep, so every wet cell
    lies in one (tiles.py); with the skip off every tile is on."""
    w, t = wp.tid()
    m = wp.float64(0.0)
    mx = wp.vec3d(0.0, 0.0, 0.0)
    if tile_on[w, t] != 0:
        first, pitch, cols = tl.first[t], tl.pitch[t], tl.cols[t]
        k = cells.chart[first]
        X = chart_xform(body_q, cbody[w, k], clocal[w, k])
        for r in range(tl.rows[t]):
            for j in range(cols):
                c = first + r * pitch + j
                hc = h[w, c]
                if hc > 0.0:
                    mc = wp.float64(rho) * hc * cells.area[c]
                    x = wp.transform_point(X, cells.pos[c] + (0.5 * wp.float32(hc)) * cells.nrm[c])
                    m += mc
                    mx += mc * wp.vec3d(x)
    mass[w, t] = m
    moment[w, t] = mx


@wp.func
def film_velocity(cells: Cells, faces: Faces, u: wp.array2d[wp.float32], w: int, c: int) -> wp.vec3:
    """Depth-mean film velocity at cell c (chart frame): per axis, the mean of its non-wall faces' u along their
    directions."""
    fs = cells.face[c]
    out = wp.vec3(0.0, 0.0, 0.0)
    for axis in range(2):
        acc = wp.vec3(0.0, 0.0, 0.0)
        n = float(0.0)
        for s in range(2):
            f = fs[2 * axis + s]
            if f >= 0 and faces.kind[f] != WALL:
                acc = acc + u[w, f] * cell_dir(faces, f, c)
                n += 1.0
        if n > 0.0:
            out = out + acc / n
    return out


@wp.struct
class DebrisSet:
    body: wp.array2d[wp.int32]       # [W, D] global body ids
    radius: wp.array[wp.float32]     # [D]


@wp.func
def _nearest_chart(charts: wp.array[Chart], prof: wp.array[wp.vec3], cbody: wp.array2d[wp.int32],
                   clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform], w: int, p: wp.vec3,
                   reach: float) -> wp.vec2i:
    """(chart, cell) under world point p with the smallest |normal distance| <= reach, (-1, -1) for none."""
    best = wp.vec2i(-1, -1)
    best_d = float(reach)
    for k in range(charts.shape[0]):
        X = chart_xform(body_q, cbody[w, k], clocal[w, k])
        near = point_cell(charts[k], prof, wp.transform_point(wp.transform_inverse(X), p))
        if near.cell >= 0 and wp.abs(near.dist) <= best_d:
            best_d = wp.abs(near.dist)
            best = wp.vec2i(k, near.cell)
    return best


@wp.func
def _debris_drag(ds: DebrisSet, d: int, w: int, charts: wp.array[Chart], prof: wp.array[wp.vec3], cells: Cells,
                 faces: Faces, cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                 body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 inv_mass: wp.array[wp.float32], h: wp.array2d[wp.float64], u: wp.array2d[wp.float32], fl: Fluid,
                 h_wet: float, dt: float, body_f: wp.array[wp.spatial_vector]):
    b = ds.body[w, d]
    rad = ds.radius[d]
    com = com_world(body_q, body_com, b)
    hit = _nearest_chart(charts, prof, cbody, clocal, body_q, w, com, 4.0 * rad)
    if hit[0] < 0:
        return
    k = hit[0]
    ch = charts[k]
    cb = cbody[w, k]
    X = chart_xform(body_q, cb, clocal[w, k])
    centre = wp.transform_point(wp.transform_inverse(X), com)
    loc = hit[1] - ch.cell_start
    win = cell_window(ch, prof, loc // ch.ny, loc % ch.ny, rad)
    scale = float(1.0)
    for p in range(2):          # pass 0: the drag coefficient sum (stability scale); pass 1: the forces
        coef = float(0.0)
        on_body = wp.spatial_vector()
        on_chart = wp.spatial_vector()
        chart_com = wp.vec3(0.0, 0.0, 0.0)
        if cb >= 0:
            chart_com = com_world(body_q, body_com, cb)
        for ii in range(win[0], win[1] + 1):
          for jj in range(win[2], win[3] + 1):
            c = ch.cell_start + ii * ch.ny + (jj % ch.ny + ch.ny) % ch.ny
            hc = wp.float32(h[w, c])
            if hc <= h_wet:
                continue
            rel = cells.pos[c] - centre
            rel = rel - wp.dot(rel, cells.nrm[c]) * cells.nrm[c]
            if wp.length(rel) > rad:
                continue
            a = wp.float32(cells.area[c])
            x = wp.transform_point(X, cells.pos[c] + hc * cells.nrm[c])
            n = wp.transform_vector(X, cells.nrm[c])
            v_surface = wp.transform_vector(X, SURFACE_SPEED * film_velocity(cells, faces, u, w, c))
            if cb >= 0:
                v_surface = v_surface + point_velocity(body_qd, chart_com, cb, x)
            v = point_velocity(body_qd, com, b, x) - v_surface
            v = v - wp.dot(v, n) * n
            k_c = film_shear(fl, w, c, hc, v) * a
            if p == 0:
                coef += k_c
                continue
            f = -k_c * v
            on_body = on_body + wrench_at(f, x, com)
            on_chart = on_chart - wrench_at(f, x, chart_com)
        if p == 0:
            if coef <= 0.0:
                return
            scale = wp.min(1.0, 0.5 / wp.max(coef * inv_mass[b] * dt, 1.0e-12))
            continue
        add_wrench(body_f, b, scale * on_body)
        if cb >= 0:
            add_wrench(body_f, cb, scale * on_chart)


@wp.kernel
def apply_kernel(tile_start: wp.array[wp.int32], mass: wp.array2d[wp.float64], moment: wp.array2d[wp.vec3d],
                 weight: int, cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                 body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], gravity: wp.array[wp.vec3],
                 g_chart: wp.array2d[wp.vec3], tick: wp.array[wp.int32], impulse: wp.array2d[wp.vec3],
                 imp_moment: wp.array2d[wp.vec3], dt: float, body_f: wp.array[wp.spatial_vector]):
    """Per world: each chart's film weight and absorbed impulse on the chart's body; the impulse store is zeroed.
    The film pushes its carrier with ``M·(g - a)``: the chart frame's effective gravity (``g_chart``, from the film's
    last exchange, turned into the world frame), or plain gravity before the film's first exchange."""
    w = wp.tid()
    for k in range(cbody.shape[1]):
        b = cbody[w, k]
        g = gravity[w]
        if tick[w] > 0:
            g = wp.transform_vector(chart_xform(body_q, b, clocal[w, k]), g_chart[w, k])
        J = impulse[w, k]
        L = imp_moment[w, k]
        impulse[w, k] = wp.vec3(0.0, 0.0, 0.0)
        imp_moment[w, k] = wp.vec3(0.0, 0.0, 0.0)
        if b < 0:
            continue
        com = com_world(body_q, body_com, b)
        if weight != 0:
            m = wp.float64(0.0)
            mx = wp.vec3d(0.0, 0.0, 0.0)
            for t in range(tile_start[k], tile_start[k + 1]):
                m += mass[w, t]
                mx += moment[w, t]
            if m > 0.0:
                c = mx / m
                add_wrench(body_f, b, wrench_at(wp.float32(m) * g, wp.vec3(wp.float32(c[0]), wp.float32(c[1]),
                                                                              wp.float32(c[2])), com))
        tau = L - wp.cross(com, J)
        add_wrench(body_f, b, wp.spatial_vector(J[0] / dt, J[1] / dt, J[2] / dt, tau[0] / dt, tau[1] / dt,
                                                tau[2] / dt))


@wp.kernel
def debris_kernel(ds: DebrisSet, n_debris: int, charts: wp.array[Chart], prof: wp.array[wp.vec3], cells: Cells,
                  faces: Faces, cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                  body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                  body_com: wp.array[wp.vec3], inv_mass: wp.array[wp.float32], h: wp.array2d[wp.float64],
                  u: wp.array2d[wp.float32], fl: Fluid, h_wet: float, dt: float,
                  body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    for d in range(n_debris):
        _debris_drag(ds, d, w, charts, prof, cells, faces, cbody, clocal, body_q, body_qd, body_com, inv_mass, h, u,
                     fl, h_wet, dt, body_f)


def chart_tile_starts(cell_chart: np.ndarray, tile_first: np.ndarray, charts: int) -> np.ndarray:
    """[K + 1] first tile of each chart (a chart's tiles are contiguous)."""
    tile_chart = cell_chart[tile_first]
    if len(tile_chart) and np.any(np.diff(tile_chart) < 0):
        raise ValueError("surface_flow: tiles are not grouped by chart")
    return np.searchsorted(tile_chart, np.arange(charts + 1)).astype(np.int32)


class BodyForces:
    """The film's forces on bodies for one ``SurfaceFlow`` (module docstring). ``film`` is set up already."""

    def __init__(self, film: Any, weight: bool = True, debris: tuple[Debris, ...] = ()):
        ctx, t, tl = film.ctx, film.tables, film.tl
        W, K = ctx.world_count, len(film.config.charts)
        self.film, self.weight = film, bool(weight)
        self.debris = tuple(Debris(**d) if isinstance(d, dict) else d for d in debris)
        first = tl.tiles.first.numpy()
        self.tile_start = ctx.array(chart_tile_starts(t.cchart.numpy(), first, K), wp.int32)
        self.mass = ctx.zeros((W, max(tl.count, 1)), wp.float64)
        self.moment = ctx.zeros((W, max(tl.count, 1)), wp.vec3d)
        # drops.absorb adds Σ m·v_n here (world frame) and Σ p × (m·v_n); state, so a reset or restore carries it
        self.impulse = film.add_state("drops_impulse", ctx.zeros((W, K), wp.vec3))
        self.imp_moment = film.add_state("drops_imp_moment", ctx.zeros((W, K), wp.vec3))
        self.ds = DebrisSet()
        bodies = [ctx.bodies(d.body).numpy() for d in self.debris]
        self.ds.body = ctx.array(np.stack(bodies, 1) if bodies else np.zeros((W, 1), np.int32), wp.int32)
        self.ds.radius = ctx.array([d.radius for d in self.debris] or [0.0], wp.float32)

    def apply(self, state: Any, forces: Any, dt: float, h_wet: float) -> None:
        f, ctx, t = self.film, self.film.ctx, self.film.tables
        W, dev = ctx.world_count, ctx.device
        if self.weight and f.tl.count:
            wp.launch(weight_kernel, dim=(W, f.tl.count),
                      inputs=[f.tl.tiles, f.tile_on, f.cells, t.charts, t.cbody, t.clocal, state.body_q, f.h,
                              float(f.mat.density)], outputs=[self.mass, self.moment], device=dev)
        wp.launch(apply_kernel, dim=W,
                  inputs=[self.tile_start, self.mass, self.moment, int(self.weight), t.cbody, t.clocal, state.body_q,
                          ctx.model.body_com, ctx.model.gravity, f.frame.g, f.tick, self.impulse, self.imp_moment,
                          float(dt)],
                  outputs=[forces.body_f], device=dev)
        if self.debris:
            wp.launch(debris_kernel, dim=W,
                      inputs=[self.ds, len(self.debris), t.charts, t.prof, f.cells, f.faces, t.cbody, t.clocal,
                              state.body_q, state.body_qd, ctx.model.body_com, ctx.model.body_inv_mass, f.h, f.u,
                              f.fluid, float(h_wet), float(dt)],
                      outputs=[forces.body_f], device=dev)


__all__ = ["BodyForces", "Debris", "apply_kernel", "chart_tile_starts", "debris_kernel", "film_velocity",
           "weight_kernel"]
