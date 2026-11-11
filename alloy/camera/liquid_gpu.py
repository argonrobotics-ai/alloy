# SPDX-License-Identifier: Apache-2.0
"""Live SPH liquids in the batched camera: every render turns each liquid's poured particles of every world into one
closed, smoothed surface mesh per world on the device (:mod:`alloy.camera.liquid_kernels`), with no host copy of the
particles.

Each world's mesh has a fixed capacity (``ALLOY_CAMERA_LIQUID_TRIS`` triangles, default 24 per particle and at least
8,192; a warning names the knob when a surface needs more). The topology changes every render, so the meshes' BVHs
are rebuilt, never refitted: a new ``wp.Mesh`` per world on the Warp backend, an in-place GAS rebuild on OptiX (a
fixed GAS buffer sized for the capacity, so its traversable handle stays). The grids of all worlds share one node
budget (``ALLOY_CAMERA_LIQUID_NODES``, default 2^26); a larger extent coarsens the cells for that render (warned once).

Example::

    surf = LiquidSurface(obj, worlds=64, stride=layout.stride, device="cuda:0")
    nv, nt = surf.build(state.particle_q, flags)    # host (worlds,) vertex and triangle counts
    surf.pts, surf.tris                             # (worlds, cap) vec3 and (worlds, 3 cap) int32 device buffers
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import liquid_kernels as lk
from .envvars import env_int
from . import liquid_blocks as lb
from .liquid_blocks import mark_blocks

log = logging.getLogger(__name__)

LAMBDA = 0.9  # Yu-Turk smoothing weight
SUPPORT_DX = 2.0  # kernel support, in particle spacings
RADIUS_DX = 1.0  # Zhu-Bridson particle radius, in spacings (GA: 0.55 at 2 mm particles; see notes below)
CELL_DX = 0.5  # grid cell, in spacings
N_EPS = 13  # fewer neighbours within the support (bulk: ~33, its free surface ~17, a falling plate 9): a stream segment
L_MAX = 2.0  # longest stream half segment, in supports (joins layers up to 3.3 supports apart)
V_MIN = 0.05  # m/s: slower sparse particles keep round kernels
TAUBIN = (0.5, -0.53)  # (lambda, mu) of one Taubin pass
TAUBIN_PASSES = 2
DENSE_FIELD = False  # True: evaluate every block (A / B checks of liquid_blocks; same bits)
TRIS_PER_PARTICLE = 24  # default capacity: a pour mid-stream needed 21 per particle (lone drops: ~100 each)
MIN_TRIS = 8192


@dataclass(frozen=True)
class LiquidObject:
    """One SPH liquid: ``count`` particles per world from ``start`` (world 0), ``spacing`` metres apart."""

    name: str
    start: int
    count: int
    spacing: float
    material: str = "water"


def capacity(obj: LiquidObject) -> int:
    """Triangles per world mesh (``ALLOY_CAMERA_LIQUID_TRIS`` or 24 per particle, at least 8,192)."""
    cap = env_int("ALLOY_CAMERA_LIQUID_TRIS", 0)
    return int(cap) if cap > 0 else max(MIN_TRIS, -(-TRIS_PER_PARTICLE * obj.count // 1024) * 1024)


def _grid_shape(lo: np.ndarray, hi: np.ndarray, ok: np.ndarray, cell: float, pad: float) -> tuple[np.ndarray, tuple]:
    """(world-space grid corners (W, 3), node counts (nx, ny, nz) shared by every world) for cell size ``cell``."""
    lo64 = np.where(ok[:, None], lo, 0.0).astype(np.float64)
    hi64 = np.where(ok[:, None], hi, 0.0).astype(np.float64)
    corner = np.floor((lo64 - pad) / cell) * cell  # snapped: the surface does not shimmer as the bounds move
    n = np.ceil((hi64 + pad - corner) / cell).astype(np.int64) + 1
    dims = np.maximum(n[ok].max(0) if ok.any() else np.full(3, 2), 2)
    return corner, (int(dims[0]), int(dims[1]), int(dims[2]))


class LiquidSurface:
    """The surfaces of one liquid over every world, rebuilt by :meth:`build` (see the module docstring)."""

    def __init__(self, obj: LiquidObject, worlds: int, stride: int, device: str):
        self.obj, self.worlds, self.stride, self.device = obj, int(worlds), int(stride), str(device)
        self.tcap = capacity(obj)
        # a closed marching-cubes surface has V = T / 2 + 2 per component (lone drops: one component each)
        self.vcap = self.tcap // 2 + max(4096, 2 * obj.count)
        w, dev, n = self.worlds, self.device, obj.count
        self.pts = wp.zeros((w, self.vcap), dtype=wp.vec3, device=dev)
        self.tmp = wp.zeros((w, self.vcap), dtype=wp.vec3, device=dev)
        self.vn = wp.zeros((w, self.vcap), dtype=wp.vec3, device=dev)
        self.vkey = wp.zeros((w, self.vcap), dtype=wp.int32, device=dev)
        self.tris = wp.zeros((w, 3 * self.tcap), dtype=wp.int32, device=dev)
        self.act = wp.zeros(w * n + 1, dtype=wp.int32, device=dev)  # + a zero tail: the exclusive scan ends in the total
        self.slot = wp.zeros(w * n + 1, dtype=wp.int32, device=dev)
        m = max(w * n, 1)
        self.packed = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.smoothed = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.vel = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.pworld = wp.zeros(m, dtype=wp.int32, device=dev)
        self.ncount = wp.zeros(m, dtype=wp.int32, device=dev)
        self.flag = wp.zeros(m + 1, dtype=wp.int32, device=dev)
        self.aslot = wp.zeros(m + 1, dtype=wp.int32, device=dev)
        self.axis = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.half = wp.zeros(m, dtype=float, device=dev)
        self.an = lk.Aniso()
        self.an.pts = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.an.axis = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.an.half = wp.zeros(m, dtype=float, device=dev)
        self.hash_a = wp.HashGrid(128, 128, 128, device=dev, dtype=wp.float64)
        self.a64 = wp.zeros(m, dtype=wp.vec3d, device=dev)
        self.lo = wp.zeros(w, dtype=wp.vec3, device=dev)
        self.hi = wp.zeros(w, dtype=wp.vec3, device=dev)
        self.nv = wp.zeros(w, dtype=wp.int32, device=dev)
        self.nt = wp.zeros(w, dtype=wp.int32, device=dev)
        self.hash = wp.HashGrid(128, 128, 128, device=dev, dtype=wp.float64)  # float64: see liquid_kernels.to_f64
        self.p64 = wp.zeros(m, dtype=wp.vec3d, device=dev)
        self.tables = lk.make_tables(dev)
        self.g = lk.Grid()
        self.g.vb0 = wp.zeros(w, dtype=wp.int32, device=dev)
        self.g.tb0 = wp.zeros(w, dtype=wp.int32, device=dev)
        self._nodes = 0  # grid buffers sized for this many nodes over all worlds
        self._blocks = 0  # g.blocks entries
        self.max_nodes = env_int("ALLOY_CAMERA_LIQUID_NODES", 1 << 26)
        self.counts = (np.zeros(w, np.int64), np.zeros(w, np.int64))
        self._warned = set()

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _active(self, pq: wp.array, flags: wp.array) -> int:
        """Mark the poured particles and their bounds per world; returns how many there are (host sync)."""
        o, dev = self.obj, self.device
        self.lo.fill_(wp.vec3(1.0e30))
        self.hi.fill_(wp.vec3(-1.0e30))
        wp.launch(lk.mark_active, dim=(self.worlds, o.count), inputs=[pq, flags, o.start, self.stride, o.count],
                  outputs=[self.act, self.lo, self.hi], device=dev)
        wp.utils.array_scan(self.act, self.slot, inclusive=False)
        n = self.worlds * o.count
        return int(self.slot[n:n + 1].numpy()[0])

    def _layout(self) -> None:
        """Grid corners, node counts, cell size and tiles from this render's bounds (host)."""
        lo, hi = self.lo.numpy(), self.hi.numpy()
        ok = (lo <= hi).all(axis=1)
        dx = self.obj.spacing
        self.radius = SUPPORT_DX * dx
        pad = self.radius + 2.0 * CELL_DX * dx
        cell = CELL_DX * dx
        corner, dims = _grid_shape(lo, hi, ok, cell, pad)
        while self.worlds * math.prod(dims) > self.max_nodes:
            cell *= 1.05 * (self.worlds * math.prod(dims) / self.max_nodes) ** (1.0 / 3.0)
            corner, dims = _grid_shape(lo, hi, ok, cell, max(pad, self.radius + 2.0 * cell))
            self._warn_once("coarse", "camera: liquid %r spans %s grid nodes over %d worlds at %.2g m cells; drawing it "
                            "with %.2g m cells (raise ALLOY_CAMERA_LIQUID_NODES, %d now, or render fewer worlds)",
                            self.obj.name, dims, self.worlds, CELL_DX * dx, cell, self.max_nodes)
        k = max(1, math.ceil(self.worlds ** (1.0 / 3.0)))
        span = (np.asarray(dims, np.float64) - 1.0) * cell + 2.0 * self.radius + cell
        w = np.arange(self.worlds)
        tile = np.stack([w % k, (w // k) % k, w // (k * k)], 1) * span
        g, dev = self.g, self.device
        g.nx, g.ny, g.nz = dims
        g.nn, g.cell = int(math.prod(dims)), float(cell)
        g.bshift = lb.BLOCK_SHIFT
        g.nbx, g.nby, g.nbz = lb.block_counts(*dims, g.bshift)
        need = self.worlds * g.nbx * g.nby * g.nbz
        if self._blocks < need:
            self._blocks = int(need * 1.5) + 1
            g.blocks = wp.zeros(self._blocks, dtype=wp.int32, device=self.device)
        g.lo = wp.array(corner.astype(np.float32), dtype=wp.vec3, device=dev)
        g.tile = wp.array(tile.astype(np.float32), dtype=wp.vec3, device=dev)
        self._grid_buffers(self.worlds * g.nn)

    def _grid_buffers(self, nodes: int) -> None:
        """Node buffers for ``nodes`` nodes (grown by half again when too small; + a zero tail for the scans)."""
        if nodes <= self._nodes:
            return
        self._nodes = int(nodes * 1.5) + 1
        g, dev = self.g, self.device
        g.field = wp.zeros(self._nodes, dtype=float, device=dev)
        self.count = wp.zeros(self._nodes + 1, dtype=wp.int32, device=dev)
        g.vbase = wp.zeros(self._nodes + 1, dtype=wp.int32, device=dev)
        g.tbase = wp.zeros(self._nodes + 1, dtype=wp.int32, device=dev)

    def _implicit(self, pq: wp.array, qd: wp.array, total: int) -> None:
        o, g, dev, R = self.obj, self.g, self.device, self.radius
        wp.launch(lk.compact, dim=(self.worlds, o.count),
                  inputs=[pq, qd, self.act, self.slot, o.start, self.stride, o.count, g],
                  outputs=[self.packed, self.vel, self.pworld], device=dev)
        pk, sm = self.packed[:total], self.smoothed[:total]
        self._build_grid(self.hash, pk, self.p64, R)
        wp.launch(lk.smooth, dim=total, inputs=[self.hash.id, pk, R, LAMBDA], outputs=[sm, self.ncount], device=dev)
        self._anisotropy(pk, total)
        self._build_grid(self.hash, sm, self.p64, R)
        if DENSE_FIELD:
            g.blocks.fill_(1)
        else:
            g.blocks.zero_()
            wp.launch(mark_blocks, dim=total, inputs=[sm, self.pworld, self.flag, self.half, g, R], device=dev)
        wp.launch(lk.implicit, dim=self.worlds * g.nn, inputs=[self.hash.id, sm, self.flag, self.an, g, R,
                                                               RADIUS_DX * o.spacing], outputs=[g.field], device=dev)

    def _build_grid(self, grid: wp.HashGrid, pts: wp.array, buf: wp.array, cell: float) -> None:
        """Build a float64 hash grid over float32 ``pts`` (via ``buf``; see ``liquid_kernels.to_f64``)."""
        n = pts.shape[0]
        wp.launch(lk.to_f64, dim=n, inputs=[pts], outputs=[buf], device=self.device)
        grid.build(buf[:n], cell)

    def _anisotropy(self, pk: wp.array, total: int) -> None:
        """Stream segments of the sparse moving particles (``lk.stretch``), gathered with their own hash grid."""
        dev, R, an = self.device, self.radius, self.an
        wp.launch(lk.stretch, dim=total, inputs=[self.hash.id, pk, self.vel, self.ncount, N_EPS, L_MAX * R, V_MIN],
                  outputs=[self.axis, self.half, self.flag], device=dev)
        self.flag[total:total + 1].zero_()
        wp.utils.array_scan(self.flag[: total + 1], self.aslot[: total + 1], inclusive=False)
        an.n = int(self.aslot[total:total + 1].numpy()[0])
        an.reach = float((1.0 + L_MAX) * R)
        if an.n:
            wp.launch(lk.gather_aniso, dim=total, inputs=[self.flag, self.aslot, self.smoothed, self.axis, self.half],
                      outputs=[an.pts, an.axis, an.half], device=dev)
            self._build_grid(self.hash_a, an.pts[: an.n], self.a64, an.reach)
            an.grid = self.hash_a.id

    def _scan(self, kernel, inputs: list, out: wp.array) -> None:
        n = self.worlds * self.g.nn
        cnt = self.count[: n + 1]
        cnt.zero_()
        wp.launch(kernel, dim=n, inputs=inputs, outputs=[cnt], device=self.device)
        wp.utils.array_scan(cnt, out[: n + 1], inclusive=False)

    def _cubes(self) -> None:
        """Marching cubes: counts, scans, vertices and triangles; the per-world counts to the host."""
        g, tb, dev = self.g, self.tables, self.device
        self._scan(lk.vert_count, [g], g.vbase)
        self._scan(lk.tri_count, [g, tb], g.tbase)
        wp.launch(lk.world_counts, dim=self.worlds, inputs=[g], outputs=[self.nv, self.nt], device=dev)
        n = self.worlds * g.nn
        wp.launch(lk.emit_verts, dim=n, inputs=[g, self.vcap], outputs=[self.pts, self.vkey], device=dev)
        wp.launch(lk.emit_tris, dim=n, inputs=[g, tb, self.vcap, self.tcap], outputs=[self.tris], device=dev)

    def _fair(self) -> None:
        """Taubin passes, then area-weighted vertex normals."""
        g, tb, dev, dim = self.g, self.tables, self.device, (self.worlds, self.vcap)
        for _ in range(TAUBIN_PASSES):
            for wgt, (src, dst) in zip(TAUBIN, ((self.pts, self.tmp), (self.tmp, self.pts))):
                wp.launch(lk.taubin, dim=dim, inputs=[g, tb, self.vkey, self.nv, self.vcap, wgt, src], outputs=[dst],
                          device=dev)
        wp.launch(lk.vertex_normals, dim=dim, inputs=[g, tb, self.vkey, self.nv, self.vcap, self.pts],
                  outputs=[self.vn], device=dev)

    def build(self, pq: wp.array, flags: wp.array, qd: wp.array | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild every world's surface from ``pq`` (``state.particle_q``), ``qd`` (``state.particle_qd``; None: no
        stream kernels) and ``flags`` (int32, model numbering).
        Returns host (vertex counts, triangle counts) per world, clamped to the capacity (at least 1 triangle: an
        empty world keeps a degenerate one)."""
        total = self._active(pq, flags)
        self._layout()
        if total == 0:
            self.nv.zero_()
            self.nt.zero_()
        else:
            if qd is None:  # no velocities: round kernels everywhere
                qd = wp.zeros_like(pq)
            self._implicit(pq, qd, total)
            self._cubes()
            self._fair()
        wp.launch(lk.finish_empty, dim=self.worlds, inputs=[self.nt, self.g], outputs=[self.pts, self.tris],
                  device=self.device)
        nv, nt = self.nv.numpy().astype(np.int64), self.nt.numpy().astype(np.int64)
        if (nt > self.tcap).any() or (nv > self.vcap).any():
            self._warn_once("cap", "camera: liquid %r needs %d triangles in a world (capacity %d): the surface has "
                            "holes; set ALLOY_CAMERA_LIQUID_TRIS higher", self.obj.name, int(nt.max()), self.tcap)
        self.counts = (np.minimum(nv, self.vcap), np.maximum(np.minimum(nt, self.tcap), 1))
        return self.counts

    def mesh(self, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host copy of world ``w``'s last surface: (verts (V, 3), tris (T, 3), vertex normals (V, 3))."""
        nv, nt = int(self.counts[0][w]), int(self.counts[1][w])
        if int(self.nt.numpy()[w]) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32)
        return (self.pts.numpy()[w, :nv].copy(), self.tris.numpy()[w, : 3 * nt].reshape(-1, 3).copy(),
                self.vn.numpy()[w, :nv].copy())
