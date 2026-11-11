# SPDX-License-Identifier: Apache-2.0
"""Wavefront host loop: buffers, waves, the per-bounce cast / sort / shade cycle and the final accumulation.

One wave = up to ``max_paths`` (sample, pixel) work items, each a camera path. Per bounce: tracer closest hit over the
active queue -> [sort by material] -> [Bevel pre-pass + Bevel cast] -> Warp shade kernel (the megakernel's loop body:
writes the NEE shadow ray + its contribution and the next ray, appends to the next / shadow queues) -> shadow rays
(terminate on first hit) -> Warp adds the unoccluded contributions. At the end of a wave every path's radiance /
albedo / normal goes to per-slot pixel accumulators (deterministic: one item per pixel and slot per wave);
``resolve`` divides by spp. One host sync per bounce (the queue sizes).

Integrators plug in :class:`Stages` (camera init + shade vertex, optional Bevel and sort callbacks).

Example::

    wf = Wavefront(tracer, n_pixels, "cuda:0", img_world, max_paths=1 << 24)
    wf.run(spp, Stages(camera, shade), rgb, albedo, normal)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import warp as wp

from .state import PathState, Wave, accumulate, alloc_path_state, iota, resolve, shadow_add

log = logging.getLogger(__name__)


@dataclass
class Stages:
    """Integrator callbacks of one render.

    camera(wv, n): init PathState and the first ray of the wave's ``n`` items.
    shade(wv, n, q_in, q_out, q_sh, counts, first): one vertex of every queued path (``first``: identity queue).
    bevel(wv, n, q_in, first, counts): optional pass between the closest hits and shading (Bevel rays).
    sort_key(wv, n, q_in, keys): optional, writes a material key per queued path; the queue is then sorted by it
        (results do not depend on the order, shading is coherent).
    sorted_cb(n, keys): optional, called right after each sort (``keys[:n]`` sorted).
    after_trace(wv, n, q_in, first): optional, right after the closest-hit cast (e.g. restore hits that were not
        traced, such as subsurface exits).
    """

    camera: Callable
    shade: Callable
    bevel: Callable | None = None
    sort_key: Callable | None = None
    sorted_cb: Callable | None = None
    after_trace: Callable | None = None


class Wavefront:
    """Buffers + host loop. ``self.timing`` (seconds per stage, synchronised, when ``profile`` is True) splits a
    render; ``self.stats`` counts rays and vertices."""

    def __init__(self, tracer, npix: int, device, img_world: wp.array, max_paths: int, sort_min: int = 1 << 16,
                 max_bounces: int = 64, max_paths_cap: int = 1 << 24):
        self.tracer, self.npix, self.device, self.img_world = tracer, int(npix), device, img_world
        self.max_paths = min(int(max_paths), max_paths_cap)  # a sort key packs the path id below its key class
        self.max_iters, self.sort_min = int(max_bounces), int(sort_min)
        # the most paths one wave may hold; buffers grow up to it as renders need (see reserve)
        self.cap_paths = (min(self.npix * max(1, self.max_paths // self.npix), self.max_paths)
                          if self.npix <= self.max_paths else self.max_paths)
        self.cap_slots = max(1, self.cap_paths // self.npix)
        self.n_paths, self.max_slots = 0, 0
        self.ps: PathState | None = None
        self.q: list[wp.array] = []
        self.acc: list[wp.array] = []
        self.keys = None
        self.sorted_now = False  # the active queue of this bounce is sorted (integrators may launch per range)
        self.counts = wp.zeros(4, dtype=wp.int32, device=device)
        self.counts_host = wp.zeros(4, dtype=wp.int32, device="cpu", pinned=True)
        self.profile = False
        self.timing: dict[str, float] = {}
        self.stats: dict[str, int] = {}
        self._tmark = 0.0
        self._qa = self._qb = None

    def reserve(self, spp: int) -> bool:
        """Grow the path buffers to fit a ``spp``-sample render (never past the cap); True when they were allocated.
        The wave schedule is the same as with buffers at the cap, so images do not depend on earlier renders."""
        need = self.cap_paths if self.npix > self.cap_paths else self.npix * min(self.cap_slots, max(1, int(spp)))
        if need <= self.n_paths:
            return False
        dev = self.device
        self.n_paths, self.max_slots = need, max(1, need // self.npix)
        self.ps = alloc_path_state(need, dev)
        # queues are 2n long: wp.utils.radix_sort_pairs sorts (keys, queue) in place and needs the scratch half
        self.q = [wp.zeros(2 * need, dtype=wp.int32, device=dev) for _ in range(3)]
        self.keys = None
        self.acc = [wp.zeros(self.max_slots * self.npix, dtype=wp.vec3, device=dev) for _ in range(3)]
        log.info("wavefront: %d paths per wave (%d pixels, %d slots)", self.n_paths, self.npix, self.max_slots)
        return True

    def read_counts(self) -> np.ndarray:
        """The device counters (next queue, shadow queue, Bevel queue) on the host; waits for the stream."""
        wp.copy(self.counts_host, self.counts)
        wp.synchronize_stream(wp.get_stream(self.device))
        return self.counts_host.numpy().copy()

    def _tick(self, key: str) -> None:
        if not self.profile:
            return
        wp.synchronize_device(self.device)
        now = time.perf_counter()
        self.timing[key] = self.timing.get(key, 0.0) + (now - max(self._tmark, 0.0))
        self._tmark = now

    def mark(self, key: str) -> None:
        """Profiling split inside an integrator callback: time since the last stage boundary goes to ``key``."""
        self._tick(key)

    def waves(self, spp: int):
        """(g0, n_items, n_slots) of every wave of a ``spp``-sample render."""
        if self.npix <= self.n_paths:
            n_slots = min(self.max_slots, spp)
            for s0 in range(0, spp, n_slots):
                yield s0 * self.npix, min(n_slots, spp - s0) * self.npix, n_slots
            return
        for s in range(spp):
            for c0 in range(0, self.npix, self.n_paths):
                yield s * self.npix + c0, min(self.n_paths, self.npix - c0), 1

    def run(self, spp: int, stages: Stages, out_rgb: wp.array, out_alb: wp.array, out_nrm: wp.array) -> None:
        """Render ``spp`` samples per pixel into the three output arrays (means)."""
        self.reserve(spp)
        self.timing, self.stats = {}, {"rays": 0, "shadow_rays": 0, "vertices": 0, "bevel_calls": 0}
        self._tmark = time.perf_counter()
        if self.profile:
            wp.synchronize_device(self.device)
            self._tmark = time.perf_counter()
        for a in self.acc:
            a.zero_()
        n_slots_used = 1
        for g0, n, n_slots in self.waves(spp):
            n_slots_used = n_slots
            wv = Wave()
            wv.g0, wv.npix, wv.n_slots, wv.img_world = g0, self.npix, n_slots, self.img_world
            stages.camera(wv, n)
            self._tick("camera")
            self._trace_wave(wv, n, stages)
            wp.launch(accumulate, dim=n, inputs=[self.ps, wv], outputs=self.acc, device=self.device)
        wp.launch(resolve, dim=self.npix, inputs=[*self.acc, n_slots_used, self.npix, 1.0 / float(spp)],
                  outputs=[out_rgb, out_alb, out_nrm], device=self.device)
        self._tick("accumulate")

    def _trace_wave(self, wv: Wave, n: int, stages: Stages) -> None:
        self._qa, self._qb = self.q[0], self.q[1]
        n_cur, first = n, True
        for _ in range(self.max_iters):
            n_cur = self._bounce(wv, n_cur, first, stages)
            first = False
            if n_cur == 0:
                break

    def _bounce(self, wv: Wave, n: int, first: bool, stages: Stages) -> int:
        """One cast / sort / shade / shadow cycle over the ``n`` queued paths; returns the next queue size."""
        ps, tracer = self.ps, self.tracer
        tracer.trace_closest(n, ps.o, ps.d, ps.tmax, ps.world, ps.hit_t, ps.hit_u, ps.hit_v, ps.hit_prim, ps.hit_inst,
                             queue=None if first else self._qa)
        self.stats["rays"] += n
        self._tick("trace")
        if stages.after_trace is not None:
            stages.after_trace(wv, n, self._qa, first)
        self.counts.zero_()
        self.sorted_now = False
        if stages.sort_key is not None and n >= self.sort_min:
            first = self._sort(wv, n, first, stages)
        if stages.bevel is not None:
            stages.bevel(wv, n, self._qa, first, self.counts)
            self._tick("bevel")
        stages.shade(wv, n, self._qa, self._qb, self.q[2], self.counts, first)
        self.stats["vertices"] += n
        self._tick("shade")
        c = self.read_counts()
        self.stats["bevel_calls"] += int(c[2])
        if c[1] > 0:
            self._shadow(int(c[1]))
        self._qa, self._qb = self._qb, self._qa
        return int(c[0])

    def _sort(self, wv: Wave, n: int, first: bool, stages: Stages) -> bool:
        """Sort the active queue by the integrator's key; returns the new ``first`` (False: the queue is explicit)."""
        if self.keys is None:
            self.keys = wp.zeros(2 * self.n_paths, dtype=wp.int32, device=self.device)
        if first:
            wp.launch(iota, dim=n, inputs=[self._qa], device=self.device)
        stages.sort_key(wv, n, self._qa, self.keys)
        wp.utils.radix_sort_pairs(self.keys, self._qa, n)
        self.sorted_now = True
        if stages.sorted_cb is not None:
            stages.sorted_cb(n, self.keys)
        self._tick("sort")
        return False

    def _shadow(self, n_sh: int) -> None:
        ps = self.ps
        self.tracer.trace_shadow(n_sh, ps.sh_o, ps.sh_d, ps.sh_tmax, ps.world, ps.occ, queue=self.q[2])
        self.stats["shadow_rays"] += n_sh
        self._tick("trace_shadow")
        wp.launch(shadow_add, dim=n_sh, inputs=[ps, self.q[2]], device=self.device)
        self._tick("shade")
