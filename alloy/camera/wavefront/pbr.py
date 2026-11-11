# SPDX-License-Identifier: Apache-2.0
"""Wavefront renderer for a :class:`~alloy.camera.pbr.renderer.PbrRenderer` (``backend`` "optix" or
"wavefront-warp"): camera kernel, material sort, Bevel pre-pass + cast, material-specialised shade kernels, accumulation.

Example::

    wf = PbrWavefront(renderer, tracer, WavefrontOptions.from_env())
    wf.render(spp, prm)            # fills renderer.rgb / albedo / normal
"""
from __future__ import annotations

import logging
import weakref

import numpy as np
import warp as wp

from ..errors import CameraError
from ..pbr.integrator import BumpTab
from .bevel import MAX_SAMPLES, alloc_bevel
from .bevel_flat import build_bevel_flat
from .host import Stages, Wavefront
from .knobs import WavefrontOptions
from .pbr_shade import ShadeKernels
from .pbr_stages import wf_bevel_prepass, wf_camera, wf_sss_fixup
from .sort import key_bins, key_shift, sort_key_kernel, spec_launches
from .state import pixel_rng_kernel

log = logging.getLogger(__name__)
_KERNELS: dict = {}


def _shade_kernels(disp, block_dim: int, warps: int) -> ShadeKernels:
    key = (disp.__name__, block_dim, warps)
    if key not in _KERNELS:
        _KERNELS[key] = ShadeKernels(disp, block_dim, warps)
    return _KERNELS[key]


def _has_subsurface(static: dict, names: list[str]) -> bool:
    """Whether any material can pick a BSSRDF (Subsurface Weight linked or non-zero)."""
    for nm in names:
        sw = static["materials"][nm].get("principled", {}).get("inputs", {}).get("Subsurface Weight", {})
        if sw.get("linked") or float(sw.get("value", 0.0) or 0.0) != 0.0:
            return True
    return False


class PbrWavefront:
    """Wavefront state and launches for one :class:`PbrRenderer` (kernels are cached per material module)."""

    def __init__(self, r, tracer, opts: WavefrontOptions):
        # a weak link back: the renderer owns this object, and a cycle would keep GPU buffers alive until gc runs
        self.r, self.opts, self.dev, self.tracer = weakref.proxy(r), opts, r.sc.device, tracer
        ps, dev = r.ps, self.dev
        bt: BumpTab = ps.bump
        self.bevel_on = bool(bt.bevel_on.numpy().any())
        if self.bevel_on and int(bt.bevel_samples.numpy().max()) > MAX_SAMPLES:
            raise CameraError(f"a Bevel node with more than {MAX_SAMPLES} samples",
                              f"use at most {MAX_SAMPLES} samples on the Bevel node for backend='optix'")
        self.kernels = _shade_kernels(ps.disp, r.block_dim, opts.shade_warps)
        self.has_sss = _has_subsurface(r.ps.pack.static, r.ps.mat_names)
        self.n_mats = len(ps.mat_names) + 1
        self.key_shift = key_shift(self.n_mats + 2)
        n = r.n_img * r.w * r.h
        self.wf = Wavefront(tracer, n, dev, r.cams.world, opts.paths_per_wave(dev, True), opts.sort_min,
                            max_paths_cap=1 << self.key_shift)
        self.pix_rng = wp.zeros(n, dtype=wp.uint32, device=dev)
        self._init_bevel()
        self._init_spec()
        self._prm = None

    def _init_bevel(self) -> None:
        """The flat-region table; Bevel buffers come with the path buffers (:meth:`_reserve`)."""
        r, dev = self.r, self.dev
        self.bb, self._bb_paths = alloc_bevel(1, dev), 1  # a one-path dummy until a render needs more
        n_tris = r.ps.shade.tri_obj.shape[0]  # the Bevel mesh's triangles (deformables, last, have no Bevel)
        self.bf, self.bevel_flat_stats = build_bevel_flat(r.ps.shade, r.ps.bump, n_tris, dev,
                                                          self.bevel_on and self.opts.bevel_flat)

    def _init_spec(self) -> None:
        """Material-specialised shade kernels, one per material range of the sorted queue."""
        self.spec = None
        if not self.opts.spec:
            return
        bev, bmp = self.r.ps.bump.bevel_on.numpy(), self.r.ps.bump.on.numpy()
        gen = list(self.r.ps.info.get("generic", [])) + [False]  # generic slots read their rows: one kernel for all
        self.spec = [self.kernels.generic if gen[k] else self.kernels.specialised(k, int(bev[k]), int(bmp[k]))
                     for k in range(self.n_mats)]
        # paths per material of the sorted queue (key_bins, right after the sort), copied to pinned host memory and read
        # when the shade kernels launch: the host waits for the sort only, the Bevel passes queue behind it
        self.hist = wp.zeros(self.n_mats + 2, dtype=wp.int32, device=self.dev)
        self.hist_host = wp.zeros(self.n_mats + 2, dtype=wp.int32, device="cpu", pinned=True)
        self.hist_ev = None

    def render(self, spp: int, prm) -> None:
        """Mean over ``spp`` samples into ``r.rgb`` / ``r.albedo`` / ``r.normal``."""
        r = self.r
        self._prm = prm
        self._reserve(spp)
        wp.launch(pixel_rng_kernel, dim=self.pix_rng.shape[0], inputs=[prm], outputs=[self.pix_rng], device=self.dev)
        stages = Stages(self._camera, self._shade, self._bevel if self.bevel_on else None,
                        self._sort_key if self.opts.sort else None, self._sorted if self.spec else None,
                        self._sss_fixup if self.has_sss else None)
        self.wf.run(spp, stages, r.rgb, r.albedo, r.normal)

    def _reserve(self, spp: int) -> None:
        """Grow the path buffers for ``spp`` samples, and the Bevel buffers with them when a material bevels."""
        self.wf.reserve(spp)
        if self.bevel_on and self._bb_paths < self.wf.n_paths:
            self.bb, self._bb_paths = alloc_bevel(self.wf.n_paths, self.dev), self.wf.n_paths

    def _camera(self, wv, n: int) -> None:
        r = self.r
        wp.launch(wf_camera, dim=n, inputs=[self.wf.ps, wv, r.cams, self._prm, r.ftab, r.sobol, self.pix_rng],
                  device=self.dev)

    def _shade_args(self, wv, q_in, first, q_out, q_sh, counts, off: int) -> list:
        r, ps = self.r, self.r.ps
        return [self.wf.ps, wv, q_in, int(first), q_out, q_sh, counts, r.sc.geo, ps.shade, ps.bump, ps.lights, r.cams,
                self._prm, r.p2, r.sobol, ps.lut, ps.images, self.pix_rng, self.bb, off]

    def _shade(self, wv, n, q_in, q_out, q_sh, counts, first) -> None:
        bd = self.r.block_dim
        if self.spec is None or not self.wf.sorted_now:
            wp.launch(self.kernels.generic, dim=n, inputs=self._shade_args(wv, q_in, first, q_out, q_sh, counts, 0),
                      device=self.dev, block_dim=bd)
            return
        wp.synchronize_event(self.hist_ev)
        for kern, off, count in spec_launches(self.hist_host.numpy().tolist(), self.n_mats, self.opts.spec_min):
            k = self.kernels.generic if kern < -1 else (self.kernels.escape if kern == -1 else self.spec[kern])
            wp.launch(k, dim=count, inputs=self._shade_args(wv, q_in, first, q_out, q_sh, counts, off),
                      device=self.dev, block_dim=bd)

    def _sss_fixup(self, wv, n, q_in, first) -> None:
        wp.launch(wf_sss_fixup, dim=n, inputs=[self.wf.ps, q_in, int(first)], device=self.dev)

    def _bevel(self, wv, n, q_in, first, counts) -> None:
        r, wf = self.r, self.wf
        q_bv = wf.q[2]  # the shadow queue is free until the shade kernel runs
        wp.launch(wf_bevel_prepass, dim=n, inputs=[wf.ps, wv, q_in, int(first), q_bv, counts, r.sc.geo, r.ps.shade,
                                                   r.ps.bump, self._prm, r.sobol, self.pix_rng, self.bb, self.bf],
                  device=self.dev)
        wf.mark("bevel_rays")
        self.tracer.trace_bevel(n, q_bv, self.bb, n_dev=counts[2:3], anyhit=self.opts.bevel == "anyhit")
        wf.mark("bevel_trace")

    def _sort_key(self, wv, n, q_in, keys) -> None:
        wp.launch(sort_key_kernel, dim=n, inputs=[self.wf.ps, q_in, self.r.ps.shade, self.n_mats, self.key_shift],
                  outputs=[keys], device=self.dev)

    def _sorted(self, n, keys) -> None:
        """Material ranges for the specialised shade kernels, read when they launch."""
        wp.launch(key_bins, dim=self.n_mats + 2, inputs=[keys, n, self.key_shift], outputs=[self.hist], device=self.dev)
        wp.copy(self.hist_host, self.hist)
        self.hist_ev = wp.record_event()
