# SPDX-License-Identifier: Apache-2.0
"""Renderer: settings, lights, materials and per-image cameras -> one megakernel launch per batch, outputs on device.

Example::

    sc = GpuScene(scene_data, device="cuda:0")
    r = Renderer(sc, 128, 128, RenderSettings())  # one image per world (or pass worlds=[...])
    r.set_cameras(eye, rot, extent, clip)       # numpy (N, 3), (N, 3, 3), (N, 2), (N, 2)
    rgb, albedo, normal = r.render(spp=16)      # wp.array(vec3) shape (N, H, W), linear, row 0 = top

``RenderSettings(backend="optix")`` traces on RT cores through the wavefront (``alloy.camera.wavefront``) instead of
the megakernel; same outputs, same sampler.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..scene import Lighting
from ..settings import SAMPLER_SOBOL_BURLEY, RenderSettings
from .gpu_scene import GpuScene
from .integrator import Cams, Mats, Params, render_kernel
from .textures import fill_mats, pack_textures
from .sky import Lights
from .tables import SOBOL_BURLEY_TABLE, background_marginal_cdf, filter_table, hash_uint


def _sun_fields(lt: Lights, sun, st: RenderSettings) -> None:
    half = np.float32(0.5 * sun.angle)
    omc = float(np.float32(2.0) * np.sin(np.float32(0.5) * half) ** 2)  # light.cpp:1292
    area = float(np.pi * np.sin(half) ** 2)
    eval_fac = 1.0 / area if area > 0 else 1.0
    lt.has_sun = 1
    lt.sun_to = wp.vec3(*[float(x) for x in sun.direction_to_sun])
    lt.sun_radiance = wp.vec3(*[float(c) * eval_fac for c in sun.strength])
    lt.sun_one_minus_cos = omc
    lt.sun_cos_half, lt.sun_sin_half = float(np.cos(half)), float(np.sin(half))
    lt.sun_pdf = 1.0 / (2.0 * math.pi * omc) if omc > 0 else 1.0
    lt.sun_energy = float(np.mean(sun.strength))
    lt.sun_use_mis = int(st.sun_mis and half > 0)


def make_lights(lighting: Lighting, st: RenderSettings, device: str) -> Lights:
    """Device lights for one sun + a constant world (Cycles' conventions; see ``sky.py``)."""
    if len(lighting.suns) > 1:
        from ..errors import CameraError
        raise CameraError(f"{len(lighting.suns)} suns", "use at most one sun with the diffuse renderer")
    lt = Lights()
    lt.has_sun, lt.sun_to, lt.sun_radiance = 0, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0)
    lt.sun_one_minus_cos, lt.sun_cos_half, lt.sun_sin_half, lt.sun_pdf, lt.sun_energy = 0.0, 1.0, 0.0, 1.0, 0.0
    lt.sun_use_mis, lt.sun_visible_camera = 0, int(st.sun_visible_camera)
    for sun in lighting.suns:
        _sun_fields(lt, sun, st)
    bg = np.asarray(lighting.world_color, np.float64) * lighting.world_strength
    lt.has_bg = int(np.any(bg != 0))
    lt.bg_color = wp.vec3(*[float(c) for c in bg])
    # scene/light.cpp:287-296: a constant world is not sampled as a light (no NEE), only hit by BSDF rays
    lt.bg_is_light = int(st.world_mis and lt.has_bg and st.force_world_light)
    avg = float(np.mean(np.abs(bg)))
    lt.bg_energy = avg * math.pi * avg * st.bg_energy_scale
    lt.bg_use_map = int(st.world_map)
    lt.bg_res_x, lt.bg_res_y = 1024, 512
    lt.bg_marg = wp.array(background_marginal_cdf(avg, 1024, 512), dtype=wp.vec2, device=device)
    lt.bg_first = int(st.light_order_bg_first)
    return lt


def make_params(st: RenderSettings, width: int, height: int, spp: int, sample_offset: int, total_spp: int,
                seed: int | None = None) -> Params:
    """Kernel parameters for one batch (Cycles' kernel-side bounce limits, clamps and sampler seeds)."""
    p = Params()
    p.width, p.height = int(width), int(height)
    p.spp, p.sample_offset, p.total_spp = int(spp), int(sample_offset), int(total_spp)
    p.max_bounce, p.max_diffuse, p.min_bounce = st.kernel_bounce_limits()
    p.clamp_direct, p.clamp_indirect = st.kernel_clamps()
    p.sampler = int(st.sampler)
    seed = st.seed if seed is None else int(seed)
    seed = hash_uint(seed) if st.sampler != SAMPLER_SOBOL_BURLEY else seed  # integrator.cpp:296-303
    p.seed_hash = wp.uint32(seed)
    npot = 1 << max(0, int(total_spp - 1).bit_length())  # next_power_of_two(total_spp - 1)
    p.sobol_mask = wp.uint32(int(f"{(npot - 1) & 0xFFFFFFFF:032b}"[::-1], 2))
    p.off_abs, p.off_rel = st.ray_offset_abs, st.ray_offset_rel
    p.normal_camera = int(st.normal_space == "camera")
    p.first_centered = int(st.first_sample_centered)
    return p


def image_worlds(worlds, num_worlds: int) -> np.ndarray:
    """The world of each image (int32): ``worlds``, or one image per world when None. A world outside
    0..num_worlds-1 raises ``SimulationError``: the render kernel would read past its per-world arrays (a blank image
    reported as fine, or a crash)."""
    if worlds is None:
        return np.arange(num_worlds, dtype=np.int32)
    out = np.asarray(worlds, np.int64).reshape(-1)
    bad = out[(out < 0) | (out >= num_worlds)]
    if bad.size:
        from alloy.api.errors import SimulationError  # noqa: PLC0415 - the error Sim.state raises for the same world
        raise SimulationError(f"world {int(bad[0])} is outside 0..{num_worlds - 1}", "pass world=0 for the first")
    return out.astype(np.int32)


class Renderer:
    """Diffuse path tracer over a :class:`GpuScene`; image i looks into world ``worlds[i]`` (default: one per world)."""

    def __init__(self, scene: GpuScene, width: int, height: int, settings: RenderSettings | None = None,
                 worlds=None, block_dim: int = 32, wavefront=None):
        """``wavefront``: optional :class:`~alloy.camera.wavefront.knobs.WavefrontOptions` (backend != "warp")."""
        self.sc, self.w, self.h = scene, int(width), int(height)
        self.st = settings or RenderSettings()
        self.block_dim = int(block_dim)
        self.worlds = image_worlds(worlds, scene.num_worlds)
        self.n_img = len(self.worlds)
        dev = scene.device
        self.cams = Cams()
        self.cams.world = wp.array(self.worlds, dtype=wp.int32, device=dev)
        self.cams.eye = wp.zeros(self.n_img, dtype=wp.vec3, device=dev)
        self.cams.rot = wp.zeros(self.n_img, dtype=wp.mat33, device=dev)
        self.cams.extent = wp.zeros(self.n_img, dtype=wp.vec2, device=dev)
        self.cams.clip = wp.zeros(self.n_img, dtype=wp.vec2, device=dev)
        self.mats = Mats()
        self.mats.kind = wp.zeros(scene.num_instances, dtype=wp.int32, device=dev)
        self.mats.color = wp.array(scene.sd.inst_color[scene.order], dtype=wp.vec3, device=dev)
        fill_mats(self.mats, pack_textures(scene.sd, scene.order), dev)
        self.lights = make_lights(scene.sd.lighting, self.st, dev)
        self.ftab = wp.array(filter_table(self.st.filter_type, self.st.filter_width), dtype=float, device=dev)
        self.sobol = wp.array(SOBOL_BURLEY_TABLE.reshape(-1), dtype=wp.uint32, device=dev)
        n = self.n_img * self.w * self.h
        self.rgb, self.albedo, self.normal = (wp.zeros(n, dtype=wp.vec3, device=dev) for _ in range(3))
        self.wf = self._make_wavefront(wavefront)

    def _make_wavefront(self, options):
        """The wavefront renderer when ``settings.backend`` asks for one ("optix", "wavefront-warp", "auto"), else None."""
        from ..wavefront.tracers import make_tracer, resolve_backend
        backend = resolve_backend(self.st.backend, self.sc.device)
        if backend == "warp":
            return None
        from ..wavefront.diffuse import DiffuseWavefront
        from ..wavefront.knobs import WavefrontOptions
        return DiffuseWavefront(self, make_tracer(self.sc, backend), options or WavefrontOptions.from_env())

    def set_cameras(self, eye, rot, extent, clip) -> None:
        """Per-image camera (host arrays): eye (N, 3), rot (N, 3, 3) columns = camera x, y, z (looks along -z),
        image-plane half extents at z = 1 (N, 2), (clip_start, clip_end) (N, 2)."""
        self.cams.eye.assign(np.asarray(eye, np.float32).reshape(self.n_img, 3))
        self.cams.rot.assign(np.asarray(rot, np.float32).reshape(self.n_img, 3, 3))
        self.cams.extent.assign(np.asarray(extent, np.float32).reshape(self.n_img, 2))
        self.cams.clip.assign(np.asarray(clip, np.float32).reshape(self.n_img, 2))

    def render(self, spp: int, sample_offset: int = 0, total_spp: int | None = None,
               seed: int | None = None) -> tuple[wp.array, ...]:
        """Mean over samples [sample_offset, sample_offset + spp) of a ``total_spp``-sample image: (rgb, albedo,
        normal), each wp.array(vec3) of shape (n_images, H, W), linear, row 0 = top."""
        prm = make_params(self.st, self.w, self.h, spp, sample_offset, total_spp or spp, seed)
        if self.wf is not None:
            self.wf.render(spp, prm)
        else:
            wp.launch(render_kernel, dim=self.n_img * self.w * self.h,
                      inputs=[self.sc.geo, self.mats, self.lights, self.cams, prm, self.ftab, self.sobol],
                      outputs=[self.rgb, self.albedo, self.normal], device=self.sc.device, block_dim=self.block_dim)
        shp = (self.n_img, self.h, self.w)
        return tuple(a.reshape(shp) for a in (self.rgb, self.albedo, self.normal))
