# SPDX-License-Identifier: Apache-2.0
"""Random walks of the subsurface closure in isolation, for tests: the guided walk of ``pbr.subsurface`` and an
independent analog walk (isotropic, grey, no guiding or MIS) in a half-space, both entered at normal incidence.

Chiang's remap turns the surface albedo A into a single-scattering albedo; both walks then return the fraction of
the entering weight that leaves the medium again, so they must agree (Cycles' walk is the guided one)."""
from __future__ import annotations

import numpy as np
import warp as wp

from alloy.camera.pbr.bsdf.bssrdf import bssrdf_new, bssrdf_setup, walk_remap
from alloy.camera.pbr.subsurface import random_walk
from alloy.camera.scene import SceneData
from alloy.camera.tracer.geometry import Geo
from alloy.camera.tracer.gpu_scene import GpuScene
from alloy.camera.tracer.sampling import PRNG_BOUNCE_NUM, Rng, pixel_rng
from alloy.camera.tracer.tables import SOBOL_BURLEY_TABLE
from tests.camera.helpers import cube

TOP = 0.3  # the cube's top face (half size 0.3 m: a half-space for sub-millimetre free paths)


@wp.kernel
def _walks(geo: Geo, sobol: wp.array(dtype=wp.uint32), base: float, radius: float, push: float, spp: int,
           out: wp.array(dtype=float)):
    i = wp.tid()
    rng = Rng()
    rng.kind = 2
    rng.pixel = pixel_rng(2, i % 1024, i // 1024, wp.uint32(7))
    rng.seq_len = wp.uint32(spp)
    rng.mask = wp.uint32(0xFFFFFFFF)
    n = wp.vec3(0.0, 0.0, 1.0)
    b, _d = bssrdf_setup(bssrdf_new(wp.vec3(base)), n, wp.vec3(base), wp.vec3(radius), 0.25, 1.5, 0.0, 0)
    p0 = wp.vec3(0.0, 0.0, TOP)
    acc = float(0.0)
    for s in range(spp):
        w = random_walk(geo, 0, p0, p0 - n * push, -n, n, b, wp.vec3(base), rng, s, PRNG_BOUNCE_NUM * 3, sobol)
        if w.ok != 0:
            acc = acc + w.thr[0]
    out[i] = acc / float(spp)


def guided_albedo(base: float, radius: float, n: int = 1024, spp: int = 8, push: float = 1.5e-5) -> float:
    """Mean weight leaving the medium after ``n * spp`` guided walks; ``push`` = the first ray-cast's inward offset."""
    sc = GpuScene(SceneData.single_world([cube(TOP)], [(0.8, 0.8, 0.8)]), device="cpu")
    sobol = wp.array(SOBOL_BURLEY_TABLE.reshape(-1), dtype=wp.uint32, device="cpu")
    out = wp.zeros(n, dtype=float, device="cpu")
    wp.launch(_walks, dim=n, inputs=[sc.geo, sobol, base, radius, push, spp], outputs=[out], device="cpu")
    return float(out.numpy().mean())


@wp.kernel
def _remap(a: wp.array(dtype=float), out: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    out[i] = walk_remap(a[i], 1.0, 0.0)


def analog_albedo(base: float, n: int = 200000, seed: int = 1, max_bounces: int = 256) -> float:
    """The same quantity from a plain analog walk in numpy (unit sigma_t, isotropic scattering, Cycles' minimum
    single-scattering albedo of 0.2 compensated in the weight)."""
    out = wp.zeros(1, dtype=wp.vec2, device="cpu")
    wp.launch(_remap, dim=1, inputs=[wp.array([base], dtype=float, device="cpu")], outputs=[out], device="cpu")
    alpha = float(out.numpy()[0, 1])
    a = max(alpha, 0.2)
    rng = np.random.default_rng(seed)
    z, dz, w = np.zeros(n), -np.ones(n), np.ones(n)
    alive, res = np.ones(n, bool), np.zeros(n)
    for _ in range(max_bounces):
        znew = z + dz * -np.log1p(-rng.random(n))
        ex = alive & (znew >= 0.0)
        res[ex] = w[ex]
        alive &= ~ex
        if not alive.any():
            break
        z, w, dz = np.where(alive, znew, z), w * a, 1.0 - 2.0 * rng.random(n)
    return float(res.mean()) * alpha / a
