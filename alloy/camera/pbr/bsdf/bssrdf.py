# SPDX-License-Identifier: Apache-2.0
"""Principled subsurface scattering closure of Cycles 4.5 (kernel/closure/bssrdf.h, kernel/integrator/subsurface*.h,
tag v4.5.0): the ``RANDOM_WALK`` method (Principled's default), its setup and the volume coefficients of the walk.

The closure itself has no BSDF: when it is picked, the path enters the object through a GGX-refracted entry bounce
and random-walks inside (``pbr/subsurface.py``) until it leaves; the exit point is lit as a white Lambertian surface.
After a diffuse bounce (PATH_RAY_DIFFUSE_ANCESTOR), or for colour channels with a radius below BSSRDF_MIN_RADIUS,
Cycles replaces it with a plain diffuse closure (``sss_diffuse`` here).

Example (inside a Warp function)::

    b, d = bssrdf_setup(bssrdf_new(base * sss_weight * w), n, base, radius * scale, alpha, ior, aniso, ancestor)
"""
import warp as wp

from .lobes import Diffuse
from .util import CLOSURE_WEIGHT_CUTOFF, average3, vec3s

BSSRDF_MIN_RADIUS = wp.constant(1e-8)
LABEL_SUBSURFACE_SCATTER = wp.constant(128)


@wp.struct
class Bssrdf:
    active: int           # 1: random walk when picked; 0: none (or replaced by the diffuse fallback)
    weight: wp.vec3
    sample_weight: float  # |average(weight)| x (channels with a usable radius), as bssrdf_setup
    n: wp.vec3
    radius: wp.vec3       # mean free path per channel after bssrdf_setup_radius (x 1 / 4 pi)
    albedo: wp.vec3       # clamped base colour
    alpha: float          # GGX alpha of the entry bounce (roughness^2)
    ior: float            # entry IOR, clamped to [1.01, 3.8]
    anisotropy: float     # Henyey-Greenstein g, clamped to [0, 0.9]


@wp.func
def bssrdf_new(w_in: wp.vec3) -> Bssrdf:
    """bssrdf_alloc: inactive below CLOSURE_WEIGHT_CUTOFF."""
    b = Bssrdf()
    b.weight = w_in
    b.sample_weight = wp.abs(average3(w_in))
    b.active = wp.where(b.sample_weight >= CLOSURE_WEIGHT_CUTOFF, 1, 0)
    return b


@wp.func
def bssrdf_setup(b_in: Bssrdf, n: wp.vec3, albedo: wp.vec3, radius: wp.vec3, alpha: float, ior: float,
                 anisotropy: float, diffuse_ancestor: int):
    """bssrdf_setup for CLOSURE_BSSRDF_RANDOM_WALK_ID. Returns (bssrdf, diffuse fallback closure)."""
    b = b_in
    d = Diffuse()
    d.active = 0
    if b.active == 0:
        return b, d
    b.n = n
    b.albedo = albedo
    b.alpha = alpha
    b.ior = wp.clamp(ior, 1.01, 3.8)
    b.anisotropy = wp.clamp(anisotropy, 0.0, 0.9)
    b.radius = wp.max(radius, vec3s(0.0))
    channels = int(3)
    dw = vec3s(0.0)
    if diffuse_ancestor != 0:
        channels = 0
        dw = b.weight
    else:
        for c in range(3):
            if b.radius[c] < BSSRDF_MIN_RADIUS:
                dw[c] = b.weight[c]
                b.weight[c] = 0.0
                b.radius[c] = 0.0
                channels = channels - 1
    if channels < 3:
        d.weight = dw
        d.sample_weight = wp.abs(average3(dw))
        d.active = wp.where(d.sample_weight >= CLOSURE_WEIGHT_CUTOFF, 1, 0)
        d.n = n
    if channels > 0:
        b.sample_weight = wp.abs(average3(b.weight)) * float(channels)
        b.radius = b.radius * (0.25 * 0.31830988618379067154)
    else:
        b.active = 0
        b.sample_weight = 0.0
    return b, d


@wp.func
def _poly7(c0: float, c1: float, c2: float, c3: float, c4: float, c5: float, c6: float, c7: float, g: float) -> float:
    return c0 + g * (c1 + g * (c2 + g * (c3 + g * (c4 + g * (c5 + g * (c6 + g * c7))))))


@wp.func
def walk_remap(albedo: float, d: float, g: float) -> wp.vec2:
    """subsurface_random_walk_remap: (sigma_t, single-scattering albedo alpha) from the surface albedo, the mean
    free path d and the anisotropy g (fits from "Practical and Controllable Subsurface Scattering", Chiang 2016)."""
    a = _poly7(1.8260523782, -1.28451056436, -1.79904629312, 9.19393289202, -22.8215585862, 32.0234874259,
               -23.6264803333, 7.21067002658, g)
    b = 4.98511194385 + 0.127355959438 * wp.exp(_poly7(0.0, 31.1491581433, -201.847017512, 841.576016723,
                                                         -2018.09288505, 2731.71560286, -1935.41424244,
                                                         559.009054474, g))
    c = _poly7(1.09686102424, -0.394704063468, 1.05258115941, -8.83963712726, 28.8643230661, -46.8802913581,
               38.5402837518, -12.7181042538, g)
    dd = _poly7(0.496310210422, 0.360146581622, -2.15139309747, 17.8896899217, -55.2984010333, 82.065982243,
                -58.5106008578, 15.8478295021, g)
    e = 4.23190299701 + 0.00310603949088 * wp.exp(_poly7(0.0, 76.7316253952, -594.356773233, 2448.8834203,
                                                          -5576.68528998, 7116.60171912, -4763.54467887,
                                                          1303.5318055, g))
    f = _poly7(2.40602999408, -2.51814844609, 9.18494908356, -79.2191708682, 259.082868209, -403.613804597,
               302.85712436, -87.4370473567, g)
    blend = wp.pow(albedo, 0.25)
    alpha = (1.0 - blend) * a * wp.pow(wp.atan(b * albedo), c) + blend * dd * wp.pow(wp.atan(e * albedo), f)
    alpha = wp.clamp(alpha, 0.0, 0.999999)
    sigma_t_prime = 1.0 / wp.max(d, 1e-16)
    return wp.vec2(sigma_t_prime / (1.0 - g), alpha)


@wp.func
def walk_coefficients(albedo: wp.vec3, radius: wp.vec3, g: float, thr_in: wp.vec3):
    """subsurface_random_walk_coefficients: (sigma_t, alpha, throughput with the albedo divided out and the
    low-alpha compensation applied)."""
    sigma_t = wp.vec3(0.0)
    alpha = wp.vec3(0.0)
    thr = thr_in
    for c in range(3):
        r = walk_remap(albedo[c], radius[c], g)
        sigma_t[c] = r[0]
        alpha[c] = r[1]
        thr[c] = wp.where(albedo[c] != 0.0, thr[c] / albedo[c], 0.0)
        if alpha[c] < 0.2:
            thr[c] = thr[c] * alpha[c] / 0.2
            alpha[c] = 0.2
    return sigma_t, alpha, thr
