# SPDX-License-Identifier: Apache-2.0
"""Hashes, object random and 3D Perlin / FBM noise texture (Cycles 4.5 svm/noise.h, noisetex.h, fractal_noise.h)."""
import numpy as np
import warp as wp

# ---------------------------------------------------------------------------------------------------------
# Hashes (util/hash.h, Jenkins lookup3 `final`)
# ---------------------------------------------------------------------------------------------------------
_U = wp.uint32
# Typed uint32 (GEO P5, 2026-11-11): each is above INT32_MAX, and as a plain int constant it compiled to
# `const wp::int32 v = 3735928584;` with a clang -Wconstant-conversion warning; the same bits, read through _U below.
_H1 = wp.constant(wp.uint32((0xDEADBEEF + (1 << 2) + 13) & 0xFFFFFFFF))
_H2 = wp.constant(wp.uint32((0xDEADBEEF + (2 << 2) + 13) & 0xFFFFFFFF))
_H3 = wp.constant(wp.uint32((0xDEADBEEF + (3 << 2) + 13) & 0xFFFFFFFF))


@wp.func
def _rot(x: wp.uint32, k: int) -> wp.uint32:
    return (x << _U(k)) | (x >> _U(32 - k))


@wp.func
def _jenkins_final(a: wp.uint32, b: wp.uint32, c: wp.uint32) -> wp.uint32:
    c = c ^ b
    c = c - _rot(b, 14)
    a = a ^ c
    a = a - _rot(c, 11)
    b = b ^ a
    b = b - _rot(a, 25)
    c = c ^ b
    c = c - _rot(b, 16)
    a = a ^ c
    a = a - _rot(c, 4)
    b = b ^ a
    b = b - _rot(a, 14)
    c = c ^ b
    c = c - _rot(b, 24)
    return c


@wp.func
def hash_uint(kx: wp.uint32) -> wp.uint32:
    init = _U(_H1)
    return _jenkins_final(init + kx, init, init)


@wp.func
def hash_uint2(kx: wp.uint32, ky: wp.uint32) -> wp.uint32:
    init = _U(_H2)
    return _jenkins_final(init + kx, init + ky, init)


@wp.func
def hash_uint3(kx: wp.uint32, ky: wp.uint32, kz: wp.uint32) -> wp.uint32:
    init = _U(_H3)
    return _jenkins_final(init + kx, init + ky, init + kz)


@wp.func
def uint_to_float_incl(n: wp.uint32) -> float:
    return float(n) * (1.0 / 4294967296.0)  # (float)0xFFFFFFFFu == 2^32 in float32


def _float_bits_py(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def _random_float3_offset_py(seed: float):
    """noisetex.h random_float3_offset (constant per seed; Warp has no float->bits builtin, so precomputed)."""
    out = []
    for axis in range(3):
        h = hash_uint2_py(_float_bits_py(seed), _float_bits_py(float(axis)))
        out.append(np.float32(100.0) + np.float32(np.float32(h) * np.float32(1.0 / 4294967296.0)) * np.float32(100.0))
    return wp.vec3(*[float(v) for v in out])


def hash_string(name: str) -> int:
    """util/hash.h hash_string: i = i * 37 + c (uint32, c is the byte)."""
    i = 0
    for c in name.encode("utf-8"):
        i = (i * 37 + c) & 0xFFFFFFFF
    return i


def _hash_final_py(a, b, c):
    M = 0xFFFFFFFF

    def rot(x, k):
        return ((x << k) | (x >> (32 - k))) & M

    c ^= b; c = (c - rot(b, 14)) & M
    a ^= c; a = (a - rot(c, 11)) & M
    b ^= a; b = (b - rot(a, 25)) & M
    c ^= b; c = (c - rot(b, 16)) & M
    a ^= c; a = (a - rot(c, 4)) & M
    b ^= a; b = (b - rot(a, 14)) & M
    c ^= b; c = (c - rot(b, 24)) & M
    return c


def hash_uint2_py(kx: int, ky: int) -> int:
    init = (0xDEADBEEF + (2 << 2) + 13) & 0xFFFFFFFF
    return _hash_final_py((init + kx) & 0xFFFFFFFF, (init + ky) & 0xFFFFFFFF, init)


_OFF0 = wp.constant(_random_float3_offset_py(0.0))
_OFF1 = wp.constant(_random_float3_offset_py(1.0))
_OFF2 = wp.constant(_random_float3_offset_py(2.0))
_OFF3 = wp.constant(_random_float3_offset_py(3.0))
_OFF4 = wp.constant(_random_float3_offset_py(4.0))


def object_random(name: str) -> float:
    """Object Info > Random for a non-instanced object (blender/object.cpp:331, scene/object.cpp:491):
    random_id = hash_uint2(hash_string(name), 0); value = float32(random_id) * float32(1/0xFFFFFFFF)
    with (float)0xFFFFFFFF == 2^32 in float32. `name` is the Blender object name (b_ob.name())."""
    rid = hash_uint2_py(hash_string(name), 0)
    return float(np.float32(rid) * np.float32(1.0 / 4294967296.0))


# ---------------------------------------------------------------------------------------------------------
# Perlin noise, 3D (kernel/svm/noise.h)
# ---------------------------------------------------------------------------------------------------------
@wp.func
def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@wp.func
def _negate_if(val: float, cond: int) -> float:
    if cond != 0:
        return -val
    return val


@wp.func
def _grad3(hash: wp.uint32, x: float, y: float, z: float) -> float:
    h = int(hash & _U(15))
    u = y
    if h < 8:
        u = x
    vt = z
    if h == 12 or h == 14:
        vt = x
    v = vt
    if h < 4:
        v = y
    return _negate_if(u, h & 1) + _negate_if(v, h & 2)


@wp.func
def _tri_mix(v0: float, v1: float, v2: float, v3: float, v4: float, v5: float, v6: float, v7: float,
             x: float, y: float, z: float) -> float:
    x1 = 1.0 - x
    y1 = 1.0 - y
    z1 = 1.0 - z
    return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) + z * (
        y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x))


@wp.func
def perlin_3d(x: float, y: float, z: float) -> float:
    flx = wp.floor(x)
    fly = wp.floor(y)
    flz = wp.floor(z)
    X = int(flx)
    Y = int(fly)
    Z = int(flz)
    fx = x - flx
    fy = y - fly
    fz = z - flz
    u = _fade(fx)
    v = _fade(fy)
    w = _fade(fz)
    ux = _U(X)
    uy = _U(Y)
    uz = _U(Z)
    ux1 = _U(X + 1)
    uy1 = _U(Y + 1)
    uz1 = _U(Z + 1)
    return _tri_mix(
        _grad3(hash_uint3(ux, uy, uz), fx, fy, fz),
        _grad3(hash_uint3(ux1, uy, uz), fx - 1.0, fy, fz),
        _grad3(hash_uint3(ux, uy1, uz), fx, fy - 1.0, fz),
        _grad3(hash_uint3(ux1, uy1, uz), fx - 1.0, fy - 1.0, fz),
        _grad3(hash_uint3(ux, uy, uz1), fx, fy, fz - 1.0),
        _grad3(hash_uint3(ux1, uy, uz1), fx - 1.0, fy, fz - 1.0),
        _grad3(hash_uint3(ux, uy1, uz1), fx, fy - 1.0, fz - 1.0),
        _grad3(hash_uint3(ux1, uy1, uz1), fx - 1.0, fy - 1.0, fz - 1.0),
        u, v, w)


@wp.func
def _precision_wrap(c: float) -> float:
    # fmod(p, 100000) + 0.5 * (|p| >= 1e6)
    corr = 0.0
    if wp.abs(c) >= 1000000.0:
        corr = 0.5
    return wp.mod(c, 100000.0) + corr


@wp.func
def snoise_3d(p: wp.vec3) -> float:
    """Signed noise in [-1, 1] (noise_scale3 = 0.9820)."""
    return 0.9820 * perlin_3d(_precision_wrap(p[0]), _precision_wrap(p[1]), _precision_wrap(p[2]))


@wp.func
def noise_3d(p: wp.vec3) -> float:
    return 0.5 * snoise_3d(p) + 0.5


@wp.func
def noise_fbm_3d(p: wp.vec3, detail: float, roughness: float, lacunarity: float, normalize: int) -> float:
    """fractal_noise.h noise_fbm(float3, ...). detail already clamped to [0, 15]."""
    fscale = float(1.0)
    amp = float(1.0)
    maxamp = float(0.0)
    s = float(0.0)
    n = int(detail)
    for i in range(n + 1):
        t = snoise_3d(fscale * p)
        s += t * amp
        maxamp += amp
        amp *= roughness
        fscale *= lacunarity
    rmd = detail - wp.floor(detail)
    if rmd != 0.0:
        t2 = snoise_3d(fscale * p)
        sum2 = s + t2 * amp
        if normalize != 0:
            a = 0.5 * s / maxamp + 0.5
            b = 0.5 * sum2 / (maxamp + amp) + 0.5
            return a * (1.0 - rmd) + b * rmd
        return s * (1.0 - rmd) + sum2 * rmd
    if normalize != 0:
        return 0.5 * s / maxamp + 0.5
    return s


@wp.func
def tex_noise_3d(vector: wp.vec3, scale: float, detail: float, roughness: float, lacunarity: float,
                 distortion: float, normalize: int):
    """Noise Texture node, 3D, FBM (svm_node_tex_noise + noise_texture_3d). Returns (Fac, Color).
    Offset/gain are unused by FBM. `vector` is the node's Vector input (unscaled)."""
    d = wp.clamp(detail, 0.0, 15.0)
    r = wp.max(roughness, 0.0)
    p = vector * scale
    if distortion != 0.0:
        p = p + wp.vec3(
            snoise_3d(p + _OFF0) * distortion,
            snoise_3d(p + _OFF1) * distortion,
            snoise_3d(p + _OFF2) * distortion)
    value = noise_fbm_3d(p, d, r, lacunarity, normalize)
    color = wp.vec3(value,
                    noise_fbm_3d(p + _OFF3, d, r, lacunarity, normalize),
                    noise_fbm_3d(p + _OFF4, d, r, lacunarity, normalize))
    return value, color


