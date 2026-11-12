# SPDX-License-Identifier: Apache-2.0
"""Device-side sampling, copied from Cycles 4.5 kernel/sample/{pattern,sobol_burley,util,mapping}.h.

The default sampler is Cycles' background-render default (sampling_pattern Automatic -> BLUE_NOISE_PURE):
one Owen-scrambled Sobol-Burley sequence, each pixel taking a block of `total_spp` consecutive indices, with the
pixel's block chosen by a base-4 nested-uniform scramble of its Morton code (blue-noise error distribution).
"""
from __future__ import annotations

import warp as wp

U32_INV_EXCL = wp.constant(1.0 / 4294967808.0)
PI = wp.constant(3.14159265358979323846)
PRNG_BOUNCE_NUM = wp.constant(16)
# per-vertex dimension offsets (kernel/types.h PathTraceDimension)
PRNG_FILTER = wp.constant(0)
PRNG_TERMINATE = wp.constant(0)
PRNG_LIGHT = wp.constant(1)
PRNG_SURFACE_BSDF = wp.constant(3)

# Hash and scramble constants above INT32_MAX, as typed uint32 constants (GEO P5, 2026-11-11). Written inline,
# wp.uint32(0xD35A2D97) compiles to `const wp::int32 v = 3545902487;` before its cast, and clang warns at every
# use (-Wconstant-conversion: a flood on the first camera compile, hundreds of lines on macOS). A uint32 constant
# compiles to `3545902487u`: the same bits, no warning. Constants up to INT32_MAX stay inline.
_H_MUL2 = wp.constant(wp.uint32(0xD35A2D97))
_H_XOR = wp.constant(wp.uint32(0xE6FE3BEB))
_U32_ALL = wp.constant(wp.uint32(0xFFFFFFFF))
_GOLDEN = wp.constant(wp.uint32(0x9E3779B9))
_S1_INDEX = wp.constant(wp.uint32(0xBFF95BFE))
_S2_INDEX = wp.constant(wp.uint32(0xF8ADE99A))
_S2_DIM0 = wp.constant(wp.uint32(0xE0AAAF76))
_S2_DIM1 = wp.constant(wp.uint32(0x94964D4E))
_S3_INDEX = wp.constant(wp.uint32(0xCAA726AC))
_S3_DIM0 = wp.constant(wp.uint32(0x9E78E391))


@wp.struct
class Rng:
    kind: wp.int32  # settings.SAMPLER_*
    pixel: wp.uint32  # rng_pixel (blue noise: scrambled Morton index; sobol/pcg: per-pixel hash)
    seq_len: wp.uint32  # blue_noise_sequence_length = total spp of the image
    mask: wp.uint32  # sobol_index_mask (SOBOL_BURLEY only)


@wp.func
def reverse_bits(v: wp.uint32) -> wp.uint32:
    x = v
    x = ((x >> wp.uint32(1)) & wp.uint32(0x55555555)) | ((x & wp.uint32(0x55555555)) << wp.uint32(1))
    x = ((x >> wp.uint32(2)) & wp.uint32(0x33333333)) | ((x & wp.uint32(0x33333333)) << wp.uint32(2))
    x = ((x >> wp.uint32(4)) & wp.uint32(0x0F0F0F0F)) | ((x & wp.uint32(0x0F0F0F0F)) << wp.uint32(4))
    x = ((x >> wp.uint32(8)) & wp.uint32(0x00FF00FF)) | ((x & wp.uint32(0x00FF00FF)) << wp.uint32(8))
    return (x >> wp.uint32(16)) | (x << wp.uint32(16))


@wp.func
def hash_hp_uint(v: wp.uint32) -> wp.uint32:
    i = v
    i = i ^ (i >> wp.uint32(16))
    i = i * wp.uint32(0x21F0AAAD)
    i = i ^ (i >> wp.uint32(15))
    i = i * _H_MUL2
    i = i ^ (i >> wp.uint32(15))
    return i ^ _H_XOR


@wp.func
def hash_iqnt2d(x: wp.uint32, y: wp.uint32) -> wp.uint32:
    qx = wp.uint32(1103515245) * ((x >> wp.uint32(1)) ^ y)
    qy = wp.uint32(1103515245) * ((y >> wp.uint32(1)) ^ x)
    return wp.uint32(1103515245) * (qx ^ (qy >> wp.uint32(3)))


@wp.func
def reversed_bit_owen(v: wp.uint32, seed: wp.uint32) -> wp.uint32:
    n = v
    n = n ^ (n * wp.uint32(0x3D20ADEA))
    n = n + seed
    n = n * ((seed >> wp.uint32(16)) | wp.uint32(1))
    n = n ^ (n * wp.uint32(0x05526C56))
    n = n ^ (n * wp.uint32(0x53A22864))
    return n


@wp.func
def reversed_bit_owen_base4(v: wp.uint32, seed: wp.uint32) -> wp.uint32:
    n = v
    m = wp.uint32(0x55555555)
    n = n ^ (n * wp.uint32(0x3D20ADEA))
    n = n ^ ((n >> wp.uint32(1)) & (n << wp.uint32(1)) & m)
    n = n + seed
    n = n * ((seed >> wp.uint32(16)) | wp.uint32(1))
    n = n ^ ((n >> wp.uint32(1)) & (n << wp.uint32(1)) & m)
    n = n ^ (n * wp.uint32(0x05526C56))
    n = n ^ (n * wp.uint32(0x53A22864))
    return n


@wp.func
def expand_bits(v: wp.uint32) -> wp.uint32:
    x = v & wp.uint32(0x0000FFFF)
    x = (x ^ (x << wp.uint32(8))) & wp.uint32(0x00FF00FF)
    x = (x ^ (x << wp.uint32(4))) & wp.uint32(0x0F0F0F0F)
    x = (x ^ (x << wp.uint32(2))) & wp.uint32(0x33333333)
    x = (x ^ (x << wp.uint32(1))) & wp.uint32(0x55555555)
    return x


@wp.func
def pixel_rng(kind: wp.int32, x: wp.int32, y: wp.int32, kseed: wp.uint32) -> wp.uint32:
    """path_rng_pixel_init (pattern.h): blue noise -> nested_uniform_scramble_base4(morton2d(x, y), seed)."""
    ux = wp.uint32(x)
    uy = wp.uint32(y)
    if kind == 0:
        morton = (expand_bits(ux) << wp.uint32(1)) | expand_bits(uy)
        return reverse_bits(reversed_bit_owen_base4(reverse_bits(morton), kseed))
    return hash_iqnt2d(ux, uy) ^ kseed


@wp.func
def sobol_burley(rev_bit_index: wp.uint32, dimension: wp.int32, scramble_seed: wp.uint32,
                 table: wp.array(dtype=wp.uint32)) -> float:
    result = wp.uint32(0)
    if dimension == 0:
        result = reverse_bits(rev_bit_index)
    else:
        idx = reverse_bits(rev_bit_index)  # bit k of idx selects direction number k
        k = dimension * 32
        while idx != wp.uint32(0):
            if (idx & wp.uint32(1)) != wp.uint32(0):
                result = result ^ table[k]
            idx = idx >> wp.uint32(1)
            k += 1
    result = reverse_bits(reversed_bit_owen(result, scramble_seed))
    return wp.float32(result) * U32_INV_EXCL


@wp.func
def _seq(rng: Rng, sample: wp.int32) -> wp.vec3ui:
    """(index, seed, mask) for sobol_burley_sample_ND, as blue_noise_indexing() / the Sobol-Burley pattern."""
    if rng.kind == 0:
        return wp.vec3ui(wp.uint32(sample) + rng.pixel * rng.seq_len, wp.uint32(0), _U32_ALL)
    return wp.vec3ui(wp.uint32(sample), rng.pixel, rng.mask)


@wp.func
def _pcg(sample: wp.int32, dim: wp.int32, k: wp.int32, pixel: wp.uint32) -> float:
    h = hash_hp_uint(pixel ^ hash_hp_uint(wp.uint32(sample) * _GOLDEN + wp.uint32(dim * 4 + k)))
    return wp.float32(h) * U32_INV_EXCL


@wp.func
def rand1(rng: Rng, sample: wp.int32, dim: wp.int32, table: wp.array(dtype=wp.uint32)) -> float:
    if rng.kind == 2:
        return _pcg(sample, dim, 0, rng.pixel)
    s = _seq(rng, sample)
    seed = s[1] ^ hash_hp_uint(wp.uint32(dim))
    index = reversed_bit_owen(reverse_bits(s[0]), seed ^ _S1_INDEX) & s[2]
    return sobol_burley(index, 0, seed ^ wp.uint32(0x635C77BD), table)


@wp.func
def rand2(rng: Rng, sample: wp.int32, dim: wp.int32, table: wp.array(dtype=wp.uint32)) -> wp.vec2:
    if rng.kind == 2:
        return wp.vec2(_pcg(sample, dim, 0, rng.pixel), _pcg(sample, dim, 1, rng.pixel))
    s = _seq(rng, sample)
    seed = s[1] ^ hash_hp_uint(wp.uint32(dim))
    index = reversed_bit_owen(reverse_bits(s[0]), seed ^ _S2_INDEX) & s[2]
    return wp.vec2(sobol_burley(index, 0, seed ^ _S2_DIM0, table),
                   sobol_burley(index, 1, seed ^ _S2_DIM1, table))


@wp.func
def rand3(rng: Rng, sample: wp.int32, dim: wp.int32, table: wp.array(dtype=wp.uint32)) -> wp.vec3:
    if rng.kind == 2:
        return wp.vec3(_pcg(sample, dim, 0, rng.pixel), _pcg(sample, dim, 1, rng.pixel),
                       _pcg(sample, dim, 2, rng.pixel))
    s = _seq(rng, sample)
    seed = s[1] ^ hash_hp_uint(wp.uint32(dim))
    index = reversed_bit_owen(reverse_bits(s[0]), seed ^ _S3_INDEX) & s[2]
    return wp.vec3(sobol_burley(index, 0, seed ^ _S3_DIM0, table),
                   sobol_burley(index, 1, seed ^ wp.uint32(0x67C33241), table),
                   sobol_burley(index, 2, seed ^ wp.uint32(0x78C395C5), table))


@wp.func
def lookup_table(tab: wp.array(dtype=float), x: float) -> float:
    """kernel/util/lookup_table.h lookup_table_read (linear interpolation)."""
    size = tab.shape[0]
    xs = wp.clamp(x, 0.0, 1.0) * float(size - 1)
    index = wp.min(int(xs), size - 1)
    nindex = wp.min(index + 1, size - 1)
    t = xs - float(index)
    d0 = tab[index]
    if t == 0.0:
        return d0
    return (1.0 - t) * d0 + t * tab[nindex]


@wp.func
def make_orthonormals(n: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    a = wp.vec3(0.0)
    if n[0] != n[1] or n[0] != n[2]:
        a = wp.vec3(n[2] - n[1], n[0] - n[2], n[1] - n[0])
    else:
        a = wp.vec3(n[2] - n[1], n[0] + n[2], -n[1] - n[0])
    a = wp.normalize(a)
    return a, wp.cross(n, a)


@wp.func
def sample_uniform_disk(r: wp.vec2) -> wp.vec2:
    a = 2.0 * r[0] - 1.0
    b = 2.0 * r[1] - 1.0
    if a == 0.0 and b == 0.0:
        return wp.vec2(0.0)
    rr = float(0.0)
    phi = float(0.0)
    if a * a > b * b:
        rr = a
        phi = (0.25 * PI) * (b / a)
    else:
        rr = b
        phi = 0.5 * PI - (0.25 * PI) * (a / b)
    return wp.vec2(rr * wp.cos(phi), rr * wp.sin(phi))


@wp.func
def sample_cos_hemisphere(n: wp.vec3, r: wp.vec2) -> tuple[wp.vec3, float]:
    d = sample_uniform_disk(r)
    cos_t = wp.sqrt(wp.max(1.0 - wp.dot(d, d), 0.0))
    t, b = make_orthonormals(n)
    return d[0] * t + d[1] * b + cos_t * n, cos_t / PI


@wp.func
def sample_uniform_cone(n: wp.vec3, one_minus_cos: float, r: wp.vec2) -> wp.vec3:
    if one_minus_cos > 0.0:
        xy = sample_uniform_disk(r)
        r2 = wp.dot(xy, xy)
        cos_t = 1.0 - r2 * one_minus_cos
        xy = xy * wp.sqrt(wp.max(one_minus_cos * (2.0 - one_minus_cos * r2), 0.0))
        t, b = make_orthonormals(n)
        return xy[0] * t + xy[1] * b + cos_t * n
    return n


@wp.func
def sample_uniform_sphere(r: wp.vec2) -> wp.vec3:
    z = 1.0 - 2.0 * r[0]
    s = wp.sqrt(wp.max(1.0 - z * z, 0.0))
    phi = 2.0 * PI * r[1]
    return wp.vec3(s * wp.cos(phi), s * wp.sin(phi), z)


@wp.func
def power_heuristic(a: float, b: float) -> float:
    return (a * a) / (a * a + b * b)
