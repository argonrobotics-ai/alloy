# SPDX-License-Identifier: Apache-2.0
"""Host-side tables and hashes copied from Cycles 4.5 (computed in float32 / uint32 like the C++ code).

filter_table: scene/film.cpp filter_table() + util/math_cdf.{h,cpp} util_cdf_inverted(make_symmetric=true).
hash_uint:    util/hash.h (Jenkins lookup3, one word), used for the blue-noise seed (scene/integrator.cpp:296-299).
SOBOL_BURLEY_TABLE: kernel/tables.h sobol_burley_table[4][32].
"""
from __future__ import annotations

import numpy as np

FILTER_TABLE_SIZE = 1024
M_2PI = np.float32(2.0 * np.pi)


def _bh(v: np.float32, width: np.float32) -> np.float32:
    v = M_2PI * (v / width + np.float32(0.5))
    return (np.float32(0.35875) - np.float32(0.48829) * np.cos(v) + np.float32(0.14128) * np.cos(np.float32(2) * v)
            - np.float32(0.01168) * np.cos(np.float32(3) * v))


def _gauss(v: np.float32, width: np.float32) -> np.float32:
    v = v * (np.float32(6.0) / width)
    return np.exp(np.float32(-2.0) * v * v)


def filter_table(kind: str = "BLACKMAN_HARRIS", width: float = 1.5) -> np.ndarray:
    """Inverted-CDF table (1024 floats): maps u in [0,1] to the raster offset added to the integer pixel index."""
    w = np.float32(width)
    if kind == "BLACKMAN_HARRIS":
        func, w = _bh, w * np.float32(2.0)
    elif kind == "GAUSSIAN":
        func, w = _gauss, w * np.float32(3.0)
    elif kind == "BOX":
        func = lambda v, ww: np.float32(1.0)  # noqa: E731
    else:
        raise ValueError(kind)
    res = FILTER_TABLE_SIZE
    frm, to = np.float32(0.0), w * np.float32(0.5)
    rng = to - frm
    # util_cdf_evaluate(resolution - 1, ...)
    n = res - 1
    cdf = np.zeros(n + 1, np.float32)
    for i in range(n):
        x = frm + rng * np.float32(i) / np.float32(n - 1)
        cdf[i + 1] = cdf[i] + np.abs(np.float32(func(np.float32(x), w)))
    fac = np.float32(0.0) if cdf[n] == 0 else np.float32(1.0) / cdf[n]
    cdf = (cdf * fac).astype(np.float32)
    cdf[n] = np.float32(1.0)
    # util_cdf_invert(resolution, ..., make_symmetric=true); entry res-1 stays 0 as in Cycles (vector zero-init)
    inv = np.zeros(res, np.float32)
    half = (res - 1) // 2
    size = len(cdf)
    for i in range(half + 1):
        x = np.float32(i) / np.float32(half)
        index = int(np.searchsorted(cdf, x, side="right"))  # std::upper_bound
        if index < size - 1:
            t = (x - cdf[index]) / (cdf[index + 1] - cdf[index])
        else:
            t = np.float32(0.0)
            index = size - 1
        y = ((np.float32(index) + np.float32(t)) / np.float32(res - 1)) * (np.float32(2.0) * rng)
        inv[half + i] = np.float32(0.5) * (np.float32(1.0) + y)
        inv[half - i] = np.float32(0.5) * (np.float32(1.0) - y)
    return inv


def _rot(x: int, k: int) -> int:
    x &= 0xFFFFFFFF
    return ((x << k) | (x >> (32 - k))) & 0xFFFFFFFF


def hash_uint(kx: int) -> int:
    """util/hash.h hash_uint: lookup3 'final' on (a = init + kx, b = c = init)."""
    m = 0xFFFFFFFF
    a = b = c = (0xDEADBEEF + (1 << 2) + 13) & m
    a = (a + kx) & m
    c ^= b; c = (c - _rot(b, 14)) & m  # noqa: E702
    a ^= c; a = (a - _rot(c, 11)) & m  # noqa: E702
    b ^= a; b = (b - _rot(a, 25)) & m  # noqa: E702
    c ^= b; c = (c - _rot(b, 16)) & m  # noqa: E702
    a ^= c; a = (a - _rot(c, 4)) & m  # noqa: E702
    b ^= a; b = (b - _rot(a, 14)) & m  # noqa: E702
    c ^= b; c = (c - _rot(b, 24)) & m  # noqa: E702
    return c


def hash_hp_uint(i: int) -> int:
    m = 0xFFFFFFFF
    i &= m
    i ^= i >> 16
    i = (i * 0x21F0AAAD) & m
    i ^= i >> 15
    i = (i * 0xD35A2D97) & m
    i ^= i >> 15
    return i ^ 0xE6FE3BEB


SOBOL_BURLEY_TABLE = np.array([
    [1 << k for k in range(32)],
    [0x00000001, 0x00000003, 0x00000005, 0x0000000f, 0x00000011, 0x00000033, 0x00000055, 0x000000ff,
     0x00000101, 0x00000303, 0x00000505, 0x00000f0f, 0x00001111, 0x00003333, 0x00005555, 0x0000ffff,
     0x00010001, 0x00030003, 0x00050005, 0x000f000f, 0x00110011, 0x00330033, 0x00550055, 0x00ff00ff,
     0x01010101, 0x03030303, 0x05050505, 0x0f0f0f0f, 0x11111111, 0x33333333, 0x55555555, 0xffffffff],
    [0x00000001, 0x00000003, 0x00000006, 0x00000009, 0x00000017, 0x0000003a, 0x00000071, 0x000000a3,
     0x00000116, 0x00000339, 0x00000677, 0x000009aa, 0x00001601, 0x00003903, 0x00007706, 0x0000aa09,
     0x00010117, 0x0003033a, 0x00060671, 0x000909a3, 0x00171616, 0x003a3939, 0x00717777, 0x00a3aaaa,
     0x01170001, 0x033a0003, 0x06710006, 0x09a30009, 0x16160017, 0x3939003a, 0x77770071, 0xaaaa00a3],
    [0x00000001, 0x00000003, 0x00000004, 0x0000000a, 0x0000001f, 0x0000002e, 0x00000045, 0x000000c9,
     0x0000011b, 0x000002a4, 0x0000079a, 0x00000b67, 0x0000101e, 0x0000302d, 0x00004041, 0x0000a0c3,
     0x0001f104, 0x0002e28a, 0x000457df, 0x000c9bae, 0x0011a105, 0x002a7289, 0x0079e7db, 0x00b6dba4,
     0x0100011a, 0x030002a7, 0x0400079e, 0x0a000b6d, 0x1f001001, 0x2e003003, 0x45004004, 0xc900a00a],
], dtype=np.uint64).astype(np.uint32)


def background_marginal_cdf(avg_value: float, res_x: int = 1024, res_y: int = 512) -> np.ndarray:
    """(res_y + 1, 2) float32 marginal CDF of Cycles' world importance map for a CONSTANT world (scene/light.cpp
    background_cdf + device_update_background): x = row total (avg colour * sin theta_row), y = normalised CDF;
    entry res_y holds the overall total in x and 1 in y. Constant rows make every conditional CDF uniform."""
    f32 = np.float32
    rows = np.empty(res_y, f32)
    for i in range(res_y):
        st = f32(np.sin(f32(np.pi) * (f32(i) + f32(0.5)) / f32(res_y)))
        val = f32(avg_value) * st
        y = f32(0.0)
        for _ in range(1, res_x):  # cond_cdf[j].y = cond_cdf[j-1].y + x/res_x (only the total is needed)
            y = f32(y + val / f32(res_x))
        rows[i] = f32(y + val / f32(res_x))
    marg = np.zeros((res_y + 1, 2), f32)
    marg[0] = (rows[0], 0.0)
    for i in range(1, res_y):
        marg[i, 0] = rows[i]
        marg[i, 1] = f32(marg[i - 1, 1] + marg[i - 1, 0] / f32(res_y))
    total = f32(marg[res_y - 1, 1] + marg[res_y - 1, 0] / f32(res_y))
    marg[res_y, 0] = total
    if total > 0:
        marg[1:res_y, 1] = (marg[1:res_y, 1] / total).astype(f32)
    marg[res_y, 1] = 1.0
    return marg
