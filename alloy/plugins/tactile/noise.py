# SPDX-License-Identifier: Apache-2.0
"""The tactile noise layer, after Tactile Genesis (arXiv 2606.22332): crosstalk, hysteresis, drift, white noise, dead
taxels, saturation and ADC steps. Every term is off at its default. It is seeded per world, and draws depend only
on (seed, world, tick, taxel, channel), so runs repeat bit for bit.

Order per channel: x = raw + crosstalk·Σ(4 neighbours' raw); hysteresis y follows x with time constant tau_load
while x > y and tau_unload otherwise (0 = no lag); bias does a random walk of drift·sqrt(dt) per step; out = y + bias
+ noise·N(0, 1); a dead taxel reads 0; then the clip at ±saturation and rounding to lsb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class NoiseParams:
    """The layer's settings, in SI units (N, N/sqrt(s), s)."""

    noise: float = 0.0
    drift: float = 0.0
    tau_load: float = 0.0
    tau_unload: float = 0.0
    dead: float = 0.0
    crosstalk: float = 0.0
    saturation: float = 0.0
    lsb: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.noise, self.drift, self.tau_load, self.tau_unload, self.crosstalk, self.saturation,
               self.lsb) < 0.0:
            raise ValueError("tactile noise settings must be >= 0")
        if not 0.0 <= self.dead <= 1.0:
            raise ValueError(f"dead is a fraction of taxels in [0, 1], got {self.dead}")


def dead_mask(worlds: int, slots_per_world: int, fraction: float, seed: int) -> np.ndarray:
    """int32 [worlds·slots]: 1 for a dead taxel, exactly round(fraction·slots) per world, seeded per world."""
    out = np.zeros((worlds, slots_per_world), np.int32)
    n = int(round(fraction * slots_per_world))
    for w in range(worlds):
        if n:
            out[w, np.random.default_rng([seed, w]).choice(slots_per_world, n, replace=False)] = 1
    return out.reshape(-1)


def grid_neighbours(nx: int, ny: int) -> np.ndarray:
    """int32 [nx·ny·4]: the 4-neighbours of each grid taxel (row-major, j = iy·nx + ix), -1 where none."""
    out = -np.ones((ny, nx, 4), np.int32)
    for iy in range(ny):
        for ix in range(nx):
            for e, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                if 0 <= ix + dx < nx and 0 <= iy + dy < ny:
                    out[iy, ix, e] = (iy + dy) * nx + ix + dx
    return out.reshape(-1)


@wp.func
def _gauss(seed: int, offset: int) -> float:
    rs = wp.rand_init(seed, offset)
    return wp.randn(rs)


@wp.func
def _shape(v: float, saturation: float, lsb: float) -> float:
    out = v
    if saturation > 0.0:
        out = wp.clamp(out, -saturation, saturation)
    if lsb > 0.0:
        out = wp.round(out / lsb) * lsb
    return out


@wp.func
def _lag(x: float, y: float, dt: float, tau_load: float, tau_unload: float) -> float:
    tau = tau_unload
    if x > y:
        tau = tau_load
    if tau <= 0.0:
        return x
    return y + (x - y) * (1.0 - wp.exp(-dt / tau))


@wp.kernel
def apply_noise(raw: wp.array(dtype=wp.float32), neighbours: wp.array(dtype=wp.int32), n: int, pads: int,
                tick: wp.array(dtype=wp.int32), dead: wp.array(dtype=wp.int32), p: wp.vec4, q: wp.vec4, seed: int,
                dt: float, hyst: wp.array(dtype=wp.float32), bias: wp.array(dtype=wp.float32),
                out: wp.array(dtype=wp.float32)):
    """One thread per taxel slot. p = (noise, drift, tau_load, tau_unload); q = (crosstalk, saturation, lsb, -)."""
    k = wp.tid()
    g = k // n
    j = k - g * n
    w = g // pads
    base = (tick[w] * pads * n + (k - w * pads * n)) * 6
    for c in range(3):
        i = k * 3 + c
        x = raw[i]
        if q[0] > 0.0:
            for e in range(4):
                nb = neighbours[j * 4 + e]
                if nb >= 0:
                    x = x + q[0] * raw[(g * n + nb) * 3 + c]
        y = _lag(x, hyst[i], dt, p[2], p[3])
        hyst[i] = y
        b = bias[i]
        if p[1] > 0.0:
            b = b + p[1] * wp.sqrt(dt) * _gauss(seed + w * 7919, base + c * 2)
            bias[i] = b
        v = y + b
        if p[0] > 0.0:
            v = v + p[0] * _gauss(seed + w * 7919, base + c * 2 + 1)
        if dead[k] != 0:
            v = 0.0
        out[i] = _shape(v, q[1], q[2])


@wp.kernel
def advance_tick(tick: wp.array(dtype=wp.int32)):
    w = wp.tid()
    tick[w] = tick[w] + 1
