# SPDX-License-Identifier: Apache-2.0
"""Fixed64 (SPEC §7 WP5 acceptance): order-independent sums (shuffled inputs give bitwise-equal results) that match
float64 to 1 ulp of the grid for terms on the grid; non-finite terms fail closed."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.precision.fixed64 import (
    FIXED64_EMPTY, fixed64_accumulate, fixed64_alloc, fixed64_mantissa_bits, fixed64_prepare, fixed64_sum,
    fixed64_values,
)


@pytest.mark.parametrize("n, bits", [(1, 61), (2, 60), (3, 59), (1024, 51), (1025, 50), (1_000_000, 41)])
def test_mantissa_bits(n: int, bits: int) -> None:
    # 61 - bitlen(n - 1): n terms of magnitude < 2**bits sum below 2**61 < 2**63
    assert fixed64_mantissa_bits(n) == bits
    assert n * 2**bits <= 2**61


def test_mantissa_bits_rejects_zero() -> None:
    with pytest.raises(ValueError):
        fixed64_mantissa_bits(0)


def _sums(values: np.ndarray, dest: np.ndarray, n_dest: int, device: str) -> np.ndarray:
    out = fixed64_sum(wp.array(values, dtype=wp.float32, device=device), wp.array(dest, dtype=wp.int32, device=device),
                      n_dest)
    return out.numpy()


def _grid(values: np.ndarray, dest: np.ndarray, d: int, mbits: int) -> float:
    x = np.abs(values[dest == d].astype(np.float64))
    e = math.frexp(float(x.max()))[1]  # smallest e with max|x| < 2**e
    return math.ldexp(1.0, e - mbits)


def test_shuffled_bitwise_and_one_grid_ulp(device: str, rng: np.random.Generator) -> None:
    n, n_dest = 20_000, 9
    mbits = fixed64_mantissa_bits(n)
    span = mbits - 24  # binades below the largest term that stay exact on the grid
    exps = rng.integers(-span + 2, 1, n)
    values = (rng.choice([-1.0, 1.0], n) * rng.uniform(1.0, 2.0, n) * np.ldexp(1.0, exps)).astype(np.float32)
    dest = rng.integers(0, n_dest, n).astype(np.int32)
    ref = _sums(values, dest, n_dest, device)
    for _ in range(4):
        perm = rng.permutation(n)
        got = _sums(values[perm], dest[perm], n_dest, device)
        assert np.array_equal(got.view(np.uint64), ref.view(np.uint64))
    for d in range(n_dest):
        exact = math.fsum(values[dest == d].astype(np.float64))
        assert abs(ref[d] - exact) <= _grid(values, dest, d, mbits), (d, ref[d], exact)


def test_wide_range_error_is_bounded_by_rounding(device: str, rng: np.random.Generator) -> None:
    n = 5000
    values = (rng.standard_normal(n) * np.exp(rng.uniform(-12, 12, n))).astype(np.float32)
    dest = np.zeros(n, dtype=np.int32)
    got = _sums(values, dest, 1, device)[0]
    mbits = fixed64_mantissa_bits(n)
    grid = _grid(values, dest, 0, mbits)
    exact = math.fsum(values.astype(np.float64))
    # each term rounds by <= half a grid step; the final float64 conversion adds <= half an ulp of the result
    assert abs(got - exact) <= 0.5 * grid * n + abs(exact) * 2.0**-53


def test_empty_zero_and_poison(device: str) -> None:
    values = np.array([0.0, 1.5, np.inf, -2.0, 0.0], dtype=np.float32)
    dest = np.array([0, 1, 2, 2, 3], dtype=np.int32)
    got = _sums(values, dest, 5, device)
    assert got[0] == 0.0 and got[1] == 1.5 and np.isnan(got[2]) and got[3] == 0.0 and got[4] == 0.0


def test_term_missed_by_prepare_poisons(device: str) -> None:
    f = fixed64_alloc(1, device)
    small = wp.array(np.array([1.0], np.float32), device=device)
    big = wp.array(np.array([8.0], np.float32), device=device)
    dest = wp.zeros(1, dtype=wp.int32, device=device)
    fixed64_prepare(f, small, dest)
    fixed64_accumulate(f, small, dest, 60)
    fixed64_accumulate(f, big, dest, 60)  # not covered by pass 1: fail closed, never overflow silently
    out = wp.zeros(1, dtype=wp.float64, device=device)
    fixed64_values(f, 60, out)
    assert np.isnan(out.numpy()[0])
    assert int(f.exp.numpy()[0]) != FIXED64_EMPTY


def test_signed_zero_and_subnormal(device: str) -> None:
    sub = np.float32(1e-45)  # smallest subnormal
    values = np.array([-0.0, sub, sub, -sub], dtype=np.float32)
    got = _sums(values, np.zeros(4, np.int32), 1, device)[0]
    assert got == float(sub)
