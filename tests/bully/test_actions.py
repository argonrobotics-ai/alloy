# SPDX-License-Identifier: Apache-2.0
"""Knot splines and legalisation (alloy.bully.actions)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import BullyError
from alloy.bully.actions import knot_basis, knots_to_trace, legalize, tick_count, trace_to_knots


def test_tick_count():
    assert tick_count(2.0, 50.0) == 100
    with pytest.raises(BullyError):
        tick_count(0.0, 50.0)


def test_basis_partition_of_unity_and_interpolation():
    m = knot_basis(8, 99)  # knots at ticks 0, 14, 28, ..., 98
    np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-12)
    for k in range(8):
        np.testing.assert_allclose(m[14 * k], np.eye(8)[k], atol=1e-12)


def test_linear_reproduced_between_interior_knots():
    # Catmull-Rom reproduces linear data on interior segments (all four neighbours exist)
    m = knot_basis(6, 51)
    lin = np.arange(6, dtype=float)
    t = np.arange(10, 41)  # segments 1..3 of 5 (span 10 ticks)
    np.testing.assert_allclose(m[t] @ lin, t / 10.0, atol=1e-12)


def test_knot_round_trip():
    rng = np.random.default_rng(0)
    low, high = np.array([-1.0, 0.0]), np.array([1.0, 0.5])
    k = rng.uniform(0.1, 0.9, size=(8, 2))
    trace = knots_to_trace(k, low, high, 100)
    assert trace.shape == (100, 2)
    np.testing.assert_allclose(trace_to_knots(trace, low, high, 8), k, atol=1e-9)
    batch = knots_to_trace(np.stack([k, k]), low, high, 100)
    np.testing.assert_allclose(batch[1], trace)


def test_legalize_clips_and_rate_limits():
    low, high, dmax = np.array([-1.0]), np.array([1.0]), np.array([0.5])
    raw = np.array([[[0.0], [2.0], [2.0], [2.0], [-3.0]]])  # [W=1, T=5, A=1]
    out = legalize(raw, low, high, dmax, initial=np.array([0.0]))
    np.testing.assert_allclose(out.trace[0, :, 0], [0.0, 0.5, 1.0, 1.0, 0.5])
    # changed: t1 (clip + rate), t2 (clip), t3 (clip), t4 (clip + rate) -> 4 entries
    assert out.clips.tolist() == [4]
    assert out.trace.dtype == np.float32


def test_legalize_is_identity_on_a_legal_trace():
    low, high, dmax = np.full(3, -1.0), np.full(3, 1.0), np.full(3, 0.1)
    t = np.linspace(0, 1, 20)
    raw = np.stack([0.5 * np.sin(t)] * 3, axis=-1)[None]
    out = legalize(raw, low, high, dmax)
    np.testing.assert_allclose(out.trace, raw, atol=1e-7)
    assert out.clips.tolist() == [0]
    with pytest.raises(BullyError):
        legalize(raw, low, high, np.zeros(3))
