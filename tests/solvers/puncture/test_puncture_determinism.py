# SPDX-License-Identifier: Apache-2.0
"""PUN-9 determinism: events, lists and gathers run in fixed index order with integer-only atomics, so a puncture
run repeats bitwise, and identical worlds of a batch end bitwise identical (per-world kernels, no cross-world sums)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import Block, Drive, build_block

pytestmark = pytest.mark.compile_heavy
BL = Block(size=4e-3, thick=3e-3, material=TissueMaterial(fp=0.3, fcut=0.1, flen=50.0))


def _run(worlds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(BL, worlds=worlds)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=10,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, 5e-4)
        for _ in range(round((BL.gap + 1.5e-3) / 1e-2 / 5e-4)):
            d.step(lambda t: 1e-2 * t, BL.gap, record=False)
        return d.s0.particle_q.numpy(), s.anchors.w.numpy(), s.tracks.punctures.numpy()


def test_pun9_repeat_is_bitwise() -> None:
    a, b = _run(2), _run(2)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert a[2].tolist() == [1, 1]


def test_pun9_identical_worlds_match() -> None:
    q, w, punct = _run(2)
    half = q.shape[0] // 2
    assert np.array_equal(q[:half], q[half:])
    A = PunctureConfig().anchors_per_world
    assert np.array_equal(w[:A], w[A:2 * A])
