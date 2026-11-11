# SPDX-License-Identifier: Apache-2.0
"""Pass gate: a pinned source endpoint (effective mass 0) does not enter the per-world force residual. Its feedback
cannot move anything, and on SPH-VBD P2's cloth under water (WP11, L7) pinned proxy vertices held the per-vertex max.
The raw harvest is still recorded, and the kinematic residuals still count."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.coupling import residuals as rs


def _run(meff: list[float]) -> tuple[float, np.ndarray]:
    dev = "cpu"
    n = len(meff)
    i32 = lambda a: wp.array(a, dtype=int, device=dev)                       # noqa: E731
    v3 = lambda a: wp.array(np.asarray(a, np.float32), dtype=wp.vec3, device=dev)   # noqa: E731
    inp = rs.ResidualInputs()
    inp.gravity, inp.force_floor, inp.staggered = 9.81, 1.0e-6, 0
    raw = [[0.0, 0.0, 5.0], [0.0, 0.0, 1.0]][:n]                 # endpoint 0: a 5 N change; endpoint 1: 1 N
    res = [wp.zeros(1, dtype=float, device=dev) for _ in range(3)]
    f_raw = wp.zeros(n, dtype=wp.vec3, device=dev)
    zeros = v3(np.zeros((n, 3)))
    wp.launch(rs.particle_residual_kernel, dim=n, device=dev,
              inputs=[inp, wp.zeros(1, dtype=wp.int32, device=dev), i32(range(n)), i32(range(n)), i32(range(n)),
                      i32([0] * n), wp.array(meff, dtype=float, device=dev), wp.zeros(1, dtype=wp.int32, device=dev),
                      zeros, zeros, zeros, zeros, v3(raw), v3(np.zeros((n, 3))), f_raw, v3(np.zeros((n, 3))),
                      v3(np.zeros((n, 3))), *res])
    return float(res[2].numpy()[0]), f_raw.numpy()


def test_pinned_endpoints_do_not_gate_the_force_residual() -> None:
    both, _ = _run([0.1, 0.1])
    pinned_first, f_raw = _run([0.0, 0.1])
    assert both == np.float32(5.0 / 5.0) and pinned_first == np.float32(1.0 / 1.0) * 1.0     # 1 N over its 1 N scale
    np.testing.assert_allclose(f_raw[0], [0.0, 0.0, 5.0])                                   # still recorded
    assert _run([0.0, 0.0])[0] == 0.0
