# SPDX-License-Identifier: Apache-2.0
"""Regression test: contact_touch_kernel must ignore negative body_world on shared bodies."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.bully import kernels as bully_kernels


def test_contact_touch_kernel_ignores_shared_negative_body_world():
    """contact_touch_kernel must not write to world_robot_touch[-1] when a shared body has body_world == -1."""
    big = wp.zeros(3, dtype=wp.int32, device="cpu")
    world_touch = big[1:]
    wp.launch(
        bully_kernels.contact_touch_kernel,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([0, 1], dtype=wp.int32, device="cpu"),
            wp.array([0, 1], dtype=wp.int32, device="cpu"),
            wp.array([-1, 0], dtype=wp.int32, device="cpu"),
            wp.zeros(2, dtype=wp.int32, device="cpu"),
            wp.zeros(2, dtype=wp.int32, device="cpu"),
            world_touch,
        ],
        device="cpu",
    )
    assert np.all(big.numpy() == 0), big.numpy()
