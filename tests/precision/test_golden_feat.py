# SPDX-License-Identifier: Apache-2.0
"""The FEAT golden cases (robot randomize, VectorEnv resets, camera) run, move and repeat bit for bit on the CPU, and
the GPU golden table lists them (rc1-1110: one L4 golden row per route a FEAT slice touches)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.precision import golden, golden_feat

SERIES = ("robot_randomize_series", "vectorenv_reset_series", "camera_series")


@pytest.mark.parametrize("name", SERIES)
def test_a_feat_series_repeats_moves_and_tells_worlds_apart(name: str) -> None:
    fn = getattr(golden_feat, name)
    a, b = fn(2, 2), fn(2, 2)
    assert len(a) == 3
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "the same case gave other bits on a second run"
    assert any(not np.array_equal(a[k], a[0]) for k in (1, 2)), "nothing moved: the case digests nothing"
    assert not np.array_equal(a[-1][0], a[-1][1]), "both worlds digest the same: the per-world draws are not seen"


def test_health_bookkeeping_is_left_out_of_the_feat_digests() -> None:
    assert golden_feat._is_health("mujoco/tunnel.in_shape") and golden_feat._is_health("tunnel.fast")
    assert not golden_feat._is_health("mujoco/mjw.qacc_warmstart")


def test_the_gpu_golden_table_lists_the_feat_rows() -> None:
    names = {c.name for c in golden._gpu_cases()}
    assert {"feat_robot_randomize", "feat_vectorenv_reset", "feat_camera"} <= names
    assert all(c.series is not None for c in golden._gpu_cases() if c.name.startswith("feat_"))
