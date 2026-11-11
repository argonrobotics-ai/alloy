# SPDX-License-Identifier: Apache-2.0
"""Ordered gravity compensation repeats on the GPU (coordinator ruling 2026-11-06 06:23Z, WP5).

The arm of ``test_mjw_route`` with gravcomp on all four bodies, plus two free bodies on a plane, at 512 worlds on
CUDA, on Alloy's ordered route. Two solvers built from the same MjData step 30 times, and ``qfrc_gravcomp``, ``qpos``
and ``qvel`` must be bitwise equal after every step. MuJoCo Warp's own ``passive._gravity_force`` adds each body's
share with a float atomic_add, which C16's entry showed racing (docs/notes/wp5-mjw-determinism.md, round 6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy._vendor import mujoco_warp as mjw
from alloy.precision import mjw_launch, mjw_route

sys.path.insert(0, str(Path(__file__).parent))
from test_mjw_route import SolverMuJoCo, _xml  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.det]
WORLDS, STEPS = 512, 30


@pytest.fixture(autouse=True)
def cuda_and_clean():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device (gpu marker)")
    with wp.ScopedDevice("cuda:0"):
        yield
    mjw_route.uninstall()


def test_gravcomp_arm_repeats_on_the_route() -> None:
    a, b = (SolverMuJoCo(_xml(gravcomp=1), nworld=WORLDS) for _ in range(2))
    assert mjw_route.enable_route(a) is None and mjw_route.enable_route(b) is None
    before = mjw_launch.substituted_calls().get("passive._gravity_force", 0)
    for k in range(STEPS):
        mjw.step(a.mjw_model, a.mjw_data)
        mjw.step(b.mjw_model, b.mjw_data)
        for name in ("qfrc_gravcomp", "qpos", "qvel"):
            x, y = getattr(a.mjw_data, name).numpy(), getattr(b.mjw_data, name).numpy()
            assert x.tobytes() == y.tobytes(), f"{name} differs at step {k}: {np.count_nonzero(x != y)} entries"
    assert mjw_launch.substituted_calls().get("passive._gravity_force", 0) == before + 2 * STEPS
    assert np.count_nonzero(a.mjw_data.qfrc_gravcomp.numpy()) > 0
