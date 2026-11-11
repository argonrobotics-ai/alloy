# SPDX-License-Identifier: Apache-2.0
"""PUN-5 no tunnelling: a needle that is inside tissue without a track is reported. A needle spawned through a block
(no puncture ever happened) raises F_TUNNEL for its tip and untracked depth for its body; a needle that punctured and
advances normally reports none."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.events import F_TUNNEL
from alloy.solvers.puncture.scenes import Block, Drive, build_block

pytestmark = pytest.mark.compile_heavy
BL = Block(size=4e-3, thick=3e-3, material=TissueMaterial(fp=0.3, fcut=0.1, flen=50.0), needle_length=6e-3)


def _run(bl: Block, depth: float) -> SolverPuncture:
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(bl)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=10,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, 5e-4)
        for _ in range(round(depth / 1e-2 / 5e-4)):
            d.step(lambda t: 1e-2 * t, bl.gap, record=False)
        return s


def test_pun5_spawned_through_is_flagged() -> None:
    s = _run(replace(BL, gap=-2.0e-3), 0.5e-3)          # tip starts 2 mm inside the block
    assert int(s.tracks.flags.numpy()[0]) & int(F_TUNNEL)
    assert int(s.untracked_count.numpy()[0]) > 0 and float(s.world_untracked_depth.numpy()[0]) > 0.0


def test_pun5_normal_insertion_is_clean() -> None:
    s = _run(BL, BL.gap + 2.0e-3)
    assert int(s.tracks.punctures.numpy()[0]) == 1 and int(s.tracks.flags.numpy()[0]) == 0
    assert int(s.untracked_count.numpy()[0]) == 0 and float(s.world_untracked_depth.numpy()[0]) == 0.0
