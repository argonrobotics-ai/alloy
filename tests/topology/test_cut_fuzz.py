# SPDX-License-Identifier: Apache-2.0
"""TOP-3: random blade strokes across pre-split tet meshes with random bond-break subsets.

Pass means: no crash, no NaN, spare-pool and event-cap overflows counted, and VBD adjacency and colouring bitwise
unchanged. The scene is 16 worlds of a cantilevered 20 x 15 x 15 mm block (gravity on, fracture strain 0.03),
pre-split every 10 mm in x and y. Each round restores every world from a WorldSnapshot that includes the
controller's state arrays, breaks a random 0-40 % subset of each world's bonds, then sweeps one blade per world
along a random line with a random edge direction for 10 steps of 1 ms. The event cap is 4 per world per step, so
overflow happens. CPU runs 1,000 strokes; the GPU nightly runs 10,000
(``alloy.testing.topology.run_cut_fuzz``).
"""

from __future__ import annotations

import pytest

from alloy.testing.topology import run_cut_fuzz


def _check(stats: dict[str, float], strokes: int) -> None:
    print("TOP-3", {k: v for k, v in stats.items()})
    assert stats["strokes"] >= strokes
    assert stats["nan_strokes"] == 0
    assert stats["vbd_arrays_unchanged"] == 1.0
    assert stats["entered_cutting"] > 0                 # blades did gate bonds into the cutting phase
    assert stats["events"] > 0 and stats["overflow"] > 0   # breaks happened and the cap was hit and counted


def test_cut_fuzz_cpu_1000(device):
    if device != "cpu":
        pytest.skip("CPU acceptance size; the GPU size is the nightly test")
    _check(run_cut_fuzz(device, 1000), 1000)


@pytest.mark.gpu
@pytest.mark.nightly
def test_cut_fuzz_gpu_10000(device):
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA")
    _check(run_cut_fuzz(device, 10_000, seed=1), 10_000)
