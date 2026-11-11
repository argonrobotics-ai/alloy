# SPDX-License-Identifier: Apache-2.0
"""CT-10: SDF memory per part ≤ 64 MB and per scene ≤ 1 GB (meshed peg and blind hole at the 50 µm default voxel).
The CPU part checks the §4.7 estimate; the GPU part measures the baked texture data."""

from __future__ import annotations

import pytest

from alloy.contact.sdf_cache import SDF_SCENE_BUDGET_B, check_sdf_budget
from alloy.contact.validate import SDF_PART_BUDGET_B, mesh_info, sdf_bytes_estimate
from alloy.testing.contact import PegHoleSpec, blind_hole_tris, cylinder_tris, peg_hole_scene

from peg_harness import record


def test_estimate_for_test_parts():
    g = PegHoleSpec()
    for v, t in (cylinder_tris(0.5 * g.peg_d, 0.5 * g.peg_len, g.segments), blind_hole_tris(g)):
        est = sdf_bytes_estimate(mesh_info(v, t).area, 1e-3, 5e-5)
        assert est > 0


@pytest.mark.gpu
@pytest.mark.smoke
def test_ct10_sdf_budget():
    scene = peg_hole_scene(1, device="cuda:0", mode="hydroelastic")
    total = check_sdf_budget(scene.sdf)
    row = dict(metric="CT-10", parts_mb=[i.nbytes / 2**20 for i in scene.sdf],
               estimate_mb=[i.estimate_bytes / 2**20 for i in scene.sdf], scene_mb=total / 2**20,
               voxel_m=scene.spec.voxel_size)
    record("ct10", row)
    assert all(i.nbytes <= SDF_PART_BUDGET_B for i in scene.sdf), row
    assert total <= SDF_SCENE_BUDGET_B, row
