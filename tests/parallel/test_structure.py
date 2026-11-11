# SPDX-License-Identifier: Apache-2.0
"""classify_structure (SPEC §3.9 v1 scope, parallel §4.4): chain, Franka tree and box pile labels."""

from __future__ import annotations

import pytest

from alloy._vendor import newton
from alloy.parallel import rung_arm, rung_scenes
from alloy.parallel.structure import classify_structure
from alloy.parallel.types import StructureClass


def finalize(template: newton.ModelBuilder, n: int = 2) -> newton.Model:
    main = newton.ModelBuilder()
    main.replicate(template, copy_count=n)
    return main.finalize(device="cpu")


def test_20_link_chain_is_chain_pcr_and_unavailable():
    plan = classify_structure(finalize(rung_scenes.cable_chain(20)[0]))
    (isl,) = plan.islands
    assert isl.cls is StructureClass.CHAIN_PCR and isl.nv == 60 and isl.chain_len == 20 and isl.depth == 20
    assert not isl.available and plan.unavailable == ("chain_pcr",)
    assert plan.recommended_solver == "mujoco"


def test_franka_tree_is_dense_small():
    b = newton.ModelBuilder()
    rung_arm.panda_arm(b)
    plan = classify_structure(finalize(b))
    (isl,) = plan.islands
    # 7 arm hinges + 2 finger slides; link0 is welded to the world, so the tree is link1..hand + 2 fingers
    assert isl.nv == 9 and isl.cls is StructureClass.DENSE_SMALL and isl.available
    assert isl.chain_len < 16 and isl.depth >= 9
    assert plan.unavailable == ()


def test_64_box_pile_is_matrix_free_colored():
    plan = classify_structure(finalize(rung_scenes.box_pile(64)[0]))
    (isl,) = plan.islands  # potential-contact island: every box may touch every box
    assert isl.nv == 384 and isl.cls is StructureClass.MATRIX_FREE_COLORED and not isl.available
    assert isl.chain_len == 1 and isl.depth == 1


@pytest.mark.parametrize("count,cls", [(4, StructureClass.DENSE_SMALL), (16, StructureClass.DENSE_BLOCKED),
                                       (22, StructureClass.MATRIX_FREE_COLORED)])
def test_pile_class_follows_island_size(count, cls):
    (isl,) = classify_structure(finalize(rung_scenes.box_pile(count)[0])).islands
    assert isl.nv == 6 * count and isl.cls is cls


def test_collision_groups_split_islands():
    b = newton.ModelBuilder()
    for group in (1, 2, 2):
        body = b.add_body()
        b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(collision_group=group))
    plan = classify_structure(finalize(b))
    assert sorted(i.nv for i in plan.islands) == [6, 12]


def test_parallel_unit_by_world_count():
    model = finalize(rung_scenes.box_pile(4)[0], n=8)
    assert classify_structure(model, sm_count=0).parallel_unit == "world"  # CPU
    assert classify_structure(model, sm_count=2).parallel_unit == "world"  # 8 >= 4 * 2
    assert classify_structure(model, sm_count=58).parallel_unit == "color"  # 1 island x 8 worlds < 232
    chains = newton.ModelBuilder()
    for k in range(40):  # one collision group per chain, so the chains are separate islands
        first = chains.shape_count
        chains.add_builder(rung_scenes.cable_chain(10)[0])
        chains.shape_collision_group[first:] = [k + 1] * (chains.shape_count - first)
    assert classify_structure(finalize(chains, n=8), sm_count=58).parallel_unit == "island"  # 40 x 8 >= 232
