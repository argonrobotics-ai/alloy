# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-1 item 2 (WP12's share of alloy/scene/entries.py): a coupled MuJoCo entry with precision pairs is
built at one step per exchange; every other entry keeps today's rung (CPU)."""

from __future__ import annotations

from alloy.cards.library import CardLibrary
from alloy.scene import entries as E
from alloy.scene.assembler import SceneAssembler


def test_max_substeps_pins_only_a_coupled_mujoco_entry_with_precision_pairs():
    assert E.max_substeps("mujoco", coupled=True, has_precision=True) == 1
    assert E.max_substeps("mujoco", coupled=True, has_precision=False) == E.SUBSTEP_LADDER_MAX
    assert E.max_substeps("sph", coupled=True, has_precision=True) == E.SUBSTEP_LADDER_MAX
    assert E.max_substeps("vbd", coupled=True, has_precision=False) == E.SUBSTEP_LADDER_MAX
    assert E.max_substeps("mujoco", coupled=False, has_precision=True) == 1


def test_coupled_library_scene_without_pairs_keeps_the_ladder():
    """demo_cloth_drop (MuJoCo + VBD, no precision pair) is unchanged: both entries at the D21 ladder max."""
    lib = CardLibrary()
    built = SceneAssembler(lib).assemble(lib.scene("demo_cloth_drop"), "cpu", world_count=1)
    assert not built.precision_pairs
    assert {e.capability: e.max_substeps for e in built.entries if e.name} == {"mujoco": 64, "vbd": 64}
