# SPDX-License-Identifier: Apache-2.0
"""CARD-6: 20 rebuilds of the same cards give an equal route report, `model_digest` and entry ownership (20/20).

Two scenes: `box_cloth_cable` (MuJoCo rigid + CHAIN + VBD cloth, two entries through the proxy) and `randomized`
(per-world draws, which are part of the digest)."""

from __future__ import annotations

import pytest

from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.scene.assembler import SceneAssembler
from alloy.testing import scene as S

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}
REBUILDS = 20


def _signature(built) -> tuple:
    rep = built.report
    decisions = tuple(sorted((n, str(d.route), d.entry, tuple(sorted(d.numbers.items())), d.scene_effects)
                             for n, d in rep.decisions.items()))
    owners = tuple((e.name, e.bodies, e.particles) for e in built.entries)
    return built.model_digest, decisions, tuple(rep.scene_effects), tuple(rep.unsupported), owners


@pytest.mark.parametrize("scene", ["box_cloth_cable", "randomized"])
def test_card6_twenty_rebuilds_are_identical(scene):
    sigs = []
    with isolated_registry(empty=True):
        for _ in range(REBUILDS):
            _, card, lib = next(s for s in S.mixed_scenes() if s[0] == scene)
            built = SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS))).assemble(card, "cpu")
            sigs.append(_signature(built))
    same = sum(1 for s in sigs if s == sigs[0])
    S.record("CARD-6.rebuilds", same / REBUILDS, "pass" if same == REBUILDS else "fail", scene=scene,
             rebuilds=REBUILDS, digest=sigs[0][0])
    assert same == REBUILDS


def test_card6_mesh_scene_rebuilds_are_identical():
    """A mesh scene (T09: STL plug, receptacle and robot meshes): two rebuilds give the same `model_digest`. Newton's
    per-build mesh ids (`shape_source_ptr`) and the unwritten tail of `bvh_shape_enabled` (`wp.empty`) made every
    rebuild's digest differ; the digest now hashes the mesh content instead."""
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    lib = CardLibrary()
    digests = []
    with isolated_registry(empty=True):
        for _ in range(2):
            asm = SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS)))
            digests.append(asm.assemble(lib.scene("task_t09"), "cpu", world_count=2).model_digest)
    assert digests[0] == digests[1]
