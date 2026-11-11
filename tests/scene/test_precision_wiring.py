# SPDX-License-Identifier: Apache-2.0
"""Precision pairs reach WP3's contact layer: each pair is a collision filter pair in every world (so neither the
bulk pipeline nor MuJoCo's native narrow phase collides it twice) and the layer builds on both native values."""

from __future__ import annotations

import pytest

from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.scene.assembler import SceneAssembler

SCENES = ("demo_peg_d12", "task_t09")
PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


def _assembler(lib) -> SceneAssembler:
    return SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS)))   # the USB cable self-test is slow


def _library():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    return CardLibrary()


@pytest.mark.parametrize("scene", SCENES)
def test_precision_pairs_are_filtered_in_every_world(scene):
    lib = _library()
    with isolated_registry(empty=True):
        built = _assembler(lib).assemble(lib.scene(scene), "cpu", world_count=3)
    assert built.precision_pairs
    filt = {tuple(sorted(map(int, p))) for p in built.model.shape_collision_filter_pairs}
    shapes_per_world = len(next(iter(built.handles.values())).world_stride.shape)
    for spec in built.precision_pairs:
        for w in range(3):
            pair = tuple(sorted((spec.shape_a + w * shapes_per_world, spec.shape_b + w * shapes_per_world)))
            assert pair in filt, (scene, w, pair)
    mj = getattr(built.model, "mujoco", None)
    if mj is not None and getattr(mj, "pair_geom1", None) is not None:
        rows = {tuple(sorted(map(int, g))) for g in zip(mj.pair_geom1.numpy(), mj.pair_geom2.numpy())}
        for spec in built.precision_pairs:   # an explicit MuJoCo pair bypasses the filter (T09 plug ejected, WP3)
            assert tuple(sorted((spec.shape_a, spec.shape_b))) not in rows, (scene, "explicit MuJoCo pair row")


@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_contact_layer_accepts_the_assembled_pairs(native):
    layer_mod = pytest.importorskip("alloy.contact.pipeline")
    lib = _library()
    with isolated_registry(empty=True):
        built = _assembler(lib).assemble(lib.scene("demo_peg_d12"), "cpu", world_count=2)
    layer = layer_mod.AlloyContactLayer(built.model, built.precision_pairs, native=native)
    assert len(layer.pairs) == len(built.precision_pairs)
