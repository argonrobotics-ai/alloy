# SPDX-License-Identifier: Apache-2.0
"""D1-memory laws: the check probe is sized from MuJoCo's capacities (and fails closed), the graph out-of-memory
fallback is said once and recorded."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.check import contacts as cc
from alloy.check.contacts import ContactProbe

G1_MIB_PER_WORLD_LIMIT = 3.0


def _bytes_allocated(fn):
    total = {"b": 0}
    orig = wp.array._init_new

    def hooked(self, dtype, shape, strides, device, pinned):
        orig(self, dtype, shape, strides, device, pinned)
        total["b"] += int(getattr(self, "capacity", 0) or 0)

    wp.array._init_new = hooked
    try:
        out = fn()
    finally:
        wp.array._init_new = orig
    return out, total["b"]


def _boxes(worlds: int):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("b", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.0505), mass=0.2)
    return alloy.build(scene, device="cpu")


def test_g1_probe_bytes_per_world_bounded():
    from alloy.cards.library import default_root

    path = default_root() / "robots" / "emb_e6_g1" / "scene.xml"
    if not path.exists():
        pytest.skip("G1 card not available")
    worlds = cc.BUDGET_MIN_WORLDS
    scene = alloy.Scene(worlds=worlds, dt=0.005)
    scene.add_robot("g1", path)
    sim = alloy.build(scene, device="cpu")
    probe, nbytes = _bytes_allocated(lambda: ContactProbe(sim.engine.built))
    assert nbytes / worlds / 2**20 < G1_MIB_PER_WORLD_LIMIT     # was 21.7 MiB per world at Newton's size
    probe.collide(sim.engine.state)
    assert not probe.fell_back


def test_small_scene_keeps_newton_size():
    sim = _boxes(2)
    assert cc.probe_budget(sim.engine.built.model) is None


def test_probe_overflow_falls_back_and_says_so(monkeypatch):
    sim = _boxes(cc.BUDGET_MIN_WORLDS)
    from alloy.scene.steppers import ContactBudget

    monkeypatch.setattr(cc, "probe_budget", lambda model: ContactBudget(triangle_pairs=1, rigid_contacts=1))
    probe = ContactProbe(sim.engine.built)
    with pytest.warns(RuntimeWarning, match="overflowed"):
        c = probe.collide(sim.engine.state)
    assert probe.fell_back
    monkeypatch.undo()
    built = sim.engine.built
    saved = {k: getattr(built.model, k, None) for k in cc.MODEL_CAPACITY}
    monkeypatch.setattr(cc, "probe_budget", lambda model: None)
    ref = ContactProbe(built)                                    # an unsized probe, as on main
    n_ref = int(ref.collide(sim.engine.state).rigid_contact_count.numpy()[0])
    # OSS-7: the rebuild is really at Newton's size (not the sized value the first pipeline published on the model)
    assert int(c.rigid_contact_max) == int(ref.contacts.rigid_contact_max) > cc.BUDGET_MIN_WORLDS
    assert int(c.rigid_contact_count.numpy()[0]) == n_ref >= cc.BUDGET_MIN_WORLDS
    assert int(probe.layer.overflow.numpy()[0]) == 0 and not probe.overflowed
    assert {k: getattr(built.model, k, None) for k in cc.MODEL_CAPACITY} == saved   # nothing left on the model


def test_oom_fallback_warns_once_and_records():
    sim = _boxes(1)
    eng = sim.engine

    class _Graph:
        def invalidate(self):
            pass

    err = RuntimeError("CUDA error: out of memory")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eng._out_of_memory("k", 4, _Graph(), err)
        eng._out_of_memory("k", 2, _Graph(), err)
    ours = [x for x in w if "step graph" in str(x.message)]
    assert len(ours) == 1
    assert "fewer worlds" in str(ours[0].message).lower()
    assert len(eng.memory_notes) == 2 and "4-exchange" in eng.memory_notes[0]
    assert sim.check()["ok"]
    assert np.isfinite(sim.state(world=0)["b.pos"]).all()
