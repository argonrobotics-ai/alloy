# SPDX-License-Identifier: Apache-2.0
"""D1-memory laws: the check probe is sized from MuJoCo's capacities (and grows from them when it loses contacts), the
graph out-of-memory fallback is said once and recorded."""

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
    assert not probe.lost and probe.dropped is None


def test_small_scene_keeps_newton_size():
    sim = _boxes(2)
    assert cc.probe_budget(sim.engine.built.model) is None


def test_probe_overflow_grows_from_its_budget(monkeypatch):
    """A sized probe that loses contacts grows from its budget (never rebuilt at Newton's automatic size: 12 GiB at
    1,024 worlds of 134 shapes on an L4, where the run itself fit), ends with every contact an unsized probe finds,
    says nothing when nothing is lost, and leaves nothing on the shared model."""
    from alloy.scene.steppers import ContactBudget

    sim = _boxes(cc.BUDGET_MIN_WORLDS)
    built = sim.engine.built
    saved = {k: getattr(built.model, k, None) for k in cc.MODEL_CAPACITY}
    monkeypatch.setattr(cc, "probe_budget", lambda model: None)
    ref = ContactProbe(built)                                    # Newton's automatic size: the reference count
    n_ref = int(ref.collide(sim.engine.state).rigid_contact_count.numpy()[0])
    monkeypatch.setattr(cc, "probe_budget", lambda model: ContactBudget(triangle_pairs=1, rigid_contacts=1))
    monkeypatch.setattr(cc, "sim_contacts", lambda built: 0)    # no floor from the simulation's own capacity
    made, real = [], cc.ContactProbe._make_layer
    monkeypatch.setattr(cc.ContactProbe, "_make_layer", lambda self, b: made.append(b) or real(self, b))
    probe = ContactProbe(built)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # every contact found after growing: no warning
        c = probe.collide(sim.engine.state)
    assert None not in made and made[-1].rigid_contacts > made[0].rigid_contacts == 1
    assert int(c.rigid_contact_count.numpy()[0]) == n_ref >= cc.BUDGET_MIN_WORLDS
    assert int(c.rigid_contact_max) < int(ref.contacts.rigid_contact_max)    # still under Newton's size
    assert not probe.lost and not probe.overflowed and not probe.fell_back
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
