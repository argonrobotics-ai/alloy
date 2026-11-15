# SPDX-License-Identifier: Apache-2.0
"""OSS-7: the check's contact probe on a many-world scene never jumps to Newton's automatic size (at 1,024 worlds of
134 shapes on an L4 that asked 12 GiB where the run itself fit). It starts from its sized budget, at least the
simulation's own contacts per world, grows from that budget when it loses contacts, and collides the worlds in chunks
when its buffers do not fit the free device memory, with the same pair depths as one pass. CPU proxy: the free
memory and the failed allocations are faked.

L4 row (variety_1024's kitchen sample, 1,024 worlds of an arm and 134 shapes, contact_capacity=640; 11.5 s of its
physics, then a learned-denoise camera rendered twice and deleted, 20.5 GB of 22.5 GB in use, then sim.check()):
main af73866a9 ran out of memory in the probe (Failed to allocate 676331528 bytes, as in the sample's run); this change:
check ok, one rebuild (the contact reducer dropped candidates at the sized budget), no chunks needed, 21.6 GB peak.
185 s on a cold kernel cache (the probe's collision kernels compile on first use)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import alloy
import alloy.check
from alloy.cards import types as ct
from alloy.check import contacts as cc
from alloy.check import probe_chunks as pc
from alloy.check.objects import ObjectIndex
from alloy.scene.steppers import ContactBudget

WORLDS = cc.BUDGET_MIN_WORLDS
TABLE = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))
CUBE = ct.Placement("cube_beech_30", (0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 1.0))      # 3 mm into the table top


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(worlds=WORLDS, dt=0.01)
    scene.add_box("table", size=(0.6, 0.6, 0.02), pos=(0, 0, -0.01), static=True)
    for i in range(3):                                  # three boxes 2 mm into the table top, 0.1 mm more per world
        z = 0.023 - 1e-4 * np.arange(WORLDS)
        pos = np.stack([np.full(WORLDS, 0.1 * i - 0.1), np.zeros(WORLDS), z], axis=1)
        scene.add_box(f"b{i}", size=(0.05, 0.05, 0.05), pos=pos, mass=0.2)
    return alloy.build(scene, device="cpu", contact_capacity=96)


def _rows(probe, sim):
    index = ObjectIndex.build(sim.engine.built)
    return [(p.a, p.b, round(p.depth, 9), p.world, sorted(p.worlds_past(2.5e-3)))
            for p in probe.depths(sim.engine.state, index, limits=(2.5e-3,))]


def _built_layers(monkeypatch):
    """Record the budget of every full-scene pipeline the probe builds (None: Newton's automatic size)."""
    real, made = cc.ContactProbe._make_layer, []

    def make(self, budget):
        made.append(budget)
        return real(self, budget)

    monkeypatch.setattr(cc.ContactProbe, "_make_layer", make)
    return made


def test_budget_floored_at_the_sims_own_contacts(sim):
    built = sim.engine.built
    assert cc.sim_contacts(built) == 96
    probe = cc.ContactProbe(built)
    default = cc.probe_budget(built.model)
    assert default.rigid_contacts < 96 == probe.budget.rigid_contacts
    assert probe.budget.triangle_pairs == cc.TRIANGLE_PAIRS_PER_CONTACT * 96


def test_a_lossy_probe_grows_from_its_budget_never_to_newtons_size(sim, monkeypatch):
    ref = _rows(cc.ContactProbe(sim.engine.built), sim)
    # 4 contacts per world, a third of what the boxes make: two doublings (Newton reads 64 for 64 slots, no count)
    monkeypatch.setattr(cc, "probe_budget", lambda model: ContactBudget(triangle_pairs=1, rigid_contacts=4))
    monkeypatch.setattr(cc, "sim_contacts", lambda built: 0)
    made = _built_layers(monkeypatch)
    probe = cc.ContactProbe(sim.engine.built)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # growing that ends with every contact says nothing
        rows = _rows(probe, sim)
    assert None not in made and len(made) >= 2 and made[-1].rigid_contacts > made[0].rigid_contacts == 4
    assert rows == ref and len(ref) == 3 and not probe.lost and probe.dropped is None and not probe.fell_back


def test_chunks_when_the_probe_does_not_fit_free_memory(sim, monkeypatch):
    built = sim.engine.built
    ref = _rows(cc.ContactProbe(built), sim)
    budget = cc.ContactProbe(built).budget
    monkeypatch.setattr(pc, "device_free", lambda device: pc.probe_bytes(budget, 5) / pc.USABLE)   # 5 worlds fit
    probe = cc.ContactProbe(built)
    assert probe.layer is None and probe.chunks == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert _rows(probe, sim) == ref
    c = probe.collide(sim.engine.state)
    assert int(c.rigid_contact_count.numpy()[0]) == int(cc.ContactProbe(built).collide(sim.engine.state)
                                                         .rigid_contact_count.numpy()[0]) > 0


def test_a_lossy_probe_in_chunks_grows_from_the_worst_chunk(sim, monkeypatch):
    built = sim.engine.built
    ref = _rows(cc.ContactProbe(built), sim)
    small = ContactBudget(triangle_pairs=1, rigid_contacts=4)
    monkeypatch.setattr(cc, "probe_budget", lambda model: small)
    monkeypatch.setattr(cc, "sim_contacts", lambda built: 0)
    monkeypatch.setattr(pc, "device_free", lambda device: pc.probe_bytes(small, 6) / pc.USABLE)
    probe = cc.ContactProbe(built)
    assert probe.chunks is not None and len(probe.chunks) > 1
    assert _rows(probe, sim) == ref and probe.counted >= 1 and probe.budget.rigid_contacts > 4 and not probe.lost
    assert probe.chunks is not None and max(hi - lo for lo, hi in probe.chunks) < 6      # re-planned for the budget


def test_the_worst_chunk_sizes_the_growth(sim):
    """One full chunk of four among four: the need is that chunk's, scaled to the 16 worlds (summed, the 16 contacts
    for 40 slots would hide it)."""
    built, calls = sim.engine.built, []

    def need(pipeline, contacts):
        calls.append(1)
        return {"triangle_pairs": 0, "triangle_slots": 0, "rigid_contacts": 10 if len(calls) == 1 else 2,
                "rigid_slots": 10, "reducer": 0}

    ranges = [(0, 4), (4, 8), (8, 12), (12, 16)]
    _, n, lost = pc.collide_chunks(built.model, cc.ContactProbe(built).budget, ranges, sim.engine.state, need)
    assert lost and n["rigid_contacts"] == n["rigid_slots"] == 40 and pc.full(n)


def test_a_failed_allocation_collides_in_chunks(sim, monkeypatch):
    built = sim.engine.built
    ref = _rows(cc.ContactProbe(built), sim)

    def no_room(self, budget):
        raise RuntimeError("Failed to allocate 676331528 bytes on device 'cuda:0'")

    monkeypatch.setattr(cc.ContactProbe, "_make_layer", no_room)
    probe = cc.ContactProbe(built)
    assert probe.chunks is not None and len(probe.chunks) > 1 and _rows(probe, sim) == ref


def test_a_chunk_that_does_not_fit_halves_the_chunks(sim, monkeypatch):
    built = sim.engine.built
    ref = _rows(cc.ContactProbe(built), sim)
    real = pc.ChunkLayer.__init__

    def at_most_two(self, model, budget, worlds, first):
        if worlds[1] - worlds[0] > 2:
            raise RuntimeError("Warp CUDA error 2: out of memory")
        real(self, model, budget, worlds, first)

    monkeypatch.setattr(pc.ChunkLayer, "__init__", at_most_two)
    monkeypatch.setattr(pc, "device_free", lambda device: 1.0)          # chunked from the start
    probe = cc.ContactProbe(built)
    probe.chunks = [(0, WORLDS)]                                        # one chunk of all: it fails, then halves
    assert _rows(probe, sim) == ref and max(hi - lo for lo, hi in probe.chunks) <= 2


def test_check_runs_in_chunks_with_the_same_report(monkeypatch):
    card = ct.SceneCard(robots=(), objects=(TABLE, CUBE))
    clean = alloy.check.check_load(card, worlds=WORLDS, device="cpu")
    monkeypatch.setattr(pc, "device_free", lambda device: 1e5)          # about 4 worlds per chunk
    ranges, real = [], pc.collide_chunks
    monkeypatch.setattr(pc, "collide_chunks", lambda model, b, r, *a: ranges.append(r) or real(model, b, r, *a))
    chunked = alloy.check.check_load(card, worlds=WORLDS, device="cpu")
    assert ranges and len(ranges[0]) > 1
    assert chunked.checks == clean.checks and chunked.checks["overlap"] == "fail"
    assert chunked.metrics["overlap"] == clean.metrics["overlap"]
    assert [f.message for f in chunked.findings] == [f.message for f in clean.findings]


def test_plan_and_halving():
    budget = ContactBudget(triangle_pairs=2000, rigid_contacts=1000)

    class _Model:
        world_count, device = 1024, "cuda:0"

    assert pc.plan_chunks(_Model(), budget, free=1e9) is None
    assert pc.plan_chunks(_Model(), budget, free=2e8) == [(0, 512), (512, 1024)]
    assert pc.plan_chunks(_Model(), None, free=1.0) is None
    assert pc.halved([(0, 5), (5, 6)]) == [(0, 3), (3, 5), (5, 6)] and pc.halved([(0, 1), (1, 2)]) is None
