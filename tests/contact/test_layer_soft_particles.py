# SPDX-License-Identifier: Apache-2.0
"""Soft-contact slots of the Alloy contact layer's Newton pipeline (Gemini audit S1-012, 0.1.2).

One ``collision="sdf"`` mesh makes ``contacts="auto"`` collide on Newton (``AlloyContactLayer(native="newton")``),
whose stock pipeline sized its soft buffer as every particle x shape x world, contact-free SPH and MPM particles
included: 22,022 slots for 847 water particles and 13 shapes at 2 worlds, 0 under ``contacts="mujoco"``; a 16-world
scene asked for 98.6 M (1.18 GB) and ran out of memory on an L4. The wiring now gives the layer the contact-free
particles (``no_soft``), as the fallback buffer already left them out (SCR-JOB2-4):
- water, and water with sand, beside an SDF bowl get no soft slots (the control, the old wiring: W x P x S);
- a cloth beside them keeps Newton's candidates, so the scene steps bit for bit as with the old wiring, and the water
  in the bowl with it (its solver reads no contacts).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.coupling.proxy_contacts import soft_pair_count
from alloy.integration import wiring

pytestmark = pytest.mark.timeout(1800)
W = 2


def _bowl(path: Path, r_out: float = 0.06, r_in: float = 0.048, h: float = 0.045, seg: int = 12) -> Path:
    """A closed 12-sided bowl (the audit's repro mesh), faces wound outward."""
    verts, faces = [], []
    for k in range(seg):
        c, s = math.cos(2 * math.pi * k / seg), math.sin(2 * math.pi * k / seg)
        verts += [(r_out * c, r_out * s, 0.0), (r_out * c, r_out * s, h), (r_in * c, r_in * s, h),
                  (r_in * c, r_in * s, 0.008)]
    verts += [(0.0, 0.0, 0.0), (0.0, 0.0, 0.008)]
    b_out, b_in = len(verts) - 1, len(verts)
    for k in range(seg):
        b0, b1 = 4 * k + 1, 4 * ((k + 1) % seg) + 1
        faces += [(b0, b1, b1 + 1), (b0, b1 + 1, b0 + 1), (b0 + 1, b1 + 1, b1 + 2), (b0 + 1, b1 + 2, b0 + 2),
                  (b0 + 2, b1 + 2, b1 + 3), (b0 + 2, b1 + 3, b0 + 3), (b_out, b1, b0), (b_in, b0 + 3, b1 + 3)]
    path.write_text("\n".join([f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
                              + [f"f {a} {b} {c}" for a, b, c in faces]) + "\n")
    return path


def _scene(bowl: Path, *, sand: bool = False, cloth: bool = False) -> alloy.Scene:
    s = alloy.Scene(worlds=W, dt=0.01)
    s.add_ground()
    s.add_open_box("sink", (0.30, 0.30, 0.16), wall=0.015, pos=(0.0, 0.0, 0.0), static=True)
    s.add_mesh("bowl", str(bowl), pos=(0.32, 0.0, 0.0), collision="sdf", static=True)
    s.add_liquid("water", size=(0.04, 0.04, 0.02), pos=(0.32, 0.0, 0.019))           # in the bowl, on its floor
    if sand:
        s.add_box("sand", size=(0.035, 0.035, 0.025), pos=(0.0, 0.32, 0.03), material="sand_silica_dry")
    if cloth:
        s.add_cloth("sheet", size=(0.14, 0.14), pos=(0.32, 0.0, 0.05), resolution=8, mass=0.01, self_contact=False)
    return s


def _build(scene: alloy.Scene, monkeypatch: pytest.MonkeyPatch, old: bool):
    with monkeypatch.context() as mp:
        if old:                                            # the wiring before S1-012: every particle a candidate
            mp.setattr(wiring, "contact_free_particles", lambda entries: np.zeros(0, np.int64))
        sim = alloy.build(scene, device="cpu", contacts="auto", contact_capacity=64)
    layer = sim.engine.contact_layer
    assert type(layer).__name__ == "AlloyContactLayer" and layer.native == "newton"   # the SDF bowl's route
    return sim, int(layer.contacts().soft_contact_max)


@pytest.mark.parametrize("sand", [False, True], ids=["water", "water_and_sand"])
def test_contact_free_particles_get_no_soft_slots_beside_an_sdf_mesh(sand, tmp_path, monkeypatch):
    scene = _scene(_bowl(tmp_path / "bowl.obj"), sand=sand)
    sim, slots = _build(scene, monkeypatch, old=False)
    model = sim.engine.model
    full = int(model.particle_count) * int(model.shape_count) // W          # every particle x shape, per world
    sim.close()
    old, old_slots = _build(scene, monkeypatch, old=True)
    old.close()
    print(f"soft slots: {slots} (old wiring {old_slots}; every particle x shape of a world {full})")
    assert slots == 0 and old_slots >= full > 0


def test_a_cloth_beside_an_sdf_mesh_keeps_its_contacts_bit_for_bit(tmp_path, monkeypatch):
    scene = _scene(_bowl(tmp_path / "bowl.obj"), cloth=True)
    out = {}
    for old in (False, True):
        sim, slots = _build(scene, monkeypatch, old)
        model = sim.engine.model
        cloth = np.zeros(int(model.particle_count), bool)
        for r in (sim.index.particles["sheet"],):
            for w in range(W):
                start = int(sim.index.per_world(r.start, "particle")[w])
                cloth[start:start + len(r)] = True
        sim.step(15)                                       # the sheet lands on the bowl's rim
        touching = int(sim.engine.contact_layer.contacts().soft_contact_count.numpy()[0])
        st = sim.state(["sheet", "water"])
        out[old] = (slots, soft_pair_count(model, particle_ok=cloth), st["sheet.points"].copy(),
                    st["water.points"].copy(), sim.check()["ok"], touching)
        sim.close()
    (slots, want, sheet, water, ok, touching), (slots0, _, sheet0, water0, ok0, _) = out[False], out[True]
    print(f"soft slots: {slots} (the cloth's candidates {want}; old wiring {slots0}); cloth contacts {touching}")
    assert slots == want and 0 < want < slots0 and touching > 0
    assert np.array_equal(sheet, sheet0) and np.array_equal(water, water0)
    assert ok and ok0
