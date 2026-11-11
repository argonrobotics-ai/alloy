# SPDX-License-Identifier: Apache-2.0
"""SCR-WP4-6/7 through ``wire`` (CPU): the fallback contact buffer is built by wiring with the profile's per-world
budget; soft-only when MuJoCo runs its own contacts and owns every rigid shape (build-time check, failing closed),
with the soft contacts of the full layer; its overflow turns every world INVALID."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.integration import build_simulation, profile, wire
from alloy.integration.wiring import _fallback_layer
from alloy.precision.types import HealthFlags
from alloy.scene.steppers import ContactBudget, NewtonContacts
from alloy.testing import integration as I
from alloy.testing import scene as S

_SOFT = ("soft_contact_particle", "soft_contact_shape", "soft_contact_body_pos", "soft_contact_normal")


def _randomized():
    return {n: (c, lib) for n, c, lib in S.mixed_scenes()}["randomized"]       # table + box (MuJoCo) + cloth (VBD)


def _cloth_onto_box(built, state) -> None:
    """Lower each world's cloth to 0.5 mm into its box's top face (box half size 0.03 m), so every world has soft
    contacts with the MuJoCo box."""
    m = built.model
    mj = next(e for e in built.entries if e.capability == "mujoco")
    bq, bw, pw = state.body_q.numpy(), m.body_world.numpy(), m.particle_world.numpy()
    q = state.particle_q.numpy()
    for b in mj.bodies:
        q[pw == bw[b], 2] = bq[b, 2] + 0.03 - 5e-4
    state.particle_q.assign(q)


def _soft_rows(layer: NewtonContacts) -> dict[str, np.ndarray]:
    c = layer.contacts()
    n = int(c.soft_contact_count.numpy()[0])
    return {f: getattr(c, f).numpy()[:n] for f in _SOFT}


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_coupled_native_mujoco_gets_the_soft_only_layer():
    card, lib = _randomized()
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 2)
        parts = wire(built, "standard")
        layer = parts.contact_layer
        assert isinstance(layer, NewtonContacts) and not layer.rigid
        assert int(layer.contacts().rigid_contact_max) == 0
        assert any("fallback contacts: soft-only, rigid contacts from MuJoCo" in n for n in parts.notes)
        full = NewtonContacts(built.model)                                  # Newton's own pipeline, rigid + soft
        state = built.model.state()
        _cloth_onto_box(built, state)
        for c in (layer, full):
            c.collide(state, c.contacts(), dt=1e-3)
        a, b = _soft_rows(full), _soft_rows(layer)
        assert a["soft_contact_particle"].size > 0
        for f in _SOFT:                                                     # same candidates, same order
            np.testing.assert_array_equal(a[f], b[f], err_msg=f)


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_rigid_owner_outside_mujoco_gets_the_full_layer():
    card, lib = _randomized()
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 1)
        mj = next(e for e in built.entries if e.capability == "mujoco")
        other = next(e for e in built.entries if e.name and e.capability != "mujoco")
        moved = mj.bodies[-1]                              # a body the check must see outside MuJoCo
        entries = tuple(dataclasses.replace(e, bodies=tuple(b for b in e.bodies if b != moved)) if e is mj else
                        dataclasses.replace(e, bodies=(*e.bodies, moved)) if e is other else e for e in built.entries)
        notes: list[str] = []
        layer = _fallback_layer(dataclasses.replace(built, entries=entries), "mujoco", profile("standard"), notes)
        assert isinstance(layer, NewtonContacts) and layer.rigid
        assert int(layer.contacts().rigid_contact_max) > 0 and "rigid + soft" in notes[-1]


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_fallback_overflow_turns_every_world_invalid():
    card, lib = _randomized()
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 2)
        sim = build_simulation(built, "standard", contact_budget=ContactBudget(soft_contacts=1))
        _cloth_onto_box(built, sim.state)
        sim.step_period()
        assert int(sim.contact_layer.overflow.numpy()[0]) > 0
        report = sim.health.poll()
        assert report.invalid_worlds == (0, 1)
        assert all(int(f) & int(HealthFlags.CONTACT_OVERFLOW) for f in report.flags)
