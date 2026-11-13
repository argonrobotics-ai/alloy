# SPDX-License-Identifier: Apache-2.0
"""``IntegrationProfile.contacts_auto`` (audit S2-008 of 0.1.2) changes only scenes MuJoCo's precision stream refused:
every scene that built before steps bit-identically with it on. Non-auto contacts never set it (``alloy.api._build``),
so their profile is the one they had."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.core.registry import isolated_registry
from alloy.integration import build_simulation, wiring
from alloy.testing import integration as I

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]


def _engine_run(contacts_auto: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    """The precision pair beside a VBD pad that MuJoCo's stream isolates (``contacts_for`` runs its check, which
    passes): 10 control periods."""
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

        register_builtin_solvers()
        sim = build_simulation(I.dc5_like(2, "cpu"), "standard", contacts_auto=contacts_auto)
        for _ in range(10):
            sim.step_period()
        s = sim.state
        return {"body_q": s.body_q.numpy(), "body_qd": s.body_qd.numpy(), "particle_q": s.particle_q.numpy(),
                "particle_qd": s.particle_qd.numpy()}, list(sim.parts.notes)


def test_contacts_auto_steps_an_isolated_precision_scene_bit_identically():
    on, notes_on = _engine_run(True)
    off, notes_off = _engine_run(False)
    assert notes_on == notes_off and not any("contacts auto" in n for n in notes_on)
    for key, value in off.items():
        assert np.array_equal(on[key], value), key


def _usb_beside_a_box() -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_object("usb_port", "usb_a_receptacle", pos=(0.0, 0.0, 0.02), static=True)
    sc.add_object("usb_plug", "usb_a_plug", pos=(0.08, 0.0, 0.02))
    sc.add_box("cube", size=(0.03, 0.03, 0.03), pos=(0.5, 0.0, 0.02), mass=0.05)
    return sc


def _api_run(monkeypatch: pytest.MonkeyPatch, off: bool) -> tuple[dict[str, np.ndarray], list[bool]]:
    """20 control periods; ``off``: the profile the API built before the field (contacts_auto False)."""
    seen: list[bool] = []
    base = wiring.profile

    def spy(prof, **kw):
        p = base(prof, **({**kw, "contacts_auto": False} if off else kw))
        seen.append(p.contacts_auto)
        return p

    monkeypatch.setattr(wiring, "profile", spy)
    sim = alloy.build(_usb_beside_a_box(), device="cpu", contacts="auto")
    try:
        sim.step(20)
        return {k: np.asarray(v) for k, v in sim.state().items()}, seen
    finally:
        sim.close()
        monkeypatch.undo()


def test_contacts_auto_steps_an_api_precision_scene_bit_identically(monkeypatch):
    """A precision pair beside a rigid box through ``alloy.build(contacts='auto')``, which now sets the field."""
    on, seen_on = _api_run(monkeypatch, off=False)
    off, seen_off = _api_run(monkeypatch, off=True)
    assert seen_on and all(seen_on) and seen_off and not any(seen_off)
    assert on.keys() == off.keys()
    for key, value in off.items():
        assert np.array_equal(on[key], value), key
