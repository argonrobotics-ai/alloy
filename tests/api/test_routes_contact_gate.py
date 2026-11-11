# SPDX-License-Identifier: Apache-2.0
"""``sim.routes()`` names WP10's opt-in contact gate (``ClockPolicy.contact_gate``, mixed-speed lever #2) on every
object of a gated coupled pair, and says nothing new under today's clock (coordinator 2026-11-08 04:01Z). The gate has
no public switch yet: here it is turned on at the engine layer, on the built coupler."""

from __future__ import annotations

import dataclasses

import pytest

import alloy
from alloy.coupling.contact_gate import gate_for


def _scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.3), resolution=6, mass=0.02)
    scene.add_box("block", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.5), mass=0.2)
    return scene


@pytest.mark.timeout(600)
def test_routes_name_the_contact_gate_only_when_it_is_on() -> None:
    sim = alloy.build(_scene(), device="cpu")
    today = sim.routes()
    assert not any("exchange_clock" in r for r in today.values())
    st = sim.engine.stepper
    st._alloy_policy = dataclasses.replace(st._alloy_policy,
                                           clock=dataclasses.replace(st._alloy_policy.clock, contact_gate=True))
    st._gate = gate_for(st)
    routes = sim.routes()
    named = {name for name, r in routes.items() if "exchange_clock" in r}
    assert named == {"sheet", "block"}, routes
    assert "contact gate" in routes["sheet"]["exchange_clock"]
    assert "coupling-contact-gate-settling" in routes["block"]["exchange_clock"]
    assert {k: {f: v for f, v in r.items() if f != "exchange_clock"} for k, r in routes.items()} == today
