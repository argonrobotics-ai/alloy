# SPDX-License-Identifier: Apache-2.0
"""FAST_CONTACT laws (GAP-E2 round 3): where it is raised, that it is not sticky, and that it ignores contact gaps.

The flag counts down (64 rigid steps), so every law reads health after every period and works on the union of what it
showed. Rule (``health_tunnel``): approach per rigid step > 5 mm (a constant, not a shape's gap), a static shape that
is not a plane and thinner along the approach than twice the travel, the body's near surface ending within 5 mm of it.

F1  the 1 cm plate: a 4 m drop (11.1 mm per step) and a 2 m drop (7.8 mm) land with FAST_CONTACT, a 0.5 m drop (3 mm)
    does not; none tunnels; long after the landing the flag is gone and the world reads OK.
F2  no flag at any period: a 5 cm cube landing on the ground and on a 40 cm table at 3 m/s, a 30 cm 2 kg cube dropped
    1.5 m and 3 m on the ground, a ball dropped 3 m on a 40 cm table.
F3  a body whose shapes carry a 1e-4 m contact gap, landing at 1 m/s on a 1 cm plate: no flag (the gap is not the limit).
F4  shipped robots at rest for 0.2 s (FR3 Duo, Panda): no FAST_CONTACT and no TUNNEL at any period.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

ROUTES = ["mjc", "warp"]
R = 0.01
BITS = {"FAST_CONTACT", "TUNNEL"}


def _seen(sim, seconds: float) -> list[set[str]]:
    """Per world, the union over every period of the run of the TUNNEL / FAST_CONTACT flags health showed."""
    seen: list[set[str]] = [set() for _ in range(sim.worlds)]
    for _ in range(int(round(seconds / sim.dt))):
        sim.step(1)
        for w, row in enumerate(sim.health()["worlds"]):
            seen[w] |= BITS & set(row["flags"])
    return seen


def _build(route, monkeypatch, parts, worlds=1):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    parts(scene)
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("route", ROUTES)
def test_f1_the_1_cm_plate_flags_the_4_m_and_2_m_drops_not_the_half_metre_and_the_flag_clears(route, monkeypatch) -> None:
    def parts(scene):
        scene.add_box("plate", size=(0.5, 0.5, 0.01), pos=(0, 0, 0.3), static=True)
        scene.add_sphere("ball", R, pos=np.array([[0, 0, 0.315 + h] for h in (4.0, 2.0, 0.5)]), density=7800.0)

    sim = _build(route, monkeypatch, parts, worlds=3)
    texts: list[str] = []
    seen: list[set[str]] = [set() for _ in range(3)]
    for _ in range(300):
        sim.step(1)
        for w, row in enumerate(sim.health()["worlds"]):
            seen[w] |= BITS & set(row["flags"])
            if w == 0 and "FAST_CONTACT" in row["flags"]:
                texts.append(row["problem"])
    assert texts and 'moved' in texts[0] and '"plate"' in texts[0] and "5 mm travel limit" in texts[0], texts[:1]
    assert "FAST_CONTACT" in seen[0] and "FAST_CONTACT" in seen[1], seen
    assert "FAST_CONTACT" not in seen[2], seen
    assert not any("TUNNEL" in s for s in seen), seen                       # all three land on the plate
    rows = sim.health()["worlds"]                                          # 3 s: the 64-step hold ran out long ago
    assert all("FAST_CONTACT" not in r["flags"] for r in rows)
    assert [r["severity"] for r in rows] == ["OK"] * 3, rows


@pytest.mark.parametrize("route", ROUTES)
def test_f2_landings_on_thick_parts_and_the_bare_ground_carry_no_flag(route, monkeypatch) -> None:
    def parts(scene):
        scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(1.0, 0, 0.2), static=True)
        scene.add_box("cube_g", size=(0.05,) * 3, pos=(-1.0, 0, 0.025 + 0.46), mass=0.1)           # 3 m/s on the ground
        scene.add_box("cube_t", size=(0.05,) * 3, pos=(1.0, 0, 0.4 + 0.025 + 0.46), mass=0.1)      # 3 m/s on the table
        scene.add_box("big_1", size=(0.3,) * 3, pos=(-3.0, 0, 0.15 + 1.5), mass=2.0)
        scene.add_box("big_3", size=(0.3,) * 3, pos=(-5.0, 0, 0.15 + 3.0), mass=2.0)
        scene.add_sphere("ball", R, pos=(1.0, 0.2, 0.4 + R + 3.0), density=7800.0)              # 7.7 m/s on the table

    sim = _build(route, monkeypatch, parts)
    assert _seen(sim, 1.5) == [set()]


@pytest.mark.parametrize("route", ROUTES)
def test_f3_a_contact_gap_of_a_tenth_of_a_millimetre_is_not_the_limit(route, monkeypatch) -> None:
    """Shipped task cards carry tiny gaps; the travel limit is the constant 5 mm, so a ball landing at 1 m/s (1.25 mm per
    rigid step) on a plate whose shapes all carry a gap of 1e-4 m raises nothing. (The default gap of an ordinary shape is
    ``shape_config``'s keyword-only default, lowered here.)"""
    from alloy.scene.emitters import common

    monkeypatch.setitem(common.shape_config.__kwdefaults__, "gap", 1.0e-4)

    def parts(scene):
        scene.add_box("plate", size=(0.5, 0.5, 0.01), pos=(0, 0, 0.3), static=True)
        scene.add_sphere("ball", R, pos=(0, 0, 0.315 + 0.05), density=7800.0)                # lands at about 1 m/s

    sim = _build(route, monkeypatch, parts)
    assert float(sim.engine.model.shape_gap.numpy().max()) == pytest.approx(1.0e-4), "the gap was not lowered"
    assert _seen(sim, 0.6) == [set()]
    assert sim.state()["ball.pos"][0, 2] == pytest.approx(0.315, abs=0.01)                  # it did land on the plate


def _robot_scene(name: str):
    def parts(scene):
        from alloy.cards.library import default_root

        scene.add_robot("r", default_root() / "robots" / name / "scene.xml")
    return parts


@pytest.mark.parametrize("route, name", [("warp", "emb_e5_fr3_duo"), ("warp", "emb_e2_panda"), ("mjc", "emb_e2_panda"),
                                         ("mjc", "emb_e5_fr3_duo")])
@pytest.mark.timeout(1500)
def test_f4_a_shipped_robot_at_rest_raises_nothing_at_any_period(route, name, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(dt=0.01)
    _robot_scene(name)(scene)
    sim = alloy.build(scene, device="cpu")
    assert _seen(sim, 0.2) == [set()]
