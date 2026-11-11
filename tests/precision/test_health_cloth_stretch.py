# SPDX-License-Identifier: Apache-2.0
"""Cloth stretch in health (SCR-WP5-17, GO 2026-11-08 14:18Z; CPU).

Before: WP2's VBD drape on a 5 mm mesh blew up (edges at 500x their rest length, 2.2 m/s, every particle finite) and
sim.check() read ok: NONFINITE needs a NaN, SPEED_FAIL 20 m/s. CLOTH_STRETCH is DEGRADED past 50 % stretch (sticky
for the run) and INVALID past 500 %.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.precision.cloth_stretch import ClothStretch


def _sheet(**build):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)
    scene.add_cloth("sheet", size=(0.4, 0.4), pos=(0, 0, 0.35), resolution=12, mass=0.05)
    return alloy.build(scene, device="cpu", **build)


def _lift(sim, vertex: int, dz: float) -> None:
    """Move one sheet vertex up by ``dz`` between two steps (what a blow-up does to a sheet, in one go)."""
    q = sim.engine.state.particle_q.numpy()
    q[vertex, 2] += dz
    sim.engine.state.particle_q.assign(q)


def test_the_measure_is_the_largest_principal_stretch_of_a_triangle() -> None:
    sim = _sheet()
    eng = sim.engine
    cs = ClothStretch(eng.model, sim.worlds)
    assert cs.update(eng.state).numpy()[0] == pytest.approx(0.0, abs=1e-6)
    q = eng.state.particle_q.numpy()
    c = q.mean(axis=0)
    q[:, 0] = c[0] + 1.8 * (q[:, 0] - c[0])                    # every triangle 80 % longer along x, y unchanged
    eng.state.particle_q.assign(q)
    assert cs.update(eng.state).numpy()[0] == pytest.approx(0.8, abs=1e-4)


@pytest.mark.law
def test_a_plain_drape_is_ok() -> None:
    sim = _sheet()
    sim.run(0.5)
    w = sim.health()["worlds"][0]
    assert "CLOTH_STRETCH" not in w["flags"] and "cloth_stretch" not in w and sim.check()["ok"]
    assert float(sim.engine.health.poll().cloth_stretch[0]) < 0.05              # a drape stretches ~0.4 %


@pytest.mark.law
def test_a_sheet_stretched_past_half_its_size_is_degraded_by_name_and_stays_flagged() -> None:
    sim = _sheet()
    sim.step(1)
    _lift(sim, 40, 0.06)                                        # ~1.7 edges: ~96 % stretch on its triangles
    sim.step(1)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "DEGRADED" and "CLOTH_STRETCH" in w["flags"]
    assert w["cloth_stretch"]["cloth"] == "sheet" and w["cloth_stretch"]["level"] == "warn"
    assert 50.0 < w["cloth_stretch"]["percent"] < 500.0 and "sheet stretched" in w["problem"]
    sim.run(0.2)                                                # relaxed again: the warning stays for the run
    assert "CLOTH_STRETCH" in sim.health()["worlds"][0]["flags"] and sim.check()["ok"]


@pytest.mark.law
def test_a_blown_sheet_is_invalid_and_check_names_it() -> None:
    sim = _sheet()
    sim.step(1)
    _lift(sim, 40, 0.5)
    sim.step(1)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and "CLOTH_STRETCH" in w["flags"] and w["cloth_stretch"]["level"] == "fail"
    assert w["problem"].startswith("sheet stretched") and ": it blew up" in w["problem"]
    assert not sim.check()["ok"]


def test_only_the_stretched_world_is_flagged_and_it_names_its_own_cloth() -> None:
    """OSS-3: the per-world name lookup past world 0 (2 worlds, two cloths; world 1's second cloth stretched)."""
    scene = alloy.Scene(dt=0.01, worlds=2)
    scene.add_ground()
    scene.add_cloth("left", size=(0.2, 0.2), pos=(-0.3, 0, 0.3), resolution=8, mass=0.02)
    scene.add_cloth("right", size=(0.2, 0.2), pos=(0.3, 0, 0.3), resolution=8, mass=0.02)
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    q = sim.engine.state.particle_q.numpy()
    per_world = len(q) // 2
    q[per_world + 64 + 27, 2] += 0.5                            # world 1, "right" (64 vertices each), an inner vertex
    sim.engine.state.particle_q.assign(q)
    sim.step(1)
    w0, w1 = sim.health()["worlds"]
    assert "CLOTH_STRETCH" not in w0["flags"] and "cloth_stretch" not in w0
    assert w1["severity"] == "INVALID" and w1["cloth_stretch"]["cloth"] == "right"


def test_under_the_precision_preset_a_blown_sheet_raises() -> None:
    from alloy.core.errors import PrecisionError

    sim = _sheet(profile="precision")
    sim.step(1)
    _lift(sim, 40, 0.5)
    with pytest.raises((PrecisionError, alloy.SimulationError)):
        sim.step(1)
        sim.engine.health.poll()
    assert "CLOTH_STRETCH" in sim.health()["worlds"][0]["flags"]


def test_a_scene_without_cloth_has_no_measure() -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.2)
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    assert sim.engine.health.ranges.cloth is None and sim.engine.health.poll().cloth_stretch is None
    assert sim.health()["worlds"][0]["flags"] == []


def test_a_soft_bodys_surface_triangles_are_not_cloth() -> None:
    """A soft body's faces are its tets' (STRAIN_RANGE judges them): a knife cut stretched them 66 %, lawfully."""
    from types import SimpleNamespace

    from alloy.precision.cloth_stretch import cloth_triangles

    model = SimpleNamespace(tri_count=3, particle_count=7, tet_count=1,
                            tri_indices=SimpleNamespace(numpy=lambda: np.array([[0, 1, 2], [4, 5, 6], [2, 3, 4]])),
                            tet_indices=SimpleNamespace(numpy=lambda: np.array([[3, 4, 5, 6]])))
    assert cloth_triangles(model).tolist() == [0]


def test_check_lists_a_blown_cloth_even_when_the_physics_checks_miss_it() -> None:
    """WP2's blown drape: INVALID by CLOTH_STRETCH alone, and check_sim listed nothing (check() read ok)."""
    from alloy.api.health import cloth_problems

    h = {"worlds": [{"severity": "OK"}, {"severity": "INVALID", "cloth_stretch": {"percent": 524.9, "cloth": "disc",
                                                                                 "level": "fail"}}]}
    (p,) = cloth_problems(h)
    assert p["world"] == 1 and p["what"] == "disc stretched 525 % past its rest size: it blew up"
    assert cloth_problems(h, [{"world": 1, "what": "disc stretched 525 % ..."}]) == []      # listed once
