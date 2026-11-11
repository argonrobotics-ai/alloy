# SPDX-License-Identifier: Apache-2.0
"""A J2 metal sheet through the public API (SCR-WP7-15 P3): ``add_cloth`` with a ``plastic.model: j2`` material routes to
the film with its J2 hinges (the build's readback checks their section), and the state carries
``"<sheet>.plastic_strain"``: per vertex, the largest layer ε̄p of the J2 hinges along it. A plain cloth has no such
key. CPU."""

from __future__ import annotations

import numpy as np
import pytest

import alloy

pytestmark = pytest.mark.compile_heavy


def _scene(material: str) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("sheet", size=(0.06, 0.06), pos=(0.0, 0.0, 0.02), resolution=7, thickness=2.5e-4,
                    material=material)
    return scene


def test_a_j2_cloth_is_a_film_sheet_with_a_plastic_strain_readout():
    sim = alloy.build(_scene("aluminium_6061_t6"), device="cpu")
    route = sim.routes()["sheet"]
    assert route["route"] == "cloth_film" and any("j2 sheet" in r for r in route["reasons"]), route
    st = sim.state(world=0)
    assert st["sheet.plastic_strain"].shape == (64,) and float(st["sheet.plastic_strain"].max()) == 0.0
    film = next(s for s in sim.engine.solvers if getattr(s, "plastic_strain", None) is not None)
    j2 = film._j2
    lit = np.zeros(j2.ebar_max.shape[0], np.float32)
    hinge = int(np.flatnonzero(j2.sy.numpy() > 0.0)[0])
    lit[hinge] = 0.01
    j2.ebar_max.assign(lit)                                         # the plumbing: one hinge's ε̄p reaches its line
    got = sim.state(world=0)["sheet.plastic_strain"]
    assert np.count_nonzero(got) == 2 and got.max() == pytest.approx(0.01), got      # the hinge's two edge vertices
    sim.run(0.05)
    assert np.isfinite(sim.state(world=0)["sheet.points"]).all() and sim.check()["ok"]


def test_a_plain_cloth_has_no_plastic_strain():
    sim = alloy.build(_scene("cotton_woven"), device="cpu")
    assert "sheet.plastic_strain" not in sim.state(world=0)
    film = [s for s in sim.engine.solvers if hasattr(s, "has_plastic_strain")]
    assert all(s._accel.cfg.initial_guess == "inertial" for s in film)          # off: the stock start


def test_a_small_thick_j2_cloth_is_still_a_sheet_and_hangs_without_yielding():
    """OSS-17: a 6 cm, 0.5 mm 6061 cloth routed rigid by the bending test (no points, no plastic strain); add_cloth
    asks for a sheet, so a j2 material's sheet routes cloth_film whatever its stiffness. A J2 film runs 20 VBD
    iterations: hung over a 20 mm bar at VBD's 10 the 0.25 mm sheet drooped 9 mm and yielded (ebar 1e-3, a dent from
    nothing); at 20 it droops 5.1 mm and does not yield."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("plate", size=(0.06, 0.06), pos=(0.1, 0.0, 0.02), resolution=5, thickness=5.0e-4,
                    material="aluminium_6061_t6")
    scene.add_box("bar", size=(0.02, 0.2, 0.02), pos=(-0.1, 0.0, 0.01), static=True)
    scene.add_cloth("sheet", size=(0.06, 0.06), pos=(-0.1, 0.0, 0.0215), resolution=13, thickness=2.5e-4,
                    material="aluminium_6061_t6")
    sim = alloy.build(scene, device="cpu")
    for name in ("plate", "sheet"):
        assert sim.routes()[name]["route"] == "cloth_film", sim.routes()[name]
    assert any("add_cloth" in r for r in sim.routes()["plate"]["reasons"])
    film = next(s for s in sim.engine.solvers if getattr(s, "has_plastic_strain", False))
    assert film._accel.cfg.initial_guess == "inertial" and film.iterations == 20
    sim.run(0.5)
    st = sim.state(world=0)
    assert float(st["sheet.plastic_strain"].max()) == 0.0 and sim.check()["ok"]
    assert "plate.points" in st


@pytest.mark.parametrize("material", ["paper_80gsm", "film_ldpe"])
def test_a_paper_or_polymer_cloth_builds_on_the_film(material):
    """These fell to the paste route and the build failed (CapacityError); add_cloth sends them to the film."""
    sim = alloy.build(_scene(material), device="cpu")
    assert sim.routes()["sheet"]["route"] == "cloth_film", sim.routes()["sheet"]
    sim.run(0.05)
    assert np.isfinite(sim.state(world=0)["sheet.points"]).all() and sim.check()["ok"]


@pytest.mark.timeout(900)
def test_a_dented_sheet_released_fast_stays_on_its_supports():
    """OSS-17 (b650fd2fd): from VBD's adaptive start, a sheet dented between two round supports and released at
    0.1 m/s walked sideways and climbed off them with health OK (CPU, 1.5 s: top 74 mm, x -22 mm). From the inertial
    start at 20 iterations it stays (top 53 mm, x -0.6 mm)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for name, x in (("sup_a", -0.02), ("sup_b", 0.02)):
        scene.add_cylinder(name, radius=0.005, height=0.07, pos=(x, 0.0, 0.045), rpy=(np.pi / 2, 0.0, 0.0),
                           static=True)
    scene.add_cloth("sheet", size=(0.06, 0.06), pos=(0.0, 0.0, 0.0503), resolution=13, thickness=2.5e-4,
                    material="aluminium_6061_t6")
    scene.add_sphere("pusher", 0.005, pos=(0.0, 0.0, 0.056), kinematic=True)
    sim = alloy.build(scene, device="cpu")
    sim.step(20)
    sim.drive("pusher", velocity=(0.0, 0.0, -0.05), duration=0.3)
    sim.step(30)
    sim.drive("pusher", velocity=(0.0, 0.0, 0.1), duration=0.3)
    sim.step(100)
    st = sim.state(world=0)
    p = st["sheet.points"]
    assert float(st["sheet.plastic_strain"].max()) > 0.005, "the press dents the sheet"
    assert p[:, 2].max() < 0.056 and abs(float(p[:, 0].mean())) < 0.003, (p[:, 2].max(), p[:, 0].mean())
    assert sim.check()["ok"]
