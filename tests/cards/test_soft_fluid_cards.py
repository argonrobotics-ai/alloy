# SPDX-License-Identifier: Apache-2.0
"""Liquids and grains from the card library: the fill cards, their open box liquid_box_56, the sourced materials, the
solver each fill routes to, and a short settle on CPU (examples/soft/soft_fluid_fills.py)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.cards.library import CardLibrary
from alloy.sourcing.tags import tag_of

LIB = CardLibrary(assets=None)
INNER, WALL, DEPTH = 0.056, 0.004, 0.080        # liquid_box_56: cavity 56 x 56 x 80 mm, floor 4 mm thick
FILLS = {  # card: (solver, ml)
    "water_box_200ml": ("sph", 200), "glycerol_80_box_100ml": ("sph", 100), "oil_olive_box_150ml": ("sph", 150),
    "honey_box_150ml": ("mpm", 150), "sauce_ketchup_box_150ml": ("mpm", 150), "sauce_mayonnaise_box_150ml": ("mpm", 150),
    "sand_box_200ml": ("mpm", 200),
    "rice_box_200ml": ("mpm", 200),
    "batter_bread_box_150ml": ("mpm", 150), "batter_cake_box_150ml": ("mpm", 150), "puree_fruit_box_150ml": ("mpm", 150),
    "tomato_concentrate_box_150ml": ("mpm", 150), "yogurt_box_150ml": ("mpm", 150), "peanut_butter_box_150ml": ("mpm", 150),
}
#: shear-thinning food pastes (Herschel-Bulkley K and n on the card): material -> (density tag, has a yield stress)
HB_FOODS = {"batter_bread_wheat": ("ESTIMATE", True), "batter_cake_quinoa": ("DOC", False),
            "puree_fruit": ("ESTIMATE", False), "tomato_concentrate": ("DOC", False),
            "yogurt_plain": ("SNIPPET", True), "peanut_butter": ("ESTIMATE", True)}


CUP_R, CUP_FLOOR, CUP_Z = 0.0388, 0.0015, 0.0232   # tumbler_pp inside; water_tumbler_200ml sits at the cup's pos + CUP_Z


def box_scene(fill: str) -> alloy.Scene:
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_object("box", "liquid_box_56", static=True)
    scene.add_object("fill", fill)
    return scene


def test_box_card_has_no_errors() -> None:
    assert not [i for i in LIB.object_issues("liquid_box_56") if i.level == "error"]


@pytest.mark.parametrize("name", sorted(FILLS))
def test_fill_card_has_no_errors_and_fills_the_box(name: str) -> None:
    assert not [i for i in LIB.object_issues(name) if i.level == "error"]
    lo, hi = (np.asarray(c) for c in LIB.peek_object(name).shape.region)
    assert np.allclose(hi[:2], 0.5 * INNER) and np.allclose(lo[:2], -0.5 * INNER)    # the cavity's footprint
    assert lo[2] == pytest.approx(WALL) and hi[2] < WALL + DEPTH                       # on the floor, below the rim
    assert float(np.prod(hi - lo)) == pytest.approx(FILLS[name][1] * 1e-6, rel=0.01)    # the volume in its name


@pytest.mark.parametrize("name, fields", [("glycerol_80", ("density", "viscosity", "surface_tension")),
                                          ("sauce_ketchup", ("density", "viscosity")),
                                          ("water", ("density", "viscosity", "surface_tension")),
                                          ("honey", ("density", "viscosity")),
                                          ("sand_silica_dry", ("density", "friction_angle")),
                                          ("rice_grain", ("density", "friction_angle"))])
def test_fill_materials_cite_their_sources(name: str, fields: tuple[str, ...]) -> None:
    m = LIB.material(name)
    fluid, plastic = m.fluid, m.plastic
    q = {"density": m.density, "viscosity": fluid and fluid.viscosity, "surface_tension": fluid and fluid.surface_tension,
         "friction_angle": plastic and plastic.friction_angle}
    assert {f: tag_of(q[f]) for f in fields} == {f: "DOC" for f in fields}


@pytest.mark.parametrize("name", sorted(HB_FOODS))
def test_shear_thinning_foods_carry_a_sourced_fit(name: str) -> None:
    m = LIB.material(name)
    density_tag, yields = HB_FOODS[name]
    f = m.fluid
    assert tag_of(m.density) == density_tag
    assert {tag_of(q) for q in (f.viscosity, f.consistency, f.flow_index)} == {"DOC"}
    assert 0.3 < f.flow_index.value < 0.7                                               # shear-thinning
    assert (m.plastic is not None and tag_of(m.plastic.yield_stress) == "DOC") == yields
    k, n = f.consistency.value, f.flow_index.value                                     # starting viscosity: K at 10 1/s
    assert f.viscosity.value == pytest.approx(k * 10.0 ** (n - 1.0), rel=0.01)       # (the plastic part when it yields)


@pytest.mark.parametrize("name", sorted(FILLS))
def test_fill_routes_to_its_solver(name: str) -> None:
    assert alloy.estimate(box_scene(name), device="cpu")["solvers"] == [FILLS[name][0]]


@pytest.mark.parametrize("name, seconds", [("water_box_200ml", 0.2), ("honey_box_150ml", 0.1)])
def test_fill_settles_in_the_box_on_cpu(name: str, seconds: float) -> None:
    sim = alloy.build(box_scene(name), device="cpu")
    p = sim.run(seconds, record=["fill"], every=10).world(0)["fill.points"]
    p0, p1 = p[0], p[-1]
    half = 0.5 * INNER + 1e-4
    assert np.all(np.abs(p1[:, :2]) < half) and np.all(p1[:, 2] > WALL - 1e-4)       # nothing left the box
    top0, top1 = (np.percentile(q[:, 2], 99) for q in (p0, p1))     # the surface (one SPH particle may hop ~1 mm)
    assert top1 <= top0 + 1e-3                                                         # it settles, it does not rise
    assert sim.check()["ok"]


def test_shear_thinning_fill_steps_with_the_rheology_solver_on_cpu() -> None:
    from alloy.solvers.mpm_rheology.solver import SolverRheologyMPM  # noqa: PLC0415 - heavy

    sim = alloy.build(box_scene("yogurt_box_150ml"), device="cpu")
    assert any(isinstance(s, SolverRheologyMPM) for s in sim.engine.solvers)
    p1 = sim.run(0.1, record=["fill"], every=10).world(0)["fill.points"][-1]
    assert np.all(np.abs(p1[:, :2]) < 0.5 * INNER + 1e-4) and np.all(p1[:, 2] > WALL - 1e-4)
    assert sim.check()["ok"]


def cup_scene() -> alloy.Scene:
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_object("cup", "tumbler_pp", static=True)
    scene.add_object("fill", "water_tumbler_200ml", pos=(0, 0, CUP_Z))
    return scene


def test_round_cup_fill_holds_200_ml_and_routes_to_sph() -> None:
    assert not [i for i in LIB.object_issues("water_tumbler_200ml") if i.level == "error"]
    r, half = LIB.peek_object("water_tumbler_200ml").shape.size
    assert r == pytest.approx(CUP_R) and np.pi * r * r * 2 * half == pytest.approx(200e-6, rel=0.01)
    assert CUP_Z - half == pytest.approx(CUP_FLOOR + 5e-4, abs=1e-4)                  # bottom 0.5 mm above the floor
    assert alloy.estimate(cup_scene(), device="cpu")["solvers"] == ["sph"]


def test_round_cup_fill_stays_in_the_cup_on_cpu() -> None:
    sim = alloy.build(cup_scene(), device="cpu")
    p = sim.run(0.2, record=["fill"], every=10).world(0)["fill.points"]
    p0, p1 = p[0], p[-1]
    assert np.all(np.hypot(p1[:, 0], p1[:, 1]) < CUP_R + 1e-4) and np.all(p1[:, 2] > CUP_FLOOR - 1e-4)
    top0, top1 = (np.percentile(q[:, 2], 99) for q in (p0, p1))
    assert top1 <= top0 + 1e-3                                                         # it rests, no jet
    assert sim.check()["ok"]
