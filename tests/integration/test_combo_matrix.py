# SPDX-License-Identifier: Apache-2.0
"""The material-combination matrix (user ask 2026-11-07; [internal note] notes/combo-matrix-1107.md), pinned on the CPU.

Every pair of the nine material kinds (and six triples) is one scene from ``test_combo_matrix_kit``: the kinds
stacked on a static plate, each falling onto the one below. The 2026-11-07 sweep (2 sim-s, 2 CPU worlds and 64 L4
worlds) sorted each into works / errors cleanly / runs but wrong / blows up / too slow. These tests keep that matrix
true: when a fix changes a combo's status, the test for it fails, and the matrix and the expectation here change with
the fix.

- ``BUILD_ERRORS``: ``alloy.build`` refuses the scene with an Alloy error that says what to change.
- F12: water filled into a free soft cup is refused at build; a static cup builds.
- ``UNSUPPORTED_PAIRS``: route pairs Alloy cannot step together, refused at build by WP12's registry
  (``alloy.coupling.unsupported``, 3d9a5538f). The sweep found them building and then failing at the first step with
  a solver's own error (F1), or running with the cloth through the paper (F4). mpm+liquid left it in SCR-WP12-MIX
  (the MPM particles are a particle wall of the liquid) and is in ``CLOCKS``.
- ``RUNS``: cheap on the CPU; 0.1 s (10 periods) runs healthy, with each kind's books (liquid particles kept, film
  ledger exact, heat taken). ``NO_FILM_CONTACT``: the surface film neither moves what lands on it nor feels it.
- ``CLOCKS``: coupled combos too slow for a CPU test (1.7-36 s per period): the build-time exchange clock, the
  number of exchanges per 10 ms period (2^j) that the pair bound sets, and the routes.
- Slow (``-m slow``): heated paste beside the pusher (a longer CPU run).
"""

from __future__ import annotations

import math

import alloy
import numpy as np
import pytest

from alloy.coupling.unsupported import FILLED_SOFT_CONTAINER, unsupported
from alloy.route.types import Route
from tests.integration.test_combo_matrix_kit import FILM_DEPTH, TOP, build, combo

NO_HEAT = ("SceneError", "nothing to heat")
PEANUT_HEAT = ("SceneError", "no heat data for material 'peanut_butter'")
BUILD_ERRORS = {
    "film+heat": NO_HEAT, "heat": NO_HEAT, "heat+paper": NO_HEAT, "heat+cloth": NO_HEAT,
    "heat+soft": NO_HEAT, "heat+rigid": NO_HEAT, "heat+robot": NO_HEAT, "heat+mpm": PEANUT_HEAT,
    "heat+mpm+robot": PEANUT_HEAT,
}
UNSUPPORTED_PAIRS = ("cloth+mpm", "mpm+soft", "paper+soft", "paper+cloth")   # mpm+liquid steps (SCR-WP12-MIX)
ROUTES = {"film": None, "paper": "cloth_film", "cloth": "cloth_vbd", "mpm": "paste_mpm", "liquid": "liquid_sph",
          "soft": "soft_vbd", "rigid": "rigid", "robot": "rigid", "heat": None}
NAMES = {"paper": "paper", "cloth": "cloth", "mpm": "paste", "liquid": "water", "soft": "sponge", "rigid": "block",
         "robot": "pusher"}
RUNS = ("film", "liquid", "cloth", "rigid", "robot", "soft", "paper", "rigid+robot", "film+rigid", "film+robot",
        "film+rigid+robot", "film+cloth", "film+liquid", "heat+liquid", "liquid+rigid", "liquid+robot",
        "heat+liquid+rigid", "cloth+liquid", "film+paper")
RUN_TIMEOUT = {"cloth+liquid": 900}   # s: ~200 s on host-1 since cloth's 40-iteration default (1de04ca01; 456 s on a
#                                      loaded VM, was 184 s), past pytest's 60 s default (WP11, 2026-11-08)
CLOCKS = {          # combo: exchanges per 10 ms period at build (2^j), the coupled stepper
    "cloth+rigid": (128, "Scheduler"), "paper+rigid": (128, "Scheduler"), "soft+rigid": (128, "Scheduler"),
    "robot+rigid+cloth": (128, "Scheduler"), "paper+rigid+robot": (128, "Scheduler"),
    "cloth+liquid+rigid": (128, "Scheduler"), "paper+robot": (64, "Scheduler"),   # one chain: WP4 3dddcdfaa
    "cloth+robot": (64, "Scheduler"), "soft+robot": (64, "Scheduler"), "mpm+rigid": (8, "Scheduler"),
    "mpm+robot": (8, "Scheduler"), "paper+mpm": (8, "Scheduler"), "paper+liquid": (8, "Scheduler"),
    "liquid+soft": (8, "Scheduler"), "mpm+liquid": (8, "Scheduler"),   # SCR-WP12-MIX: MPM particles as SPH walls
}


def _kinds(spec: str) -> tuple[str, ...]:
    return tuple(spec.split("+"))


def _routes(sim) -> dict[str, str]:
    return {n: d["route"] for n, d in sim.routes().items()}


def _check_routes(sim, spec: str) -> None:
    got = _routes(sim)
    for k in _kinds(spec):
        if ROUTES[k] is not None:
            name = NAMES[k]
            assert got[name] == ROUTES[k], (spec, name, got)


@pytest.mark.parametrize("spec", sorted(BUILD_ERRORS))
def test_build_refuses_with_an_alloy_error(spec: str) -> None:
    cls, text = BUILD_ERRORS[spec]
    with pytest.raises(getattr(alloy, cls), match=text) as e:
        build(combo(_kinds(spec), worlds=1))
    assert e.value.fix                                   # the error says what to change


@pytest.mark.parametrize("spec", UNSUPPORTED_PAIRS)
def test_unsupported_pairs_are_refused_at_build(spec: str) -> None:
    """WP12's registry refuses the pair at build (it used to build and then fail at the first step, F1, or run with
    the cloth through the paper, F4): a BuildError naming both routes, the registry's alternative as its fix, and the
    sweep pair as the registry entry's finding."""
    a, b = (Route(ROUTES[k]) for k in _kinds(spec))
    u = unsupported(a, b)
    assert u is not None and spec in u.finding, (spec, u)
    with pytest.raises(alloy.BuildError, match="not supported in one scene yet") as e:
        build(combo(_kinds(spec), worlds=1))
    assert str(a) in e.value.message and str(b) in e.value.message
    assert e.value.fix == u.alternative


def _cup(static: bool) -> alloy.Scene:
    s = alloy.Scene(worlds=1, dt=0.01)
    s.add_ground()
    s.add_object("cup", "tumbler_pp", pos=(0.0, 0.0, 0.0), static=static)
    s.add_object("water", "water_tumbler_200ml", pos=(0.0, 0.0, 0.0232))
    return s


def test_water_filled_into_a_free_soft_cup_is_refused_at_build() -> None:
    """F12 (the code read's "floating containers"): a free tumbler_pp routes to the cloth solver, and water filled into
    it at build latched ENERGY_JUMP 0.16 ms into the first step (WP12's FILLED_SOFT_CONTAINER, 12481e58d, refuses it
    with the alternative as its fix). The same water in a static cup builds."""
    with pytest.raises(alloy.BuildError, match="starts inside") as e:
        alloy.build(_cup(static=False), device="cpu")
    assert e.value.fix == FILLED_SOFT_CONTAINER.alternative
    sim = alloy.build(_cup(static=True), device="cpu")
    assert _routes(sim)["water"] == "liquid_sph" and _routes(sim)["cup"] == "static"


def _points(sim, name: str) -> np.ndarray:
    return np.asarray(sim.state([name])[f"{name}.points"], dtype=np.float64)


def _books(sim, c, spec: str, start: dict) -> None:
    kinds = _kinds(spec)
    for k in ("paper", "cloth", "mpm", "liquid", "soft"):
        if k in kinds:
            p = _points(sim, NAMES[k])
            assert np.isfinite(p).all(), (spec, k)
            assert p[..., 2].min() > TOP - 1.0e-3 or k == "liquid", (spec, k, p[..., 2].min())   # not through
    if "liquid" in kinds:
        assert sim.health()["retired_particles"] == [0]                 # every particle kept
        assert sim.liquid_volume("water") == start["liquid"]
    if c.flow is not None:
        led = np.asarray(c.flow.read()["ledger"], dtype=np.float64)
        assert np.abs(led - start["film"]).max() <= 1.0e-12 * np.abs(start["film"]).max()
    if "heat" in kinds:
        assert float(sim.heat("water")["mean_K"][0]) > start["heat"]


@pytest.mark.parametrize("spec", [pytest.param(s, marks=pytest.mark.timeout(RUN_TIMEOUT[s])) if s in RUN_TIMEOUT else s
                                  for s in RUNS])
def test_cheap_combos_run_healthy(spec: str) -> None:
    c = combo(_kinds(spec), worlds=1)
    sim = build(c)
    _check_routes(sim, spec)
    start = {"liquid": sim.liquid_volume("water") if "liquid" in c.kinds else None,
             "film": np.asarray(c.flow.read()["ledger"], dtype=np.float64).copy() if c.flow is not None else None,
             "heat": float(sim.heat("water")["mean_K"][0]) if "heat" in c.kinds else None}
    sim.run(0.1, targets={"lift": (0.0, -0.5 * c.push)} if c.push else None)
    assert sim.check()["ok"], sim.check()["problems"]
    _books(sim, c, spec, start)


@pytest.mark.parametrize("spec", ["film+rigid", "film+cloth"])
def test_the_surface_film_does_not_touch_what_lands_on_it(spec: str) -> None:
    """Runs but wrong: a block or a cloth comes down through 1 mm of water film onto the plate as if it were not
    there, and the film (a plug-in on the plate's chart) keeps every cell's depth: no displacement either way."""
    c = combo(_kinds(spec), worlds=1)
    sim = build(c)
    sim.run(0.2)
    depth = c.flow.depth(0)[0]
    assert np.array_equal(depth, np.full(depth.shape, FILM_DEPTH))
    if "rigid" in c.kinds:
        bottom = float(sim.state(["block"])["block.pos"][0][2]) - 0.02
        assert abs(bottom - TOP) < 1.0e-4                                # on the plate, through the film
    else:
        assert abs(_points(sim, "cloth")[..., 2].min() - (TOP + 1.0e-3)) < 1.0e-4


@pytest.mark.parametrize("spec", sorted(CLOCKS))
def test_slow_combos_keep_their_exchange_clock(spec: str) -> None:
    """The exchanges per control period the coupled stepper plans at build: control_dt / 2^j <= the tightest pair's
    bound (contact stiffness against the lighter side's mass). 128 per period is what makes a block on cloth, paper or
    a sponge cost 7-36 s of CPU per 10 ms period (the sweep: block on sponge escalates further, to 1024)."""
    per, stepper = CLOCKS[spec]
    sim = build(combo(_kinds(spec), worlds=1))
    _check_routes(sim, spec)
    st = sim.engine.stepper
    assert type(st).__name__ == stepper
    bound = st._interface_bound()
    j = max(3, math.ceil(math.log2(sim.dt / bound))) if math.isfinite(bound) else 3
    assert 2 ** j == per, (spec, bound)


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_heated_paste_beside_a_robot_takes_heat_at_the_default_gap() -> None:
    """Heated paste (MPM, porridge heat data) on the heater plate with the pusher in the scene (never touching it)
    takes heat at the default heater gap, the MPM voxel (WP7's d73527b5d, the matrix's F5). At the sweep's base the
    default was a liquid's quarter spacing and this scene took 0 J: its bottom layer rested 1.5 mm higher than alone."""
    sim = build(combo(("heat", "mpm", "robot"), worlds=1, heat_material="porridge_oat"))
    sim.step(10)
    assert float(np.asarray(sim.plugins.read()["particle_thermal"]["heat_in_J"]).sum()) > 0.0
