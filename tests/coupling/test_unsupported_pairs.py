# SPDX-License-Identifier: Apache-2.0
"""Route pairs Alloy cannot step together fail closed at build (alloy.coupling.unsupported, checked by
alloy.integration.wiring.wire; WP13's combination sweep 2026-11-07, F1 and F4): an Alloy error naming the
pair and the nearest supported alternative, before any step. F12: a liquid that starts inside a free cloth or soft body
(a soft cup filled at build) is refused at assembly (alloy.scene.entries.make_entries). SCR-WP12-MIX: an SPH liquid
beside an MPM paste or powder builds; a second liquid on the MPM route (R1) and a liquid starting inside the MPM
material (R3, ``PREWETTED``) are refused."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling import unsupported as U
from alloy.route.types import Route

R = Route
SWEEP = {  # combination sweep pair -> the routes it puts in two entries
    "cloth+mpm": (R.CLOTH_VBD, R.PASTE_MPM), "mpm+soft": (R.PASTE_MPM, R.SOFT_VBD),
    "paper+soft": (R.CLOTH_FILM, R.SOFT_VBD), "paper+cloth": (R.CLOTH_FILM, R.CLOTH_VBD)}
WORKS = [(R.CLOTH_VBD, R.LIQUID_SPH), (R.CLOTH_FILM, R.PASTE_MPM), (R.CLOTH_FILM, R.LIQUID_SPH),
         (R.PASTE_MPM, R.LIQUID_SPH), (R.GRAIN_MPM, R.LIQUID_SPH),        # SCR-WP12-MIX: particle walls
         (R.SOFT_VBD, R.LIQUID_SPH), (R.RIGID, R.PASTE_MPM), (R.RIGID, R.CLOTH_VBD), (R.RIGID, R.LIQUID_SPH)]


@pytest.mark.parametrize("name", sorted(SWEEP))
def test_every_failing_sweep_pair_is_in_the_registry_both_ways(name: str) -> None:
    x, y = SWEEP[name]
    u = U.unsupported(x, y)
    assert u is not None and u is U.unsupported(y, x) and name in u.finding and u.alternative


@pytest.mark.parametrize("pair", WORKS, ids=lambda p: f"{p[0]}+{p[1]}")
def test_pairs_the_sweep_ran_are_not_refused(pair) -> None:
    assert U.unsupported(*pair) is None and U.unsupported(*pair[::-1]) is None


def test_every_mpm_route_meets_the_same_refusals() -> None:
    for mpm in U.MPM_ROUTES:
        for other in (R.CLOTH_VBD, R.SOFT_VBD):
            assert U.unsupported(mpm, other) is not None, (mpm, other)


def test_a_second_liquid_on_the_mpm_route_stays_refused() -> None:
    """MIX R1: the MPM liquid would be a solid particle wall to the SPH one; paste and powder beside SPH build."""
    u = U.unsupported(R.LIQUID_SPH, R.LIQUID_MPM)
    assert u is not None and u is U.unsupported(R.LIQUID_MPM, R.LIQUID_SPH) and "both liquids on SPH" in u.alternative
    assert U.unsupported(R.LIQUID_SPH, R.PASTE_MPM) is None and U.unsupported(R.GRAIN_MPM, R.LIQUID_SPH) is None


def _entry(name: str, *routes: Route, objects=("a",)) -> SimpleNamespace:
    return SimpleNamespace(name=name, routes=frozenset(routes), objects=tuple(objects))


def test_a_built_scene_is_refused_before_wiring_and_names_both_sides() -> None:
    d = lambda route, entry: SimpleNamespace(route=route, entry=entry)  # noqa: E731
    built = SimpleNamespace(report=SimpleNamespace(decisions={
        "table": d(R.STATIC, None), "block": d(R.RIGID, "mujoco"), "towel": d(R.CLOTH_VBD, "vbd"),
        "sponge": d(R.SOFT_VBD, "vbd"), "paste": d(R.PASTE_MPM, "mpm")}))
    with pytest.raises(U.UnsupportedPairError) as e:
        U.check_built(built)
    text = str(e.value)
    assert "towel, sponge (cloth) on route cloth_vbd and paste (paste) on route paste_mpm" in text      # no entry names
    assert "Nearest supported" in text and "vbd:" not in text and "sweep" not in text
    assert e.value.pair == (R.CLOTH_VBD, R.PASTE_MPM)
    assert e.value.fix == U.unsupported(R.CLOTH_VBD, R.PASTE_MPM).alternative
    del built.report.decisions["paste"]
    U.check_built(built)                                        # rigid + cloth + soft: allowed


def test_routes_inside_one_entry_are_that_solvers_business() -> None:
    U.check_entries([_entry("vbd", R.CLOTH_VBD, R.SOFT_VBD), _entry("sph", R.LIQUID_SPH)])     # cloth+soft works
    U.check_entries([SimpleNamespace(name="mpm", routes=frozenset({R.PASTE_MPM, R.GRAIN_MPM}), objects=("a", "b"))])


def _scene(kinds: tuple[str, ...]):
    """WP13's sweep stack (tests/integration/test_combo_matrix_kit.py) for two kinds, through the public API."""
    import alloy  # noqa: PLC0415

    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.4, 0.4, 0.02), pos=(0.0, 0.0, 0.01), static=True)
    add = {"paper": lambda: scene.add_object("paper", "paper_kraft_sheet", pos=(0.0, 0.0, 0.022)),
           "cloth": lambda: scene.add_cloth("cloth", size=(0.14, 0.14), pos=(0.0, 0.0, 0.03), resolution=10, mass=0.02),
           "mpm": lambda: scene.add_object("paste", "peanut_butter_box_150ml", pos=(0.0, 0.0, 0.04)),
           "liquid": lambda: scene.add_liquid("water", size=(0.04, 0.04, 0.02), pos=(0.0, 0.0, 0.11)),
           "soft": lambda: scene.add_box("sponge", size=(0.05, 0.05, 0.03), pos=(0.0, 0.0, 0.15),
                                         material="pu_foam_sponge")}
    for k in kinds:
        add[k]()
    return scene


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", sorted(SWEEP))
def test_the_public_build_fails_closed_with_the_alternative(name: str) -> None:
    """Before: the build passed and the first step raised NotImplementedError / ValueError (F1), or the cloth fell
    through the paper (F4)."""
    import alloy  # noqa: PLC0415

    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_scene(tuple(name.split("+"))), device="cpu")
    x, y = SWEEP[name]
    assert "UnsupportedPairError" in e.value.message and "not supported in one scene yet" in e.value.message
    assert {str(x), str(y)} <= set(e.value.message.replace("(", " ").split())
    assert U.unsupported(x, y).alternative in e.value.message and e.value.fix == U.unsupported(x, y).alternative


@pytest.mark.timeout(600)
def test_a_build_with_plugins_is_refused_too() -> None:
    """With plug-ins the API wires the scene itself (alloy.api._plugins.build_engine, not build_simulation); before,
    that path skipped the check and the cloth + paste scene built."""
    import alloy  # noqa: PLC0415

    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_scene(("cloth", "mpm")), device="cpu", plugins=[lambda built: None])
    assert "UnsupportedPairError" in e.value.message
    assert e.value.fix == U.unsupported(R.CLOTH_VBD, R.PASTE_MPM).alternative


def _shell(kind: str) -> list[tuple[float, float, float]]:
    """Particles of a 40 mm cup (walls and a floor at z 0), a flat 40 mm sheet, or a dome open at the bottom (a towel
    crumpled over whatever is under it: its rim at z 0.01, its top at 0.06)."""
    ring = [(0.02 * math.cos(a), 0.02 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    floor = [(x, y) for x in np.linspace(-0.015, 0.015, 7) for y in np.linspace(-0.015, 0.015, 7)]
    if kind == "dome":
        return [(x, y, z) for x, y in ring for z in (0.01, 0.04)] + [(0.0, 0.0, 0.06)]
    walls = [(x, y, z) for x, y in ring for z in ((0.0,) if kind == "flat" else (0.0, 0.03, 0.06))]
    return walls + [(x, y, 0.0) for x, y in floor]


def _cup_and_water(water_at=(0.0, 0.0, 0.03), *, shell="cup", motion="dynamic", active=1):
    """A soft shell (``_shell``) and a 3 x 3 x 3 block of water particles centred at ``water_at``: template objects
    and a stand-in builder."""
    ring = _shell(shell)
    g = np.linspace(-0.005, 0.005, 3)
    water = [(water_at[0] + x, water_at[1] + y, water_at[2] + z) for x in g for y in g for z in g]
    n = len(ring)
    obj = lambda route, card, lo, hi: SimpleNamespace(  # noqa: E731
        name={R.LIQUID_SPH: "water"}.get(route, "cup"), obj=SimpleNamespace(card=card),
        decision=SimpleNamespace(route=route), ranges=SimpleNamespace(particle=range(lo, hi)))
    objs = [obj(R.CLOTH_VBD, SimpleNamespace(motion=motion), 0, n),
            obj(R.LIQUID_SPH, SimpleNamespace(motion="dynamic"), n, n + len(water))]
    return objs, SimpleNamespace(particle_q=ring + water, particle_flags=[1] * n + [active] * len(water))


def test_a_liquid_starting_inside_a_free_soft_cup_is_refused() -> None:
    objs, builder = _cup_and_water()
    with pytest.raises(U.UnsupportedPairError) as e:
        U.check_objects(objs, builder)
    assert "water (liquid_sph) starts inside cup (cloth_vbd, free): 27 liquid particles in its box" in str(e.value)
    assert e.value.pair == (R.CLOTH_VBD, R.LIQUID_SPH) and e.value.fix == U.FILLED_SOFT_CONTAINER.alternative


@pytest.mark.parametrize("case", ["beside", "flat sheet", "under a dome", "not yet poured", "kinematic", "no builder"])
def test_a_liquid_outside_or_a_held_or_flat_soft_object_is_allowed(case: str) -> None:
    """WP13's review: a box alone is wider than "inside" (a towel crumpled over a basin holds nothing)."""
    shell = {"flat sheet": "flat", "under a dome": "dome"}.get(case, "cup")
    objs, builder = _cup_and_water(water_at=(0.1, 0.0, 0.03) if case == "beside" else (0.0, 0.0, 0.03),
                                   shell=shell, active=0 if case == "not yet poured" else 1,
                                   motion="kinematic" if case == "kinematic" else "dynamic")
    if case == "under a dome":                    # the water is inside the dome's box
        q = np.asarray(builder.particle_q)
        assert np.all((q[-27:] > q[:25].min(axis=0)) & (q[-27:] < q[:25].max(axis=0)))
    U.check_objects(objs, None if case == "no builder" else builder)


def test_held_by_needs_a_part_of_the_shell_under_each_point() -> None:
    cup = np.asarray(_shell("cup"))
    x = np.asarray([(0.0, 0.0, 0.03), (0.0, 0.0, -0.01), (0.0, 0.0, 0.07), (0.05, 0.0, 0.03)])
    assert U.held_by(cup, x).tolist() == [True, False, False, False]       # in it; under it, above it, beside it
    assert not U.held_by(np.asarray(_shell("dome")), x[:1]).any()


def _cup_scene(*, static: bool = False, beside: bool = False):
    import alloy  # noqa: PLC0415

    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_object("cup", "tumbler_pp", pos=(0.0, 0.0, 0.0), static=static)
    if beside:
        scene.add_liquid("water", size=(0.06, 0.06, 0.02), pos=(0.12, 0.0, 0.0105))
    else:
        scene.add_object("water", "water_tumbler_200ml", pos=(0.0, 0.0, 0.0232))      # the fill's place in the cup
    return scene


@pytest.mark.timeout(600)
def test_the_public_build_of_a_filled_free_cup_fails_closed() -> None:
    """Before: the free cup (cloth_vbd) filled with water built, then latched ENERGY_JUMP 0.16 ms into the first step
    (WP13's sweep, F12). The static cup and water beside the free cup still build."""
    import alloy  # noqa: PLC0415

    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_cup_scene(), device="cpu")
    assert "UnsupportedPairError" in e.value.message and "water (liquid_sph) starts inside cup" in e.value.message
    assert e.value.fix == U.FILLED_SOFT_CONTAINER.alternative
    for kw in ({"static": True}, {"beside": True}):
        alloy.build(_cup_scene(**kw), device="cpu")


def _paste_and_water(gap: float, *, active: int = 1):
    """A 3 x 3 x 3 lattice of MPM paste particles (r 1.25 mm, 2.5 mm apart) and one of water (r 2 mm) whose lowest
    layer sits ``gap`` above the paste's top layer: template objects and a stand-in builder."""
    g = np.arange(3) * 0.0025
    paste = [(x, y, z) for x in g for y in g for z in g]
    water = [(x, y, 0.005 + gap + z) for x in g for y in g for z in (0.0, 0.004)]
    obj = lambda name, route, lo, hi: SimpleNamespace(  # noqa: E731
        name=name, obj=SimpleNamespace(card=SimpleNamespace(motion="dynamic")),
        decision=SimpleNamespace(route=route), ranges=SimpleNamespace(particle=range(lo, hi)))
    n, k = len(paste), len(water)
    objs = [obj("paste", R.PASTE_MPM, 0, n), obj("water", R.LIQUID_SPH, n, n + k)]
    return objs, SimpleNamespace(particle_q=paste + water, particle_flags=[1] * n + [active] * k,
                                 particle_radius=[0.00125] * n + [0.002] * k)


def test_a_liquid_starting_inside_an_mpm_paste_is_refused() -> None:
    """MIX R3: within r_liquid + r_mpm (3.25 mm) of a paste particle at the start."""
    objs, builder = _paste_and_water(0.002)
    with pytest.raises(U.UnsupportedPairError) as e:
        U.check_objects(objs, builder)
    assert "water (liquid_sph) starts inside paste (paste_mpm): 9 liquid particles within 3.25 mm" in str(e.value)
    assert e.value.pair == (R.LIQUID_SPH, R.PASTE_MPM) and e.value.fix == U.PREWETTED.alternative


@pytest.mark.parametrize("case", ["clear", "not yet poured"])
def test_a_liquid_clear_of_the_paste_or_still_to_pour_is_allowed(case: str) -> None:
    objs, builder = _paste_and_water(0.0035 if case == "clear" else 0.0, active=0 if case == "not yet poured" else 1)
    U.check_objects(objs, builder)


@pytest.mark.timeout(600)
def test_the_public_build_of_a_paste_beside_a_liquid_now_builds() -> None:
    """Before SCR-WP12-MIX the sweep's mpm+liquid stack was refused (F1); now it builds, and its MPM entry takes the
    liquid's force (alloy.integration.mpm_proxy)."""
    import alloy  # noqa: PLC0415

    sim = alloy.build(_scene(("mpm", "liquid")), device="cpu")
    assert {sim.routes()["paste"]["route"], sim.routes()["water"]["route"]} == {"paste_mpm", "liquid_sph"}
    assert any("takes sph's force on its particles" in n for n in sim.engine.parts.notes)
