# SPDX-License-Identifier: Apache-2.0
"""Router decisions (SPEC §3.3) on in-code cards; CHAIN self-tests stubbed (the real one is test_rodcheck.py)."""

from __future__ import annotations

import math

import pytest

from alloy.cards.types import Crease, CurveSource, ParticleSource, Primitive, RouteOverride
from alloy.core.errors import CapacityError, RouteError
from alloy.core.registry import isolated_registry, registered_routes
from alloy.route.router import Router
from alloy.route.types import Route, RouterConfig
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.testing import route as T

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


@pytest.fixture(scope="module")
def routes():
    with isolated_registry(empty=True):
        register_builtin_solvers()
        yield registered_routes()


@pytest.fixture
def router():
    return Router(rod_self_test=lambda rod, h: dict(PASS))


def ctx(routes, **kw):
    return T.context(routes, **kw)


def test_steel_rod_rigid(router, routes):
    d = router.route(T.rod("rod", T.material("steel", **T.STEEL), 0.005, 0.2), ctx(routes))
    assert d.route == Route.RIGID and d.entry == "mujoco"
    assert d.numbers["delta_bend"] == pytest.approx(54.3e-6, rel=2e-3)


def test_aluminium_tray_rigid(router, routes):
    tray = T.box("tray", T.material("al", **T.ALUMINIUM), 0.1, 0.1, 0.001)
    d = router.route(tray, ctx(routes))
    assert d.route == Route.RIGID
    assert d.numbers["delta_bend"] == pytest.approx(7.6e-6, rel=1e-2)


def test_screwdriver_shaft_chain_near_boundary(router, routes):
    d = router.route(T.rod("shaft", T.material("steel", **T.STEEL), 0.0025, 0.1), ctx(routes))
    assert d.route == Route.CHAIN and d.near_boundary
    assert "mujoco:integrator=discrete" in d.scene_effects
    assert d.numbers["links_required"] == 2 and d.self_test["sag_err"] == PASS["sag_err"]
    assert d.numbers["links"] == 5        # SCR-DC3-7: 3 links (the minimum) sag 14 % short; 5 is the first within 5 %


def test_pencil_chain_unless_task_load(router, routes):
    wood = T.material("wood", **T.WOOD)
    assert router.route(T.rod("pencil", wood, 0.0035, 0.17), ctx(routes)).route == Route.CHAIN
    light = T.rod("pencil", wood, 0.0035, 0.17, loads=T.loads(bend_force=0.05))
    assert router.route(light, ctx(routes)).route == Route.RIGID


def test_rubber_block_compliant(router, routes):
    # 20 N on a 4 mm thick rubber pad: δ_loc = 20(1−0.45²)/(2·2e-3·2e6) = 2 mm > τ → deformable volume (SOFT_VBD);
    # stiffer rubber 1 GPa: δ_loc = 4e-6 m ≤ τ but > 0.1τ? 4e-6 < 1e-5 → rigid. 100 MPa: 4e-5 → compliant
    rub = T.material("rubber", **{**T.RUBBER, "E": 1e8})
    d = router.route(T.box("pad", rub, 0.02, 0.02, 0.01), ctx(routes))
    assert d.route == Route.RIGID_COMPLIANT
    assert d.numbers["k_compliant"] == pytest.approx(20.0 / d.numbers["delta_loc"])


def test_sponge_soft(router, routes):
    d = router.route(T.box("sponge", T.material("foam", **T.PU_FOAM), 0.05, 0.035, 0.02), ctx(routes))
    assert d.route == Route.SOFT_VBD and d.entry == "vbd"


def test_cloth_and_panels(router, routes):
    cotton = T.material("cotton", **T.COTTON)
    cloth = T.box("towel", cotton, 0.2, 0.2, 2.5e-4)
    assert router.route(cloth, ctx(routes)).route == Route.CLOTH_VBD
    card = T.material("board", density=700.0, E=3e9, nu=0.3)
    crease = Crease(a=(0.0, -0.1, 0.0), b=(0.0, 0.1, 0.0), stiffness=T.q(0.05))
    lid = T.box("lid", card, 0.15, 0.1, 2.5e-4, features=T.features(creases=(crease,)))
    assert router.route(lid, ctx(routes)).route == Route.PANELS


def test_a_flat_cloth_mesh_routes_by_its_card_thickness(router, routes):
    """A perfectly flat sheet (OBB l3 = 0, XB4-10's drape disc) squeezes by the card's thickness instead of dividing by
    l3; without a thickness it is refused by name."""
    cotton = T.material("cotton", **T.COTTON)
    flat = T.box("disc", cotton, 0.15, 0.15, 0.0, features=T.features(thickness=2.5e-4))
    d = router.route(flat, ctx(routes))
    assert d.route == Route.CLOTH_VBD
    assert d.numbers["delta_loc"] == pytest.approx(20.0 * (1 - 0.09) / (2 * 1.25e-4 * 1e6))
    with pytest.raises(RouteError, match="disc: the shape is flat"):
        router.route(T.box("disc", cotton, 0.15, 0.15, 0.0), ctx(routes))


def test_cloth_accuracy_tags_cloth_decisions_only(routes):
    """RouterConfig.cloth_accuracy tags every CLOTH_VBD decision with its setting's effect, which the VBD entry maps
    to SolverVBDAlloy's flags (alloy.scene.cloth_accuracy, SCR-WP2-13); other routes carry no tag."""
    from alloy.scene import cloth_accuracy as CA

    cotton, rubber = T.material("cotton", **T.COTTON), T.material("rubber", **T.RUBBER)
    towel, block = T.box("towel", cotton, 0.2, 0.2, 2.5e-4), T.box("block", rubber, 0.05, 0.05, 0.05)
    plain = Router(rod_self_test=lambda rod, h: dict(PASS))
    sup = Router(RouterConfig(cloth_accuracy="super"), rod_self_test=lambda rod, h: dict(PASS))
    assert CA.SUPER_EFFECT in sup.route(towel, ctx(routes)).scene_effects
    effects = plain.route(towel, ctx(routes)).scene_effects
    assert CA.DEFAULT_EFFECT in effects and CA.SUPER_EFFECT not in effects
    fast = Router(RouterConfig(cloth_accuracy="fast"), rod_self_test=lambda rod, h: dict(PASS))
    assert CA.FAST_EFFECT in fast.route(towel, ctx(routes)).scene_effects
    assert not set(CA.EFFECT.values()) & set(sup.route(block, ctx(routes)).scene_effects)


def test_static_and_kinematic(router, routes):
    table = T.box("table", T.material("wood", **T.WOOD), 0.5, 0.5, 0.02, motion="static")
    d = router.route(table, ctx(routes))
    assert d.route == Route.STATIC and d.entry == ""


def test_pile(router, routes):
    d = router.route(T.box("block", T.material("wood", **T.WOOD), 0.01, 0.01, 0.01), ctx(routes, free_bodies=64))
    assert d.route == Route.RIGID_PILE and d.entry == "mujoco"


def test_liquid_without_sph_is_mpm_approximate(router, routes):
    water = T.material("water", density=1000.0, fluid_mu=1e-3)
    no_sph = frozenset(routes) - {"liquid_sph"}
    d = router.route(T.obj("water", ParticleSource(((0, 0, 0), (0.05, 0.05, 0.05)), 0.002), water), ctx(no_sph))
    assert d.route == Route.LIQUID_MPM and d.approximate and d.entry == "mpm"


def test_builtin_registration_loads_plugins(routes):
    # SCR-WP9-5: register_builtin_solvers() ends with load_plugins(), so WP9's liquid_sph route is registered (D15),
    # also inside a fresh isolated registry where the plug-in import is already cached
    pytest.importorskip("alloy.solvers.sph")
    assert "liquid_sph" in routes
    with isolated_registry(empty=True):
        register_builtin_solvers()
        assert "liquid_sph" in registered_routes()


def test_liquid_sph_radius_from_passage(routes):
    """The radius is half liquid_accuracy's spacing (SCR-WP9-LIQRES); a 16 mm passage is recorded for the build's
    warning, not refined into. The old LIQ-6 rule (r = passage/16) stays behind RouterConfig.sph_refine_passage."""
    r = Router(rod_self_test=lambda rod, h: dict(PASS))
    water = T.material("water", density=1000.0, fluid_mu=1e-3)
    cup = T.box("cup", T.material("pp", density=900.0, E=1.5e9), 0.04, 0.04, 0.05,
                features=T.features(passage_width=0.016))
    w = T.obj("water", ParticleSource(((0, 0, 0), (0.02, 0.02, 0.02)), 0.002), water)
    c = ctx(routes | {"liquid_sph"}, touching={"water": ("cup",)}, partners={"cup": cup})
    d = r.route(w, c)
    assert d.route == Route.LIQUID_SPH and d.numbers["particle_radius"] == pytest.approx(2e-3, abs=0)
    assert d.numbers["liquid_spacing"] == pytest.approx(4e-3, abs=0)
    assert d.numbers["passage_width"] == pytest.approx(0.016, abs=0)
    assert d.numbers["viscosity_weiler"] == 0.0
    sup = Router(RouterConfig(liquid_accuracy="super"), rod_self_test=lambda rod, h: dict(PASS)).route(w, c)
    assert sup.numbers["particle_radius"] == pytest.approx(1e-3, abs=0)
    old = Router(RouterConfig(sph_refine_passage=True), rod_self_test=lambda rod, h: dict(PASS)).route(w, c)
    assert old.numbers["particle_radius"] == pytest.approx(1e-3, abs=0)          # passage / 16
    big = T.obj("water", ParticleSource(((0, 0, 0), (0.5, 0.5, 0.5)), 0.002), water)
    with pytest.raises(CapacityError):
        r.route(big, c)


def test_paste_and_viscous(router, routes):
    honey = T.material("honey", density=1400.0, fluid_mu=10.0)
    region = ParticleSource(((0, 0, 0), (0.03, 0.03, 0.03)), 0.002)
    assert router.route(T.obj("honey", region, honey), ctx(routes)).route == Route.PASTE_MPM
    dough = T.material("dough", density=1100.0, E=1e4, yield_stress=200.0)
    assert router.route(T.box("dough", dough, 0.03, 0.03, 0.02), ctx(routes)).route == Route.PASTE_MPM


def test_granular_rules(router, routes):
    sand = T.material("sand", density=1600.0, E=1e7, friction_angle=math.radians(33))
    fine = T.obj("sand", ParticleSource(((0, 0, 0), (0.1, 0.1, 0.05)), 5e-4, grain_diameter=3e-4), sand)
    assert router.route(fine, ctx(routes)).route == Route.GRAIN_MPM          # voxel 5 mm ≥ 3·0.3 mm
    beans = T.obj("beans", ParticleSource(((0, 0, 0), (0.04, 0.04, 0.02)), 0.01, grain_diameter=0.01), sand)
    d = router.route(beans, ctx(routes))
    assert d.route == Route.GRAIN_RIGID and d.numbers["n_grains"] <= 2000
    rice = T.obj("rice", ParticleSource(((0, 0, 0), (0.2, 0.2, 0.1)), 0.005, grain_diameter=0.005), sand)
    d = router.route(rice, ctx(routes))            # SCR-SOFT-2-3: over the rigid budget, voxel < 3 grains -> continuum
    assert d.route == Route.GRAIN_MPM and d.numbers["n_grains"] > 2000
    assert any("continuum" in r for r in d.reasons)
    narrow = T.obj("rice", ParticleSource(((0, 0, 0), (0.2, 0.2, 0.1)), 0.005, grain_diameter=0.005), sand,
                   features=T.features(passage_width=0.03))          # a 6-grain passage: grains jam, no continuum
    with pytest.raises(RouteError, match="jamming.*6.0 grains"):
        router.route(narrow, ctx(routes))


def test_limp_long_rope_goes_to_the_elastic_rod(router, routes):
    """Past CHAIN's 40-link cap a rod routes to ROD_VBD (coordinator ruling 2026-11-08 11:53Z; was a RouteError)."""
    rope = T.material("rope", density=1100.0, E=1e6, nu=0.4)   # c = 4.5 cm over 3 m: 68 links > 40
    d = router.route(T.rod("rope", rope, 0.002, 3.0), ctx(routes))
    assert d.route == Route.ROD_VBD and d.entry == "vbd"
    assert any("links > CHAIN's 40" in r for r in d.reasons)


def test_curve_cable_chain_links(router, routes):
    pvc = T.material("pvc", **T.CABLE_PVC)
    pts = tuple((0.05 * i, 0.0, 0.0) for i in range(21))
    d = router.route(T.obj("cable", CurveSource(points=pts, radius=0.002), pvc), ctx(routes))
    assert d.route == Route.CHAIN
    assert d.numbers["rod_length"] == pytest.approx(1.0)
    assert d.numbers["links"] == d.numbers["links_required"]          # the sag rule, no 30-link floor (SCR-DC3-7)


def test_needs_l1_falls_back_to_discrete_when_implicitfast_fails(routes):
    calls = []

    def st(rod, h):
        calls.append(rod.integrator)
        return dict(PASS) if rod.integrator == "discrete" else {**PASS, "period_err": 0.5, "passed": 0.0}

    d = Router(rod_self_test=st).route(T.rod("shaft", T.material("steel", **T.STEEL), 0.0025, 0.1),
                                       ctx(routes, needs_l1=True))
    assert calls == ["implicitfast", "discrete"] and "mujoco:integrator=discrete" in d.scene_effects
    assert any("L1 unavailable" in r for r in d.reasons)


def test_override_keeps_numbers(router, routes):
    rod = T.rod("rod", T.material("steel", **T.STEEL), 0.005, 0.2, route=RouteOverride("chain", "task wants it"))
    d = router.route(rod, ctx(routes))
    assert d.route == Route.CHAIN and d.overridden and "delta_bend" in d.numbers


def test_worst_case_softest_modulus_changes_route(router, routes):
    steel = T.material("steel", **T.STEEL)
    stiff = T.rod("rod", steel, 0.005, 0.2)
    soft = T.rod("rod", steel, 0.005, 0.2, worst_case={"material.elastic.youngs": 50e9})
    assert router.route(stiff, ctx(routes)).route == Route.RIGID
    assert router.route(soft, ctx(routes)).route == Route.CHAIN


def test_unregistered_route_raises(router):
    with pytest.raises(RouteError, match="no registered"):
        router.route(T.box("sponge", T.material("foam", **T.PU_FOAM), 0.05, 0.035, 0.02),
                     T.context(frozenset({"rigid"})))


def test_router_config_thresholds(routes):
    lax = Router(RouterConfig(rigid_ratio=1.0), rod_self_test=lambda rod, h: dict(PASS))
    rub = T.material("rubber", **{**T.RUBBER, "E": 1e8})
    assert lax.route(T.box("pad", rub, 0.02, 0.02, 0.01), ctx(routes)).route == Route.RIGID


def test_links_rise_until_the_self_test_passes(routes):
    """SCR-DC3-7: a chain whose self-test fails at the sag count (here: period below 7 links) gets ~25 % more links
    per try; a card's features.chain_links is kept as given."""
    tried = []

    def st(rod, h):
        tried.append(rod.links)
        return dict(PASS) if rod.links >= 7 else {**PASS, "period_err": 0.04, "passed": 0.0}

    shaft = T.rod("shaft", T.material("steel", **T.STEEL), 0.0025, 0.1)
    d = Router(rod_self_test=st).route(shaft, ctx(routes))
    assert tried == [5, 7] and d.numbers["links"] == 7 and d.self_test["passed"] == 1.0
    assert any("5 failed the self-test, 7 pass" in r for r in d.reasons)
