# SPDX-License-Identifier: Apache-2.0
"""D5 coupler selection (SPEC §3.4 coupler_select, §3.8 fluid pairs, SCR-WP2-6, SCR-WP9-2): entry count, the strong
predicate m_a/m_s > 0.5 (strict), saturating actuators and static holds, fluid added mass C_a·ρ_f·V_sub with the
lightest body deciding, fluid pairs never ADMM, floating containers routed up to a measured ratio bound (a
named error above it) and SPH with two other entries refused."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import AssemblyError, RouteError
from alloy.route.types import Route
from alloy.scene.coupler_select import (FLOATING_CONTAINER_RATIO_MAX, BodyInfo, CouplerInputs, EntryMass,
                                        FloatingContainerError, StrongPairBesideLiquidError, fluid_added_mass,
                                        select_coupler, strong_pair)
from alloy.coupling.types import PairPolicy

RIGID = frozenset({Route.RIGID})
WATER = 1000.0


def rigid(mass: float, objects=("box",)) -> EntryMass:
    return EntryMass("mujoco", mass, RIGID, tuple(objects))


def cloth(mass: float) -> EntryMass:
    return EntryMass("vbd", mass, frozenset({Route.CLOTH_VBD}), ("towel",))


def water(mass: float = 1.0, name: str = "sph") -> EntryMass:
    return EntryMass(name, mass, frozenset({Route.LIQUID_SPH}), ("water",), fluid_density=WATER)


def ball(name: str, density: float, r: float = 0.02, **kw) -> BodyInfo:
    v = 4.0 / 3.0 * 3.141592653589793 * r**3
    return BodyInfo(name, density * v, v, **kw)


def test_single_entry_has_no_coupler():
    assert select_coupler(CouplerInputs([rigid(1.0)])).kind == "none"


def test_light_deformable_on_heavy_rigid_is_lagged_proxy():
    c = select_coupler(CouplerInputs([cloth(0.05), rigid(1.0)]))
    assert c.kind == "proxy" and not c.strong_unrouted
    (p,) = c.pairs
    assert (p.source, p.destination, p.mode) == ("mujoco", "vbd", "lagged")    # the rigid entry drives
    assert p.added_mass == pytest.approx(0.05)


@pytest.mark.parametrize("m_dst,strong", [(0.4, False), (0.5, False), (0.5000001, True), (2.0, True)])
def test_strong_threshold_is_strict_ratio_above_half(m_dst, strong):
    c = select_coupler(CouplerInputs([rigid(1.0), cloth(m_dst)]))
    assert (c.pairs[0].mode == "admm") is strong
    assert (c.kind == "admm") is strong
    assert bool(c.strong_unrouted) is strong


@pytest.mark.parametrize("flag", ["saturating", "static_hold"])
def test_saturating_actuator_or_static_hold_makes_a_pair_strong(flag):
    c = select_coupler(CouplerInputs([rigid(1.0), cloth(0.01)], **{flag: frozenset({("mujoco", "vbd")})}))
    assert c.pairs[0].mode == "admm" and c.strong_unrouted == (("mujoco", "vbd"),)


def test_precision_flag_is_carried():
    c = select_coupler(CouplerInputs([rigid(1.0), cloth(0.01)], precision=frozenset({("mujoco", "vbd")})))
    assert c.pairs[0].precision and c.pairs[0].mode == "lagged"


def test_three_entries_use_admm():
    """Three dynamic entries (a film sheet beside sand steps; cloth beside sand is refused: see below)."""
    mpm = EntryMass("mpm", 0.01, frozenset({Route.GRAIN_MPM}), ("sand",))
    film = EntryMass("film", 0.01, frozenset({Route.CLOTH_FILM}), ("wrap",))
    c = select_coupler(CouplerInputs([rigid(1.0), film, mpm]))
    assert c.kind == "admm" and len(c.pairs) == 3


def test_strong_pair_onto_mpm_stays_lagged():
    """WP12 (engine item 2): SolverImplicitMPM never reads ADMM's contact-row forces, so a strong body-MPM pair keeps
    the lagged proxy (its grid collider), still reported strong-unrouted."""
    sand = EntryMass("mpm", 2.0, frozenset({Route.GRAIN_MPM}), ("sand",))
    c = select_coupler(CouplerInputs([rigid(0.1, ("cup",)), sand]))
    assert c.kind == "proxy" and c.pairs[0].mode == "lagged"
    assert c.strong_unrouted == (("mujoco", "mpm"),)
    held = select_coupler(CouplerInputs([rigid(1.0, ("cup",)), sand], static_hold=frozenset({("mujoco", "mpm")})))
    assert held.kind == "proxy" and held.pairs[0].mode == "lagged" and held.strong_unrouted == (("mujoco", "mpm"),)


def test_soft_body_in_water_is_vbd_to_sph_staggered_and_strong():
    """SPH-VBD P1 (WP9): the liquid is the destination and a soft source always runs staggered (lagged soft walls
    are refused). Without the soft objects' volumes the added mass is C_a·m_s (neutral buoyancy): strong, ω0 0.1."""
    sponge = EntryMass("vbd", 0.05, frozenset({Route.SOFT_VBD}), ("sponge",))
    c = select_coupler(CouplerInputs([sponge, water()]))
    (p,) = c.pairs
    assert (p.source, p.destination, p.mode) == ("vbd", "sph", "staggered") and p.added_mass == pytest.approx(0.05)
    assert c.kind == "proxy" and c.strong_unrouted == (("vbd", "sph"),)


def test_light_soft_body_in_water_is_strong_with_its_volume():
    sponge = EntryMass("vbd", 0.05, frozenset({Route.SOFT_VBD}), ("sponge",))
    bag = BodyInfo("sponge", 0.05, 1.0e-3, container=True)          # floats (rho 50 << 1000): no refusal for soft
    c = select_coupler(CouplerInputs([sponge, water()], {"sponge": bag}))
    (p,) = c.pairs
    assert p.mode == "staggered" and p.added_mass == pytest.approx(WATER * 1.0e-3)
    assert c.strong_unrouted == (("vbd", "sph"),)


def test_fluid_added_mass_is_rho_f_times_volume():
    b = ball("b", 7850.0)
    assert fluid_added_mass(b, WATER) == pytest.approx(WATER * b.volume, rel=1e-15)


def test_dense_body_in_water_is_lagged():
    steel = ball("steel", 7850.0)                    # m_a/m_s = 1000/7850 = 0.127
    c = select_coupler(CouplerInputs([rigid(steel.mass, ("steel",)), water()], bodies={"steel": steel}))
    (p,) = c.pairs
    assert c.kind == "proxy" and p.mode == "lagged" and not c.strong_unrouted
    assert p.added_mass == pytest.approx(WATER * steel.volume)


def test_light_body_in_water_is_strong_but_never_admm():
    pp = ball("pp", 900.0)                           # m_a/m_s = 1.11 > 0.5
    c = select_coupler(CouplerInputs([rigid(pp.mass, ("pp",)), water()], bodies={"pp": pp}))
    (p,) = c.pairs
    assert p.mode == "staggered" and c.kind == "proxy"      # SCR-WP9-2: fluid pairs run staggered, not ADMM
    assert c.strong_unrouted == (("mujoco", "sph"),)


def test_lightest_dynamic_body_decides_the_fluid_pair():
    steel, cork = ball("steel", 7850.0), ball("cork", 240.0, r=0.01)
    fixed = BodyInfo("post", 0.001, 1.0, dynamic=False)     # static bodies never count
    entry = rigid(steel.mass + cork.mass, ("steel", "cork", "post"))
    c = select_coupler(CouplerInputs([entry, water()], bodies={"steel": steel, "cork": cork, "post": fixed}))
    (p,) = c.pairs
    assert p.added_mass == pytest.approx(WATER * cork.volume) and p.mode == "staggered"


def test_floating_container_routes_like_a_floating_solid_up_to_the_bound():
    """WP9 (a1): a floating container (V_sub = its outline, the air inside counted) routes as a strong body-liquid
    pair (staggered, never ADMM; the liquid's solver integrates its proxy) while D5's m_a/m_s <= the measured bound."""
    cup = BodyInfo("cup", 0.02, 2e-4, container=True)       # 20 g, 200 ml outline: m_a/m_s = 10
    c = select_coupler(CouplerInputs([rigid(0.02, ("cup",)), water()], bodies={"cup": cup}))
    (p,) = c.pairs
    assert c.kind == "proxy" and p.mode == "staggered" and p.added_mass == pytest.approx(WATER * 2e-4)
    assert c.strong_unrouted == (("mujoco", "sph"),)
    rng = np.random.default_rng(0)                          # exactly at the bound routes for any (ρ, V): WP4's S4
    for rho, vol in zip(rng.uniform(500.0, 2000.0, 200), rng.uniform(1e-5, 1e-2, 200)):
        edge = BodyInfo("cup", rho * vol / FLOATING_CONTAINER_RATIO_MAX, vol, container=True)
        fluid = EntryMass("sph", 1.0, frozenset({Route.LIQUID_SPH}), ("water",), fluid_density=rho)
        select_coupler(CouplerInputs([rigid(edge.mass, ("cup",)), fluid], bodies={"cup": edge}))


def test_floating_container_beyond_the_bound_is_a_named_error():
    """Above the bound (a 2 g cup of a 200 ml outline: 100x) the coupled step is not shown to hold: a
    FloatingContainerError (a RouteError) naming the ratio, the bound and the mass that would route."""
    cup = BodyInfo("cup", 0.002, 2e-4, container=True)
    with pytest.raises(FloatingContainerError, match=r"100x lighter.*mass >= 0.004 kg") as e:
        select_coupler(CouplerInputs([rigid(0.002, ("cup",)), water()], bodies={"cup": cup}))
    assert isinstance(e.value, RouteError)
    odd = BodyInfo("cup", 0.002, 2.0617e-4, container=True)        # needs 4.1234 g: printed rounded up, 4.13 g
    with pytest.raises(FloatingContainerError, match=r"mass >= 0.00413 kg"):
        select_coupler(CouplerInputs([rigid(0.002, ("cup",)), water()], bodies={"cup": odd}))
    ok = BodyInfo("cup", 0.00413, 2.0617e-4, container=True)        # and that mass routes
    select_coupler(CouplerInputs([rigid(0.00413, ("cup",)), water()], bodies={"cup": ok}))
    solid = BodyInfo("cup", 0.002, 2e-4)                    # not a container: a light solid routes as before
    select_coupler(CouplerInputs([rigid(0.002, ("cup",)), water()], bodies={"cup": solid}))


def _routes(c):
    return [(p.source, p.destination, p.mode, p.one_way) for p in c.pairs]


def test_hub_partner_and_sph_make_sph_a_one_way_follower():
    """SCR-DC5-7: MuJoCo hub + VBD (two-way, lagged) + SPH (one-way follower of the hub), for a partner without SPH
    boundary faces (a rope)."""
    rope = EntryMass("vbd", 0.1, frozenset({Route.ROD_VBD}), ("rope",))
    c = select_coupler(CouplerInputs([rigid(1.0), rope, water(0.03)]))
    assert c.kind == "proxy" and c.strong_unrouted == ()
    assert _routes(c) == [("mujoco", "vbd", "lagged", False), ("mujoco", "sph", "lagged", True)]


@pytest.mark.parametrize("route", [Route.CLOTH_VBD, Route.SOFT_VBD])
def test_hub_soft_partner_and_sph_make_the_p3_chain(route):
    """Engine item 3 P3 (option A, coordinator 11-06): a partner whose surface SPH takes as boundary faces couples to
    the liquid too: hub -> partner lagged, partner -> SPH staggered (strong without volumes: C_a·m_s), hub -> SPH
    one-way. One Scheduler steps the three entries (alloy.coupling.compose)."""
    bag = EntryMass("vbd", 0.1, frozenset({route}), ("bag",))
    c = select_coupler(CouplerInputs([rigid(1.0), bag, water(0.03)]))
    assert c.kind == "proxy"
    assert _routes(c) == [("mujoco", "vbd", "lagged", False), ("vbd", "sph", "staggered", False),
                          ("mujoco", "sph", "lagged", True)]
    assert c.strong_unrouted == (("vbd", "sph"),)
    assert c.pairs[1].added_mass == pytest.approx(0.1)


def test_a_light_rigid_hub_beside_cloth_and_liquid_is_refused_with_the_mass_that_routes():
    """Audit WORLDS-3: the hub pair is strong (cloth over rigid mass above 0.5) and ADMM cannot step a liquid, so
    the scene is refused. The error names the objects and the lightest rigid mass that routes (it said
    "mujoco/vbd is strong (D5) and needs ADMM, which has no one-way stage for SPH")."""
    sheet = EntryMass("vbd", 0.02, frozenset({Route.CLOTH_VBD}), ("sheet",))
    with pytest.raises(StrongPairBesideLiquidError) as err:
        select_coupler(CouplerInputs([rigid(0.0349, ("ball",)), sheet, water(0.03)]))
    text = str(err.value)
    assert "['ball'] weigh 0.0349 kg" in text and "['sheet'] (0.02 kg)" in text and "liquid" in text
    assert "at least 0.04 kg" in err.value.fix and "static=True or kinematic=True" in err.value.fix
    assert "liquid-cloth-light-rigid.md" in err.value.fix
    c = select_coupler(CouplerInputs([rigid(0.04, ("ball",)), sheet, water(0.03)]))        # that mass routes
    assert c.kind == "proxy" and _routes(c)[0] == ("mujoco", "vbd", "lagged", False)
    held = CouplerInputs([rigid(1.0, ("hand",)), sheet, water(0.03)], static_hold=frozenset({("mujoco", "vbd")}))
    with pytest.raises(StrongPairBesideLiquidError, match="a grip that must not creep") as err:
        select_coupler(held)
    assert "separate scenes" in err.value.fix


def test_p3_chain_uses_the_soft_volume():
    bag = EntryMass("vbd", 0.5, frozenset({Route.SOFT_VBD}), ("bag",))
    dense = BodyInfo("bag", 0.5, 1.0e-4)                     # 5000 kg/m^3: m_a/m_s = 0.2, not strong
    c = select_coupler(CouplerInputs([rigid(2.0), bag, water(0.03)], {"bag": dense}))
    assert c.pairs[1].mode == "staggered" and c.pairs[1].added_mass == pytest.approx(WATER * 1.0e-4)
    assert c.strong_unrouted == ()


def test_sph_beside_two_two_way_partners_raises():
    """A liquid beside a rigid hub and two deformable solvers: refused, every solver's objects named (audit S2-009)."""
    sand = EntryMass("mpm", 0.2, frozenset({Route.GRAIN_MPM}), ("sand",))
    film = EntryMass("film", 0.1, frozenset({Route.CLOTH_FILM}), ("wrap",))
    with pytest.raises(AssemblyError, match="not supported in one scene yet") as err:
        select_coupler(CouplerInputs([rigid(1.0), film, sand, water()]))
    assert all(n in str(err.value) for n in ("water", "wrap", "sand")) and "keep only one of" in err.value.fix


def test_sph_beside_a_strong_hub_pair_raises():
    with pytest.raises(AssemblyError, match="tighter coupling"):
        select_coupler(CouplerInputs([rigid(1.0), cloth(0.9), water()]))      # 0.9 / 1.0 > 0.5


def test_strong_pair_predicate():
    pol = PairPolicy("a", "b")
    assert not strong_pair(0.5, 1.0, pol) and strong_pair(0.51, 1.0, pol)
    assert strong_pair(1.0, 0.0, pol)                       # massless source: infinite ratio
    assert strong_pair(0.0, 1.0, PairPolicy("a", "b", saturating_actuator=True))


def test_deformable_objects_get_their_emitted_volume():
    """SPH-VBD (engine item 3, WP4): D5 needs a soft source's volume for its fluid added mass, so deformable objects
    get a BodyInfo too. The volume is the emitted one (particle mass / density): the sponge's box volume, the towel's
    area x thickness. Rigid objects keep their shape volume."""
    from alloy.route.router import Router
    from alloy.scene.assembler import SceneAssembler
    from alloy.scene.finish import coupler_inputs
    from alloy.testing import scene as S

    card, lib = S.scene([S.table(), S.box(), S.sponge(), S.cloth()],
                        [S.place("table"), S.place("box", z=0.051), S.place("sponge", x=0.15, z=0.045),
                         S.place("towel", x=-0.15, z=0.03)], worlds=1)
    built = SceneAssembler(lib, Router()).assemble(card, "cpu")
    assert built.handles["sponge"].decision.route == Route.SOFT_VBD
    bodies = coupler_inputs(built._plan, built.model).bodies
    assert bodies["sponge"].volume == pytest.approx(0.08 * 0.06 * 0.04, rel=1e-3)
    assert bodies["towel"].volume == pytest.approx(0.2 * 0.2 * 5e-4, rel=1e-3)
    assert bodies["box"].volume == pytest.approx(0.06**3, rel=1e-6)
    assert bodies["sponge"].dynamic and not bodies["sponge"].container
    assert "table" not in bodies                                # static: no entry, as before


def test_select_coupler_refuses_cloth_beside_sand_naming_both():
    """Audit S1-001 of 0.1.2: a route pair no solver steps together is refused at the coupler choice, which estimate
    and the quick check reach too (they assemble), naming the objects."""
    from alloy.coupling.unsupported import UnsupportedPairError

    sand = EntryMass("mpm", 0.01, frozenset({Route.GRAIN_MPM}), ("sand",))
    with pytest.raises(UnsupportedPairError) as err:
        select_coupler(CouplerInputs([rigid(1.0), cloth(0.01), sand]))
    assert "towel" in str(err.value) and "sand" in str(err.value)

