# SPDX-License-Identifier: Apache-2.0
"""Router step 2 (phase: liquid, paste, granular) and the rod branch of step 5 (CHAIN vs ROD_VBD, D7/D8).

Feature sizes come from the partners the object meets (`SceneContext.partners`, SCR-WP2-4; the rigid
partners' `Features.passage_width` and their middle OBB extent as the container inner size) and the object itself.
A rod or cable (a curve card) is no container: its middle extent is its diameter, so it sets no inner size (WP13's
combination sweep F11: a 4 mm lace beside 150 ml of paste asked for 1,041,859 MPM particles). Nor is a moving solid
primitive (a free or kinematic box, sphere, capsule, cylinder, cone or ellipsoid: a tool, a stirrer, a dropped part):
its middle extent is its own width (SOFT-2, SC10: a kinematic 5 mm rod dragged through mayonnaise in a tub asked for
830,208 MPM particles, a 1.5 mm whisk wire 31 M). Static boxes still count: add_open_box's walls are the cavity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from alloy.cards.types import CurveSource, ParticleSource, Primitive, ResolvedObject
from alloy.core.errors import CapacityError, RouteError
from alloy.route import checks as ck
from alloy.route import rodcheck
from alloy.route.types import LIQUID_SPACING, Route, RouterConfig, SceneContext

if TYPE_CHECKING:
    from alloy.route.router import Router, _Path

log = logging.getLogger("alloy.route")


def _partners(obj: ResolvedObject, ctx: SceneContext) -> list[ResolvedObject]:
    table = ctx.partners
    names = ctx.rigid_partners.get(obj.card.name, ())
    return [table[n] for n in names if n in table]


def feature_size(obj: ResolvedObject, ctx: SceneContext) -> tuple[float, float | None]:
    """(d_feature, passage) = min over partner passage widths and container inner sizes, and the passage alone."""
    passages = [float(o.card.features.passage_width.value) for o in _partners(obj, ctx)
                if o.card.features.passage_width is not None]
    if obj.card.features.passage_width is not None:
        passages.append(float(obj.card.features.passage_width.value))
    inner = [float(o.extents[1]) for o in _partners(obj, ctx) if _may_contain(o)]
    sizes = passages + inner + [float(obj.extents[2])]
    return min(sizes), (min(passages) if passages else None)


_SOLIDS = ("box", "sphere", "capsule", "cylinder", "cone", "ellipsoid")


def _may_contain(o: ResolvedObject) -> bool:
    """Whether a rigid partner's middle extent can be a container's inner size: not a rod (curve card), not a moving
    solid primitive (module docstring); static parts and meshes (cups, bowls) can."""
    if isinstance(o.card.shape, CurveSource):
        return False
    moving_solid = isinstance(o.card.shape, Primitive) and o.card.shape.kind in _SOLIDS and o.card.motion != "static"
    return not moving_solid


def _volume(obj: ResolvedObject) -> float:
    from alloy.scene.emitters.common import object_volume

    return object_volume(obj)


def route_phase(cfg: RouterConfig, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route | None:
    """Liquid → SPH (or MPM, approximate); paste → MPM; granular → MPM or rigid grains. None for solids. An MPM route's
    reasons end with the rheology solve tolerance it builds with: read from the environment, which Scene JSON does not
    carry (``alloy.integration.mpm_solve``; WP2's review)."""
    route = _route_phase(cfg, obj, ctx, p)
    if route in (Route.GRAIN_MPM, Route.PASTE_MPM, Route.LIQUID_MPM):
        from alloy.integration.mpm_solve import describe  # noqa: PLC0415

        p.reasons.append(f"{route}: {describe()}")
    return route


def _route_phase(cfg: RouterConfig, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route | None:
    from alloy.route.router import worst

    mat = obj.material
    plastic = mat.plastic
    yield_stress = worst(obj, "material.plastic.yield_stress", 0.0) or 0.0 if plastic is not None else 0.0
    if mat.fluid is not None:
        mu = worst(obj, "material.fluid.viscosity")
        t = ck.Threshold("viscosity", mu, cfg.liquid_viscosity_max)
        p.thresholds.append(t)
        p.numbers["viscosity"] = mu
        if yield_stress <= 0.0 and t.passed:
            return _liquid(cfg, obj, ctx, p, mu)
        p.reasons.append(f"paste: yield stress {yield_stress:.3g} Pa or viscosity {mu:.3g} Pa·s > "
                         f"{cfg.liquid_viscosity_max}")
        return _mpm(cfg, obj, ctx, p, Route.PASTE_MPM)
    if plastic is not None and plastic.friction_angle is not None:
        return _granular(cfg, obj, ctx, p)
    if yield_stress > 0.0:
        p.reasons.append(f"paste: yield stress {yield_stress:.3g} Pa")
        return _mpm(cfg, obj, ctx, p, Route.PASTE_MPM)
    return None


SPH_REST_JITTER_SPEED = 1.5e-3      # m/s, SolverDFSPH's rest_jitter_speed (SCR-WP9-8); the rule runs in the solver


def _liquid(cfg: RouterConfig, obj: ResolvedObject, ctx: SceneContext, p: _Path, mu: float) -> Route:
    if str(Route.LIQUID_SPH) not in ctx.registered_routes:
        p.reasons.append("liquid_mpm (approximate, D15): no liquid_sph capability registered")
        return _mpm(cfg, obj, ctx, p, Route.LIQUID_MPM)
    _, passage = feature_size(obj, ctx)
    r = 0.5 * LIQUID_SPACING[cfg.liquid_accuracy]       # the setting's spacing (SCR-WP9-LIQRES)
    if cfg.sph_refine_passage:                           # LIQ-6: finer below 8 particles across a passage
        r = ck.sph_radius(passage, r, cfg.sph_passage_diameters)
    spacing = 2.0 * r
    n = ck.sph_particle_count(_volume(obj), r)
    p.numbers.update(particle_radius=r, n_particles=float(n), viscosity_weiler=float(mu > cfg.sph_viscous_above),
                     rest_jitter_speed=SPH_REST_JITTER_SPEED, liquid_spacing=spacing)
    if passage is not None:
        p.numbers["passage_width"] = float(passage)      # the build warns under 8 particles across (_liquid_accuracy)
    if n > obj.card.budget.particles:
        raise CapacityError(f"{obj.card.name}: {n} SPH particles at r = {r:.3g} m > budget "
                            f"{obj.card.budget.particles}")
    model = "weiler2018" if mu > cfg.sph_viscous_above else "xsph"
    p.reasons.append(f"liquid_sph: liquid_accuracy {cfg.liquid_accuracy!r}, spacing {spacing * 1e3:g} mm "
                     f"(r = {r:.3g} m), {n} particles, viscosity {model}; density tolerance min(1e-3, "
                     f"max({SPH_REST_JITTER_SPEED * 1e3:g} mm/s, 0.5 U)·dt/H) per world, U the liquid's mean speed "
                     "(SCR-WP9-8; the effective value: SolverDFSPH.report().density_tol)")
    return Route.LIQUID_SPH


def _mpm(cfg: RouterConfig, obj: ResolvedObject, ctx: SceneContext, p: _Path, route: Route) -> Route:
    d_feature, _ = feature_size(obj, ctx)
    voxel = ck.mpm_voxel(d_feature, cfg.mpm_voxel_max)
    p.numbers.update(voxel=voxel, d_feature=d_feature)
    p.thresholds.append(ck.Threshold("mpm_voxel_max", 0.25 * d_feature, cfg.mpm_voxel_max))
    p.reasons.append(f"{route}: voxel = min(d_feature/4, {cfg.mpm_voxel_max}) = {voxel:.3g} m")
    return route


def _granular(cfg: RouterConfig, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route:
    d_feature, _ = feature_size(obj, ctx)
    voxel = ck.mpm_voxel(d_feature, cfg.mpm_voxel_max)
    src = obj.card.shape
    d_grain = getattr(src, "grain_diameter", None) if isinstance(src, ParticleSource) else None
    p.numbers.update(voxel=voxel, d_feature=d_feature)
    if d_grain is None:
        p.reasons.append("grain_mpm: no grain_diameter on the card, continuum assumed")
        return Route.GRAIN_MPM
    t = ck.Threshold("voxel/grain", voxel / d_grain, cfg.mpm_voxel_grain_min)
    p.thresholds.append(t)
    p.numbers["grain_diameter"] = d_grain
    if not t.passed or math.isclose(t.value, t.limit):
        p.reasons.append(f"grain_mpm: voxel {voxel:.3g} m ≥ {cfg.mpm_voxel_grain_min}·d_grain")
        return Route.GRAIN_MPM
    n = ck.grain_count(_volume(obj), d_grain)
    p.numbers["n_grains"] = float(n)
    p.thresholds.append(ck.Threshold("n_grains", float(n), float(cfg.grain_rigid_max)))
    if n <= cfg.grain_rigid_max:
        p.reasons.append(f"grain_rigid: {n} grains ≤ {cfg.grain_rigid_max}, voxel < 3·d_grain")
        return Route.GRAIN_RIGID
    if ck.mpm_voxel(d_feature, math.inf) < cfg.mpm_voxel_grain_min * d_grain:     # the feature, not the voxel cap
        raise RouteError(f"{obj.card.name}: jamming regime exceeds rigid budget ({n} grains > {cfg.grain_rigid_max}, "
                         f"narrowest feature {d_feature:.3g} m = {d_feature / d_grain:.1f} grains of {d_grain:.3g} m)")
    return _grain_continuum(obj, p, n, cfg.grain_rigid_max, voxel / d_grain)


def _grain_continuum(obj: ResolvedObject, p: _Path, n: int, rigid_max: int, voxel_grains: float) -> Route:
    """Above the rigid-grain budget, with a voxel under 3 grains only because of the voxel cap: GRAIN_MPM as a
    continuum (SCR-SOFT-2-3; was a RouteError). The MPM resolves the bulk flow and the friction angle, not single
    grains (a 3 mm rice grain in a 5 mm voxel). A feature under 4 x 3 grains wide (an orifice a few grains across,
    where grains arch and jam) still raises in ``_granular`` (WP2's review)."""
    why = (f"grain_mpm (continuum, SCR-SOFT-2-3): {n} grains > {rigid_max} rigid, voxel {voxel_grains:.2f} grains "
           "(< 3): the bulk flows as a continuum, single grains are not resolved")
    p.reasons.append(why)
    log.warning("%s: %s", obj.card.name, why)
    return Route.GRAIN_MPM


def _asks_rod_vbd(obj: ResolvedObject) -> bool:
    """The card's route override names ROD_VBD (a clamp: ``Scene.add_rod(clamp=)`` writes it)."""
    return obj.card.route is not None and str(obj.card.route.route) == str(Route.ROD_VBD)


def _asks_chain(obj: ResolvedObject) -> bool:
    """The card's route override names CHAIN (``Scene.add_rod(route="chain")`` writes it)."""
    return obj.card.route is not None and str(obj.card.route.route) == str(Route.CHAIN)


def _rod_vbd(obj: ResolvedObject, sec, links: int, override: int | None, p: _Path) -> Route:
    """ROD_VBD: Newton's Cosserat rod on SolverVBDRod. Segments as CHAIN counts links (the card's chain_links, else
    raised until the discrete chain sags within 5 %), with no CHAIN cap and no MuJoCo self-test: the direct solve is
    exact per chain, and the route's own laws (A11.*.rod_vbd) judge it."""
    feats = obj.card.features
    rod = rodcheck.rod_card(sec.length, sec.radius, sec.area, sec.inertia, obj.material, links,
                            self_contact=feats.self_contact, features=feats)
    if override is None:
        rod = replace(rod, links=rodcheck.sag_links(rod, ck.max_chain_links(sec.length, sec.radius, 10**6)))
    p.numbers.update(rodcheck.rod_numbers(rod))
    p.reasons.append(f"rod_vbd: {rod.links} segments, γ = {rod.gamma:.3g}, Cosserat rod on SolverVBDRod")
    return Route.ROD_VBD


def rod_route(router: Router, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route:
    """CHAIN if the required count ⌈L/c⌉+1 ≤ cap, else ROD_VBD (coordinator ruling 2026-11-08 11:53Z, option A: the
    elastic rod wherever CHAIN cannot do the job; was a RouteError, D8). A card override naming ROD_VBD (a clamp)
    goes there at any count. Links: the card's features.chain_links, else the required count raised until the
    discrete chain sags within 5 % (SCR-DC3-7)."""
    cfg = router.cfg
    sec = router._section(obj)
    rho = float(obj.material.density.value)
    EI = p.numbers["EI"]
    c = ck.rod_bending_length(rho, sec.area, EI)
    override = getattr(obj.card.features, "chain_links", None)      # card override (SCR-DC3-7, WP1's field)
    links, required = ck.chain_links(sec.length, c, sec.radius, override, cfg.chain_links_min)
    p.numbers.update(links_required=float(required), bending_length=c)
    p.thresholds.append(ck.Threshold("links", float(required), float(cfg.chain_links_max)))
    if _asks_rod_vbd(obj):
        return _rod_vbd(obj, sec, links, override, p)
    plastic = obj.material.plastic
    yields = plastic is not None and plastic.model == "j2"
    if yields and not _asks_chain(obj):                          # the ruling's third case: CHAIN has no plasticity
        p.reasons.append("rod_vbd: the material yields (plastic.model j2), and only the elastic rod keeps a set")
        return _rod_vbd(obj, sec, links, override, p)
    if yields:      # route="chain" asked: the chain's own links and self-test, not rod_vbd's (those blew up at once)
        p.reasons.append("chain asked: the material yields (plastic.model j2); the chain is elastic, keeps no set")
    if required > cfg.chain_links_max:
        p.reasons.append(f"rod_vbd: needs {required} links > CHAIN's {cfg.chain_links_max} (bending length {c:.3g} m "
                         f"over {sec.length:.3g} m)")
        return _rod_vbd(obj, sec, links, override, p)
    integ = "implicitfast" if ctx.needs_l1 else "discrete"
    feats = obj.card.features
    rod = rodcheck.rod_card(sec.length, sec.radius, sec.area, sec.inertia, obj.material, links,
                            integrator=integ, self_contact=feats.self_contact, features=feats)
    if override is None:                # SCR-DC3-7: up from ⌈L/c⌉ + 1 until the discrete chain sags within 5 %
        links = rodcheck.sag_links(rod, ck.max_chain_links(sec.length, sec.radius, links))
        rod = replace(rod, links=links)
    p.reasons.append(_links_reason(links, required, c, override))
    result = dict(router._self_test(rod, obj.card_hash))
    if integ == "implicitfast" and not rodcheck.passes(result):
        p.reasons.append("implicitfast chain failed its self-test: discrete, gap-ladder L1 unavailable (D7)")
        rod = rodcheck.rod_card(sec.length, sec.radius, sec.area, sec.inertia, obj.material, links, features=feats)
        result = dict(router._self_test(rod, obj.card_hash))
        integ = "discrete"
    if override is None and not rodcheck.passes(result):
        rod, result = _raise_until_pass(router, obj, rod, result, ck.max_chain_links(sec.length, sec.radius, 10**6))
        if rod.links != links:
            p.reasons.append(f"links: {links} failed the self-test, {rod.links} pass (SCR-DC3-7)")
        links = rod.links
    if result.get("finite", 1.0) <= 0.0:
        raise RouteError(f"{obj.card.name}: CHAIN self-test blew up at {links} links ({integ}, dt "
                         f"{router.chain_dt} s)")
    p.self_test = result
    p.numbers.update(rodcheck.rod_numbers(rod))
    p.numbers["chain_dt"] = router.chain_dt
    p.effects.append(f"mujoco:integrator={integ}")
    p.reasons.append(f"chain: {links} links, γ = {rod.gamma:.3g}, self-test sag {100 * result['sag_err']:+.2f} %"
                     + ("" if rodcheck.passes(result) else " (FAILED)"))
    return Route.CHAIN


LINKS_RETRY_CEILING = 30        # the old floor: every library cable passed its self-test there


def _raise_until_pass(router: Router, obj: ResolvedObject, rod, result: dict, most: int) -> tuple:
    """SCR-DC3-7: while the self-test fails (finite), ~25 % more links per try, up to LINKS_RETRY_CEILING (or `most`,
    segments no shorter than a diameter). The sag is already within bound (`rodcheck.sag_links`); this covers the
    release numbers: sc2's 4 mm leads at 5 links ran period +3.9 % (bound 2 %)."""
    top = min(most, max(LINKS_RETRY_CEILING, rod.links))
    while not rodcheck.passes(result) and result.get("finite", 1.0) > 0.0 and rod.links < top:
        rod = replace(rod, links=min(top, max(rod.links + 1, math.ceil(1.25 * rod.links))))
        result = dict(router._self_test(rod, obj.card_hash))
    return rod, result


def _links_reason(links: int, required: int, c: float, override: int | None) -> str:
    if override is not None:
        return (f"links: {links} from the card's features.chain_links {override}"
                + (f", below the sag rule's {required}" if links < required else "") + " (SCR-DC3-7)")
    why = "" if links == required else f", raised from {required} until the discrete chain sags within 5 %"
    return f"links: {links}, sag rule ⌈L/c⌉ + 1 = {required} at bending length {c * 1e3:.3g} mm{why} (SCR-DC3-7)"
