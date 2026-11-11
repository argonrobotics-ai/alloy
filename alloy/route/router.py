# SPDX-License-Identifier: Apache-2.0
"""Router: physics checks → RouteDecision (SPEC §3.3 steps 1-7, D7, D8, D15, D16), evaluated at `obj.worst_case`.

Numbers in `RouteDecision.numbers` are SI. `margin` is the smallest log10 distance over the thresholds evaluated on
the decision path; `near_boundary` = margin < `RouterConfig.near_boundary_log10`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from alloy.cards.types import CurveSource, ParticleSource, ResolvedObject
from alloy.core.errors import CapacityError, RouteError
from alloy.route import checks as ck
from alloy.route import rodcheck
from alloy.route.articulated import articulated_joints
from alloy.route.types import RodCard, Route, RouteDecision, RouterConfig, SceneContext
from alloy.scene import cloth_accuracy, derive, stiff_sheet

log = logging.getLogger("alloy.route")
_J2_FALLBACK_WARNED: set[str] = set()      # card names warned once per process (estimate + build, variants: WP2)

ROUTE_ENTRY: dict[Route, str] = {
    Route.STATIC: "", Route.RIGID: "mujoco", Route.RIGID_COMPLIANT: "mujoco", Route.CHAIN: "mujoco",
    Route.PANELS: "mujoco", Route.GRAIN_RIGID: "mujoco", Route.CLOTH_VBD: "vbd", Route.SOFT_VBD: "vbd",
    Route.ROD_VBD: "vbd", Route.CLOTH_STYLE3D: "style3d", Route.GRAIN_MPM: "mpm", Route.PASTE_MPM: "mpm",
    Route.LIQUID_MPM: "mpm", Route.LIQUID_SPH: "sph", Route.CLOTH_FILM: "film",
}
SelfTest = Callable[[RodCard, str], Mapping[str, float]]


@dataclass
class _Path:
    """Decision being built: route, numbers, thresholds and reasons accumulated along the router steps."""

    numbers: dict[str, float] = field(default_factory=dict)
    thresholds: list[ck.Threshold] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    self_test: Mapping[str, float] | None = None
    articulated: str = ""                   # card name of a fixed-base articulated MJCF object (SCR-DC4-2)


def worst(obj: ResolvedObject, path: str, default: float | None = None) -> float | None:
    """Value of a dotted card path at the routing-worst end of its randomization range, else the card value.

    `path` starts at the ResolvedObject ("material.elastic.youngs", "card.loads.tolerance"); `worst_case` keys may
    also omit the leading "card." (Randomization.field paths on the object card)."""
    for key in (path, path.removeprefix("card.")):
        if key in obj.worst_case:
            return float(obj.worst_case[key])
    node: object = obj
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return default
    return float(getattr(node, "value", node))


def squeeze_depth(obj: ResolvedObject) -> float:
    """The depth the local-squeeze check reads: the OBB's l3, or the card's thickness when the shape is perfectly flat
    (a flat cloth mesh has l3 = 0; XB4-10's drape disc raised ZeroDivisionError). A flat shape without a thickness
    has no depth to judge and is refused."""
    l3 = obj.extents[2]
    if l3 > 0.0:
        return l3
    t = worst(obj, "card.features.thickness", 0.0)
    if not t or t <= 0.0:
        raise RouteError(f"{obj.card.name}: the shape is flat (OBB l3 = 0) and the card has no thickness; give "
                         "features.thickness (a cloth's sheet thickness, e.g. 2 mm)")
    return t


class Router:
    def __init__(self, cfg: RouterConfig = RouterConfig(), *, rod_self_test: SelfTest | None = None,
                 chain_dt: float = 1.25e-3, entries: Mapping[Route, str] | None = None) -> None:
        """`entries` overrides the capability serving a route (e.g. {Route.RIGID: "featherstone"} for bench runs on
        an alternative solver); the default map is ROUTE_ENTRY."""
        self.cfg = cfg
        self.chain_dt = chain_dt
        self.entries = {**ROUTE_ENTRY, **dict(entries or {})}
        self._self_test = rod_self_test or (lambda rod, h: rodcheck.self_test(rod, h, chain_dt))

    def route(self, obj: ResolvedObject, ctx: SceneContext) -> RouteDecision:
        p = _Path()
        route = self._route(obj, ctx, p)
        if route == Route.CLOTH_VBD:
            stiff_sheet.tag(obj, p.effects, p.reasons)          # a stiff sheet starts from the inertial guess
        if obj.card.route is not None and obj.card.route.route == str(Route.CLOTH_FILM) and _is_j2(obj) \
                and str(Route.CLOTH_FILM) not in ctx.registered_routes:          # add_cloth's j2 sheet, no film
            _no_film(obj, p, f"the card asks for cloth_film ({obj.card.route.reason}), but the film plug-in is not "
                             f"registered: it stays {route} and keeps no dent")
            return self._decision(route, ctx, p)
        if obj.card.route is not None and obj.card.route.route == str(Route.PASTE_MPM) and _is_j2(obj):
            self._j2_solid_on_mpm(obj, ctx, p)  # GAP #6: the MPM numbers emit needs, the opt-in's reason and effect
        if obj.card.route is not None:          # the override first; the natural route's reasons follow it
            p.reasons.insert(0, f"override {route} -> {obj.card.route.route}: {obj.card.route.reason} (the reasons "
                                f"after this line are the {route} pick it replaced)")
            return self._decision(Route(obj.card.route.route), ctx, p, overridden=True)
        return self._decision(route, ctx, p)

    def _j2_solid_on_mpm(self, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> None:
        """A j2 solid whose card overrides to PASTE_MPM runs on implicit MPM as a von Mises solid (GAP #6 slice 1,
        opt-in): the voxel numbers a paste gets (`phase._mpm`), a reason, and the tight-solve effect
        (`mpm_solid.EFFECT`). Solid volumes only: a j2 sheet or rod raises."""
        from alloy.route.phase import _mpm
        from alloy.solvers.plastic import mpm_solid

        morph, _ = ck.morphology(obj.extents, isinstance(obj.card.shape, CurveSource),
                                 obj.card.features.thickness is not None)
        if morph != "volume":
            raise RouteError(f"{obj.card.name}: the route override paste_mpm for a plastic.model j2 material takes "
                             f"solid volumes only, not a {morph}. Fix: drop the override (sheets: cloth_film, "
                             "rods: chain / rod_vbd)")
        _mpm(self.cfg, obj, ctx, p, Route.PASTE_MPM)
        p.effects.append(mpm_solid.EFFECT)
        p.reasons.append("j2 solid on implicit MPM as a von Mises solid (mpm yield_stress = sigma_y / sqrt(3), "
                         "friction 0, tension held), opt-in by the card's route override, because soft_vbd yields "
                         "spuriously below sigma_y / E 10 % "
                         "(docs/known_gaps/plasticity-soft-tets-low-yield.md); the entry's "
                         f"rheology solve (all MPM materials in it) runs at tolerance {mpm_solid.TOLERANCE:g}, at most "
                         f"{mpm_solid.MAX_ITERATIONS} iterations per step")
        if (text := mpm_solid.stiffness_warning(obj.card.name, obj.material, p.numbers["voxel"])) is not None:
            p.reasons.append(text)
            _warn_once(obj.card.name, text)

    def _decision(self, route: Route, ctx: SceneContext, p: _Path, overridden: bool = False) -> RouteDecision:
        entry = self._articulated_entry(p.articulated, ctx) if route == Route.STATIC and p.articulated \
            else self._entry(route, ctx)
        if route == Route.CLOTH_VBD:
            acc = self.cfg.cloth_accuracy
            p.effects.append(cloth_accuracy.EFFECT[acc])
            p.reasons.append(f"cloth_accuracy {acc}: the VBD entry runs {dict(cloth_accuracy.KWARGS[acc])}")
        margin = ck.min_margin(p.thresholds)
        p.numbers.update({f"thr:{t.name}": t.value / t.limit for t in p.thresholds if t.limit > 0})
        return RouteDecision(route=route, entry=entry, numbers=dict(p.numbers), margin=margin,
                             near_boundary=margin < self.cfg.near_boundary_log10, reasons=tuple(p.reasons),
                             scene_effects=tuple(p.effects), overridden=overridden, self_test=p.self_test)

    def _entry(self, route: Route, ctx: SceneContext) -> str:
        if route == Route.RIGID_PILE:
            return self.cfg.pile_backend
        entry = self.entries[route]
        if entry and str(route) not in ctx.registered_routes:
            fix = "; register the film plug-in: alloy.solvers.film.plugin.register()" \
                if route == Route.CLOTH_FILM else ""
            raise RouteError(f"route {route} has no registered solver capability (registered: "
                             f"{sorted(ctx.registered_routes)}){fix}")
        return entry

    # --- steps --------------------------------------------------------------------------------------------------

    def _articulated_entry(self, name: str, ctx: SceneContext) -> str:
        """The MuJoCo entry for a fixed-base articulated object; it alone steps the nested joints (SCR-DC4-2)."""
        entry = self.entries[Route.RIGID]
        if entry != "mujoco" or str(Route.RIGID) not in ctx.registered_routes:
            raise RouteError(f"{name}: motion=static with nested MJCF joints needs the mujoco entry to step them "
                             f"(rigid route served by {entry!r}, registered routes {sorted(ctx.registered_routes)})")
        return entry

    def _route(self, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route:
        if obj.card.motion in ("static", "kinematic"):
            joints = articulated_joints(obj)
            if joints:
                p.articulated = obj.card.name
                p.numbers["articulated_joints"] = float(len(joints))
                p.reasons.append(f"motion=static with {len(joints)} nested MJCF joint(s) "
                                 f"({', '.join(sorted(set(joints)))}): fixed base, the mujoco entry steps the "
                                 "articulated bodies (SCR-DC4-2)")
            else:
                p.reasons.append(f"motion={obj.card.motion}: shapes in the shared pipeline, no entry")
            return Route.STATIC
        phase = self._phase(obj, ctx, p)
        if phase is not None:
            return phase
        morph, ratios = ck.morphology(obj.extents, isinstance(obj.card.shape, CurveSource),
                                      obj.card.features.thickness is not None)
        p.thresholds.extend(ratios)
        p.reasons.append(f"morphology {morph} (l = {tuple(round(x, 6) for x in obj.extents)})")
        curve = isinstance(obj.card.shape, CurveSource)
        if obj.material.elastic is None:
            if curve:       # the rigid emitter has no shape for a curve
                raise RouteError(f"{obj.card.name}: a rod needs a material that bends, and {obj.material.name} has no "
                                 "elastic data. Fix: a rod material such as copper_c110 or cable_usb_composite, or "
                                 "add_capsule for a straight rigid bar")
            p.reasons.append("material.elastic is None: rigid")
            return self._rigid_split(Route.RIGID, ctx, p)
        rigid = self._rigidity(obj, morph, p)
        if rigid is not None and curve:     # a stiff rod keeps the rod routes (and its .points keys): no rigid curve
            p.reasons.append(f"{rigid} by stiffness, but a curve has no rigid shape: the rod routes step it")
            return self._rod(obj, ctx, p)
        if rigid is not None:
            return self._rigid_split(rigid, ctx, p)
        return self._deformable(obj, morph, ctx, p)

    def _phase(self, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route | None:
        from alloy.route.phase import route_phase

        plastic = obj.material.plastic
        if plastic is not None and plastic.model == "j2":   # a solid that yields keeps its solid route (SCR-WP7-15)
            p.reasons.append("plastic.model j2: a solid that yields, not a paste")
            return None
        return route_phase(self.cfg, obj, ctx, p)

    def _rigidity(self, obj: ResolvedObject, morph: str, p: _Path) -> Route | None:
        """Step 4: RIGID / RIGID_COMPLIANT, or None (deformable)."""
        E = worst(obj, "material.elastic.youngs")
        nu = worst(obj, "material.elastic.poisson")
        tau = worst(obj, "card.loads.tolerance", 1e-4)
        F_g = worst(obj, "card.loads.grip_force", 20.0)
        d_loc = ck.local_squeeze(F_g, E, nu, squeeze_depth(obj))
        p.numbers.update(delta_loc=d_loc, tolerance=tau, grip_force=F_g)
        bend_ok = self._bending(obj, morph, E, nu, tau, p)
        t_rigid = ck.Threshold("delta_loc/rigid", d_loc, self.cfg.rigid_ratio * tau)
        t_comp = ck.Threshold("delta_loc/compliant", d_loc, tau)
        p.thresholds.extend([t_rigid, t_comp])
        if bend_ok and t_rigid.passed:
            p.reasons.append(f"rigid: δ_loc {d_loc:.3g} m ≤ {self.cfg.rigid_ratio}·τ, bending passes")
            return Route.RIGID
        if bend_ok and t_comp.passed:
            p.numbers["k_compliant"] = F_g / d_loc
            p.reasons.append(f"rigid_compliant: δ_loc {d_loc:.3g} m ≤ τ, K = F_g/δ_loc = {F_g / d_loc:.3g} N/m")
            return Route.RIGID_COMPLIANT
        p.reasons.append("deformable: " + ("local squeeze" if bend_ok else "whole-body bending") + " exceeds τ")
        return None

    def _bending(self, obj: ResolvedObject, morph: str, E: float, nu: float, tau: float, p: _Path) -> bool:
        F_b = worst(obj, "card.loads.bend_force", 2.0)
        rho = worst(obj, "material.density")
        if morph == "sheet":
            h = worst(obj, "card.features.thickness", obj.extents[2])
            D = derive.sheet_bending(obj.material, E, nu, h)    # the card's bending, not E·h³ at h (R2-E1)
            d_b = ck.sheet_bend(F_b, 0.5 * obj.extents[1], D)
            p.numbers.update(delta_bend=d_b, plate_D=D, thickness=h, bending_length=ck.sheet_bending_length(D, rho, h))
            t = ck.Threshold("delta_bend", d_b, tau)
            p.thresholds.append(t)
            return t.passed
        if morph == "rod":
            sec = self._section(obj)
            bend = getattr(obj.card.features, "bend_stiffness", None)
            EI = float(bend.value) if bend is not None else E * sec.inertia   # SCR-WP1-7 measured composite EI
            d_b, gamma = ck.rod_bend(F_b, sec.length, EI), ck.rod_gamma(rho, sec.area, sec.length, EI)
            p.numbers.update(delta_bend=d_b, gamma=gamma, EI=EI, bending_length=ck.rod_bending_length(rho, sec.area, EI))
            tb, tg = ck.Threshold("delta_bend", d_b, tau), ck.Threshold("gamma", gamma, self.cfg.rod_sag_rigid)
            p.thresholds.extend([tb, tg])
            return tb.passed and tg.passed
        return True

    @staticmethod
    def _section(obj: ResolvedObject) -> ck.RodSection:
        shape = obj.card.shape
        if isinstance(shape, CurveSource):
            pts = shape.points
            length = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
            return ck.rod_section(None, (), obj.extents, length, shape.radius)
        return ck.rod_section(getattr(shape, "kind", None), tuple(getattr(shape, "size", ())), obj.extents)

    def _rigid_split(self, route: Route, ctx: SceneContext, p: _Path) -> Route:
        """Step 6: more than `pile_threshold` free bodies per world → RIGID_PILE on `pile_backend`."""
        t = ck.Threshold("free_bodies", float(ctx.free_bodies_per_world), float(self.cfg.pile_threshold))
        p.thresholds.append(t)
        p.numbers["free_bodies_per_world"] = float(ctx.free_bodies_per_world)
        if ctx.free_bodies_per_world > self.cfg.pile_threshold:
            p.reasons.append(f"rigid_pile: {ctx.free_bodies_per_world} free bodies > {self.cfg.pile_threshold}")
            return Route.RIGID_PILE
        return route

    def _deformable(self, obj: ResolvedObject, morph: str, ctx: SceneContext, p: _Path) -> Route:
        """Step 5: rod → CHAIN / ROD_VBD; sheet → PANELS / CLOTH_STYLE3D / CLOTH_VBD; volume → SOFT_VBD."""
        if morph == "rod":
            return self._rod(obj, ctx, p)
        if morph == "sheet":
            if obj.card.features.creases:
                p.reasons.append(f"panels: {len(obj.card.features.creases)} creases")
                return Route.PANELS
            sheet = self._j2_sheet(obj, ctx, p)
            if sheet is not None:
                return sheet
            aniso = obj.material.elastic.cloth is not None
            others = ctx.other_dynamic_entries
            if aniso and others == 0 and str(Route.CLOTH_STYLE3D) in ctx.registered_routes:
                p.reasons.append("cloth_style3d: anisotropic cloth, no other dynamic entry (D16)")
                return Route.CLOTH_STYLE3D
            p.reasons.append("cloth_vbd")
            return Route.CLOTH_VBD
        p.reasons.append("soft_vbd (snh)")
        return Route.SOFT_VBD

    @staticmethod
    def _j2_sheet(obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route | None:
        """A deformable sheet of a ``plastic.model: j2`` material (a metal sheet: ``scene.add_cloth(material=...)``)
        goes to the film, whose hinges carry its fibre section and keep a dent (SCR-WP7-15 P3); None otherwise. It
        runs after the creases check (a card's fold lines are its author's choice: PANELS; the film refuses creases on
        a J2 sheet) and before the anisotropic (Style3D) one.
        Without the film plug-in the sheet falls back to the elastic sheet routes and keeps no dent: a reason and a
        user warning say so (WP2's review)."""
        if not _is_j2(obj):
            return None
        if str(Route.CLOTH_FILM) not in ctx.registered_routes:
            _no_film(obj, p, "plastic.model j2 sheet, but the film plug-in is not registered: it falls back to the "
                             "elastic sheet routes and keeps no dent")
            return None
        p.reasons.append("cloth_film: a j2 sheet (hinges with the fibre section keep a dent)")
        return Route.CLOTH_FILM

    def _rod(self, obj: ResolvedObject, ctx: SceneContext, p: _Path) -> Route:
        from alloy.route.phase import rod_route

        return rod_route(self, obj, ctx, p)


def _is_j2(obj: ResolvedObject) -> bool:
    """Whether the object's material yields by J2 (``plastic.model: j2``, a metal card). Example:
    ``_is_j2(obj)`` -> ``True`` for ``add_cloth(material="steel_carbon")``, ``False`` for ``film_ldpe_sheet``."""
    plastic = obj.material.plastic
    return plastic is not None and plastic.model == "j2"


def _no_film(obj: ResolvedObject, p: _Path, why: str) -> None:
    """A j2 sheet without the film plug-in: the reason, and one user warning per card name naming the fix (WP2)."""
    p.reasons.append(why)
    if obj.card.name in _J2_FALLBACK_WARNED:
        return
    _J2_FALLBACK_WARNED.add(obj.card.name)
    from alloy.api._say import warn  # noqa: PLC0415 - the API's user warning (the package logger is silent)

    warn("%s: %s; register the film plug-in (alloy.solvers.film.plugin.register()) or pass rigid=True",
         obj.card.name, why)


_STIFF_WARNED: set[str] = set()             # j2 solids too stiff for MPM: one user warning per card name


def _warn_once(name: str, text: str) -> None:
    """A user warning for ``name`` once per process (estimate and build route the same objects)."""
    if name in _STIFF_WARNED:
        return
    _STIFF_WARNED.add(name)
    from alloy.api._say import warn  # noqa: PLC0415 - the API's user warning (the package logger is silent)

    warn("%s", text)


def particle_region_volume(src: ParticleSource) -> float:
    lo, hi = src.region
    return abs((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]))


__all__ = ["CapacityError", "ROUTE_ENTRY", "Router", "particle_region_volume", "worst"]
