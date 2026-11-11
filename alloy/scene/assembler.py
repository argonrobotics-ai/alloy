# SPDX-License-Identifier: Apache-2.0
"""SceneAssembler (SPEC §3.4): cards → routed world template → replicate → finalize → pair table, precision pairs,
entries, coupler, randomization, readback, model digest. Deterministic for equal cards (CARD-6)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import PairCard, ResolvedObject, SceneCard
from alloy.core.errors import AssemblyError
from alloy.core.protocols import CardLibraryLike, RouterLike
from alloy.core.registry import SolverCapability, registered_routes, solver
from alloy.parallel.types import PackSpec
from alloy.route.router import Router
from alloy.route.types import Route, SceneContext
from alloy.scene.inertia import restore_authored_inertia
from alloy.scene import fast_profile as FP, fixed_pairs as FIX, integrator_route as IR, ortho_membrane as OMB
from alloy.scene import mjcf_options as MO, template as tp, vbd_contact as VC
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.scene.emitters.rigid import emit_part
from alloy.scene.entries import chain_on_newton, entry_overrides, hold_warning, make_entries, sliding_warning
from alloy.scene.finish import finish_scene, material_entries
from alloy.scene.rigid_step import RigidStepRequest, choose_rigid_step, today_step
from alloy.scene.types import BuiltScene, ObjectHandle, RouteDecision, RouteReport
from alloy.solvers.vbd import cloth_profile as CP

log = logging.getLogger("alloy.scene")


@dataclass
class Plan:
    """Host-side result of resolve + route: everything the template build needs."""

    scene: SceneCard
    objects: list[tp.TemplateObject] = field(default_factory=list)
    resolved: dict[str, ResolvedObject] = field(default_factory=dict)
    caps: list[SolverCapability] = field(default_factory=list)
    report: RouteReport = field(default_factory=RouteReport.empty)
    integrator: IR.IntegratorChoice | None = None    # the MuJoCo entry's rk4 decision (`_template`, SCR-WP6-4)
    fast: FP.FastChoice | None = None                # the opt-in fast route profile's decision (`_template`)
    device: str | None = None                        # the build's device: F6 R1 reads its MuJoCo route (IR.r1_off)


class SceneAssembler:
    def __init__(self, library: CardLibraryLike, router: RouterLike | None = None, *,
                 route_profile: str = "default", newton_candidate: bool = False, cloth_profile: str = "default",
                 rigid_step: RigidStepRequest | None = None):
        """`route_profile`: "default" (the route) or "fast", the opt-in profile for precision scenes (`alloy.scene.
        fast_profile`, SCR-WP3-8; speed reports print its cable creep beside the speed). `newton_candidate`: build a
        chain attachment's entry on Newton without the SCR-DC1-2 rule (its probe and evidence runs, `alloy.scene.
        newton_rule`). `cloth_profile`: "default" or the opt-in "accurate" (`alloy.solvers.vbd.cloth_profile`)."""
        if route_profile not in FP.ROUTE_PROFILES:
            raise AssemblyError(f"unknown route profile {route_profile!r}; expected one of {FP.ROUTE_PROFILES}")
        register_builtin_solvers()
        self.library = library
        self.router = router or Router()
        self.route_profile, self.cloth_profile = route_profile, CP.checked_profile(cloth_profile)
        self.newton_candidate, self.rigid_step = newton_candidate, rigid_step     # alloy.scene.rigid_step (the API)

    def assemble(self, scene: SceneCard, device: str, *, world_count: int | None = None, world_id_offset: int = 0,
                 capacities: PackSpec | None = None, world_ids: Sequence[int] | None = None) -> BuiltScene:
        """`world_ids` (SCR-WP8-2): the true global id of each local world for repacked packs (RNG keys, D9)."""
        if world_ids is not None:
            world_count = len(world_ids) if world_count is None else world_count
            if len(world_ids) != world_count:
                raise AssemblyError(f"world_ids has {len(world_ids)} entries for {world_count} worlds")
        worlds = int(world_count if world_count is not None else scene.worlds)
        if worlds < 1:
            raise AssemblyError(f"world_count must be >= 1, got {worlds}")
        plan = self.plan(scene, device)
        tmpl = self._template(plan)
        plan._template, plan._assembler = tmpl, self
        pairs, tcounts = self._precision(plan, tmpl, device), tp.counts(tmpl.builder)
        main = tp.make_builder(plan.caps)
        main.replicate(tmpl.builder, copy_count=worlds, spacing=(0.0, 0.0, 0.0))
        if main.particle_count or any(t.decision.route == Route.ROD_VBD for t in tmpl.objects):
            OMB.colour(main)   # VBD needs colours (finalize leaves particle_colors unset); woven triangles count too
        model = main.finalize(device=device)
        restore_authored_inertia(main, model)
        log.info("assembled %d objects x %d worlds on %s: %d bodies, %d shapes, %d particles", len(plan.objects),
                 worlds, device, model.body_count, model.shape_count, model.particle_count)
        return finish_scene(self, plan, tmpl, model, tcounts, worlds, world_id_offset, capacities, pairs, world_ids)

    # --- resolve + route ----------------------------------------------------------------------------------------

    def plan(self, scene: SceneCard, device: str | None = None) -> Plan:
        plan = Plan(scene, device=device)
        names = self._resolve(scene, plan)
        ctx = self._context(plan, names)
        for name, placement in zip(names, scene.objects):
            obj = plan.resolved[name]
            d = self.router.route(obj, ctx)
            plan.objects.append(tp.TemplateObject(name, obj, d, placement, self._material_key(obj, name, d, placement)))
            plan.report.decisions[name] = d
            plan.report.default_leaves[name] = tuple(obj.default_leaves)
            plan.report.scene_effects.extend(e for e in d.scene_effects if e not in plan.report.scene_effects)
            if d.near_boundary:
                plan.report.warnings.append(f"{name}: near a routing boundary (margin {d.margin:.3f})")
        for robot in scene.robots:
            d = RouteDecision(route=_rigid(), entry="mujoco", numbers={}, margin=float("inf"), near_boundary=False,
                              reasons=("robots are always RIGID on mujoco (§3.3 step 6)",))
            plan.objects.append(tp.TemplateObject(robot.name, None, d, robot, f"robot:{robot.name}"))
            plan.report.decisions[robot.name] = d
        plan.caps = [solver(n) for n in sorted({t.decision.entry for t in plan.objects if t.decision.entry})]
        CP.apply_to_plan(plan, self.cloth_profile)
        return plan

    def _resolve(self, scene: SceneCard, plan: Plan) -> list[str]:
        names, seen = [], {}
        for placement in scene.objects:
            card = self.library.object(placement.object)
            k = seen.get(card.name, 0)
            seen[card.name] = k + 1
            name = card.name if k == 0 else f"{card.name}#{k}"
            plan.resolved[name] = self._with_asset_paths(self.library.resolve(card, placement))
            names.append(name)
        return names

    def asset(self, rel: str) -> str:
        """A card-relative asset path resolved against the library root (WP1 `asset_path`), else as given."""
        fn = getattr(self.library, "asset_path", None)
        found = fn(rel) if fn is not None else None
        return str(found) if found is not None else rel

    def _with_asset_paths(self, obj: ResolvedObject) -> ResolvedObject:
        """Absolute asset paths on the shape source, each resolved by ``asset``: ``path`` (mesh, MJCF, USD, SDF, tet
        .npz) and a TetSource's ``from_mesh`` (WP4: it stayed relative to the cwd). card_hash unchanged."""
        shape = obj.card.shape
        paths = {f: self.asset(getattr(shape, f)) for f in ("path", "from_mesh") if getattr(shape, f, None)}
        return replace(obj, card=replace(obj.card, shape=replace(shape, **paths))) if paths else obj

    def _context(self, plan: Plan, names: Sequence[str]) -> SceneContext:
        objs = [plan.resolved[n] for n in names]
        touching = {n: tuple(m for m in names if m != n) for n in names}
        return SceneContext(rigid_partners=touching, free_bodies_per_world=tp.free_body_count(objs),
                            registered_routes=registered_routes(), partners=dict(plan.resolved),
                            needs_l1=any(o.card.precision_features for o in objs),
                            other_dynamic_entries=self._other_dynamic(objs))

    def _other_dynamic(self, objs: Sequence[ResolvedObject]) -> int:
        """Dynamic entries besides a sheet's own (D16 Style3D rule): MuJoCo if any other dynamic object exists."""
        dyn = [o for o in objs if o.card.motion == "dynamic"]
        return 1 if len(dyn) > 1 else 0

    @staticmethod
    def _material_key(obj: ResolvedObject, name: str, d: RouteDecision, placement) -> str:
        """Own pair-table row for compliant objects and objects whose surface values are randomized."""
        own = tp.is_compliant(d) or any(r.field.startswith("material.surface.") for r in placement.randomize)
        return f"{obj.material.name}@{name}" if own else obj.material.name

    # --- template ------------------------------------------------------------------------------------------------

    def _template(self, plan: Plan) -> tp.Template:
        topo = needs_topology(plan)
        b = tp.make_builder(plan.caps, topology=topo)
        make_sub = lambda: tp.make_builder(plan.caps, topology=topo)  # noqa: E731
        robot_mats = {}
        tree = self._emit_objects(b, plan, make_sub, robot_mats)
        if plan.scene.attachments:
            from alloy.scene.attachments import add_attachments
            add_attachments(b, plan.objects, plan.scene.attachments, today_step(plan.scene.control_dt), plan.report,
                            tree)
            for e in (e for t in plan.objects for e in t.decision.scene_effects):
                if e not in plan.report.scene_effects:
                    plan.report.scene_effects.append(e)
        keys = material_keys(plan.objects)
        for i, t in enumerate(plan.objects):
            tp.tag(b, t.ranges, i, keys.index(t.material_key))
        from alloy.scene.mujoco_pairs import add_native_pairs
        from alloy.scene.native_gains import apply_native_gains, floor_template, rigid_dt_max

        _filter_precision_shapes(b, plan.objects)       # before the native pair rows (mujoco_pairs._excluded)
        FIX.filter_fixed_pairs(b)                       # kinematic vs static / kinematic: no contact, as above
        plan.integrator = self._integrator(plan, b, rigid_dt_max(plan.scene.control_dt))   # gains' cap depends on it
        plan.fast = self._fast(plan, b)

        def gains(b: newton.ModelBuilder, integ: str, request, report) -> tp.Template:   # integ: the gains mode
            dt_max = choose_rigid_step(b, plan.objects, plan.scene, integ, request, report).dt
            apply_native_gains(b, plan.objects, dt_max, integ, report)
            floor_template(b, plan.objects, dt_max, plan.scene, report)    # after the rk4 choice (inertia floor)
            tmpl = tp.Template(b, plan.objects, robot_mats)
            if any(t.decision.entry == "mujoco" for t in plan.objects):
                add_native_pairs(b, plan.objects, material_entries(plan, tmpl),
                                 lambda a, c: self.pair_card(a, c, plan.scene), plan.scene.combine, dt_max, integ)
            return tmpl
        plan.integrator = IR.own_gains(plan.integrator, b, plan, self.rigid_step, gains)   # step 2c (own gains)
        return gains(b, plan.integrator.gains, self.rigid_step, plan.report)   # rk4's quarter cap under rk4 and R1

    def _emit_objects(self, b: newton.ModelBuilder, plan: Plan, make_sub, robot_mats: dict) -> dict:
        """Emit robots and objects; chain-end welds become tree joints (SCR-WP2-9, `alloy.scene.tree_weld`), with
        each such chain emitted right after its parent. Returns {attachment index: (report line, chain)}."""
        from alloy.scene import tree_weld as TW

        atts = plan.scene.attachments
        cand = TW.candidates(plan.objects, atts) if atts else {}
        plan.objects[:] = TW.emission_order(plan.objects, cand)
        objs, done, roots = {t.name: t for t in plan.objects}, {}, {}      # roots: kinematic fixtures' bodies
        for t in plan.objects:
            if t.obj is None:
                t.ranges = tp.emit_robot(b, t.placement, make_sub, self.asset)
                robot_mats[t.material_key] = tp.robot_material(t.name, b, t.ranges.shape)
                continue
            cap = solver(t.decision.entry) if t.decision.entry else None
            tw = cand.get(t.name)
            if tw is not None and not self._tree_root(b, objs, tw, cap, make_sub, plan.report):
                tw = None
            t.ranges = emit_part(b, t, cap, make_sub, roots)        # tp.emit_object; a fixture part: its root's body
            if tw is not None:
                body = TW.parent_body(b, objs, tw)
                TW.attach(b, t, body, TW.parent_shapes(objs, tw))
                done[atts.index(tw.att)] = (TW.report_line(tw, body), t.name)
            spares = t.obj.card.topology.spare_particles if t.obj.card.topology is not None else 0
            if spares:
                extra = tp.add_spares(b, t.obj, spares)
                t.ranges = tp.IndexRanges(**{**t.ranges.__dict__, "particle": range(t.ranges.particle.start,
                                                                                     extra.stop)})
        return done

    @staticmethod
    def _tree_root(b, objs, tw, cap, make_sub, report: RouteReport) -> bool:
        """Mark the chain's root fixed when the weld lands on link 0 and its parent's articulation is the last."""
        from alloy.scene import tree_weld as TW

        link = TW.root_link(b, objs, tw, cap, make_sub)
        body = TW.parent_body(b, objs, tw)
        why = (f"the weld lands on link {link}, not the root" if link != 0 else
               None if TW.adjacent(b, body) else f"{tw.parent_ref} is not in the last articulation")
        if why is not None:
            report.entry_options.append(f"mujoco weld {tw.att.a}–{tw.att.b}: equality row, no tree joint: {why}")
            return False
        TW.mark_root(objs[tw.chain])
        return True

    def _integrator(self, plan: Plan, b: newton.ModelBuilder, dt: float | None = None) -> IR.IntegratorChoice:
        """SCR-WP6-4 rule (and F6 R1, its floor test at today's step ``dt``) on the template, logged and routed."""
        from alloy.scene import precision_pairs as pp
        from alloy.scene.noslip_route import scene_noslip

        precision = bool(pp.routed_features(plan.report, plan.objects))    # each pair named in Sim.routes()
        noslip = scene_noslip(plan.scene.robots, self.asset, plan.objects) is not None
        choice = IR.choose_integrator(b, plan.objects, precision=precision, noslip=noslip, dt=dt, device=plan.device,
                                      newton_contacts=getattr(self.rigid_step, "newton_contacts", False))
        return IR.name_in_routes(plan.report, plan.objects, choice)

    def _fast(self, plan: Plan, b: newton.ModelBuilder) -> FP.FastChoice:
        """The fast route profile's decision; under it, chain dofs get Newton's armature in the template."""
        if self.route_profile != "fast":
            return FP.OFF
        integ = IR.effect_integrator(e for t in plan.objects if t.decision.entry == "mujoco"
                                     for e in t.decision.scene_effects)
        choice = FP.choose_fast(plan.objects, _precision_shape_pairs(b, plan.objects), integ)
        log.info("%s", choice.reason)
        if choice.newton:
            n = FP.raise_chain_armature(b, plan.objects)
            log.info("fast profile: chain armature >= %g kg·m² (%d dofs raised)", FP.CHAIN_ARMATURE, n)
        return choice

    def _precision(self, plan: Plan, tmpl: tp.Template, device: str) -> list:
        """PrecisionPairs from the cards (template ids, template masses); flat solimp imp on their shapes, the pair
        filtered from native/bulk collision (the contact layer collides it) and, on CUDA, the feature SDFs."""
        from alloy.scene import precision_pairs as pp

        b = tmpl.builder
        objs = {t.name: t for t in plan.objects if t.obj is not None}
        spairs = _shape_pairs(b, plan.objects)
        if not spairs:
            return []
        integ = IR.effect_integrator(e for t in plan.objects for e in t.decision.scene_effects)
        if integ == "rk4":      # the rk4 rule (entries) excludes precision scenes; a per-object rk4 effect may not
            raise AssemblyError("precision pairs cannot run under rk4 (SPEC §3.5 integrator table: no gap ladder)")
        rigid_dt = plan.report.rigid_step.dt      # today's step: precision pairs keep it (alloy.scene.rigid_step)
        out = []
        for sp in spairs:           # SCR-DC3-5: one pair per touched shape pair (alloy.scene.precision_shapes)
            ta, tc = objs[sp.a], objs[sp.b]
            inp = pp.PairInputs(sp.shape_a, sp.shape_b, _shape_mass(b, ta, sp.shape_a), _shape_mass(b, tc, sp.shape_b),
                                max(_q(ta.obj.card.loads.design_force), _q(tc.obj.card.loads.design_force)),
                                min(_q(ta.obj.card.loads.contact_speed), _q(tc.obj.card.loads.contact_speed)))
            out.append(pp.build_pair(sp.a, sp.b, sp.feature_a, sp.feature_b, inp, plan.scene.control_dt, rigid_dt,
                                     integ))
        _flat_solimp(b, out)            # the pair's shapes were filtered from native collision in _template
        _feature_sdfs(b, out, device)
        return out

    def make_handles(self, plan: Plan, tcounts: dict[str, int]) -> dict[str, ObjectHandle]:
        stride = tp.stride_ranges(tcounts)
        return {t.name: ObjectHandle(t.name, t.obj.card_hash if t.obj else "", t.decision, t.ranges, stride,
                                     blade=_blade(t)) for t in plan.objects}

    def entries(self, plan: Plan, tcounts: dict[str, int], worlds: int, capacities, precision_dt, coupled: bool):
        from alloy.scene.noslip_route import held_effects, report_line, scene_noslip

        req = scene_noslip(plan.scene.robots, self.asset, plan.objects)   # every MJCF the scene loads (SCR-SC6-3)
        fast = plan.fast or FP.OFF
        noslip = FP.fast_noslip(req, fast.pairs) if fast.on else (None if req is None else (req.iterations,
                                                                                            req.tolerance))
        if noslip is not None:
            plan.report.scene_effects.append(f"mujoco:noslip={noslip[0]}")
        if req is not None:
            plan.report.entry_options.append(report_line(req))
        if fast.on:
            plan.report.warnings.append(FP.FAST_CAVEAT)
        choice = plan.integrator or self._integrator(plan, plan._template.builder)
        scene_fx = choice.effects() + fast.effects() + held_effects(req) + plan.report.rigid_step.effects()
        plan.report.scene_effects.extend(scene_fx)
        mjcf = MO.scene_options(MO.option_files(plan.scene.robots, plan.objects, self.asset))
        make = lambda objs: make_entries(objs, tcounts, worlds, plan.scene.control_dt,  # noqa: E731
                                         capacities=capacities, precision_dt=precision_dt,
                                         has_precision=precision_dt is not None, coupled=coupled, noslip=noslip,
                                         scene_effects=scene_fx, mjcf=mjcf, builder=plan._template.builder,
                                         rigid_dt=plan.report.rigid_step.dt)
        self._newton_rule(plan, make, fast)
        specs = IR.with_gains(make(plan.objects), choice)      # R1's gains flag for the light-body rule
        VC.warn_small_bodies(plan.objects)                     # a body that fits between a sheet's vertices (GAPS-11)
        mj = next((e for e in specs if e.name == "mujoco"), None)
        if mj is not None:
            effects = {e for t in plan.objects for e in t.decision.scene_effects} | set(scene_fx)
            group = [t for t in plan.objects if t.decision.entry == "mujoco"]
            chain_nw = chain_on_newton(group, effects, mjcf)
            plan.report.entry_options.extend(MO.report_lines(mjcf, mj.solver_kwargs,
                                                             entry_overrides(group, effects, precision_dt is not None,
                                                                             mjcf, chain_nw)))
            warn = sliding_warning(mjcf, mj.solver_kwargs)
            if warn is None and chain_nw and "impratio" not in mj.solver_kwargs:
                warn = hold_warning(mj.solver_kwargs)
            plan.report.warnings.extend([] if warn is None else [warn])
        return specs

    def _newton_rule(self, plan: Plan, make, fast: FP.FastChoice) -> None:
        """SCR-DC1-2: a chain attachment keeps MuJoCo Warp's Newton solver when the rule's probe shows the float32
        Hessian positive definite, or with evidence (`alloy.scene.newton_rule`, `alloy.scene.newton_evidence`)."""
        from alloy.scene import newton_rule as NR

        cand = NR.candidates(plan.objects, fast.on and fast.newton)
        if cand is None:
            return
        mj = next(e for e in make(cand) if e.name == "mujoco")
        key = NR.probe_key(plan._template.builder, mj.solver_kwargs, mj.target_dt, plan.scene.control_dt)
        if self.newton_candidate:
            NR.apply_newton(plan.objects, plan.report)
            plan.report.entry_options.append(f"mujoco solver: Newton candidate of the SCR-DC1-2 probe; probe key {key}")
            return
        verdict = NR.decide(self, plan.scene, key)
        if verdict.newton:
            NR.apply_newton(plan.objects, plan.report)
        plan.report.entry_options.append(verdict.line())
        log.info("%s", verdict.line())

    def pair_card(self, a: str, b: str, scene: SceneCard) -> PairCard | None:
        key = tuple(sorted((a, b)))
        for p in scene.pairs:
            if tuple(sorted((p.a, p.b))) == key:
                return p
        return self.library.pair(a, b)


def _blade(t: tp.TemplateObject):
    """The card's BladeSpec for the handle (SCR-SC5-1); its edge is in a body frame, so a blade needs one body."""
    blade = t.obj.card.blade if t.obj is not None else None
    if blade is not None and len(t.ranges.body) != 1:
        raise AssemblyError(f"{t.name}: blade card on {len(t.ranges.body)} bodies; the edge is given in one body's "
                            "frame, so a blade object needs exactly one body")
    return blade


def _q(x) -> float:
    return float(getattr(x, "value", x))


def _mass(b: newton.ModelBuilder, t: tp.TemplateObject) -> float:
    """Mass of an object's dynamic bodies in the template [kg]: 0 for static/kinematic, the bodies its joints move
    for a fixed-base articulated MJCF object (SCR-DC4-2)."""
    from alloy.scene.articulated import is_fixed_base_articulated, moving_mass

    if is_fixed_base_articulated(t):
        return moving_mass(t.ranges.body, b.body_mass, b.joint_parent, b.joint_child, b.joint_type)
    if t.obj is not None and t.obj.card.motion != "dynamic":
        return 0.0
    return float(sum(b.body_mass[i] for i in t.ranges.body))


def _filter_precision_shapes(b: newton.ModelBuilder, objs: Sequence[tp.TemplateObject]) -> None:
    """Filter each precision pair's shapes (every touched shape pair, as `_precision` builds them) from
    native/bulk collision before any explicit MuJoCo pair row is written: an explicit pair bypasses
    contype/conaffinity, and T09's plug/socket row (μ 0.575 ≠ max 0.5) made MuJoCo collide their hulls, 4 contacts
    at 8.65 mm pushing a seated plug out at 1.14 m/s in one step (WP3, L4). The contact layer collides the pair."""
    for sa, sc in _precision_shape_pairs(b, objs):
        b.add_shape_collision_filter_pair(sa, sc)


def _shape_pairs(b: newton.ModelBuilder, objs: Sequence[tp.TemplateObject]) -> list:
    """The precision shape pairs of the cards' partner relations (SCR-DC3-5, alloy.scene.precision_shapes)."""
    from alloy.scene import precision_pairs as pp
    from alloy.scene import precision_shapes as ps

    named = {t.name: t for t in objs if t.obj is not None}
    rels = pp.pair_features({n: t.obj for n, t in named.items()})
    return ps.shape_pairs(b, rels, {n: t.ranges.shape for n, t in named.items()}) if rels else []


def _precision_shape_pairs(b: newton.ModelBuilder, objs: Sequence[tp.TemplateObject]) -> tuple[tuple[int, int], ...]:
    """Template shape ids of each precision pair, as `_precision` builds them."""
    return tuple((p.shape_a, p.shape_b) for p in _shape_pairs(b, objs))


def _shape_mass(b: newton.ModelBuilder, t: tp.TemplateObject, s: int) -> float:
    """Mass behind a precision shape: the object's for one body, else the shape's body if it moves (SCR-DC3-5)."""
    from alloy.scene.articulated import is_fixed_base_articulated, moving_bodies
    from alloy.scene.precision_shapes import shape_mass

    moving = (moving_bodies(t.ranges.body, b.joint_parent, b.joint_child, b.joint_type)
              if is_fixed_base_articulated(t) else None)
    return shape_mass(b, t.ranges.body, s, _mass(b, t), moving)


def _flat_solimp(b: newton.ModelBuilder, pairs) -> None:
    """Precision shapes get MuJoCo solimp (imp, imp, 0.001, 0.5, 2) with imp = spec.mj_imp (D22)."""
    key = "mujoco:geom_solimp"
    if key not in b.custom_attributes:
        return
    values = b.custom_attributes[key].values
    for p in pairs:
        imp = p.spec.mj_imp
        for s in (p.spec.shape_a, p.spec.shape_b):
            values[s] = (imp, imp, 0.001, 0.5, 2.0)


def _feature_sdfs(b: newton.ModelBuilder, pairs, device: str) -> None:
    """Bake WP3's feature SDFs on the precision shapes before finalize (texture SDFs need CUDA)."""
    if not pairs:
        return
    if not wp.get_device(device).is_cuda:
        log.info("precision feature SDFs not baked on %s (texture SDFs need CUDA)", device)
        return
    try:
        from alloy.contact.sdf_cache import attach_pair_sdfs  # WP3
    except ImportError:
        log.warning("alloy.contact.sdf_cache not available: precision feature SDFs not baked")
        return
    for p in pairs:
        attach_pair_sdfs(b, p.spec, device=device)


def needs_topology(plan: Plan) -> bool:
    """WP10 attributes are registered only when the scene has attachments or an object with a TopologyBudget."""
    return bool(plan.scene.attachments) or any(
        t.obj is not None and t.obj.card.topology is not None for t in plan.objects)


def material_keys(objs: Sequence[tp.TemplateObject]) -> list[str]:
    """Pair-table material order: sorted unique keys (deterministic ids)."""
    return sorted({t.material_key for t in objs})


def _rigid():
    from alloy.route.types import Route

    return Route.RIGID


_POINTER_ARRAYS = frozenset({"shape_source_ptr"})     # per-build mesh ids: the mesh content is digested instead


def digest_model(model: newton.Model) -> str:
    """sha256 over every numeric Model array (name, dtype, shape, bytes), sorted by attribute name, and the shapes'
    mesh content (CARD-6). Pointer arrays are left out; `bvh_shape_enabled` and Newton 1.8's `_bvh_shape_active_leaves`
    (`wp.empty`, model.py:2279) are cut to their valid prefix, so two builds of a mesh scene no longer differ there."""
    import hashlib

    h = hashlib.sha256()
    arrays = []
    for name in sorted(vars(model)):
        val = getattr(model, name)
        if isinstance(val, wp.array) and name not in _POINTER_ARRAYS:
            arrays.append((name, val))
        elif hasattr(val, "__dict__") and name in ("alloy", "mujoco", "mpm", "style3d", "sph"):
            arrays.extend((f"{name}.{k}", v) for k, v in sorted(vars(val).items()) if isinstance(v, wp.array))
    for name, arr in arrays:
        data = np.ascontiguousarray(arr.numpy())
        if name in ("bvh_shape_enabled", "_bvh_shape_active_leaves"):       # unwritten tails (docstring)
            data = data[: int(model.bvh_shape_count_enabled)]
        h.update(f"{name}|{data.dtype.str}|{data.shape}|".encode())
        h.update(data.tobytes())
    _digest_meshes(h, getattr(model, "shape_source", None) or ())
    return h.hexdigest()


def _digest_meshes(h, sources) -> None:
    """Vertices and indices of every mesh shape source, in shape order (None and primitives: their slot only)."""
    for i, src in enumerate(sources):
        h.update(f"source|{i}|{type(src).__name__}|".encode())
        for field in ("vertices", "indices"):
            data = getattr(src, field, None)
            if data is not None:
                h.update(np.ascontiguousarray(np.asarray(data)).tobytes())
