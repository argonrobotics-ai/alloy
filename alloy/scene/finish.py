# SPDX-License-Identifier: Apache-2.0
"""Post-finalize assembly steps (SPEC §3.4): pair table, soft-contact globals, precision pairs, entries, coupler,
randomization, readback and the model digest."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.coupling.types import PairPolicy
from alloy.route.types import Route
from alloy.scene import pairs as pt
from alloy.scene import template as tp
from alloy.scene.coupler_select import BodyInfo, CouplerInputs, EntryMass, select_coupler
from alloy.scene.emitters.common import object_volume
from alloy.scene.precision_pairs import PrecisionPair, precision_dt
from alloy.scene.types import BuiltScene, CouplerChoice, PairTable

if TYPE_CHECKING:
    from alloy.scene.assembler import Plan, SceneAssembler

log = logging.getLogger("alloy.scene")
RIGID_RULE_ORDER = ("mujoco", "featherstone", "semi_implicit", "kamino", "xpbd", "vbd")
DEFORMABLE_ROUTES = frozenset({Route.CLOTH_VBD, Route.CLOTH_STYLE3D, Route.CLOTH_FILM, Route.SOFT_VBD, Route.ROD_VBD})
SOFT_SPREAD_WARN = 0.10


def material_entries(plan: Plan, tmpl: tp.Template) -> list[pt.MaterialEntry]:
    from alloy.scene.assembler import material_keys

    cards = {}
    for t in plan.objects:
        if t.obj is None:
            cards[t.material_key] = (tmpl.robot_materials[t.material_key], None)
        else:
            k = t.decision.numbers.get("k_compliant") if tp.is_compliant(t.decision) else None
            cards[t.material_key] = (t.obj.material, k)
    return [pt.MaterialEntry(key, *cards[key]) for key in material_keys(plan.objects)]


def rule_entry(plan: Plan) -> str:
    entries = {t.decision.entry for t in plan.objects if t.decision.entry}
    return next((e for e in RIGID_RULE_ORDER if e in entries), "mujoco")


def shape_imp(model: newton.Model) -> np.ndarray | None:
    """Per-shape flat imp (dmax of mujoco:geom_solimp), the MuJoCo damping correction input (SCR-WP2-3)."""
    mj = getattr(model, "mujoco", None)
    solimp = getattr(mj, "geom_solimp", None) if mj is not None else None
    if solimp is None:
        return None
    return np.asarray(solimp.numpy(), np.float64).reshape(model.shape_count, -1)[:, 1]


def build_table(asm: SceneAssembler, plan: Plan, tmpl: tp.Template, model: newton.Model) -> PairTable:
    from alloy.core.registry import solver

    entries = material_entries(plan, tmpl)
    built = pt.build_tables(entries, lambda a, b: asm.pair_card(a, b, plan.scene), plan.scene.combine)
    plan.report.unsourced_pairs.extend(built.unsourced)
    entry = rule_entry(plan)
    cap = solver(entry)
    imp = shape_imp(model) if entry == "mujoco" else None
    table = pt.make_pair_table(entries, built, model, mu_mix_rule=cap.friction_mix_code, shape_imp=imp)
    if not cap.per_contact_friction:
        plan.report.unsupported.append(("pairs.mu", f"{entry} ignores per-contact friction: pair μ reaches it only "
                                                    f"through its own {cap.friction_mix} mix of shape μ"))
    if not cap.restitution_path:
        plan.report.unsupported.append(("surface.restitution", f"{entry} has no per-contact damping path"))
    return table


def set_soft_globals(plan: Plan, model: newton.Model) -> None:
    """Global particle/soft-contact scalars from the dominant (heaviest) deformable material (§3.4)."""
    defs = [t for t in plan.objects if t.obj is not None and t.decision.route in DEFORMABLE_ROUTES]
    if not defs:
        return
    masses = [float(t.obj.material.density.value) * object_volume(t.obj) for t in defs]
    dom = defs[int(np.argmax(masses))].obj.material
    mu = float(dom.surface.mu_dynamic.value)
    others = [float(t.obj.material.surface.mu_dynamic.value) for t in defs]
    if max(others) - min(others) > SOFT_SPREAD_WARN * max(abs(mu), 1e-12):
        plan.report.warnings.append(f"deformable materials differ > 10 % in μ; soft contacts use {dom.name}")
    k = pt.material_k(pt.MaterialEntry(dom.name, dom))
    model.soft_contact_mu = mu
    model.particle_mu = mu
    model.soft_contact_ke = k
    model.soft_contact_restitution = float(dom.surface.restitution.value)
    plan.report.unsupported.append(("soft_contact.per_pair", "particle-shape contacts use one global μ/K/e "
                                                             f"(dominant deformable {dom.name})"))


def precision_overrides(table: PairTable, pairs: Sequence[PrecisionPair], model: newton.Model,
                        shape_stride: int) -> None:
    """Per-shape slots + [P, P] K for precision shape pairs (all worlds share the template slot)."""
    if not pairs:
        return
    shapes = sorted({s for p in pairs for s in (p.spec.shape_a, p.spec.shape_b)})
    slot = {s: i for i, s in enumerate(shapes)}
    kmat = np.zeros((len(shapes), len(shapes)), np.float32)
    for p in pairs:
        a, b = slot[p.spec.shape_a], slot[p.spec.shape_b]
        kmat[a, b] = kmat[b, a] = p.k_declared
    per_shape = np.full(model.shape_count, -1, np.int32)
    for g in range(model.shape_count):
        per_shape[g] = slot.get(g % shape_stride if shape_stride else g, -1)
    table.override_slot = wp.array(per_shape, dtype=wp.int32, device=model.device)
    table.override_k = wp.array(kmat, dtype=wp.float32, device=model.device)


def coupler(plan: Plan, model: newton.Model, worlds: int) -> CouplerChoice:
    return select_coupler(coupler_inputs(plan, model))


def coupler_inputs(plan: Plan, model: newton.Model) -> CouplerInputs:
    """D5's inputs: per-entry masses, and a BodyInfo for every object of a rigid entry and every deformable object.
    coupler_select reads a source's bodies only when the destination is a liquid. So the deformable ones (WP4's
    SPH-VBD soft source in a liquid, engine item 3) change nothing until a soft or cloth entry is the source of a
    liquid pair."""
    entries, bodies = [], {}
    for name in sorted({t.decision.entry for t in plan.objects if t.decision.entry}):
        objs = [t for t in plan.objects if t.decision.entry == name]
        mass = sum(_object_mass(t, model) for t in objs)
        fluid = next((float(t.obj.material.density.value) for t in objs if t.obj is not None
                      and t.decision.route in (Route.LIQUID_SPH, Route.LIQUID_MPM)), None)
        entries.append(EntryMass(name, mass, frozenset(t.decision.route for t in objs),
                                 tuple(t.name for t in objs), fluid))
    for t in plan.objects:
        if t.obj is None:
            continue
        rigid = t.decision.entry in pt_rigid_entries()
        if not rigid and t.decision.route not in DEFORMABLE_ROUTES:
            continue
        mass = _object_mass(t, model)
        volume = object_volume(t.obj) if rigid else _deformable_volume(t, mass)
        bodies[t.name] = BodyInfo(t.name, mass, volume, container=t.obj.card.features.passage_width is not None,
                                  dynamic=t.obj.card.motion == "dynamic")
    return CouplerInputs(entries, bodies)


def _deformable_volume(t: tp.TemplateObject, mass: float) -> float:
    """A deformable object's volume as emitted, its particle mass over its density [m³]:
    - a SOFT_VBD solid: the card's volume (the emitter scales the tet density to the card's mass);
    - a sheet: area x thickness (areal density rho h).
    Falls back to the shape's OBB when there is no mass."""
    rho = float(t.obj.material.density.value)
    return mass / rho if mass > 0.0 and rho > 0.0 else object_volume(t.obj)


def pt_rigid_entries() -> frozenset[str]:
    from alloy.scene.coupler_select import RIGID_ENTRIES

    return RIGID_ENTRIES


def _object_mass(t: tp.TemplateObject, model: newton.Model) -> float:
    """World-0 mass of an object's bodies and particles [kg]; for a fixed-base articulated MJCF object only the
    bodies its joints move (SCR-DC4-2)."""
    from alloy.scene.articulated import is_fixed_base_articulated, moving_mass

    if is_fixed_base_articulated(t):
        return moving_mass(t.ranges.body, model.body_mass.numpy(), model.joint_parent.numpy(),
                           model.joint_child.numpy(), model.joint_type.numpy())
    m = 0.0
    if t.ranges.body:
        m += float(np.sum(model.body_mass.numpy()[t.ranges.body.start:t.ranges.body.stop]))
    if t.ranges.particle:
        m += float(np.sum(model.particle_mass.numpy()[t.ranges.particle.start:t.ranges.particle.stop]))
    return m


def finish_scene(asm: SceneAssembler, plan: Plan, tmpl: tp.Template, model: newton.Model, tcounts: dict[str, int],
                 worlds: int, world_id_offset: int, capacities, pairs: Sequence[PrecisionPair] = (),
                 world_ids: Sequence[int] | None = None) -> BuiltScene:
    from alloy.scene.assembler import digest_model
    from alloy.scene.randomize import apply_randomization, has_randomization
    from alloy.scene.readback import readback

    table = build_table(asm, plan, tmpl, model)
    precision_overrides(table, pairs, model, tcounts["shape"])
    for p in pairs:                 # several shape pairs per object pair (SCR-DC3-5): report the lowest K_real/K
        prev = plan.report.k_real.get((p.a, p.b))
        if prev is None or p.k_real / p.k_declared < prev[1] / prev[0]:
            plan.report.k_real[(p.a, p.b)] = (p.k_declared, p.k_real)
        plan.report.ladder[(p.a, p.b)] = p.ladder
    set_soft_globals(plan, model)
    cpl = coupler(plan, model, worlds)
    entries = asm.entries(plan, tcounts, worlds, capacities, precision_dt(pairs), coupled=cpl.kind != "none")
    plan.report.warnings.extend(discrete_solref_warnings(model, entries, tcounts["shape"]))
    built = BuiltScene(model=model, entries=entries, coupler=cpl, handles=asm.make_handles(plan, tcounts),
                       pair_table=table, precision_pairs=tuple(p.spec for p in pairs),
                       control_dt=plan.scene.control_dt, world_count=worlds, world_id_offset=world_id_offset,
                       world_origin=wp.zeros(worlds, dtype=wp.vec3d, device=model.device),
                       bonds=None, spares=_spares(plan), emitters=tuple(plan.scene.emitters),
                       report=plan.report, model_digest="",
                       world_ids=None if world_ids is None else wp.array(np.asarray(world_ids, np.int32),
                                                                         dtype=wp.int32, device=model.device))
    built.bonds = _bonds(built, asm.library)
    _copies(built)
    if has_randomization(plan.scene):
        apply_randomization(built, seed=0, plan=plan)
    readback(built, asm.library, plan=plan)
    built.model_digest = digest_model(model)
    return built


def discrete_solref_warnings(model: newton.Model, entries, shapes_per_world: int) -> list[str]:
    """MuJoCo entries on `discrete`: geom/pair solref time constants below 2·dt (MJW clamps τ ≥ 2·dt only for the
    continuous integrators; WP13's T09 bench at solref 2.5e-4 s, solmix 1e4 went NaN within 3 periods)."""
    out = []
    mj = getattr(model, "mujoco", None)
    for e in entries:
        if e.capability != "mujoco" or e.solver_kwargs.get("integrator") != "discrete" or mj is None:
            continue
        floor = 2.0 * e.target_dt / max(e.substeps, 1)
        for name in ("geom_solref", "pair_solref"):
            arr = getattr(mj, name, None)
            if arr is None or arr.shape[0] == 0:
                continue
            tau = arr.numpy().reshape(-1, 2)[:, 0]
            tau = tau[: shapes_per_world] if name == "geom_solref" else tau
            low = np.flatnonzero((tau > 0.0) & (tau < floor))
            if low.size:
                out.append(f"{e.name}: {low.size} {name} rows with solref[0] < 2·dt = {floor:.3g} s (min "
                           f"{tau[low].min():.3g} s) under the discrete integrator; contacts may go unstable")
    return out


def _bonds(built: BuiltScene, library):
    """WP10's BondSet from the bond rows the emitters wrote (None when the scene has no `topology:*` rows)."""
    from alloy.topology.bonds import bonds_from_cards, has_bonds

    return bonds_from_cards(built, library) if has_bonds(built.model) else None


def _copies(built: BuiltScene) -> None:
    """WP10's virtual-node copies (cards with ``topology: {cut_model: copies}``): recolour them before any solver is
    built (a copy takes its hub's colour + 1); ``alloy.integration.wiring`` builds their cutter."""
    from alloy.topology.copies_model import prepare_copies

    prepare_copies(built.model)


def _spares(plan: Plan) -> dict[str, int]:
    out = {"particles": 0, "vertices": 0, "tets": 0}
    for t in plan.objects:
        tb = t.obj.card.topology if t.obj is not None else None
        if tb is not None:
            out["particles"] += tb.spare_particles
            out["vertices"] += tb.spare_vertices
            out["tets"] += tb.spare_tets
    return out


__all__ = ["PairPolicy", "finish_scene"]
