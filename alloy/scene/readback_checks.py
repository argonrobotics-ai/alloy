# SPDX-License-Identifier: Apache-2.0
"""Per-route readback checkers: expected card-derived values vs compiled Model arrays, [W, ...] per world.

Each checker returns (checks, unsupported) where `unsupported` maps a card path to the reason it cannot reach the
Model for that route. Expected values are recomputed from the cards here (not read back from the builder state).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from alloy.cards.types import MeshSource, Primitive
from alloy.route.types import Route
from alloy.scene import derive
from alloy.scene.emitters.common import primitive_volume
from alloy.scene.types import BuiltScene
from alloy.solvers.vbd import cloth_profile

if TYPE_CHECKING:
    from alloy.scene.assembler import Plan
    from alloy.scene.template import TemplateObject


@dataclass(frozen=True)
class Check:
    path: str
    expected: np.ndarray
    actual: np.ndarray
    atol: float = 0.0


Result = tuple[list[Check], dict[str, str]]


def routing_inputs() -> set[str]:
    return set()


def per_world(built: BuiltScene, t: TemplateObject, field: str, nominal: float) -> np.ndarray:
    """[W] value of a card field: the randomized draw where present, else the nominal value."""
    vals = getattr(built, "randomized", {}).get((t.name, field))
    return np.full(built.world_count, nominal, np.float64) if vals is None else np.asarray(vals, np.float64)


def gids(t: TemplateObject, built: BuiltScene, kind: str) -> np.ndarray:
    """[W, n] global ids of the object's entities of `kind`."""
    rng = getattr(t.ranges, kind)
    stride = len(getattr(built.handles[t.name].world_stride, kind))
    return np.arange(built.world_count)[:, None] * stride + np.asarray(list(rng), np.int64)[None, :]


def drawn_surface_values(built: BuiltScene, t: TemplateObject, leaf: str) -> np.ndarray:
    """[W] surface value ``leaf`` of the object's card with each world's draws combined (``randomize.drawn_surface``:
    a μ_static draw also sets μ_d)."""
    from alloy.scene.randomize import drawn_surface, surface_draws

    mat = t.obj.material
    draws = [surface_draws(built, t.name, w) for w in range(built.world_count)]
    return np.asarray([float(getattr(drawn_surface(mat, d).surface, leaf).value) if d else
                       float(getattr(mat.surface, leaf).value) for d in draws], np.float64)


def _surface_checks(built: BuiltScene, t: TemplateObject) -> list[Check]:
    m, s = built.model, t.obj.material.surface
    sh = gids(t, built, "shape")
    if sh.size == 0:
        return []
    mu_d = drawn_surface_values(built, t, "mu_dynamic")          # MuJoCo geom friction follows each world's draw
    out = [Check("material.surface.mu_dynamic", np.broadcast_to(mu_d[:, None], sh.shape),
                 m.shape_material_mu.numpy()[sh]),
           Check("material.surface.restitution", np.full(sh.shape, s.restitution.value),
                 m.shape_material_restitution.numpy()[sh])]
    if s.rolling_lever is not None:
        out.append(Check("material.surface.rolling_lever", np.full(sh.shape, s.rolling_lever.value),
                         m.shape_material_mu_rolling.numpy()[sh]))
    if s.torsional_lever is not None:
        out.append(Check("material.surface.torsional_lever", np.full(sh.shape, s.torsional_lever.value),
                         m.shape_material_mu_torsional.numpy()[sh]))
    return out + _pair_row_checks(built, t)


def _pair_row_checks(built: BuiltScene, t: TemplateObject) -> list[Check]:
    """The object's own pair-table entry (with itself) vs its cards: the same-material PairCard where sourced,
    else the material's μ_s, μ_d, e; K = series of two equal springs."""
    table = built.pair_table
    i = table.material_index(t.material_key)
    s = t.obj.material.surface
    plan = built._plan
    card = plan._assembler.pair_card(t.obj.material.name, t.obj.material.name, plan.scene)
    if card is not None:
        return _sourced_self_checks(built, t, table, i, card)
    wt = getattr(table, "world_table", None)
    rows = wt.numpy()[:, i, i] if wt is not None else np.broadcast_to(table.table.numpy()[i, i],
                                                                         (built.world_count, 4))
    mu_s, mu_d, e = (drawn_surface_values(built, t, leaf) for leaf in ("mu_static", "mu_dynamic", "restitution"))
    out = [Check("material.surface.mu_static", mu_s, rows[:, 0]), Check("pair.self.mu_dynamic", mu_d, rows[:, 1]),
           Check("pair.self.restitution", e, rows[:, 2])]
    if s.contact_stiffness is not None and not t.decision.numbers.get("k_compliant"):
        k = per_world(built, t, "material.surface.contact_stiffness", s.contact_stiffness.value)
        ws = getattr(table, "world_stiffness", None)
        got = ws.numpy()[:, i, i] if ws is not None else np.full(built.world_count, table.stiffness.numpy()[i, i])
        out.append(Check("material.surface.contact_stiffness", 0.5 * k, got))
    return out


def _sourced_self_checks(built: BuiltScene, t: TemplateObject, table, i: int, card) -> list[Check]:
    from alloy.scene import pairs as pt

    entry = pt.MaterialEntry(t.material_key, t.obj.material)
    exp = pt.pair_values(entry, entry, card, built._plan.scene.combine)
    wt = getattr(table, "world_table", None)
    rows = wt.numpy()[:, i, i] if wt is not None else np.broadcast_to(table.table.numpy()[i, i],
                                                                         (built.world_count, 4))
    w = built.world_count
    return [Check("pairs.self.mu_static", np.full(w, exp.mu_s), rows[:, 0]),
            Check("pairs.self.mu_dynamic", np.full(w, exp.mu_d), rows[:, 1]),
            Check("pairs.self.restitution", np.full(w, exp.e), rows[:, 2])]


def _body_mass_check(built: BuiltScene, t: TemplateObject, volume: float) -> list[Check]:
    bodies = gids(t, built, "body")
    if bodies.size == 0 or volume <= 0.0:
        return []
    rho = per_world(built, t, "material.density", t.obj.material.density.value)
    got = built.model.body_mass.numpy()[bodies].sum(axis=1)
    return [Check("material.density", rho * volume + _riders_mass(built, t), got)]


def _riders_mass(built: BuiltScene, t: TemplateObject) -> np.ndarray | float:
    """[W] mass a kinematic fixture's other parts add to its root part's body (``rigid.emit_part``: one body carries
    every part, with each part's own density); 0 for any other object."""
    from alloy.scene.emitters.rigid import FIXTURE_GROUP, FIXTURE_ROOT

    numbers = t.decision.numbers
    if not numbers.get(FIXTURE_ROOT):
        return 0.0
    riders = [o for o in built._plan.objects if o is not t and o.obj is not None
              and o.decision.numbers.get(FIXTURE_GROUP) == numbers.get(FIXTURE_GROUP)]
    return sum((per_world(built, o, "material.density", o.obj.material.density.value)
                * primitive_volume(o.obj.card.shape) for o in riders), 0.0)


def decompose_mass_source(shape: MeshSource) -> tuple[float, str]:
    """(card-mesh volume, where the inertia comes from) of a ``collision="decompose"`` mesh (SCR-OSS-10): the emitter
    gives the body density x the card mesh's volume in both branches, with the inertia from the card mesh when it is
    watertight, else from the hulls (their density scaled to that mass)."""
    from alloy.cards.geometry import MeshData, is_watertight
    from alloy.cards.mass import mesh_mass
    from alloy.scene.emitters.common import load_mesh

    mesh = load_mesh(shape)
    data = MeshData(mesh.vertices, mesh.indices)
    return mesh_mass(data, 1.0), "card mesh" if is_watertight(data) else "hulls"


def check_rigid(built: BuiltScene, plan: Plan, t: TemplateObject) -> Result:
    unsupported = {}
    shape = t.obj.card.shape
    volume = primitive_volume(shape) if isinstance(shape, Primitive) else 0.0
    if isinstance(shape, MeshSource) and shape.collision == "decompose" and t.decision.route != Route.STATIC:
        volume, source = decompose_mass_source(shape)
        unsupported["inertia"] = (f"collision='decompose': inertia from the {source}"
                                  + ("" if source == "card mesh" else " (card mesh not watertight)") + ", not checked")
    checks = _surface_checks(built, t) + _body_mass_check(built, t, volume)
    if t.decision.route == Route.STATIC:
        unsupported["material.density"] = "static shapes carry no mass"
    elif volume == 0.0:
        unsupported["material.density"] = "non-primitive shape: mass from the mesh/import, not checked in v1"
    if t.obj.material.elastic is not None:
        for leaf in ("youngs", "poisson", "rayleigh_beta"):
            unsupported[f"material.elastic.{leaf}"] = f"{t.decision.route}: elastic values only decide the route"
    if t.decision.numbers.get("k_compliant"):
        unsupported["material.surface.contact_stiffness"] = "replaced by K = F_g/δ_loc (RIGID_COMPLIANT)"
    return checks, unsupported


def check_cloth(built: BuiltScene, plan: Plan, t: TemplateObject) -> Result:
    from alloy.scene.emitters.deformable import sheet_thickness

    el, m = t.obj.material.elastic, built.model
    if el.cloth is not None and t.decision.route == Route.CLOTH_VBD:
        return check_ortho(built, t)
    E = per_world(built, t, "material.elastic.youngs", el.youngs.value)
    nu = float(el.poisson.value)
    beta = float(el.rayleigh_beta.value) if el.rayleigh_beta is not None else 0.0
    h = sheet_thickness(t.obj)
    lame = np.asarray([derive.cloth_stiffness(e, nu, h)[:2] for e in E])       # (mu, lambda) per world
    tri_ke, tri_ka = lame[:, 0], lame[:, 1]
    edge_ke = np.asarray([derive.sheet_bending(t.obj.material, e, nu, h) for e in E])   # a drawn E scales it
    bend_leaf = "material.elastic.poisson" if el.bending is None else "material.elastic.bending"
    tris, edges, parts = gids(t, built, "tri"), gids(t, built, "edge"), gids(t, built, "particle")
    tm, eb = m.tri_materials.numpy(), m.edge_bending_properties.numpy()
    edge_exp, edge_atol = cloth_profile.expected_edge_ke(m, edges, edge_ke, t.decision.route == Route.CLOTH_VBD)
    checks = [Check("material.elastic.youngs", tri_ke[:, None] * np.ones(tris.shape[1]), tm[tris, 0]),
              Check("material.elastic.poisson", tri_ka[:, None] * np.ones(tris.shape[1]), tm[tris, 1]),
              Check(bend_leaf, edge_exp, eb[edges, 0], edge_atol),
              Check("material.elastic.rayleigh_beta", beta * tri_ke[:, None] * np.ones(tris.shape[1]), tm[tris, 2],
                    atol=1e-9)]
    area = _cloth_area(m, tris[0])
    rho = per_world(built, t, "material.density", t.obj.material.density.value)
    checks.append(Check("material.density", rho * h * area, m.particle_mass.numpy()[parts].sum(axis=1)))
    checks.append(Check("card.features.thickness", np.full(parts.shape, 0.5 * h), m.particle_radius.numpy()[parts]))
    return checks + _soft_global_checks(built, t), {}


def check_ortho(built: BuiltScene, t: TemplateObject) -> Result:
    """A woven card on CLOTH_VBD (``alloy.scene.ortho_membrane``): the alloy:tri_ortho_k rows are the ClothAniso
    stretch / shear values and beta, Newton's membrane is off, and each hinge sits between the plate map at B_weft and
    at B_warp (its scale depends on the edge's direction in the panel); mass and radius as for VBD cloth."""
    from alloy.scene import ortho_membrane as OM
    from alloy.scene.emitters.deformable import sheet_thickness

    el, m = t.obj.material.elastic, built.model
    an, h = el.cloth, sheet_thickness(t.obj)
    beta = float(el.rayleigh_beta.value) if el.rayleigh_beta is not None else 0.0
    tris, edges, parts = gids(t, built, "tri"), gids(t, built, "edge"), gids(t, built, "particle")
    k, tm = getattr(m.alloy, OM.K_ATTR).numpy(), m.tri_materials.numpy()
    checks = [Check(f"material.elastic.cloth.{n}", np.full(tris.shape, float(getattr(an, n).value)), k[tris, i])
              for i, n in enumerate(("stretch_warp", "stretch_weft", "shear"))]
    checks += [Check("material.elastic.rayleigh_beta", np.full(tris.shape, beta), k[tris, 3], atol=1e-9),
               Check("material.elastic.youngs", np.zeros(tris.shape), tm[tris, 0]),
               Check("material.elastic.poisson", np.zeros(tris.shape), tm[tris, 1])]
    d_card = float(derive.sheet_bending(t.obj.material, float(el.youngs.value), float(el.poisson.value), h))
    plate, _ = cloth_profile.expected_edge_ke(m, edges, np.full(edges.shape[0], d_card), True)
    ratio = np.divide(m.edge_bending_properties.numpy()[edges, 0], plate, out=np.zeros(edges.shape), where=plate > 0)
    lo, hi = sorted((float(an.bend_warp.value) / d_card, float(an.bend_weft.value) / d_card))
    on, f32 = plate > 0, 1e-5                                    # edge_ke is stored in float32
    checks.append(Check("material.elastic.cloth.bend_warp", np.where(on, np.clip(ratio, lo * (1 - f32), hi * (1 + f32)),
                                                                     0.0), ratio, atol=1e-6 * hi))
    area = _cloth_area(m, tris[0])
    rho = per_world(built, t, "material.density", t.obj.material.density.value)
    checks.append(Check("material.density", rho * h * area, m.particle_mass.numpy()[parts].sum(axis=1)))
    checks.append(Check("card.features.thickness", np.full(parts.shape, 0.5 * h), m.particle_radius.numpy()[parts]))
    return checks + _soft_global_checks(built, t), {}


def check_style3d(built: BuiltScene, plan: Plan, t: TemplateObject) -> Result:
    """CLOTH_STYLE3D reads the ClothAniso values (style3d:tri_aniso_ke = weft, warp, shear; style3d:aniso_ke per
    edge = bend weft, warp, twist), not E/ν; mass, radius and damping as for VBD cloth."""
    from alloy.scene.emitters.deformable import sheet_thickness

    el, m = t.obj.material.elastic, built.model
    an = el.cloth
    if an is None:
        return check_cloth(built, plan, t)
    tris, edges, parts = gids(t, built, "tri"), gids(t, built, "edge"), gids(t, built, "particle")
    tri_k, edge_k = m.style3d.tri_aniso_ke.numpy(), m.style3d.aniso_ke.numpy()
    h = sheet_thickness(t.obj)
    checks = [Check(f"material.elastic.cloth.{k}", np.full(tris.shape, float(getattr(an, k).value)), tri_k[tris, i])
              for i, k in enumerate(("stretch_weft", "stretch_warp", "shear"))]
    checks += [Check(f"material.elastic.cloth.{k}", np.full(edges.shape, float(getattr(an, k).value)), edge_k[edges, i])
               for i, k in enumerate(("bend_weft", "bend_warp"))]
    area = _cloth_area(m, tris[0])
    rho = per_world(built, t, "material.density", t.obj.material.density.value)
    checks.append(Check("material.density", rho * h * area, m.particle_mass.numpy()[parts].sum(axis=1)))
    checks.append(Check("card.features.thickness", np.full(parts.shape, 0.5 * h), m.particle_radius.numpy()[parts]))
    why = "SolverStyle3D uses the ClothAniso stretch/bending values, not the isotropic modulus"
    return checks + _soft_global_checks(built, t), {"material.elastic.youngs": why, "material.elastic.poisson": why}


def _cloth_area(m, tri_ids: np.ndarray) -> float:
    idx = m.tri_indices.numpy()[tri_ids]
    x = m.particle_q.numpy().astype(np.float64)
    a, b, c = x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _soft_global_checks(built: BuiltScene, t: TemplateObject) -> list[Check]:
    s = t.obj.material.surface
    if abs(built.model.soft_contact_mu - s.mu_dynamic.value) > 1e-6 * max(1.0, s.mu_dynamic.value):
        return []      # not the dominant deformable: listed unsupported by set_soft_globals
    return [Check("material.surface.mu_dynamic", np.asarray([s.mu_dynamic.value]),
                  np.asarray([built.model.soft_contact_mu]))]


def check_soft(built: BuiltScene, plan: Plan, t: TemplateObject) -> Result:
    el, m = t.obj.material.elastic, built.model
    E = per_world(built, t, "material.elastic.youngs", el.youngs.value)
    nu = float(el.poisson.value)
    beta = float(el.rayleigh_beta.value) if el.rayleigh_beta is not None else 0.0
    lame = np.asarray([derive.lame(e, nu) for e in E])
    from alloy.topology.copies_model import whole_tets  # noqa: PLC0415 - WP10: copies' fragments carry no material

    tets = whole_tets(m, gids(t, built, "tet"))
    tm = m.tet_materials.numpy()[tets]
    ones = np.ones(tets.shape[1])
    checks = [Check("material.elastic.youngs", lame[:, 0:1] * ones, tm[..., 0]),
              Check("material.elastic.poisson", lame[:, 1:2] * ones, tm[..., 1]),
              Check("material.elastic.rayleigh_beta", beta * lame[:, 0:1] * ones, tm[..., 2], atol=1e-9)]
    # Particles include WP10's dormant copies: the density and radius rows rely on their zero mass and hub radius.
    shape, parts = t.obj.card.shape, gids(t, built, "particle")
    if isinstance(shape, Primitive):
        rho = per_world(built, t, "material.density", t.obj.material.density.value)
        checks.append(Check("material.density", rho * primitive_volume(shape),
                            m.particle_mass.numpy()[parts].sum(axis=1)))
    radius_path = "card.budget.tets" if isinstance(shape, Primitive) else "card.shape"
    checks.append(Check(radius_path, np.full(parts.shape, _soft_radius(t)), m.particle_radius.numpy()[parts]))
    return checks + _soft_global_checks(built, t), {}


def _soft_radius(t: TemplateObject) -> float:
    """The SOFT_VBD particle radius the rule gives (SCR-XB4-5-1), from the mesh the card's shape and tet budget make.
    A surface mesh's tets come from the build's fTetWild cache (SCR OSS-9-5), so this is the emitted mesh; from the
    Model's float32 world positions it missed rtol 1e-6 by rounding (sponge_kitchen). A readback outside the building
    process, or after the cache dir is cleared, re-runs fTetWild, which is not repeatable: it can fail a correct build.
    The pre-split path duplicates seam vertices in place, so its tets are the plain mesh's."""
    from alloy.scene.emitters.deformable import soft_mesh, soft_particle_radius

    verts, tets, _ = soft_mesh(t.obj)
    return soft_particle_radius(verts, tets)


def check_chain(built: BuiltScene, plan: Plan, t: TemplateObject) -> Result:
    n, m = t.decision.numbers, built.model
    links = int(n["links"])
    l = n["rod_length"] / (links - 1)
    E_ratio = per_world(built, t, "material.elastic.youngs", t.obj.material.elastic.youngs.value)
    E_ratio = E_ratio / float(t.obj.material.elastic.youngs.value)
    bodies = gids(t, built, "body")
    rho = per_world(built, t, "material.density", t.obj.material.density.value)
    lin = rho * math.pi * n["rod_radius"] ** 2 * (n["rod_density"] / float(t.obj.material.density.value))
    checks = [Check("material.density", lin * n["rod_length"], m.body_mass.numpy()[bodies].sum(axis=1))]
    if t.decision.entry == "mujoco":
        qd = m.joint_qd_start.numpy()
        stiff = m.joint_stiffness.numpy()                # Newton 1.8: the core passive springs (the hinges')
        joints = gids(t, built, "joint")[:, 1:]          # joint 0 is the free root
        got = np.asarray([[stiff[d] for j in row for d in range(qd[j], qd[j + 1])] for row in joints])
        per_joint = np.asarray([n["EI"] / l, n["EI"] / l, n["GJ"] / l])
        exp = E_ratio[:, None] * np.tile(per_joint, joints.shape[1])[None, :]
        checks.append(Check("material.elastic.youngs", exp, got))
        checks.append(Check("material.elastic.poisson", exp, got))
    return checks + _surface_checks(built, t), {"material.elastic.rayleigh_beta": "sets the hinge damping time"}


MPM_PATHS = {"young_modulus": "material.elastic.youngs", "poisson_ratio": "material.elastic.poisson",
             "friction": "material.plastic.friction_angle", "yield_pressure": "material.plastic.yield_pressure",
             "yield_stress": "material.plastic.yield_stress", "hardening": "material.plastic.hardening",
             "dilatancy": "material.plastic.dilatancy", "viscosity": "material.fluid.viscosity"}


def check_mpm(built: BuiltScene, plan: Plan, t: TemplateObject) -> Result:
    from alloy.scene.emitters.deformable import mpm_density, mpm_values

    m, parts = built.model, gids(t, built, "particle")
    checks, unsupported = [], {}
    ns = getattr(m, "mpm", None)
    for name, val in mpm_values(t.obj.material).items():
        arr = getattr(ns, name, None) if ns is not None else None
        if arr is None:
            unsupported[MPM_PATHS.get(name, f"mpm:{name}")] = f"mpm:{name} not registered on the Model"
            continue
        checks.append(Check(MPM_PATHS.get(name, f"mpm:{name}"), np.full(parts.shape, val), arr.numpy()[parts]))
    spacing = 0.5 * t.decision.numbers["voxel"]
    bulk = mpm_density(t.obj, t.decision) / float(t.obj.material.density.value)     # grain_mpm: GRAIN_PACKING
    rho = bulk * per_world(built, t, "material.density", t.obj.material.density.value)
    checks.append(Check("material.density", rho[:, None] * spacing**3 * np.ones(parts.shape[1]),
                        m.particle_mass.numpy()[parts]))
    unsupported["material.surface.mu_dynamic"] = "MPM collider friction comes from the collider shapes"
    return checks, unsupported


CHECKERS: dict[Route, Callable[[BuiltScene, "Plan", "TemplateObject"], Result]] = {
    Route.RIGID: check_rigid, Route.STATIC: check_rigid, Route.RIGID_COMPLIANT: check_rigid,
    Route.RIGID_PILE: check_rigid, Route.PANELS: check_rigid, Route.GRAIN_RIGID: check_rigid,
    Route.CHAIN: check_chain, Route.CLOTH_VBD: check_cloth, Route.CLOTH_STYLE3D: check_style3d,
    Route.SOFT_VBD: check_soft, Route.GRAIN_MPM: check_mpm, Route.PASTE_MPM: check_mpm,
    Route.LIQUID_MPM: check_mpm, Route.CLOTH_FILM: lambda built, plan, t: _film_check()(built, plan, t),
}


def _film_check():
    """SC8: the film plug-in's readback (scaled membrane, per-hinge bending, film:* attributes), imported lazily."""
    from alloy.solvers.film.readback import check_film

    return check_film
