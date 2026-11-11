# SPDX-License-Identifier: Apache-2.0
"""Scene attachments (`SceneCard.attachments`) as WP10 bonds on the world template (SPEC §3.8, D19).

- weld → `bonds.add_weld_bond`: a MuJoCo weld equality at the placed relative pose (MuJoCo infers relpose from
  qpos0, newton@ac7f7830:newton/_src/solvers/mujoco/equality.py:522-524). Weld and joint impedance follow
  `alloy.scene.weld_impedance` (SCR-DC3-3), not MuJoCo's 0.02 s default. A weld on a CHAIN filters the contact of
  its two bodies (SCR-DC3-14, as on the tree path of `alloy.scene.tree_weld`).
- joint → `bonds.add_connect_bond`: a ball at the anchor, in a's body frame.
- spring → `bonds.add_spring_bond`: a particle spring between the two nearest active particles.

Bond rows and native elements are written once on the template; `replicate` offsets them per world (their columns
declare `references=`, alloy/topology/bonds.py).

Endpoints are "object" or "object:link". The link is a robot body label, or a chain link index. A static object is
the world (body −1). With several bodies (a chain, or a robot without a link), the endpoint body is the one nearest
the anchor point: the anchor in a's frame, else a's origin. Weld and joint need MuJoCo-owned bodies, because
equality rows exist only in SolverMuJoCo. A spring needs particles on both ends: Newton ac7f7830 has no
particle–body spring, so a cloth–rigid spring raises AssemblyError.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import Attachment
from alloy.core.errors import AssemblyError
from alloy.route.types import Route
from alloy.scene import derive
from alloy.scene import pairs as pt
from alloy.scene import weld_impedance as WI
from alloy.scene.template import TemplateObject, pose_transform
from alloy.scene.types import RouteReport

log = logging.getLogger("alloy.scene")
EQUALITY_ENTRIES = frozenset({"mujoco"})
CG_EFFECT = "mujoco:solver=cg"


@dataclass(frozen=True)
class Endpoint:
    name: str
    body: int                     # −1 = world (static object)
    particles: tuple[int, ...]    # active particles of the object
    entry: str | None
    frame: wp.transform           # the endpoint's frame in the template (body pose, else placement pose)


def _q(x) -> float | None:
    return None if x is None else float(x.value)


def _placement_pose(t: TemplateObject) -> wp.transform:
    pose = getattr(t.placement, "pose", None) or getattr(t.placement, "base_pose", None)
    return pose_transform(pose) if pose is not None else wp.transform_identity()


def _body_pos(b: newton.ModelBuilder, body: int) -> np.ndarray:
    return np.asarray(b.body_q[body][:3], np.float64)


def _link_bodies(b: newton.ModelBuilder, t: TemplateObject, link: str) -> list[int]:
    bodies = list(t.ranges.body)
    if not link:
        return bodies
    if t.obj is None:
        hit = [i for i in bodies if b.body_label[i] == link or b.body_label[i].endswith("/" + link)]
    else:
        hit = [bodies[int(link)]] if link.isdigit() and int(link) < len(bodies) else []
    if not hit:
        raise AssemblyError(f"attachment endpoint {t.name}:{link}: no such link ({len(bodies)} bodies)")
    return hit[:1]


def _nearest_body(b: newton.ModelBuilder, bodies: Sequence[int], point: np.ndarray) -> int:
    if not bodies:
        return -1
    d = [np.linalg.norm(_body_pos(b, i) - point) for i in bodies]
    return int(bodies[int(np.argmin(d))])


def _active_particles(b: newton.ModelBuilder, t: TemplateObject) -> tuple[int, ...]:
    active = int(newton.ParticleFlags.ACTIVE)
    return tuple(i for i in t.ranges.particle if int(b.particle_flags[i]) & active)


def endpoint(b: newton.ModelBuilder, objs: dict[str, TemplateObject], ref: str, point: np.ndarray | None) -> Endpoint:
    """Resolve "object[:link]"; with several candidate bodies, take the one nearest `point` (default: origin)."""
    name, _, link = ref.partition(":")
    if name not in objs:
        raise AssemblyError(f"attachment endpoint {ref!r}: no placed object or robot named {name!r}")
    t = objs[name]
    place = _placement_pose(t)
    target = np.asarray(place.p, np.float64) if point is None else point
    body = _nearest_body(b, _link_bodies(b, t, link), target)
    frame = wp.transform(*b.body_q[body]) if body >= 0 else place
    return Endpoint(ref, body, _active_particles(b, t), t.decision.entry, frame)


def _require_equality(att: Attachment, *ends: Endpoint) -> None:
    if all(e.body < 0 for e in ends):
        raise AssemblyError(f"{att.kind} attachment {att.a}–{att.b}: both ends are static")
    for e in ends:
        if e.body >= 0 and e.entry not in EQUALITY_ENTRIES:
            raise AssemblyError(f"{att.kind} attachment {att.a}–{att.b}: {e.name} runs on {e.entry!r}; weld/joint "
                                "attachments are MuJoCo equality rows (SolverMuJoCo only)")


def _anchor_world(a: Endpoint, att: Attachment) -> np.ndarray:
    local = wp.vec3(*att.anchor[:3]) if att.anchor is not None else wp.vec3(0.0, 0.0, 0.0)
    return np.asarray(wp.transform_point(a.frame, local), np.float64)


def _connect_anchor(b: newton.ModelBuilder, a: Endpoint, other: Endpoint, att: Attachment) -> tuple[float, ...]:
    """The ball anchor in a's body frame (world frame when a is static): the card anchor, else b's origin."""
    if att.anchor is not None:
        world = _anchor_world(a, att)
    else:
        world = _body_pos(b, other.body) if other.body >= 0 else np.asarray(other.frame.p, np.float64)
    if a.body < 0:
        return tuple(float(x) for x in world)
    local = wp.transform_point(wp.transform_inverse(a.frame), wp.vec3(*world))
    return tuple(float(x) for x in local)


def _nearest_particle(b: newton.ModelBuilder, parts: Sequence[int], point: np.ndarray) -> int:
    q = np.asarray([b.particle_q[i] for i in parts], np.float64)
    return int(parts[int(np.argmin(np.linalg.norm(q - point, axis=1)))])


def _closest_particles(b: newton.ModelBuilder, a: Endpoint, c: Endpoint, att: Attachment) -> tuple[int, int]:
    """With an anchor: a's particle nearest it, then c's nearest that. Without: alternate nearest-point projections
    from c's centroid until the pair repeats (the closest pair for convex-ish sheets and blocks)."""
    pos = lambda i: np.asarray(b.particle_q[i], np.float64)  # noqa: E731
    if att.anchor is not None:
        pa = _nearest_particle(b, a.particles, _anchor_world(a, att))
        return pa, _nearest_particle(b, c.particles, pos(pa))
    centroid = np.mean([pos(i) for i in c.particles], axis=0)
    pa = _nearest_particle(b, a.particles, centroid)
    pc = _nearest_particle(b, c.particles, pos(pa))
    for _ in range(64):
        pa_new = _nearest_particle(b, a.particles, pos(pc))
        pc_new = _nearest_particle(b, c.particles, pos(pa_new))
        if (pa_new, pc_new) == (pa, pc):
            break
        pa, pc = pa_new, pc_new
    return pa, pc


def _spring(b: newton.ModelBuilder, att: Attachment, a: Endpoint, c: Endpoint, mats: tuple) -> int:
    """Particle spring a–c: K = series of the two materials' contact stiffness, damping from the pair's e."""
    for e in (a, c):
        if not e.particles:
            raise AssemblyError(f"spring attachment {att.a}–{att.b}: {e.name} has no particles; newton ac7f7830 "
                                "has no particle–body spring (use kind weld/joint between bodies)")
    pa, pc = _closest_particles(b, a, c, att)
    ka, kc = (pt.material_k(pt.MaterialEntry(f"{e.name}", m)) for e, m in zip((a, c), mats, strict=True))
    K = derive.pair_stiffness(ka, kc)
    ma, mc = float(b.particle_mass[pa]), float(b.particle_mass[pc])
    e_pair = min(float(m.surface.restitution.value) for m in mats)
    kd = derive.contact_damping(e_pair, K, ma * mc / (ma + mc))
    from alloy.topology import bonds

    return bonds.add_spring_bond(b, pa, pc, K, kd, break_force=_q(att.break_force),
                                 break_strain=_q(att.break_strain))


def _design_force(objs: dict[str, TemplateObject], *refs: str) -> float | None:
    """The larger card design force of the two endpoints (robots have none)."""
    vals = [_q(t.obj.card.loads.design_force) for t in (objs[r.partition(":")[0]] for r in refs) if t.obj is not None]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _impedance(b: newton.ModelBuilder, objs: dict[str, TemplateObject], att: Attachment, a: Endpoint, c: Endpoint,
               dt: float) -> WI.WeldImpedance:
    """SCR-DC3-3: the weld/connect impedance from the reduced mass at the design step, or the card's overrides."""
    mass = [float(b.body_mass[e.body]) if e.body >= 0 else 0.0 for e in (a, c)]
    return WI.weld_impedance(WI.reduced_mass(*mass), dt, _design_force(objs, att.a, att.b),
                             _q(att.timeconst), _q(att.dampratio))


def _set_eq_solimp(b: newton.ModelBuilder, bond: int, solimp: tuple[float, ...]) -> None:
    """bonds.add_weld_bond/add_connect_bond take a solref only; the equality row's solimp is written here."""
    from alloy.topology.bonds import NS

    eq = int(b.custom_attributes[f"{NS}:bond_eq"].values[bond])
    b.custom_attributes["mujoco:eq_solimp"].values[eq] = tuple(float(x) for x in solimp)


def add_attachment(b: newton.ModelBuilder, objs: dict[str, TemplateObject], att: Attachment,
                   dt: float | None = None, report: RouteReport | None = None, note: str = "") -> int:
    """Add one attachment's native element and bond row; returns the bond index. A weld/joint gets the SCR-DC3-3
    impedance at design step `dt` (None: MuJoCo's defaults, the pre-SCR behaviour); its line goes to `report`."""
    from alloy.topology import bonds

    a = endpoint(b, objs, att.a, None)
    c = endpoint(b, objs, att.b, _anchor_world(a, att))
    if att.kind == "spring":
        mats = tuple(objs[e.name.partition(":")[0]].obj for e in (a, c))
        if any(o is None for o in mats):
            raise AssemblyError(f"spring attachment {att.a}–{att.b}: robots have no particles")
        return _spring(b, att, a, c, tuple(o.material for o in mats))
    _require_equality(att, a, c)
    imp = _impedance(b, objs, att, a, c, dt) if dt is not None else None
    kw = {"break_force": _q(att.break_force), "break_strain": _q(att.break_strain),
          "solref": None if imp is None else imp.solref}
    if att.kind == "weld":
        _filter_chain_weld(b, objs, a, c)
        idx = bonds.add_weld_bond(b, a.body, c.body, **kw)
    else:
        idx = bonds.add_connect_bond(b, a.body, c.body, _connect_anchor(b, a, c, att), **kw)
    if imp is not None:
        _set_eq_solimp(b, idx, imp.solimp)
        _report(report, att, imp, note)
    return idx


def _weld_shapes(b: newton.ModelBuilder, objs: dict[str, TemplateObject], e: Endpoint) -> Sequence[int]:
    """The endpoint body's shapes; a static object's own shapes when the end is the world."""
    return b.body_shapes.get(e.body, ()) if e.body >= 0 else objs[e.name.partition(":")[0]].ranges.shape


def _filter_chain_weld(b: newton.ModelBuilder, objs: dict[str, TemplateObject], a: Endpoint, c: Endpoint) -> int:
    """A weld equality row on a CHAIN: the two welded bodies do not collide (SCR-DC3-14 part 3, the same rule as the
    tree path's, `alloy.scene.tree_weld.filter_welded`). Rigid-rigid welds are not covered."""
    from alloy.scene.tree_weld import filter_welded

    if not any(objs[e.name.partition(":")[0]].decision.route == Route.CHAIN for e in (a, c)):
        return 0
    return filter_welded(b, _weld_shapes(b, objs, a), _weld_shapes(b, objs, c))


def _report(report: RouteReport | None, att: Attachment, imp: WI.WeldImpedance, note: str = "") -> None:
    """The impedance line in RouteReport.entry_options; a target the solimp cap cannot meet in its warnings."""
    if report is None:
        return
    report.entry_options.append(WI.report_line(att.kind, att.a, att.b, imp, note))
    warn = WI.separation_warning(att.kind, att.a, att.b, imp)
    if warn is not None:
        log.warning("%s", warn)
        report.warnings.append(warn)


def add_attachments(b: newton.ModelBuilder, objs: Sequence[TemplateObject], atts: Sequence[Attachment],
                    dt: float | None = None, report: RouteReport | None = None,
                    tree: Mapping[int, tuple[str, str]] | None = None) -> int:
    """Every scene attachment on the template (assembler hook, after all objects are emitted); weld/joint
    impedance at design step `dt`, one route-report line each in `report`. `tree`: attachments already realised
    as tree joints (index -> (report line, chain), `alloy.scene.tree_weld`), which add no equality row."""
    by_name, tree = {t.name: t for t in objs}, dict(tree or {})
    rooted = {chain for _, chain in tree.values()}
    for i, att in enumerate(atts):
        if i in tree:
            if report is not None:
                report.entry_options.append(tree[i][0])
            continue
        loop = rooted & {att.a.partition(":")[0], att.b.partition(":")[0]}
        note = f" (closes the loop: {', '.join(sorted(loop))} root is a tree joint)" if loop else ""
        idx = add_attachment(b, by_name, att, dt, report, note)
        log.info("attachment %s %s–%s: bond %d", att.kind, att.a, att.b, idx)
    _solver_effects(by_name, atts)
    return len(atts)


def _solver_effects(objs: dict[str, TemplateObject], atts: Sequence[Attachment]) -> None:
    """A weld/joint on a CHAIN object, an equality row or a tree weld (SCR-WP2-9), asks its MuJoCo entry for the CG
    solver. Measured on T09 (plug welded to the 1 m USB cable's end link, 0.19 g): MJW's Newton solver returned NaN
    forces on the first cable-bench contact (step 32, both integrators, weld or connect, with or without impratio
    300); CG stepped 256 steps finite. Newton factorises H = M + JᵀDJ in float32; CG never does. With the tree weld
    (no equality row) T09's cable on Newton went non-finite in the first period too (CPU, 2026-11-03), so the cause
    is the light chain, not the weld row. The SCR-DC1-2 rule then decides, per scene, whether the entry keeps Newton
    (`alloy.scene.newton_rule`, applied in `SceneAssembler.entries`)."""
    for att in atts:
        if att.kind == "spring":
            continue
        for ref in (att.a, att.b):
            t = objs[ref.partition(":")[0]]
            if t.decision.route == Route.CHAIN and CG_EFFECT not in t.decision.scene_effects:
                t.decision = replace(t.decision, scene_effects=(*t.decision.scene_effects, CG_EFFECT))
                log.warning("%s: equality attachment on a chain; its MuJoCo entry uses CG unless the SCR-DC1-2 "
                            "rule keeps Newton", t.name)
