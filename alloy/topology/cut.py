# SPDX-License-Identifier: Apache-2.0
"""Blade cutting of pre-split tet meshes: TET_CUT bonds in three phases (SPEC §3.8, D19).

Phase 0 (intact weld). The twin spring is the stiff weld written by presplit. A blade is in contact when its
cutting edge, a world segment from the body pose, passes within ``edge_radius + reach + particle radius`` of the
twins' midpoint (``nearest_blade``). Cutting starts when contact holds **and** the gate opens (``gate_open``):
tensile or compressive strain of the incident tets ≥ ``cut_strain``, **or** twin force ≥ ``cut_force``. If both are
unset, contact alone opens the gate. Gating on depth as well as force follows [internal reference]: force
alone misses drops and steep tools.
Phase 1 (cutting). The twin spring takes the blade's ``cut_spring_ke`` and the cut damage model lowers it to zero
while the blade acts on the bond. ``damage`` reports the fraction of ``cut_spring_ke`` lost (0 on entry, 1 at zero).

Cut damage model (clean-room, docs/notes/cleanroom-cut-log.md; Heiden et al., RSS 2021 arXiv:2105.12244 §III-D
Eq. 2 and IJRR 2023 arXiv:2203.10263 §3.4 Eq. 2, §5.5). The stiffness falls linearly with the knife's work on the
bond, so the rate is per unit blade travel, never per step:
- the *edge depth* of a bond is how far the edge reaches into its contact ball:
  ``depth = max(0, edge_radius + reach + particle_radius[a] − d)``, d the twins' midpoint-to-edge distance. It is
  ``reach`` when the edge touches the twins' particles and grows past it as the edge presses into them;
- the knife pressure on the bond is taken linear in the depth (Alloy's soft contacts are linear penalties);
- over a step, the *knife work* is the depth times the travel of the edge, relative to the bond, at the edge point
  nearest the bond (trapezoid rule over the step). Travel along the edge counts: slicing cuts (IJRR §5.5);
- the twin stiffness loses ``cut_spring_ke · softness · work / grip²`` and stops at exactly 0. The *grip*
  ``edge_radius + reach`` is the knife's own contact radius (the papers' edge round plus contact margin).

``softness`` is dimensionless: the fraction of ``cut_spring_ke`` lost per grip of edge travel at a depth of one grip
(the edge pressed ``edge_radius`` into the twins' particles). At the default 3.0 an edge closing in on a bond severs
it as it comes into contact (depth ≈ ``reach``). Travel is measured against the bond's nearest blade at the previous
pre-step, kept per bond (``CutDamageTrack``, part of the controller's state arrays); without that history a step
adds no work. A bond whose gate opens in a step is damaged for that step's travel too (the papers have no gate;
deferring it would lag the cut by one step of travel and make progress depend on dt).
Phase 2 (broken) is set by the controller's commit when the stiffness reaches 0 (tear.py), like any other break.

Seam alignment (gap #9, 2026-11-09; ``TopologyController.Config.cut_alignment``, default 0.5; None: no gate). A blade parts
material in its own plane, so with the gate a blade is in contact only with bonds whose current seam normal n
(``deformed_normal``: F^-T n0, the blade contact's) has ``|n . face normal| >= cut_alignment``; a bond's blade is the
nearest such blade in reach (``nearest_cutting_blade``). A seam the edge merely crosses stays welded. A phase 1
bond that turns out of alignment keeps its stiffness and damage and takes no more. A bond without a seam normal
(zero ``bond_normal``) is not gated; the controller logs how many. Without the gate, a blade in the x seam of a
block softened and cut the y seam along the line where its edge crossed it, and a blade pressed face-on onto flat
cut rounds split them into layers (JOBRK finding 1: 115-117 of 134 broken bonds under |n . face| 0.5).
``BladeContact`` puts only aligned twins on the blade faces (``min_alignment``), so use its value
(``SEAM_ALIGNMENT``): every bond the cutter damages is then one the contact opens.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.blade_contact_kernels import SEAM_ALIGNMENT, deformed_normal  # noqa: F401 - the gate's value
from alloy.topology.geometry import point_segment_distance
from alloy.topology.tables import item_worlds
from alloy.topology.tear import K_TET, tet_strain_range
from alloy.topology.types import Blade

log = logging.getLogger("alloy.topology")

PHASE_WELD, PHASE_CUTTING = 0, 1
DEFAULT_REACH = 5.0e-4        # [m] RSS Table A2 / IJRR Table A2 ``sdf_radius`` (contact radius around the knife)
NO_LIMIT = wp.constant(1.0e30)  # [m] reach for "nearest blade in the world, at any distance"


def default_reach() -> float:
    """[m] Default extra blade contact reach beyond ``edge_radius`` (``TopologyController.Config.blade_reach=None``).

    0.5 mm, the radius around the knife surface that the papers' contact model considers (``sdf_radius``, RSS
    Table A2 p.16, IJRR Table A2 p.27)."""
    return DEFAULT_REACH


def resolve_reach(reach: float | None) -> float:
    """[m] ``Config.blade_reach`` with ``None`` resolved to ``default_reach()``; ``AssemblyError`` if not >= 0."""
    value = default_reach() if reach is None else float(reach)
    if not np.isfinite(value) or value < 0.0:
        raise AssemblyError(f"TopologyController.Config.blade_reach must be a finite length >= 0 m, got {reach!r}")
    log.debug("cut damage: blade reach %.3g m (%s)", value, "default" if reach is None else "Config.blade_reach")
    return value


def resolve_alignment(alignment: float | None) -> float:
    """``Config.cut_alignment`` as the kernel's threshold: 0 (no gate) for None; ``AssemblyError`` unless in (0, 1]."""
    if alignment is None:
        return 0.0
    value = float(alignment)
    if not (np.isfinite(value) and 0.0 < value <= 1.0):
        raise AssemblyError(f"TopologyController.Config.cut_alignment must be in (0, 1] or None, got {alignment!r}")
    return value


@dataclass
class BladeTables:
    count: int
    body: wp.array               # [K] int32
    edge_a: wp.array             # [K] vec3 body frame
    edge_b: wp.array
    radius: wp.array             # [K] float32
    softness: wp.array
    cut_ke: wp.array
    world_start: wp.array        # [W+1]
    world_order: wp.array        # [K]
    p0: wp.array                 # [K] world segment, refreshed each pre_step
    p1: wp.array
    axis: wp.array               # [K] vec3 unit face normal, body frame (``blade_contact.blade_axis``)
    face: wp.array               # [K] vec3 the same in the world, refreshed each pre_step


def build_blade_tables(model: newton.Model, blades: Sequence[Blade], world_count: int) -> BladeTables:
    from alloy.topology.blade_contact import blade_axis

    dev = model.device
    n = len(blades)
    bodies = np.array([bl.body for bl in blades], dtype=np.int64)
    if n and (bodies.min() < 0 or bodies.max() >= model.body_count):
        raise AssemblyError(f"Blade body index out of range (bodies {bodies.tolist()}, body_count {model.body_count})")
    bw = model.body_world.numpy()[bodies] if n else np.zeros(0, np.int64)
    world = item_worlds(bw, world_count, "Blade")
    order = np.argsort(world, kind="stable").astype(np.int32)
    wstart = np.concatenate([[0], np.cumsum(np.bincount(world, minlength=world_count)[:world_count])]).astype(np.int32)
    v3 = lambda f: wp.array(np.array([f(bl) for bl in blades], np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)  # noqa: E731
    f32 = lambda f: wp.array(np.array([f(bl) for bl in blades], np.float32), dtype=wp.float32, device=dev)  # noqa: E731
    for bl in blades:
        _check_blade(bl)
    return BladeTables(
        count=n, body=wp.array(bodies.astype(np.int32), dtype=wp.int32, device=dev), edge_a=v3(lambda bl: bl.edge_a),
        edge_b=v3(lambda bl: bl.edge_b), radius=f32(lambda bl: bl.edge_radius), softness=f32(lambda bl: bl.softness),
        cut_ke=f32(lambda bl: bl.cut_spring_ke), world_start=wp.array(wstart, dtype=wp.int32, device=dev),
        world_order=wp.array(order, dtype=wp.int32, device=dev), p0=wp.zeros(n, dtype=wp.vec3, device=dev),
        p1=wp.zeros(n, dtype=wp.vec3, device=dev), axis=v3(lambda bl: blade_axis(model, bl)),
        face=wp.zeros(n, dtype=wp.vec3, device=dev))


def _check_blade(bl: Blade) -> None:
    if bl.edge_radius <= 0.0:
        raise AssemblyError(f"Blade on body {bl.body}: edge_radius must be > 0 (a real edge round, GOTCHAS:538)")
    if not (np.isfinite(bl.softness) and bl.softness >= 0.0):
        raise AssemblyError(f"Blade on body {bl.body}: cut damage softness must be finite and >= 0, got {bl.softness}")
    if not (np.isfinite(bl.cut_spring_ke) and bl.cut_spring_ke > 0.0):
        raise AssemblyError(f"Blade on body {bl.body}: cut_spring_ke must be finite and > 0 N/m, got "
                            f"{bl.cut_spring_ke}")


@wp.kernel
def blade_segments(body: wp.array[wp.int32], edge_a: wp.array[wp.vec3], edge_b: wp.array[wp.vec3],
                   axis: wp.array[wp.vec3], body_q: wp.array[wp.transform], p0: wp.array[wp.vec3],
                   p1: wp.array[wp.vec3], face: wp.array[wp.vec3]):
    k = wp.tid()
    xf = body_q[body[k]]
    p0[k] = wp.transform_point(xf, edge_a[k])
    p1[k] = wp.transform_point(xf, edge_b[k])
    face[k] = wp.transform_vector(xf, axis[k])


@wp.func
def nearest_blade(mid: wp.vec3, reach: float, start: int, end: int, order: wp.array[wp.int32],
                  p0: wp.array[wp.vec3], p1: wp.array[wp.vec3], radius: wp.array[wp.float32]) -> int:
    best = int(-1)
    best_d = float(1.0e30)
    for k in range(start, end):
        bl = order[k]
        d = point_segment_distance(mid, p0[bl], p1[bl])
        if d <= radius[bl] + reach and d < best_d:
            best = bl
            best_d = d
    return best


@wp.struct
class CutBlades:
    world_start: wp.array[wp.int32]
    order: wp.array[wp.int32]
    p0: wp.array[wp.vec3]
    p1: wp.array[wp.vec3]
    radius: wp.array[wp.float32]
    softness: wp.array[wp.float32]
    cut_ke: wp.array[wp.float32]
    face: wp.array[wp.vec3]


@wp.func
def nearest_cutting_blade(mid: wp.vec3, reach: float, start: int, end: int, blades: CutBlades, n: wp.vec3,
                          alignment: float) -> int:
    """``nearest_blade`` among the blades that may cut a seam with current normal ``n``: those with
    ``|n . face| >= alignment``. Every blade with no gate (``alignment`` 0) or no normal (``n`` zero). A nearer
    blade that only crosses the seam does not hide an aligned one behind it."""
    gated = alignment > 0.0 and wp.length_sq(n) > 0.0
    best = int(-1)
    best_d = float(1.0e30)
    for k in range(start, end):
        bl = blades.order[k]
        d = point_segment_distance(mid, blades.p0[bl], blades.p1[bl])
        if d > blades.radius[bl] + reach or d >= best_d:
            continue
        if gated and wp.abs(wp.dot(n, blades.face[bl])) < alignment:
            continue
        best = bl
        best_d = d
    return best


@wp.func
def gate_open(strain: wp.vec2, force: float, cut_strain: float, cut_force: float) -> bool:
    if wp.isinf(cut_strain) and wp.isinf(cut_force):
        return True
    return wp.max(strain[0], strain[1]) >= cut_strain or force >= cut_force


@dataclass
class CutDamageTrack:
    """Per-bond cut damage history between pre-steps (allocated once; exported by the controller's state arrays).

    ``blade[i]`` is the bond's contact blade at the last pre-step, else its nearest blade in its world at any
    distance (−1 = none yet). ``edge0`` / ``edge1`` are that blade's edge end points relative to the twins' midpoint
    then. The next pre-step measures the edge's travel relative to the bond against them."""

    blade: wp.array              # [B] int32
    edge0: wp.array              # [B] vec3, m
    edge1: wp.array              # [B] vec3, m

    def arrays(self) -> dict[str, wp.array]:
        return {"topology.cut_damage.blade": self.blade, "topology.cut_damage.edge0": self.edge0,
                "topology.cut_damage.edge1": self.edge1}


def build_cut_damage_track(count: int, device) -> CutDamageTrack:
    n = max(int(count), 1)
    return CutDamageTrack(blade=wp.full(n, -1, dtype=wp.int32, device=device),
                          edge0=wp.zeros(n, dtype=wp.vec3, device=device),
                          edge1=wp.zeros(n, dtype=wp.vec3, device=device))


@wp.func
def edge_depth(e0: wp.vec3, e1: wp.vec3, ball: float) -> float:
    """[m] How far the edge segment (relative to the twins' midpoint at the origin) reaches into the contact ball."""
    return wp.max(ball - point_segment_distance(wp.vec3(0.0, 0.0, 0.0), e0, e1), 0.0)


@wp.func
def knife_work(old0: wp.vec3, old1: wp.vec3, new0: wp.vec3, new1: wp.vec3, ball: float) -> float:
    """[m²] Edge depth integrated over the edge's travel relative to the bond during one step (trapezoid rule).

    The travel is that of the edge point now nearest the bond, so pressing and slicing along the edge both count
    (IJRR §5.5: the knife's work, force times distance, is what opens the cut)."""
    e = new1 - new0
    u = wp.clamp(wp.dot(-new0, e) / wp.max(wp.dot(e, e), 1.0e-30), 0.0, 1.0)
    travel = wp.length((new0 + u * e) - (old0 + u * (old1 - old0)))
    return 0.5 * (edge_depth(old0, old1, ball) + edge_depth(new0, new1, ball)) * travel


@wp.func
def cut_damage_stiffness(ke: float, cut_ke: float, softness: float, work: float, grip: float) -> float:
    """[N/m] Twin stiffness after a step of knife ``work``: linear loss (RSS/IJRR Eq. 2), floored at exactly 0."""
    return wp.max(ke - cut_ke * softness * work / (grip * grip), 0.0)


@wp.func
def weld_gate(i: int, xa: wp.vec3, xb: wp.vec3, ke: float, elem_start: wp.array[wp.int32],
              elem: wp.array[wp.int32], tet_indices: wp.array2d[wp.int32], tet_poses: wp.array[wp.mat33],
              x: wp.array[wp.vec3], cut_strain: wp.array[wp.float32], cut_force: wp.array[wp.float32]) -> bool:
    """Phase 0 → 1 test of a bond in blade contact (``gate_open`` on the incident tets' strain and the twin force)."""
    strain = wp.vec2(0.0, 0.0)
    if not wp.isinf(cut_strain[i]):
        strain = tet_strain_range(elem_start[i], elem_start[i + 1], elem, tet_indices, tet_poses, x)
    return gate_open(strain, ke * wp.length(xa - xb), cut_strain[i], cut_force[i])


@wp.kernel
def cut_pre_step(
    kind: wp.array[wp.int8], a: wp.array[wp.int32], b: wp.array[wp.int32], world: wp.array[wp.int32],
    intact: wp.array[wp.int8], spring: wp.array[wp.int32], elem_start: wp.array[wp.int32], elem: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32], tet_poses: wp.array[wp.mat33], x: wp.array[wp.vec3],
    particle_radius: wp.array[wp.float32], cut_strain: wp.array[wp.float32], cut_force: wp.array[wp.float32],
    blades: CutBlades, reach: float, phase: wp.array[wp.int8], spring_stiffness: wp.array[wp.float32],
    damage: wp.array[wp.float32], contact: wp.array[wp.int32], track_blade: wp.array[wp.int32],
    track_edge0: wp.array[wp.vec3], track_edge1: wp.array[wp.vec3], normal: wp.array[wp.vec3],
    side_tet: wp.array[wp.int32], alignment: float,
):
    """One thread per bond: blade contact (with ``alignment`` > 0, of seams aligned with the blade face only), phase
    0 gating, then cut damage of phase 1 bonds for this step's knife work. Writes ``contact``, ``phase``,
    ``damage``, the twin spring's ``spring_stiffness`` and the track."""
    i = wp.tid()
    contact[i] = -1
    if wp.int32(kind[i]) != K_TET or intact[i] == wp.int8(0):
        return
    xa = x[a[i]]
    xb = x[b[i]]
    mid = 0.5 * (xa + xb)
    lo = blades.world_start[world[i]]
    hi = blades.world_start[world[i] + 1]
    rp = particle_radius[a[i]]
    n = normal[i]
    if alignment > 0.0 and wp.length_sq(n) > 0.0:
        n = deformed_normal(n, side_tet[i], tet_indices, tet_poses, x)
    k = nearest_cutting_blade(mid, reach + rp, lo, hi, blades, n, alignment)   # -1: no blade, or all only cross it
    tracked = k
    if tracked < 0:
        tracked = nearest_blade(mid, NO_LIMIT, lo, hi, blades.order, blades.p0, blades.p1, blades.radius)
    old_blade = track_blade[i]
    old0 = track_edge0[i]
    old1 = track_edge1[i]
    track_blade[i] = tracked
    if tracked >= 0:
        track_edge0[i] = blades.p0[tracked] - mid
        track_edge1[i] = blades.p1[tracked] - mid
    contact[i] = k
    if k < 0:
        return
    s = spring[i]
    if phase[i] == wp.int8(PHASE_WELD):
        if not weld_gate(i, xa, xb, spring_stiffness[s], elem_start, elem, tet_indices, tet_poses, x, cut_strain,
                         cut_force):
            return
        phase[i] = wp.int8(PHASE_CUTTING)
        spring_stiffness[s] = blades.cut_ke[k]
        damage[i] = 0.0
    if phase[i] != wp.int8(PHASE_CUTTING) or old_blade != k:
        return
    grip = blades.radius[k] + reach
    work = knife_work(old0, old1, blades.p0[k] - mid, blades.p1[k] - mid, grip + rp)
    ke = cut_damage_stiffness(spring_stiffness[s], blades.cut_ke[k], blades.softness[k], work, grip)
    spring_stiffness[s] = ke
    damage[i] = wp.clamp(1.0 - ke / blades.cut_ke[k], 0.0, 1.0)


def cut_blades_struct(t: BladeTables) -> CutBlades:
    s = CutBlades()
    s.world_start, s.order, s.p0, s.p1 = t.world_start, t.world_order, t.p0, t.p1
    s.radius, s.softness, s.cut_ke, s.face = t.radius, t.softness, t.cut_ke, t.face
    return s
