# SPDX-License-Identifier: Apache-2.0
"""GapMonitor: exact precision-pair penetration and tunnels against geometry independent of the narrow phase (L3).

Each world is measured against its own pair geometry: worlds whose pair shapes differ (type, scale, mesh, margin; CT-2's
plate per world) get their own samples and reference, a geometry slot, and identical worlds share one (SCR-XB4-10-1;
the work is one (sample, world) task each, so it stays linear in the world count).

Reference geometry per pair (never the texture SDF the narrow phase uses, contact §4.4): analytic SDFs for
primitives (``analytic_sdf``), else the partner's original mesh as a ``wp.Mesh`` queried with Warp's angle-weighted
pseudo-normal sign, an "inside" confirmed by the winding number (Warp's pseudo-normal sign flips on the mid-plane
between two facing walls, e.g. inside the USB-A plug's tongue cavity; ``winding=True`` uses the winding number for
every sample). The current pose of a sample on a closed mesh gets its sign from a per-sample anchor: the winding
number settles the sign at the anchor, and the sample keeps it while it stays within the anchor's distance of the
anchor (the partner is rigid, so no surface lies in that ball: exact, whatever the path or a reset). Anchored signs
are therefore the exact inside/outside, which the pseudo-normal alone misreads both ways on the USB-A meshes, and a
seated pair pays for closest-point queries only (T09: nearly every seated sample took the winding confirm). Surface
samples of the smaller side (``samples.py``) are posed with the state, measured in the partner's frame, and

- enter the p99 population if their exact distance ≤ 0 or they lie within ``sample_spacing`` of an active row of the
  same pair and world (force > 1e-3 N when ``rigid_contact_force`` is filled, else separation ≤ 0);
- go to a per-pair 1-µm histogram (integer atomics) and per-pair/world maxima (``atomic_max``): deterministic;
- count a tunnel when the segment prev→cur, expressed in the partner frame, passes through the partner interior with
  the sample outside at both ends, or goes from outside to deeper than ``tunnel_depth``. The segment is marched at
  half the thinnest wall (analytic: the partner's thinnest half-extent; meshes: ``wall_thickness/2`` or 0.1 mm) with
  exact signed distances; ray-crossing counts are not used (rays grazing a wall plane miscount). A march point is
  inside only deeper than ``cross_floor`` (the pair's own contact depth, at most a quarter of the march step): the
  straight segment of a sample sliding along a wall it touches dips into the wall by its contact depth, and counting
  that as a tunnel made worlds of demo_peg_d12 INVALID on an L4 (PAIR_PEN_FAIL from a 0.1-1.2 um dip of a peg falling
  down its bore at 0.35 m/s, 1 to 3 worlds in 16; audit PHYSA-4, 2026-11-10). The previous-pose
  distance is evaluated only when a tunnel is possible (1-Lipschitz: a segment of length ``m`` that ends at distance
  ``d1 ≥ m`` stays outside; a shallow ``-tunnel_depth ≤ d1 < 0`` cannot count).

``update`` is device-only (graph-safe) and writes ``StepHealth.max_pair_pen_m`` with ``max_pair_id`` (the index
of that pair in the layer's pairs; -1 when none penetrates), ``min_pair_gap_m`` and the PAIR_PEN_WARN /
PAIR_PEN_FAIL / CONTACT_OVERFLOW flags (D3, D11); severity stays with HealthMonitor (WP5).
"""

from __future__ import annotations

import hashlib

import numpy as np
import warp as wp

from alloy.contact.analytic_sdf import GEO_BOX, GEO_CAPSULE, GEO_CYLINDER, GEO_SPHERE, has_exact_sdf, sd_primitive
from alloy.contact.samples import surface_area_estimate, surface_samples
from alloy.contact.types import GapReport
from alloy.contact.validate import is_watertight
from alloy.core.log import get_logger
from alloy.precision.types import HealthFlags, StepHealth

log = get_logger("contact.monitor")

BINS = GapReport.HISTOGRAM_BINS              # 1-µm bins to 2 mm
MAX_ROWS_PER_PAIR_WORLD = 64
FORCE_MIN_N = 1.0e-3
F_WARN = wp.constant(wp.uint32(int(HealthFlags.PAIR_PEN_WARN)))
F_FAIL = wp.constant(wp.uint32(int(HealthFlags.PAIR_PEN_FAIL)))
F_OVER = wp.constant(wp.uint32(int(HealthFlags.CONTACT_OVERFLOW)))
REF_ANALYTIC, REF_WINDING, REF_NORMAL, REF_OPEN = 0, 1, 2, 3  # REF_NORMAL: closed mesh, pseudo-normal sign (inside
# confirmed by the winding number); REF_OPEN: open mesh, pseudo-normal sign only
ANCHOR_SAFETY = 0.999  # a sign anchor covers the open ball of 0.999 x its distance (float rounding)


@wp.struct
class PairRef:
    """Reference geometry per slot (one per distinct geometry of a pair across worlds), limits per pair, shape ids per
    (pair, world)."""

    kind: wp.array(dtype=wp.int32)          # [S] REF_*
    geo: wp.array(dtype=wp.int32)           # [S] partner GeoType (analytic)
    scale: wp.array(dtype=wp.vec3)          # [S] partner scale (analytic)
    mesh: wp.array(dtype=wp.uint64)         # [S] partner wp.Mesh id (mesh kinds)
    margin: wp.array(dtype=wp.float32)      # [S] margin sum
    march: wp.array(dtype=wp.float32)       # [S] analytic march step [m]
    tunnel_depth: wp.array(dtype=wp.float32)    # [S]
    cross_floor: wp.array(dtype=wp.float32)     # [S] a tunnel march point counts as inside past this depth [m]
    slot_pair: wp.array(dtype=wp.int32)     # [S] the pair of each slot
    s2: wp.array(dtype=wp.float32)          # [P] PAIR_PEN_WARN limit
    s1: wp.array(dtype=wp.float32)          # [P] PAIR_PEN_FAIL limit
    samp_shape: wp.array(dtype=wp.int32)    # [P*W]
    part_shape: wp.array(dtype=wp.int32)    # [P*W]


@wp.struct
class Accum:
    hist: wp.array(dtype=wp.int32)          # [P*BINS]
    pair_max: wp.array(dtype=wp.float32)    # [P]
    world_max: wp.array(dtype=wp.float32)   # [W]
    tunnels: wp.array(dtype=wp.int32)       # [P*W]
    step_max: wp.array(dtype=wp.float32)    # [P*W], this step
    step_gap: wp.array(dtype=wp.float32)    # [W], this step
    step_tunnel: wp.array(dtype=wp.int32)   # [W], this step
    row_n: wp.array(dtype=wp.int32)         # [P*W]
    row_pt: wp.array(dtype=wp.vec3)         # [P*W*R]


@wp.struct
class SignAnchor:
    """Per (world, sample) partner-frame point where the sign was last settled and its signed distance (0: none)."""

    q: wp.array(dtype=wp.vec3)              # [W*S]
    d: wp.array(dtype=wp.float32)           # [W*S]


@wp.func
def _pose(body: int, body_q: wp.array[wp.transform], x_bs: wp.transform) -> wp.transform:
    if body < 0:
        return x_bs
    return wp.transform_multiply(body_q[body], x_bs)


@wp.func
def _dist(ref: PairRef, p: int, q: wp.vec3) -> float:
    k = ref.kind[p]
    if k == REF_ANALYTIC:
        return sd_primitive(ref.geo[p], ref.scale[p], q)
    if k == REF_WINDING:
        r = wp.mesh_query_point_sign_winding_number(ref.mesh[p], q, 1.0e3)
        return r.sign * wp.length(q - wp.mesh_eval_position(ref.mesh[p], r.face, r.u, r.v))
    r = wp.mesh_query_point_sign_normal(ref.mesh[p], q, 1.0e3)
    sign = r.sign
    if sign < 0.0 and k == REF_NORMAL:
        # Warp's pseudo-normal sign reads "inside" on the mid-plane between two facing walls of a closed mesh (the
        # USB-A plug's tongue cavity: insert top vs shell top, 0.975 mm deep at the seated pose); the winding number
        # settles inside samples (rare, so the extra query is cheap)
        sign = wp.mesh_query_point_sign_winding_number(ref.mesh[p], q, 1.0e3).sign
    return sign * wp.length(q - wp.mesh_eval_position(ref.mesh[p], r.face, r.u, r.v))


@wp.func
def _dist_anchored(ref: PairRef, p: int, q: wp.vec3, anchor: SignAnchor, i: int) -> float:
    """Signed distance to a closed mesh with sample i's anchored sign: inside the anchor's ball a closest-point
    query only; outside it the winding number settles the sign and q becomes the anchor. Analytic and open-mesh
    partners use ``_dist``."""
    k = ref.kind[p]
    if k == REF_ANALYTIC or k == REF_OPEN:
        return _dist(ref, p, q)
    a = anchor.d[i]
    if wp.length(q - anchor.q[i]) < ANCHOR_SAFETY * wp.abs(a):
        r = wp.mesh_query_point_no_sign(ref.mesh[p], q, 1.0e3)
        return wp.sign(a) * wp.length(q - wp.mesh_eval_position(ref.mesh[p], r.face, r.u, r.v))
    rw = wp.mesh_query_point_sign_winding_number(ref.mesh[p], q, 1.0e3)
    d = rw.sign * wp.length(q - wp.mesh_eval_position(ref.mesh[p], rw.face, rw.u, rw.v))
    anchor.q[i] = q
    anchor.d[i] = d
    return d


@wp.func
def _crosses(ref: PairRef, p: int, a: wp.vec3, b: wp.vec3) -> bool:
    """Does the segment a->b (partner frame) pass through the partner interior?"""
    seg = b - a
    length = wp.length(seg)
    if length <= ref.march[p]:
        return False                 # shorter than half the thinnest wall: cannot pass through
    steps = wp.min(int(wp.ceil(length / ref.march[p])), 256)
    for k in range(1, steps):
        if _dist(ref, p, a + seg * (float(k) / float(steps))) < -ref.cross_floor[p]:
            return True
    return False


@wp.kernel
def collect_rows(count: wp.array[wp.int32], pair: wp.array[wp.int32], world: wp.array[wp.int32],
                 flags: wp.array[wp.uint8], force: wp.array[wp.vec3], use_force: int, body_q: wp.array[wp.transform],
                 shape_body: wp.array[wp.int32], shape0: wp.array[wp.int32], point0: wp.array[wp.vec3],
                 world_count: int, acc: Accum):
    i = wp.tid()
    if i >= count[0]:
        return
    p = pair[i]
    if p < 0:
        return
    active = (int(flags[i]) & 1) == 0                      # not SPECULATIVE: separation <= 0 at collide time
    if use_force != 0:
        active = wp.length(force[i]) > FORCE_MIN_N
    if not active:
        return
    pw = p * world_count + wp.max(world[i], 0)
    slot = wp.atomic_add(acc.row_n, pw, 1)
    if slot < MAX_ROWS_PER_PAIR_WORLD:
        b = shape_body[shape0[i]]
        x = point0[i]
        if b >= 0:
            x = wp.transform_point(body_q[b], x)
        acc.row_pt[pw * MAX_ROWS_PER_PAIR_WORLD + slot] = x


@wp.func
def _near_row(acc: Accum, pw: int, x: wp.vec3, spacing: float) -> bool:
    n = wp.min(acc.row_n[pw], MAX_ROWS_PER_PAIR_WORLD)
    for k in range(n):
        if wp.length(acc.row_pt[pw * MAX_ROWS_PER_PAIR_WORLD + k] - x) <= spacing:
            return True
    return False


@wp.kernel
def measure(world_count: int, spacing: float, task_sample: wp.array[wp.int32], task_world: wp.array[wp.int32],
            sample_slot: wp.array[wp.int32], sample_pt: wp.array[wp.vec3], body_q: wp.array[wp.transform],
            body_q_prev: wp.array[wp.transform], shape_body: wp.array[wp.int32], shape_xform: wp.array[wp.transform],
            ref: PairRef, acc: Accum, anchor: SignAnchor):
    tid = wp.tid()
    j, w = task_sample[tid], task_world[tid]
    g = sample_slot[j]
    p = ref.slot_pair[g]
    pw = p * world_count + w
    ss = ref.samp_shape[pw]
    ps = ref.part_shape[pw]
    xs = _pose(shape_body[ss], body_q, shape_xform[ss])
    xs0 = _pose(shape_body[ss], body_q_prev, shape_xform[ss])
    xp = _pose(shape_body[ps], body_q, shape_xform[ps])
    xp0 = _pose(shape_body[ps], body_q_prev, shape_xform[ps])
    xw = wp.transform_point(xs, sample_pt[j])
    q1 = wp.transform_point(wp.transform_inverse(xp), xw)
    q0 = wp.transform_point(wp.transform_inverse(xp0), wp.transform_point(xs0, sample_pt[j]))
    d1 = _dist_anchored(ref, g, q1, anchor, tid) - ref.margin[g]
    if d1 > 0.0:
        wp.atomic_min(acc.step_gap, w, d1)
    tunnel = bool(False)
    if d1 < -ref.tunnel_depth[g] or (d1 >= 0.0 and d1 < wp.length(q1 - q0)):
        d0 = _dist(ref, g, q0) - ref.margin[g]
        tunnel = d0 >= 0.0 and d1 < -ref.tunnel_depth[g]
        if d0 >= 0.0 and d1 >= 0.0:
            tunnel = _crosses(ref, g, q0, q1)
    if tunnel:
        wp.atomic_add(acc.tunnels, pw, 1)
        wp.atomic_add(acc.step_tunnel, w, 1)
    if d1 > 0.0 and not _near_row(acc, pw, xw, spacing):
        return
    pen = wp.max(0.0, -d1)
    b = wp.min(int(pen * 1.0e6), BINS - 1)
    wp.atomic_add(acc.hist, p * BINS + b, 1)
    wp.atomic_max(acc.step_max, pw, pen)
    wp.atomic_max(acc.pair_max, p, pen)
    wp.atomic_max(acc.world_max, w, pen)


@wp.kernel
def write_health(n_pairs: int, world_count: int, ref: PairRef, acc: Accum, overflow: wp.array[wp.int32],
                 health: wp.array[StepHealth]):
    w = wp.tid()
    h = health[w]
    flags = h.flags
    worst = float(0.0)
    worst_pair = int(-1)  # the pair of the deepest pen; -1 when no pair penetrates
    for p in range(n_pairs):
        pen = acc.step_max[p * world_count + w]
        if pen > worst:
            worst = pen
            worst_pair = p
        if pen > ref.s1[p]:
            flags = flags | F_FAIL
        elif pen > ref.s2[p]:
            flags = flags | F_WARN
    if acc.step_tunnel[w] > 0:
        flags = flags | F_FAIL
    if overflow[0] > 0:
        flags = flags | F_OVER
    h.flags = flags
    if worst > h.max_pair_pen_m:  # the inbox may already hold a deeper pen (and its pair): keep both then
        h.max_pair_pen_m = worst
        h.max_pair_id = worst_pair
    h.min_pair_gap_m = wp.min(h.min_pair_gap_m, acc.step_gap[w])
    health[w] = h


@wp.kernel
def _add_counter(src: wp.array[wp.int32], idx: int, dst: wp.array[wp.int32]):
    if src[idx] > 0:
        wp.atomic_add(dst, 0, src[idx])


def _thin(geo: int, scale: np.ndarray) -> float:
    if geo == GEO_BOX:
        return float(np.min(scale))
    if geo == GEO_SPHERE:
        return float(scale[0])
    if geo in (GEO_CAPSULE, GEO_CYLINDER):
        return float(min(scale[0], scale[1]))
    return 1e-3


def cross_floor(max_pen: float, march: float) -> float:
    """The depth past which a tunnel march point counts as inside: the pair's own contact depth (``max_pen``), so a
    sample sliding along a wall it touches is not a tunnel, capped at a quarter of the march step. A segment that
    crosses a wall of the thinnest thickness 2 x ``march`` has a march point at least march / 2 deep, so that crossing
    is still seen."""
    return min(float(max_pen), 0.25 * float(march))


class GapMonitor:
    """See module docstring. Satisfies ``GapMonitorLike``."""

    def __init__(self, layer, *, sample_spacing: float = 5e-4, tunnel_depth: float | None = None,
                 winding: bool = False) -> None:
        self.layer, self.spacing, self._winding = layer, float(sample_spacing), winding
        model = layer.model
        self.device, self.P, self.W = layer.device, len(layer.pairs), model.world_count
        self._meshes: list = []                       # keep wp.Mesh objects alive
        self._source_keys: dict[int, str] = {}
        self.reference: list[str] = []
        pts: list[np.ndarray] = []
        cols = self._pair_columns(model, tunnel_depth, pts)
        self.n_samples = int(sum(len(x) for x in pts))
        flat = np.concatenate(pts) if pts else np.zeros((0, 3))
        dev = self.device
        i32 = lambda x: wp.array(np.asarray(x, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
        self.sample_pt = wp.array(flat.astype(np.float32), dtype=wp.vec3, device=dev)
        self.sample_slot = i32(np.concatenate([np.full(len(x), g) for g, x in enumerate(pts)]) if pts else [])
        self.n_tasks = len(cols["task_sample"])
        self.task_sample, self.task_world = i32(cols["task_sample"]), i32(cols["task_world"])
        self.slots = len(cols["kind"])
        self.ref = self._make_ref(cols)
        self.acc = self._make_acc()
        self.anchor = SignAnchor()
        self.anchor.q = wp.zeros(max(self.n_tasks, 1), dtype=wp.vec3, device=dev)
        self.anchor.d = wp.zeros(max(self.n_tasks, 1), dtype=wp.float32, device=dev)
        self._ovf = wp.zeros(1, dtype=wp.int32, device=dev)
        log.info("GapMonitor: %d pairs, %d geometry slots, %d samples, %d (sample, world) tasks, references %s",
                 self.P, self.slots, self.n_samples, self.n_tasks, self.reference)

    # ---------------------------------------------------------------------------------------------- setup ---
    def _source_key(self, source) -> str:
        """Content key of a shape's mesh source (worlds may hold equal copies), '' for none."""
        if source is None:
            return ""
        if id(source) not in self._source_keys:
            h = hashlib.sha1(np.ascontiguousarray(source.vertices, np.float64).tobytes())
            h.update(np.ascontiguousarray(source.indices, np.int64).tobytes())
            self._source_keys[id(source)] = h.hexdigest()
        return self._source_keys[id(source)]

    def _variants(self, model, spec, starts) -> dict[tuple, list[int]]:
        """Worlds grouped by the pair's geometry (each shape's type, scale and mesh, the margin sum): one slot each."""
        types, scales, margins = model.shape_type.numpy(), model.shape_scale.numpy(), model.shape_margin.numpy()
        out: dict[tuple, list[int]] = {}
        for w in range(self.W):
            ga, gb = int(starts[w] + spec.shape_a), int(starts[w] + spec.shape_b)
            key = tuple((int(types[g]), tuple(float(x) for x in scales[g]), self._source_key(model.shape_source[g]))
                        for g in (ga, gb)) + (float(margins[ga] + margins[gb]),)
            out.setdefault(key, []).append(w)
        return out

    def _pair_columns(self, model, tunnel_depth, pts) -> dict[str, list]:
        """Reference columns per slot, shape ids per (pair, world) and the (sample, world) task list. Each world is
        measured against its own pair geometry (worlds may differ, e.g. CT-2's plate per world: SCR-XB4-10-1)."""
        starts = model.shape_world_start.numpy()
        cols: dict[str, list] = {k: [] for k in ("kind", "geo", "scale", "mesh", "margin", "march", "td", "floor",
                                                 "s2", "s1", "samp", "part", "slot_pair", "task_sample", "task_world")}
        for p, spec in enumerate(self.layer.pairs):
            samp, part, refs = [0] * self.W, [0] * self.W, []
            for worlds in self._variants(model, spec, starts).values():
                n0 = sum(len(x) for x in pts)
                samp_t, part_t, name = self._slot(model, cols, spec, int(starts[worlds[0]]), tunnel_depth, pts)
                cols["slot_pair"].append(p)
                refs.append(name)
                for w in worlds:
                    samp[w], part[w] = int(starts[w] + samp_t), int(starts[w] + part_t)
                    cols["task_sample"] += range(n0, n0 + len(pts[-1]))
                    cols["task_world"] += [w] * len(pts[-1])
            cols["s2"].append(spec.s2_limit())
            cols["s1"].append(spec.s1_limit())
            cols["samp"] += samp
            cols["part"] += part
            self.reference.append("+".join(dict.fromkeys(refs)))
        return cols

    def _slot(self, model, cols, spec, start: int, tunnel_depth, pts) -> tuple[int, int, str]:
        """One slot from the world whose first shape is ``start``: samples of the smaller side, the partner's
        reference; returns the template ids (sampled, partner) and the reference's name."""
        types, scales, margins = model.shape_type.numpy(), model.shape_scale.numpy(), model.shape_margin.numpy()
        ga, gb = start + spec.shape_a, start + spec.shape_b
        area = [surface_area_estimate(int(types[g]), scales[g], model.shape_source[g]) for g in (ga, gb)]
        samp_t, part_t, crop = ((spec.shape_a, spec.shape_b, spec.crop_a) if area[0] <= area[1]
                                else (spec.shape_b, spec.shape_a, spec.crop_b))
        gs, gp = start + samp_t, start + part_t
        pts.append(surface_samples(int(types[gs]), scales[gs], model.shape_source[gs], self.spacing, crop))
        wall = spec.wall_thickness
        self._mesh_march = 0.5 * wall if wall else 1e-4
        name = self._partner(cols, int(types[gp]), scales[gp], model.shape_source[gp])
        cols["floor"].append(cross_floor(spec.resolved_max_penetration(), cols["march"][-1]))
        cols["td"].append(tunnel_depth if tunnel_depth is not None else min(1e-3, 0.5 * wall) if wall else 1e-3)
        cols["margin"].append(float(margins[ga] + margins[gb]))
        return samp_t, part_t, name

    def _partner(self, cols: dict, geo: int, scale: np.ndarray, source) -> str:
        """Append one slot's partner reference columns; returns the reference's name."""
        cols["geo"].append(geo)
        cols["scale"].append(tuple(float(x) for x in scale))
        if has_exact_sdf(geo, scale):
            cols["march"].append(max(0.5 * _thin(geo, scale), 1e-6))
            cols["kind"].append(REF_ANALYTIC)
            cols["mesh"].append(0)
            return "analytic"
        v = np.asarray(source.vertices, dtype=np.float32) * np.asarray(scale, np.float32)
        t = np.asarray(source.indices, dtype=np.int32).reshape(-1)
        closed = is_watertight(t.reshape(-1, 3))
        winding = closed and self._winding
        mesh = wp.Mesh(wp.array(v, dtype=wp.vec3, device=self.device), wp.array(t, dtype=wp.int32, device=self.device),
                       support_winding_number=closed)
        self._meshes.append(mesh)
        cols["march"].append(self._mesh_march)
        cols["kind"].append(REF_WINDING if winding else REF_NORMAL if closed else REF_OPEN)
        cols["mesh"].append(int(mesh.id))
        return "winding_number" if winding else "pseudo_normal" if closed else "normal_sign_open"

    def _make_ref(self, c: dict) -> PairRef:
        dev, r = self.device, PairRef()
        arr = lambda x, dt: wp.array(np.asarray(x if len(x) else [0]), dtype=dt, device=dev)  # noqa: E731
        r.kind, r.geo = arr(c["kind"], wp.int32), arr(c["geo"], wp.int32)
        r.scale = wp.array(np.asarray(c["scale"] or [(0, 0, 0)], np.float32), dtype=wp.vec3, device=dev)
        r.mesh = wp.array(np.asarray(c["mesh"] or [0], np.uint64), dtype=wp.uint64, device=dev)
        r.margin, r.march = arr(c["margin"], wp.float32), arr(c["march"], wp.float32)
        r.tunnel_depth, r.s2, r.s1 = arr(c["td"], wp.float32), arr(c["s2"], wp.float32), arr(c["s1"], wp.float32)
        r.cross_floor = arr(c["floor"], wp.float32)
        r.samp_shape, r.part_shape = arr(c["samp"], wp.int32), arr(c["part"], wp.int32)
        r.slot_pair = arr(c["slot_pair"], wp.int32)
        return r

    def _make_acc(self) -> Accum:
        dev, a, P, W = self.device, Accum(), max(self.P, 1), self.W
        a.hist = wp.zeros(P * BINS, dtype=wp.int32, device=dev)
        a.pair_max, a.world_max = wp.zeros(P, dtype=wp.float32, device=dev), wp.zeros(W, dtype=wp.float32, device=dev)
        a.tunnels = wp.zeros(P * W, dtype=wp.int32, device=dev)
        a.step_max = wp.zeros(P * W, dtype=wp.float32, device=dev)
        a.step_gap = wp.full(W, np.inf, dtype=wp.float32, device=dev)
        a.step_tunnel = wp.zeros(W, dtype=wp.int32, device=dev)
        a.row_n = wp.zeros(P * W, dtype=wp.int32, device=dev)
        a.row_pt = wp.zeros(P * W * MAX_ROWS_PER_PAIR_WORLD, dtype=wp.vec3, device=dev)
        return a

    # ---------------------------------------------------------------------------------------------- run ---
    def update(self, state_prev, state, health: wp.array | None = None) -> None:
        """Measure every sample of every pair and world at ``state`` (tunnels from ``state_prev``). Graph-safe."""
        if self.P == 0:
            return
        a, lay, m = self.acc, self.layer, self.layer.model
        a.step_max.zero_()
        a.step_gap.fill_(np.inf)
        a.step_tunnel.zero_()
        a.row_n.zero_()
        c, side = lay.contacts(), lay.sidecar()
        binding = getattr(lay, "_binding", None)
        use_force = int(binding is not None and binding.installed)
        if use_force:
            binding.precision_forces(c)
        wp.launch(collect_rows, dim=c.rigid_contact_max, device=self.device, inputs=[
            c.rigid_contact_count, side.pair, side.world, side.flags, c.rigid_contact_force, use_force, state.body_q,
            m.shape_body, c.rigid_contact_shape0, c.rigid_contact_point0, self.W, a])
        wp.launch(measure, dim=self.n_tasks, device=self.device, inputs=[
            self.W, self.spacing, self.task_sample, self.task_world, self.sample_slot, self.sample_pt, state.body_q,
            state_prev.body_q, m.shape_body, m.shape_transform, self.ref, a, self.anchor])
        if health is not None:
            self._overflow()
            wp.launch(write_health, dim=self.W, device=self.device,
                      inputs=[self.P, self.W, self.ref, a, self._ovf, health])

    def _overflow(self) -> None:
        self._ovf.zero_()
        wp.launch(_add_counter, dim=1, inputs=[self.layer.overflow, 0, self._ovf], device=self.device)
        wp.launch(_add_counter, dim=1, inputs=[self.layer.hydro_overflow, 0, self._ovf], device=self.device)
        for buf, _ in self.layer.precision_streams:
            wp.launch(_add_counter, dim=1, inputs=[buf.contact_counters, 2, self._ovf], device=self.device)

    def reset(self) -> None:
        """Clear the accumulated statistics (histograms, maxima, tunnel counts)."""
        a = self.acc
        for arr in (a.hist, a.pair_max, a.world_max, a.tunnels):
            arr.zero_()

    def report(self) -> GapReport:
        """Host readback of everything accumulated since construction or ``reset``."""
        P, W = self.P, self.W
        hist = self.acc.hist.numpy()[: P * BINS].reshape(P, BINS).astype(np.int64)
        total = hist.sum(axis=1)
        p99 = np.zeros(P)
        for p in range(P):
            if total[p] > 0:
                b = int(np.searchsorted(np.cumsum(hist[p]), 0.99 * total[p]))
                p99[p] = (b + 1) * 1e-6           # upper bin edge: conservative
        return GapReport(pair_p99_m=p99, pair_max_m=self.acc.pair_max.numpy()[:P].astype(np.float64),
                         pair_samples=total, world_max_m=self.acc.world_max.numpy().astype(np.float64),
                         tunnel_events=self.acc.tunnels.numpy()[: P * W].reshape(P, W).astype(np.int64),
                         histogram_um=hist, reference=tuple(self.reference))
