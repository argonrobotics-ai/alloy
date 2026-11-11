# SPDX-License-Identifier: Apache-2.0
"""Sweep CCD (SPEC §3.5 ``ccd="sweep"``): conservative advancement on the partner's reference geometry; appends one
SWEPT speculative row per (pair, world) whose predicted motion would reach the partner beyond L0's reach.

Prediction: each sample's velocity relative to the partner from ``state.body_qd`` (COM velocity + ω × r), times
``dt·SAFETY``, in the partner frame (``state_prev`` is accepted for the SPEC signature and not needed). Each surface sample of the smaller side (GapMonitor's samples and reference geometry: analytic
SDF or the original mesh, never the narrow-phase texture) is advanced along that segment with steps ``t += d/L``
(1-Lipschitz distance: never skips the surface). Samples already within the pair's ``speculative_gap_max`` are left
to the pipeline's velocity-expanded rows (L0), so rows are not doubled. Per (pair, world) up to ``ROWS`` samples get
a row: the earliest time of impact and the four extremes of the hitting samples across the motion (a single rim row
lets a flat face pivot through a thin wall; ties: lowest sample index). Each row has the partner as shape0, the
outward partner normal at that sample's hit point and separation = its current distance along that normal, flags
SPECULATIVE | SWEPT | NEW, weight 1/rows and the pair's K.
The solver's speculative path (L1: MJWarp append or Newton speculative contacts) then limits the approach to that
separation per step. Row slots come from a device scan, so the result is deterministic and graph-safe.
"""

from __future__ import annotations

import weakref

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.monitor import GapMonitor, PairRef, _dist, _pose
from alloy.contact.types import ContactFlags
from alloy.core.log import get_logger

log = get_logger("contact.ccd")

SAFETY = 1.25                 # predicted sweep = 1.25 x the last step's relative motion
MAX_ITERS = 32
HIT_TOL = 1e-6                # m
NO_HIT = 2.0
ROWS = 5                      # earliest hit + 4 extremes of the impending contact patch
NONE = 2147483647
F_SWEPT_ROW = wp.constant(wp.uint8(int(ContactFlags.SPECULATIVE | ContactFlags.SWEPT | ContactFlags.NEW)))
_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


@wp.struct
class SweepIn:
    world_count: int
    task_sample: wp.array(dtype=wp.int32)    # GapMonitor's (sample, world) tasks
    task_world: wp.array(dtype=wp.int32)
    sample_slot: wp.array(dtype=wp.int32)    # sample -> geometry slot (PairRef columns; slot_pair -> pair)
    sample_pt: wp.array(dtype=wp.vec3)
    sweep_on: wp.array(dtype=wp.int32)       # [P] 1 if ccd == "sweep"
    reach: wp.array(dtype=wp.float32)        # [P] L0 reach (speculative_gap_max)
    rows: int                                # ROWS / 4 / 3 as runtime values: Warp unrolls constant-range loops,
    axes: int                                # and unrolled inlined distance queries made the CUDA build of this
    dims: int                                # module take > 12 min
    body_q: wp.array(dtype=wp.transform)
    body_qd: wp.array(dtype=wp.spatial_vector)
    body_com: wp.array(dtype=wp.vec3)
    dt: float
    shape_body: wp.array(dtype=wp.int32)
    shape_xform: wp.array(dtype=wp.transform)


@wp.func
def _point_vel(s: SweepIn, body: int, x: wp.vec3) -> wp.vec3:
    if body < 0:
        return wp.vec3(0.0)
    qd = s.body_qd[body]
    com = wp.transform_point(s.body_q[body], s.body_com[body])
    return wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), x - com)


@wp.func
def _frames(s: SweepIn, ref: PairRef, pw: int, j: int):
    """Sample position now (q1) and its predicted sweep over the next step (seg), in the partner shape frame."""
    ss, ps = ref.samp_shape[pw], ref.part_shape[pw]
    bs, bp = s.shape_body[ss], s.shape_body[ps]
    xw = wp.transform_point(_pose(bs, s.body_q, s.shape_xform[ss]), s.sample_pt[j])
    xp = _pose(bp, s.body_q, s.shape_xform[ps])
    v = _point_vel(s, bs, xw) - _point_vel(s, bp, xw)
    q1 = wp.transform_point(wp.transform_inverse(xp), xw)
    seg = wp.transform_vector(wp.transform_inverse(xp), v) * (s.dt * SAFETY)
    return q1, seg


@wp.func
def _task_pw(s: SweepIn, ref: PairRef, tid: int) -> int:
    """The (pair, world) index of task ``tid``."""
    w = s.task_world[tid]
    return ref.slot_pair[s.sample_slot[s.task_sample[tid]]] * s.world_count + w


@wp.func
def _advance(ref: PairRef, p: int, q1: wp.vec3, seg: wp.vec3, d1: float) -> float:
    """Conservative advancement from q1 along seg; time of impact in [0, 1) or NO_HIT."""
    length = wp.length(seg)
    if length <= d1:
        return NO_HIT
    t = float(0.0)
    d = d1
    for _ in range(MAX_ITERS):
        if d <= HIT_TOL:
            return t
        t = t + d / length
        if t >= 1.0:
            return NO_HIT
        d = _dist(ref, p, q1 + seg * t) - ref.margin[p]
    return NO_HIT


@wp.kernel
def _toi(s: SweepIn, ref: PairRef, toi: wp.array[wp.float32], best: wp.array[wp.float32]):
    tid = wp.tid()
    j = s.task_sample[tid]
    g = s.sample_slot[j]
    p = ref.slot_pair[g]
    toi[tid] = NO_HIT
    if s.sweep_on[p] == 0:
        return
    pw = p * s.world_count + s.task_world[tid]
    q1, seg = _frames(s, ref, pw, j)
    d1 = _dist(ref, g, q1) - ref.margin[g]
    if d1 <= s.reach[p]:
        return                                   # inside L0's reach (or penetrating): the pipeline's rows
    t = _advance(ref, g, q1, seg, d1)
    toi[tid] = t
    if t < NO_HIT:
        wp.atomic_min(best, pw, t)


@wp.kernel
def _pick(s: SweepIn, ref: PairRef, toi: wp.array[wp.float32], best: wp.array[wp.float32],
          cand: wp.array[wp.int32]):
    tid = wp.tid()
    pw = _task_pw(s, ref, tid)
    if toi[tid] < NO_HIT and toi[tid] == best[pw]:
        wp.atomic_min(cand, pw * ROWS, tid)      # a task id: one (pair, world)'s tasks run in sample order


@wp.func
def _proj(s: SweepIn, ref: PairRef, pw: int, j: int, j0: int, k: int) -> float:
    """Extent of sample j along axis k of (u, v, -u, -v), the plane across the earliest hit's motion."""
    q0, seg0 = _frames(s, ref, pw, j0)
    d = wp.normalize(seg0)
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(d[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    u = wp.normalize(wp.cross(d, a))
    v = wp.cross(d, u)
    q1, seg = _frames(s, ref, pw, j)
    x = wp.where(k % 2 == 0, wp.dot(q1, u), wp.dot(q1, v))
    return wp.where(k < 2, x, -x)


@wp.kernel
def _extent(s: SweepIn, ref: PairRef, toi: wp.array[wp.float32], cand: wp.array[wp.int32],
            ext: wp.array[wp.float32]):
    tid = wp.tid()
    if toi[tid] >= NO_HIT:
        return
    pw = _task_pw(s, ref, tid)
    j = s.task_sample[tid]
    j0 = s.task_sample[cand[pw * ROWS]]
    for k in range(s.axes):
        wp.atomic_max(ext, pw * 4 + k, _proj(s, ref, pw, j, j0, k))


@wp.kernel
def _extent_pick(s: SweepIn, ref: PairRef, toi: wp.array[wp.float32], ext: wp.array[wp.float32],
                 cand: wp.array[wp.int32]):
    tid = wp.tid()
    if toi[tid] >= NO_HIT:
        return
    pw = _task_pw(s, ref, tid)
    j = s.task_sample[tid]
    j0 = s.task_sample[cand[pw * ROWS]]
    for k in range(s.axes):
        if _proj(s, ref, pw, j, j0, k) == ext[pw * 4 + k]:
            wp.atomic_min(cand, pw * ROWS + 1 + k, tid)


@wp.func
def _is_new(cand: wp.array[wp.int32], pw: int, c: int) -> bool:
    j = cand[pw * ROWS + c]
    if j == NONE:
        return False
    for e in range(c):
        if cand[pw * ROWS + e] == j:
            return False
    return True


@wp.kernel
def _mask(cand: wp.array[wp.int32], rows: int, mask: wp.array[wp.int32]):
    pw = wp.tid()
    m = int(0)
    for c in range(rows):
        if _is_new(cand, pw, c):
            m += 1
    mask[pw] = m


@wp.func
def _normal(ref: PairRef, p: int, q: wp.vec3, fallback: wp.vec3, dims: int) -> wp.vec3:
    """Outward partner normal at q: central differences of the reference distance (1e-5 m); ``dims`` = 3 at run
    time (a runtime loop keeps one inlined distance query pair instead of six)."""
    g = wp.vec3(0.0)
    for a in range(dims):
        e = wp.vec3(0.0)
        e[a] = 1.0e-5
        g[a] = _dist(ref, p, q + e) - _dist(ref, p, q - e)
    n = wp.length(g)
    if n < 1.0e-12:
        return fallback
    return g / n


@wp.struct
class RowsOut:
    count: wp.array(dtype=wp.int32)
    start: wp.array(dtype=wp.int32)          # scratch: the count before this sweep
    overflow: wp.array(dtype=wp.int32)
    shape0: wp.array(dtype=wp.int32)
    shape1: wp.array(dtype=wp.int32)
    point0: wp.array(dtype=wp.vec3)
    point1: wp.array(dtype=wp.vec3)
    offset0: wp.array(dtype=wp.vec3)
    offset1: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    margin0: wp.array(dtype=wp.float32)
    margin1: wp.array(dtype=wp.float32)
    stiffness: wp.array(dtype=wp.float32)
    damping: wp.array(dtype=wp.float32)
    friction: wp.array(dtype=wp.float32)
    match_index: wp.array(dtype=wp.int32)    # length 1 when the buffer has none
    row_k: wp.array(dtype=wp.float32)
    cls: wp.array(dtype=wp.int8)
    pair: wp.array(dtype=wp.int32)
    flags: wp.array(dtype=wp.uint8)
    world: wp.array(dtype=wp.int32)
    weight: wp.array(dtype=wp.float32)
    area: wp.array(dtype=wp.float32)
    toi: wp.array(dtype=wp.float32)
    pair_class: wp.array(dtype=wp.int8)
    pair_k: wp.array(dtype=wp.float32)
    shape_margin: wp.array(dtype=wp.float32)


@wp.kernel
def _start(out: RowsOut):
    out.start[0] = out.count[0]


@wp.func
def _write_sidecar(out: RowsOut, k: int, p: int, w: int, t: float, weight: float):
    out.cls[k] = out.pair_class[p]
    out.pair[k] = p
    out.flags[k] = F_SWEPT_ROW
    out.world[k] = w
    out.weight[k] = weight
    out.area[k] = 0.0
    out.toi[k] = t
    out.stiffness[k] = out.pair_k[p]
    out.row_k[k] = out.pair_k[p]
    out.damping[k] = 0.0
    out.friction[k] = 1.0
    if out.match_index.shape[0] > k:
        out.match_index[k] = -1


@wp.func
def _emit_row(s: SweepIn, ref: PairRef, out: RowsOut, pw: int, j: int, t: float, k: int, weight: float):
    p = pw // s.world_count
    g = s.sample_slot[j]
    q1, seg = _frames(s, ref, pw, j)
    qh = q1 + seg * t
    n = _normal(ref, g, qh, -wp.normalize(seg), s.dims)
    ph = qh - n * (_dist(ref, g, qh) - ref.margin[g])        # on the partner surface (margin included)
    ss, ps = ref.samp_shape[pw], ref.part_shape[pw]
    xp = _pose(s.shape_body[ps], s.body_q, s.shape_xform[ps])
    out.shape0[k] = ps
    out.shape1[k] = ss
    out.point0[k] = wp.transform_point(s.shape_xform[ps], ph)              # partner body frame (world if static)
    out.point1[k] = wp.transform_point(s.shape_xform[ss], s.sample_pt[j])  # sampled body frame
    out.offset0[k] = wp.vec3(0.0)
    out.offset1[k] = wp.vec3(0.0)
    out.normal[k] = wp.transform_vector(xp, n)
    out.margin0[k] = out.shape_margin[ps]
    out.margin1[k] = out.shape_margin[ss]
    _write_sidecar(out, k, p, pw - p * s.world_count, t, weight)


@wp.kernel
def _emit(s: SweepIn, ref: PairRef, cand: wp.array[wp.int32], toi: wp.array[wp.float32],
          mask: wp.array[wp.int32], slots: wp.array[wp.int32], out: RowsOut):
    pw = wp.tid()
    m = mask[pw]
    r = int(0)
    for c in range(s.rows):
        k = out.start[0] + slots[pw] + r
        if _is_new(cand, pw, c) and k < out.shape0.shape[0]:
            task = cand[pw * ROWS + c]
            _emit_row(s, ref, out, pw, s.task_sample[task], toi[task], k, 1.0 / float(m))
            r += 1


@wp.kernel
def _bump(mask: wp.array[wp.int32], slots: wp.array[wp.int32], out: RowsOut):
    last = mask.shape[0] - 1
    end = out.start[0] + slots[last] + mask[last]
    cap = out.shape0.shape[0]
    out.count[0] = wp.min(end, cap)
    if end > cap:
        wp.atomic_add(out.overflow, 0, end - cap)


class SweepCCD:
    """Device state of ``sweep_ccd`` for one layer (samples and reference geometry shared with a GapMonitor)."""

    def __init__(self, layer, *, sample_spacing: float = 5e-4) -> None:
        self.layer, self.device = layer, layer.device
        self.mon = GapMonitor(layer, sample_spacing=sample_spacing)
        P, W = len(layer.pairs), layer.model.world_count
        dev = self.device
        self.sweep_on = wp.array(np.asarray([int(s.ccd == "sweep") for s in layer.pairs] or [0], np.int32),
                                 dtype=wp.int32, device=dev)
        reach = [s.gap if s.mode == "hydroelastic" else max(s.gap, s.speculative_gap_max) for s in layer.pairs]
        self.reach = wp.array(np.asarray(reach or [0.0], np.float32), dtype=wp.float32, device=dev)
        self.n = self.mon.n_tasks
        self.toi = wp.zeros(max(self.n, 1), dtype=wp.float32, device=dev)
        self.best = wp.zeros(max(P * W, 1), dtype=wp.float32, device=dev)
        self.cand = wp.zeros(max(P * W, 1) * ROWS, dtype=wp.int32, device=dev)
        self.ext = wp.zeros(max(P * W, 1) * 4, dtype=wp.float32, device=dev)
        self.mask = wp.zeros(max(P * W, 1), dtype=wp.int32, device=dev)
        self.slots = wp.zeros(max(P * W, 1), dtype=wp.int32, device=dev)
        self.out = self._make_out()
        log.info("sweep CCD: %d of %d pairs, %d (sample, world) tasks", int(self.sweep_on.numpy().sum()), P, self.n)

    def _make_out(self) -> RowsOut:
        lay, c, side = self.layer, self.layer.contacts(), self.layer.sidecar()
        o = RowsOut()
        for name in ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1",
                     "stiffness", "damping", "friction", "count"):
            setattr(o, name, getattr(c, f"rigid_contact_{name}"))
        match = getattr(c, "rigid_contact_match_index", None)
        o.match_index = match if match is not None else wp.zeros(1, dtype=wp.int32, device=self.device)
        o.start = wp.zeros(1, dtype=wp.int32, device=self.device)
        o.overflow, o.row_k = lay.overflow, lay.row_stiffness
        for name in ("cls", "pair", "flags", "world", "weight", "area", "toi"):
            setattr(o, name, getattr(side, name))
        o.pair_class, o.pair_k, o.shape_margin = lay._tables.pair_class, lay.pair_k, lay.model.shape_margin
        return o

    def _inputs(self, state: newton.State, dt: float) -> SweepIn:
        s, m = SweepIn(), self.layer.model
        s.world_count, s.sample_pt, s.sample_slot = m.world_count, self.mon.sample_pt, self.mon.sample_slot
        s.task_sample, s.task_world = self.mon.task_sample, self.mon.task_world
        s.sweep_on, s.reach = self.sweep_on, self.reach
        s.body_q, s.body_qd, s.body_com, s.dt = state.body_q, state.body_qd, m.body_com, float(dt)
        s.rows, s.axes, s.dims = ROWS, 4, 3
        s.shape_body, s.shape_xform = m.shape_body, m.shape_transform
        return s

    def run(self, state: newton.State, dt: float) -> None:
        if self.n == 0:
            return
        s, ref, dev = self._inputs(state, dt), self.mon.ref, self.device
        self.best.fill_(NO_HIT)
        self.cand.fill_(NONE)
        self.ext.fill_(-1.0e30)
        nw = self.n
        wp.launch(_toi, dim=nw, inputs=[s, ref, self.toi, self.best], device=dev)
        wp.launch(_pick, dim=nw, inputs=[s, ref, self.toi, self.best, self.cand], device=dev)
        wp.launch(_extent, dim=nw, inputs=[s, ref, self.toi, self.cand, self.ext], device=dev)
        wp.launch(_extent_pick, dim=nw, inputs=[s, ref, self.toi, self.ext, self.cand], device=dev)
        wp.launch(_mask, dim=self.mask.shape[0], inputs=[self.cand, ROWS, self.mask], device=dev)
        wp.utils.array_scan(self.mask, self.slots, inclusive=False)
        wp.launch(_start, dim=1, inputs=[self.out], device=dev)
        wp.launch(_emit, dim=self.mask.shape[0], inputs=[s, ref, self.cand, self.toi, self.mask, self.slots,
                                                         self.out], device=dev)
        wp.launch(_bump, dim=1, inputs=[self.mask, self.slots, self.out], device=dev)


def sweep_ccd(layer, state_prev: newton.State, state: newton.State, contacts, dt: float) -> None:
    """Append SWEPT speculative rows for ``ccd="sweep"`` pairs (call after ``layer.collide``, before the solver
    step). Graph-safe after the first call (which builds the samples)."""
    from alloy.core.errors import AssemblyError  # noqa: PLC0415

    if contacts is not layer.contacts():
        raise AssemblyError("sweep_ccd must be given the layer's own contacts() buffer")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    sw = _CACHE.get(layer)
    if sw is None:
        sw = _CACHE[layer] = SweepCCD(layer)
    sw.run(state, dt)
