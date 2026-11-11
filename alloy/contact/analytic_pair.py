# SPDX-License-Identifier: Apache-2.0
"""Analytic precision pairs (SPEC §3.5 ``analytic_id``, law A9): contact rows from an analytic partner SDF, no texture.

A pair has a *partner* (shape0, ``spec.shape_a``) measured by an analytic SDF, an ISO thread (``sd_thread``) or a flat
plate (a washer), and a *sampled* side (shape1, ``spec.shape_b``): thread flanks on their exact helical profile or a
flat annular face (a nut's bearing face), on a (θ, axial) grid at ``sample_spacing`` (default ``spec.voxel_size``; a
contact-localisation resolution, the partner SDF is analytic). Thread samples cover the centred ``flank_band`` of each
flank (default 80 %) and start half a pitch inside each slab end, where real threads have root radii, crest
truncation and countersinks: a sample on a sharp profile corner meets the partner's corner with a normal 60° off the
flank (measured: one such end row carried the whole 5 N load and raised the run-down torque by 60 %). A slab end with
an ``EndChamfer`` models that geometry instead (``analytic_samples``; SCR-SC3-2: a flat-cut end let a nut's crest run
44 µm into the bolt's last partial turn, C5). Features and sampling live in ``analytic_samples``.

Rows per (pair, world) are reduced over fixed bins: thread bins are (angular sector, tooth, flank), face bins are
sectors. A bin emits one row when its deepest sample is within ``spec.gap``; the row's sample is the one nearest the
bin centre among those within ``TIE_ABS + TIE_FRAC·|d_min|`` of the deepest (samples are pre-sorted by that distance,
ties go to the lowest index). The centre is the bin's uniform-pressure torque radius ``(2/3)(ro³-ri³)/(ro²-ri²)``: an
evenly loaded flank or face is represented there, an edge contact keeps its deepest point. A full M10 nut (8 mm, 12
sectors) gives at most 12·7·2 = 168 thread rows per world, 12·7·3 = 252 with crest rows (``ThreadFeature.crest``).

Row: shape0 = partner, normal = outward partner gradient (central differences, world frame), point0 = the sample
projected onto the partner surface (partner body frame), point1 = the sample (sampled body frame), separation = SDF
value minus the shape margins. Sidecar: PRECISION_SDF, flags NEW (| SPECULATIVE when separation > 0), ``pair`` -1
until the layer owns analytic pairs (its pair tables are sized by its own pairs). Stiffness ``K_pair·weight``: with
``split_k`` (D22) weight = 1/n over the rows of the same (pair, world, normal group +z/-z/radial), counting
speculative rows too (rows are re-selected each step without matching, so a load-bearing count would jump a row's K
by up to the patch size as rows touch); without it every row gets ``K_pair``. MuJoCo's row timeconst comes from the
row's own K and the body's invweight, so splitting K over n rows makes each row √n slower: a 69-row M10 flank patch at
the D22 K let the drive screw the nut 90 µm into the flank (thread twins use ``split_k=False``). Slots come from a
device scan over (world, bin): deterministic and graph-safe; rows past the capacity are counted in ``layer.overflow``.
``_start``/``_bump`` repeat ccd's: launching those builds the whole ccd module (> 12 min, 16 GB on CUDA, unfinished).
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.ccd import RowsOut
from alloy.contact.monitor import _pose
from alloy.contact.precision import pair_stiffness
from alloy.contact.analytic_samples import (END_BAND, END_FLANK_BAND, END_TRIM, SECTORS, FaceFeature,  # noqa: F401
                                            SampleBins, SlabFeature, ThreadFeature, face_samples, sample_bins,
                                            thread_samples, torque_radius)
from alloy.contact.thread_sdf import ThreadParams, parse_thread_id, sd_thread, thread_sdf_params
from alloy.contact.types import ContactClass, ContactFlags, PrecisionPairSpec
from alloy.core.errors import AssemblyError, PrecisionSpecError
from alloy.core.log import get_logger

log = get_logger("contact.analytic_pair")

TIE_ABS = 5.0e-7                    # m: samples this close to the deepest count as tied
TIE_FRAC = 0.1                      # plus this fraction of the deepest distance
GROUP_NZ = 0.25                     # |n_z| above this: an axial (+z / -z) group, else radial
NORMAL_H = 1.0e-5                   # m, central-difference step of the partner gradient
DEFAULT_DESIGN_FORCE_N = 50.0       # as AlloyContactLayer when pair_stiffness is unset
KIND_THREAD, KIND_SLAB = 0, 1
NONE = 2147483647
F_NEW = wp.constant(wp.uint8(int(ContactFlags.NEW)))
F_SPEC_NEW = wp.constant(wp.uint8(int(ContactFlags.NEW | ContactFlags.SPECULATIVE)))
_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class AnalyticPair:
    """``spec.shape_a`` is the partner (shape0), ``spec.shape_b`` the sampled side (template shape ids)."""

    spec: PrecisionPairSpec
    partner: ThreadFeature | SlabFeature
    sampled: ThreadFeature | FaceFeature
    split_k: bool = True            # row K = K_pair/n per patch (D22); False: every row gets K_pair


def partner_params(feat: ThreadFeature | SlabFeature) -> tuple[int, ThreadParams]:
    """(kind, Warp params) of a partner feature; a slab uses only z0/z1 of ``ThreadParams``."""
    if isinstance(feat, ThreadFeature):
        spec = parse_thread_id(feat.analytic_id)
        return KIND_THREAD, thread_sdf_params(spec, z0=feat.z0, z1=feat.z1, outer_radius=feat.outer_radius,
                                              chamfer0=feat.chamfer0, chamfer1=feat.chamfer1).to_warp()
    if isinstance(feat, SlabFeature):
        prm = ThreadParams()
        prm.z0, prm.z1 = float(feat.z0), float(feat.z1)
        return KIND_SLAB, prm
    raise PrecisionSpecError(f"partner must be a ThreadFeature or SlabFeature, got {type(feat).__name__}")


# ---------------------------------------------------------------- device side ---------------------------------------

@wp.struct
class PairIn:
    world_count: int
    n_bins: int
    bin_start: wp.array(dtype=wp.int32)       # [B+1]
    bin_pair: wp.array(dtype=wp.int32)        # [B]
    sample_pt: wp.array(dtype=wp.vec3)        # [N], sampled shape frame
    partner: wp.array(dtype=ThreadParams)     # [P]
    partner_kind: wp.array(dtype=wp.int32)    # [P]
    gap: wp.array(dtype=wp.float32)           # [P]
    split: wp.array(dtype=wp.int32)           # [P] 1: weight 1/n, 0: weight 1
    part_shape: wp.array(dtype=wp.int32)      # [P*W]
    samp_shape: wp.array(dtype=wp.int32)      # [P*W]
    body_q: wp.array(dtype=wp.transform)
    shape_body: wp.array(dtype=wp.int32)
    shape_xform: wp.array(dtype=wp.transform)
    shape_margin: wp.array(dtype=wp.float32)


@wp.func
def _partner_sd(s: PairIn, p: int, q: wp.vec3) -> float:
    prm = s.partner[p]
    if s.partner_kind[p] == KIND_SLAB:
        return wp.max(prm.z0 - q[2], q[2] - prm.z1)
    return sd_thread(prm, q)


@wp.func
def _partner_normal(s: PairIn, p: int, q: wp.vec3) -> wp.vec3:
    h = NORMAL_H
    g = wp.vec3(_partner_sd(s, p, q + wp.vec3(h, 0.0, 0.0)) - _partner_sd(s, p, q - wp.vec3(h, 0.0, 0.0)),
                _partner_sd(s, p, q + wp.vec3(0.0, h, 0.0)) - _partner_sd(s, p, q - wp.vec3(0.0, h, 0.0)),
                _partner_sd(s, p, q + wp.vec3(0.0, 0.0, h)) - _partner_sd(s, p, q - wp.vec3(0.0, 0.0, h)))
    n = wp.length(g)
    if n < 1.0e-12:
        return wp.vec3(0.0, 0.0, 1.0)
    return g / n


@wp.func
def _relative(s: PairIn, pw: int) -> wp.transform:
    """Sampled shape frame -> partner shape frame."""
    ss = s.samp_shape[pw]
    ps = s.part_shape[pw]
    xs = _pose(s.shape_body[ss], s.body_q, s.shape_xform[ss])
    xp = _pose(s.shape_body[ps], s.body_q, s.shape_xform[ps])
    return wp.transform_multiply(wp.transform_inverse(xp), xs)


@wp.func
def _margins(s: PairIn, pw: int) -> float:
    return s.shape_margin[s.part_shape[pw]] + s.shape_margin[s.samp_shape[pw]]


@wp.func
def _group(n: wp.vec3) -> int:
    if n[2] > GROUP_NZ:
        return 0
    if n[2] < -GROUP_NZ:
        return 1
    return 2


@wp.func
def _deepest(s: PairIn, p: int, rel: wp.transform, lo: int, hi: int, msum: float) -> float:
    d = float(1.0e30)
    for j in range(lo, hi):
        d = wp.min(d, _partner_sd(s, p, wp.transform_point(rel, s.sample_pt[j])) - msum)
    return d


@wp.func
def _first_within(s: PairIn, p: int, rel: wp.transform, lo: int, hi: int, lim: float, msum: float) -> int:
    for j in range(lo, hi):
        if _partner_sd(s, p, wp.transform_point(rel, s.sample_pt[j])) - msum <= lim:
            return j
    return lo


@wp.kernel
def _select(s: PairIn, tie_abs: float, tie_frac: float, cand: wp.array[wp.int32], cand_d: wp.array[wp.float32],
            mask: wp.array[wp.int32], group_n: wp.array[wp.int32]):
    tid = wp.tid()
    w = tid // s.n_bins
    b = tid - w * s.n_bins
    p = s.bin_pair[b]
    pw = p * s.world_count + w
    lo = s.bin_start[b]
    hi = s.bin_start[b + 1]
    cand[tid] = NONE
    mask[tid] = 0
    if hi <= lo:
        return
    rel = _relative(s, pw)
    msum = _margins(s, pw)
    dmin = _deepest(s, p, rel, lo, hi, msum)
    if dmin > s.gap[p]:
        return
    j = _first_within(s, p, rel, lo, hi, dmin + tie_abs + tie_frac * wp.abs(dmin), msum)
    q = wp.transform_point(rel, s.sample_pt[j])
    d = _partner_sd(s, p, q) - msum
    cand[tid] = j
    cand_d[tid] = d
    mask[tid] = 1
    wp.atomic_add(group_n, pw * 3 + _group(_partner_normal(s, p, q)), 1)


@wp.kernel
def _start(out: RowsOut):
    out.start[0] = out.count[0]


@wp.kernel
def _bump(mask: wp.array[wp.int32], slots: wp.array[wp.int32], out: RowsOut):
    """New count after the scan (clamped to the capacity; dropped rows go to ``out.overflow``)."""
    last = mask.shape[0] - 1
    end = out.start[0] + slots[last] + mask[last]
    cap = out.shape0.shape[0]
    out.count[0] = wp.min(end, cap)
    if end > cap:
        wp.atomic_add(out.overflow, 0, end - cap)


@wp.func
def _write_extras(out: RowsOut, k: int, p: int, w: int, d: float, weight: float):
    out.cls[k] = out.pair_class[p]
    out.pair[k] = -1
    out.flags[k] = wp.where(d > 0.0, F_SPEC_NEW, F_NEW)
    out.world[k] = w
    out.weight[k] = weight
    out.area[k] = 0.0
    out.toi[k] = 1.0
    out.stiffness[k] = out.pair_k[p] * weight
    out.row_k[k] = out.pair_k[p] * weight
    out.damping[k] = 0.0
    out.friction[k] = 1.0
    if out.match_index.shape[0] > k:
        out.match_index[k] = -1


@wp.kernel
def _emit(s: PairIn, cand: wp.array[wp.int32], cand_d: wp.array[wp.float32], mask: wp.array[wp.int32],
          slots: wp.array[wp.int32], group_n: wp.array[wp.int32], out: RowsOut):
    tid = wp.tid()
    k = out.start[0] + slots[tid]
    if mask[tid] == 0 or k >= out.shape0.shape[0]:
        return
    w = tid // s.n_bins
    p = s.bin_pair[tid - w * s.n_bins]
    pw = p * s.world_count + w
    j = cand[tid]
    ss = s.samp_shape[pw]
    ps = s.part_shape[pw]
    q = wp.transform_point(_relative(s, pw), s.sample_pt[j])
    n = _partner_normal(s, p, q)
    ph = q - n * _partner_sd(s, p, q)                                   # on the partner surface (core, no margin)
    out.shape0[k] = ps
    out.shape1[k] = ss
    out.point0[k] = wp.transform_point(s.shape_xform[ps], ph)          # partner body frame (world if static)
    out.point1[k] = wp.transform_point(s.shape_xform[ss], s.sample_pt[j])
    out.offset0[k] = wp.vec3(0.0)
    out.offset1[k] = wp.vec3(0.0)
    out.normal[k] = wp.transform_vector(_pose(s.shape_body[ps], s.body_q, s.shape_xform[ps]), n)
    out.margin0[k] = out.shape_margin[ps]
    out.margin1[k] = out.shape_margin[ss]
    weight = float(1.0)
    if s.split[p] != 0:
        weight = 1.0 / float(wp.max(group_n[pw * 3 + _group(n)], 1))
    _write_extras(out, k, p, w, cand_d[tid], weight)


# ---------------------------------------------------------------- host ----------------------------------------------

def _world_shapes(model: newton.Model, spec: PrecisionPairSpec) -> tuple[np.ndarray, np.ndarray]:
    """Global (partner, sampled) shape ids per world from template ids (``replicate``/``begin_world`` layout)."""
    starts = model.shape_world_start.numpy()
    a = starts[: model.world_count] + int(spec.shape_a)
    b = starts[: model.world_count] + int(spec.shape_b)
    if np.any(np.maximum(a, b) >= starts[1 : model.world_count + 1]):
        raise AssemblyError(f"analytic pair template shapes ({spec.shape_a}, {spec.shape_b}) outside a world")
    return a.astype(np.int32), b.astype(np.int32)


def _check_pairs(layer, pairs: Sequence[AnalyticPair], shapes: list[tuple[np.ndarray, np.ndarray]]) -> None:
    filt = {tuple(sorted((int(a), int(b)))) for a, b in layer.model.shape_collision_filter_pairs}
    owned = {tuple(sorted(map(int, ab))) for ab in np.asarray(layer.world_pairs).reshape(-1, 2)}
    for pair, (a, b) in zip(pairs, shapes):
        if pair.spec.mode != "sdf":
            raise PrecisionSpecError(f"analytic pair ({pair.spec.shape_a}, {pair.spec.shape_b}) must use mode='sdf'")
        for x, y in zip(a.tolist(), b.tolist()):
            key = tuple(sorted((x, y)))
            if key not in filt:
                raise AssemblyError(f"analytic pair shapes {key} not in the collision filter pairs (collided twice)")
            if key in owned:
                raise AssemblyError(f"analytic pair shapes {key} are also a layer precision pair (rows would double)")
    if layer.native == "mujoco" and not layer.precision_streams:
        raise AssemblyError("native='mujoco' layer without precision pipelines has a 1-row buffer that is never "
                            "cleared; use native='newton' until AlloyContactLayer owns analytic pairs")


class AnalyticPairContacts:
    """Device state for the analytic pairs of one layer (samples, bins, scratch, the row writer)."""

    def __init__(self, layer, pairs: Sequence[AnalyticPair], *, sample_spacing: float | None = None,
                 sectors: int = SECTORS, design_force: float = DEFAULT_DESIGN_FORCE_N) -> None:
        if not pairs:
            raise PrecisionSpecError("attach at least one analytic pair")
        self.layer, self.pairs, self.device = layer, tuple(pairs), layer.device
        model, W, dev = layer.model, layer.model.world_count, layer.device
        shapes = [_world_shapes(model, pr.spec) for pr in self.pairs]
        _check_pairs(layer, self.pairs, shapes)
        bins = [sample_bins(pr.sampled, sample_spacing or pr.spec.voxel_size, sectors) for pr in self.pairs]
        self.bins_per_pair = [sb.n_bins for sb in bins]
        self.n_bins, self.n_samples = sum(self.bins_per_pair), sum(len(sb.points) for sb in bins)
        self.s = self._inputs(model, shapes, bins)
        kinds, prms = zip(*(partner_params(pr.partner) for pr in self.pairs))
        self.s.partner = wp.array(list(prms), dtype=ThreadParams, device=dev)
        self.s.partner_kind = wp.array(np.asarray(kinds, np.int32), dtype=wp.int32, device=dev)
        self.pair_k = np.asarray([pair_stiffness(pr.spec, design_force) for pr in self.pairs], np.float32)
        nbw = self.n_bins * W
        self.cand = wp.zeros(nbw, dtype=wp.int32, device=dev)
        self.cand_d = wp.zeros(nbw, dtype=wp.float32, device=dev)
        self.mask = wp.zeros(nbw, dtype=wp.int32, device=dev)
        self.slots = wp.zeros(nbw, dtype=wp.int32, device=dev)
        self.group_n = wp.zeros(len(self.pairs) * W * 3, dtype=wp.int32, device=dev)
        self.out = self._make_out()
        cap = layer.contacts().rigid_contact_max
        if cap < nbw:
            log.warning("contact capacity %d < %d analytic bins x worlds: rows beyond it are dropped (layer.overflow)",
                        cap, nbw)
        log.info("analytic pairs: %d, %d samples, %d bins/world (at most that many rows), %d worlds",
                 len(self.pairs), self.n_samples, self.n_bins, W)

    def _inputs(self, model, shapes, bins) -> PairIn:
        dev, W = self.device, model.world_count
        s = PairIn()
        s.world_count, s.n_bins = W, self.n_bins
        starts, offset = [], 0
        for sb in bins:
            starts.append(sb.start[:-1] + offset)
            offset += len(sb.points)
        s.bin_start = wp.array(np.concatenate(starts + [[offset]]).astype(np.int32), dtype=wp.int32, device=dev)
        s.bin_pair = wp.array(np.repeat(np.arange(len(bins)), self.bins_per_pair).astype(np.int32), dtype=wp.int32,
                              device=dev)
        s.sample_pt = wp.array(np.concatenate([sb.points for sb in bins]), dtype=wp.vec3, device=dev)
        s.gap = wp.array(np.asarray([pr.spec.gap for pr in self.pairs], np.float32), dtype=wp.float32, device=dev)
        s.split = wp.array(np.asarray([int(pr.split_k) for pr in self.pairs], np.int32), dtype=wp.int32, device=dev)
        s.part_shape = wp.array(np.concatenate([a for a, _ in shapes]), dtype=wp.int32, device=dev)
        s.samp_shape = wp.array(np.concatenate([b for _, b in shapes]), dtype=wp.int32, device=dev)
        s.shape_body, s.shape_xform, s.shape_margin = model.shape_body, model.shape_transform, model.shape_margin
        s.body_q = model.body_q
        return s

    def _make_out(self) -> RowsOut:
        lay, dev = self.layer, self.device
        c, side = lay.contacts(), lay.sidecar()
        o = RowsOut()
        for name in ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1",
                     "stiffness", "damping", "friction", "count"):
            setattr(o, name, getattr(c, f"rigid_contact_{name}"))
        match = getattr(c, "rigid_contact_match_index", None)
        o.match_index = match if match is not None else wp.zeros(1, dtype=wp.int32, device=dev)
        o.start = wp.zeros(1, dtype=wp.int32, device=dev)
        o.overflow, o.row_k = lay.overflow, lay.row_stiffness
        for name in ("cls", "pair", "flags", "world", "weight", "area", "toi"):
            setattr(o, name, getattr(side, name))
        o.pair_class = wp.array(np.full(len(self.pairs), int(ContactClass.PRECISION_SDF), np.int8), dtype=wp.int8,
                                device=dev)
        o.pair_k = wp.array(self.pair_k, dtype=wp.float32, device=dev)
        o.shape_margin = lay.model.shape_margin
        return o

    def run(self, state: newton.State) -> None:
        """Append this step's rows after the rows already in ``layer.contacts()`` (device-only)."""
        dev, nbw = self.device, self.mask.shape[0]
        self.s.body_q = state.body_q
        self.group_n.zero_()
        wp.launch(_select, dim=nbw, inputs=[self.s, TIE_ABS, TIE_FRAC, self.cand, self.cand_d, self.mask,
                                            self.group_n], device=dev)
        wp.utils.array_scan(self.mask, self.slots, inclusive=False)
        wp.launch(_start, dim=1, inputs=[self.out], device=dev)
        wp.launch(_emit, dim=nbw, inputs=[self.s, self.cand, self.cand_d, self.mask, self.slots, self.group_n,
                                          self.out], device=dev)
        wp.launch(_bump, dim=1, inputs=[self.mask, self.slots, self.out], device=dev)


def attach_analytic_pairs(layer, pairs: Sequence[AnalyticPair], **kwargs) -> AnalyticPairContacts:
    """Register analytic pairs on ``layer`` (shapes filtered from native collision, not layer precision pairs)."""
    state = AnalyticPairContacts(layer, pairs, **kwargs)
    _CACHE[layer] = state
    return state


def analytic_collide(layer, state: newton.State, contacts, dt: float) -> None:
    """Append the analytic pairs' rows; call after ``layer.collide`` and before the solver step."""
    if contacts is not layer.contacts():
        raise AssemblyError("analytic_collide must be given the layer's own contacts() buffer")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    rows = _CACHE.get(layer)
    if rows is None:
        raise AssemblyError("no analytic pairs attached to this layer (attach_analytic_pairs)")
    rows.run(state)
