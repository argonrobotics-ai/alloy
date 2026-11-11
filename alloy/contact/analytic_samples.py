# SPDX-License-Identifier: Apache-2.0
"""Host side of the analytic precision pairs (``analytic_pair``): the features a pair is made of, and the surface
samples and bins of its sampled side. Split from ``analytic_pair`` (500-line limit) when chamfered thread ends were
added (SCR-SC3-2); ``analytic_pair`` re-exports every public name here.

Thread samples cover the centred ``flank_band`` of each flank and start ``END_TRIM`` pitches inside a flat-cut slab
end (see ``analytic_pair``). At a slab end with an ``EndChamfer`` they reach the end instead: only samples the
feature's own cone leaves in material are kept, and within ``END_BAND`` pitches of that end ``END_FLANK_BAND`` of each
flank is sampled, crests included, so a lead-in turn has rows where the crests meet the partner's chamfer.

``crest=True`` also samples the centred ``CREST_BAND`` of each crest flat (internal: the minor flat, external: the major
flat) into its own bin per (tooth, sector). At a chamfered end it also samples, into the same bins, the cone's face
where it cuts the thread (edge included) and, within ``END_BAND`` pitches, the two crest/flank corner helices exactly:
the lead-in turn's convex edges, which a grid misses by up to half a step. Why crests: the profile's root flat is only
|deviation| / 2 radially from the partner's crest (M10 6g/6H: 16 um, the same radial play as the flanks), and on the
lead-in turn the end cones trim the flanks, so an off-centre nut (C5, 20-50 um at the catch) put its unsampled crest
20 um into the bolt core. Off by default (no change to existing pairs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.contact.thread_sdf import EndChamfer, parse_thread_id, profile_radius
from alloy.core.errors import PrecisionSpecError

SECTORS = 12
END_TRIM = 0.5                      # thread samples start this many pitches inside each flat-cut slab end
END_BAND = 1.0                      # pitches from a chamfered end sampled with END_FLANK_BAND
END_FLANK_BAND = 0.95               # crests included, the exact profile corners not
CREST_BAND = 0.8                    # crest=True: centred fraction of each crest flat sampled (its corners not)


@dataclass(frozen=True)
class ThreadFeature:
    """An ISO thread on a shape: axis = shape +z, phase as ``thread_sdf``, cut to [z0, z1] (internal: solid to
    ``outer_radius``)."""

    analytic_id: str
    z0: float
    z1: float
    outer_radius: float | None = None
    flank_band: float = 0.8         # sampled side: centred fraction of each flank's height that gets samples
    chamfer0: EndChamfer | None = None   # conical end at z0 / z1 (``iso4753_chamfer``, ``nut_countersink``); None: flat
    chamfer1: EndChamfer | None = None
    crest: bool = False             # sampled side: rows on the crest flats too (radial contact with the partner's root)


@dataclass(frozen=True)
class FaceFeature:
    """A flat annular face z = ``z`` of a shape, r in [r_in, r_out] (sampled side only, e.g. a nut bearing face)."""

    z: float
    r_in: float
    r_out: float


@dataclass(frozen=True)
class SlabFeature:
    """A plate z0 <= z <= z1 of unbounded radius (partner side only, e.g. a washer)."""

    z0: float
    z1: float


@dataclass(frozen=True)
class SampleBins:
    """Surface samples [N, 3] (shape frame) grouped by bin: bin b owns ``points[start[b]:start[b+1]]``."""

    points: np.ndarray
    start: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.start) - 1


def torque_radius(r_in: float, r_out: float) -> float:
    """Radius at which a uniform pressure on the annulus [r_in, r_out] has its friction torque."""
    return 2.0 / 3.0 * (r_out**3 - r_in**3) / (r_out**2 - r_in**2)


def _sector(theta: np.ndarray, sectors: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.minimum((theta * sectors / (2.0 * math.pi)).astype(np.int64), sectors - 1)
    return k, (k + 0.5) * 2.0 * math.pi / sectors


def _bins(points: np.ndarray, bin_id: np.ndarray, key: np.ndarray, n_bins: int) -> SampleBins:
    order = np.lexsort((np.arange(len(key)), key, bin_id))
    start = np.searchsorted(bin_id[order], np.arange(n_bins + 1)).astype(np.int32)
    return SampleBins(points[order].astype(np.float32), start)


def thread_samples(feat: ThreadFeature, spacing: float, sectors: int = SECTORS) -> SampleBins:
    """Samples of the flanks of the helical thread surface on [z0, z1]; bins (tooth, sector, flank)."""
    spec = parse_thread_id(feat.analytic_id)
    p, lead = spec.pitch, spec.lead_2pi
    n_theta = sectors * math.ceil(2.0 * math.pi * spec.r_major / spacing / sectors)
    theta = 2.0 * math.pi * np.arange(n_theta) / n_theta
    z = np.linspace(feat.z0, feat.z1, math.ceil((feat.z1 - feat.z0) / (0.5 * spacing)) + 1)
    th, zz = np.meshgrid(theta, z)
    u = zz - lead * th
    r = profile_radius(spec, u)
    period = np.floor(u / p)
    w = u - period * p
    half = (w >= 0.5 * p).astype(np.int64)
    first = math.floor((feat.z0 - p) / p)
    teeth = math.floor(feat.z1 / p) - first + 1
    tooth = np.clip(period.astype(np.int64) - first, 0, teeth - 1)
    sector, theta_c = _sector(th, sectors)
    edge = 0.5 * (1.0 - feat.flank_band) * 5.0 * p / 16.0               # in u; the flank spans 5p/16
    near_end = np.zeros_like(zz, dtype=bool)
    if feat.chamfer0 is not None:
        near_end |= zz < feat.z0 + END_BAND * p
    if feat.chamfer1 is not None:
        near_end |= zz > feat.z1 - END_BAND * p
    edge_s = np.where(near_end, 0.5 * (1.0 - END_FLANK_BAND) * 5.0 * p / 16.0, edge)
    r_lo, r_hi = spec.r_minor + math.sqrt(3.0) * edge, spec.r_major - math.sqrt(3.0) * edge
    r_t = torque_radius(r_lo, r_hi)
    w_c = np.where(half == 0, p / 16.0 + (spec.r_major - r_t) / math.sqrt(3.0),
                   5.0 * p / 8.0 + (r_t - spec.r_minor) / math.sqrt(3.0))
    keep = (np.abs(w - np.where(half == 0, 7.0 * p / 32.0, 25.0 * p / 32.0)) <= 5.0 * p / 32.0 - edge_s)
    key = (r_t * (th - theta_c)) ** 2 + (2.0 * (w - w_c)) ** 2
    kind, kinds = half, 2
    if feat.crest:
        on, c_tooth, c_key = _crest(spec, u, th - theta_c, first, teeth)
        keep |= on
        kind, kinds = np.where(on, 2, half), 3
        tooth, key = np.where(on, c_tooth, tooth), np.where(on, c_key, key)
    keep &= zz >= feat.z0 + (0.0 if feat.chamfer0 is not None else END_TRIM * p)
    keep &= zz <= feat.z1 - (0.0 if feat.chamfer1 is not None else END_TRIM * p)
    keep &= _inside_cones(spec, feat, r, zz)
    pts = np.stack([r * np.cos(th), r * np.sin(th), zz], -1)[keep]
    bin_id = ((tooth * sectors + sector) * kinds + kind)[keep]
    key = key[keep]
    if feat.crest:
        cp, c_tooth, c_sector, c_key = _merge(_cone_faces(spec, feat, theta, spacing, sectors),
                                              _crest_corners(spec, feat, spacing, sectors))
        c_tooth = np.clip(c_tooth - first, 0, teeth - 1)
        pts, key = np.concatenate([pts, cp]), np.concatenate([key, c_key])
        bin_id = np.concatenate([bin_id, (c_tooth * sectors + c_sector) * kinds + 2])
    return _bins(pts, bin_id, key, teeth * sectors * kinds)


def _crest(spec, u: np.ndarray, dth: np.ndarray, first: int, teeth: int):
    """Crest-flat samples (internal: minor flat centred on u = p/2, external: major flat on u = 0): mask, tooth, key."""
    p = spec.pitch
    centre, half_w = (0.0, p / 16.0) if spec.external else (0.5 * p, p / 8.0)
    k = np.floor((u - centre) / p + 0.5)
    du = u - centre - k * p
    on = np.abs(du) <= CREST_BAND * half_w
    tooth = np.clip(k.astype(np.int64) - first, 0, teeth - 1)
    r_c = spec.r_major if spec.external else spec.r_minor
    return on, tooth, (r_c * dth) ** 2 + (2.0 * du) ** 2


def _crest_corners(spec, feat: ThreadFeature, spacing: float, sectors: int):
    """crest=True within END_BAND pitches of a chamfered end: samples exactly on the two helical crest/flank corners,
    at a quarter of the spacing (the convex edges a bolt chamfer meets on the lead-in turn; between grid rows a corner
    went 8 um into the chamfer, C5 c5cx4). Points, absolute tooth, sector, key."""
    p, lead = spec.pitch, spec.lead_2pi
    centre, half_w = (0.0, p / 16.0) if spec.external else (0.5 * p, p / 8.0)
    r_c = spec.r_major if spec.external else spec.r_minor
    n = sectors * math.ceil(2.0 * math.pi * r_c / (0.25 * spacing) / sectors)
    th = 2.0 * math.pi * np.arange(n) / n
    bands = [(feat.z0, feat.z0 + END_BAND * p)] if feat.chamfer0 is not None else []
    bands += [(feat.z1 - END_BAND * p, feat.z1)] if feat.chamfer1 is not None else []
    zs, ths = [], []
    for lo, hi in bands:
        for du in (-half_w, half_w):
            base = centre + du + lead * th
            for k in range(math.floor((lo - base.max()) / p), math.ceil((hi - base.min()) / p) + 1):
                z = base + k * p
                keep = (z >= max(lo, feat.z0)) & (z <= min(hi, feat.z1))
                zs.append(z[keep])
                ths.append(th[keep])
    if not zs:
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    z, th = np.concatenate(zs), np.concatenate(ths)
    keep = _inside_cones(spec, feat, np.full_like(z, r_c), z)
    z, th = z[keep], th[keep]
    u = z - lead * th
    k_t = np.floor((u - centre) / p + 0.5)
    sector, th_c = _sector(th, sectors)
    pts = np.stack([r_c * np.cos(th), r_c * np.sin(th), z], -1)
    return pts, k_t.astype(np.int64), sector, (r_c * (th - th_c)) ** 2 + (2.0 * half_w) ** 2


def _merge(*parts):
    """Concatenate (points, tooth, sector, key) sample sets."""
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def _cone_faces(spec, feat: ThreadFeature, theta: np.ndarray, spacing: float, sectors: int):
    """crest=True at a chamfered end: samples on the end cone's face where the thread leaves material, from its edge
    with the crest flat out to the other profile radius (a nut's countersink meets a bolt's chamfer there on the
    lead-in turn; the flat grid misses that edge by up to half a z step). Points, absolute tooth, sector, key."""
    side = 1.0 if spec.external else -1.0
    pts, tooth, sector, key = [], [], [], []
    for ch, z_face, sign in ((feat.chamfer0, feat.z0, -1.0), (feat.chamfer1, feat.z1, 1.0)):
        if ch is None:
            continue
        a = math.radians(ch.angle_deg)
        if spec.external:
            lo, hi = max(spec.r_minor, ch.r_face), spec.r_major
        else:
            lo, hi = spec.r_minor, min(spec.r_major, ch.r_face)
        if hi <= lo:
            continue
        rr, th = np.meshgrid(np.linspace(lo, hi, math.ceil((hi - lo) / (0.5 * spacing * math.sin(a))) + 1), theta)
        zz = z_face - side * sign * (rr - ch.r_face) / math.tan(a)
        u = zz - spec.lead_2pi * th
        keep = side * (rr - profile_radius(spec, u)) <= 1e-10          # thread material behind the cone face
        keep &= (zz >= feat.z0) & (zz <= feat.z1)
        k, th_c = _sector(th, sectors)
        pts.append(np.stack([rr * np.cos(th), rr * np.sin(th), zz], -1)[keep])
        tooth.append(np.floor(u / spec.pitch).astype(np.int64)[keep])
        sector.append(k[keep])
        key.append(((rr * (th - th_c)) ** 2 + (rr - lo) ** 2)[keep])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(pts), np.concatenate(tooth), np.concatenate(sector), np.concatenate(key)


def _inside_cones(spec, feat: ThreadFeature, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Samples the feature's own end cones leave in material (the SDF's cone terms <= 0)."""
    side = 1.0 if spec.external else -1.0
    ok = np.ones_like(z, dtype=bool)
    for ch, z_face, sign in ((feat.chamfer0, feat.z0, -1.0), (feat.chamfer1, feat.z1, 1.0)):
        if ch is not None:
            a = math.radians(ch.angle_deg)
            ok &= side * (r - ch.r_face) * math.cos(a) + sign * (z - z_face) * math.sin(a) <= 0.0
    return ok


def face_samples(feat: FaceFeature, spacing: float, sectors: int = SECTORS) -> SampleBins:
    """Samples of the annular face on a polar grid; one bin per angular sector."""
    n_theta = sectors * math.ceil(2.0 * math.pi * feat.r_out / spacing / sectors)
    theta = 2.0 * math.pi * np.arange(n_theta) / n_theta
    radii = np.linspace(feat.r_in, feat.r_out, math.ceil((feat.r_out - feat.r_in) / spacing) + 1)
    th, rr = np.meshgrid(theta, radii)
    sector, theta_c = _sector(th, sectors)
    r_t = torque_radius(feat.r_in, feat.r_out)
    key = (r_t * (th - theta_c)) ** 2 + (rr - r_t) ** 2
    pts = np.stack([rr * np.cos(th), rr * np.sin(th), np.full_like(rr, feat.z)], -1).reshape(-1, 3)
    return _bins(pts, sector.ravel(), key.ravel(), sectors)


def sample_bins(feat: ThreadFeature | FaceFeature, spacing: float, sectors: int = SECTORS) -> SampleBins:
    if isinstance(feat, ThreadFeature):
        return thread_samples(feat, spacing, sectors)
    if isinstance(feat, FaceFeature):
        return face_samples(feat, spacing, sectors)
    raise PrecisionSpecError(f"sampled side must be a ThreadFeature or FaceFeature, got {type(feat).__name__}")


__all__ = ["END_BAND", "END_FLANK_BAND", "END_TRIM", "SECTORS", "FaceFeature", "SampleBins", "SlabFeature",
           "ThreadFeature", "face_samples", "sample_bins", "thread_samples", "torque_radius"]
