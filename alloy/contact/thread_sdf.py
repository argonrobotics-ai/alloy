# SPDX-License-Identifier: Apache-2.0
"""Analytic SDF of ISO metric threads (``analytic_id`` "iso_thread:M10x1.5:6g", SPEC §3.5, law A9).

Profile: the ISO 68-1 basic profile for both sides, a periodic polyline in the meridional half-plane: flat of width
p/8 at the major radius, 60° flanks (slope dr/du = √3), flat of width p/4 at the minor radius d1/2. External roots
are this flat d1 root, not the rounded ISO 965-1 root (d3 is reported for information only). The ISO 965-1
fundamental deviation shifts every diameter (es for e/f/g/h, EI for G/H); tolerance widths are ignored, so each
part sits at the deviation-shifted basic size. A bolt and a nut with the same frame and phase mate on the basic
profile.

Frame: axis +Z, right-hand thread, axial phase u = z - p·θ/2π; the centre of the major flat is at u = 0 (θ = 0,
z = 0). External threads are solid inside the profile, internal threads solid outside it up to ``outer_radius``;
both are cut to the slab z0 <= z <= z1.

Distance: the screw motion is an isometry, so the exact distance is a geodesic distance in the (r, u) half-plane
with metric dr² + s(r)²du², s(r) = r/√(r² + L²), L = p/2π. We take the Euclidean distance in (r, s_lo·u) with
s_lo = s(r_lo), r_lo = r_minor - reach. That field is 1-Lipschitz in 3D for r >= r_lo, exact on the flats, and is
clamped at ``reach`` on the axis side (where the phase is undefined), so the whole field is 1-Lipschitz, zero on
the surface and never larger in magnitude than the true distance. Error: 0 <= |true| - |sdf| <= |d|·(s(r_major +
|d|)/s_lo - 1) for |d| <= reach away from the slab ends (``sdf_error_bound``): 0.58 µm at 0.5 mm for M10x1.5.

Ends (optional, SCR-SC3-2): a slab end is a flat cut unless it has an ``EndChamfer``, a cone of revolution through
(r_face, end face) at ``angle_deg`` to the axis: an ISO 4753 chamfered end on an external thread (``iso4753_chamfer``,
removes the material outside the cone, so the last turn has no feather edge) or a countersink on an internal thread
(``nut_countersink``, removes the material inside it). Its term is the exact distance to that cone in the meridional
(r, z) half-plane, intersected (max) with the thread: still 1-Lipschitz, never larger than the true distance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy.contact.validate import sdf_bytes_estimate
from alloy.core.errors import PrecisionSpecError
from alloy.core.log import get_logger

log = get_logger("contact.thread")

SQRT3 = math.sqrt(3.0)
SEC30 = 2.0 / SQRT3
REACH_M = 5.0e-4                    # SPEC narrow band half-width: accurate |d| range, axis-side clamp
NARROW_BAND_M = 1.0e-3              # SPEC PrecisionPairSpec.narrow_band width
THREAD_ID_RE = re.compile(r"^iso_thread:M(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?):(\d)([A-Za-z])$")
# ISO 965-1 fundamental deviations: sign·(c + 11·P) µm, P in mm; h and H are 0
_DEVIATION_UM = {"e": (-1.0, 50.0), "f": (-1.0, 30.0), "g": (-1.0, 15.0), "h": (0.0, 0.0),
                 "G": (1.0, 15.0), "H": (0.0, 0.0)}
_GRADES = range(3, 10)              # ISO 965-1 tolerance grades


class ThreadSpecError(PrecisionSpecError, ValueError):
    """An ISO thread id or thread geometry is invalid (also a ``ValueError``)."""


@dataclass(frozen=True)
class EndChamfer:
    """A conical slab end [m, deg]: the cone meets the end face at radius ``r_face`` and makes ``angle_deg`` with the
    thread axis (45 = a 90° included chamfer)."""

    r_face: float
    angle_deg: float = 45.0

    def __post_init__(self) -> None:
        if not (self.r_face > 0.0 and 0.0 < self.angle_deg < 90.0):
            raise ThreadSpecError(f"end chamfer needs r_face > 0 and 0 < angle < 90 deg, got {self}")


@dataclass(frozen=True)
class ThreadSpec:
    """One ISO metric thread [m]. ``deviation`` is es (<= 0, external) or EI (>= 0, internal) on every diameter."""

    d: float
    pitch: float
    external: bool
    tol_class: str
    deviation: float

    def __post_init__(self) -> None:
        if not (self.d > 0.0 and self.pitch > 0.0 and self.r_minor > 0.0):
            raise ThreadSpecError(f"invalid thread M{self.d * 1e3:g}x{self.pitch * 1e3:g}: minor radius "
                                  f"{self.r_minor:.4g} m")

    @property
    def H(self) -> float:
        """Fundamental triangle height (√3/2)·p."""
        return 0.5 * SQRT3 * self.pitch

    @property
    def d2(self) -> float:
        """Basic pitch diameter d - 0.6495·p."""
        return self.d - 0.75 * self.H

    @property
    def d1(self) -> float:
        """Basic minor diameter d - 1.0825·p (the flat root of this profile)."""
        return self.d - 1.25 * self.H

    @property
    def d3(self) -> float:
        """ISO 965-1 rounded-root minor diameter of a bolt, d1 - H/6 (informative; not used by the profile)."""
        return self.d1 - self.H / 6.0

    @property
    def lead_2pi(self) -> float:
        """Axial advance per radian, p/2π (single start)."""
        return self.pitch / (2.0 * math.pi)

    @property
    def r_major(self) -> float:
        """Profile radius at the major flat, deviation applied."""
        return 0.5 * (self.d + self.deviation)

    @property
    def r_minor(self) -> float:
        """Profile radius at the minor flat, deviation applied."""
        return 0.5 * (self.d1 + self.deviation)


def fundamental_deviation(letter: str, pitch: float) -> float:
    """ISO 965-1 fundamental deviation [m] for position ``letter`` (e, f, g, h, G, H) and ``pitch`` [m]."""
    if letter not in _DEVIATION_UM:
        raise ThreadSpecError(f"tolerance position {letter!r} not supported (ISO 965-1: e f g h / G H)")
    sign, c_um = _DEVIATION_UM[letter]
    return sign * (c_um + 11.0 * pitch * 1e3) * 1e-6


def parse_thread_id(analytic_id: str) -> ThreadSpec:
    """Parse "iso_thread:M<d>x<p>:<grade><position>" (mm); lowercase position = external, uppercase = internal."""
    m = THREAD_ID_RE.match(analytic_id)
    if m is None:
        raise ThreadSpecError(f"expected 'iso_thread:M<d>x<pitch>:<class>' (e.g. M10x1.5:6g), got {analytic_id!r}")
    d, pitch = float(m.group(1)) * 1e-3, float(m.group(2)) * 1e-3
    grade, letter = int(m.group(3)), m.group(4)
    if grade not in _GRADES:
        raise ThreadSpecError(f"{analytic_id!r}: tolerance grade {grade} outside ISO 965-1 grades 3-9")
    spec = ThreadSpec(d, pitch, letter.islower(), f"{grade}{letter}", fundamental_deviation(letter, pitch))
    log.debug("thread %s: d2=%.6g d1=%.6g deviation=%.3g m", analytic_id, spec.d2, spec.d1, spec.deviation)
    return spec


def profile_radius(spec: ThreadSpec, u: np.ndarray) -> np.ndarray:
    """Profile radius [m] at axial phase ``u`` [m] (periodic in p; major flat centred on u = 0)."""
    p = spec.pitch
    w = np.abs(np.asarray(u, dtype=np.float64) - p * np.floor(np.asarray(u) / p + 0.5))
    frac = np.clip((w - p / 16.0) / (5.0 * p / 16.0), 0.0, 1.0)
    return spec.r_major + frac * (spec.r_minor - spec.r_major)


# ---------------------------------------------------------------- SDF parameters ------------------------------------

@wp.struct
class ThreadParams:
    """Warp twin of ``ThreadSDFParams`` (float32)."""

    pitch: float
    lead_2pi: float
    squeeze: float
    r_major: float
    r_minor: float
    crest_half: float
    flank_end: float
    side: float
    reach: float
    z0: float
    z1: float
    outer_radius: float
    ch0: float                      # 1: chamfered end at z0, 0: flat cut
    ch0_r: float
    ch0_cos: float
    ch0_sin: float
    ch1: float                      # 1: chamfered end at z1, 0: flat cut
    ch1_r: float
    ch1_cos: float
    ch1_sin: float


@dataclass(frozen=True)
class ThreadSDFParams:
    """Everything the SDF needs [m]; ``side`` +1 external / -1 internal, ``outer_radius`` inf for external."""

    pitch: float
    lead_2pi: float
    squeeze: float                  # s_lo = r_lo / √(r_lo² + L²)
    r_major: float
    r_minor: float
    crest_half: float               # p/16, half width of the major flat
    flank_end: float                # 3p/8, start of the minor flat
    side: float
    reach: float
    z0: float
    z1: float
    outer_radius: float
    ch0: float = 0.0                # end cones (EndChamfer at z0 / z1): on flag, face radius, cos and sin of the angle
    ch0_r: float = 0.0
    ch0_cos: float = 1.0
    ch0_sin: float = 0.0
    ch1: float = 0.0
    ch1_r: float = 0.0
    ch1_cos: float = 1.0
    ch1_sin: float = 0.0

    def to_warp(self) -> ThreadParams:
        """The ``ThreadParams`` struct for ``sd_thread``."""
        out = ThreadParams()
        for name in self.__dataclass_fields__:
            setattr(out, name, float(getattr(self, name)))
        return out


def _squeeze(r: float, lead_2pi: float) -> float:
    return r / math.hypot(r, lead_2pi)


def _check_extent(spec: ThreadSpec, z0: float, z1: float, outer_radius: float | None) -> None:
    if not z1 > z0:
        raise ThreadSpecError(f"thread slab needs z1 > z0, got [{z0}, {z1}]")
    if spec.external and outer_radius is not None:
        raise ThreadSpecError("outer_radius applies to internal threads only")
    if not spec.external and (outer_radius is None or outer_radius <= spec.r_major):
        raise ThreadSpecError(f"internal thread needs outer_radius > major radius {spec.r_major:.6g} m, "
                              f"got {outer_radius}")


def iso4753_chamfer(spec: ThreadSpec) -> EndChamfer:
    """Chamfered end (ISO 4753 "CH") of an external thread: a cone reaching the end face at the rounded-root minor
    diameter d3 (below the flat root of this profile), so no partial turn ends in a feather edge. The 45° angle and
    the face diameter are the usual shop values, NOT checked against the (paywalled) standard: pass your own
    ``EndChamfer`` where it matters."""
    if not spec.external:
        raise ThreadSpecError("iso4753_chamfer is for external threads; use nut_countersink")
    return EndChamfer(0.5 * (spec.d3 + spec.deviation), 45.0)


def nut_countersink(spec: ThreadSpec, da_over_d: float = 1.04, included_deg: float = 120.0) -> EndChamfer:
    """Countersink on a nut face: diameter d_a at the face and ``included_deg`` included angle. d_a: DIN 934 M10 table
    d_a min 10.00 / max 10.80 mm, i.e. d <= d_a <= 1.08 d (https://www.fasteners.eu/standards/din/934/); default the
    middle. The 120° angle is the usual shop value (90-120°), NOT checked against ISO 4032."""
    if spec.external:
        raise ThreadSpecError("nut_countersink is for internal threads; use iso4753_chamfer")
    return EndChamfer(0.5 * da_over_d * spec.d, 0.5 * included_deg)


def _chamfer_fields(ch: EndChamfer | None) -> tuple[float, float, float, float]:
    if ch is None:
        return 0.0, 0.0, 1.0, 0.0
    a = math.radians(ch.angle_deg)
    return 1.0, float(ch.r_face), math.cos(a), math.sin(a)


def thread_sdf_params(spec: ThreadSpec, *, z0: float, z1: float, outer_radius: float | None = None,
                      reach: float = REACH_M, chamfer0: EndChamfer | None = None,
                      chamfer1: EndChamfer | None = None) -> ThreadSDFParams:
    """SDF parameters for the thread cut to [z0, z1] (internal: solid up to ``outer_radius``), with optional conical
    ends ``chamfer0`` at z0 and ``chamfer1`` at z1 (flat cuts when None)."""
    _check_extent(spec, z0, z1, outer_radius)
    r_lo = spec.r_minor - reach
    if not (0.0 < reach and r_lo > 0.0):
        raise ThreadSpecError(f"reach {reach} m must be in (0, minor radius {spec.r_minor:.6g} m)")
    p, lead = spec.pitch, spec.lead_2pi
    prm = ThreadSDFParams(p, lead, _squeeze(r_lo, lead), spec.r_major, spec.r_minor, p / 16.0, 3.0 * p / 8.0,
                          1.0 if spec.external else -1.0, reach, z0, z1,
                          math.inf if outer_radius is None else float(outer_radius),
                          *_chamfer_fields(chamfer0), *_chamfer_fields(chamfer1))
    log.debug("thread sdf params: %s", prm)
    return prm


def sdf_error_bound(prm: ThreadSDFParams, distance: float) -> float:
    """Upper bound on |true| - |sdf| [m] at true distance ``distance`` <= reach (away from the slab ends)."""
    d = abs(distance)
    return d * (_squeeze(prm.r_major + d, prm.lead_2pi) / prm.squeeze - 1.0)


# ---------------------------------------------------------------- numpy reference -----------------------------------

def _profile_sd_np(prm: ThreadSDFParams, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Signed distance in the squeezed (r, s·u) plane to the folded half profile, positive at r > R(u)."""
    s = prm.squeeze
    x, a, b = s * w, s * prm.crest_half, s * prm.flank_end
    ex, er = b - a, prm.r_minor - prm.r_major
    d_major = np.hypot(np.maximum(x - a, 0.0), r - prm.r_major)
    d_minor = np.hypot(np.maximum(b - x, 0.0), r - prm.r_minor)
    t = np.clip(((x - a) * ex + (r - prm.r_major) * er) / (ex * ex + er * er), 0.0, 1.0)
    d_flank = np.hypot(x - a - t * ex, r - prm.r_major - t * er)
    dist = np.minimum(np.minimum(d_major, d_minor), d_flank)
    radius = prm.r_major + np.clip((x - a) / ex, 0.0, 1.0) * er
    return np.where(r > radius, dist, -dist)


def thread_sdf_np(prm: ThreadSDFParams, points: np.ndarray) -> np.ndarray:
    """Signed distance [m] of ``points`` [N, 3] (thread frame) to the threaded solid; float64."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.hypot(x, y)
    u = z - prm.lead_2pi * np.arctan2(y, x)
    w = np.abs(u - prm.pitch * np.floor(u / prm.pitch + 0.5))
    d = prm.side * np.maximum(_profile_sd_np(prm, r, w), -prm.reach)
    d = np.maximum(d, r - prm.outer_radius)
    d = np.maximum(d, np.maximum(prm.z0 - z, z - prm.z1))
    if prm.ch0 > 0.5:
        d = np.maximum(d, prm.side * (r - prm.ch0_r) * prm.ch0_cos + (prm.z0 - z) * prm.ch0_sin)
    if prm.ch1 > 0.5:
        d = np.maximum(d, prm.side * (r - prm.ch1_r) * prm.ch1_cos + (z - prm.z1) * prm.ch1_sin)
    return d


# ---------------------------------------------------------------- Warp ----------------------------------------------

@wp.func
def sd_thread_profile(prm: ThreadParams, r: float, w: float) -> float:
    """Warp twin of ``_profile_sd_np``."""
    s = prm.squeeze
    x = s * w
    a = s * prm.crest_half
    b = s * prm.flank_end
    ex = b - a
    er = prm.r_minor - prm.r_major
    dr = r - prm.r_major
    d_major = wp.sqrt(wp.max(x - a, 0.0) * wp.max(x - a, 0.0) + dr * dr)
    d_minor = wp.sqrt(wp.max(b - x, 0.0) * wp.max(b - x, 0.0) + (r - prm.r_minor) * (r - prm.r_minor))
    t = wp.clamp(((x - a) * ex + dr * er) / (ex * ex + er * er), 0.0, 1.0)
    fx = x - a - t * ex
    fr = dr - t * er
    dist = wp.min(wp.min(d_major, d_minor), wp.sqrt(fx * fx + fr * fr))
    radius = prm.r_major + wp.clamp((x - a) / ex, 0.0, 1.0) * er
    if r > radius:
        return dist
    return -dist


@wp.func
def sd_thread(prm: ThreadParams, p: wp.vec3) -> float:
    """Signed distance [m] of ``p`` (thread frame) to the threaded solid; Warp twin of ``thread_sdf_np``."""
    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    u = p[2] - prm.lead_2pi * wp.atan2(p[1], p[0])
    w = wp.abs(u - prm.pitch * wp.floor(u / prm.pitch + 0.5))
    d = prm.side * wp.max(sd_thread_profile(prm, r, w), -prm.reach)
    d = wp.max(d, r - prm.outer_radius)
    d = wp.max(d, wp.max(prm.z0 - p[2], p[2] - prm.z1))
    if prm.ch0 > 0.5:
        d = wp.max(d, prm.side * (r - prm.ch0_r) * prm.ch0_cos + (prm.z0 - p[2]) * prm.ch0_sin)
    if prm.ch1 > 0.5:
        d = wp.max(d, prm.side * (r - prm.ch1_r) * prm.ch1_cos + (p[2] - prm.z1) * prm.ch1_sin)
    return d


# ---------------------------------------------------------------- mesh ----------------------------------------------

def _z_levels(z0: float, z1: float, du: float) -> np.ndarray:
    """z0, every multiple of ``du`` strictly inside (z0, z1) (profile corners land on these rows), z1."""
    eps = 1e-6 * du
    inner = du * np.arange(math.ceil((z0 + eps) / du), math.floor((z1 - eps) / du) + 1)
    inner = inner[(inner > z0 + eps) & (inner < z1 - eps)]
    return np.concatenate([[z0], inner, [z1]])


def _strip(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quads (a_i, a_i+1, b_i+1, b_i) of two closed rings (rows of index arrays) as triangles."""
    a1, b1 = np.roll(a, -1, axis=-1), np.roll(b, -1, axis=-1)
    return np.concatenate([np.stack([a, a1, b1], -1).reshape(-1, 3), np.stack([a, b1, b], -1).reshape(-1, 3)])


def _fan(center: int, ring: np.ndarray) -> np.ndarray:
    return np.stack([np.full_like(ring, center), ring, np.roll(ring, -1)], -1)


def _ring(radius: float, z: float, n: int) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.full(n, z)], -1)


def _chamfer_radius(spec: ThreadSpec, radius: np.ndarray, z: np.ndarray, ends: tuple) -> np.ndarray:
    """Radii pulled onto the end cones (external: no larger than the cone, internal: no smaller)."""
    for ch, z_face, sign in ends:
        if ch is None:
            continue
        grow = sign * (z_face - z) * math.tan(math.radians(ch.angle_deg))   # >= 0 inside the slab
        radius = np.minimum(radius, ch.r_face + grow) if spec.external else np.maximum(radius, ch.r_face - grow)
    return radius


def _threaded_grid(spec: ThreadSpec, z: np.ndarray, n: int, ends: tuple = ()) -> np.ndarray:
    """Vertices [len(z)·n, 3] on the thread surface at θ_i = 2πi/n, row-major in z (radii clipped by end cones)."""
    theta = 2.0 * np.pi * np.arange(n) / n
    radius = profile_radius(spec, z[:, None] - spec.lead_2pi * theta[None, :])
    radius = _chamfer_radius(spec, radius, np.broadcast_to(z[:, None], radius.shape), ends)
    xyz = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.broadcast_to(z[:, None], radius.shape)], -1)
    return xyz.reshape(-1, 3)


def _flip(tris: np.ndarray) -> np.ndarray:
    return tris[:, [0, 2, 1]]


def thread_mesh(spec: ThreadSpec, length: float, segments_per_turn: int = 96, *, z0: float = 0.0,
                outer_radius: float | None = None, chamfer0: EndChamfer | None = None,
                chamfer1: EndChamfer | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Closed, outward-wound mesh (float64 verts [m], int32 tris) of the solid ``thread_sdf_np`` describes on
    [z0, z0 + length]. Rows are p/segments_per_turn apart along helices, so profile corners are mesh edges;
    ``segments_per_turn`` must be a multiple of 16. Crest chord error is r_major·(1 - cos(π/segments_per_turn)).
    End cones move vertices radially onto the cone (vertex-exact; facets between rows cut the cone by the chord)."""
    n = int(segments_per_turn)
    if n < 16 or n % 16:
        raise ThreadSpecError(f"segments_per_turn must be a positive multiple of 16, got {segments_per_turn}")
    _check_extent(spec, z0, z0 + length, outer_radius)
    z = _z_levels(z0, z0 + length, spec.pitch / n)
    side = _threaded_grid(spec, z, n, ((chamfer0, z0, -1.0), (chamfer1, z0 + length, 1.0)))
    rows = np.arange(len(side)).reshape(len(z), n)
    wall = _strip(rows[:-1], rows[1:])
    if spec.external:
        c0, c1 = len(side), len(side) + 1
        verts = np.concatenate([side, [[0.0, 0.0, z[0]], [0.0, 0.0, z[-1]]]])
        tris = [wall, _flip(_fan(c0, rows[0])), _fan(c1, rows[-1])]
    else:
        o0 = len(side) + np.arange(n)
        o1 = o0 + n
        verts = np.concatenate([side, _ring(outer_radius, z[0], n), _ring(outer_radius, z[-1], n)])
        tris = [_flip(wall), _strip(o0, o1), _strip(o1, rows[-1]), _strip(rows[0], o0)]
    tris = np.concatenate(tris).astype(np.int32)
    log.debug("thread mesh: %d verts, %d tris, %d rows", len(verts), len(tris), len(z))
    return verts, tris


# ---------------------------------------------------------------- sizes and law A9 ----------------------------------

def mean_profile_r2(spec: ThreadSpec) -> float:
    """Mean of R(u)² over one pitch [m²]: solid cross-section area = π·mean (external)."""
    ra, rb = spec.r_major, spec.r_minor
    return ra * ra / 8.0 + rb * rb / 4.0 + 5.0 / 8.0 * (ra * ra + ra * rb + rb * rb) / 3.0


def thread_surface_area(spec: ThreadSpec, length: float) -> float:
    """Area [m²] of the helical thread surface over axial ``length``: 2π·(length/p)·∫_0^p √(R²(1+R'²) + L²R'²) du."""
    lead2 = 3.0 * spec.lead_2pi ** 2

    def flank(r: float) -> float:    # ∫ √(4R² + 3L²) dR
        root = math.sqrt(4.0 * r * r + lead2)
        return 0.5 * r * root + 0.25 * lead2 * math.log(2.0 * r + root)

    p = spec.pitch
    per_pitch = p / 8.0 * spec.r_major + p / 4.0 * spec.r_minor
    per_pitch += 2.0 / SQRT3 * (flank(spec.r_major) - flank(spec.r_minor))
    return 2.0 * math.pi * length / p * per_pitch


def texture_sdf_bytes(spec: ThreadSpec, length: float, voxel: float, band: float = NARROW_BAND_M) -> float:
    """Sparse texture SDF bytes for the thread surface alone (a lower bound for the part), contact §4.7."""
    return sdf_bytes_estimate(thread_surface_area(spec, length), band, voxel)


def advance_per_rev(spec: ThreadSpec) -> float:
    """Axial advance per revolution [m] (A9: = p, single start)."""
    return spec.pitch


def rundown_torque(spec: ThreadSpec, axial_force: float, mu: float) -> float:
    """A9 run-down torque [N·m]: F·(d2/2)·(p + π·μ·d2·sec30°)/(π·d2 - μ·p·sec30°), basic d2."""
    d2, p = spec.d2, spec.pitch
    return axial_force * 0.5 * d2 * (p + math.pi * mu * d2 * SEC30) / (math.pi * d2 - mu * p * SEC30)
