# SPDX-License-Identifier: Apache-2.0
"""R1 rig (peg insertion, 0.2 mm diametral clearance; bench §4.4 R1): geometry, holder, protocol, metrics, mesh.

Shared by the twin (``alloy.contact.rig_twins``) and the synthetic dataset (``alloy.contact.rig_synth``) so that the
simulated and the measured series are reduced by the same code.

Conventions: the peg leans toward +x (tilt = rotation about +y) and its tip is offset by ``offset`` along +x, in the
tilt plane (the worst case of the protocol's tilt × offset grid), both measured on the peg hanging in the holder
before the stroke; depth is the depth of the peg-tip centre below the hole's top face [m]; insertion force is the
tared axial holder force, compression positive [N].

Metric definitions (per trial or per world; the protocol's reduction of a force-depth series):
- profile: force binned on a 20 µm depth grid and smoothed by a 0.1 mm moving average;
- jam: the stage stopped on the force limit (max force >= 0.95 ``f_stop``) or the tip never got within 0.5 mm of
  the stroke end;
- sliding_force: mean profile force over depths [15, 18] mm (undefined for a jam);
- peak_force / peak_depth: the two-point peak, i.e. the profile maximum before the sliding window and the
  force-weighted centroid (weights F - 0.9 F_max) of the contiguous depths where the profile is within 10 % of it.
  Defined only when it stands >= 0.5 N above the sliding force. The centroid replaces the argmax: a Whitney two-point
  peak is broad (F ~ a/h - b/h²), so moving its argmax by 0.1 mm changes the force by ~1 mN, far below the cell
  noise. On the synthetic 2° trials the weighted centroid scatters 9 µm under 0.05 N cell noise (the plain 5 % band
  centroid 37 µm), which is what lets n = 5 valid trials judge ± 0.1 mm (TUR >= 3).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

import numpy as np

from alloy.bench.types import Condition
from alloy.testing.contact import PegHoleSpec, blind_hole_tris

LAW_ID = "R1.peg_insertion"
METRICS: tuple[str, ...] = ("sliding_force", "peak_force", "peak_depth", "jam")
UNITS: Mapping[str, str] = {"sliding_force": "N", "peak_force": "N", "peak_depth": "m", "jam": "1"}
DEFAULT_MU = 0.35                     # steel on reamed Al with a light oil film (card `objects.pair.mu`)


@dataclass(frozen=True)
class R1Geometry:
    """Ø9.80 steel peg into a Ø10.00 reamed Al blind hole, 20 mm deep (card `objects`)."""

    peg_d: float = 9.80e-3
    bore_d: float = 10.00e-3
    peg_len: float = 40.0e-3
    bore_depth: float = 20.0e-3
    outer_d: float = 30.0e-3
    height: float = 25.0e-3
    chamfer: float = 0.5e-3           # 45° hole chamfer width of the chamfered conditions
    density: float = 7800.0
    segments: int = 96                # mesh tessellation (chord error 2.7 µm at r = 5 mm), not a card value

    @property
    def clearance(self) -> float:
        """Diametral clearance D - d [m]."""
        return self.bore_d - self.peg_d

    @property
    def peg_mass(self) -> float:
        return self.density * math.pi * (0.5 * self.peg_d) ** 2 * self.peg_len

    def hole_spec(self) -> PegHoleSpec:
        return PegHoleSpec(peg_d=self.peg_d, peg_len=self.peg_len, bore_d=self.bore_d, outer_d=self.outer_d,
                           height=self.height, depth=self.bore_depth, segments=self.segments)

    def objects(self, mu: float) -> dict[str, dict[str, float]]:
        """Card `objects` with measured dimensions and u95 (SI)."""
        return {"peg": {"diameter_m": self.peg_d, "diameter_m_u95": 2e-6, "length_m": self.peg_len,
                        "density_kg_m3": self.density, "mass_kg": self.peg_mass, "mass_kg_u95": 1e-4},
                "hole": {"diameter_m": self.bore_d, "diameter_m_u95": 3e-6, "depth_m": self.bore_depth,
                         "outer_diameter_m": self.outer_d, "height_m": self.height, "chamfer_m": self.chamfer},
                "pair": {"mu": mu, "mu_u95": 0.02}}

    @classmethod
    def from_objects(cls, objects: Mapping[str, Mapping[str, float]]) -> R1Geometry:
        peg, hole = objects["peg"], objects["hole"]
        return cls(peg_d=float(peg["diameter_m"]), bore_d=float(hole["diameter_m"]), peg_len=float(peg["length_m"]),
                   bore_depth=float(hole["depth_m"]), outer_d=float(hole["outer_diameter_m"]),
                   height=float(hole["height_m"]), chamfer=float(hole["chamfer_m"]),
                   density=float(peg["density_kg_m3"]))


@dataclass(frozen=True)
class R1Holder:
    """The measured compliant holder (card `holder`): springs at a compliance centre ``l_g`` above the peg tip."""

    k_x: float = 2000.0               # lateral [N/m]
    k_theta: float = 1.0              # angular [N·m/rad]
    l_g: float = 50.0e-3              # compliance centre above the tip [m] (10 mm above a 40 mm peg's top)
    k_z: float = 1.0e5                # axial, holder and 20 kg cell in series [N/m]
    tilt_sigma: float = math.radians(0.003)   # stage setting repeatability, 1 sigma [rad]
    offset_sigma: float = 2.0e-6              # [m]

    def card(self, proto: R1Protocol) -> dict[str, float]:
        return {**asdict(self), "stage_speed": proto.speed, "f_stop": proto.f_stop, "z_end": proto.z_end,
                "approach": proto.approach}

    @classmethod
    def from_card(cls, holder: Mapping[str, float]) -> R1Holder:
        return cls(**{f.name: float(holder[f.name]) for f in fields(cls) if f.name in holder})


@dataclass(frozen=True)
class R1Protocol:
    """Stage profile and series reduction (fixed by the protocol version, not measured)."""

    speed: float = 1.0e-3             # stage [m/s]
    quick_speed: float = 5.0e-3       # twin quick mode (CT-3 allows 1-5 mm/s; the metrics are quasi-static)
    approach: float = 0.5e-3          # tip starts this far above the top face
    z_end: float = 19.0e-3            # stroke end, tip depth (bore floor at 20 mm)
    f_stop: float = 30.0              # stage stops on this force (jam)
    t_settle: float = 0.05
    t_dwell: float = 0.05
    bin_m: float = 20.0e-6
    smooth_m: float = 0.1e-3
    window: tuple[float, float] = (15.0e-3, 18.0e-3)
    peak_min_n: float = 0.5
    peak_band: float = 0.10
    jam_margin: float = 0.5e-3
    jam_force_frac: float = 0.95

    @property
    def stroke(self) -> float:
        return self.approach + self.z_end

    def duration(self, speed: float) -> float:
        """Run length [s] at stage ``speed``: settle, approach + stroke, dwell."""
        return self.t_settle + self.stroke / speed + self.t_dwell


@dataclass(frozen=True)
class InsertionMetrics:
    sliding_force: float
    peak_force: float
    peak_depth: float
    jam: float
    jam_depth: float

    def scalars(self) -> dict[str, float]:
        """Trial scalars: the defined (finite) R1 metrics only."""
        vals = {m: float(getattr(self, m)) for m in METRICS}
        return {k: v for k, v in vals.items() if math.isfinite(v)}


def r1_conditions(quick: bool = False, geometry: R1Geometry = R1Geometry()) -> tuple[Condition, ...]:
    """Tilt {0, 0.5, 1, 2}° × offset {0, 0.05, 0.1} mm × chamfer {on, off}; quick: {1, 2}° × 0.05 mm, chamfered."""
    tilts, offsets, chamfers = ((1.0, 2.0), (0.05,), (True,)) if quick else \
        ((0.0, 0.5, 1.0, 2.0), (0.0, 0.05, 0.1), (True, False))
    out = []
    for ch in chamfers:
        for deg in tilts:
            for mm in offsets:
                cid = f"t{deg:g}_o{mm:g}_{'ch' if ch else 'sh'}"
                out.append(Condition(cid, {"tilt": math.radians(deg), "offset": mm * 1e-3,
                                           "chamfer": geometry.chamfer if ch else 0.0}))
    return tuple(out)


def force_profile(depth: np.ndarray, force: np.ndarray, proto: R1Protocol) -> tuple[np.ndarray, np.ndarray]:
    """(bin centres, smoothed mean force) on [0, z_end]; NaN where no sample falls within the smoothing window."""
    n = int(round(proto.z_end / proto.bin_m))
    d, f = np.asarray(depth, np.float64), np.asarray(force, np.float64)
    ok = np.isfinite(d) & np.isfinite(f) & (d >= 0.0)
    idx = np.floor(d[ok] / proto.bin_m).astype(np.int64)
    keep = idx < n
    sums = np.bincount(idx[keep], weights=f[ok][keep], minlength=n)
    cnt = np.bincount(idx[keep], minlength=n).astype(np.float64)
    box = np.ones(max(1, int(round(proto.smooth_m / proto.bin_m))))
    s, c = np.convolve(sums, box, "same"), np.convolve(cnt, box, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan)
    return (np.arange(n) + 0.5) * proto.bin_m, prof


def _peak(z: np.ndarray, prof: np.ndarray, sliding: float, proto: R1Protocol) -> tuple[float, float]:
    """Two-point peak force and depth (force-weighted centroid of the top band), or NaNs if there is none."""
    pre = (z < proto.window[0]) & np.isfinite(prof)
    if not pre.any() or not math.isfinite(sliding):
        return math.nan, math.nan
    i = int(np.argmax(np.where(pre, prof, -np.inf)))
    fmax = float(prof[i])
    if fmax - sliding < proto.peak_min_n:
        return math.nan, math.nan
    cut = fmax * (1.0 - proto.peak_band)
    out = np.flatnonzero(~(pre & (prof >= cut)))
    lo = out[out < i].max() + 1 if (out < i).any() else 0
    hi = out[out > i].min() if (out > i).any() else len(z)
    w = prof[lo:hi] - cut
    return fmax, float(np.sum(z[lo:hi] * w) / np.sum(w)) if np.sum(w) > 0.0 else float(z[i])


def insertion_metrics(depth: np.ndarray, force: np.ndarray, proto: R1Protocol = R1Protocol()) -> InsertionMetrics:
    """The R1 metrics of one force-depth series (see the module docstring)."""
    d, f = np.asarray(depth, np.float64), np.asarray(force, np.float64)
    fin_d, fin_f = d[np.isfinite(d)], f[np.isfinite(f)]
    if fin_d.size == 0 or fin_f.size == 0:
        return InsertionMetrics(math.nan, math.nan, math.nan, math.nan, math.nan)
    reach = float(fin_d.max())
    jam = bool(fin_f.max() >= proto.jam_force_frac * proto.f_stop or reach < proto.z_end - proto.jam_margin)
    if jam:
        return InsertionMetrics(math.nan, math.nan, math.nan, 1.0, reach)
    z, prof = force_profile(d, f, proto)
    win = (z >= proto.window[0]) & (z <= proto.window[1]) & np.isfinite(prof)
    sliding = float(np.mean(prof[win])) if win.any() else math.nan
    peak_f, peak_z = _peak(z, prof, sliding, proto)
    return InsertionMetrics(sliding, peak_f, peak_z, 0.0, math.nan)


def mean_profile(series: list[tuple[np.ndarray, np.ndarray]], proto: R1Protocol) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the per-series profiles (NaN-aware) on the common depth grid."""
    profs = [force_profile(d, f, proto) for d, f in series]
    stack = np.stack([p for _, p in profs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bins stay NaN
        return profs[0][0], np.nanmean(stack, axis=0)


def curve_error(sim: np.ndarray, real: np.ndarray) -> float:
    """CT-3: sqrt(mean_z (F_sim - F_rig)²) / max F_rig over the depths where both profiles are defined."""
    s, r = np.asarray(sim, np.float64), np.asarray(real, np.float64)
    ok = np.isfinite(s) & np.isfinite(r)
    scale = float(np.max(np.abs(r[ok]))) if ok.any() else 0.0
    if scale <= 0.0:
        return math.nan
    return float(np.sqrt(np.mean((s[ok] - r[ok]) ** 2)) / scale)


# --------------------------------------------------------------------------------------------- hole mesh ---
def _ring(radius: float, z: float, n: int) -> np.ndarray:
    th = np.arange(n) * (2.0 * math.pi / n)
    return np.stack([radius * np.cos(th), radius * np.sin(th), np.full(n, z)], axis=1)


def _strip(lo: int, hi: int, n: int, kind: int) -> list[tuple[int, int, int, int]]:
    out = []
    for i in range(n):
        j = (i + 1) % n
        out += [(lo + i, lo + j, hi + j, kind), (lo + i, hi + j, hi + i, kind)]
    return out


def _fan(center: int, ring: int, n: int, kind: int) -> list[tuple[int, int, int, int]]:
    return [(center, ring + i, ring + (i + 1) % n, kind) for i in range(n)]


_OUT, _IN, _UP, _DOWN, _CONE = range(5)


def _orient(verts: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Flip each triangle so its normal points along its face kind's outward direction."""
    f = np.asarray(faces, np.int64)
    tri, kind = f[:, :3], f[:, 3]
    v = verts.astype(np.float64)
    nrm = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    c = v[tri].mean(axis=1)
    radial = np.concatenate([c[:, :2], np.zeros((len(c), 1))], axis=1)
    up = np.tile([0.0, 0.0, 1.0], (len(c), 1))
    rxy = np.linalg.norm(c[:, :2], axis=1, keepdims=True)
    want = np.select([kind[:, None] == _OUT, kind[:, None] == _IN, kind[:, None] == _UP, kind[:, None] == _DOWN],
                     [radial, -radial, up, -up], default=-radial + up * rxy)
    flip = np.einsum("ij,ij->i", nrm, want) < 0.0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri.astype(np.int32)


def chamfered_hole_tris(spec: PegHoleSpec, chamfer: float) -> tuple[np.ndarray, np.ndarray]:
    """``blind_hole_tris`` with a 45° entry chamfer of width ``chamfer`` (0 = the sharp-edged hole)."""
    if chamfer <= 0.0:
        return blind_hole_tris(spec)
    n, ro, rb, h, zb = spec.segments, 0.5 * spec.outer_d, 0.5 * spec.bore_d, spec.height, spec.bottom_z
    if not 0.0 < chamfer < min(ro - rb, h - zb):
        raise ValueError(f"chamfer {chamfer} m does not fit the hole (wall {ro - rb} m, depth {h - zb} m)")
    rings = [(ro, 0.0), (ro, h), (rb + chamfer, h), (rb, h - chamfer), (rb, zb)]
    verts = np.concatenate([_ring(r, z, n) for r, z in rings] + [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, zb]])])
    c0, cb = 5 * n, 5 * n + 1
    faces = (_strip(0, n, n, _OUT) + _strip(2 * n, n, n, _UP) + _strip(3 * n, 2 * n, n, _CONE)
             + _strip(4 * n, 3 * n, n, _IN) + _fan(cb, 4 * n, n, _UP) + _fan(c0, 0, n, _DOWN))
    return verts.astype(np.float32), _orient(verts, faces)


__all__ = ["LAW_ID", "METRICS", "UNITS", "DEFAULT_MU", "R1Geometry", "R1Holder", "R1Protocol", "InsertionMetrics",
           "r1_conditions", "force_profile", "insertion_metrics", "mean_profile", "curve_error",
           "chamfered_hole_tris"]
