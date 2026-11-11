# SPDX-License-Identifier: Apache-2.0
"""ISO thread analytic SDF (law A9, SPEC A9 amendment): id parsing, accuracy and conservativeness against a
brute-force distance, the 1-Lipschitz bound, the Warp twin, the mesh, the A9 formulas and the texture budget.

The brute-force reference is independent of the module: the profile is rebuilt from the ISO 68-1 fundamental
triangle, and the 3D distance is the minimum over a dense set of meridional half-planes (angle φ from the point's
own) of the exact in-plane distance to that polyline, combined with the out-of-plane offset r·sin φ. It is itself
checked against a dense sampling of the helical surface.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.contact import thread_sdf as ts
from alloy.contact.validate import SDF_PART_BUDGET_B, is_watertight
from alloy.core.errors import PrecisionSpecError

BOLT, NUT = "iso_thread:M10x1.5:6g", "iso_thread:M10x1.5:6H"
Z0, Z1 = -0.01, 0.01
NUT_OUTER = 8.5e-3                  # inscribed radius of the 17 mm hex (DIN 934 M10)
BAND = 5.0e-4                       # accuracy goal: points within 0.5 mm of the thread
GOAL = 1.0e-6                       # |sdf - true| <= 1 µm inside BAND


def _params(analytic_id: str):
    spec = ts.parse_thread_id(analytic_id)
    outer = None if spec.external else NUT_OUTER
    return spec, ts.thread_sdf_params(spec, z0=Z0, z1=Z1, outer_radius=outer)


# ---------------------------------------------------------------- independent reference ----------------------------

def _iso_radius(spec, u):
    """ISO 68-1 basic profile from the fundamental triangle: V of height H with its apex at u = 0 on radius
    d/2 + H/8, cut at d/2 (crest) and d/2 - 5H/8 (root), shifted by the fundamental deviation."""
    p = spec.pitch
    h = math.sqrt(3.0) / 2.0 * p
    w = np.abs(u - p * np.round(u / p))
    return np.clip(spec.d / 2 + h / 8 - 2.0 * h * w / p, spec.d / 2 - 5 * h / 8, spec.d / 2) + spec.deviation / 2


def _iso_polyline(spec):
    """Vertices (u, r) of the profile on u in [-3p/2, 3p/2] (corners where the V meets the two cuts)."""
    p = spec.pitch
    h = math.sqrt(3.0) / 2.0 * p
    uc, ur = (h / 8) * p / (2 * h), (h / 8 + 5 * h / 8) * p / (2 * h)
    us = sorted({k * p + s * c for k in (-1, 0, 1) for s in (-1, 1) for c in (uc, ur)} | {-1.5 * p, 1.5 * p})
    us = np.array(us)
    return np.stack([us, _iso_radius(spec, us)], -1)


def _dist_polyline(u, r, verts):
    """Euclidean distance from points (u, r) to the polyline ``verts`` (any array shape)."""
    best = np.full(np.shape(u), np.inf)
    for (ua, ra), (ub, rb) in zip(verts[:-1], verts[1:]):
        eu, er = ub - ua, rb - ra
        t = np.clip(((u - ua) * eu + (r - ra) * er) / (eu * eu + er * er), 0.0, 1.0)
        best = np.minimum(best, np.hypot(u - ua - t * eu, r - ra - t * er))
    return best


def _plane_dist(spec, verts, r, uf, phi):
    """Distance to the surface points in the half-plane at angle φ from the point (r, phase uf)."""
    p = spec.pitch
    u = uf - spec.lead_2pi * phi
    u = u - p * np.round(u / p)
    return np.hypot(_dist_polyline(u, r * np.cos(phi), verts), r * np.sin(phi))


def _reference(spec, pts):
    """Brute-force signed distance to the infinite thread surface: coarse φ grid, then a fine grid at the minimum."""
    verts = _iso_polyline(spec)
    r = np.hypot(pts[:, 0], pts[:, 1])[:, None]
    uf = (pts[:, 2] - spec.lead_2pi * np.arctan2(pts[:, 1], pts[:, 0]))[:, None]
    coarse = np.linspace(-0.2, 0.2, 1201)
    best = coarse[np.argmin(_plane_dist(spec, verts, r, uf, coarse[None, :]), axis=1)]
    step = coarse[1] - coarse[0]
    fine = best[:, None] + np.linspace(-2.0 * step, 2.0 * step, 401)[None, :]
    dist = _plane_dist(spec, verts, r, uf, fine).min(axis=1)
    outside = r[:, 0] > _iso_radius(spec, uf[:, 0])
    return np.where(outside == spec.external, dist, -dist)


def _band_points(spec, rng, n):
    r = rng.uniform(spec.r_minor - 0.6e-3, spec.r_major + 0.6e-3, n)
    t = rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t), rng.uniform(-3e-3, 3e-3, n)], -1)


# ---------------------------------------------------------------- parsing -------------------------------------------

def test_parse_m10_bolt_and_nut():
    bolt, nut = ts.parse_thread_id(BOLT), ts.parse_thread_id(NUT)
    assert bolt.external and not nut.external
    assert (bolt.tol_class, nut.tol_class) == ("6g", "6H")
    assert bolt.d2 == pytest.approx(9.026e-3, abs=5e-7)          # ISO 724
    assert bolt.d1 == pytest.approx(8.376e-3, abs=5e-7)
    assert bolt.d3 == pytest.approx(8.160e-3, abs=5e-7)
    assert bolt.deviation == pytest.approx(-31.5e-6, abs=1e-12)  # es = -(15 + 11·1.5) µm
    assert nut.deviation == 0.0
    assert ts.parse_thread_id("iso_thread:M10x1.5:6G").deviation == pytest.approx(31.5e-6, abs=1e-12)
    assert ts.parse_thread_id("iso_thread:M6x1:6g").deviation == pytest.approx(-26e-6, abs=1e-12)
    assert bolt.r_major == pytest.approx(5e-3 - 15.75e-6, abs=1e-12)


@pytest.mark.parametrize("bad", ["iso_thread:M10:6g", "iso_thread:M10x1.5:6a", "iso_thread:M10x1.5:6E",
                                 "iso_thread:M10x1.5:2g", "iso_thread:M10x1.5", "unc_thread:M10x1.5:6g",
                                 "iso_thread:M1x1:6g"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        ts.parse_thread_id(bad)
    with pytest.raises(PrecisionSpecError):
        ts.parse_thread_id(bad)


def test_param_validation():
    bolt, nut = ts.parse_thread_id(BOLT), ts.parse_thread_id(NUT)
    with pytest.raises(ValueError):
        ts.thread_sdf_params(nut, z0=0.0, z1=0.01)                       # internal needs outer_radius
    with pytest.raises(ValueError):
        ts.thread_sdf_params(bolt, z0=0.0, z1=0.01, outer_radius=0.01)   # external has none
    with pytest.raises(ValueError):
        ts.thread_sdf_params(bolt, z0=0.01, z1=0.0)
    with pytest.raises(ValueError):
        ts.thread_sdf_params(bolt, z0=0.0, z1=0.01, reach=bolt.r_minor)
    with pytest.raises(ValueError):
        ts.thread_mesh(bolt, 0.01, segments_per_turn=100)


# ---------------------------------------------------------------- geometry and accuracy -----------------------------

def test_right_hand_phase():
    """The major flat runs along the right-hand helix (r_major, φ, p·φ/2π) through θ = 0, z = 0."""
    spec, prm = _params(BOLT)
    phi = np.linspace(-3.0, 3.0, 13)
    on = np.stack([spec.r_major * np.cos(phi), spec.r_major * np.sin(phi), spec.lead_2pi * phi], -1)
    assert np.abs(ts.thread_sdf_np(prm, on)).max() < 1e-12
    quarter = np.array([-0.5 * np.pi, 0.5 * np.pi])                     # a left-hand helix is half a pitch off
    left = np.stack([spec.r_major * np.cos(quarter), spec.r_major * np.sin(quarter), -spec.lead_2pi * quarter], -1)
    assert ts.thread_sdf_np(prm, left).min() > 0.5 * (spec.r_major - spec.r_minor)


def test_reference_matches_dense_surface_samples(rng):
    """The φ-search reference equals the nearest of a dense (φ, arclength) sampling of the surface to within the
    sampling bound sqrt(r²+L²)·Δφ/2 + Δs/2 (and never exceeds it: samples lie on the surface)."""
    for analytic_id in (BOLT, NUT):
        spec, _ = _params(analytic_id)
        pts = _band_points(spec, rng, 200)
        ref = _reference(spec, pts)
        pick = pts[(np.abs(ref) > 5e-5) & (np.abs(ref) < BAND)][:3]
        verts = _iso_polyline(spec)
        seg = np.hypot(*np.diff(verts, axis=0).T)
        s = np.arange(0.0, seg.sum(), 0.5e-6)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        su, sr = np.interp(s, cum, verts[:, 0]), np.interp(s, cum, verts[:, 1])
        phi = np.arange(-0.15, 0.15, 1.5e-4)
        bound = math.hypot(spec.r_major, spec.lead_2pi) * 0.75e-4 + 0.25e-6
        for pt, rv in zip(pick, _reference(spec, pick)):
            r, uf = math.hypot(pt[0], pt[1]), pt[2] - spec.lead_2pi * math.atan2(pt[1], pt[0])
            uf -= spec.pitch * round(uf / spec.pitch)
            near = np.abs(su - uf) < 0.6e-3
            dz = uf - su[near][None, :] - spec.lead_2pi * phi[:, None]
            d2 = r * r + sr[near] ** 2 - 2.0 * r * sr[near] * np.cos(phi)[:, None] + dz * dz
            gap = math.sqrt(d2.min()) - abs(rv)
            assert -1e-10 <= gap <= bound, (analytic_id, pt, rv, gap, bound)


@pytest.mark.parametrize("analytic_id", [BOLT, NUT])
def test_accuracy_vs_bruteforce(analytic_id, rng):
    """Within 0.5 mm of the thread: same sign, never larger in magnitude, error <= 1 µm and <= sdf_error_bound.
    The reference overestimates by its own φ resolution, ~(r·Δφ)²/(8|d|) with r·Δφ < 20 nm, so the bound is checked
    from |d| >= 1 µm."""
    spec, prm = _params(analytic_id)
    pts = _band_points(spec, rng, 3000)
    ref = _reference(spec, pts)
    keep = np.abs(ref) <= BAND
    pts, ref = pts[keep], ref[keep]
    got = ts.thread_sdf_np(prm, pts)
    assert keep.sum() > 1500
    assert np.all(np.sign(got[np.abs(ref) > 1e-9]) == np.sign(ref[np.abs(ref) > 1e-9]))
    err = np.abs(ref) - np.abs(got)
    assert err.min() >= -1e-12                                           # conservative: |sdf| <= |true| <= |ref|
    assert err.max() <= GOAL, err.max()
    far = np.abs(ref) >= 1e-6
    bound = np.array([ts.sdf_error_bound(prm, x) for x in ref[far]])
    assert np.all(err[far] <= bound), (err[far] - bound).max()
    assert ts.sdf_error_bound(prm, BAND) < GOAL


@pytest.mark.parametrize("analytic_id", [BOLT, NUT])
def test_lipschitz(analytic_id, rng):
    """|f(a) - f(b)| <= |a - b| everywhere, including the axis, the slab ends and the nut's outer wall."""
    spec, prm = _params(analytic_id)
    n = 200_000
    r = rng.uniform(0.0, NUT_OUTER + 1e-3, n)
    t = rng.uniform(-np.pi, np.pi, n)
    a = np.stack([r * np.cos(t), r * np.sin(t), rng.uniform(Z0 - 1e-3, Z1 + 1e-3, n)], -1)
    b = a + rng.normal(0.0, 1.0e-4, (n, 3))
    lhs = np.abs(ts.thread_sdf_np(prm, a) - ts.thread_sdf_np(prm, b))
    assert np.all(lhs <= np.linalg.norm(a - b, axis=1) * (1.0 + 1e-9) + 1e-15)


# ---------------------------------------------------------------- Warp twin -----------------------------------------

@wp.kernel
def _eval_thread(prm: ts.ThreadParams, pts: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    out[i] = ts.sd_thread(prm, pts[i])


@pytest.mark.parametrize("analytic_id", [BOLT, NUT])
def test_warp_matches_numpy(analytic_id, device, rng):
    spec, prm = _params(analytic_id)
    n = 4000
    pts = np.concatenate([_band_points(spec, rng, n // 2),
                          rng.uniform(-0.012, 0.012, (n // 2, 3))]).astype(np.float32)
    out = wp.zeros(n, dtype=float, device=device)
    wp.launch(_eval_thread, dim=n, inputs=[prm.to_warp(), wp.array(pts, dtype=wp.vec3, device=device), out],
              device=device)
    ref = ts.thread_sdf_np(prm, pts.astype(np.float64))
    assert np.abs(out.numpy() - ref).max() < 1e-8                       # float32 on ~1e-2 m coordinates


# ---------------------------------------------------------------- mesh ----------------------------------------------

def _signed_volume(v, t):
    return float(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0)


def _mean_r2(spec):
    u = np.linspace(0.0, spec.pitch, 200_001)
    return float(np.trapezoid(_iso_radius(spec, u) ** 2, u) / spec.pitch)


@pytest.mark.parametrize("analytic_id", [BOLT, NUT])
def test_mesh_watertight_volume_and_zero_set(analytic_id):
    """Closed and consistently wound; volume = π·L·mean(R²) (bolt) or π·L·(R_out² - mean(R²)) (nut) up to the
    N-gon factor sin(x)/x, x = 2π/N; every vertex lies on the SDF zero set."""
    spec, _ = _params(analytic_id)
    n, z0, length = 96, -3.0e-3, 6.3e-3                                 # length not a multiple of p/n
    outer = None if spec.external else NUT_OUTER
    v, t = ts.thread_mesh(spec, length, n, z0=z0, outer_radius=outer)
    assert is_watertight(t)
    area = np.pi * _mean_r2(spec) if spec.external else np.pi * (NUT_OUTER**2 - _mean_r2(spec))
    ngon = math.sin(2 * math.pi / n) / (2 * math.pi / n)
    assert _signed_volume(v, t) / (area * length * ngon) == pytest.approx(1.0, abs=2e-5)
    assert ts.mean_profile_r2(spec) == pytest.approx(_mean_r2(spec), rel=1e-9)
    prm = ts.thread_sdf_params(spec, z0=z0, z1=z0 + length, outer_radius=outer)
    assert np.abs(ts.thread_sdf_np(prm, v)).max() < 1e-12


def _wall_area(spec, length, n):
    v, t = ts.thread_mesh(spec, length, n)
    wall = t[np.ptp(v[t][:, :, 2], axis=1) > 0.0]                      # cap triangles are flat in z
    return 0.5 * np.linalg.norm(np.cross(v[wall[:, 1]] - v[wall[:, 0]], v[wall[:, 2]] - v[wall[:, 0]]), axis=1).sum()


def test_mesh_area_converges_to_surface_area():
    """The bolt mesh wall converges to the closed-form thread_surface_area from below at second order in 1/N."""
    spec = ts.parse_thread_id(BOLT)
    length = 4 * spec.pitch
    exact = ts.thread_surface_area(spec, length)
    gap96, gap192 = (1.0 - _wall_area(spec, length, n) / exact for n in (96, 192))
    assert 0.0 < gap192 < 1.5e-4
    assert gap96 / gap192 == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------- law A9 and the texture budget ---------------------

def test_a9_advance_and_rundown_torque():
    spec = ts.parse_thread_id(BOLT)
    assert ts.advance_per_rev(spec) == 1.5e-3
    f, d2, p = 5.0, spec.d2, spec.pitch
    assert ts.rundown_torque(spec, f, 0.0) == pytest.approx(f * p / (2 * math.pi), rel=1e-12)  # work balance
    for mu in (0.1, 0.14, 0.3):                         # T = F·d2/2·tan(λ + ρ'), tan λ = p/πd2, tan ρ' = μ·sec30°
        lam, rho = math.atan(p / (math.pi * d2)), math.atan(mu / math.cos(math.radians(30.0)))
        assert ts.rundown_torque(spec, f, mu) == pytest.approx(f * d2 / 2 * math.tan(lam + rho), rel=1e-12)


def test_texture_sdf_exceeds_part_budget():
    """SPEC A9: a texture at the voxel the 6g/6H clearance demands (radial clearance on d2, (EI - es)/2 = 15.75 µm,
    "16 µm") exceeds 64 MB for the 8 mm nut thread alone; the 50 µm default voxel would fit."""
    bolt, nut = ts.parse_thread_id(BOLT), ts.parse_thread_id(NUT)
    voxel = 0.5 * (nut.deviation - bolt.deviation)
    assert voxel == pytest.approx(15.75e-6, abs=1e-12) and voxel <= 16e-6
    assert ts.texture_sdf_bytes(nut, 8e-3, 16e-6) > SDF_PART_BUDGET_B
    assert ts.texture_sdf_bytes(bolt, 60e-3, 16e-6) > SDF_PART_BUDGET_B
    assert ts.texture_sdf_bytes(nut, 8e-3, 5e-5) < SDF_PART_BUDGET_B
