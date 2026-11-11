# SPDX-License-Identifier: Apache-2.0
"""Thread end cones (SCR-SC3-2): ISO 4753 chamfered bolt ends and nut countersinks on the analytic thread SDF."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.contact.thread_sdf import (EndChamfer, ThreadParams, ThreadSpecError, iso4753_chamfer, nut_countersink,
                                      parse_thread_id, sd_thread, thread_mesh, thread_sdf_np, thread_sdf_params)

BOLT, NUT = "iso_thread:M10x1.5:6g", "iso_thread:M10x1.5:6H"
Z0, Z1 = -4e-3, 4e-3


def _params(tid: str, chamfered: bool):
    spec = parse_thread_id(tid)
    ch = (iso4753_chamfer(spec) if spec.external else nut_countersink(spec)) if chamfered else None
    kw = {} if spec.external else {"outer_radius": spec.r_major + 2e-3}
    return spec, ch, kw, thread_sdf_params(spec, z0=Z0, z1=Z1, chamfer0=ch, chamfer1=ch, **kw)


def _points(n: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-7e-3, 7e-3, (n, 2)), rng.uniform(-5e-3, 5e-3, n)]


@wp.kernel
def _eval(prm: ThreadParams, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = sd_thread(prm, pts[i])


@pytest.mark.parametrize("tid", [BOLT, NUT])
def test_chamfer_only_removes_material_near_the_ends(tid: str) -> None:
    _, _, _, flat = _params(tid, False)
    _, _, _, cham = _params(tid, True)
    pts = _points()
    a, b = thread_sdf_np(cham, pts), thread_sdf_np(flat, pts)
    assert (a >= b - 1e-15).all()                                     # a cone only cuts
    spec = parse_thread_id(tid)
    mid = np.abs(pts[:, 2]) < 2e-3                                    # cones reach < 1.2 mm into the slab
    mid &= np.hypot(pts[:, 0], pts[:, 1]) > spec.r_minor - 5e-4       # (inside a nut's bore the countersink line runs
    assert np.array_equal(a[mid], b[mid])                             # past its apex: still a valid lower bound)
    assert (a > b + 1e-6).any()                                       # and it does cut at the ends


@pytest.mark.parametrize("tid", [BOLT, NUT])
def test_default_params_unchanged(tid: str) -> None:
    spec = parse_thread_id(tid)
    kw = {} if spec.external else {"outer_radius": spec.r_major + 2e-3}
    prm = thread_sdf_params(spec, z0=Z0, z1=Z1, **kw)
    assert prm.ch0 == 0.0 and prm.ch1 == 0.0


@pytest.mark.parametrize("tid", [BOLT, NUT])
def test_chamfered_sdf_is_1_lipschitz(tid: str) -> None:
    _, _, _, prm = _params(tid, True)
    a = _points(5000, 1)
    b = a + np.random.default_rng(2).normal(0.0, 2e-4, a.shape)
    da, db = thread_sdf_np(prm, a), thread_sdf_np(prm, b)
    assert (np.abs(da - db) <= np.linalg.norm(a - b, axis=1) + 1e-12).all()


@pytest.mark.parametrize("tid", [BOLT, NUT])
def test_warp_twin_matches_numpy(tid: str) -> None:
    _, _, _, prm = _params(tid, True)
    pts = _points(4096, 3)
    out = wp.zeros(len(pts), dtype=float, device="cpu")
    wp.launch(_eval, len(pts), inputs=[prm.to_warp(), wp.array(pts.astype(np.float32), dtype=wp.vec3, device="cpu"),
                                       out], device="cpu")
    assert np.abs(out.numpy() - thread_sdf_np(prm, pts.astype(np.float32))).max() < 2e-7


@pytest.mark.parametrize("tid", [BOLT, NUT])
def test_mesh_lies_on_the_chamfered_surface(tid: str) -> None:
    spec, ch, kw, prm = _params(tid, True)
    v, _ = thread_mesh(spec, Z1 - Z0, 96, z0=Z0, chamfer0=ch, chamfer1=ch, **kw)
    r = np.hypot(v[:, 0], v[:, 1])
    side = (r < spec.r_major + 1e-4) if spec.external else (r < kw["outer_radius"] - 1e-4)   # thread-surface verts
    assert np.abs(thread_sdf_np(prm, v[side])).max() < 2e-6


def test_iso_values() -> None:
    bolt, nut = parse_thread_id(BOLT), parse_thread_id(NUT)
    ch = iso4753_chamfer(bolt)
    assert ch.angle_deg == 45.0 and ch.r_face < bolt.r_minor                      # below the root: no feather edge
    cs = nut_countersink(nut)
    assert 0.5 * nut.d <= cs.r_face <= 0.5 * 1.08 * nut.d and cs.angle_deg == 60.0  # d <= d_a <= 1.08 d, 120°
    with pytest.raises(ThreadSpecError):
        iso4753_chamfer(nut)
    with pytest.raises(ThreadSpecError):
        EndChamfer(1e-3, 95.0)


def _nut_feature(chamfered: bool, band: float = 0.2):
    from alloy.contact.analytic_samples import ThreadFeature
    spec = parse_thread_id(NUT)
    cs = nut_countersink(spec) if chamfered else None
    return spec, ThreadFeature(NUT, Z0, Z1, 8.5e-3, flank_band=band, chamfer0=cs, chamfer1=cs)


def test_flat_ends_sample_as_before() -> None:
    from alloy.contact.analytic_samples import END_TRIM, thread_samples
    spec, feat = _nut_feature(False)
    pts = thread_samples(feat, 5e-5).points
    assert pts[:, 2].min() >= Z0 + END_TRIM * spec.pitch - 1e-9 and pts[:, 2].max() <= Z1 - END_TRIM * spec.pitch + 1e-9


def test_chamfered_ends_sample_the_lead_in() -> None:
    from alloy.contact.analytic_samples import END_BAND, END_TRIM, thread_samples
    spec, flat = _nut_feature(False)
    _, cham = _nut_feature(True)
    a, b = thread_samples(cham, 5e-5).points, thread_samples(flat, 5e-5).points
    p = spec.pitch
    end = (a[:, 2] < Z0 + END_TRIM * p) | (a[:, 2] > Z1 - END_TRIM * p)
    assert end.any()                                                  # samples inside the old half-pitch trim
    r = np.hypot(a[:, 0], a[:, 1])
    lead = a[:, 2] < Z0 + END_BAND * p
    crest_band = r < spec.r_minor + 0.3 * (spec.r_major - spec.r_minor)
    assert (lead & crest_band).any()                                  # crest-side samples near a chamfered end
    assert not ((~lead & (a[:, 2] < Z1 - END_BAND * p)) & crest_band).any()   # mid-slab keeps the 20 % band
    prm = thread_sdf_params(spec, z0=Z0, z1=Z1, outer_radius=8.5e-3, chamfer0=cham.chamfer0, chamfer1=cham.chamfer1)
    assert (thread_sdf_np(prm, a) <= 1e-7).all()                      # none the countersink cut away
    mid = np.abs(b[:, 2]) < 1e-3
    assert np.array_equal(np.sort(a[np.abs(a[:, 2]) < 1e-3], axis=0), np.sort(b[mid], axis=0))
