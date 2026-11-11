# SPDX-License-Identifier: Apache-2.0
"""Analytic thread contact rows (alloy.contact.analytic_pair) and the A9/R2 thread twins' CPU parts: mated and pushed
bolt/nut rows, bearing face on a washer, determinism, capacity, assembly checks, the shared metric estimator, the
synthetic R2 dataset and the law specs. The stepped A9 twin runs here once on CPU (compile_heavy); A9/R2 on GPU are
in test_thread_gpu.py."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.bench.law import register_law, unregister_law
from alloy.contact import analytic_pair as ap
from alloy.contact import thread_rig as tr
from alloy.contact import thread_twins as tt
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.thread_sdf import parse_thread_id, rundown_torque
from alloy.contact.types import ContactClass, ContactFlags, PrecisionPairSpec
from alloy.core.errors import AssemblyError

BASIC_BOLT, NUT = "iso_thread:M10x1.5:6h", "iso_thread:M10x1.5:6H"
COS30 = math.cos(math.radians(30.0))


def _model(device: str, worlds: int = 1, capacity: int = 512, washer_gap: float | None = None) -> newton.Model:
    """Per world: bolt (template shape 0), nut (1) and, with ``washer_gap``, a static 2 mm washer plate (2) whose
    lower face is ``washer_gap`` above the nut's top face."""
    b = newton.ModelBuilder()
    for _ in range(worlds):
        b.begin_world(gravity=(0.0, 0.0, 0.0))
        tt.build_thread_world(b, tt.ThreadRig())
        if washer_gap is not None:
            bolt, nut = b.shape_count - 2, b.shape_count - 1
            pose = wp.transform((0.0, 0.0, tt.NUT_HALF + washer_gap + 1e-3), wp.quat_identity())
            washer = b.add_shape_box(-1, xform=pose, hx=0.01, hy=0.01, hz=1e-3, label="washer")
            b.add_shape_collision_filter_pair(washer, nut)
            b.add_shape_collision_filter_pair(washer, bolt)
        b.end_world()
    model = b.finalize(device=device)
    model.rigid_contact_max = capacity * worlds
    return model


def _thread_pair(gap: float = 1e-4, split_k: bool = True, bolt: str = BASIC_BOLT) -> ap.AnalyticPair:
    spec = PrecisionPairSpec(0, 1, mode="sdf", voxel_size=1e-4, gap=gap, pair_stiffness=1e6, ccd="off")
    return ap.AnalyticPair(spec, ap.ThreadFeature(bolt, *tt.BOLT_Z),
                           ap.ThreadFeature(NUT, -tt.NUT_HALF, tt.NUT_HALF, tt.NUT_OUTER), split_k=split_k)


def _rows(model: newton.Model, pairs, dz: float = 0.0, layer=None):
    """Collide once with every nut moved by ``dz`` along z; returns the layer and its live rows (separation
    recomputed from the points)."""
    layer = layer or AlloyContactLayer(model, [], native="newton")
    ap.attach_analytic_pairs(layer, pairs)
    state = model.state()
    q = state.body_q.numpy()
    nuts = [i for i, lab in enumerate(model.body_label) if str(lab).endswith("nut")]
    q[nuts, 2] += dz
    state.body_q.assign(q)
    layer.collide(state, layer.contacts(), dt=1e-3)
    ap.analytic_collide(layer, state, layer.contacts(), 1e-3)
    c, side = layer.contacts(), layer.sidecar()
    n = int(c.rigid_contact_count.numpy()[0])
    nrm, p0, p1 = (getattr(c, f"rigid_contact_{k}").numpy()[:n] for k in ("normal", "point0", "point1"))
    body = model.shape_body.numpy()[c.rigid_contact_shape1.numpy()[:n]]
    x1 = p1 + q[body, :3]                                               # nut bodies are not rotated here
    out = {"n": n, "normal": nrm, "sep": np.einsum("ij,ij->i", nrm, x1 - p0), "x1": x1,
           "weight": side.weight.numpy()[:n], "k": c.rigid_contact_stiffness.numpy()[:n],
           "flags": side.flags.numpy()[:n], "cls": side.cls.numpy()[:n], "world": side.world.numpy()[:n]}
    return layer, out


# ---------------------------------------------------------------- rows ----------------------------------------------

def test_mated_basic_profile_has_no_deep_rows(device):
    _, r = _rows(_model(device), [_thread_pair()])
    assert r["n"] > 50
    assert r["sep"].min() >= -1e-6                                      # coincident flanks at the basic profile
    assert np.all(r["cls"] == int(ContactClass.PRECISION_SDF))


@pytest.mark.parametrize("delta", [5e-6, 2e-5])
def test_axial_push_loads_flanks_at_30_degrees(device, delta):
    """Nut pushed down by δ: the loaded rows sit at -δ·cos 30° with normals 30° off the axis (+z), the opposite
    flanks open by the same amount."""
    _, r = _rows(_model(device), [_thread_pair()], dz=-delta)
    up = r["normal"][:, 2] > 0.25
    assert up.sum() > 20 and (~up).sum() > 20
    np.testing.assert_allclose(r["sep"][up], -delta * COS30, rtol=0.01)
    np.testing.assert_allclose(r["sep"][~up], delta * COS30, rtol=0.01)
    angle = np.degrees(np.arccos(np.abs(r["normal"][:, 2])))
    np.testing.assert_allclose(angle, 30.0, atol=0.5)
    flags = r["flags"]
    assert np.all(flags[up] == int(ContactFlags.NEW)) and np.all(flags[~up] & int(ContactFlags.SPECULATIVE))


def test_rows_stay_within_bins_and_capacity(device):
    model = _model(device, worlds=2, capacity=8)
    layer, r = _rows(model, [_thread_pair()], dz=-1e-5)
    assert r["n"] == 16 and int(layer.overflow.numpy()[0]) > 0
    model2 = _model(device, worlds=2)
    layer2, r2 = _rows(model2, [_thread_pair()], dz=-1e-5)
    rows = ap._CACHE[layer2]
    assert r2["n"] <= rows.n_bins * 2 and int(layer2.overflow.numpy()[0]) == 0
    assert set(np.unique(r2["world"]).tolist()) == {0, 1}


def test_rows_are_deterministic(device):
    a = _rows(_model(device), [_thread_pair()], dz=-7e-6)[1]
    b = _rows(_model(device), [_thread_pair()], dz=-7e-6)[1]
    for key in ("normal", "sep", "x1", "k"):
        assert np.array_equal(a[key], b[key])


def test_patch_weights(device):
    _, split = _rows(_model(device), [_thread_pair(split_k=True)], dz=-1e-5)
    up = split["normal"][:, 2] > 0.25
    assert split["weight"][up].sum() == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(split["k"], 1e6 * split["weight"], rtol=1e-6)
    _, whole = _rows(_model(device), [_thread_pair(split_k=False)], dz=-1e-5)
    assert np.all(whole["weight"] == 1.0) and np.all(whole["k"] == 1e6)


def test_bearing_face_on_washer(device):
    """Nut face pushed 3 µm into the washer: one row per sector at -3 µm, normal -z, at the face's torque radius. The
    washer sits one pitch above the nut, so the 6g/6H thread is mated again (3 µm < its 9 µm axial play)."""
    gap = parse_thread_id(NUT).pitch
    face = ap.AnalyticPair(PrecisionPairSpec(2, 1, mode="sdf", voxel_size=2e-4, gap=2e-6, pair_stiffness=2.4e8),
                           ap.SlabFeature(-1e-3, 1e-3), ap.FaceFeature(tt.NUT_HALF, tt.BEARING_R_IN, tt.NUT_OUTER))
    pairs = [_thread_pair(gap=1e-9, bolt="iso_thread:M10x1.5:6g"), face]   # 6g/6H play: no thread rows here
    _, r = _rows(_model(device, washer_gap=gap), pairs, dz=gap + 3e-6)
    face = r["normal"][:, 2] < -0.99
    assert face.sum() == ap.SECTORS and r["n"] == ap.SECTORS
    np.testing.assert_allclose(r["sep"][face], -3e-6, atol=2e-8)
    radius = np.hypot(r["x1"][face, 0], r["x1"][face, 1])
    np.testing.assert_allclose(radius, ap.torque_radius(tt.BEARING_R_IN, tt.NUT_OUTER), atol=1e-4)
    np.testing.assert_allclose(r["k"][face], 2.4e8 / ap.SECTORS, rtol=1e-6)


def test_assembly_checks(device):
    model = _model(device)
    layer = AlloyContactLayer(model, [], native="newton")
    state = model.state()
    with pytest.raises(AssemblyError):
        ap.analytic_collide(layer, state, layer.contacts(), 1e-3)          # nothing attached
    bad = ap.AnalyticPair(PrecisionPairSpec(0, 0, mode="sdf"), ap.ThreadFeature(BASIC_BOLT, -0.01, 0.01),
                          ap.ThreadFeature(NUT, -4e-3, 4e-3, 8e-3))
    with pytest.raises(AssemblyError):
        ap.attach_analytic_pairs(layer, [bad])                            # shape pair not filtered
    ap.attach_analytic_pairs(layer, [_thread_pair()])
    other = newton.Contacts(16, 0, device=device)
    with pytest.raises(AssemblyError):
        ap.analytic_collide(layer, state, other, 1e-3)
    with pytest.raises(ValueError):
        ap.analytic_collide(layer, state, layer.contacts(), 0.0)


def test_sample_bins():
    feat = ap.ThreadFeature(NUT, -4e-3, 4e-3, 8.5e-3, flank_band=0.5)
    sb = ap.thread_samples(feat, 1e-4)
    spec = parse_thread_id(NUT)
    r = np.hypot(sb.points[:, 0], sb.points[:, 1])
    lo = spec.r_minor + 0.25 * (spec.r_major - spec.r_minor)
    assert r.min() >= lo - 1e-6 and r.max() <= spec.r_major - (lo - spec.r_minor) + 1e-6
    assert np.all(np.abs(sb.points[:, 2]) <= 4e-3 - 0.5 * spec.pitch + 1e-9)
    assert sb.n_bins == ap.SECTORS * 2 * (math.floor(4e-3 / spec.pitch) - math.floor(-5.5e-3 / spec.pitch) + 1)
    assert np.all(np.diff(sb.start) >= 0) and sb.start[-1] == len(sb.points)
    face = ap.face_samples(ap.FaceFeature(4e-3, 5.25e-3, 8.5e-3), 2e-4)
    assert face.n_bins == ap.SECTORS and np.all(np.diff(face.start) > 0)


# ---------------------------------------------------------------- metrics, data, laws --------------------------------

def test_thread_metrics_recover_closed_forms():
    rng = np.random.default_rng(3)
    truth = tr.TRUTH["dry"]
    s = tr._trial_series(truth, 2.0, rng)
    m = tt.thread_metrics(s["t"], s["theta"], s["z"], s["torque"])
    assert m["advance_per_rev"] == pytest.approx(1.5e-3, abs=2e-6)
    assert m["rundown_torque"] == pytest.approx(tr.thread_factor(truth.mu_thread) * tr.F_AXIAL, rel=0.03)
    assert m["seating_slope"] == pytest.approx(tr.seating_slope(truth.mu_thread, truth.mu_bearing, tr.K_CLAMP),
                                               rel=0.03)
    assert tr.fit_mu(tr.thread_factor(0.12) * 5.0, 5.0) == pytest.approx(0.12, rel=1e-12)
    assert tr.fit_mu_bearing(tr.seating_slope(0.12, 0.09, 2e8), 0.12, 2e8) == pytest.approx(0.09, rel=1e-12)


def test_synthetic_r2_dataset():
    a, b = tr.synthetic_r2_dataset(1), tr.synthetic_r2_dataset(1)
    assert a.sha256 == b.sha256 and a.sha256 != tr.synthetic_r2_dataset(2).sha256
    assert len(a.split("calib")) == len(a.split("valid")) == 10
    assert a.card.protocol_version.startswith("synthetic")
    twin = tr.ThreadR2Twin()
    twin.bind_calib(a.card, a.split("calib"))
    for cid, truth in tr.TRUTH.items():
        assert twin.fit[cid].mu_thread == pytest.approx(truth.mu_thread, rel=0.02)
        assert twin.fit[cid].mu_bearing == pytest.approx(truth.mu_bearing, rel=0.05)


def test_law_specs_register():
    for law in (tt.A9_LAW, tr.R2_LAW):
        assert law.producer == "WP3" and law.solvers == ("mujoco",)
        register_law(law, tt.ThreadA9Twin if law is tt.A9_LAW else tr.ThreadR2Twin)
        unregister_law(law.id)
    assert tt.A9_LAW.id.startswith("A9.") and tr.R2_LAW.id.startswith("R2.")
    ref = tt.ThreadA9Twin().reference(tt.ThreadA9Twin().conditions(True)[0])
    assert ref["advance_per_rev"] == 1.5e-3
    assert ref["rundown_torque"] == pytest.approx(rundown_torque(parse_thread_id(tt.BOLT_ID), 5.0, 0.14))
    assert tt.row_stiffness(tt.A9_DT) > 0.0 and tt.A9_DT <= 1e-3


@pytest.mark.compile_heavy
def test_a9_twin_cpu():
    """The stepped A9 twin on CPU (quick: both μ, one turn after settling)."""
    from alloy.bench.runner import run_law  # noqa: PLC0415

    wp.config.quiet = True
    res = run_law(tt.A9_LAW, tt.ThreadA9Twin(), None, device="cpu", quick=True)
    assert res.monitors.get("finite"), res.monitors
    assert res.verdict == "pass", [(v.name, v.error, v.bound) for v in res.verdicts]
