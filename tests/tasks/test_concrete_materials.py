# SPDX-License-Identifier: Apache-2.0
"""Tests of the ported concrete material and process laws (alloy.tasks.concrete_materials)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.tasks import concrete_materials as cm

T21 = 21.1
MIN = 60.0


def test_masco_reference_and_scaling():
    assert cm.masco_factor((70 - 32) / 1.8) == pytest.approx(1.0, abs=1e-12)
    assert cm.masco_factor(T21) == pytest.approx(1.0, abs=2e-3)         # 21.1 C is 69.98 F
    assert cm.masco_factor(10.0) == pytest.approx(1.72, abs=0.02)
    assert cm.masco_factor(32.0) == pytest.approx(0.45, abs=0.02)
    assert cm.scaled_start_age_s(600.0, 10.0) == pytest.approx(600.0 * cm.masco_factor(10.0))
    c = cm.Concrete()
    s = cm.masco_factor(27.0)                                           # same stage at another temperature
    assert float(c.tau_bulk(100 * MIN * s, 27.0)) == pytest.approx(float(c.tau_bulk(100 * MIN, (70 - 32) / 1.8)))


def test_concrete_tau0_and_set_anchors():
    assert cm.TAU_B0 == pytest.approx(1566.0, abs=0.5)
    assert cm.SET_M == pytest.approx(5.13, abs=0.01)
    c = cm.Concrete()
    t_is, t_fs = c.initial_set_time(T21), c.final_set_time(T21)
    assert float(c.tau_bulk(t_is, T21)) == pytest.approx(3.45e6 / 9, rel=1e-6)
    assert float(c.tau_bulk(t_fs, T21)) == pytest.approx(27.6e6 / 9, rel=1e-6)
    s = cm.masco_factor(T21)
    assert t_is == pytest.approx(4 * 3600 * s, rel=0.02)
    assert t_fs == pytest.approx(6 * 3600 * s, rel=0.01)
    assert float(c.tau_bulk(6 * 3600 * s, T21)) == pytest.approx(27.6e6 / 9, rel=5e-3)   # t_FS anchor, offsets ~0.2 %


def test_concrete_bleed_triangle():
    c = cm.Concrete()
    te = c.bleed_end_time(T21)
    assert float(c.tau_bulk(te, T21)) == pytest.approx(1.0e4, rel=1e-6)
    assert 105 * MIN < te < 115 * MIN
    t = np.linspace(0, te, 200001)
    r = c.bleed_rate(t, T21)
    assert np.trapezoid(r, t) == pytest.approx(0.15 / 3600 * te, rel=1e-6)
    assert t[np.argmax(r)] == pytest.approx(te / 3, rel=1e-4)
    assert r.max() == pytest.approx(2 * 0.15 / 3600, rel=1e-3)
    assert c.bleed_rate(te * 1.2, T21) == 0.0 and c.bleed_rate(0.0, T21) == 0.0


def test_repair_mortar_thirty_kpa_window():
    rm = cm.RepairMortar()
    t0 = rm.time_to_tau(30e3 / 0.5, T21) / MIN       # D = 0: tau_s = 0.5 tau_b
    t1 = rm.time_to_tau(30e3 / 1.0, T21) / MIN       # D = 1: tau_s = tau_b
    assert t0 == pytest.approx(34.5, abs=0.15)
    assert t1 == pytest.approx(29.9, abs=0.15)
    assert float(rm.tau_surf(rm.tau_bulk(t0 * MIN, T21), 0.0)) == pytest.approx(30e3, rel=1e-6)
    assert rm.bleed_mean == 0.05 and rm.initial_set_time(T21) < cm.Concrete().initial_set_time(T21)


def test_masonry_mortar_courses():
    mm = cm.MasonryMortar()
    ages = np.array([45, 37, 28, 20]) * MIN * cm.masco_factor(T21)
    kpa = mm.tau_bulk(ages, T21) / 1e3
    np.testing.assert_allclose(kpa, [18.0, 10.4, 5.1, 2.5], atol=0.15)
    assert float(mm.tau_surf(1000.0, 1.0)) == pytest.approx(1300.0)
    assert mm.time_to_tau(2e4, T21) / MIN == pytest.approx(46.7, abs=0.3)
    assert float(mm.bleed_rate(100.0)) == 0.0 and math.isinf(mm.initial_set_time())


def test_evaporation():
    ev = cm.air_evaporation_kg_m2_h("still", 21.1)
    assert 0.1 < ev < 1.0
    assert cm.air_evaporation_kg_m2_h("fan") > ev


def test_per_pass_closed_forms_and_composition():
    assert cm.embed_update(0.3, 1000.0, 1.0) == pytest.approx(0.3 * math.exp(-8.3))
    assert cm.embed_update(0.3, 1e-3, 1.0) == pytest.approx(0.3 * math.exp(-0.83))      # pressure clip 0.1
    assert math.log(cm.embed_update(0.3, 1e9, 1.0) / 0.3) == pytest.approx(-8.3 * 20, rel=1e-9, abs=0)   # clip 20
    assert cm.embed_update(cm.embed_update(1.0, 1000.0, 1.0), 1000.0, 2.0) == pytest.approx(
        cm.embed_update(1.0, 1000.0, 3.0))
    a2 = cm.texture_update(cm.texture_update(8e-4, 3e-5, 1.0, 1.0), 3e-5, 1.0, 1.0)
    assert a2 == pytest.approx(cm.texture_update(8e-4, 3e-5, 2.0, 1.0))
    assert cm.texture_update(8e-4, 3e-5, 1.0, 1.0) == pytest.approx(3e-5 + 7.7e-4 * math.exp(-2.2))
    # Wt = 0.3 * 11200 Pa * 0.01 m = 33.6 J/m2 -> exp(-0.3)
    assert cm.densify_update(0.1, 11200.0, 0.01, 0.3, 1.0) == pytest.approx(1 - 0.9 * math.exp(-0.3))
    assert cm.seal_update(0.0, 0.1, 1.0, 1.0, 5000.0) == pytest.approx(1 - math.exp(-0.1 * 0.6 * 5.0 / 0.1))
    assert cm.seal_update(0.0, 0.1, 0.0, 1.0, 5000.0) == 0.0


def test_interface_factor_rule():
    assert cm.interface_factor(300.0, 0.0) == pytest.approx(0.05)
    assert cm.interface_factor(30e3, 0.0) == pytest.approx(0.30)
    assert cm.interface_factor(3e3, 0.0) == pytest.approx(0.05 + 0.25 * 0.5)
    assert cm.interface_factor(3e3, 3e-5) == pytest.approx(0.05)


def test_densification_monotone_bounded_and_composes():
    D = 0.1
    seq = [D]
    for _ in range(40):
        D = float(cm.densify_update(D, 8e3, 0.05, 0.2, 1.0))
        seq.append(D)
    assert all(0.0 <= x <= 1.0 for x in seq) and all(b >= a for a, b in zip(seq, seq[1:]))
    one = cm.densify_update(0.1, 8e3, 0.4, 0.2, 1.0)
    two = cm.densify_update(cm.densify_update(0.1, 8e3, 0.2, 0.2, 1.0), 8e3, 0.2, 0.2, 1.0)
    assert one == pytest.approx(two)


def test_compaction_dose_rule():
    tau = 10e3
    p = 20.0 / 1e-4
    dose = p / (p + tau) * 0.3 * 0.05 / 0.012
    assert cm.compact_update(0.0, p, 0.05, tau, 0.3) == pytest.approx(1 - math.exp(-dose))
    assert cm.compact_update(0.5, 0.0, 0.05, tau, 0.3) == pytest.approx(0.5)
    c = cm.compact_update(np.linspace(0, 1, 5), p, 0.05, tau, 0.3)
    assert np.all(np.diff(c) >= -1e-12) and c.max() <= 1.0


def test_blister_lift_and_bearing():
    q = np.zeros((40, 40))
    q[20, 20] = 3e-4
    lift = cm.blister_lift(q, 0.0015)                                  # k = 5, window 11 cells
    assert lift.max() == pytest.approx(4e-4 / 121)
    assert lift.sum() == pytest.approx(4e-4)                            # blur conserves volume
    q[:] = 1e-3
    assert cm.blister_lift(q, 0.002).max() == pytest.approx(3e-3)       # cap
    assert cm.blister_lift(np.full((8, 8), 1e-4), 0.002).max() == 0.0
    tb = 1e4
    assert cm.bearing_pressure(tb, tb, 0.05) == pytest.approx(cm.NC * tb)
    assert cm.bearing_pressure(tb, tb, 0.05, length=0.05, depth=0.125) == pytest.approx(
        cm.NC * 1.2 * 1.5 * tb)                                         # 9.25 tau_b square punch 2.5 widths deep
    thin = cm.bearing_pressure(tb, 0.3 * tb, 0.05, ts=0.001)           # squeeze term 0.3 tb (2 + 12.5)
    assert thin == pytest.approx(0.3 * tb * 14.5)
    assert thin < cm.NC * tb * 1.01 or thin == pytest.approx(cm.NC * tb)


def _grid():
    rng = np.random.default_rng(0)
    f = lambda lo, hi, n=64: rng.uniform(lo, hi, n).astype(np.float32).astype(np.float64)
    return dict(age=f(0, 20000), tau=f(500, 5e4), D=f(0, 0.99), p=f(50, 8000), ds=f(0, 0.2), passes=f(0, 1.5),
                m_i=f(0.05, 0.3), w=np.where(rng.random(64) < 0.3, 3e-5, 0.0).astype(np.float32).astype(np.float64),
                a=f(3e-5, 8e-4))


TAU0 = wp.constant(cm.TAU_B0)
A_THIX = wp.constant(cm.A_THIX)
TAU_IS = wp.constant(cm.TAU_IS)
T_IS = wp.constant(cm.T_IS)
SET_M = wp.constant(cm.SET_M)


@wp.kernel
def _all_laws(age: wp.array(dtype=float), tau: wp.array(dtype=float), D: wp.array(dtype=float),
              p: wp.array(dtype=float), ds: wp.array(dtype=float), passes: wp.array(dtype=float),
              m_i: wp.array(dtype=float), w: wp.array(dtype=float), a: wp.array(dtype=float),
              te: float, out: wp.array2d(dtype=float)):
    i = wp.tid()
    s = float(1.3)
    tb = cm.wp_tau_bulk_setting(age[i], s, TAU0, A_THIX, TAU_IS, T_IS, SET_M)
    out[0, i] = tb
    out[1, i] = cm.wp_tau_surf_setting(tb, D[i], 0.15, 0.30)
    out[2, i] = cm.wp_tau_bulk_mortar(age[i], s, 1000.0, 8000.0, 2100.0, 3.0)
    out[3, i] = cm.wp_tau_surf_mortar(tau[i], D[i], 0.30)
    out[4, i] = cm.wp_bleed_rate(age[i], te, 0.15)
    out[5, i] = cm.wp_embed_update(D[i], p[i], passes[i])
    out[6, i] = cm.wp_interface_factor(tau[i], w[i])
    out[7, i] = cm.wp_texture_update(a[i], 3.0e-5, passes[i], p[i], tau[i], m_i[i])
    out[8, i] = cm.wp_densify_update(D[i], tau[i], ds[i], m_i[i], 0.6)
    out[9, i] = cm.wp_seal_update(D[i], ds[i], 0.3, passes[i], p[i])
    out[10, i] = cm.wp_compact_update(D[i], p[i], ds[i], tau[i], m_i[i])


def test_warp_twins_match_numpy():
    g = _grid()
    n = len(g["age"])
    dev = "cpu"
    arr = {k: wp.array(v.astype(np.float32), dtype=float, device=dev) for k, v in g.items()}
    out = wp.zeros((11, n), dtype=float, device=dev)
    te = 6000.0
    wp.launch(_all_laws, dim=n, inputs=[arr["age"], arr["tau"], arr["D"], arr["p"], arr["ds"], arr["passes"],
                                        arr["m_i"], arr["w"], arr["a"], te, out], device=dev)
    r = out.numpy().astype(np.float64)
    c = cm.Concrete()
    s = 1.3
    tb = c.tau0 + c.a_thix * g["age"] / s + c.tau_is * (g["age"] / (c.t_is * s)) ** c.m
    peak = 2 * 0.15 / 3600
    up = peak * g["age"] / (te / 3)
    down = peak * (te - g["age"]) / (2 * te / 3)
    bleed = np.clip(np.where(g["age"] < te / 3, up, down), 0, None)
    want = [tb, (0.15 + 0.15 * g["D"]) * tb,
            1000.0 + 8000.0 * (g["age"] / s / 2100.0) ** 3,
            cm.MasonryMortar().tau_surf(g["tau"], g["D"]), bleed,
            cm.embed_update(g["D"], g["p"], g["passes"]), cm.interface_factor(g["tau"], g["w"]),
            cm.texture_update(g["a"], 3e-5, g["passes"], cm.texture_yield(g["p"], g["tau"], g["m_i"])),
            cm.densify_update(g["D"], g["tau"], g["ds"], g["m_i"], 0.6),
            cm.seal_update(g["D"], g["ds"], 0.3, g["passes"], g["p"]),
            np.array([cm.compact_update(g["D"][i], g["p"][i], g["ds"][i], g["tau"][i], g["m_i"][i]) for i in range(n)])]
    for k, ref in enumerate(want):
        np.testing.assert_allclose(r[k], ref, rtol=1e-6, atol=1e-12, err_msg=f"law {k}")


def test_the_concrete_setting_curve_is_surface_flows_setting_law():
    """One law: the concrete's tau_b(age, T) equals surface flow's Setting (process_specs) with s_t = MASCO(T)."""
    from alloy.plugins.surface_flow.process_specs import Setting

    mat = cm.Concrete()
    ages = np.linspace(0.0, 6 * 3600.0, 50)
    for temp_c in (10.0, 21.1, 32.0):
        law = Setting(tau_0=mat.tau0, a_thix=mat.a_thix, tau_is=mat.tau_is, t_is=mat.t_is, m=mat.m,
                      s_t=float(cm.masco_factor(temp_c)))
        assert np.allclose(mat.tau_bulk(ages, temp_c), law.yield_stress(ages), rtol=1e-12, atol=0)


def test_the_sealant_and_the_worked_water_rise():
    """Their sealant (M:126-142): 600 Pa at every age, the surface 3x once skinned at 2 h x MASCO (20 min for the fast
    grade); no bleed, no initial set. A worked-in film w raises the top 6 mm's w/c by 1000 w / (350 x 0.006)."""
    s = cm.Sealant()
    assert (s.rho, s.tau0, s.mu) == (1400.0, 600.0, 50.0)
    skin = s.skin_time(21.1)
    assert skin == pytest.approx(7200.0 * cm.masco_factor(21.1), rel=1e-12)
    fast = cm.Sealant(skin_s=cm.FAST_SKIN).skin_time(21.1)
    assert fast == pytest.approx(1200.0 * cm.masco_factor(21.1), rel=1e-12)
    age = np.array([0.0, skin - 1.0, skin, 5 * skin])
    tb = s.tau_bulk(age)
    assert np.all(tb == 600.0)
    assert np.allclose(s.tau_surf(tb, age), [600.0, 600.0, 1800.0, 1800.0], rtol=0, atol=0)
    assert np.all(s.bleed_rate(age) == 0.0) and math.isinf(s.initial_set_time())
    assert cm.dwc_rise(2.1e-4) == pytest.approx(0.1)                   # 0.21 kg/m² into 2.1 kg/m² of cement


@wp.kernel
def _embed(E: wp.array(dtype=float), p: wp.array(dtype=float), n: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = cm.wp_embed_update(E[i], p[i], n[i])


def test_the_warp_embed_law_caps_the_pressure_at_20_kpa():
    """Above 20 kPa the pressure factor stops growing (clip at 20), on the device as in numpy: small passes keep both
    sides off the underflow floor, so a missing cap would show (exp(-8.3 x 1000 x 0.01) vs exp(-8.3 x 20 x 0.01))."""
    p = np.array([2.0e4, 5.0e4, 2.0e5, 1.0e6], np.float32)
    E = np.full(4, 0.3, np.float32)
    n = np.full(4, 0.01, np.float32)
    out = wp.zeros(4, dtype=float, device="cpu")
    wp.launch(_embed, dim=4, inputs=[wp.array(E, dtype=float, device="cpu"), wp.array(p, dtype=float, device="cpu"),
                                     wp.array(n, dtype=float, device="cpu"), out], device="cpu")
    want = 0.3 * math.exp(-8.3 * 20 * 0.01)
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-5, atol=0)
    np.testing.assert_allclose(cm.embed_update(E.astype(float), p.astype(float), n.astype(float)), want, rtol=1e-6,
                               atol=0)
