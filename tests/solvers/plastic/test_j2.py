# SPDX-License-Identifier: Apache-2.0
"""J2 return maps and sections (alloy/solvers/plastic/j2.py) against the closed forms (closed_forms.py): the 1-D cycle,
the tension-shear fibre, the volume return, and the sheet and round-wire sections with their spring-back (SCR-WP7-15,
P0; CPU, light)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.solvers.plastic import closed_forms as C
from alloy.solvers.plastic import j2

pytestmark = pytest.mark.law
DEV = "cpu"


@wp.kernel
def _path_1d(eps: wp.array(dtype=float), E: float, sy0: float, hmod: float, out: wp.array2d(dtype=float)):
    ep = float(0.0)
    eb = float(0.0)
    for i in range(eps.shape[0]):
        s, ep1, eb1 = j2.return_1d(eps[i], ep, eb, E, sy0, hmod)
        ep = ep1
        eb = eb1
        out[i, 0] = s
        out[i, 1] = ep
        out[i, 2] = eb


@wp.kernel
def _fibre(st: wp.array2d(dtype=float), E: float, G: float, sy0: float, hmod: float, out: wp.array2d(dtype=float)):
    i = wp.tid()
    s, t, ep, gp, eb = j2.return_fibre(st[i, 0], st[i, 1], st[i, 2], st[i, 3], st[i, 4], E, G, sy0, hmod)
    out[i, 0] = s
    out[i, 1] = t
    out[i, 2] = ep
    out[i, 3] = gp
    out[i, 4] = eb


@wp.kernel
def _vol(F: wp.array(dtype=wp.mat33), fp: wp.array(dtype=wp.mat33), eb: wp.array(dtype=float), mu: float, sy0: float,
         hmod: float):
    i = wp.tid()
    f, e = j2.return_3d(F[i], fp[i], eb[i], mu, sy0, hmod)
    fp[i] = f
    eb[i] = e


@wp.kernel
def _sheet(eps: float, kap: float, t: float, Ep: float, syp: float, hp: float, q: j2.Quad,
           ep: wp.array2d(dtype=float), eb: wp.array2d(dtype=float), out: wp.array(dtype=wp.vec2)):
    e = wp.tid()
    out[e] = j2.sheet_section(e, eps, kap, t, Ep, syp, hp, q, ep, eb)


@wp.kernel
def _rod(eps: float, k1: float, k2: float, tw: float, R: float, E: float, G: float, sy0: float, hmod: float,
         q: j2.Quad, ep: wp.array2d(dtype=float), gp: wp.array2d(dtype=float), eb: wp.array2d(dtype=float),
         out: wp.array(dtype=wp.vec4)):
    e = wp.tid()
    out[e] = j2.rod_section(e, eps, k1, k2, tw, R, E, G, sy0, hmod, q, ep, gp, eb)


def _ref_1d(eps: np.ndarray, E: float, sy0: float, hmod: float) -> np.ndarray:
    ep = eb = 0.0
    out = []
    for e in eps:
        s = E * (e - ep)
        f = abs(s) - (sy0 + hmod * eb)
        if f > 0.0:
            dl = f / (E + hmod)
            ep += math.copysign(dl, s)
            eb += dl
            s -= math.copysign(E * dl, s)
        out.append((s, ep, eb))
    return np.array(out)


@pytest.mark.parametrize("hmod", [0.0, 0.05])
def test_1d_cycle_yields_unloads_elastically_and_reverse_yields_isotropically(hmod):
    E, sy0 = 1.0, 1.0e-3
    ey = sy0 / E
    path = np.concatenate([np.linspace(0, 4 * ey, 101), np.linspace(4 * ey, -4 * ey, 201)[1:],
                           np.linspace(-4 * ey, 0, 101)[1:]])
    out = wp.zeros((len(path), 3), dtype=float, device=DEV)
    wp.launch(_path_1d, dim=1, inputs=[wp.array(path, dtype=float, device=DEV), E, sy0, hmod * E, out], device=DEV)
    got, ref = out.numpy(), _ref_1d(path, E, sy0, hmod * E)
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6 * sy0)
    assert got[100, 0] == pytest.approx(float(C.stress_1d(4 * ey, E, sy0, hmod * E)), rel=1e-5)   # the peak
    assert got[101, 0] - got[100, 0] == pytest.approx(-E * (path[100] - path[101]), rel=1e-4)   # unloads with slope E
    first_reverse = np.argmax(got[101:, 2] > got[100, 2] + 1e-9 * ey) + 101                    # reverse yield
    assert got[first_reverse - 1, 0] == pytest.approx(-(sy0 + hmod * E * got[100, 2]), rel=2e-2)


def test_fibre_return_lands_on_the_yield_surface_with_normal_flow():
    E, nu, sy0, hmod = 200e9, 0.3, 250e6, 1e9
    G = E / (2 * (1 + nu))
    rng = np.random.default_rng(0)
    n = 400
    st = np.zeros((n, 5))
    st[:, 0] = rng.uniform(-4, 4, n) * sy0 / E
    st[:, 1] = rng.uniform(-4, 4, n) * sy0 / G
    st[:, 2] = rng.uniform(-1, 1, n) * sy0 / E
    st[:, 3] = rng.uniform(-1, 1, n) * sy0 / G
    st[:, 4] = rng.uniform(0, 2, n) * sy0 / E
    out = wp.zeros((n, 5), dtype=float, device=DEV)
    wp.launch(_fibre, dim=n, inputs=[wp.array(st, dtype=float, device=DEV), E, G, sy0, hmod, out], device=DEV)
    s, t, ep, gp, eb = out.numpy().T.astype(float)
    trial = np.hypot(E * (st[:, 0] - st[:, 2]), math.sqrt(3) * G * (st[:, 1] - st[:, 3]))
    yielded = trial > sy0 + hmod * st[:, 4]
    assert 100 < yielded.sum() < n
    assert np.allclose(s, E * (st[:, 0] - ep), rtol=1e-4, atol=1e-4 * sy0)
    assert np.allclose(t, G * (st[:, 1] - gp), rtol=1e-4, atol=1e-4 * sy0)
    y = sy0 + hmod * eb
    assert np.allclose(np.hypot(s, math.sqrt(3) * t)[yielded], y[yielded], rtol=1e-5)
    eb0 = st[:, 4].astype(np.float32)                                    # the state as the kernel read it
    assert np.all(eb >= eb0) and np.array_equal(eb[~yielded], eb0[~yielded])
    dl = (eb - eb0)[yielded]                                       # normality: dε_p = dλ σ/q, dγ_p = dλ 3τ/q
    assert np.allclose((ep - st[:, 2])[yielded], dl * s[yielded] / y[yielded], rtol=1e-3, atol=1e-6 * sy0 / E)
    assert np.allclose((gp - st[:, 3])[yielded], dl * 3 * t[yielded] / y[yielded], rtol=1e-3, atol=1e-6 * sy0 / G)


def test_volume_return_follows_the_uniaxial_law_is_isochoric_and_objective():
    mu, sy0, hmod = 5.0e4, 750.0, 2.0e3
    lam = np.linspace(1.0, 1.06, 121)
    rot = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    rot *= np.sign(np.linalg.det(rot))
    fp = wp.array(np.repeat(np.eye(3)[None], 2, axis=0), dtype=wp.mat33, device=DEV)
    eb = wp.zeros(2, dtype=float, device=DEV)
    for x in lam:
        F = np.diag([x, x ** -0.5, x ** -0.5])
        wp.launch(_vol, dim=2, inputs=[wp.array(np.stack([F, rot @ F]), dtype=wp.mat33, device=DEV), fp, eb, mu, sy0,
                                       hmod], device=DEV)
    want = max(0.0, (3 * mu * math.log(lam[-1]) - sy0) / (3 * mu + hmod))   # σ = 3μ(ε - ε_p) = σ_y0 + H ε̄p
    assert eb.numpy() == pytest.approx([want, want], rel=2e-3)          # float32 SVD over 120 increments: 0.1 %
    assert np.linalg.det(fp.numpy()) == pytest.approx([1.0, 1.0], abs=1e-5)
    vol = wp.zeros(1, dtype=float, device=DEV)                              # pure volume change never yields
    wp.launch(_vol, dim=1, inputs=[wp.array([np.eye(3) * 1.2], dtype=wp.mat33, device=DEV),
                                   wp.array([np.eye(3)], dtype=wp.mat33, device=DEV), vol, mu, sy0, hmod], device=DEV)
    assert vol.numpy()[0] == 0.0
    flat = wp.array([np.eye(3)], dtype=wp.mat33, device=DEV)                 # inverted: left alone (WP9's review)
    wp.launch(_vol, dim=1, inputs=[wp.array([np.diag([-1.2, 1.0, 1.0])], dtype=wp.mat33, device=DEV), flat, vol, mu,
                                   sy0, hmod], device=DEV)
    assert vol.numpy()[0] == 0.0 and np.array_equal(flat.numpy()[0], np.eye(3))


def _sheet_run(kappas, t, Ep, syp, hp):
    q = j2.sheet_quad(DEV)
    ep = wp.zeros((1, j2.fibres(q)), dtype=float, device=DEV)
    eb = wp.zeros_like(ep)
    out = wp.zeros(1, dtype=wp.vec2, device=DEV)
    moments = []
    for k in kappas:
        wp.launch(_sheet, dim=1, inputs=[0.0, float(k), t, Ep, syp, hp, q, ep, eb, out], device=DEV)
        moments.append(float(out.numpy()[0][1]))
    return np.array(moments), eb


@pytest.mark.parametrize("hmod", [0.0, 1.0e9])
def test_sheet_section_moment_onset_and_gardiner_springback(hmod):
    E, nu, sy0, t = 200e9, 0.3, 250e6, 1.0e-3
    Ep, syp, hp = j2.plane_strain(E, nu, sy0, hmod)
    ky = 2 * syp / (Ep * t)
    kmax = ky / math.sqrt(3 * 0.02)                                       # M = 0.98 M_p when perfectly plastic
    kappas = np.concatenate([[0.999 * ky, 1.001 * ky], np.linspace(1.01 * ky, kmax, 200)])
    M, eb = _sheet_run(kappas, t, Ep, syp, hp)
    ref = np.array([C.rect_moment(k, t, Ep, syp, hp) for k in kappas])
    assert M[0] == pytest.approx(ref[0], rel=1e-5)                        # elastic below M_y, exactly
    assert np.abs(M / ref - 1).max() < 0.004                               # j2 doc: within 0.4 %
    m1, _ = _sheet_run(kappas[:1], t, Ep, syp, hp)
    _, eb_on = _sheet_run(kappas[:2], t, Ep, syp, hp)
    _, eb_off = _sheet_run(kappas[:1], t, Ep, syp, hp)
    assert eb_off.numpy().max() == 0.0 and eb_on.numpy().max() > 0.0      # the onset is M_y's curvature
    k0 = kappas[-1]                                                        # release from 0.98 M_p
    kf = C.springback_curvature(k0, M[-1], Ep * t ** 3 / 12)
    if hmod == 0.0:
        assert k0 / kf == pytest.approx(C.gardiner_ratio(1 / k0, t, Ep, syp) ** -1 * 1.0, rel=5e-3)
    before = eb.numpy().copy()
    q = j2.sheet_quad(DEV)
    out = wp.zeros(1, dtype=wp.vec2, device=DEV)
    ep2 = wp.zeros((1, j2.fibres(q)), dtype=float, device=DEV)
    eb2 = wp.zeros_like(ep2)
    for k in kappas:
        wp.launch(_sheet, dim=1, inputs=[0.0, float(k), t, Ep, syp, hp, q, ep2, eb2, out], device=DEV)
    wp.launch(_sheet, dim=1, inputs=[0.0, float(kf), t, Ep, syp, hp, q, ep2, eb2, out], device=DEV)
    assert abs(float(out.numpy()[0][1])) < 1e-4 * syp * t ** 2           # released: no moment left
    assert np.array_equal(eb2.numpy(), before)                              # and unloading stayed elastic


def test_round_section_bends_in_any_direction_and_twists_within_the_doc_bounds():
    E, nu, sy0, R = 200e9, 0.3, 250e6, 1.0e-3
    G = E / (2 * (1 + nu))
    from scipy.optimize import brentq

    ky = sy0 / (E * R)
    kmax = brentq(lambda k: C.round_moment(k, R, E, sy0) - 0.98 * 4 * sy0 * R ** 3 / 3, ky, 1e3 * ky)
    kappas = np.linspace(ky, kmax, 120)
    q = j2.rod_quad(DEV)
    worst = skew = 0.0
    for ang in np.linspace(0.0, 2 * math.pi / j2.N_T, 4):
        ep = wp.zeros((1, j2.fibres(q)), dtype=float, device=DEV)
        gp, eb = wp.zeros_like(ep), wp.zeros_like(ep)
        out = wp.zeros(1, dtype=wp.vec4, device=DEV)
        for k in kappas:
            wp.launch(_rod, dim=1, inputs=[0.0, float(k * math.cos(ang)), float(k * math.sin(ang)), 0.0, R, E, G,
                                           sy0, 0.0, q, ep, gp, eb, out], device=DEV)
            n, m1, m2, _ = out.numpy()[0]
            worst = max(worst, abs(math.hypot(m1, m2) / C.round_moment(k, R, E, sy0) - 1))
            skew = max(skew, abs(m1 * math.sin(ang) - m2 * math.cos(ang)) / math.hypot(m1, m2))
    assert skew < 0.01                                                     # j2 doc: M within 0.6° of κ's direction
    assert worst < 0.0035                                                  # j2 doc: within 0.35 %
    ty = sy0 / math.sqrt(3)
    pmax = brentq(lambda p: C.round_torque(p, R, G, sy0) - 0.98 * 2 * math.pi * R ** 3 * ty / 3, ty / (G * R), 1e3)
    ep = wp.zeros((1, j2.fibres(q)), dtype=float, device=DEV)
    gp, eb = wp.zeros_like(ep), wp.zeros_like(ep)
    out = wp.zeros(1, dtype=wp.vec4, device=DEV)
    errs = []
    for p in np.linspace(ty / (G * R), pmax, 200):
        wp.launch(_rod, dim=1, inputs=[0.0, 0.0, 0.0, float(p), R, E, G, sy0, 0.0, q, ep, gp, eb, out], device=DEV)
        errs.append(abs(out.numpy()[0][3] / C.round_torque(p, R, G, sy0) - 1))
    assert max(errs) < 0.0035                                              # j2 doc: within 0.35 %
