# SPDX-License-Identifier: Apache-2.0
"""CT-5: centred 12.0 mm peg in a 12.2 mm bore: reported gap 0.100 ± 0.005 mm; play angle in 36 directions vs the
closed form ``d·cosθ + l·sinθ = D`` within 5 % (kinematic, no dynamics).

Play angle search: worlds = (tilt θ, shift s) grid; the peg axis passes through the bore-top point shifted by ``s``
along direction φ, its bottom end is ``l`` = 10 mm below that plane along the axis. A pose is free when every reported
contact separation is ≥ -1 µm. θ_max(φ) = largest θ with a free s; a second, finer grid refines it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.contact.bias import _min_separation
from alloy.contact.pipeline import AlloyContactLayer
from alloy.testing.contact import PegHoleSpec, peg_hole_scene

from peg_harness import record

pytestmark = pytest.mark.gpu
ENGAGE = 10e-3
N_T, N_S = 41, 41


def closed_form_angle(d: float, D: float, l: float) -> float:
    """Solve d·cosθ + l·sinθ = D for θ [rad] (bisection on [0, π/4])."""
    lo, hi = 0.0, math.pi / 4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if d * math.cos(mid) + l * math.sin(mid) < D else (lo, mid)
    return 0.5 * (lo + hi)


def _quat_tilt(phi: float, theta: float) -> np.ndarray:
    ax = np.array([-math.sin(phi), math.cos(phi), 0.0])           # tilting toward +phi
    return np.r_[ax * math.sin(theta / 2), math.cos(theta / 2)]


def _poses(g: PegHoleSpec, phi: float, thetas: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = []
    for th in thetas:
        q = _quat_tilt(phi, th)
        a = np.array([math.cos(phi) * math.sin(th), math.sin(phi) * math.sin(th), math.cos(th)])
        for s in shifts:
            top = np.array([s * math.cos(phi), s * math.sin(phi), g.height])
            center = top - ENGAGE * a + 0.5 * g.peg_len * a
            out.append(np.r_[center, q])
    return np.asarray(out, np.float32)


def _free(layer, scene, poses: np.ndarray) -> np.ndarray:
    st = scene.model.state()
    q = st.body_q.numpy()
    q[scene.peg_bodies] = poses
    st.body_q.assign(q)
    layer.collide(st, layer.contacts(), dt=1e-3)
    sep = _min_separation(scene.model, st, layer.contacts(), scene.model.world_count)
    return sep >= -1e-6


def _theta_max(layer, scene, phi: float, thetas: np.ndarray, shifts: np.ndarray) -> float:
    free = _free(layer, scene, _poses(scene.geometry, phi, thetas, shifts)).reshape(len(thetas), len(shifts))
    ok = np.nonzero(free.any(axis=1))[0]
    return float(thetas[ok.max()]) if len(ok) else 0.0


def test_ct5_gap_and_play_angle():
    g = PegHoleSpec()
    scene = peg_hole_scene(N_T * N_S, device="cuda:0", mode="sdf", start_gap=5e-3)
    layer = AlloyContactLayer(scene.model, [scene.spec], native="newton", precision_contacts_per_world=64)
    # (a) reported gap of the centred peg (first world; every pose is the centred start pose)
    st = scene.model.state()
    layer.collide(st, layer.contacts(), dt=1e-3)
    gap = float(_min_separation(scene.model, st, layer.contacts(), scene.model.world_count)[0])
    # (b) play angle
    theta_c = closed_form_angle(g.peg_d, g.bore_d, ENGAGE)
    shifts = np.linspace(-g.radial_clearance * 1.6, g.radial_clearance * 1.6, N_S)
    ratios = []
    for k in range(36):
        phi = 2 * math.pi * k / 36
        coarse = _theta_max(layer, scene, phi, np.linspace(0.0, 1.5 * theta_c, N_T), shifts)
        step = 1.5 * theta_c / (N_T - 1)
        fine = _theta_max(layer, scene, phi, np.linspace(coarse, coarse + step, N_T), shifts)
        ratios.append(fine / theta_c)
    ratios = np.asarray(ratios)
    row = dict(metric="CT-5", gap_m=gap, theta_closed_deg=math.degrees(theta_c), engage_m=ENGAGE,
               ratio_min=float(ratios.min()), ratio_max=float(ratios.max()), ratio_mean=float(ratios.mean()),
               err_max=float(np.abs(ratios - 1).max()))
    record("ct5", row)
    assert abs(gap - 1e-4) <= 5e-6, row
    assert np.abs(ratios - 1.0).max() <= 0.05, row
