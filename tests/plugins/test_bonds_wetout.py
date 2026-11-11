# SPDX-License-Identifier: Apache-2.0
"""Wet-out of the Bonds adhesive (alloy/plugins/bonds/wetout.py, SCR-WP8-11 Part B1): the standard roller card and
its calibration, laws W-1 (the exact ODE per step) and W-2 (rises with pressure, none below p_form) on the update
kernel, W-3 (one roller pass gives the Hertz contact's wet-out whatever the cell size) on the kernel and, slow, on the
roller rig. CPU."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.bonds import kernels as K
from alloy.plugins.bonds.adhesives import adhesive
from alloy.plugins.bonds.laws import ROLL_SPEED, bonds_laws
from alloy.plugins.bonds.plugin import Bonds
from alloy.plugins.bonds.wetout import (P_REF, STANDARD_ROLLER, STD_PASSES, STD_SPEED, W_STD, Roller, exposure,
                                        rollers_struct, wet_after, wet_tau)

P_FORM = adhesive("3m_7818/stainless_steel").p_form
TAU = wet_tau(P_FORM)
DT = 1.0e-3


def test_the_standard_roller_is_d3330s_and_calibrates_to_w_std():
    """The ASTM D3330 roller (2.04 kg, 95.25 mm, 44.45 mm wide, Shore A 80): 450 N/m, a 1.48 mm Hertz half-width and
    a 194 kPa peak; two passes at the standard speed give W_STD exactly."""
    r = STANDARD_ROLLER
    q = r.line_load()
    assert q == pytest.approx(450.2, rel=1e-3)
    assert r.half_width(q) == pytest.approx(1.476e-3, rel=5e-3)
    assert float(r.pressure(0.0, q)) == pytest.approx(1.94e5, rel=5e-3)
    e = STD_PASSES * exposure(r, q, P_FORM, STD_SPEED)
    assert 1.0 - math.exp(-e / TAU) == pytest.approx(W_STD, rel=1e-12)
    b = r.half_width(q)                                  # p >= p_form over |s| <= b·sqrt(1 - (p_form/p_max)²)
    a = math.sqrt(1.0 - (P_FORM / float(r.pressure(0.0, q))) ** 2)
    closed = float(r.pressure(0.0, q)) / P_REF * b / STD_SPEED * (a * math.sqrt(1 - a * a) + math.asin(a))
    assert exposure(r, q, P_FORM, STD_SPEED) == pytest.approx(closed, rel=1e-4)    # M = 1's closed form


def _press(n: int, worlds: int = 1, roller: Roller | None = None, shape_xform: np.ndarray | None = None) -> K.Press:
    pr = K.Press()
    load = wp.array(np.full((worlds, 1), roller.line_load() if roller else 0.0), dtype=float, device="cpu")
    cards = (roller,) if roller else ()
    pr.rollers = rollers_struct(cards, np.zeros((worlds, 1), np.int64), 1, load, wp.zeros_like(load), 2.0e-3, "cpu")
    pr.world = wp.zeros(n, dtype=wp.int32, device="cpu")
    pr.shape_xform = wp.array(shape_xform if shape_xform is not None else np.zeros((1, 7), np.float32),
                              dtype=wp.transform, device="cpu")
    return pr


def _surface() -> K.Surface:
    sf = K.Surface()
    for name, v in (("substrate", 1), ("p_form", P_FORM), ("dwell", 0.05), ("ke", 2e4), ("margin", 0.0),
                    ("tau_w", TAU)):
        setattr(sf, name, wp.array([v], dtype=wp.int32 if name == "substrate" else float, device="cpu"))
    return sf


def _bonded(state: np.ndarray) -> K.Bonded:
    n = len(state)
    bd = K.Bonded()
    bd.state = wp.array(state.astype(np.int32), dtype=wp.int32, device="cpu")
    bd.shape = wp.zeros(n, dtype=wp.int32, device="cpu")
    bd.world = wp.zeros(n, dtype=wp.int32, device="cpu")
    bd.base = wp.zeros(1, dtype=wp.int32, device="cpu")
    bd.anchor = wp.zeros(n, dtype=wp.vec3, device="cpu")
    bd.k = wp.zeros(n, dtype=float, device="cpu")
    return bd


def _step(key: np.ndarray, state: np.ndarray, wet: wp.array, x: np.ndarray, pr: K.Press, steps: int,
          move=None) -> np.ndarray:
    n = len(key)
    k = wp.array(key.astype(np.int64), dtype=wp.int64, device="cpu")
    sf, bd = _surface(), _bonded(state)
    q = wp.array(x.astype(np.float32), dtype=wp.vec3, device="cpu")
    normal = wp.array(np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cpu")
    no_body = wp.zeros(1, dtype=wp.transform, device="cpu")
    shape_body = wp.array([-1], dtype=wp.int32, device="cpu")
    radius = wp.full(n, 5.0e-4, dtype=float, device="cpu")
    for j in range(steps):
        if move is not None:
            pr.shape_xform.assign(move(j))
        wp.launch(K.wet_update, dim=n, inputs=[k, sf, pr, bd, q, radius, normal, no_body, shape_body, DT],
                  outputs=[wet], device="cpu")
    return wet.numpy()


def _key(p: float) -> int:
    return int(p) * int(K.KEY_SHIFT) + 0


def test_w1_a_held_pressure_follows_the_exact_ode():
    """W-1: free vertices pressed at 1.2, 2 and 5 x p_form for 0.3 s: w = 1 - exp(-(p/P_REF)·t/tau_w), stepped exactly
    (300 float32 steps: 1e-5)."""
    ps = np.array([1.2, 2.0, 5.0]) * P_FORM
    wet = wp.zeros(3, dtype=float, device="cpu")
    got = _step(np.array([_key(p) for p in ps]), np.zeros(3), wet, np.zeros((3, 3)), _press(3), 300)
    want = [wet_after(0.0, float(int(p)), 0.3, P_FORM, TAU) for p in ps]
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_w2_wet_out_rises_with_pressure_and_none_forms_below_p_form():
    """W-2: higher pressure wets faster (rate ∝ (p/P_REF)^M, M = 1); a free vertex off the press (no key: below
    p_form or not touching) stays at 0 and a pressed one that leaves it goes back to 0; a bonded one keeps its w."""
    wet = wp.zeros(4, dtype=float, device="cpu")
    key = np.array([_key(1.5 * P_FORM), _key(3.0 * P_FORM), 0, 0])
    got = _step(key, np.array([0, 0, 0, 1]), wet, np.zeros((4, 3)), _press(4), 100)
    rate = -np.log1p(-got[:2])
    assert rate[1] / rate[0] == pytest.approx(2.0, rel=1e-4) and got[2] == 0.0 and got[3] == 0.0
    wet.assign(np.array([0.5, 0.5, 0.5, 0.5], np.float32))
    got = _step(np.zeros(4), np.array([0, 0, 1, 1]), wet, np.zeros((4, 3)), _press(4), 1)
    np.testing.assert_array_equal(got, [0.0, 0.0, 0.5, 0.5])


def _roll_pass(dx: float) -> np.ndarray:
    """Bonded vertices on z = 0 every ``dx`` over 10 mm, under the standard roller (axis y, radius R, its bottom at
    the film) moved along x at ROLL_SPEED from -3 to 13 mm."""
    r = STANDARD_ROLLER
    xs = np.arange(0.0, 0.010 + 1e-9, dx)
    pts = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
    rot = np.array([-math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)], np.float32)   # local z -> world y

    def pose(j: int) -> np.ndarray:
        x = -3.0e-3 + ROLL_SPEED * (j + 1) * DT
        return np.array([[x, 0.0, r.radius, *rot]], np.float32)
    wet = wp.zeros(len(xs), dtype=float, device="cpu")
    steps = int(round(0.016 / ROLL_SPEED / DT))
    return _step(np.zeros(len(xs)), np.ones(len(xs)), wet, pts, _press(len(xs), roller=r, shape_xform=pose(0)),
                 steps, move=pose)


def test_w3_a_roller_pass_gives_the_hertz_wet_out_at_any_cell_size():
    """W-3 on the kernel: one pass of the standard roller at 20 mm/s over vertices 2.5 mm and 1.25 mm apart gives the
    Hertz contact's exact wet-out (wetout.exposure) at every vertex, within 1 % (the step's sampling of the profile)."""
    want = 1.0 - math.exp(-exposure(STANDARD_ROLLER, STANDARD_ROLLER.line_load(), P_FORM, ROLL_SPEED) / TAU)
    for dx in (2.5e-3, 1.25e-3):
        np.testing.assert_allclose(_roll_pass(dx), want, rtol=1e-2)
    assert want == pytest.approx(0.4377, rel=1e-3)


def test_the_roller_card_must_match_its_shape_and_loads_its_weight():
    with pytest.raises(ValueError, match="radius, width, mass"):
        Roller(radius=0.0)
    cfg = Bonds.Config(substrates=("plate",), adhesives=("3m_7818/stainless_steel",), rollers=({"shape": "r"},))
    assert cfg.rollers[0] == Roller(shape="r")


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["W-3_roll_dx2.5", "W-3_roll_dx1.25", "W-4_roll_servo", "W-5_roll_lifted"])
def test_the_roller_laws_on_the_rig(name):
    """W-3..W-5 on the coupled rig (a 10 x 10 mm patch, a frictionless capsule roller on the gantry): the roller's line
    load measured on the film is the force it presses with (5 %) and the patch's mean w the Hertz pass's at that load
    (2.5 %) at both cell sizes; a roller that never touches does nothing."""
    from alloy.plugins.harness.run import run_scene

    law = next(x for x in bonds_laws() if x.name == name)
    run = run_scene(law.scene, [Bonds(**dict(law.config))], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, res.detail


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_front_peels_at_g_scaled_by_its_wet_out():
    """The cards' rows are a standard application's (w = W_STD). The 90° peel law's strip wetted to half that peels
    at Kendall's force for G/2: F/b = E·t·(sqrt(1 + G/(E·t)) - 1) = 190 N/m for the 380 N/m row, within 5 %."""
    from dataclasses import replace

    from alloy.plugins.harness.run import run_scene

    law = next(x for x in bonds_laws() if x.name == "peel_90")
    plug = Bonds(**dict(law.config))

    def init(sim):
        law.scene.init(sim)
        plug._flat["wet"].fill_(0.5 * W_STD)
    run = run_scene(replace(law.scene, init=init), [plug], "cpu")
    assert run.error is None, run.error
    t = np.asarray(run.trace.plugin(plug.label, "peel_force"), np.float64)[:, 0]
    b = np.asarray(run.trace.plugin(plug.label, "width"), np.float64)[:, 0]
    front = np.asarray(run.trace.plugin(plug.label, "front"), np.float64)[:, 0]
    win = (front >= 0.010) & (front <= 0.030)
    et = Bonds.Config().face_et
    g = 0.5 * adhesive("3m_7818/stainless_steel").g90
    want = et * (math.sqrt(1.0 + 2.0 * g / et) - 1.0)
    assert win.sum() >= 3 and float(np.mean(t[win] / b[win])) == pytest.approx(want, rel=0.05)


def test_gas_criterion_is_relative_to_a_standard_application():
    """WP13's pass rule: w_task / w_standard >= 0.90 over the label and over its edge band, so a standard application
    passes with margin (1.0), not on the threshold."""
    from alloy.plugins.bonds.wetout import ga_wet_out

    area, band = np.ones(4), np.array([True, True, False, False])
    assert ga_wet_out(np.full(4, W_STD), area, band) == {"label": 1.0, "edge_band": 1.0, "pass": True}
    weak_edge = ga_wet_out(np.array([0.5, 0.5, 0.9, 0.9]) * W_STD / 0.9, area, band)
    assert not weak_edge["pass"] and weak_edge["edge_band"] == pytest.approx(0.5 / 0.9)
