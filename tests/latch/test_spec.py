# SPDX-License-Identifier: Apache-2.0
"""alloy.latch spec, mass regularisation, design checks and geometric state (pure numpy, no simulation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.latch.design import (cantilever_hinge, cantilever_strain, check_flip, check_ramp, check_spec,
                                lock_angle, wedge_ratio)
from alloy.latch.mass import box_props, member_props, mode_omega, stop_solref
from alloy.latch.mjcf import latch_mjcf, mjcf_masses
from alloy.latch.spec import ROOT, Box, Coupling, LatchSpec, LatchSpecError, Member, Spring
from alloy.latch.state import catch_state, compose
from alloy.testing.latch import TabRig, keeper_spec, tab_spec


def _slide(name="sleeve", rate=800.0, preload=4.0, mass=0.01, parent=ROOT):
    return Member(name, "slide", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.004, Spring(rate, preload),
                  (Box(f"{name}_box", (0.005, 0.005, 0.005), role="catch"),), mass, parent=parent)


def test_spring_preload_sits_on_the_stop():
    s = Spring(rate=0.2, preload=0.01)
    assert s.springref == pytest.approx(-0.05)
    assert s.force(0.0) == pytest.approx(-0.01) and s.force(0.1) == pytest.approx(-0.03)


def test_spec_rejects_bad_input():
    with pytest.raises(LatchSpecError):
        Box("b", (0.0, 1.0, 1.0))
    with pytest.raises(LatchSpecError):
        Box("b", (1.0, 1.0, 1.0), role="spring")
    with pytest.raises(LatchSpecError):
        LatchSpec("x", (Box("a", (1, 1, 1)),), 1.0, (_slide(parent="nope"),))
    with pytest.raises(LatchSpecError):
        LatchSpec("x", (Box("sleeve_box", (1, 1, 1)),), 1.0, (_slide(),))         # duplicate shape name
    with pytest.raises(LatchSpecError):
        LatchSpec("x", (Box("a", (1, 1, 1)),), 1.0, (_slide(),), couplings=(Coupling("sleeve", "ghost"),))


def test_box_inertia_matches_closed_form():
    p = box_props((Box("b", (0.01, 0.02, 0.03)),), 0.6)
    want = 0.6 / 12.0 * np.array([0.04**2 + 0.06**2, 0.02**2 + 0.06**2, 0.02**2 + 0.04**2])
    assert np.allclose(np.diag(p.inertia), want, rtol=1e-9) and p.principal_ok()


@pytest.mark.parametrize("dt", [6.25e-4, 1.25e-3])
def test_regularisation_meets_omega_h(dt):
    """CON-45: a light tab's spring mode is slowed to omega·h = 0.35 at the design step with inertia, not mass."""
    tab = tab_spec(TabRig()).member("tab")
    p = member_props(tab, dt)
    assert p.added_inertia > 0.0 and p.mass == tab.mass and p.principal_ok()
    assert mode_omega(tab, p) * dt == pytest.approx(0.35, rel=1e-9)
    s = _slide(rate=1.0e4, mass=1e-4)
    ps = member_props(s, dt)
    assert ps.added_mass > 0.0 and mode_omega(s, ps) * dt == pytest.approx(0.35, rel=1e-9)


def test_heavy_member_is_not_regularised():
    s = _slide(rate=10.0, mass=0.5)
    assert member_props(s, 1.25e-3).added_mass == 0.0


def test_design_laws():
    assert wedge_ratio(0.0, 0.4) == pytest.approx(0.4)
    assert math.isinf(wedge_ratio(lock_angle(0.5) + 1e-6, 0.5))
    assert lock_angle(1.0) == pytest.approx(math.pi / 4)
    assert not check_ramp("ok", math.radians(30), 0.42)
    assert check_ramp("lock", math.radians(70), 0.42)[0].level == "error"
    assert check_ramp("near", math.radians(60), 0.6)[0].level in ("warn", "error")
    k = cantilever_hinge(2.35e9, 3e-3, 1e-3, 0.01)
    assert k == pytest.approx(3 * 2.35e9 * 3e-3 * 1e-9 / 12 / 0.01)
    assert cantilever_strain(1e-3, 0.01, 2e-3) == pytest.approx(0.03)
    assert check_flip("hold", 89.0, 3.2e5, 1e-3) == []
    assert check_flip("soft", 89.0, 2e4, 1e-3)[0].level == "error"
    assert check_spec(tab_spec(TabRig())) == []


def test_stop_solref_is_direct_and_scales_with_design_dt():
    a = stop_solref(tab_spec(TabRig(design_dt=1e-3)))
    b = stop_solref(tab_spec(TabRig(design_dt=5e-4)))
    assert a[0] < 0 and a[1] < 0 and b[0] == pytest.approx(4 * a[0])


def test_mjcf_piece_is_deterministic_and_complete():
    spec = tab_spec(TabRig())
    txt = latch_mjcf(spec)
    assert txt == latch_mjcf(spec)
    assert '<freejoint name="rig_carrier_free"/>' in txt
    assert 'type="hinge"' in txt and 'stiffness="0.17625"' in txt and 'springref="-0.0261799388"' in txt
    assert '<exclude body1="rig_carrier" body2="rig_carrier_tab"/>' in txt
    assert 'density="0"' in txt and 'fullinertia=' in txt
    static = latch_mjcf(keeper_spec(TabRig()))
    assert "freejoint" not in static and "<contact>" not in static
    masses = mjcf_masses(spec)
    assert masses["rig_carrier_tab"] == pytest.approx(spec.member("tab").mass)


def test_coupling_emits_equality():
    s1, s2 = _slide("bail", parent=ROOT), _slide("slider", parent=ROOT)
    spec = LatchSpec("cage", (Box("cage_body", (0.01, 0.01, 0.01)),), 0.02, (s1, s2),
                     couplings=(Coupling("bail", "slider", (0.0, 0.5)),))
    assert '<joint joint1="cage_slider_joint" joint2="cage_bail_joint" polycoef="0 0.5 0 0 0"/>' in latch_mjcf(spec)


def test_catch_state_reads_geometry():
    """A catch box behind a keeper box, overlapping 1 mm in lift: engaged; lifted clear or pulled out: not."""
    keeper_q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    kc, kR = compose(keeper_q, (0.0, 0.0, 0.0), (0, 0, 0, 1))
    kh = (0.002, 0.003, 0.001)                         # keeper x in [-2, 2] mm, z in [-1, 1] mm

    def state(x, z):
        cc, cR = compose(np.array([[x, 0.0, z, 0.0, 0.0, 0.0, 1.0]]), (0.0, 0.0, 0.0), (0, 0, 0, 1))
        return catch_state(cc, cR, (0.001, 0.003, 0.001), kc, kR, kh, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    s = state(0.0035, -0.001)                          # 0.5 mm behind, top 1 mm above the keeper's lower face
    assert s.engaged[0] and s.behind[0] == pytest.approx(5e-4) and s.lift[0] == pytest.approx(1e-3)
    assert not state(0.0035, -0.0021).engaged[0]       # pressed below the keeper
    assert not state(-0.0035, -0.001).engaged[0]       # in front of the keeper (pulled out)
