# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11 (docs/notes/heat-v3-design.md §Slice 11): the numpy reference of a slice's free surface with the
liquid's weight (``grid_surface_arc``, polar form about the joint's root) against solutions it does not use: the
circle (weightless), the 2-D puddle's first integral (gravity, the 88 deg cap) and GA's height function at 384
nodes (gravity, a fillet)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from alloy.plugins.thermal import grid_surface_arc as A
from alloy.plugins.thermal import grid_surface_ref as HF

GAMMA, RHO = 1.2, 6950.0                 # liquid steel, GA params.py
DOWN = np.array([0.0, -9.81])
ORIGIN = np.zeros(2)
PLATE, BRACKET, LEFT = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])
GA_TOE = math.radians(88.0)              # GA's cap: a toe at 88 deg to its chord (pool.py _caps), not A.TOE_LIMIT


def puddle_height(width: float, theta: float) -> float:
    """A 2-D puddle's apex height on a flat plate, its face meeting the plate at theta, from the first integral of
    gamma kappa = P - rho g z: gamma (1 - cos phi) = rho g d^2 / 2 + gamma k0 d at depth d below the apex (phi the
    face's slope, k0 the apex curvature), the half width the integral of gamma cos phi / (rho g d + gamma k0) dphi."""
    rg = RHO * 9.81

    def half(h: float) -> float:
        k0 = (GAMMA * (1.0 - math.cos(theta)) - 0.5 * rg * h * h) / (GAMMA * h)

        def depth(phi: float) -> float:
            b = GAMMA * k0
            return (-b + math.sqrt(b * b + 2.0 * rg * GAMMA * (1.0 - math.cos(phi)))) / rg

        knee = k0 * math.sqrt(2.0 * GAMMA / rg)                         # where the weight takes over from k0
        cuts = [c for c in (knee * 10.0 ** j for j in range(-2, 40)) if 0.0 < c < theta]      # one a decade
        return quad(lambda phi: GAMMA * math.cos(phi) / (rg * depth(phi) + GAMMA * k0), 0.0, theta,
                    epsabs=0.0, epsrel=1e-11, limit=400, points=cuts or None)[0] - 0.5 * width

    top = math.sqrt(2.0 * GAMMA * (1.0 - math.cos(theta)) / rg)          # the infinitely wide puddle (k0 = 0)
    return brentq(half, 0.3 * top, 0.999 * top, xtol=1e-14)      # 40 mm stands 1.1 % below the wide limit


def test_weightless_surface_is_the_circle():
    """D-8's reference: no weight, no arc, so the surface is the circular cap holding the area (D-3's joint)."""
    ch = HF.Chord(ORIGIN, PLATE, BRACKET, 4e-3, 3e-3)
    s = A.pressure_for(ch, 12e-6, A.Physics(GAMMA, RHO, ORIGIN))
    rest = 12e-6 - ch.wedge                                       # the segment on the 5 mm chord
    th = brentq(lambda t: (ch.length / (2 * math.sin(t / 2))) ** 2 / 2 * (t - math.sin(t)) - rest, 1e-6, math.pi)
    radius = ch.length / (2 * math.sin(th / 2))
    assert s.status == "held" and s.area == pytest.approx(12e-6, rel=1e-9)
    assert s.pressure == pytest.approx(GAMMA / radius, rel=1e-6)
    assert s.toe_chord == pytest.approx((th / 2, th / 2), abs=1e-6)
    centre = ch.p1 + 0.5 * (ch.p2 - ch.p1) - radius * math.cos(th / 2) * ch.normal
    assert np.hypot(*(s.points - centre).T) == pytest.approx(radius, rel=1e-7)


@pytest.mark.parametrize("width", [10e-3, 20e-3, 40e-3])
def test_puddle_capped_at_88_degrees_has_the_first_integrals_height(width):
    """D-9's reference: on a flat plate the surface capped at GA's 88 deg toe stands as high as the exact 2-D
    puddle (a 24-node height function stands 21-45 % higher: the first interval cannot resolve the toe)."""
    ch = HF.Chord(ORIGIN, PLATE, LEFT, width / 2, width / 2)
    cap = A.top(ch, A.Physics(GAMMA, RHO, DOWN))
    assert cap.status == "full" and cap.toe_chord == pytest.approx((GA_TOE, GA_TOE), abs=1e-6)
    assert float(cap.points[:, 1].max()) == pytest.approx(puddle_height(width, GA_TOE), rel=2e-5)
    assert puddle_height(width, GA_TOE) < 2 * math.sqrt(GAMMA / (RHO * 9.81)) * math.sin(GA_TOE / 2)


@pytest.mark.parametrize("area", [27.5e-6, 24.0e-6, 20.0e-6])
def test_fillet_with_gravity_matches_the_fine_height_function(area):
    """GA's steady fillet (toes 7 / 7 mm) with the liquid's weight, convex to concave: the same pressure as GA's own
    height-function surface at 384 nodes (which needs no toe angle while the toes stay below the cap)."""
    ch = HF.Chord(ORIGIN, PLATE, BRACKET, 7e-3, 7e-3)
    s = A.pressure_for(ch, area, A.Physics(GAMMA, RHO, DOWN))
    p_hf, _ = HF.pressure_for(ch, area, GAMMA, RHO, DOWN, n=384)
    assert s.status == "held" and s.area == pytest.approx(area, rel=1e-7)
    assert s.pressure == pytest.approx(p_hf, abs=0.05)


def test_the_cap_binds_one_toe_and_holds_the_most():
    """Gravity tilted 30 deg along the plate: toe 1 binds at 88 deg to the chord, toe 2 stays below; the held area
    grows with the pressure to the cap, through the draining branch (toe 2 receding, meeting the wall
    tangentially) and on."""
    tilt = 9.81 * np.array([math.sin(math.radians(30)), -math.cos(math.radians(30))])
    ch = HF.Chord(ORIGIN, PLATE, BRACKET, 7e-3, 7e-3)
    ph = A.Physics(GAMMA, RHO, tilt)
    cap = A.top(ch, ph)
    assert cap.toe_chord[0] == pytest.approx(GA_TOE, abs=1e-6) and cap.toe_chord[1] < GA_TOE
    rows = [A.surface(ch, p, ph, cap) for p in np.linspace(cap.pressure - 500.0, cap.pressure, 11)]
    assert all(r is not None for r in rows)
    areas = [r.area for r in rows]
    assert all(b > a for a, b in zip(areas, areas[1:])) and areas[-1] == cap.area
    wet = [r for r in rows if r.status == "wet"]
    assert wet and all(r.toes[1] < 7e-3 and r.toe_solid[1] < 1e-2 for r in wet)


def test_a_draining_slice_recedes_from_the_wall():
    """15 mm2 between toes 7 / 7 mm (wedge 24.5 mm2), gravity down: no surface reaches toe 2 at a positive angle,
    so toe 2 recedes down the bracket to where the face meets it tangentially; the area is still held exactly
    (the unbounded height function would cross into the bracket instead)."""
    ch = HF.Chord(ORIGIN, PLATE, BRACKET, 7e-3, 7e-3)
    s = A.pressure_for(ch, 15e-6, A.Physics(GAMMA, RHO, DOWN))
    assert s.status == "wet" and s.area == pytest.approx(15e-6, rel=1e-7)
    assert 6.0e-3 < s.toes[1] < 7e-3 and s.toe_solid[1] < 1e-2 and float(s.points[:, 0].min()) >= -1e-12


SURFACE_LAWS = None


def _surface_laws():
    from alloy.plugins.thermal.grid_surface_laws import surface_laws  # noqa: PLC0415

    return {law.name: law for law in surface_laws()}


@pytest.mark.law
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["surface_weightless", "surface_puddle", "surface_fillet", "surface_dent",
                                  "surface_touchdown", "surface_contact"])
def test_surface_law(name):
    from alloy.plugins.harness.check import _law, fresh  # noqa: PLC0415
    from alloy.plugins.thermal.grid import GridThermal  # noqa: PLC0415
    from alloy.plugins.thermal.grid_laws import BUSY  # noqa: PLC0415

    law = _surface_laws()[name]
    res = _law(law, None, [fresh(GridThermal(**BUSY), law.config)], "cpu")
    assert res.passed, (res.value, res.detail, res.fix)


def test_a_frozen_fillet_overhangs_then_touches_down():
    """GA's steady fillet with its toes held (the contact frozen), gravity down: past the cap (43.4 mm2) its face
    turns past 88 deg and overhangs under volume control; the family ends when the face at toe 1 lies along the plate
    (180 deg to it, 57.0 mm2 at any step count), past which the bulge touches down at the toe."""
    ch = HF.Chord(ORIGIN, PLATE, BRACKET, 7e-3, 7e-3)
    ph = A.Physics(GAMMA, RHO, DOWN)
    cap = A.top(ch, ph)
    over = A.pinned_for(ch, 50e-6, ph)
    assert over.status == "overhang" and over.area == pytest.approx(50e-6, rel=1e-6)
    assert over.toe_chord[0] > GA_TOE and over.pressure > cap.pressure
    edges = [A.pinned_for(ch, 70e-6, ph, steps=n) for n in (32, 128)]
    assert all(e.status == "touch" and e.touch == pytest.approx(7e-3) for e in edges)
    assert edges[0].area == pytest.approx(edges[1].area, rel=1e-4)
    assert edges[1].area == pytest.approx(57.0225e-6, abs=0.1e-6)        # AUDIT-BENCH's own shooting: 57.023 mm2
    assert edges[1].toe_solid[0] == pytest.approx(math.pi, abs=1e-3)


def test_the_arc_dents_a_flat_pool_as_the_linear_closed_form():
    """A wide flat pool at a fixed pressure under a small Gaussian load p0 exp(-y^2 / 2 s^2) (GA's arc pressure on
    one slice): gamma eta'' - rho g eta = p, so the dent below the unloaded surface is
    eta(0) = -(p0 / (2 gamma k)) 2 s sqrt(pi / 2) exp(k^2 s^2 / 2) erfc(k s / sqrt 2), k = sqrt(rho g / gamma)."""
    ch = HF.Chord(ORIGIN, PLATE, LEFT, 20e-3, 20e-3)
    flat = A.Physics(GAMMA, RHO, DOWN)
    cap = A.top(ch, flat, steps=512)
    p = 0.95 * cap.pressure                           # held toes (0.99 of the cap finds no held branch)
    plain = A.surface(ch, p, flat, cap, steps=512)
    z0 = float(np.interp(0.0, plain.points[::-1, 0], plain.points[::-1, 1]))
    p0, sig = 20.0, 1.79e-3
    loaded = A.surface(ch, p, A.Physics(GAMMA, RHO, DOWN, arc_peak=p0, arc_at=np.array([0.0, z0]), arc_sigma=sig),
                       cap, steps=512)
    dent = float(np.interp(0.0, loaded.points[::-1, 0], loaded.points[::-1, 1])) - z0
    k = math.sqrt(RHO * 9.81 / GAMMA)
    want = -p0 / (2 * GAMMA * k) * 2 * sig * math.sqrt(math.pi / 2) * math.exp(0.5 * (k * sig) ** 2) * math.erfc(
        k * sig / math.sqrt(2))
    assert plain.status == loaded.status == "held"
    assert dent == pytest.approx(want, rel=0.02)                         # 0.115 mm; nonlinear terms ~0.3 %


def _climb_kernel():
    import warp as wp  # noqa: PLC0415

    from alloy.plugins.thermal import grid_bead_surface as BS  # noqa: PLC0415
    from alloy.plugins.thermal import grid_cap as C  # noqa: PLC0415
    from alloy.plugins.thermal import grid_surface as S  # noqa: PLC0415

    @wp.kernel
    def climb(bead: C.Bead, out: wp.array(dtype=wp.vec2)):
        out[0] = C.overflow(bead, 2.0e-3, 2.0e-3, 40.0e-6, 2.5e-4)
        fr = S.frame_of(bead.root, bead.e1, bead.e2, 2.0e-3, 2.0e-3, wp.vec2(0.0, 1.0), wp.vec2(0.0, -9.81), 6950.0,
                        1.2, 0.0, wp.vec2(0.0, 0.0), 1.0)
        out[1] = BS.overflow(bead, fr, 40.0e-6, 4.0e-6, 2.5e-4)
    return climb


@pytest.mark.parametrize("rise", [0.29, 0.31])
def test_overflow_does_not_climb_past_ga_17_degrees(rise):
    """GA's climb rule (pool.py:1029-1056): an overfull slice spreads along a member only where e . down > -0.3
    (17.5 deg). Toe 2's member rising at sin 0.29 takes the overflow, at sin 0.31 not, in both bead models."""
    import warp as wp  # noqa: PLC0415

    from alloy.plugins.thermal.grid_cap import Bead  # noqa: PLC0415

    b = Bead()
    b.root, b.e1, b.down = wp.vec2(0.0, 0.0), wp.vec2(1.0, 0.0), wp.vec2(0.0, -1.0)
    b.e2, b.advance = wp.vec2(-math.sqrt(1.0 - rise * rise), rise), 1.0e-3
    out = wp.zeros(2, dtype=wp.vec2, device="cpu")
    wp.launch(_climb_kernel(), dim=1, inputs=[b, out], device="cpu")
    moves = out.numpy()
    assert np.all(moves[:, 0] > 0.0)                                   # the plate takes it
    assert np.all(moves[:, 1] > 0.0) if rise < 0.3 else np.all(moves[:, 1] == 0.0)
