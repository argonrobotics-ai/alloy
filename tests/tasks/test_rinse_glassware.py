# SPDX-License-Identifier: Apache-2.0
"""rinse_glassware (Gym-Anything rinse-glassware): vessel profiles, the wash station's plumbing, and one glass through
the rinse bowl, the sanitizer and into the rack."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from alloy.tasks.rinse_glass_scene import GlassStation
from alloy.tasks.rinse_glassware import Glassware, run
from alloy.tasks.rinse_sink import BOWLS
from alloy.tasks.rinse_vessels import Pan, mixing_glass, rocks_glass
from alloy.tasks.rinse_wash_station import Item, WashStation


@dataclass(frozen=True)
class PanInWash(WashStation):
    """A station with one pan that starts full of the wash bowl's water."""

    def items(self) -> tuple[Item, ...]:
        return (Item("pan", Pan(), (0.0, 0.0, 0.0), fill_from="wash"),)


@pytest.mark.parametrize("glass", [rocks_glass(), mixing_glass()])
def test_a_glass_cavity_is_its_frustum(glass):
    """A linear taper's inside is a frustum: V = πh(r0² + r0 r1 + r1²)/3 (to the quadrature's 1e-7); the hull profile's
    z rises from the underside to the rim."""
    r0, r1, h = float(glass.radius(0.0)), float(glass.radius(glass.depth)), glass.depth
    assert glass.cavity_volume() == pytest.approx(math.pi * h * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0, rel=1e-7, abs=0)
    z = [n[1] for n in glass.outer()]
    assert all(b > a for a, b in zip(z[:-1], z[1:])) and z[0] == -glass.bottom


@pytest.mark.parametrize("glass", [rocks_glass(), mixing_glass()])
def test_a_glass_is_their_taper_less_the_wall(glass):
    """Their glass (ware.py:180-192): outer r = r_bot + (r_top - r_bot) z/H, inner = outer - wall; the inner floor
    sits ``bottom`` up."""
    g = glass
    assert float(g.radius(0.0)) == pytest.approx(g.r_bot + (g.r_top - g.r_bot) * g.bottom / g.height - g.wall,
                                                 abs=1e-15)
    assert float(g.radius(g.height - g.bottom)) == pytest.approx(g.r_top - g.wall, abs=1e-15)


def test_a_pan_is_their_outer_profile_offset_by_its_wall():
    """Their profile is the outer wall, r = rb + (R - rb) s^1.5 over H from the foot (ware.py:421-422); the inside is
    it offset by the wall along the normal, over a 4 mm base: the rim's inner radius is R - wall·sqrt(1 + r'(H)²),
    the depth H - base, and the cavity ∫ π r² dz over it."""
    p = Pan()
    assert p.depth == pytest.approx(p.height - p.bottom, rel=0, abs=1e-15)
    slope = 1.5 * (p.R - p.r_bot) / p.height
    assert p.rim == pytest.approx(p.R - p.wall * math.sqrt(1.0 + slope * slope), rel=0, abs=1e-12)
    z = np.linspace(0.0, p.depth, 200001)
    exact = math.pi * np.trapezoid(np.asarray(p.radius(z)) ** 2, z)
    assert p.cavity_volume() == pytest.approx(exact, rel=1e-6, abs=0)


def test_the_stations_plumbing():
    """Glasses: three bowls and six empty cavities, a pour each, a weir to every bowl; the pan starts full of wash
    water, taken from the wash."""
    st = GlassStation(weir=BOWLS)
    nodes = {n.name: n for n in st.nodes()}
    assert list(nodes)[:3] == list(BOWLS) and len(nodes) == 9
    assert all(nodes[f"glass{k}_cavity"].volume == 0.0 for k in range(6))
    assert len([e for e in st.edges() if e.name.endswith("_pour")]) == 6
    assert all(p.weir == BOWLS for p in st.pools()[3:])
    pan = PanInWash()
    wash = {n.name: n for n in pan.nodes()}
    full, v = pan.sink.area * pan.sink.fill[0], Pan().cavity_volume()
    assert wash["pan_cavity"].volume == pytest.approx(v, rel=1e-12, abs=0)
    assert wash["wash"].volume == pytest.approx(full - v, rel=1e-12, abs=0)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)                 # 167 s on an idle CPU (the compile_heavy default is 180 s)
def test_one_glass_is_rinsed_sanitized_and_racked():
    """One soapy rocks glass on a coarse chart: a rinse dunk takes its soap (film and walls) under 10 µg, 34 s in the
    sanitizer runs the clock past 30 s from a qualifying entry, and it sits mouth down in its compartment, drained;
    every gate passes and the ledgers close."""
    task = Glassware(station=GlassStation(count=1, weir=BOWLS, ds=4.0e-3, n_theta=16), rinse_s=2.0, pace=0.6)
    res = run(task, worlds=1)
    assert res.success == [True] and res.score == [1.0], res.failed
    assert res.soap_ug[0][0] < 10.0 and res.sanitized_s[0][0] >= 30.0 and res.free_water_ml[0][0] <= 2.0
    assert res.ledger_max < 1e-9


def test_the_overflow_watch():
    """A world whose MuJoCo Warp overflow word has a capacity bit is flagged; the iteration bits alone are not; a
    solver without the word reads None."""
    from types import SimpleNamespace

    from alloy._vendor.mujoco_warp import OverflowType

    from alloy.tasks.rinse_hand import contact_overflow

    bits = [0, int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS), int(OverflowType.NARROWPHASE),
            int(OverflowType.BROADPHASE | OverflowType.ITERATIONS)]
    word = SimpleNamespace(numpy=lambda: np.array(bits, np.int32))
    assert contact_overflow(SimpleNamespace(mjw_data=SimpleNamespace(overflow=word))).tolist() == [False, False,
                                                                                                    True, True]
    assert contact_overflow(None) is None


def test_face_soap_counts_the_walls():
    """The faces' soap is the film's solution plus the walls' adsorbed load, times the cells' areas (their
    (s + gamma)·A, sanitizer.py:51), on the vessel's own cells only."""
    from types import SimpleNamespace

    from alloy.tasks.rinse_sense import face_soap_kg

    arr = lambda a: SimpleNamespace(numpy=lambda: np.asarray(a))                     # noqa: E731
    mass = np.zeros((2, 5, 3))
    mass[1, 2:5, 0] = [1e-6, 2e-6, 3e-6]
    load = np.zeros((2, 5, 3))
    load[1, 2:5, 0] = [5e-8, 5e-8, 5e-8]
    load[1, 0, 0] = 1.0                                                              # another chart's cell
    flow = SimpleNamespace(sp=SimpleNamespace(mass=arr(mass)), adsorption=SimpleNamespace(load=arr(load)),
                           tables=SimpleNamespace(cells_of=lambda k: np.arange(2, 5)))
    area = np.array([0.01, 0.02, 0.03])
    assert face_soap_kg(flow, area, 1, 1) == pytest.approx(1.4e-7 + 3.0e-9, rel=1e-15, abs=0)
    flow.adsorption = None
    assert face_soap_kg(flow, area, 1, 1) == pytest.approx(1.4e-7, rel=1e-15, abs=0)


def test_the_books_close_each_ledger():
    """Each book is its change since the start against its scale: the film's water, each species (walls included),
    the supply's cross ledger. A 1e-9 relative slip in any one opens it."""
    from types import SimpleNamespace

    from alloy.tasks.rinse_books import Books

    def read(water=0.0, soap=0.0, cross=0.0):
        r = {"ledger": np.array([2e-6 + water]), "h": np.zeros((1, 3)), "supplied_volume": np.array([1e-3]),
             "species_ledger": np.array([[5e-9 + soap, 0.0]]), "species_supplied": np.zeros((1, 2, 2)),
             "species_out": np.array([[[1e-9, 0.0]]]), "pending_volume": np.array([0.0]),
             "supply_sent": np.array([1e-3 + cross]), "supply_swept": np.array([0.0]),
             "supply_withdrawn": np.array([0.0])}
        r.update({f"supply_{k}": np.array([0.0]) for k in ("to_tank", "lost", "overspray", "mist", "caught",
                                                           "lumped")})
        return r

    flow = SimpleNamespace(read=read)
    books = Books.start(flow)
    assert books.closed(flow)[0].tolist() == [True]
    for kw in ({"water": 2e-12}, {"soap": 1e-17}, {"cross": 2e-12}):
        flow.read = lambda kw=kw: read(**kw)
        assert books.closed(flow)[0].tolist() == [False], kw


def test_the_outlet_test_is_per_world():
    """A faucet running in one world counts there only: their outlet test (over 1e-4 gpm) per world."""
    from types import SimpleNamespace

    from alloy.tasks.rinse_glassware import outlet_running
    from alloy.tasks.rinse_sink import GPM

    st = GlassStation(count=1, weir=BOWLS)
    k = [e.name for e in st.edges()].index("faucet")
    q = np.zeros((3, len(st.edges())))
    q[1, k] = 2.0e-4 * GPM
    q[2, k] = 0.5e-4 * GPM
    flow = SimpleNamespace(supply=SimpleNamespace(read=lambda: {"discharge": q}))
    assert outlet_running(flow, st)().tolist() == [False, True, False]
