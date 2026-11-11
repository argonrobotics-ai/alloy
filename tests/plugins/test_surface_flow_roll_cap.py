# SPDX-License-Identifier: Apache-2.0
"""The roll's cap with spill at the blade ends (roll_cap.py; Gym-Anything's concrete coupling.py:597-636, ``_spill``):
a guided blade ploughs a standing paste; its roll ahead holds ½·(rise + lift)²·L_face and the rest lies in 20 mm
bands beyond its ends, no higher than the roll's top, with exact volume and species ledgers (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store

LENGTH, WIDTH = 0.2, 0.16                  # the plate [m]
BOX = (0.02, 0.04, 0.01)                   # the tool block's half extents; its underside 2 mm below it
DROP = 0.002
BLADE = (0.02, 0.035)                      # half extents: 40 mm along the slide (x), 70 mm across it
H0, GAP, U, X0 = 2.0e-3, 0.5e-3, 0.1, -0.06
LIFT = 3.0e-3


def _build(y0: float):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01,
                        label="plate")
        xf = body_xform(X0, y0, GAP + DROP + BOX[2])
        tool = b.add_link(xform=xf, label="tool")
        b.add_articulation([b.add_joint_prismatic(-1, tool, parent_xform=xf, axis=newton.Axis.X)])
        b.add_shape_box(tool, hx=BOX[0], hy=BOX[1], hz=BOX[2], cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    return build


def _blade(lift: float | None, pitch: float = 0.0, size=BLADE, shape: str = "rect", **edger) -> Applicator:
    """A rect blade, its leading (+x) edge raised by ``pitch`` [rad]."""
    return Applicator(body="tool", shape=shape, size=size, center=(0.0, 0.0, -BOX[2] - DROP),
                      normal=(math.sin(pitch), 0.0, -math.cos(pitch)), axis=(math.cos(pitch), 0.0, math.sin(pitch)),
                      stiffness=1.0e3, roll_lift=lift, **edger)


def _pass(nx: int, lift: float | None = LIFT, periods: int = 20, depth=H0, y0: float = 0.0, worlds: int = 1,
          **blade):
    """A pass at U (``depth``: one, or one per world); returns (flow, sim, cell centres x, y, cell area)."""
    ny = int(round(nx * WIDTH / LENGTH))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(nx, ny)),), material=sl.PASTE, inertia=False,
                       max_depth=0.03, tools=(_blade(lift, **blade),), species=("dye",))

    def init(sim) -> None:
        d = np.asarray(depth, np.float64)
        if isinstance(depth, tuple):                                 # one depth per world
            d = d.reshape(-1, 1) * np.ones((worlds, flow.h.shape[1]))
        flow.set_depth(d)
        flow.set_species(10.0 * d, "dye")                           # 10 kg/m³ everywhere
        qd = sim.state.joint_qd.numpy()
        qd[:] = U
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=_build(y0), init=init, ground=False, gravity=(0.0, 0.0, -9.81),
                                     worlds=worlds, dt=2.0e-3), [flow], "cpu", worlds)
    for _ in range(periods):
        sim.step_period()
    c = flow.cell_centres(0)
    return flow, sim, c[..., 0], c[..., 1], (LENGTH / nx) * (WIDTH / ny)


def _cap(rise: float = 0.0, face: float = 2.0 * BLADE[1]) -> float:
    return 0.5 * (rise + LIFT) ** 2 * face


def _roll(flow, sim, x, y, dA, depth: float = H0, y0: float = 0.0, half: float = BLADE[1]) -> float:
    """The film above the paste ahead of the blade's centre over its width: the roll, its apex under the leading edge
    included (under the blade the paste is cut below the paste's level)."""
    h = flow.depth(0)[0]
    ahead = (x > float(sim.state.body_q.numpy()[0][0])) & (np.abs(y - y0) <= half)
    return float(((h - depth).clip(min=0.0) * ahead).sum() * dA)


def _closed(flow, depth: float) -> None:
    """Nothing left the plate: the film's volume and its dye's mass are the start's."""
    out, v0 = flow.read(), depth * float(flow.tables.area_np.sum())
    assert float(out["film_volume"][0]) == pytest.approx(v0, rel=1.0e-12, abs=0.0)
    assert float(out["species_mass"][0, 0]) == pytest.approx(10.0 * v0, rel=1.0e-12, abs=0.0)


def _band(y, dx: float, y0: float = 0.0, side: int = 0):
    """The band cells past the blade's ends (both, or one ``side``), the cell at its end left out."""
    d = (y - y0) * (side if side else 1.0)
    d = d if side else np.abs(d)
    return (d > BLADE[1] + dx) & (d <= BLADE[1] + 0.02)


SHALLOW = 1.0e-3                           # the blade cuts 0.75 mm: the bands have room for all of it


@pytest.fixture(scope="module")
def level():
    """The level blade's pass in the 1 mm paste on both grids (shared: the 1.25 mm one takes ~45 s on a CPU)."""
    out = {}
    for nx in (80, 160):
        flow, sim, x, y, dA = _pass(nx, depth=SHALLOW)
        spill = float(flow.read()["tool_roll_spill"][0, 0])
        out[nx] = (flow, sim, x, y, dA, _roll(flow, sim, x, y, dA, SHALLOW), spill)
    return out


@pytest.mark.timeout(600)
@pytest.mark.parametrize("nx", [80, 160])
def test_the_roll_holds_its_prism_and_the_rest_lies_beyond_the_ends(level, nx):
    """Level blade, lift 3 mm, in a 1 mm paste at a 0.5 mm gap: about 5 caps ploughed in 35 mm. The roll stays at the
    cap (within one row of cells), the bands take the rest at the paste's concentration, no bead stands above the
    paste + 3 mm, nothing lies past the bands, and the volume and species ledgers close."""
    flow, sim, x, y, dA, roll, spill = level[nx]
    h, dx = flow.depth(0)[0], LENGTH / nx
    assert spill > 4.0 * _cap()
    assert roll == pytest.approx(_cap(), abs=dx * 2.0 * BLADE[1] * 0.5e-3), (roll, _cap())
    assert float((h - SHALLOW)[_band(y, dx)].max()) <= LIFT + 1.0e-9
    assert float(np.abs(h[np.abs(y) > BLADE[1] + 0.02 + dx] - SHALLOW).max()) == 0.0
    wet = h > 1.0e-6
    dye = flow.read()["species"][0, :, 0].reshape(h.shape)
    assert np.allclose(dye[wet] / h[wet], 10.0, rtol=1.0e-9, atol=0.0)    # the tags carry the roll's concentration
    _closed(flow, SHALLOW)


@pytest.mark.timeout(600)
def test_the_cap_does_not_depend_on_the_cells(level):
    """WP15: the wedge's toe test must not make the cap grid-dependent: rolls and spills at 2.5 and 1.25 mm cells."""
    (r1, s1), (r2, s2) = ((level[nx][5], level[nx][6]) for nx in (80, 160))
    assert r2 == pytest.approx(r1, rel=0.10, abs=0.0) and s2 == pytest.approx(s1, rel=0.10, abs=0.0)


@pytest.mark.timeout(600)
def test_a_centred_pass_spills_alike_at_both_ends(level):
    """WP15: a mirror-symmetric pass lays the same at both ends (a cover threshold on float noise at the blade's
    ends gave one side 30 % more)."""
    flow, sim, x, y, dA, *_ = level[80]
    h = flow.depth(0)[0]
    band = [(y * sd > BLADE[1]) & (y * sd <= BLADE[1] + 0.02) for sd in (1, -1)]   # the cells at the ends included
    up, down = (float(((h - SHALLOW).clip(min=0.0) * m).sum() * dA) for m in band)
    assert up > 0.0 and up == pytest.approx(down, rel=1.0e-3, abs=0.0), (up, down)


def test_full_bands_leave_the_rest_in_the_roll():
    """WP6's check 2: in a 2 mm paste the blade cuts 1.75 mm x 70 mm per mm of travel, more than two 20 mm bands at
    most 3 mm high take. The bands fill to the top and what does not fit stays in the roll (GA's SPILL_WIDEN = 0)."""
    flow, sim, x, y, dA = _pass(80)
    h, dx = flow.depth(0)[0], LENGTH / 80
    band = _band(y, dx) & (x < float(sim.state.body_q.numpy()[0][0]) - BLADE[0])     # the bands behind the blade
    assert float((h - H0)[band].max()) == pytest.approx(LIFT, abs=1.0e-9)            # filled to the top
    assert _roll(flow, sim, x, y, dA) > _cap()
    _closed(flow, H0)


def test_a_roll_under_its_cap_is_left_bit_for_bit():
    """A shallow cut (0.05 mm over the pass) keeps its roll under the cap: the film is the uncapped run's, bit for
    bit, though the cap's geometry ran (the tool rolls) and nothing spilled."""
    capped, *_ = _pass(80, depth=GAP + 5.0e-5, periods=4)
    free, *_ = _pass(80, lift=None, depth=GAP + 5.0e-5, periods=4)
    assert int(capped.roll_cap.geom.on.numpy()[0, 0]) == 1
    assert float(capped.read()["tool_roll_spill"][0, 0]) == 0.0
    assert np.array_equal(capped.depth(0), free.depth(0))
    assert np.array_equal(capped.read()["species"], free.read()["species"])


def test_a_pitched_blade_holds_a_bigger_roll():
    """WP6's check 3: the leading edge raised 1° (rise 40 mm·sin 1° = 0.70 mm) raises the cap to ½(rise + lift)²·L,
    1.52 x the level blade's."""
    rise = 2.0 * BLADE[0] * math.sin(math.radians(1.0))
    flow, sim, x, y, dA = _pass(80, depth=SHALLOW, pitch=math.radians(1.0))
    assert _roll(flow, sim, x, y, dA, SHALLOW) == pytest.approx(_cap(rise), abs=(LENGTH / 80) * 2.0 * BLADE[1] * 0.5e-3)
    assert _cap(rise) / _cap() == pytest.approx(1.52, abs=0.01)


def test_a_blade_sliding_along_its_length_spills_beside_its_leading_corners():
    """The end leads (GA ``_end_leads``): L_face is the short side (40 mm) and the overflow lies beside the leading
    corners, in the 20 mm band each side; none past the leading end's band, none behind the start."""
    size = (0.035, 0.02)                                       # long axis along the slide
    flow, sim, x, y, dA = _pass(80, depth=SHALLOW, size=size)
    h, dx = flow.depth(0)[0], LENGTH / 80
    lead = float(sim.state.body_q.numpy()[0][0]) + size[0]
    assert float(flow.read()["tool_roll_spill"][0, 0]) > 4.0 * _cap(face=2.0 * size[1])
    side = (np.abs(y) > size[1] + dx) & (h > SHALLOW + 1.0e-9)
    assert side.any()
    assert float(x[side].max()) <= lead + 0.02 + dx                    # within the band past the leading end
    assert not (side & (x < X0 + size[0] - 0.02 - dx)).any()           # the leading end's band at the start
    assert float((h - SHALLOW)[side].max()) <= LIFT + 1.0e-9
    _closed(flow, SHALLOW)


@pytest.mark.timeout(600)
def test_a_band_a_chart_wall_cuts_lays_its_share_on_the_cells_left(level):
    """WP15: the blade's +y end runs 10 mm short of the chart's edge, so half that band is off the chart. Each substep
    the cut-off share stays in the roll and is offered again, so the roll stays at its cap, the pass spills as much
    as a centred one, and the cells left on the cut side stand as high as their mirror cells on the open side (the
    same depth per unit of weight, under the top); the ledgers close."""
    y0 = 0.5 * WIDTH - BLADE[1] - 0.01
    flow, sim, x, y, dA = _pass(80, depth=SHALLOW, y0=y0)
    h, dx = flow.depth(0)[0], LENGTH / 80
    assert _roll(flow, sim, x, y, dA, SHALLOW, y0=y0) == pytest.approx(_cap(), abs=dx * 2.0 * BLADE[1] * 0.5e-3)
    assert float(flow.read()["tool_roll_spill"][0, 0]) == pytest.approx(level[80][6], rel=0.01, abs=0.0)
    j_up = np.flatnonzero(_band(y[0], dx, y0, 1))                      # the cut side's band columns, outward
    j_down = np.flatnonzero(_band(y[0], dx, y0, -1))[::-1][:len(j_up)]   # their mirror columns
    assert len(j_up) >= 2
    assert np.allclose(h[:, j_up], h[:, j_down], rtol=0.0, atol=1.0e-6)
    assert float((h - SHALLOW)[:, j_up].max()) <= LIFT + 1.0e-9
    _closed(flow, SHALLOW)


def test_the_top_rides_on_the_lowest_paste_ahead():
    """WP15: the roll's top is the lowest strip's s0 + rise + lift. A paste 1.0 mm deep at -y and 1.5 mm at +y: no
    bead anywhere above 1.0 + 3 mm (the highest strip's s0 would let the +y bands stand to 4.5 mm)."""
    nx, ny = 80, 64
    yc = (np.arange(ny) + 0.5) * (WIDTH / ny) - 0.5 * WIDTH
    depth = np.tile(np.where(yc < 0.0, 1.0e-3, 1.5e-3), nx)
    flow, sim, x, y, dA = _pass(nx, depth=depth)
    h, dx = flow.depth(0)[0], LENGTH / nx
    assert float(flow.read()["tool_roll_spill"][0, 0]) > 0.0
    assert float(h[_band(y, dx)].max()) <= 1.0e-3 + LIFT + 1.0e-9
    assert float(h[_band(y, dx, side=1)].max()) > 1.0e-3 + LIFT - 0.5e-3   # the +y bands reach the top


def test_a_jointer_rod_rises_by_the_groove_not_its_radius():
    """SOFT-2's jointer: a 12.7 mm rod on a slide along a 9.5 mm groove of paste struck flush, resting on the arrises
    (bottom 2.14 mm below the faces). Its rise is the span of its underside over the cells the paste reaches, the
    groove's (1.18 mm at these cell centres), not over its whole footprint (2.74 mm, the faces beside the groove
    included: GA's ``ins`` on a chart wider than the paste)."""
    r, jw, depth, nx, ny = 0.00635, 0.0095, 0.005, 200, 40
    zax = math.sqrt(r * r - (0.5 * jw) ** 2)                        # the rod's axis above the faces

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.15, hy=0.03, hz=0.01, label="plate")
        xf = body_xform(X0, 0.0, zax + 0.02)
        rod = b.add_link(xform=xf, label="rod")
        b.add_articulation([b.add_joint_prismatic(-1, rod, parent_xform=xf, axis=newton.Axis.X)])
        b.add_shape_box(rod, hx=0.01, hy=0.01, hz=0.005, cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))

    def init(sim) -> None:
        qd = sim.state.joint_qd.numpy()
        qd[:] = U
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    rod = Applicator(body="rod", shape="roller", size=(r, 0.06865), center=(0.0, 0.0, -0.02), axis=(1.0, 0.0, 0.0),
                     stiffness=1.0e3, roll_lift=1.0e-3)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(nx, ny)),), material=sl.PASTE, inertia=False,
                       max_depth=0.03, tools=(rod,))
    sim = build_simulation(TestScene(build=build, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1,
                                     dt=2.0e-3), [flow], "cpu")
    y = flow.cell_centres(0)[..., 1]
    joint = np.abs(y) <= 0.5 * jw
    flow.set_input("base", np.where(joint, -depth, 0.0).reshape(1, -1).astype(np.float32))
    flow.set_depth(np.where(joint, depth, 0.0).reshape(1, -1))
    for _ in range(3):
        sim.step_period()
    yc = np.unique(np.round(y[0], 9))
    gap = zax - np.sqrt(np.clip(r * r - yc * yc, 0.0, None))         # the rod's underside over each column
    over_paste, under_rod = np.abs(yc) <= 0.5 * jw, np.abs(yc) < r
    rise = float(flow.roll_cap.geom.rise_lift.numpy()[0, 0]) - 1.0e-3
    assert rise == pytest.approx(float(np.ptp(gap[over_paste])), abs=1.0e-5)
    assert rise < float(np.ptp(gap[under_rod])) - 1.0e-3
    assert float(flow.read()["tool_roll_spill"][0, 0]) > 0.0


EDGER = {"edge_radius": 9.5e-3, "edge_lip": 12.7e-3, "edge_lip_w": 1.5e-3}   # GA's 136D edger's profile
ALONG = (0.035, 0.02)                      # long axis along the slide; an edger's lip on the -y side (normal x axis)


def test_an_edgers_rise_is_its_flats():
    """GA report §3.7: an edger's rise is its bearing flat's, not its lip's depth. Pitched 1°, in a paste deeper than
    the sole, it caps like the same rect: an edger is a rect with a profile (its long side is L_face), and the round
    and lip (12.7 mm deep) add no rise."""
    geoms = []
    for shape, extra in (("rect", {}), ("edger", EDGER)):
        flow, *_ = _pass(80, periods=1, depth=3.0e-3, size=ALONG, pitch=math.radians(1.0), shape=shape, **extra)
        g = flow.roll_cap.geom
        geoms.append([a.numpy()[0, 0] for a in (g.ldir, g.half, g.end, g.rise_lift, g.s_low, g.cap)])
    rect, edger = geoms
    np.testing.assert_allclose(rect[1], ALONG, rtol=1.0e-6)
    assert abs(abs(float(rect[0][0])) - 1.0) < 1.0e-6 and rect[2] == 1   # the long axis along the slide; the end leads
    for r, e in zip(rect[:3], edger[:3]):
        np.testing.assert_array_equal(np.abs(e), np.abs(r))
    rise = float(rect[3]) - LIFT
    assert rise == pytest.approx(2.0 * ALONG[0] * math.sin(math.radians(1.0)), abs=(LENGTH / 80) * 0.02)
    assert float(edger[3]) == pytest.approx(float(rect[3]), abs=1.0e-9)
    assert float(edger[4]) == pytest.approx(float(rect[4]), abs=1.0e-6)     # the profile's contact moves the pose


def test_an_edger_run_along_a_form_spills_away_from_its_lip():
    """GA report §3.7: an edger run along a form leaves its overflow beside the leading corner away from its lip; the
    lip, down the form's face, closes the roll's end on its side (GA lays nothing on the form's top)."""
    flow, sim, x, y, dA = _pass(80, depth=SHALLOW, size=ALONG, shape="edger", **EDGER)
    h, dx = flow.depth(0)[0], LENGTH / 80
    assert int(flow.roll_cap.geom.shut.numpy()[0, 0]) == -1
    assert float(flow.read()["tool_roll_spill"][0, 0]) > 2.0 * _cap(face=2.0 * ALONG[1])
    open_side, lip_side = y > ALONG[1] + dx, y < -(ALONG[1] + dx)
    assert (h[open_side] > SHALLOW + 1.0e-9).any()
    assert float(np.abs(h[lip_side] - SHALLOW).max()) == 0.0
    assert float((h - SHALLOW)[open_side].max()) <= LIFT + 1.0e-9
    _closed(flow, SHALLOW)


def _full_blade() -> Store:
    return Store(capacity=CARRY, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, initial=1.0, scrape=True)


CARRY = 1.0e-7                             # m³: the scraping blade's capacity (it starts full)


def _carry_ledgers(flow, w: int = 0, depth: float = SHALLOW) -> None:
    """Film, store (WP15's store_ledger books the off-chart share) and dye ledgers exact; the dye laid back is the
    laid volume at 10 kg/m³ (taken out with the cut as species_out, back in as supplied)."""
    r, v0 = flow.read(), depth * float(flow.tables.area_np.sum())
    off, spill = float(r["tool_overflow_off"][w, 0]), float(r["tool_roll_spill"][w, 0])
    assert float(r["ledger"][w]) == pytest.approx(v0, rel=1.0e-12, abs=0.0)
    assert float(r["store_ledger"][w]) == pytest.approx(CARRY, rel=1.0e-12, abs=0.0)
    assert float(r["species_ledger"][w, 0]) == pytest.approx(10.0 * v0, rel=1.0e-12, abs=0.0)
    assert float(r["species_supplied"][w].sum()) == pytest.approx(10.0 * spill, rel=1.0e-12, abs=1.0e-300)
    assert float(r["tool_overflow_off_sp"][w, 0, 0]) == pytest.approx(10.0 * off, rel=1.0e-12, abs=1.0e-300)
    assert float(r["tool_load"][w].sum()) == pytest.approx(CARRY, rel=1.0e-12, abs=0.0)


def test_a_full_scraping_blade_lays_its_overflow_beside_its_ends():
    """GA's carrying knife (coupling.py:597-602, 620-634; WP15's stores_lift.py): a full scraping store cuts on, and
    its cut beyond capacity lies in the same 20 mm end bands, with no top, at the cut's dye concentration. The load
    stays at capacity; film, store and dye ledgers close; nothing lands under the blade or past the band."""
    flow, sim, x, y, dA = _pass(80, lift=None, depth=SHALLOW, store=_full_blade())
    r, h, dx = flow.read(), flow.depth(0)[0], LENGTH / 80
    assert flow.overflow_on == 1
    spill, pending = float(r["tool_roll_spill"][0, 0]), float(r["tool_overflow"][0, 0])
    travel = float(sim.state.body_q.numpy()[0][0]) - X0
    assert spill >= (SHALLOW - GAP) * 2.0 * BLADE[1] * travel                 # at least the cut along the pass
    assert abs(pending) <= 1.0e-15 * spill and float(r["tool_overflow_off"][0, 0]) == 0.0
    _carry_ledgers(flow)
    tags = h > SHALLOW + 1.0e-9
    assert tags.any() and (np.abs(y[tags]) > BLADE[1]).all()                     # beside the blade's ends only
    assert (np.abs(y[tags]) <= BLADE[1] + 0.02 + dx).all()                       # within the band
    wet = h > 1.0e-9
    conc = flow.read()["species"][0, :, 0].reshape(h.shape)[wet] / h[wet]
    np.testing.assert_allclose(conc, 10.0, rtol=1.0e-12)


def test_a_blade_end_half_a_cell_off_the_grid_spills_alike():
    """WP15's review: band cells the blade itself covers by more than 25 % take none (GA's _cover(sole)), else the
    ends' cells are re-cut and re-laid every exchange (+18 % spill). With the sole cutting whole cells (WP15's
    d2b07a4dd; +3.9 % before it) shifting the ends half a cell moves the spill by 0.11 %: within 1 %."""
    ny = round(80 * WIDTH / LENGTH)
    spills = [float(_pass(80, lift=None, depth=SHALLOW, y0=y0, store=_full_blade())[0].read()["tool_roll_spill"][0, 0])
              for y0 in (0.0, 0.5 * WIDTH / ny)]
    assert abs(spills[1] / spills[0] - 1.0) < 0.01, spills


def test_a_blade_wider_than_the_chart_books_its_overflow_off_the_chart():
    """GA counts the off-chart share as gone (coupling.py:779-781): a 200 mm blade on a 160 mm plate lays nothing,
    and its whole overflow goes to tool_overflow_off at the cut's dye, none left pending."""
    flow = _pass(80, lift=None, depth=SHALLOW, store=_full_blade(), size=(0.02, 0.1))[0]
    r = flow.read()
    assert float(r["tool_roll_spill"][0, 0]) == 0.0 and float(r["tool_overflow_off"][0, 0]) > 0.0
    assert abs(float(r["tool_overflow"][0, 0])) <= 1.0e-15 * float(r["tool_overflow_off"][0, 0])
    _carry_ledgers(flow)


def test_two_worlds_lay_their_own_overflow():
    """1.0 and 1.5 mm of paste in two worlds: each world's ledgers close and world 0 is bitwise its one-world run."""
    one = _pass(80, lift=None, depth=SHALLOW, store=_full_blade())[0]
    two = _pass(80, lift=None, depth=(SHALLOW, 1.5e-3), worlds=2, store=_full_blade())[0]
    np.testing.assert_array_equal(two.h.numpy()[0], one.h.numpy()[0])
    assert two.read()["tool_roll_spill"][1, 0] > two.read()["tool_roll_spill"][0, 0]
    for w, depth in ((0, SHALLOW), (1, 1.5e-3)):
        _carry_ledgers(two, w, depth)


def test_roll_lift_is_checked():
    with pytest.raises(PluginError, match="roll_lift"):
        Applicator(body="tool", shape="rect", size=BLADE, roll_lift=-1.0e-3)
    with pytest.raises(PluginError, match="roll_lift"):
        Applicator(body="tool", shape="rect", size=BLADE, roll_lift=float("nan"))
    with pytest.raises(PluginError, match="roll_lift"):
        Applicator(body="tool", shape="roller", size=(0.02, 0.05), roll_lift=1.0e-3, store={"capacity": 1.0e-6})
