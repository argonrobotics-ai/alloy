# SPDX-License-Identifier: Apache-2.0
"""spray-stencil on the surface-flow supply (CPU): the stencil mesh is the sheet cell for cell, the can's valve expels
the stand-in's wet rate, and on the crate cropped to the "3" the supply's can passes every check in two coats and
fails the coat in one. The stencil catches what it hides, nothing lands under it, and the supply's ledgers close
against the task's books. From 1.2 m the spray and mist leave over the box's rim."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks.paint_materials import paint
from alloy.tasks.paint_spray_stencil import (BOX, CELL, CONTROL_DT, CRATE_Z, Can, SprayLayout, rim_escape,
                                             schedule)
from alloy.tasks.paint_spray_supply import box_shapes, can_network, run_supply, sheet_quads

SMALL = SprayLayout(crate=(0.12, 0.19), stencil=(0.12, 0.19))        # the "3" with a margin: 22,800 cells
SHORT = dict(rows=(-0.08, -0.04, 0.0, 0.04, 0.08), span=0.10, pause=0.25, substeps=1)   # one aim a period


def open_time(can: Can) -> float:
    """How long the valve is open [s]: every control period of every row of every coat."""
    return can.coats * len(can.rows) * math.ceil(2.0 * can.span / (can.speed * CONTROL_DT) - 1e-9) * CONTROL_DT


def sent(v: dict) -> float:
    """The volume the can sent [m³], from its solids."""
    return v["books"]["emitted"] / paint("enamel_aerosol").volume_solids


def test_the_stencil_mesh_is_the_sheet_cell_for_cell() -> None:
    for layout in (SprayLayout(), SMALL):
        c = layout.centres()
        covered = np.zeros(c.shape[:2], dtype=int)
        for xa, xb, ya, yb in sheet_quads(layout):
            covered += (c[..., 0] > xa) & (c[..., 0] < xb) & (c[..., 1] > ya) & (c[..., 1] < yb)
        assert np.array_equal(covered, layout.sheet(c).astype(int))           # each sheet cell once, no other
        for xa, xb, ya, yb in sheet_quads(layout):                          # edges on the cells' edges
            assert all(abs(v / CELL - round(v / CELL)) < 1e-9 for v in (xa + 0.5 * layout.crate[0],
                                                                         ya + 0.5 * layout.crate[1]))


def test_the_box_and_the_valve() -> None:
    layout, can = SprayLayout(), Can()
    shapes = {label: (np.array(c), np.array(h)) for label, c, h in box_shapes(layout)}
    floor = layout.board_z - CRATE_Z
    c, h = shapes["box_floor"]
    assert c[2] - h[2] == pytest.approx(floor, abs=1e-12)
    for wall in ("box_wall_x0", "box_wall_x1", "box_wall_y0", "box_wall_y1"):
        c, h = shapes[wall]
        assert c[2] + h[2] == pytest.approx(floor + BOX[2], abs=1e-12)          # the rim
    assert shapes["box_wall_x1"][0][0] - shapes["box_wall_x1"][1][0] == pytest.approx(0.5 * BOX[0], abs=1e-12)
    assert shapes["box_wall_y1"][0][1] - shapes["box_wall_y1"][1][1] == pytest.approx(0.5 * BOX[1], abs=1e-12)
    valve = next(e for e in can_network(can)["edges"] if e["name"] == "valve")
    gauge = can_network(can)["nodes"][0]["pressure"] - 101325.0
    assert valve["cd_area"] * math.sqrt(2.0 * gauge / valve["density"]) == pytest.approx(can.wet_rate, rel=1e-12,
                                                                                         abs=0.0)


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("coats", "fails", "dry_um"), [(2, set(), (38.0, 46.0)), (1, {"coat"}, (17.0, 24.0))])
def test_spray_stencil_on_the_supply(coats: int, fails: set[str], dry_um: tuple[float, float]) -> None:
    can = dataclasses.replace(Can(), coats=coats, **SHORT)
    v = run_supply(SMALL, can, settle=0.25)
    assert v["dry_film"] == "stage"
    assert dry_um[0] < v["dry_target"]["p50_um"] < dry_um[1]
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == fails
    b = v["books"]
    assert v["under_sheet_um"] == 0.0 and v["rays_outside"] == 0.0     # the stencil hides, the seal holds
    assert b["on_sheet"] > 0.0 and b["outside"] == 0.0                  # 0.3 m: the nozzle is in the box
    assert v["books_closure"] < 1e-12 and v["runoff"] == 0.0
    assert sent(v) == pytest.approx(can.wet_rate * open_time(can), rel=1e-4, abs=0.0)   # the boiling holds the pressure


@pytest.mark.compile_heavy
def test_from_above_the_rim_the_spray_leaves_the_box() -> None:
    """A short pass 1.2 m over the crate and 0.15 m off its axis: the cone overflows the rim (rays that hit nothing)
    and the mist drifts over it."""
    can = dataclasses.replace(Can(), standoff=1.2, coats=1, rows=(0.15,), span=0.02, pause=0.0)
    v = run_supply(SMALL, can, settle=0.0)
    b = v["books"]
    # 20 aim steps: the valve answers an exchange late, so the step after the last aim sends the last one's paint
    assert sent(v) == pytest.approx(can.wet_rate * open_time(can), rel=1e-4, abs=0.0)
    # the mist over the rim in closed form: each step's mist leaves by its aim's share outside the rim (the first step
    # sends nothing, the step after the last aim sends at the last aim)
    aims = [a for period in schedule(can) if period for a in period]
    aims = aims[1:] + aims[-1:]
    mist = can.wet_rate * CONTROL_DT / can.substeps * (1.0 - can.efficiency) * paint("enamel_aerosol").volume_solids
    over_rim = sum(mist * rim_escape(a, can.above_rim) for a in aims)
    assert 0.0 < v["rays_outside"] and over_rim > 0.0
    assert b["outside"] == pytest.approx(v["rays_outside"] + over_rim, rel=2e-4, abs=0.0)
    assert b["outside"] > 0.05 * b["emitted"] and not v["checks"]["overspray"]["ok"]
    assert v["books_closure"] < 1e-12


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_full_crate_on_the_supply() -> None:
    """The whole 0.45 x 0.30 m crate (135,000 cells), the can's full schedule: about 10 minutes of CPU."""
    v = run_supply()
    assert v["succeeded"] and v["under_sheet_um"] == 0.0 and v["books_closure"] < 1e-12
    assert 43.0 < v["dry_target"]["p5_um"] and v["dry_target"]["p95_um"] < 46.5
