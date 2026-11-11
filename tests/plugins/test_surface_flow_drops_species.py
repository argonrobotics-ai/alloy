# SPDX-License-Identifier: Apache-2.0
"""Surface-flow drops carry the film's species (DC5's review: drips carried water only, while GA's pendant drops carry
their soap and dye). CPU, harness: a soapy film under a plate both drips (its hanging cells) and runs off its open
edge; the drops land on a board below. Exact books."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.supply_spec import Drain

G, C0, H0 = 9.81, 5.0, 6.0e-3                            # m/s², kg/m³ of soap, m of film under the plate


def _under_and_board(b: newton.ModelBuilder) -> None:
    flip = wp.quat(1.0, 0.0, 0.0, 0.0)                    # the plate's top chart faces down: a hanging film
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), flip), hx=0.05, hy=0.02, hz=0.01, label="under")
    b.add_shape_box(-1, xform=body_xform(-0.05, 0.0, 0.19), hx=0.3, hy=0.1, hz=0.01, label="board")   # top z = 0.2
    dummy = b.add_body(xform=body_xform(2.0, 0.0, 1.0), label="dummy")                 # the harness needs a body
    b.add_shape_sphere(dummy, radius=0.01)


def _flow(drain: bool = False, **kw) -> SurfaceFlow:
    if drain:                                   # a grate over part of the plate's open edge (WP15's review probe)
        kw.update(drains=(Drain("grate", chart=0, center=(-0.05, 0.012, 0.0), radius=0.009, to="sink"),),
                  network={"nodes": [{"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16}], "edges": []})
    return SurfaceFlow(charts=(PlaneChart("under", cells=(20, 8), edges=("open", "wall", "wall", "wall")),
                               PlaneChart("board", cells=(30, 10))),
                       material="water", max_depth=8.0e-3, species=("soap",),
                       drops={"floor_z": 0.0, "tau_drip": 0.02}, **kw)


@pytest.mark.compile_heavy
def test_drips_and_shed_film_carry_their_soap_to_where_the_drops_land():
    """A 6 mm film with soap under a plate (5 kg/m³ in world 0, 2 in world 1), tilted gravity: it drips and runs off
    its open edge, part of which a drain takes, and its drops land on a board. Per world (all 1e-12): the drips take
    their share of the soap, so the hanging film keeps its concentration (it rose as it dripped before); the cells'
    shed soap is c x the volume they spilled and dripped; each drop takes its share, so the board's film is at c
    wherever it is wet and the landed soap is the board's supplied; the drops' soap closes (shed = pending + in flight
    + landed + floored + to tanks + to SPH); what the film lost = shed + drained = c x (spilled + dripped + drained),
    so the drain's cells shed none (WP15's S1: booking shed under a drain too double-counts); the film's shed
    species are all collected; the film's species ledger stays at the start's."""
    flow = _flow(drain=True)
    conc = np.array([C0, 2.0])

    def init(sim) -> None:
        flow.set_depth(H0, chart=0)
        flow.set_species(np.broadcast_to(conc[:, None, None] * H0, (2, 20, 8)), "soap", chart=0)
    run = run_scene(TestScene(build=_under_and_board, init=init, ground=False, gravity=(-3.0, 0.0, -G), periods=40,
                              worlds=2), [flow], "cpu")
    assert run.error is None, run.error
    r, d = flow.read(), flow.drops.read()
    A, h, mass = flow.tables.area_np, flow.h.numpy(), flow.sp.mass.numpy()[..., 0]
    under, board = (np.arange(len(A))[flow.tables.cells_of(k)] for k in (0, 1))      # the charts' cell indices
    assert (flow.drain_of.numpy() >= 0).any() and flow.drops.n_edges > 0
    assert (flow.sp.shed.numpy() == 0.0).all()                          # the drops collected every edge's species
    for w, c in enumerate(conc):
        assert min(r["dripped_volume"][w], r["spilled_volume"][w], r["drained_total_volume"][w],
                   d["landed_volume"][w]) > 0.0
        for cells in (under, board):
            wet = cells[h[w, cells] > 1.0e-6]
            assert len(wet) > 0
            np.testing.assert_allclose(mass[w, wet] / h[w, wet], c, rtol=1e-12, atol=0.0)
        shed = r["spilled_volume"][w] + r["dripped_volume"][w]
        assert d["species_shed"][w, 0] == pytest.approx(c * shed, rel=1e-12, abs=0.0)
        assert d["species_landed"][w, 0] == pytest.approx(c * d["landed_volume"][w], rel=1e-12, abs=0.0)
        gone = sum(d[f"species_{k}"][w, 0] for k in ("pending", "in_flight", "landed", "floored", "to_tanks",
                                                       "to_sph"))
        assert gone == pytest.approx(d["species_shed"][w, 0], rel=1e-12, abs=0.0)
        assert d["species_to_sph"][w, 0] == 0.0
        got = r["species_supplied"][w, board, 0].sum() + (flow.sp.src.numpy()[w, board, 0] * A[board]).sum()
        assert got == pytest.approx(d["species_landed"][w, 0], rel=1e-12, abs=0.0)
        out, tank = r["species_out"][w, :, 0].sum(), r["supply_species_tanks"][w, 0] + r["supply_species_away"][w, 0]
        assert out == pytest.approx(d["species_shed"][w, 0] + tank, rel=1e-12, abs=0.0)
        assert out == pytest.approx(c * (shed + r["drained_total_volume"][w]), rel=1e-12, abs=0.0)
        assert r["species_ledger"][w, 0] == pytest.approx(c * H0 * A[under].sum(), rel=1e-12, abs=0.0)


def test_film_species_with_sph_drops_warn_that_they_are_lost(caplog):
    """SPH particles carry no species, so with sph=True the film's species that leave with its drops are lost (booked
    species_to_sph): the build says so, once (no SPH liquid is needed for that; it binds at first use)."""
    with caplog.at_level(logging.WARNING, logger="alloy.plugins.surface_flow"):
        build_simulation(TestScene(build=_under_and_board, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=0.01,
                                   control_dt=0.01), [_flow(sph=True)], "cpu")
    assert sum("leave with SPH drops are lost" in m and "species_to_sph" in m for m in caplog.messages) == 1
