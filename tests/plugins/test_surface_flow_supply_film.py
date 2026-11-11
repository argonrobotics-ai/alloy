# SPDX-License-Identifier: Apache-2.0
"""The supply's sparse writes on WP15's film (CPU): with a jet, a spray and a drain the dry-tile skip matches the dense
step bit for bit (every cell the supply writes is flagged in ``cwet``); the per-exchange fields are zero outside the
box the supply wrote at the last exchange when jets move between charts or stop (a plane, a wrapped cup, two charts);
a strided paint film with a spray keeps the cross ledger with inflow pending, and a part-open drain takes its share
of the standing film per film step, not per exchange."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.supply import Drain, Nozzle
from alloy.plugins.surface_flow.supply_stub import CHART, SLAB, StubFilm, slab_scene

LEDGER = ("supplied", "spilled", "evaporated", "drained_total", "clipped")
SUPPLY = ("supply_sent", "supply_swept", "supply_drained", "supply_to_tank", "supply_overspray", "supply_mist",
          "pending_volume")
FIELDS = ("impact_p", "impact_share", "jet_cover", "jet_tau")   # the jets' momentum is pending (src_mom)
CUP = ((0.006, 0.0), (0.035, 0.0), (0.035, 0.14), (0.038, 0.14), (0.038, 0.0))     # floor, wall, rim, outside


def _network(*outlets: tuple[str, float], valve: float = 5.0e-7, closed: tuple[str, ...] = ()) -> dict:
    """Mains -> valve (about 10 mL/s at 5e-7 m²) -> line -> one orifice per outlet; a sink tank drains away."""
    edges = [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + 2e5, "cd_area": valve,
              "dp_lam": 1.0}]
    edges += [{"name": n, "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": a, "dp_lam": 1.0}
              for n, a in outlets]
    edges.append({"name": "waste", "kind": "orifice", "a": "sink", "reservoir": P_ATM, "cd_area": 2e-5, "dp_lam": 1.0})
    return {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4},
                      {"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16}],
            "edges": edges, "openings": {n: 0.0 for n in closed}}


def _skip_run(skip: bool, periods: int = 12) -> tuple[dict, list[float]]:
    tap = Nozzle("tap", edge="tap", pos=(-0.05, -0.05, 0.15), exit_area=2e-5)
    mist = Nozzle("mist", edge="mist", pos=(0.05, 0.05, 0.12), exit_area=2e-6, pattern="spray",
                  angle=math.radians(20), rays=16)
    grate = Drain("grate", chart=0, center=(-0.02, -0.05, 0.0), radius=0.008, to="sink")
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(96, 96), edges=("open",) * 4),), material="water",
                       nozzles=(tap, mist), drains=(grate,), network=_network(("tap", 2e-5), ("mist", 2e-6)),
                       skip_dry=skip)
    scene = TestScene(build=sl.plate_scene(0.192, 0.192), ground=False, worlds=2, dt=2.0e-3)
    sim = build_simulation(scene, [flow], "cpu")
    active = []
    for _ in range(periods):
        sim.step_period()
        active.append(float(flow.read()["tiles_active"][0]))
    r = flow.read()
    return {k: r[k] for k in ("h", "u", "q", *LEDGER, *SUPPLY)}, active


def test_skip_with_a_supply_matches_the_dense_step_bit_for_bit():
    """WP15's review (1): a cell given ``src_acc`` without ``cwet`` would stay pending under the skip (both ledgers
    still close, pending counting as supplied), so h, u, q and every ledger must match the dense step exactly."""
    dense, _ = _skip_run(skip=False)
    skip, active = _skip_run(skip=True)
    for name, a in dense.items():
        assert np.array_equal(a, skip[name]), f"{name}: max diff {np.abs(a - skip[name]).max()}"
    assert (dense["supply_sent"] > 0).all() and (dense["supply_swept"] > 0).all()
    assert (dense["supply_overspray"] == 0).all() and (dense["supply_drained"] > 0).all()
    assert min(active) < 1.0, active                                       # tiles were skipped: the test has teeth


def _in_box(p: StubFilm, w: int) -> np.ndarray:
    """Cells inside the box (per chart) the supply wrote at the last exchange."""
    t, box = p.tables, p.supply.written.numpy()[w]
    inside = np.zeros(t.cell_count, bool)
    for k, spec in enumerate(p.config.charts):
        (nx, ny), start = t.shapes[k], t.starts[k]
        i0, i1, j0, j1 = (int(v) for v in box[k])
        if i0 > i1:
            continue
        j = np.arange(ny)
        cols = ((j - j0) % ny <= j1 - j0) if isinstance(spec, RevolutionChart) else (j >= j0) & (j <= j1)
        rows = (np.arange(nx) >= i0) & (np.arange(nx) <= i1)
        inside[start:start + nx * ny] = (rows[:, None] & cols[None, :]).ravel()
    return inside


def _fields(p: StubFilm) -> np.ndarray:
    """[W, C]: nonzero where any per-exchange field is."""
    vals = [getattr(p, f).numpy() for f in FIELDS]
    return np.any([v != 0 for v in vals], axis=0)


def _two_slabs(b) -> None:
    slab_scene()(b)
    b.add_shape_box(-1, xform=body_xform(0.6, 0.0, SLAB[2]), hx=0.1, hy=0.1, hz=SLAB[2], label="slab2")


CLEAR_CASES = {
    # charts, scene, jet A, jet B
    "plane": ((CHART,), slab_scene(), (-0.1, -0.1), (0.1, 0.1)),
    "cup": ((CHART, RevolutionChart(profile=CUP, n_theta=12, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))),
            slab_scene(slab_x=1.0, crate=(-1.0, 0.0)), (0.02, 0.0), (1.05, 0.0)),
    "two_charts": ((CHART, PlaneChart("slab2", cells=(20, 20))), _two_slabs, (0.0, 0.05), (0.6, 0.0)),
}


@pytest.mark.parametrize("pattern", ["jet", "cone"])
@pytest.mark.parametrize("case", sorted(CLEAR_CASES))
def test_the_per_exchange_fields_hold_only_the_last_box(case, pattern):
    """WP15's review (2): jet A runs, then jet B (A closed: the jet moves), then neither (stopped). After every period
    impact_p, impact_share, jet_cover and jet_tau are zero outside the box written at the last exchange; once stopped,
    zero everywhere. The same for two 16-ray cones (their boxes: supply_shadow.boxes_written)."""
    charts, build, a, b = CLEAR_CASES[case]
    cone = {"pattern": "cone", "rays": 16, "angle": math.radians(20.0)} if pattern == "cone" else {}
    jets = (Nozzle("A", edge="a", pos=(*a, 0.3), exit_area=2e-5, **cone),
            Nozzle("B", edge="b", pos=(*b, 0.3), exit_area=2e-5, **cone))
    p = StubFilm(charts=charts, nozzles=jets, network=_network(("a", 2e-5), ("b", 2e-5), closed=("b",)))
    sim = build_simulation(TestScene(build=build, ground=True, worlds=2), [p], "cpu")
    seen = []
    for phase, (open_a, open_b) in enumerate(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))):
        p.set_input("a", open_a)
        p.set_input("b", open_b)
        for _ in range(3):
            sim.step_period()
            for w in range(2):
                on = _fields(p)[w]
                assert not on[~_in_box(p, w)].any(), (phase, w, int(on[~_in_box(p, w)].sum()))
        seen.append(_fields(p)[0].copy())
    a_only = seen[0] & ~seen[1]
    assert seen[0].any() and seen[1].any() and a_only.any()               # A wrote cells B's box does not hold
    assert not seen[2].any() and not _in_box(p, 0).any()                  # stopped: nothing written, all clear


def test_strided_paint_spray_keeps_the_cross_ledger_and_drains_per_film_step():
    """WP15's review (3) and (5): latex paint (yield stress: a 0.1 mm film stands still) steps every k > 1
    exchanges. A spray paints one corner (inflow pending between film steps), and a free drain half open (0.5) in
    the other corner takes half the standing film per film step: h = h0·0.5^n after n film steps (it took it all
    when its capacity was per exchange)."""
    mist = Nozzle("mist", edge="mist", pos=(-0.05, -0.05, 0.15), exit_area=2e-6, pattern="spray",
                  angle=math.radians(20), rays=16)
    cup = Drain("cup", chart=0, center=(0.05, 0.05, 0.0), radius=0.012, to="sink")
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 40)),), material="latex_paint", inertia=False,
                       max_depth=3.0e-4, max_stride=12, nozzles=(mist,), drains=(cup,),
                       network=_network(("mist", 2e-6)))       # 12: periods (2^k exchanges) end between film steps
    h0 = 1.0e-4

    def init(sim) -> None:
        flow.set_depth(h0)
        flow.set_input("drain_open", 0.5)
    scene = TestScene(build=sl.plate_scene(0.2, 0.2), init=init, ground=False, worlds=2)
    sim = build_simulation(scene, [flow], "cpu")
    pending = []
    for _ in range(8):
        sim.step_period()
        r = flow.read()
        sent = r["supply_sent"]
        cross = sent + r["supply_swept"] - (r["supplied_volume"] + r["pending_volume"] + r["supply_to_tank"]
                                            + r["supply_lost"] + r["supply_overspray"] + r["supply_mist"]
                                            + r["supply_caught"] + r["supply_lumped"])
        assert np.abs(cross).max() <= 1e-12 * sent.max(), cross
        assert np.allclose(r["drained_total_volume"], r["supply_drained"], rtol=1e-12, atol=1e-18)
        pending.append(float(r["pending_volume"][0]))
    assert flow.stride == 12 and flow.n_sub == 1, (flow.stride, flow.n_sub)
    assert (sent > 0).all() and max(pending) > 0.0, pending                 # inflow waited for a film step
    n = sim.steps // flow.stride                                           # film steps so far (tick % k == 0)
    cells = flow.supply.dcell.numpy()
    h = flow.read()["h"][:, cells]
    assert n >= 2 and np.allclose(h, h0 * 0.5 ** n, rtol=1e-6, atol=0.0), (n, h.min(), h.max(), h0 * 0.5 ** n)
