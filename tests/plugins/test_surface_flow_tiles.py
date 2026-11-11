# SPDX-License-Identifier: Apache-2.0
"""Surface flow, the dry-tile skip (phase 2): tiles cover each cell once, and stepping only the tiles near wet cells
matches the dense step bit for bit (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.tiles import TILE, splits

G = 9.81
LEDGER = ("supplied", "spilled", "evaporated", "drained_total", "clipped")


def test_tiles_cover_every_cell_once_and_wrap_in_theta():
    assert list(splits(40)) == [0, 13, 26, 40] and list(splits(16)) == [0, 16] and list(splits(5)) == [0, 5]
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 20)),
                               RevolutionChart(profile=((0.04, 0.1), (0.04, 0.0)), n_theta=48, ds=0.01)))
    scene = TestScene(build=sl.plate_scene(0.2, 0.1), ground=False, worlds=1)
    build_simulation(scene, [flow], "cpu")
    tl = flow.tl
    ctile = tl.ctile.numpy()
    first, rows, cols, pitch = (getattr(tl.tiles, k).numpy() for k in ("first", "rows", "cols", "pitch"))
    seen = np.zeros(len(ctile), int)
    for t in range(tl.count):
        cells = (first[t] + np.arange(rows[t])[:, None] * pitch[t] + np.arange(cols[t])[None, :]).ravel()
        assert (ctile[cells] == t).all()
        seen[cells] += 1
    assert (seen == 1).all()
    assert (rows >= TILE // 2).all() and (cols >= TILE // 2).all()
    near = tl.tiles.near.numpy()
    ring = flow.tables.starts[1]
    t_first, t_last = ctile[ring], ctile[ring + 47]                       # θ = 0 and the last θ column
    assert t_last in near[t_first] and t_first in near[t_last]            # the seam is a neighbour


def _water_drop(sim, flow) -> None:
    xy = flow.cell_centres(0)[..., :2]
    r2 = ((xy[..., 0] - 0.035) ** 2 + (xy[..., 1] + 0.035) ** 2) / 0.006 ** 2
    flow.set_depth(1.5e-3 * np.clip(1.0 - r2, 0.0, None), chart=0)


def _paste_pile(sim, flow) -> None:
    xy = flow.cell_centres(0)[..., :2]
    r2 = ((xy[..., 0] + 0.035) ** 2 + (xy[..., 1] + 0.035) ** 2) / 0.01 ** 2
    flow.set_depth(1.2e-2 * np.clip(1.0 - r2, 0.0, None), chart=0)


def _events_water(flow, nx: int, ny: int) -> None:
    h = flow.depth(0)
    h[:, nx // 2:, :] = 0.0                                                # a cloth wipes the running drop off
    flow.set_depth(h, chart=0)
    rate = np.zeros((nx, ny))
    rate[:3, -3:] = 2.0e-3                                                 # a tap starts over a dry corner
    flow.set_source(rate, chart=0)


def _events_paste(flow, nx: int, ny: int) -> None:
    rate = np.zeros((nx, ny))
    rate[-4:, -4:] = 5.0e-3                                                # more paste on the far, dry corner
    flow.set_source(rate, chart=0)


CASES = {
    # water with a contact angle and evaporation (every trace counts as wet), running down a tilted plate
    "water": (FilmMaterial("water_test", contact_angle=30.0, evaporation=1.0e-7), True, (-2.0, 0.0, -G), 2.0e-3,
              _water_drop, _events_water),
    # a Bingham pile slumping (the transverse yield estimate reads the previous substep's faces)
    "paste": (FilmMaterial("paste_test", density=2400.0, viscosity=2.0, yield_stress=20.0, surface_tension=0.0),
              False, (0.0, 0.0, -G), 1.3e-2, _paste_pile, _events_paste),
}


def _run(case: str, skip: bool, periods: int = 16) -> tuple[dict, list[float]]:
    mat, inertia, gravity, depth, init, events = CASES[case]
    nx = ny = 96                                                           # 1 mm cells, 6 x 6 tiles
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(nx, ny)),), material=mat, inertia=inertia,
                       max_depth=depth, skip_dry=skip)
    scene = TestScene(build=sl.plate_scene(0.096, 0.096), init=lambda sim: init(sim, flow), ground=False,
                      gravity=gravity, worlds=2, dt=2.0e-3)
    sim = build_simulation(scene, [flow], "cpu")
    active = []
    for k in range(periods):
        if k == periods // 2:
            events(flow, nx, ny)
        sim.step_period()
        active.append(float(flow.read()["tiles_active"][0]))
    r = flow.read()
    return {k: r[k] for k in ("h", "u", "q", *LEDGER)}, active


@pytest.mark.parametrize("case", ["water", "paste"])
def test_skip_matches_the_dense_step_bit_for_bit(case):
    dense, _ = _run(case, skip=False)
    skip, active = _run(case, skip=True)
    for name, a in dense.items():
        assert np.array_equal(a, skip[name]), f"{name}: max diff {np.abs(a - skip[name]).max()}"
    assert dense["h"].max() > 1.0e-4                                       # the film is there and moved
    assert min(active) < 1.0 and max(active) > min(active), active         # tiles were skipped, the set changed


def test_the_skip_stays_exact_under_a_wiper_on_trace_film():
    """WP2 review 1b: a wiper resting on the plate sweeps a region of trace film (below the dry threshold) two tiles
    away from the wet film. Dense stepping pushes the traces into a ridge; the skip must too, bit for bit."""
    from alloy._vendor import newton  # noqa: PLC0415

    from tests.plugins.test_surface_flow_tools import OIL, blade, scene  # noqa: PLC0415

    def run(skip: bool) -> tuple[np.ndarray, float]:
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                           skip_dry=skip, tools=(blade(stiffness=1.0e3),))

        def init(sim) -> None:
            x = flow.cell_centres(0)[..., 0]
            flow.set_depth(np.where(x < -0.07, 1.0e-3, np.where(x > -0.02, 5.0e-7, 0.0)), chart=0)
            qd = sim.state.joint_qd.numpy()
            qd[:] = 0.1                                                  # the slide: 0.1 m/s along x
            sim.state.joint_qd.assign(qd)
            newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
        sim = build_simulation(TestScene(build=scene(0.0, -0.01, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                         init=init, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=2.0e-3),
                               [flow], "cpu")
        for _ in range(20):
            sim.step_period()
        return flow.h.numpy(), float(flow.depth(0)[0].max())
    (dense, ridge), (skip, _) = run(False), run(True)
    assert ridge > 2.0e-6                                                # the traces were swept into a ridge
    assert np.array_equal(dense, skip)


@pytest.mark.parametrize("skip", [False, True])
def test_a_species_source_on_a_dry_cell_is_applied_once(skip):
    """WP15 review of WP9's flooded soil (f1): a species-only source on a cell with no water (eroded soil) is spread
    over the film substeps. The skip must keep the cell's tile on until the last substep applies and clears it;
    else the pending source stays and is added again every exchange."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(48, 48)),), material="water", max_stride=1,
                       species=("soil",), skip_dry=skip)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.048, 0.048), ground=False, gravity=(0.0, 0.0, -G),
                                     worlds=1, dt=1.0e-2), [flow], "cpu")
    area = flow.tables.area_np
    c = 40 * 48 + 7                                                       # far from everything: a dry cell
    added = 0.0
    for _ in range(4):
        src, wet = flow.sp.src.numpy(), flow.cwet.numpy()
        src[0, c, 0] += 3.0e-4                                            # kg/m², as the process stage adds it
        wet[0, c] = 1
        flow.sp.src.assign(src)
        flow.cwet.assign(wet)
        added += 3.0e-4 * area[c]
        sim.step_period()
    assert flow.n_sub > 1                                                 # the source is spread over substeps
    supplied = float(flow.read()["species_supplied"][0].sum())
    assert supplied == pytest.approx(added, rel=1.0e-12, abs=0.0)
    assert float(np.abs(flow.sp.src.numpy()).max()) == 0.0               # nothing left pending
    assert float(flow.sp.mass.numpy()[0, c, 0]) * area[c] == pytest.approx(added, rel=1.0e-12, abs=0.0)
