# SPDX-License-Identifier: Apache-2.0
"""rinse_pan end to end on the CPU, cut down: the scrub on the flooded floor (the full task runs about 230 s
simulated, 90 min on a CPU; its row is on the L4)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.rinse_books import Books
from alloy.tasks.rinse_pan import PanTask, setup
from alloy.tasks.rinse_pan_policy import scrub_floor
from alloy.tasks.rinse_pan_scene import Station
from alloy.tasks.rinse_pan_verdict import on_ledge
from alloy.tasks.rinse_vessels import Pan

WIPE = 3                              # soil_removed_by's tool column (impact, shear, front, tool)


class Dunks:
    """A stand-in runner that books the rinse's dunks (bowl, soak)."""

    def __init__(self) -> None:
        self.dunks: list[tuple[str, float]] = []

    def dunk(self, i: int, bowl: str, depth: float, soak: float, pace: float = 1.0, pour: bool = True) -> None:
        assert i == 0 and not pour
        self.dunks.append((bowl, soak))


@pytest.mark.parametrize(("looks", "made"), [(None, 2), ([False], 2), ([True, False], 3), ([True, True, True], 4)])
def test_the_rinse_dunks_again_while_a_pan_is_not_visibly_clean(monkeypatch, looks, made):
    """``rinse``: the task's two dunks, then one more each time a look finds a pan not visibly clean, at most
    ``extra_rinses`` (2) more; without a look the schedule is the open-loop one (WP5's 100k pass, 11-09: world 6, the
    heaviest egg of 16, entered the sanitizer soiled and its clock never ran). Each dunk is poured out."""
    import alloy.tasks.rinse_pan as rp  # noqa: PLC0415

    pours: list[str] = []
    monkeypatch.setattr(rp, "lift_and_pour", lambda run, st, bowl, pace=1.0: pours.append(bowl))
    task, run = PanTask(), Dunks()
    seen = iter(looks or [])
    dirty = None if looks is None else (lambda: next(seen))
    assert rp.rinse(task, run, dirty) == made
    assert run.dunks == [("rinse", task.rinse_s)] * made and pours == ["rinse"] * made


def test_the_look_reads_the_worst_gauge_of_every_world_against_half_the_visible_limit(monkeypatch):
    """``visibly_dirty``: true when any world's worst 1 cm soil gauge (the one the sanitizer's entry check reads) is
    over half the 1 ug/cm² visible-residue limit."""
    from types import SimpleNamespace  # noqa: PLC0415

    import alloy.tasks.rinse_pan as rp  # noqa: PLC0415
    from alloy.tasks.rinse_tracker import VRL_KG_M2  # noqa: PLC0415

    soil = np.zeros((2, 3))
    flow = SimpleNamespace(process=SimpleNamespace(soil=SimpleNamespace(numpy=lambda: soil)))
    worst = {"g": np.array([0.4, 0.6]) * VRL_KG_M2}
    monkeypatch.setattr(rp, "gauges", lambda f, s, chart: worst["g"] if s is soil and chart == 0 else None)
    look = rp.visibly_dirty(flow)
    assert look() is True
    worst["g"] = np.array([0.4, 0.1]) * VRL_KG_M2
    assert look() is False


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_one_scrub_pass_takes_egg_off_the_flooded_floor_and_the_books_close():
    """The pan in the wash bowl under the wash water (its floor flooded), soaked egg on the floor: the sponge takes one
    pass along each of the four strips and goes back to the ledge. The pad wipes egg off the flooded cells (finite:
    the wipe's gap floors at h_wet, WP9's 889feb806; it divided by zero there before), most of what comes off is the
    tool's, the soil's books close (soil + removed = start), so do ``rinse_books``' (the film's water, the species with
    the walls' load, the supply's cross ledger), the sponge is on the ledge and no substep overflowed its contacts."""
    task = PanTask(station=Station(weir=("wash", "rinse", "sanitize"), pan=Pan(ds=5.0e-3, n_theta=24)), passes=1)
    sim, flow, run_ = setup(task, 1, "cpu")
    books = Books.start(flow)
    start = float(flow.process.read()["soil_start_mass"][0])
    run_.grip[:, 0] = 1.0
    scrub_floor(run_, task.station, task.press, task.passes)
    pr, fr = flow.process.read(), flow.read()
    assert np.isfinite(flow.process.soil.numpy()).all() and np.isfinite(fr["h"]).all()
    by = pr["soil_removed_by_mass"][0]
    assert by[WIPE] > 0.05 * start and by[WIPE] > by[:WIPE].sum(), by
    soil = float(pr["soil_mass"][0] + pr["soil_removed_mass"][0])
    assert abs(soil - start) <= 1e-9 * start
    closed, worst = books.closed(flow)
    assert closed.tolist() == [True] and set(books.residuals(flow)) == {"water", "species", "supply"}, worst
    sponge = task.station.scour_face(sim.state.body_q.numpy()[run_.tools.held.numpy()[0, 0]])
    assert on_ledge(task.station.sink, sponge), sponge
    assert not run_.overflowed.any()
