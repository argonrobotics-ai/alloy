# SPDX-License-Identifier: Apache-2.0
"""pad-edge task pieces (alloy/tasks/pad_edge.py), CPU: the material wait, the lip mark, the edger's strokes, and
the corner law.

The wait is theirs (``env.py:88-111``, ``world.py:322-338``): the material clock alone, in 1 s steps, the film
unstepped, refused while a tool is down on the paste or the film could still flow. Over 10 min of the pad-edge start
it must age the paste by exactly that and bleed as their wait does; the resting pad, stepped instead, does not move.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.tasks import concrete_materials as CM
from alloy.tasks import pad_edge as PE
from alloy.tasks import pad_metrics as PM
from alloy.tasks import pad_run as PR
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR, evaporation_rate
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram

WINDOW = (0.12, 0.10)


def _build(task: PE.PadEdge):
    f = PE.start(task, 0)
    prog, _ = PE.program(task, f, t0=1.0)
    carriage, flow, layer, lip = PE.plugins(task, f, prog, profile=False)
    sim = build_simulation(TestScene(build=PE.build_scene(task, f), ground=False, worlds=1, dt=PE.EXCHANGE_DT),
                           [carriage, flow, layer, lip], "cpu")
    PE.set_start(flow, layer, [f])
    sim.step_period()
    return f, sim, carriage, flow, layer


def _their_wait_film(w0: float, age0: float, seconds: int, task: PE.PadEdge) -> float:
    """Their wait's bleed film (``paste.py:175-185``): 1 s steps, the clock first, bleed at the new age, float32."""
    mat = CM.Concrete()
    evap = np.float32(evaporation_rate(task.temperature, task.air) / 3600.0 / 1000.0)
    w, clock = np.float32(w0), age0
    for _ in range(seconds):
        clock += 1.0
        w = np.float32(w + np.float32(float(mat.bleed_rate(clock, task.temperature)) / 1000.0))
        w = np.float32(max(float(w - evap), 0.0))
    return float(w)


def test_the_wait_ages_and_bleeds_the_paste_as_their_wait():
    task = PE.PadEdge(window=WINDOW)
    f, sim, carriage, flow, layer = _build(task)
    age0, w0 = float(layer.read()["age"].max()), float(PE.edge_wet(layer, f)[0])
    assert PE.material_wait(sim, flow, layer, carriage, 600.0) == 600.0
    r = layer.read()
    assert np.all(r["age"][0, f["paste_mask"].reshape(-1)] == age0 + 600.0)
    assert float(PE.edge_wet(layer, f)[0]) == pytest.approx(_their_wait_film(w0, age0, 600, task), rel=1e-6, abs=0)


def test_skipping_the_film_is_exact_for_the_resting_pad():
    """The wait skips the film; stepped instead, the resting pad does not move: 25 periods leave its bits."""
    task = PE.PadEdge(window=WINDOW)
    f, sim, carriage, flow, layer = _build(task)
    assert PE.rest_ratio(flow, f["h"].shape, flow._dx) < 1.0
    h0 = flow.h.numpy().copy()
    for _ in range(25):
        sim.step_period()
    np.testing.assert_array_equal(flow.h.numpy(), h0)


def test_the_wait_is_refused_while_the_film_can_flow():
    """A 20 mm column of paste beside its 2 mm neighbours (rho·g·h·slope 2 kPa over 0.4 kPa): refused. A task's own
    rest rule, when given, decides instead (pad-finish's film held at rest)."""
    task = PE.PadEdge(window=WINDOW)
    f, sim, carriage, flow, layer = _build(task)
    h = flow.h.numpy().reshape(f["h"].shape).copy()
    i, j = h.shape[0] // 2, h.shape[1] // 2
    h[i, j] = 0.020
    flow.set_depth(h.reshape(1, -1))
    assert PE.rest_ratio(flow, f["h"].shape, flow._dx) > 1.0
    age0 = layer.read()["age"].copy()
    with pytest.raises(PluginError, match="can still flow"):
        PE.material_wait(sim, flow, layer, carriage, 60.0)
    with pytest.raises(PluginError, match="task's rule"):
        PE.material_wait(sim, flow, layer, carriage, 60.0, rest=lambda fl: False)
    np.testing.assert_array_equal(layer.read()["age"], age0)
    assert PE.material_wait(sim, flow, layer, carriage, 60.0, rest=lambda fl: fl is flow) == 60.0
    on = f["paste_mask"].reshape(-1)
    assert np.all(layer.read()["age"][0, on] == age0[0, on] + 60.0)


def test_the_lip_marks_only_the_cells_it_works():
    """Their lip mark needs the sole on the paste and sliding (coupling.py:641-642, 1003-1006, 1076-1080). The edger
    slides along the south form 15 mm over its working height (its ceiling still set), then at it: nothing marked,
    then the lip's cells."""
    task = PE.PadEdge(window=(0.20, 0.10))
    f = PE.start(task, 0)
    _, p0, p1, yaw = PE.sides(f)[0]
    pitch, z = PE.attitude(p0, p1, yaw)
    hover = (*p0, z + 0.015)
    keys = (Stroke(0.0, hover, yaw=yaw, pitch=pitch, mode=HEIGHT), Stroke(0.6, (*p1, z + 0.015), yaw=yaw, pitch=pitch,
            mode=HEIGHT), Stroke(0.8, (*p1, z + 0.015), yaw=yaw, pitch=-pitch, mode=HEIGHT),
            Stroke(1.1, (*p1, z), yaw=yaw, pitch=-pitch, mode=HEIGHT), Stroke(1.7, (*p0, z), yaw=yaw, pitch=-pitch,
            mode=HEIGHT))
    task = PE.PadEdge(window=(0.20, 0.10), cart=hover)
    carriage, flow, layer, lip = PE.plugins(task, f, ToolProgram("edger", keys))
    sim = build_simulation(TestScene(build=PE.build_scene(task, f), ground=False, worlds=1, dt=PE.EXCHANGE_DT),
                           [carriage, flow, layer, lip], "cpu")
    PE.set_start(flow, layer, [f])
    h0, ceiled = flow.h.numpy().copy(), False
    for _ in range(40):                                              # 0.8 s: the hovering slide
        sim.step_period()
        ceiled |= bool(np.isfinite(flow.read()["tool_ceiling"]).any())
    assert ceiled and lip.marks().sum() == 0
    np.testing.assert_array_equal(flow.h.numpy(), h0)                  # the hover cut nothing
    for _ in range(50):
        sim.step_period()
    marked = lip.marks().reshape(f["h"].shape) > 0
    d = PM.form_distances(f["x"], f["y"])["s"]
    assert marked.sum() > 10 and np.all(d[marked] < 0.004)             # the lip's cells by the south form


def test_the_wait_is_refused_with_the_edger_on_the_paste():
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    task = PE.PadEdge(window=WINDOW, cart=(-hx + 0.08, -hy + 0.05, FORM_H + PE.EDGER["half"][2]))
    f, sim, carriage, flow, layer = _build(task)
    age0 = layer.read()["age"].copy()
    with pytest.raises(PluginError, match="tool is down"):
        PE.material_wait(sim, flow, layer, carriage, 60.0)
    np.testing.assert_array_equal(layer.read()["age"], age0)


def _paste_at(f: dict, p: np.ndarray) -> bool:
    return bool(f["paste_mask"][int(np.abs(f["x"] - p[0]).argmin()), int(np.abs(f["y"] - p[1]).argmin())])


def test_the_strokes_put_the_lip_on_each_form():
    """Each run's lip face lies on the paste's edge: the cell a quarter cell inside it is paste, the one outside form.
    That is INTERIOR / 2 on the west, south and north, and half a cell (1 mm) inside it on the east, where the 1.219 m
    interior is 609.5 cells of the 2 mm grid (SC9's east edge, 11-07: a lip from INTERIOR / 2 half-covered the form's
    top cell and tipped the blade). The blade's ends stop at the cross forms."""
    whole = PE.start(PE.PadEdge(), 0)
    s = {side: (p0, p1, yaw) for side, p0, p1, yaw in PE.sides(whole)}
    assert sorted(s) == ["e", "n", "s", "w"]
    hx, hy, cell = INTERIOR[0] / 2, INTERIOR[1] / 2, float(whole["cell"])
    w, e, so, n = PR.paste_faces(whole)
    assert (w + hx, hx - e, so + hy, hy - n) == pytest.approx((0.0, cell / 2, 0.0, 0.0), rel=0, abs=1e-9)
    reach, L = PE.EDGER["half"][1] + PE.EDGER["lip_w"], PE.EDGER["half"][0]
    for side, (p0, p1, yaw) in s.items():
        toward = np.array([-math.sin(yaw), math.cos(yaw)])          # the sole's +y (lip side) in the pad frame
        face = np.asarray(p0) + reach * toward
        assert _paste_at(whole, face - 0.25 * cell * toward) and not _paste_at(whole, face + 0.25 * cell * toward), side
        along, (lo, hi) = (0, (w, e)) if side in "sn" else (1, (so, n))      # the blade's ends at the cross forms
        assert sorted((p0[along], p1[along])) == pytest.approx([lo + L, hi - L], rel=0, abs=1e-12)
    corner = PE.start(PE.PadEdge(window=WINDOW), 0)
    assert [side for side, *_ in PE.sides(corner)] == ["s", "w"]


def _has_edger_roll_cap() -> bool:
    """WP13's roll cap with the edger's rules (rise from its flat, no spill on the lip side)."""
    from alloy.plugins.surface_flow import tool_geometry
    return hasattr(tool_geometry, "lip_in")


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.skipif(not _has_edger_roll_cap(), reason="needs WP13's roll cap with the edger's rules")
def test_edging_the_south_west_corner_passes_pad_edge():
    """Law (CPU): their pad-edge start, 60 min old, a 0.37 x 0.37 m south-west corner at 2 mm (both runs long enough
    to lift mid-side, as on the whole pad). The scripted edger waits for the sheen (71 min), runs the south and west
    forms from the corner and back, pitched 0.3° with its roll capped (their 3 mm lift), and returns to the cart.
    Every pad-edge criterion the port judges passes on the whole window: 5 of their 8 (forms_undisturbed and
    arms_clear are not judged, tools_back in part: the edger is back on the cart); the film's ledger exact. The whole
    pad's run-end corners and its closing re-cut are the L4 row's (reconcile queue)."""
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    res = PE.run(PE.PadEdge(window=(0.37, 0.37), roll_lift=0.003, cart=(-hx + 0.185, -hy + 0.185, 0.30)))
    v = res.verdicts[0]
    assert res.success == 1, {k: c for k, c in v["criteria"].items() if c["passed"] is False}
    assert v["judged"] == [5, 8] and v["criteria"]["tools_back"]["judged_part"]       # the edger is back
    assert v["scene_differs"][-1].startswith("window:")                             # a corner, not the whole pad
    assert v["criteria"]["radius_profile"]["value"] > 0.97
    assert 60 * 60 < res.waited_s < 80 * 60 and res.material_s < PE.LIMIT_S
    assert res.ledger_rel < 1e-12
