# SPDX-License-Identifier: Apache-2.0
"""Free bodies resting on many of their parts (the late ease-cpu report, 0.1.7): ``alloy.estimate`` on the CPU.

Three YCB scans (32 convex parts each) dropped into a bin stepped 2.0-2.9x the estimate once they rested (a spot n2d;
a 5 s run 1.96x there, 2.45x on an Intel n2): MuJoCo C's rk4 tests each resting part against its support in each of
its four stages, about 3x a ground plane's cost on a box. Laws (CPU; rows: ``bench/results/estimate_calib/
rest_rows.json``, spot n2d, each row measured 2-4 times):

- ``resting_parts`` counts the parts each free body rests on, from its start pose: MuJoCo touches that many at rest;
  a kinematic body is not counted. ``resting_on_shapes`` counts those on a box, mesh or body rather than the ground.
- The step time charges each body's parts beyond its first on MuJoCo C's rk4 (``COST["cpu"]["rest_s"]``,
  ``["rest_shape_s"]``), so separate bodies resting on one part each (16 boxes on a table) cost no more: every row of a
  body that settles within ``BOUND`` both ways (YCB scans in a bin, on a table and on the ground; one body of up to
  32 separate cubes; decomposed bowls of 8-64 segments upside down and upright; on the ground and on a box table; the
  calibration's bin; the YCB rows skip where the scans are not on disk). A body that keeps rocking on many parts (the
  16-segment bowl upside down: 3.3 solver iterations a step at rest) steps up to ``ROCKING`` x the estimate.
- Not charged on MuJoCo Warp's CPU route (no rows there) nor on implicitfast (robot scenes test resting parts once a
  step: ``test_estimate_calibration``'s Panda scene, whose mug rests on 24 of its 62 parts, holds its row).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.api import _estimate as E
from alloy.api import meshes as M
from tests.api.test_estimate_calibration import _bin
from tests.api.test_estimate_pairs import F, V, _bowls

ROWS = Path(__file__).resolve().parents[2] / "bench" / "results" / "estimate_calib" / "rest_rows.json"
BOUND, ROCKING, SETTLED_ITERS = 1.3, 2.5, 2.0
YCB = ("ycb_025_mug", "ycb_077_rubiks_cube", "ycb_005_tomato_soup_can")


def _base(table: bool) -> tuple[alloy.Scene, float]:
    """A scene with the ground and, with ``table``, a static 0.6 x 0.6 x 0.1 m box table; the height things stand on."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    if table:
        s.add_box("table", size=(0.6, 0.6, 0.1), pos=(0.0, 0.0, 0.05), static=True)
    return s, 0.1 if table else 0.0


def _upside_down(n: int, segments: int = 16, table: bool = False) -> alloy.Scene:
    v, f = (V, F) if segments == 16 else M.open_cylinder(inner_radius=0.04, depth=0.05, wall=0.006, segments=segments)
    s, z = _base(table)
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        s.add_mesh(f"cap_{i:02d}", vertices=v, faces=f, pos=((i % side) * 0.16, (i // side) * 0.16, 0.052 + z),
                   rpy=(np.pi, 0.0, 0.0), collision="decompose", mass=0.4)
    return s


def _upright(segments: int, table: bool = False) -> alloy.Scene:
    v, f = M.open_cylinder(inner_radius=0.04, depth=0.05, wall=0.006, segments=segments)
    s, z = _base(table)
    s.add_mesh("cap", vertices=v, faces=f, pos=(0.0, 0.0, 0.02 + z), collision="decompose", mass=0.4)
    return s


def _grid(n: int, table: bool = False, edge: float = 0.01, pitch: float = 0.02) -> alloy.Scene:
    """One free body of ``n`` separate 1 cm cubes on a 2 cm grid (one hull each), resting on all of them."""
    side = math.ceil(math.sqrt(n))
    unit = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) * edge
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tri = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    v = np.concatenate([unit + np.array([(i % side) * pitch, (i // side) * pitch, 0.0]) for i in range(n)])
    f = np.concatenate([tri + 8 * i for i in range(n)])
    s, z = _base(table)
    s.add_mesh("grid", vertices=v - v.mean(axis=0) * np.array([1, 1, 0]), faces=f, pos=(0.0, 0.0, 0.001 + z),
               collision="decompose", density=1000.0)
    return s


def _ycb3() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_box("bin", inner=(0.3, 0.3, 0.12), wall=0.01, pos=(0, 0, 0), static=True)
    for name, card, pos in zip(("mug", "cube", "can"), YCB, ((-0.07, 0.0, 0.2), (0.07, 0.0, 0.2), (0.0, 0.07, 0.3))):
        s.add_object(name, card, pos=pos)
    return s


def _ycb3_standing(table: bool) -> alloy.Scene:
    s, z = _base(table)
    for name, card, x in zip(("mug", "cube", "can"), YCB, (-0.12, 0.0, 0.12)):
        s.add_object(name, card, pos=(x, 0.0, z + 0.01))
    return s


def _scene(name: str) -> alloy.Scene:
    """A row's scene by its name in ``rest_rows.json`` (its ``scenes`` key says what each is)."""
    fixed = {"ycb3": _ycb3, "ycb3g": lambda: _ycb3_standing(False), "ycb3t": lambda: _ycb3_standing(True),
             "bin": _bin, "seg16up": lambda: _upright(16), "seg16upt": lambda: _upright(16, table=True)}
    if name in fixed:
        return fixed[name]()
    table = name.endswith("t")
    for prefix, make in (("inv", _upside_down), ("bowls", _bowls), ("grid", lambda n: _grid(n, table)),
                         ("seg", lambda k: _upside_down(1, segments=k, table=table))):
        if name.startswith(prefix):
            return make(int(name[len(prefix):].rstrip("t")))
    raise KeyError(name)


def _ycb_here() -> bool:
    from alloy.api import _anchor  # noqa: PLC0415

    try:
        for card in YCB:
            _anchor.local_box(card, ())
    except (alloy.SceneError, alloy.BuildError, KeyError, FileNotFoundError):
        return False
    return True


@pytest.fixture(scope="module")
def templates():
    """Each row scene's one-world template on the CPU's default route (MuJoCo C), as the rows were measured."""
    if not ROWS.is_file():
        pytest.skip("the calibration rows (bench/results/estimate_calib) are not shipped in the public tree")
    mp = pytest.MonkeyPatch()
    mp.setenv("ALLOY_CPU_ROUTE", "mjc")
    names = {r["scene"] for r in json.loads(ROWS.read_text())["rows"]}
    if not _ycb_here():
        names -= {"ycb3", "ycb3g", "ycb3t"}
    try:
        yield {name: E._template_counts(_scene(name), "cpu") for name in sorted(names)}
    finally:
        mp.undo()


def _step(template: tuple, worlds: int, resting: bool = True) -> float:
    """The estimate's CPU step time of a template (``resting=False``: as before the resting term)."""
    c, solvers, _, _, _, vbd, load = template
    load = {**load, "sph_active": 0.0, **({} if resting else {"rest_charged": 0, "rest_charged_on_shapes": 0})}
    return E._step_time(c, solvers, worlds, "cpu", vbd, load)


def _touching_pairs_at_rest(scene: alloy.Scene) -> int:
    """MuJoCo C's geom pairs in contact in world 0 after 0.6 s."""
    sim = alloy.build(scene, device="cpu")
    try:
        sim.step(60)
        d = sim.engine.stepper.solver.datas[0]
        return len({(int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon]})
    finally:
        sim.close()


def test_resting_parts_are_the_parts_mujoco_touches_at_rest(templates, monkeypatch):
    rows = json.loads(ROWS.read_text())["rows"]
    for r in rows:
        if r["scene"] in templates:
            load = templates[r["scene"]][6]
            assert load["resting_parts"] == r["resting_parts"], r["scene"]
            on = r["scene"] in ("ycb3", "ycb3t", "bin") or r["scene"].endswith("t")
            assert load["resting_on_shapes"] == (r["resting_parts"] if on else 0), r["scene"]
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    for name in ("seg24", "grid16t", "bin"):
        assert _touching_pairs_at_rest(_scene(name)) == templates[name][6]["resting_parts"], name
    s, _ = _base(False)
    s.add_mesh("cap", vertices=V, faces=F, pos=(0.0, 0.0, 0.052), rpy=(np.pi, 0.0, 0.0), collision="decompose",
               kinematic=True)
    s.add_box("box", size=(0.05, 0.05, 0.05), pos=(0.3, 0.0, 0.025), mass=0.2)
    assert E._template_counts(s, "cpu")[6]["resting_parts"] == 1            # the box; the driven cap does not rest


def test_the_cpu_step_time_holds_the_resting_rows(templates):
    rows = [r for r in json.loads(ROWS.read_text())["rows"] if r["scene"] in templates]
    assert {r["scene"][:3] for r in rows} >= {"inv", "bow", "bin", "gri", "seg"}
    settled, rocking = [], []
    for r in rows:
        load = templates[r["scene"]][6]
        ratio = r["measured_step_s"] / _step(templates[r["scene"]], r["worlds"])
        rocks = r["niter"] / len(r["resting"]) >= SETTLED_ITERS and not load["resting_on_shapes"]
        (rocking if rocks else settled).append((max(ratio, 1.0 / ratio), ratio, r["scene"], r["worlds"], r["pass"]))
    assert max(settled)[0] <= BOUND, max(settled)
    assert rocking and {name for _, _, name, _, _ in rocking} == {"inv1", "inv4", "inv8", "seg16"}, rocking
    assert max(rocking)[0] <= ROCKING, max(rocking)
    before = [r["measured_step_s"] / _step(templates[r["scene"]], r["worlds"], resting=False)
              for r in rows if r["scene"] in ("ycb3", "grid32t", "seg24t") and r["worlds"] == 1]
    assert min(before) > 1.4, before


def test_separate_bodies_resting_on_one_part_each_are_not_charged(monkeypatch):
    """16 separate 4 cm boxes on a box table: MuJoCo C, rk4, 16 resting parts, none charged (each body's first part
    is in the body terms; charging it read such piles 1.6-2.1x high)."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    s, z = _base(True)
    for i in range(16):
        s.add_box(f"box_{i:02d}", size=(0.04, 0.04, 0.04), pos=((i % 4) * 0.08 - 0.12, (i // 4) * 0.08 - 0.12,
                                                                z + 0.02), mass=0.1)
    t = E._template_counts(s, "cpu")
    load = t[6]
    assert load["mujoco"] == "c" and load["rk4"] and load["resting_parts"] == load["resting_on_shapes"] == 16
    assert load["rest_charged"] == load["rest_charged_on_shapes"] == 0
    assert _step(t, 1) == _step(t, 1, resting=False)


def test_mujoco_warp_on_the_cpu_is_not_charged(monkeypatch):
    """MuJoCo Warp's CPU route (``ALLOY_CPU_ROUTE=warp``) has no resting rows: its step is the one before the term."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    t = E._template_counts(_upside_down(2), "cpu")
    assert t[6]["mujoco"] == "warp" and t[6]["rk4"] and t[6]["resting_parts"] == 32
    assert _step(t, 4) == _step(t, 4, resting=False)
