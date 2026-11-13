# SPDX-License-Identifier: Apache-2.0
"""Audit S1-002 / S1-006 (Gemini, 0.1.2): ``alloy.estimate`` on scenes of many mesh shapes.

- Its call time was its one-world assembly run twice, mostly ``mujoco_pairs``' explicit pair rows (one per shape
  pair: 16 decomposed bowls make 87,912): the estimate counts them and builds none.
- On the CPU route MuJoCo C tests every explicit pair each step, which the step time left out (32 bowls: 3.41x under
  on a spot n2d): ``COST["cpu"]["pair_s"]``, within 1.3x on the measured bowls rows.
- Mesh-on-mesh contact at the start (stacked bowls: 1.9-2.9x under on an L4) is named in ``caveats``.
- A VBD entry coupled to rigid bodies (the S1-006 scene's clamped hose) holds its own contact pipeline, which the GPU
  memory left out (1.58x under at 1,024 worlds): ``GPU_MEMORY["coupled"*]``, within 1.2x on the measured rows.
Rows: ``bench/results/estimate_calib/wp7_s1_rows.json`` (CPU)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.api import _estimate as E
from alloy.api import meshes as M
from alloy.scene import mujoco_pairs as MP

ROWS = Path(__file__).resolve().parents[2] / "bench/results/estimate_calib/wp7_s1_rows.json"
V, F = M.open_cylinder(inner_radius=0.04, depth=0.05, wall=0.006, segments=16)


def _bowl(s: alloy.Scene, name: str, pos, collision: str, r: float = 0.04, depth: float = 0.05) -> None:
    v, f = (V, F) if (r, depth) == (0.04, 0.05) else M.open_cylinder(inner_radius=r, depth=depth, wall=0.006,
                                                                       segments=16)
    s.add_mesh(name, vertices=v, faces=f, pos=pos, collision=collision, mass=0.4)


def _bowls(n: int, collision: str = "decompose") -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        _bowl(s, f"bowl_{i:03d}", ((i % side) * 0.16, (i // side) * 0.16, 0.02), collision)
    return s


def _stack(towers: int, high: int, collision: str = "convex") -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    for t in range(towers):
        for h in range(high):
            _bowl(s, f"bowl_{t}_{h}", ((t % 4) * 0.16, (t // 4) * 0.16, h * 0.0562), collision)
    return s


def _hose(s: alloy.Scene, y: float) -> alloy.Scene:
    s.add_rod("hose", length=0.34, radius=0.007, pos=(0.0, y, 0.2), direction=(1, 0, 0), segments=12, clamp="start")
    return s


def _obj10(hose: bool = True) -> alloy.Scene:
    """The S1-006 repro's scene: 4 decomposed bowls, 6 boxes and (``hose``) a clamped hose."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    for i in range(4):
        _bowl(s, f"bowl_{i}", (i * 0.15, 0.0, 0.02), "decompose", r=0.038, depth=0.045)
    for i in range(6):
        s.add_box(f"box_{i}", size=(0.08, 0.07, 0.05), pos=(i * 0.15, 0.20, 0.03), mass=0.4)
    return _hose(s, 0.4) if hose else s


def _rodboxes() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    for i in range(6):
        s.add_box(f"box_{i}", size=(0.08, 0.07, 0.05), pos=(i * 0.15, 0.20, 0.03), mass=0.4)
    return _hose(s, 0.8)


def _rows(device: str, pattern: str, key: str = "measured_step_s") -> list[dict]:
    """The rows file's measured rows on ``device`` whose scene name matches ``pattern`` whole."""
    rows = json.loads(ROWS.read_text())["rows"]
    return [r for r in rows if r["device"] == device and re.fullmatch(pattern, r["scene"]) and key in r]


@pytest.fixture(scope="module")
def cpu_route():
    mp = pytest.MonkeyPatch()
    mp.setenv("ALLOY_CPU_ROUTE", "mjc")
    yield
    mp.undo()


def test_the_estimate_counts_the_pair_rows_and_builds_none(monkeypatch):
    def no_rows(*args, **kwargs):
        raise AssertionError("the estimate built an explicit MuJoCo pair row")

    monkeypatch.setattr(MP, "_maybe_row", no_rows)
    est = alloy.estimate(_bowls(6, "convex"), worlds=4, device="cpu")
    assert est["counts"]["shapes"] == 7
    assert E._template_counts(_bowls(6, "convex"), "cpu")[6]["pair_rows"] == 21       # the 6 bowls' pairs, ground none


def test_the_cpu_step_time_holds_the_bowls_rows_with_the_pair_term(cpu_route):
    """Every measured CPU bowls row (1 to 32 decomposed bowls, 1 and 4 worlds) within 1.3x both ways; the estimate
    before the term read the 32 bowls 3.41x under."""
    rows = _rows("cpu", r"bowls\d+")
    assert {r["scene"] for r in rows} >= {"bowls1", "bowls4", "bowls8", "bowls16", "bowls32"}
    worst = []
    for r in rows:
        c, solvers, _, _, _, vbd, load = E._template_counts(_bowls(int(r["scene"][5:])), "cpu")
        assert load["pair_rows"] == r["pair_rows"]
        ratio = r["measured_step_s"] / E._step_time(c, solvers, r["worlds"], "cpu", vbd, {**load, "sph_active": 0.0})
        worst.append((max(ratio, 1 / ratio), r["scene"], r["worlds"]))
    assert max(worst)[0] <= 1.3, max(worst)
    b32 = next(r for r in rows if r["scene"] == "bowls32")
    assert b32["measured_step_s"] / b32["estimate_main_step_s"] > 3.0


def test_resting_mesh_on_mesh_contact_is_named_in_the_caveats():
    assert alloy.estimate(_bowls(4, "convex"), device="cpu")["caveats"] == []
    caveats = alloy.estimate(_stack(2, 2), device="cpu")["caveats"]
    assert len(caveats) == 1 and "mesh-on-mesh" in caveats[0] and "measure=True" in caveats[0]


MEMORY_SCENES = {"obj10": _obj10, "obj10nr": lambda: _obj10(hose=False), "rodboxes": _rodboxes,
                 "rodbowls16": lambda: _hose(_bowls(16), 0.8)}


def test_gpu_memory_holds_the_coupled_hose_rows(monkeypatch):
    """The S1-006 scene with and without its clamped hose, and the hose beside 6 boxes / 16 decomposed bowls, 1 to
    1,024 worlds on an L4: measured / predicted within 1.2x both ways (templates on MuJoCo Warp's route, as the GPU
    builds them); the estimate before the term read the S1-006 scene 1.58x under at 1,024 worlds, 2.78x at one."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    rows = _rows("cuda", "|".join(MEMORY_SCENES), key="measured_bytes")
    assert {r["scene"] for r in rows} >= {"obj10", "obj10nr", "rodboxes"}
    templates = {name: E._template_counts(make(), "cpu") for name, make in MEMORY_SCENES.items()}
    ratios = [(r["measured_bytes"] / E._gpu_memory(templates[r["scene"]], r["worlds"]), r["scene"], r["worlds"])
              for r in rows]
    worst = max(ratios, key=lambda x: max(x[0], 1 / x[0]))
    assert 1 / 1.2 <= worst[0] <= 1.2, worst
    s1006 = next(r for r in rows if (r["scene"], r["worlds"]) == ("obj10", 1024))
    assert s1006["measured_bytes"] / s1006["estimate_main_bytes"] > 1.5
    oom = next(r for r in _rows("cuda", "rodbowls16", key="failed") if r["worlds"] == 256)   # ran out on the 23 GB L4
    assert oom["failed"]["msg"].startswith("Failed to allocate")
    assert E._gpu_memory(templates["rodbowls16"], 256) > 0.95 * 23034 * 2**20 > oom["estimate_main_bytes"]
