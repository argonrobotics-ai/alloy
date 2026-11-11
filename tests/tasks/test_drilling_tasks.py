# SPDX-License-Identifier: Apache-2.0
"""The six drilling task ports (Gym-Anything's drilling tasks, v1 checks): each task's worlds on CPU, the as-asked
world passing every check, each other world failing exactly the checks its script breaks, and every scored check
failing in some world (none passes by construction)."""

from __future__ import annotations

import re

import pytest

from alloy.tasks import (drilling_ledger_anchor, drilling_match_bolt, drilling_panel_holes, drilling_press_blind,
                         drilling_press_drill, drilling_timber_bolt)

pytestmark = [pytest.mark.law, pytest.mark.compile_heavy]

SPINDLE = ["spindle switched off and stopped", "quill within 2 mm of the top", "bit clear of the work"]
EXPECT = {
    "press_drill": (drilling_press_drill, {}, [
        [], [], ["hole A: entry within 0.5 mm of the mark"], ["hole A: axis within 1 deg of the face normal"],
        ["hole A: through hole at the mark"], ["hole A: through hole at the mark"], SPINDLE,
        ["hole A: diameter at most 9.25 mm"]]),
    "press_blind": (drilling_press_blind, {}, [
        [], ["full-diameter depth 15.0-15.5 mm"], ["axis within 1 deg"], ["full-diameter depth 15.0-15.5 mm",
                                                                         "not through"],
        ["entry within 0.5 mm of the mark"], SPINDLE]),
    "panel_holes": (drilling_panel_holes, {}, [
        [], [], ["hole 2: axis within 1 deg of the face normal"], ["hole 1: entry within 1.0 mm of the mark"],
        ["hole 0: through hole at the mark", "hole 3: through hole at the mark"]]),     # 3's nearest is hole 0
    "timber_bolt": (drilling_timber_bolt, {}, [
        [], ["square to the faces (axis within 1.19 deg of the face normal)"], ["entry within 2 mm of the mark"],
        ["through hole in the top board", "through hole in the bottom board", "carriage bolt through both boards",
         "bolt pushed in with less than 50 N"]]),
    "match_bolt": (drilling_match_bolt, {}, [
        [], ["angle hole 1: entry within 0.5 mm of the mark", "hole spacing within 0.5 mm of 60 mm"],
        ["angle hole 0: axis within 1 deg of the face normal"],
        ["angle hole 0: through hole at the mark", "angle hole 1: through hole at the mark",
         "bolt 0 passes through both parts", "bolt 1 passes through both parts"],
        ["angle hole 0: diameter at most 9.25 mm", "angle hole 1: diameter at most 9.25 mm",
         "two through holes in the flat bar", "bolt 0 pushed in with less than 50 N",
         "bolt 1 pushed in with less than 50 N"]]),
    "ledger_anchor": (drilling_ledger_anchor, {"marks": 1}, [
        [], ["hole 0: through the ledger within 3 mm of the mark", "hole 0: hole in the concrete at the mark",
             "no hole drilled away from the marks"],
        ["hole 0: axis within 3 deg of the wall normal", "hole 0: anchor straight (within 3 deg)"],
        ["hole 0: expansion engaged (cone drawn into a gripping clip)"],
        ["hole 0: nut torqued to T_inst (61-116 N m)"],
        ["hole 0: through the ledger within 3 mm of the mark", "hole 0: depth 108-125 mm",
         "hole 0: reinforcement damage at most 0.5 mm",
         "hole 0: anchor driven to h_nom (at least 93 mm) with its washer on the ledger",
         "hole 0: expansion engaged (cone drawn into a gripping clip)"]]),
}


@pytest.fixture(scope="module", params=sorted(EXPECT))
def scored(request):
    module, kw, expect = EXPECT[request.param]
    task_cls = next(v for k, v in vars(module).items() if isinstance(v, type) and k[0].isupper() and
                    hasattr(v, "score") and v.__module__ == module.__name__)
    verdicts, ep = module.run(task_cls(**kw), worlds=len(expect))
    return request.param, verdicts, expect, ep


def test_each_world_fails_exactly_what_its_script_breaks(scored):
    name, verdicts, expect, _ = scored
    got = [sorted(v.failed) for v in verdicts]
    assert got == [sorted(e) for e in expect], (name, [v.detail for v in verdicts], got)
    assert verdicts[0].success and verdicts[0].progress == 1.0


def test_every_scored_check_can_fail(scored):
    """Each scored check (per mark or bolt, as one kind) fails in at least one world: none passes by construction.
    What the scene cannot measure is listed in ``unscored`` instead, each with its reason."""
    name, verdicts, _, ep = scored
    kind = lambda n: re.sub(r"\b(hole|bolt) \d+", r"\1 k", n)                       # noqa: E731
    scored_kinds = {kind(c.name) for c in verdicts[0].checks}
    failing = {kind(n) for v in verdicts for n in v.failed}
    assert scored_kinds <= failing, (name, sorted(scored_kinds - failing))
    assert all(": " in u for v in verdicts for u in v.unscored), name                  # each with its reason
    assert ep.seconds > 0.0


GPU_LEDGER = list(EXPECT["ledger_anchor"][2])
GPU_LEDGER[3] = [f"hole {k}: expansion engaged (cone drawn into a gripping clip)" for k in range(4)]   # every hole
# bar_cut is the world's deepest cut, not per hole: hole 0's cut into the bar fails every hole's damage check
GPU_LEDGER[5] = GPU_LEDGER[5] + [f"hole {k}: reinforcement damage at most 0.5 mm" for k in (1, 2, 3)]


@pytest.mark.gpu
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("name", sorted(EXPECT))
def test_gpu_smoke_at_64_worlds(name):
    """GPU smoke (light-test rule: 64 worlds): every task on cuda:0 (ledger-anchor with all four marks: 136 s
    simulated, some 16 minutes on an L4, hence the ceiling), each world failing exactly what its script breaks, as on
    the CPU (worlds past the variants repeat the last one)."""
    import time

    module, kw, expect = EXPECT[name]
    if name == "ledger_anchor":
        kw, expect = {}, GPU_LEDGER
    task_cls = next(v for k, v in vars(module).items() if isinstance(v, type) and k[0].isupper() and
                    hasattr(v, "score") and v.__module__ == module.__name__)
    t0 = time.perf_counter()
    verdicts, ep = module.run(task_cls(**kw), worlds=64, device="cuda:0")
    wall = time.perf_counter() - t0
    got = [sorted(v.failed) for v in verdicts]
    want = [sorted(expect[min(w, len(expect) - 1)]) for w in range(64)]
    assert got == want, (name, [w for w in range(64) if got[w] != want[w]])
    print(f"GPU 64 worlds {name}: {ep.seconds:.1f} s simulated in {wall:.1f} s")
