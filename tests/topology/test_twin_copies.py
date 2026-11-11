# SPDX-License-Identifier: Apache-2.0
"""TOP-2 on virtual-node copies against the seam twin (alloy/topology/twin_copies.py; WP10 item 9 step 4), CPU.

Coordinator ruling 2026-11-06T18:59Z (option a): the copies are judged on the pre-cut segment and on the cut timing.
- Pre-cut segment: from first contact below the top (depth 0) to the first cut, the knife force within 15 % of the
  seam twin's (measured 2.1 % pointwise; peak 0.613 vs 0.615 N, mean 0.421 vs 0.422 N).
- Cut timing: the first node cut at the same knife depth (step 205 vs 206, 1.15 vs 1.16 mm).
- Past the first cut the seam twin holds 0.65-0.95 N. That plateau is a contact artefact of the reference: its cut
  twins stay stuck under the edge with stock contacts, and it reads ~0 N with BladeContact. It is recorded as a known
  gap of the reference (docs/notes/wp10-results.md). The copies' cut nodes go to the faces and the force falls to
  ~0.2 N until the edge reaches the next layer. The whole stroke is only guarded loosely here (today: peak 0.78 N,
  mean 0.44 N), so a broken work rule or bond still fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.testing.topology import ArrayTwinLog
from alloy.topology.laws import TOP2_KNIFE_FORCE
from alloy.topology.twin import KnifeForceTwin
from alloy.topology.twin_copies import KnifeForceTwinCopies
from alloy.topology.vnode_work import PHASE_CUT

SAMPLE = {"softness": 1.0, "cut_spring_ke": 1000.0}


def _first_cut(solver) -> bool:
    if hasattr(solver, "cutter"):
        return bool((solver.cutter.work_rule.cut.numpy() == PHASE_CUT).any())
    return bool((solver.controller.bonds.intact.numpy() == 0).any())


def _stroke(twin) -> tuple[np.ndarray, np.ndarray, int]:
    """(depth [N] m, knife force [N] N, the step of the first cut node) for one TOP-2 world (agar, softness 1)."""
    dt = TOP2_KNIFE_FORCE.dt
    steps = int(TOP2_KNIFE_FORCE.duration_s / dt)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.begin_world()
    twin.build_world(b, twin.conditions(quick=True)[0], SAMPLE)
    b.end_world()
    b.color()
    m = b.finalize(device="cpu")
    solver = twin.make_solver(m, "vbd")
    pipe = twin.make_pipeline(m, solver)
    contacts = pipe.contacts()
    log = ArrayTwinLog(1, dt, steps, device="cpu")
    s0, s1, ctrl = m.state(), m.state(), m.control()
    first = -1
    for k in range(steps):
        twin.drive(k, k * dt, s0, ctrl)
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        twin.record(k, s1, contacts, None, log)
        s0, s1 = s1, s0
        if first < 0 and _first_cut(solver):
            first = k
    depth = twin.top - (twin._worlds[0].z0 - twin.speed * log.times() - 1.0e-4)
    return depth, log.series("force")[0], first


@pytest.fixture(scope="module")
def strokes():
    return _stroke(KnifeForceTwin()), _stroke(KnifeForceTwinCopies())


def test_copies_cut_at_the_seam_twins_depth(strokes) -> None:
    (depth, _, k_seam), (_, _, k_copies) = strokes
    assert k_seam > 0 and k_copies > 0
    assert abs(depth[k_copies] - depth[k_seam]) <= 1.0e-4                     # 0.1 mm of knife travel (measured 0.01)


def test_copies_press_the_knife_as_the_seam_twin_before_the_cut(strokes) -> None:
    (depth, seam, k_seam), (_, copies, k_copies) = strokes
    seg = (depth >= 0.0) & (np.arange(len(depth)) <= min(k_seam, k_copies))
    assert seg.sum() > 100
    np.testing.assert_allclose(copies[seg], seam[seg], rtol=0.15)
    assert copies[seg].max() == pytest.approx(seam[seg].max(), rel=0.15)
    assert copies[seg].mean() == pytest.approx(seam[seg].mean(), rel=0.15)


def test_the_whole_copies_stroke_stays_in_its_measured_band(strokes) -> None:
    """Past the first cut: the reference's known gap (module docstring). A loose band on today's stroke: peak
    0.73 N, mean 0.25 N over depth >= 0 (seam twin 1.69 / 0.67 N). Until 2026-11-10 the copies read 0.78 / 0.44 N:
    their colour array was left as built (``tet_copies.recolor_model``), so a copy's weld acted on its hub only and
    the cut copies kept pressing the knife (0.45 N mean after the first cut, 0.17 N now)."""
    (depth, _, _), (_, copies, _) = strokes
    cut = depth > 0.0
    assert np.isfinite(copies).all()
    assert 0.5 < copies[cut].max() < 1.0 and 0.15 < copies[cut].mean() < 0.35


@pytest.mark.parametrize("make", [KnifeForceTwin, KnifeForceTwinCopies])
def test_both_twins_drive_the_knife_with_its_descent_velocity(make) -> None:
    """VBD's contact friction and damping read the knife body's velocity: a knife only posed would slide past the
    material with no friction, so the two twins would differ in friction as well as in the cut model (WP15, gap #9)."""
    twin = make()
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.begin_world()
    twin.build_world(b, twin.conditions(quick=True)[0], SAMPLE)
    b.end_world()
    b.color()
    m = b.finalize(device="cpu")
    twin.make_solver(m, "vbd")
    s = m.state()
    twin.drive(3, 3.0e-3, s, m.control())
    knife = twin._worlds[0].knife
    np.testing.assert_allclose(s.body_qd.numpy()[knife], [0.0, 0.0, -twin.speed, 0.0, 0.0, 0.0], atol=0.0)
    assert s.body_q.numpy()[knife][2] == pytest.approx(twin._worlds[0].z0 - twin.speed * 3.0e-3, abs=1e-7)
