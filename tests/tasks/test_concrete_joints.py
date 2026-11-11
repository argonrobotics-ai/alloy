# SPDX-License-Identifier: Apache-2.0
"""tool-joints (concrete surface-flow proof): the start port, the lanes, the trowel's strips, the whole wall's schedule
against the thumbprint window, and one jointer stroke on CPU."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from alloy.tasks import concrete_joints as J
from alloy.tasks import concrete_joints_schedule as SCH
from alloy.tasks import concrete_joints_start as JS
from alloy.tasks import concrete_joints_strips as JSTR
from alloy.tasks import concrete_materials as CM


def _xz(f: JS.Infill) -> tuple[np.ndarray, np.ndarray]:
    n = f.joint.shape
    x = f.origin[0] + (np.arange(n[0]) + 0.5)[:, None] * f.cell + np.zeros((1, n[1]))
    z = f.origin[1] + np.zeros((n[0], 1)) + (np.arange(n[1]) + 0.5)[None, :] * f.cell
    return x, z


def test_start_is_their_infill() -> None:
    """Their region (layout.py:420-484): 853 x 589 cells of 1.5 mm, joints 20 mm deep struck flush (0.4 mm rms
    noise), the short head joint 3 mm low over 280 cells, crumbs only on faces, courses 45/37/28/20 min old."""
    f = JS.infill(0)
    assert f.joint.shape == (853, 589)
    j = f.joint
    assert np.all(f.base[j] == -JS.BED_DEPTH) and np.all(f.base[~j] == 0.0)
    top = f.top[j & ~f.void]
    assert float(top.std()) == pytest.approx(JS.FLUSH_NOISE, rel=0.05) and abs(float(top.mean())) < 1e-4
    assert int(f.void.sum()) == 280 and np.allclose(f.top[f.void], -JS.VOID_SHORT, rtol=0, atol=1e-12)
    assert f.crumbs.any() and not (f.crumbs & j).any() and np.allclose(f.top[f.crumbs], JS.CRUMB_T, rtol=0, atol=0)
    tau = CM.MasonryMortar().tau_bulk(np.array(JS.AGES_MIN) * 60.0 * CM.masco_factor(21.1))
    assert tau / 1e3 == pytest.approx([18.0, 10.45, 5.1, 2.49], abs=0.02)       # Type N, as the spec


def test_rigid_faces_cover_everything_but_the_joints() -> None:
    f = JS.infill(0)
    x, z = _xz(f)
    solid = np.zeros(f.joint.shape, bool)
    for a, b, c, d in JS.blocks():
        solid |= (x >= a) & (x < b) & (z >= c) & (z < d)
    assert not (solid & f.joint).any() and (solid | f.joint).all()


def test_every_joint_cell_lies_under_a_lane() -> None:
    """Each joint cell is within half a joint (plus a cell) of some lane's centre line."""
    f = JS.infill(0)
    x, z = _xz(f)
    near = np.zeros(f.joint.shape, bool)
    for ln in J.joint_lanes():
        (a, b), (c, d) = ln.p0, ln.p1
        lo_x, hi_x, lo_z, hi_z = min(a, c), max(a, c), min(b, d), max(b, d)
        reach = JS.JOINT_W / 2 + f.cell
        near |= (x >= lo_x - reach) & (x <= hi_x + reach) & (z >= lo_z - reach) & (z <= hi_z + reach)
    assert near[f.joint].all()
    assert len(J.joint_lanes()) == 23


def test_trowel_strips_stay_off_the_scored_cells_and_cover_the_faces_beside_the_joints() -> None:
    """A loaded blade fills the lows under it, so over a tooled joint it refills the groove: no strip's swept blade
    (soft edges and the clearance included) may touch a cell the verdict scores, on the whole wall; together they cover
    every face cell within 45 mm of a joint fully (the tags lie within ~30 mm), and the faces a head joint's roll
    reaches: the rod's leading end lays it up to 95 mm past the lane's ends, over the bed and onto the next course.
    Without the centre-line strips past the beds (92 strips) the whole wall kept 8.4-8.9 cm2/m of smear there."""
    f = JS.infill(0)
    x, z = _xz(f)
    scored = f.joint & (distance_transform_edt(f.joint) * f.cell >= 0.75 * JS.JOINT_W / 2)
    hl, hw = J.TROWEL_HALF[:2]
    h = f.cell / 2
    full = np.zeros(f.joint.shape, bool)
    plan = JSTR.plan(J.joint_lanes(), JSTR.scored(f), (hl, hw), 0.10)
    for st in plan:
        ex, ez = (hl, hw) if st.yaw == 0.0 else (hw, hl)
        x0, x1 = min(st.a[0], st.b[0]) - ex, max(st.a[0], st.b[0]) + ex
        z0, z1 = min(st.a[1], st.b[1]) - ez, max(st.a[1], st.b[1]) + ez
        touch = (x > x0 - h) & (x < x1 + h) & (z > z0 - h) & (z < z1 + h)
        assert not (touch & scored).any(), st
        full |= (x >= x0 + h) & (x <= x1 - h) & (z >= z0 + h) & (z <= z1 - h)
    near = ~f.joint & (distance_transform_edt(~f.joint) * f.cell <= 0.045)
    assert full[near].all() and len(plan) == 116
    reach = J.ROD_L / 2 + J.ToolJoints().run_in
    for ln in J.joint_lanes():
        if ln.p0[0] != ln.p1[0]:
            continue                                            # a bed's rolls run on to the wall's sides
        lo, hi = sorted((ln.p0[1], ln.p1[1]))
        tail = ~f.joint & (np.abs(x - ln.p0[0]) <= J.ROD_R + h) & (z >= lo - reach) & (z <= hi + reach)
        assert full[tail].all(), ln


@pytest.mark.parametrize("window", [(3.86, 4.06, 0.783, 1.04), (4.40, 4.60, 1.40, 1.6665), (4.10, 4.22, 0.80, 0.835),
                                    None])
def test_programs_build_for_windows_and_wait_only_for_the_youngest_course(window) -> None:
    """Keyframes increase for both tools (a strip wholly outside a window is a single placement); the tools park
    for the age bump only when the window holds the youngest course."""
    task = J.ToolJoints(window=window)
    w = J.wall(task)
    jointer, trowel, t_wait, duration = J.program(task, w)
    assert jointer.keys[-1].t < duration and trowel.keys[-1].t < duration
    win = (w.f.origin[0], w.f.origin[0] + w.size[0], w.f.origin[1], w.f.origin[1] + w.size[1])
    youngest = any(ln.course == JS.COURSES - 1 for ln in (J._clip(x, win) for x in J.joint_lanes()) if ln)
    assert (t_wait < jointer.keys[-1].t) == youngest


def _ripe(task: J.ToolJoints) -> SCH.Ripe:
    w = J.wall(task)
    jointer, _, t_wait, _ = J.program(task, w)
    return SCH.ripe_share(jointer.keys, J.joint_lanes(), w.world, J.JOINTER.half, t_wait, J.youngest_wait(task),
                          task.temperature)


def test_the_whole_walls_schedule_tools_every_course_inside_its_thumbprint_window() -> None:
    """The whole wall on rc3 (4.4 h on an L4, 4 of 4 worlds): tooled_thumbprint_hard 0.873 < 0.9. The schedule alone
    says why (no simulation): at 0.10 m/s the oldest course's bed got its last pass at 102-114 s, after its mortar
    passed 20 kPa (102 s), and the next course's side strokes worked the oldest side joints again at 228-240 s. The
    program now tools that course at 0.141 m/s (done at 88 s) and keeps side strokes off the course below."""
    old = _ripe(J.ToolJoints(speed_max=0.10, side_stop=None))
    assert old.share == pytest.approx(0.859, abs=0.005) and old.hard == pytest.approx(0.141, abs=0.005), old
    now = _ripe(J.ToolJoints())
    assert now.share > 0.995 and now.never == 0.0 and now.soft == 0.0, now
    task = J.ToolJoints()
    seg = SCH.working(J.program(task, J.wall(task))[0].keys)
    first = [x for x in seg if x[1] < SCH.hard_time(0, task.temperature)]
    assert max(x[1] for x in first[:20]) < SCH.hard_time(0, task.temperature) - 10.0     # the oldest course, 20 strokes
    assert SCH.hard_time(0, task.temperature) == pytest.approx(102.0, abs=1.0)


def test_a_side_stroke_stops_above_the_older_course_and_runs_one_way() -> None:
    lanes = J.joint_lanes()
    x_side = JS.INFILL_X[0] + JS.JOINT_W / 2
    sides = [ln for ln in lanes if ln.p0[0] == ln.p1[0] and abs(ln.p0[0] - x_side) < 1e-9]
    assert [ln.course for ln in sides] == [0, 1, 2, 3]
    out = [SCH.off_older(ln, J.SIDE_STOP) for ln in sides]
    assert out[0] == sides[0] and not out[0].one_way                           # the oldest: nothing older below
    for ln, o in zip(sides[1:], out[1:], strict=True):
        assert o.one_way and o.p0 == ln.p0 and o.p1[1] == pytest.approx(ln.p1[1] + J.SIDE_STOP)
    bed = next(ln for ln in lanes if ln.p0[1] == ln.p1[1])
    assert SCH.off_older(bed, J.SIDE_STOP) == bed and SCH.off_older(sides[1], None) == sides[1]


@pytest.mark.compile_heavy
@pytest.mark.timeout(240)
def test_one_jointer_stroke_and_the_trowel_strips_on_cpu() -> None:
    """One 120 mm stroke along the bottom bed joint (course 0, 18 kPa) at 20 N, then the margin trowel's strips beside
    it: the rod rests on the arrises, so the joint's centre ends 1.4-2.8 mm deep, worked at the bulk's yield; the
    trowel takes the tags the rod laid on the faces and leaves the groove as tooled; the film's books close."""
    task = J.ToolJoints(window=(4.10, 4.22, 0.80, 0.835), passes=1, trowel_speed=0.3)
    res = J.run(task, npz=None)
    assert res.ledger_rel < 1e-12, res
    g = res.verdicts[0]["criteria"]
    assert g["tooled_thumbprint_hard"]["passed"], g                   # tau_w 18 kPa, in the 5-20 kPa window
    assert g["no_void"]["value"] == 0, g                              # the trowel cut nothing below the faces
    assert g["smear"]["judged_part"] and g["smear"]["passed"] is None, g   # it took the tags; nothing left the wall,
    assert res.off_wall_ml == [0.0], res                               # which this port cannot count as their catch
    assert g["concave"]["value"] >= 0.8, g                            # the stroke's middle (its ends set down/lift)
    assert res.trowel_ml[0] > 0.0, res
