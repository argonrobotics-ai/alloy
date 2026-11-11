# SPDX-License-Identifier: Apache-2.0
"""The pass-through meter behind SELF_CONTACT_OVERFLOW (``alloy/precision/self_contact_cross.py``), on synthetic
triangles: no solver, one BVH built here the way Newton's detector builds its own (one group per world, global
triangles last).

Mesh: triangle A lies flat in z = 0; triangle B stands upright through A's inside; triangle C lies flat 2 mm above A
(a separate stacked sheet: no crossing); triangle D shares A's first vertex and passes through A's plane."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor.newton._src.geometry.bvh import compute_bvh_group_roots
from alloy.precision import self_contact
from alloy.precision.self_contact_cross import CrossMeter, _mark_dropped, row_particles, triangles_cross

A = [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)]
B = [(0.002, 0.002, -0.005), (0.002, 0.002, 0.005), (0.004, 0.004, 0.005)]
C = [(0.0, 0.0, 0.002), (0.01, 0.0, 0.002), (0.0, 0.01, 0.002)]


def _ints(xs) -> wp.array:
    return wp.array(np.asarray(xs, dtype=np.int32), dtype=wp.int32, device="cpu")


class _Mesh:
    """Triangles over their own vertices unless ``tri`` says otherwise; ``worlds[i]`` is vertex i's world."""

    def __init__(self, points, worlds, world_count: int, tri=None) -> None:
        self.pos = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device="cpu")
        tri = np.arange(len(points)).reshape(-1, 3) if tri is None else np.asarray(tri)
        self.tri = wp.array(tri.astype(np.int32), dtype=wp.int32, device="cpu")
        corners = self.pos.numpy()[tri]
        groups = np.where(np.asarray(worlds)[tri[:, 0]] < 0, world_count, np.asarray(worlds)[tri[:, 0]])
        self.lower = wp.array(corners.min(axis=1), dtype=wp.vec3, device="cpu")
        self.upper = wp.array(corners.max(axis=1), dtype=wp.vec3, device="cpu")
        self.bvh = wp.Bvh(self.lower, self.upper, groups=_ints(groups))
        self.roots = wp.zeros(world_count + 1, dtype=wp.int32, device="cpu")
        wp.launch(compute_bvh_group_roots, dim=world_count + 1, inputs=[self.bvh.id, self.roots], device="cpu")
        self.meter = CrossMeter(self.tri, np.asarray(worlds, dtype=np.int32), world_count, "cpu")

    def crossed(self, marked=()) -> list[int]:
        marks = np.zeros(self.meter.marked.shape[0], dtype=np.int32)
        marks[list(marked)] = 1
        self.meter.marked.assign(marks)
        self.meter.count(self.bvh.id, self.roots, self.pos)
        return self.meter.crossed.numpy().tolist()


def test_two_crossing_triangles_count_once_a_vertex_of_one_is_marked() -> None:
    mesh = _Mesh(A + B, [0] * 6, 2)
    assert mesh.crossed() == [0, 0]                         # nothing dropped: nothing counted, crossing or not
    assert mesh.crossed(marked=[4]) == [1, 0]               # a vertex of B: found from B
    assert mesh.crossed(marked=[0, 4]) == [2, 0]            # marks on both: the pair counts from each side


def test_the_count_is_the_most_one_detection_saw_not_a_sum() -> None:
    mesh = _Mesh(A + B, [0] * 6, 1)
    for _ in range(3):
        assert mesh.crossed(marked=[0]) == [1]


def test_stacked_sheets_and_neighbours_do_not_count() -> None:
    assert _Mesh(A + C, [0] * 6, 1).crossed(marked=range(6)) == [0]         # 2 mm apart: wp.intersect_tri_tri's case
    d = [(0.0, 0.0, 0.0), (0.004, 0.004, -0.005), (0.004, 0.004, 0.005)]    # through A's plane, from A's vertex 0
    shared = _Mesh(A + d[1:], [0] * 5, 1, tri=[[0, 1, 2], [0, 3, 4]])
    assert shared.crossed(marked=range(5)) == [0]


def test_triangles_of_different_worlds_do_not_count() -> None:
    assert _Mesh(A + B, [0, 0, 0, 1, 1, 1], 2).crossed(marked=range(6)) == [0, 0]
    assert _Mesh(A + B, [1, 1, 1, 1, 1, 1], 2).crossed(marked=[0]) == [0, 1]


@pytest.mark.parametrize("worlds, marked, expect", [
    ([-1, -1, -1, 0, 0, 0], [0], [1, 1]),       # a marked global triangle crossing anything: every world
    ([-1, -1, -1, 0, 0, 0], [3], [1, 0]),       # a world's marked triangle against a global one: that world
    ([-1, -1, -1, 0, 0, 0], [0, 3], [2, 1]),
    ([-1] * 6, [0], [1, 1]),
])
def test_global_triangles_fail_closed(worlds, marked, expect) -> None:
    assert _Mesh(A + B, worlds, 2).crossed(marked=marked) == expect


@pytest.mark.parametrize("a, b, cross", [
    (A, B, True),
    (A, C, False),                                                              # parallel
    (A, [(0.02, 0.0, 0.0), (0.03, 0.0, 0.0), (0.02, 0.01, 0.0)], False),        # coplanar, apart
    (A, [(0.001, 0.001, 0.0), (0.003, 0.001, 0.0), (0.001, 0.003, 0.0)], False),  # coplanar, overlapping
    (A, [(0.002, 0.002, 0.0), (0.002, 0.002, 0.005), (0.004, 0.004, 0.005)], False),  # a corner touches A
    (A, [(0.002, 0.002, 0.001), (0.002, 0.002, 0.005), (0.004, 0.004, 0.005)], False),  # above A
    (A, [(0.02, 0.02, -0.005), (0.02, 0.02, 0.005), (0.04, 0.04, 0.005)], False),  # through A's plane, outside A
])
def test_the_intersection_test_is_strict(a, b, cross) -> None:
    @wp.kernel
    def run(p: wp.array[wp.vec3d], out: wp.array[wp.int32]):
        if triangles_cross(p[0], p[1], p[2], p[3], p[4], p[5]):
            out[0] = 1
        if triangles_cross(p[3], p[4], p[5], p[0], p[1], p[2]):
            out[1] = 1

    out = wp.zeros(2, dtype=wp.int32, device="cpu")
    pts = wp.array(np.asarray(a + b, dtype=np.float32).astype(np.float64), dtype=wp.vec3d, device="cpu")
    wp.launch(run, dim=1, inputs=[pts], outputs=[out], device="cpu")
    assert out.numpy().tolist() == [int(cross)] * 2


def test_a_row_that_dropped_marks_its_particles() -> None:
    """Rows (count, size): (5, 2) drops, (4, 4) does not. A vertex row marks its particle, an edge row (o0, o1, v0,
    v1) both ends, a triangle row its three corners."""
    edges = np.array([[9, 9, 4, 6], [9, 9, 7, 8]])
    tris = np.array([[1, 2, 3], [5, 7, 8]])
    for table, columns, expect in ((None, (), [0]), (edges, self_contact.MARK_COLUMNS["edge_indices"], [4, 6]),
                                   (tris, self_contact.MARK_COLUMNS["tri_indices"], [1, 2, 3])):
        marked = wp.zeros(10, dtype=wp.int32, device="cpu")
        rows = wp.array(row_particles(2, table, columns), dtype=wp.int32, device="cpu")
        wp.launch(_mark_dropped, dim=2, inputs=[_ints([5, 4]), _ints([2, 4]), rows], outputs=[marked], device="cpu")
        assert np.flatnonzero(marked.numpy()).tolist() == expect


def test_a_reset_clears_the_masked_worlds_crossings_and_marks() -> None:
    mesh = _Mesh(A + B + A + B + A + B, [0] * 6 + [1] * 6 + [-1] * 6, 2)
    every = list(range(18))
    # per world: its A and B each cross their own and the global partner (4); the global A and B each cross the
    # three partners there are and blame every world (6)
    assert mesh.crossed(marked=every) == [10, 10]
    mesh.meter.reset(_ints([0, 1]))
    assert mesh.meter.crossed.numpy().tolist() == [10, 0]
    assert np.flatnonzero(mesh.meter.marked.numpy()).tolist() == list(range(6)) + list(range(12, 18))  # global kept
    mesh.meter.reset()
    assert mesh.meter.crossed.numpy().tolist() == [0, 0] and not mesh.meter.marked.numpy().any()


def test_the_watch_reset_clears_dropped_crossed_and_marked() -> None:
    mesh = _Mesh(A + B, [0, 0, 0, 0, 0, 0], 2)
    sc = object.__new__(self_contact.SelfContactWatch)
    sc.device, sc.dropped, sc.cross = "cpu", _ints([3, 5]), mesh.meter
    assert mesh.crossed(marked=[0]) == [1, 0]
    sc.reset(_ints([1, 0]))
    assert sc.dropped.numpy().tolist() == [0, 5] and mesh.meter.crossed.numpy().tolist() == [0, 0]
    assert not mesh.meter.marked.numpy().any()


@pytest.mark.parametrize("env, policy, expect", [("", "off", "off"), ("", "invalid", "invalid"),
                                                 ("invalid", "off", "invalid"), ("report", "invalid", "report"),
                                                 ("off", "report", "off"), ("nonsense", "report", "report")])
def test_the_switch_is_the_policy_unless_the_environment_says(monkeypatch, env, policy, expect) -> None:
    from alloy.precision.types import CheckPolicy

    monkeypatch.setenv(self_contact.CROSS_ENV, env)
    assert CheckPolicy().self_contact_cross == "off"                       # the default: no meter
    assert self_contact.cross_mode(CheckPolicy(self_contact_cross=policy)) == expect


def test_a_world_an_engine_rollback_put_back_loses_its_marks_and_is_not_judged_once() -> None:
    """Rollbacks restore ``crossed`` and ``seen`` per world from health's snapshot; the meter's own count did not
    go back, so at the next detection that world's marks are cleared and its side changes skipped (its particles
    jumped back to the period start): a miss at worst, never a crossing the kept run did not make."""
    mesh = _Mesh(A + _sunk(FLAT) + A + _sunk(FLAT), [0] * 6 + [1] * 6, 2)
    meter = mesh.meter
    meter._prev.assign(np.asarray(A + FLAT + A + FLAT, dtype=np.float32))     # both worlds' B went through A
    meter.marked.assign(np.ones(12, dtype=np.int32))
    meter.seen.assign(np.array([5, 5], np.int32))
    meter._mine.assign(np.array([5, 5], np.int32))
    kept = meter.seen.numpy().copy()
    meter.begin()
    meter.count(mesh.bvh.id, mesh.roots, mesh.pos, pad=0.004)
    assert meter.crossed.numpy().tolist() == [3, 3]                         # no rollback: both counted
    meter._prev.assign(np.asarray(A + FLAT + A + FLAT, dtype=np.float32))
    meter.crossed.zero_()
    meter.seen.assign(np.array([6, int(kept[1])], np.int32))                # world 1 put back to the period start
    meter.begin()
    assert meter.marked.numpy()[6:].sum() == 0 and meter.marked.numpy()[:6].all()
    meter.count(mesh.bvh.id, mesh.roots, mesh.pos, pad=0.004)
    assert meter.crossed.numpy().tolist() == [3, 0]
    assert meter.seen.numpy().tolist() == meter._mine.numpy().tolist()      # in step again


def test_the_meter_hands_health_its_per_world_arrays_only_when_on() -> None:
    sc = object.__new__(self_contact.SelfContactWatch)
    sc.cross = None
    assert sc.state_arrays() == {} and sc.marked is None and sc.crossed is None
    sc.cross = _Mesh(A + B, [0] * 6, 2).meter
    assert sorted(sc.state_arrays()) == ["crossed", "seen"]
    assert all(a.shape[0] == 2 for a in sc.state_arrays().values())        # world-major: restored by world mask


FLAT = [(0.002, 0.002, 0.001), (0.004, 0.002, 0.001), (0.002, 0.004, 0.001)]      # a small triangle 1 mm over A's inside


def _sunk(points, dz: float = -0.002, dx: float = 0.0):
    return [(x + dx, y, z + dz) for x, y, z in points]


def _side_changes(before, after, marked, pad: float = 0.004) -> list[int]:
    """A (fixed) and a second triangle that moved from ``before`` to ``after`` between two detections."""
    mesh = _Mesh(A + after, [0] * 6, 1)
    mesh.meter._prev.assign(np.asarray(A + before, dtype=np.float32))
    marks = np.zeros(6, dtype=np.int32)
    marks[list(marked)] = 1
    mesh.meter.marked.assign(marks)
    mesh.meter.count(mesh.bvh.id, mesh.roots, mesh.pos, pad=pad)
    return mesh.meter.crossed.numpy().tolist()


def test_a_parallel_layer_that_swapped_sides_counts_though_no_triangles_intersect() -> None:
    assert _Mesh(A + FLAT, [0] * 6, 1).crossed(marked=range(6)) == [0]              # before: apart
    assert _Mesh(A + _sunk(FLAT), [0] * 6, 1).crossed(marked=range(6)) == [0]       # after: apart again, below
    assert _side_changes(FLAT, _sunk(FLAT), marked=[3, 4, 5]) == [3]                # its three vertices went through A
    assert _side_changes(FLAT, _sunk(FLAT), marked=[3]) == [1]
    assert _side_changes(FLAT, _sunk(FLAT), marked=[0, 1, 2]) == [0]                # A's own vertices changed no side


def test_side_changes_need_a_mark_the_inside_and_a_solver_sized_move() -> None:
    assert _side_changes(FLAT, _sunk(FLAT), marked=[]) == [0]                       # nothing dropped
    assert _side_changes(FLAT, _sunk(FLAT, dz=-0.0005), marked=[3, 4, 5]) == [0]    # came closer, same side
    beside = _sunk(FLAT, dz=0.0, dx=0.02)                                           # past A's edge: around, not through
    assert _side_changes(beside, _sunk(beside), marked=[3, 4, 5]) == [0]
    assert _side_changes(FLAT, _sunk(FLAT, dz=-0.05), marked=[3, 4, 5]) == [0]      # 5 cm in one go: a teleport
    assert _side_changes(FLAT, _sunk(FLAT), marked=[3, 4, 5], pad=0.0) == [0]       # pad 0: side changes left out


def test_a_vertex_straight_through_an_edge_of_the_other_sheet_counts() -> None:
    on_edge = [(0.005, 0.0, 0.001), (0.006, -0.002, 0.001), (0.004, -0.002, 0.001)]  # vertex 3 exactly over A's edge
    assert _side_changes(on_edge, _sunk(on_edge), marked=[3]) == [1]


def test_a_rerun_period_forgets_what_its_abandoned_attempt_saw() -> None:
    """Rows sized to the scene re-run a period that dropped pairs (``row_growth``): the attempt's marks, crossings and
    positions go back to the period start with health's dropped count, through the guard's own keep / restore."""
    from types import SimpleNamespace

    from alloy.integration.row_growth import RowGuard

    mesh = _Mesh(A + B, [0] * 6, 1)
    sc = object.__new__(self_contact.SelfContactWatch)
    sc.device, sc.dropped, sc.cross = "cpu", _ints([0]), mesh.meter
    guard = RowGuard({"cloth": SimpleNamespace(solver=SimpleNamespace(_alloy_self_contact_watch=sc))})
    prev = mesh.meter._prev.numpy().copy()
    guard.keep_dropped()                                             # the period starts
    sc.dropped.assign(np.array([4], np.int32))
    assert mesh.crossed(marked=[0, 4]) == [2]                        # the attempt drops pairs and crosses
    guard.restore_dropped()                                          # the rows grew: the period runs again
    assert sc.dropped.numpy().tolist() == [0] and mesh.meter.crossed.numpy().tolist() == [0]
    assert not mesh.meter.marked.numpy().any() and np.array_equal(mesh.meter._prev.numpy(), prev)
