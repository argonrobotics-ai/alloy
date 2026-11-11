# SPDX-License-Identifier: Apache-2.0
"""GAP-13A law: a self-contact overflow is INVALID only when the surfaces really passed through each other
(``docs/known_gaps/vbd-self-contact-buffer-overflow.md``).

Law (zero tolerance, geometric): a world has passed through when, after a row dropped pairs, two triangles of the
VBD entry that share no vertex intersect, and one of them has a vertex whose vertex row, or an edge row it ends,
dropped a pair; or such a vertex went through a triangle's inside to its other side between two detections. ``SelfContactWatch.crossed`` (int32 [worlds], sticky) counts such pairs, ``marked`` (int32
[particles]) the vertices that dropped.

Two scenes of two worlds; world 1's vertex and edge rows are shrunk after build, world 0 keeps the default rows as
the control, so whatever only world 1 does, its dropped pairs caused.
- calm: two sheets stacked on the ground, rows of one pair. World 1 drops up to 14 pairs a row, nothing crosses and
  the sheets rest 2.0 mm apart: DEGRADED, as before, also with the INVALID switch on.
- tilted: a sheet over a narrow stand, a second one dropped on it tilted 0.5 rad, rows of no pair. World 1's
  sheets cross from step 9 (72 pairs, 200 by step 20); world 0's never do.
- heavy: the calm stack with a 1 kg upper sheet, rows of no pair. World 1's upper sheet sinks through the lower one
  flat from step 4 (every vertex below its twin by step 6) without a triangle pair ever intersecting at a detection:
  only the law's second clause sees it (a marked vertex changed sides of a triangle through its inside since the
  detection before). Before that clause the meter read 0 here (CPU, 2026-11-10).

The reference is a host oracle (numpy, below): every edge of one triangle against the other triangle, both ways.
The detector runs inside the step, so the meter may see a crossing the step's end state no longer shows; the tests
ask only that a crossing the oracle saw with a host-marked vertex is on the meter one step later, and that the
control world never crosses. The switch is ``CheckPolicy.self_contact_cross`` ("off" by default: no meter at all,
"report", "invalid"); ``alloy.build`` takes no policy, so the tests set its override ``ALLOY_SELF_CONTACT_CROSS``
(read at build)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.integration.stepper import entry_solvers
from alloy.precision import self_contact

CROSS_ENV = "ALLOY_SELF_CONTACT_CROSS"
EDGES = ((0, 1), (1, 2), (2, 0))
SIDE = 0.2
SceneFn = Callable[[alloy.Scene], None]


def _segments_hit(p: np.ndarray, q: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Row by row: does segment p-q pass through the inside of triangle tri (Moller-Trumbore)?"""
    d, e1, e2 = q - p, tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-14                                    # a parallel segment does not cross
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p - tri[:, 0]
    c = np.cross(s, e1)
    u, v, t = f * np.einsum("ij,ij->i", s, h), f * np.einsum("ij,ij->i", d, c), f * np.einsum("ij,ij->i", e2, c)
    return ok & (u > 0) & (v > 0) & (u + v < 1) & (t > 0) & (t < 1)


def _crossing_pairs(x: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """The oracle: (n, 2) rows of ``tris`` that share no vertex and intersect at positions ``x``."""
    i, j = np.triu_indices(len(tris), 1)
    apart = ~(tris[i][:, :, None] == tris[j][:, None, :]).any(axis=(1, 2))
    i, j = i[apart], j[apart]
    hit = np.zeros(len(i), dtype=bool)
    for a, b in ((i, j), (j, i)):
        for k0, k1 in EDGES:
            hit |= _segments_hit(x[tris[a, k0]], x[tris[a, k1]], x[tris[b]])
    return np.stack([i[hit], j[hit]], axis=1)


def _calm(scene: alloy.Scene) -> None:
    scene.add_cloth("a", size=(SIDE, SIDE), pos=(0, 0, 0.003), resolution=10, mass=0.02)
    scene.add_cloth("b", size=(SIDE, SIDE), pos=(0, 0, 0.0065), resolution=10, mass=0.02)


def _tilted(scene: alloy.Scene) -> None:
    scene.add_box("stand", size=(0.08, 0.08, 0.1), pos=(0, 0, 0.05), static=True)
    scene.add_cloth("a", size=(SIDE, SIDE), pos=(0, 0, 0.103), resolution=10, mass=0.02)
    scene.add_cloth("b", size=(SIDE, SIDE), pos=(0, 0, 0.163), resolution=10, mass=0.02, rpy=(0.5, 0, 0))


def _heavy(scene: alloy.Scene) -> None:
    """The calm stack with a 1 kg upper sheet: with nothing held it sinks through the lower one flat, layer through
    layer, and no triangle ever intersects another."""
    scene.add_cloth("a", size=(SIDE, SIDE), pos=(0, 0, 0.003), resolution=10, mass=0.02)
    scene.add_cloth("b", size=(SIDE, SIDE), pos=(0, 0, 0.0065), resolution=10, mass=1.0)


@dataclass
class Run:
    """One scene stepped with the oracle beside it; the lists are per world, read after the last step."""
    sim: Any
    watch: Any
    particle_world: np.ndarray
    oracle: list[int]               # the most crossing pairs the oracle saw at any step's end
    dropped: list[int]
    crossed: list[int]
    host_marked: np.ndarray         # bool [particles]: rows over their size at a step's end (a subset of marked)
    gap_m: float                    # world 1, last step: the nearest vertices of the two sheets
    health: dict[str, Any]
    check: dict[str, Any]


def _vbd(sim: Any) -> Any:
    [v] = [s for s in entry_solvers(sim.engine.stepper).values() if isinstance(s, newton.solvers.SolverVBD)]
    return v


def _row_sizes(v: Any) -> list[tuple[str, str, np.ndarray]]:
    """(count, size, the particles each row marks [rows, k]) of the vertex and the edge rows."""
    ends = v.model.edge_indices.numpy()[:, 2:4]
    return [("vertex_colliding_triangles_count", "vertex_colliding_triangles_buffer_sizes",
             np.arange(int(v.model.particle_count))[:, None]),
            ("edge_colliding_edges_count", "edge_colliding_edges_buffer_sizes", ends)]


def _shrink_world_1(v: Any, pairs: int) -> None:
    pw = v.model.particle_world.numpy()
    for _, size, particles in _row_sizes(v):
        sizes = getattr(v.trimesh_collision_info, size)
        sizes.assign(np.where(pw[particles[:, 0]] == 1, np.minimum(sizes.numpy(), pairs), sizes.numpy())
                     .astype(np.int32))


def _mark_host(v: Any, marked: np.ndarray) -> None:
    info = v.trimesh_collision_info
    for count, size, particles in _row_sizes(v):
        over = getattr(info, count).numpy() > getattr(info, size).numpy()
        marked[particles[over].ravel()] = True


def _attributed(x: np.ndarray, tris: np.ndarray, marked: np.ndarray) -> int:
    """Oracle pairs with a marked vertex on either triangle."""
    pairs = _crossing_pairs(x, tris)
    return int(marked[tris[pairs]].any(axis=(1, 2)).sum())


def _run(add: SceneFn, pairs: int, steps: int) -> Run:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    add(scene)
    sim = alloy.build(scene, device="cpu")
    v = _vbd(sim)
    watch = self_contact.watch(v)
    _shrink_world_1(v, pairs)
    pw, tris = v.model.particle_world.numpy(), v.model.tri_indices.numpy()
    marked, oracle = np.zeros(len(pw), dtype=bool), [0, 0]
    for _ in range(steps):
        seen = list(oracle)
        sim.step(1)
        crossed = watch.crossed.numpy().tolist()
        assert crossed[0] == 0                                # the control never crosses, at any detection
        assert all(c > 0 for c, o in zip(crossed, seen) if o > 0)     # what the oracle saw a step ago is metered
        _mark_host(v, marked)
        x = sim.engine.state.particle_q.numpy()
        now = [_attributed(x, tris[pw[tris[:, 0]] == w], marked) for w in (0, 1)]
        oracle = [max(a, b) for a, b in zip(oracle, now)]
    a, b = (sim.state(world=1)[f"{n}.points"] for n in ("a", "b"))
    gap = float(np.linalg.norm(a[:, None] - b[None], axis=2).min())
    return Run(sim, watch, pw, oracle, watch.dropped.numpy().tolist(), watch.crossed.numpy().tolist(), marked, gap,
               sim.health(), sim.check())


def _switched(value: str, add: SceneFn, pairs: int, steps: int) -> Iterator[Run]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(CROSS_ENV, value)
        mp.setenv("ALLOY_SELF_CONTACT_ROWS", "fixed")       # rows sized to the scene would grow the shrunk rows back
        yield _run(add, pairs, steps)


@pytest.fixture(scope="module")
def calm_on() -> Iterator[Run]:
    yield from _switched("invalid", _calm, pairs=1, steps=5)


@pytest.fixture(scope="module")
def tilted_on() -> Iterator[Run]:
    yield from _switched("invalid", _tilted, pairs=0, steps=12)


@pytest.fixture(scope="module")
def tilted_report() -> Iterator[Run]:
    yield from _switched("report", _tilted, pairs=0, steps=12)


@pytest.fixture(scope="module")
def heavy_report() -> Iterator[Run]:
    yield from _switched("report", _heavy, pairs=0, steps=6)


def _b_under_a(sim: Any, world: int) -> float:
    """The share of the upper sheet's vertices that are below the lower sheet's vertex of the same index."""
    st = sim.state(world=world)
    return float((st["b.points"][:, 2] < st["a.points"][:, 2]).mean())


def test_the_oracle_counts_a_pierced_triangle_and_not_a_neighbour_or_a_near_miss() -> None:
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],                      # two triangles sharing an edge
                  [0.2, 0.2, -0.1], [0.2, 0.2, 0.1], [0.9, 0.9, 0.5],              # pierces the first
                  [0.6, 0.1, 0.01], [0.7, 0.1, 0.01], [0.6, 0.2, 0.01]], float)    # 10 mm over it: no crossing
    tris = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [7, 8, 9]])
    assert _crossing_pairs(x, tris).tolist() == [[0, 2]]


def test_a_calm_overflow_drops_pairs_and_stays_degraded_with_the_switch_on(calm_on: Run) -> None:
    r, worlds = calm_on, calm_on.health["worlds"]
    assert r.dropped[0] == 0 and r.dropped[1] > 0                                   # the shrink really drops pairs
    assert r.oracle == [0, 0] and r.crossed == [0, 0]                               # and nothing passed through
    assert r.gap_m > 1.5e-3                                                         # the sheets rest 2.0 mm apart
    marked = r.watch.marked.numpy()
    assert not marked[r.particle_world == 0].any() and marked[r.particle_world == 1].any()
    assert worlds[0]["severity"] == "OK" and "SELF_CONTACT_OVERFLOW" not in worlds[0]["flags"]
    assert worlds[1]["severity"] == "DEGRADED" and worlds[1]["flags"] == ["SELF_CONTACT_OVERFLOW"]
    assert r.health["degraded_worlds"] == [1] and r.health["invalid_worlds"] == []
    assert r.check["ok"] and r.check["problems"] == []


def test_a_pass_through_is_invalid_and_named_in_check_with_the_switch_on(tilted_on: Run) -> None:
    r, worlds = tilted_on, tilted_on.health["worlds"]
    assert r.dropped[0] == 0 and r.dropped[1] > 0
    assert r.crossed[0] == 0 and r.crossed[1] > 0
    assert worlds[0]["severity"] == "OK" and "SELF_CONTACT_OVERFLOW" not in worlds[0]["flags"]
    assert worlds[1]["severity"] == "INVALID" and "SELF_CONTACT_OVERFLOW" in worlds[1]["flags"]
    assert r.health["invalid_worlds"] == [1]
    assert not r.check["ok"]
    named = [p for p in r.check["problems"] if p["world"] == 1 and "passed through" in p["what"]]
    assert len(named) == 1 and named[0]["fix"]
    assert not any(p["world"] == 0 for p in r.check["problems"])


def test_a_pass_through_is_measured_and_stays_degraded_when_only_reported(tilted_report: Run) -> None:
    r, worlds = tilted_report, tilted_report.health["worlds"]
    assert r.dropped[0] == 0 and r.dropped[1] > 0
    assert r.oracle[0] == 0 and r.oracle[1] > 0                                     # the sheets really crossed
    assert r.crossed[0] == 0 and r.crossed[1] > 0
    marked = r.watch.marked.numpy()
    assert set(np.unique(marked).tolist()) <= {0, 1}
    assert not marked[r.particle_world == 0].any()
    assert marked[r.host_marked].all()                                              # every host mark is on the meter
    assert worlds[0]["severity"] == "OK"
    assert worlds[1]["severity"] == "DEGRADED" and "SELF_CONTACT_OVERFLOW" in worlds[1]["flags"]
    assert r.health["invalid_worlds"] == [] and r.check["ok"]


def test_a_reset_clears_crossed_and_marked(tilted_report: Run) -> None:
    r = tilted_report
    r.sim.step(1)                                                                   # whatever ran before: crossed
    assert r.watch.crossed.numpy()[1] > 0 and r.watch.marked.numpy().any()
    r.sim.reset()
    assert r.watch.crossed.numpy().tolist() == [0, 0]
    assert not r.watch.marked.numpy().any()
    assert r.watch.dropped.numpy().tolist() == [0, 0]


@pytest.mark.timeout(180)                        # two 6-step CPU runs' worth with the oracle; ~60 s on a busy host
def test_parallel_layers_that_swap_sides_are_a_pass_through(heavy_report: Run) -> None:
    """No triangle pair need ever intersect: the upper sheet ends below the lower one, and the meter counts it (the
    side-change clause). The switch-on path to INVALID is the same source as the tilted scene's."""
    r = heavy_report
    assert _b_under_a(r.sim, 0) == 0.0 and _b_under_a(r.sim, 1) > 0.5               # only world 1's sheets swapped
    assert r.dropped[0] == 0 and r.dropped[1] > 0
    assert r.crossed[0] == 0 and r.crossed[1] > 0
    assert r.health["worlds"][0]["severity"] == "OK"
    assert r.health["worlds"][1]["severity"] == "DEGRADED" and r.check["ok"]
    assert r.health["worlds"][1]["self_contact"]["crossed_pairs"] > 0


def _calm_stack(worlds: int = 2) -> Any:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    _calm(scene)
    return alloy.build(scene, device="cpu")


@pytest.mark.timeout(180)
def test_the_meter_is_off_by_default_and_on_changes_no_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no switch): no meter, no health row key, as before. On ("invalid"), it only reads: with nothing
    dropped the particles step bit for bit as with it off (CPU)."""
    monkeypatch.setenv("ALLOY_SELF_CONTACT_ROWS", "fixed")
    monkeypatch.delenv(CROSS_ENV, raising=False)
    off = _calm_stack()
    assert self_contact.watch(_vbd(off)).cross is None
    monkeypatch.setenv(CROSS_ENV, "invalid")
    on = _calm_stack()
    assert self_contact.watch(_vbd(on)).cross is not None
    for _ in range(3):
        off.step(1)
        on.step(1)
    assert np.array_equal(off.engine.state.particle_q.numpy(), on.engine.state.particle_q.numpy())
    assert all("self_contact" not in w for w in off.health()["worlds"])
    assert off.health()["worlds"] == on.health()["worlds"]


@pytest.mark.timeout(180)
def test_a_rollback_takes_the_crossing_back_for_the_restored_world_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both worlds' rows hold no pair and both cross; the engine's per-world snapshot of state + health (as a
    coupled re-run uses it) puts world 1 back to before the crossing: its count goes back to 0 and the replay from
    there counts nothing for it at the first detection; world 0 keeps its count."""
    from alloy.core.snapshot import WorldSnapshot, mask_from_worlds

    monkeypatch.setenv("ALLOY_SELF_CONTACT_ROWS", "fixed")
    monkeypatch.setenv(CROSS_ENV, "report")
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    _tilted(scene)
    sim = alloy.build(scene, device="cpu")
    v, eng = _vbd(sim), sim.engine
    watch = self_contact.watch(v)
    for size in ("vertex_colliding_triangles_buffer_sizes", "edge_colliding_edges_buffer_sizes"):
        getattr(v.trimesh_collision_info, size).zero_()                        # both worlds: rows of no pair
    snap = WorldSnapshot(eng.model, eng.state, {f"health/{k}": a for k, a in eng.health.state_arrays().items()})
    snap.save(eng.state)                                                       # before anything crossed
    for _ in range(12):
        sim.step(1)
    crossed = watch.crossed.numpy().tolist()
    assert crossed[0] > 0 and crossed[1] > 0
    snap.restore(eng.state, mask_from_worlds([1], 2, eng.model.device))
    assert watch.crossed.numpy().tolist() == [crossed[0], 0]
    sim.step(1)                                                                # world 1 replays from the start
    assert watch.crossed.numpy().tolist()[1] == 0 and watch.crossed.numpy().tolist()[0] >= crossed[0]
