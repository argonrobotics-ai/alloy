# SPDX-License-Identifier: Apache-2.0
"""Rigid contacts as data and drawn on a Newton viewer (alloy.viewer._contacts, alloy.viewer._contact_draw).

Laws, each on both MuJoCo routes on the CPU (MuJoCo C, Alloy's CPU default, marked ``cpu_route``; MuJoCo Warp, which
the suite's CPU tests use for the GPU kernels): a resting box's contact forces carry its weight with no net friction,
at the table top inside its footprint, normals along z, named (table, block); a sideways push below the friction
limit is met by an equal and opposite friction force on the box; a falling box has no contacts (empty arrays of the
right shapes); worlds are read apart; reading after every step leaves the run bit for bit the same; drawing logs
offset points and force arrows and clears them when there are none.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import alloy
from alloy._vendor.newton import viewer as nv
from alloy.viewer._contact_draw import FORCES, POINTS
from alloy.viewer._contacts import ContactSet, _pyramid, contacts, draw, read

pytestmark = pytest.mark.compile_heavy      # first build of a route compiles its kernels

DT = 0.01
MASS, EDGE, TOP = 0.2, 0.05, 0.4            # kg, m (full edge), table top height (m)
WEIGHT = MASS * 9.81                        # 1.962 N
REST_Z = TOP + EDGE / 2
PUSH = 0.5                                  # N, below the friction limit (mu 0.575 * 1.962 N = 1.13 N)
ROUTES = [pytest.param("mjc", marks=pytest.mark.cpu_route, id="mujoco_c"), pytest.param("warp", id="mujoco_warp")]


def _scene(starts: np.ndarray) -> alloy.Scene:
    scene = alloy.Scene(worlds=len(starts), dt=DT)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, TOP), pos=(0, 0, TOP / 2), static=True)
    scene.add_box("block", size=(EDGE, EDGE, EDGE), pos=starts, mass=MASS)
    return scene


def _build(starts: np.ndarray, route: str) -> alloy.Sim:
    """Build on ``route`` ("mjc" or "warp"; the suite's conftest forces "warp", the factory reads it at build)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ALLOY_CPU_ROUTE", route)
        sim = alloy.build(_scene(starts), device="cpu", seed=0)
    names = [type(s).__name__ for s in sim.engine.solvers]
    assert ("MjcCoreSolver" in names) == (route == "mjc"), names
    return sim


def _starts(heights: list[float]) -> np.ndarray:
    return np.array([[0.1 * w, 0.0, z] for w, z in enumerate(heights)])


def _on(cs: ContactSet, name: str) -> np.ndarray:
    """Net contact force on ``name``: as body_b, plus the reaction where it is body_a."""
    b = np.array([x == name for x in cs.body_b], bool)
    a = np.array([x == name for x in cs.body_a], bool)
    return cs.force[b].sum(axis=0) - cs.force[a].sum(axis=0)


@pytest.fixture(scope="module", params=ROUTES)
def rest(request: pytest.FixtureRequest) -> tuple[alloy.Sim, dict[int, ContactSet]]:
    """Two worlds, a box resting on the table in each, after 0.5 s; the read is taken before any test pushes."""
    sim = _build(_starts([REST_Z, REST_Z]), request.param)
    sim.step(50)
    return sim, read(sim, [0, 1])


# ---- laws -----------------------------------------------------------------------------------------------------------
def test_resting_box_contacts_carry_its_weight(rest: tuple[alloy.Sim, dict[int, ContactSet]]) -> None:
    sim, sets = rest
    for w, cs in sets.items():
        assert len(cs) >= 3, (w, len(cs))
        f = _on(cs, "block")
        assert abs(f[2] / WEIGHT - 1.0) < 1e-3, (w, f)                       # measured < 1e-6 on both routes
        assert np.hypot(f[0], f[1]) < 1e-3 * WEIGHT, (w, f)
        assert {frozenset(p) for p in zip(cs.body_a, cs.body_b)} == {frozenset(("table", "block"))}
        assert np.allclose(np.abs(cs.normal[:, 2]), 1.0, atol=1e-6, rtol=0)
        assert np.all(np.abs(cs.pos[:, 2] - TOP) < 1e-3), cs.pos
        centre = sim.state(world=w)["block.pos"]
        assert np.all(np.abs(cs.pos[:, :2] - centre[:2]) <= EDGE / 2 + 1e-3), (cs.pos, centre)
        assert np.all(np.abs(cs.depth) < 1e-3), cs.depth


def test_push_below_friction_limit_is_met_by_friction(rest: tuple[alloy.Sim, dict[int, ContactSet]]) -> None:
    sim, _ = rest
    sim.apply_wrench("block", force=(PUSH, 0.0, 0.0))
    sim.step(20)
    for w, cs in read(sim, [0, 1]).items():
        f = _on(cs, "block")
        assert abs(f[0] + PUSH) < 0.01 * PUSH, (w, f)                         # measured < 1e-6 relative
        assert abs(f[1]) < 0.01 * PUSH and abs(f[2] / WEIGHT - 1.0) < 1e-3, (w, f)
    sim.apply_wrench("block", force=(0.0, 0.0, 0.0))


@pytest.fixture(scope="module", params=ROUTES)
def apart(request: pytest.FixtureRequest) -> alloy.Sim:
    """World 0's box rests on the table; world 1's starts 1 m higher (falling for the first 0.45 s)."""
    return _build(_starts([REST_Z, REST_Z + 1.0]), request.param)


def test_free_fall_has_no_contacts_and_worlds_are_read_apart(apart: alloy.Sim) -> None:
    apart.step(3)
    sets = read(apart, [0, 1])
    assert len(sets[0]) >= 3 and abs(_on(sets[0], "block")[2] / WEIGHT - 1.0) < 0.05
    falling = sets[1]
    assert len(falling) == 0
    assert falling.pos.shape == falling.normal.shape == falling.force.shape == (0, 3) and falling.depth.shape == (0,)
    one = contacts(apart, world=1)
    assert set(one) == {"pos", "normal", "force", "depth", "body_a", "body_b"}
    assert one["pos"].shape == (0, 3) and one["body_a"] == [] and one["body_b"] == []
    assert len(contacts(apart, world=0)["body_b"]) == len(sets[0])


def test_world_is_checked(apart: alloy.Sim) -> None:
    for bad in (2, -1, "0", 0.0, True):
        with pytest.raises(alloy.SimulationError):
            contacts(apart, world=bad)
    with pytest.raises(alloy.SimulationError):
        read(apart, [0, 5])


@pytest.mark.parametrize("route", ROUTES)
def test_reading_does_not_change_the_run(route: str) -> None:
    starts = _starts([REST_Z, REST_Z + 0.05])                  # one box resting, one landing during the run
    plain, watched = _build(starts, route), _build(starts, route)
    seen = 0
    for _ in range(30):
        plain.step(1)
        watched.step(1)
        seen += sum(len(s) for s in read(watched, [0, 1]).values())
    assert seen > 0
    a, b = plain.state(), watched.state()
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k


# ---- drawing --------------------------------------------------------------------------------------------------------
class _Recorder(nv.ViewerNull):
    """ViewerNull that keeps the last points and arrows it was given (None = cleared)."""

    def __init__(self) -> None:
        super().__init__(num_frames=10)
        self.logged: dict[str, Any] = {}

    def log_points(self, name, points, radii=None, colors=None, hidden=False):  # noqa: ANN001, ANN201
        self.logged[name] = None if points is None else points.numpy().copy()

    def log_arrows(self, name, starts, ends, colors, width=0.01, hidden=False):  # noqa: ANN001, ANN201
        self.logged[name] = None if starts is None else (starts.numpy().copy(), ends.numpy().copy())


def test_draw_passes_colour_arrays(rest: tuple[alloy.Sim, dict[int, ContactSet]]) -> None:
    """Colours go to the viewer as one vec3 per item: the desktop viewer does not take a plain tuple."""
    sim, sets = rest
    seen: dict[str, object] = {}

    class _Colours(nv.ViewerNull):
        def log_points(self, name, points, radii=None, colors=None, hidden=False):
            seen["points"] = (points, colors)

        def log_arrows(self, name, starts, ends, colors, width=0.01, hidden=False):
            seen["arrows"] = (starts, colors)

    v = _Colours(num_frames=10)
    v.set_model(sim.engine.model)
    v.begin_frame(sim.t)
    draw(v, sets)
    v.end_frame()
    for items, colors in seen.values():
        assert hasattr(colors, "numpy") and colors.numpy().shape == (len(items), 3)


def test_draw_on_null_viewer(rest: tuple[alloy.Sim, dict[int, ContactSet]]) -> None:
    sim, sets = rest
    v = nv.ViewerNull(num_frames=10)
    v.set_model(sim.engine.model)
    for frame in (sets, {0: ContactSet.empty(), 1: ContactSet.empty()}, {}):
        v.begin_frame(sim.t)
        draw(v, frame)
        draw(v, frame, points=False, forces=False)
        v.end_frame()


def test_draw_offsets_points_and_clears(rest: tuple[alloy.Sim, dict[int, ContactSet]]) -> None:
    sim, sets = rest
    v = _Recorder()
    v.set_model(sim.engine.model)
    v.set_world_offsets((2.0, 0.0, 0.0))
    offsets = v.world_offsets.numpy()
    draw(v, sets)
    pts = v.logged[POINTS]
    want = np.concatenate([sets[w].pos + offsets[w] for w in (0, 1)])
    assert np.allclose(pts, want, atol=1e-5)
    starts, ends = v.logged[FORCES]
    up = ends - starts
    assert np.allclose(starts, want, atol=1e-5) and np.all(up[:, 2] > 0) and np.allclose(up[:, :2], 0, atol=1e-6)
    v.set_visible_worlds([1])
    draw(v, sets)
    assert len(v.logged[POINTS]) == len(sets[1])
    draw(v, {0: ContactSet.empty()})
    assert v.logged[POINTS] is None and v.logged[FORCES] is None
    draw(v, sets, forces=False)
    assert v.logged[POINTS] is not None and v.logged[FORCES] is None


def test_draw_skips_a_backend_without_arrows() -> None:
    class PointsOnly:
        device = "cpu"
        model = None
        world_offsets = None

        def __init__(self) -> None:
            self.calls = 0

        def log_points(self, *args: Any) -> None:
            self.calls += 1

    b = PointsOnly()
    cs = ContactSet(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), np.array([[0, 0, 2.0]]), np.zeros(1), ["a"], ["b"])
    draw(b, {0: cs})
    assert b.calls == 1


def test_pyramid_rows_decode_as_mujoco() -> None:
    """Pyramidal cone: normal = sum of the 2(dim-1) edge forces, tangent k = (f+ - f-) * mu_k (MuJoCo's decode)."""
    row = np.array([1.0, 0.5, 0.25, 0.75, 9.0])
    f = _pyramid(row, 0, np.array([0.8, 0.6, 0, 0, 0]), 3, njmax=row.size)
    assert np.allclose(f, [2.5, 0.4, -0.3])
    assert np.allclose(_pyramid(row, 4, np.zeros(5), 1, njmax=row.size), [9.0, 0, 0])


def test_force_on_sums_with_the_right_sign_per_pair() -> None:
    """A block and a ball on a table: each reads its own weight, the table minus both, whichever body MuJoCo lists
    first in a pair."""
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.43), mass=0.2)
    scene.add_sphere("ball", 0.03, pos=(0.15, 0, 0.44), mass=0.1)
    sim = alloy.build(scene, device="cpu", seed=0)
    sim.step(100)
    g = 9.81
    assert alloy.viewer.force_on(sim, "block") == pytest.approx([0, 0, 0.2 * g], abs=0.02 * 0.2 * g)
    assert alloy.viewer.force_on(sim, "ball") == pytest.approx([0, 0, 0.1 * g], abs=0.02 * 0.1 * g)
    assert alloy.viewer.force_on(sim, "table") == pytest.approx([0, 0, -0.3 * g], abs=0.02 * 0.3 * g)
    with pytest.raises(alloy.UnknownNameError):
        alloy.viewer.force_on(sim, "blok")
