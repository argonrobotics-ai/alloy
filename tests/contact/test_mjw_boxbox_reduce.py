# SPDX-License-Identifier: Apache-2.0
"""SPEED-S1, WP3's contact part of the opt-in fast rigid mode: box-box face pairs at 4 contacts
(``alloy.contact.mjw_boxbox_reduce``), routed only under ``entries.RIGID_FAST_EFFECT``.

- the pick over the overlap of two equal squares at 2-45 degrees of yaw: the largest-hull 4 points (brute force, as
  numpy finds them). The hull keeps at least ``LIMIT`` of the overlap's area and support: four of a regular octagon's
  vertices cannot keep more at 45 degrees. Support along a direction sets the tipping onset over that edge;
- a tilted pair keeps its deepest point;
- a yawed box resting on a box (CPU, Warp route): 4 contacts instead of 8, the same rest height and level to 1e-5;
- the route sets the key under the fast effect only, and the default entry is unchanged;
- law: a 2 kg, 0.4 m cube yawed 20 degrees on a box, pushed at its top's height by a force rising 1 % of m g
  every 0.1 s, tips (> 2 degrees) within ``TIP_TOL`` m g of where it tips on 8 contacts. Rows (CPU, 5 ms, 16 push
  directions): at 45 degrees on and off tip alike at 0.53-0.58 m g. At 20 degrees on tips at 0.52-0.58 and off at
  0.53-0.59 (analytic 0.50-0.54). The geometric support loss does not show: once the cube tilts, its far contacts lift
  and the pick keeps the pivot side.
On the L4 box stack (WP7's box_fast.py, 3,072 worlds, 10 ms, back-to-back on one VM): 1,239 / 1,261x -> 1,741 /
1,754x, every stack standing, settle median 1.36 -> 1.56 mm."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.contact import mjw_boxbox_reduce as R
from alloy.scene.entries import RIGID_FAST_EFFECT, mujoco_overrides

LIMIT = 0.70   # area and worst-direction support of the 4 over the overlap's (a regular octagon's 4: cos 45 deg)
ROWS = {2.0: 0.983, 5.0: 0.958, 10.0: 0.92, 20.0: 0.85, 30.0: 0.789, 45.0: 0.707}   # both ratios (CPU, this pick)
TIP_TOL = 0.03   # tipping onset with 4 contacts against 8, in m g (rows: <= 0.02)


def _square(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) @ np.array([[c, s], [-s, c]])


def _overlap(yaw: float) -> np.ndarray:
    """The overlap polygon of a 2 x 2 square and the same square yawed by ``yaw`` (both centred; Sutherland-Hodgman),
    counter-clockwise, z = 0."""
    poly, clip = list(_square(0.0)), _square(yaw)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

        def cut(p, q):
            d, e = q - p, b - a
            t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / (d[0] * e[1] - d[1] * e[0])
            return p + t * d

        out = []
        for p, q in zip(poly, poly[1:] + poly[:1]):
            if inside(q):
                if not inside(p):
                    out.append(cut(p, q))
                out.append(q)
            elif inside(p):
                out.append(cut(p, q))
        poly = out
    xy = np.asarray(poly)
    return np.column_stack([xy, np.zeros(len(xy))])


@wp.kernel
def _pick(n: int, dist: R.vec8f, pos: R.mat83f, out: wp.array(dtype=wp.vec4i)):
    out[0] = R.pick4(n, dist, pos, wp.vec3(0.0, 0.0, 1.0))


def _run_pick(pts: np.ndarray, depth: np.ndarray) -> np.ndarray:
    n = len(pts)
    dist = R.vec8f(*[float(d) for d in depth], *[1.0e10] * (8 - n))
    pos = R.mat83f(*[float(v) for p in pts for v in p], *[0.0] * (3 * (8 - n)))
    out = wp.zeros(1, dtype=wp.vec4i, device="cpu")
    wp.launch(_pick, dim=1, inputs=[n, dist, pos, out], device="cpu")
    return np.asarray(out.numpy()[0])


def _hull_area(xy: np.ndarray) -> float:
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    return float(ConvexHull(xy).volume)


def _support_ratio(kept: np.ndarray, full: np.ndarray) -> float:
    c = full.mean(axis=0)
    t = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    dirs = np.stack([np.cos(t), np.sin(t)], axis=1)
    return float(np.min(((kept - c) @ dirs.T).max(axis=0) / ((full - c) @ dirs.T).max(axis=0)))


def _best4(pts: np.ndarray) -> float:
    from itertools import combinations  # noqa: PLC0415

    return max(_hull_area(pts[list(c), :2]) for c in combinations(range(len(pts)), 4))


@pytest.mark.parametrize("yaw_deg", sorted(ROWS))
def test_the_pick_keeps_the_largest_hull_of_4(yaw_deg: float) -> None:
    pts = _overlap(math.radians(yaw_deg))
    assert len(pts) == 8
    depth = -1.0e-3 * (1.0 + 1.0e-3 * np.random.default_rng(int(yaw_deg)).random(len(pts)))  # level: 1 um noise
    keep = _run_pick(pts, depth)
    assert len(set(keep.tolist())) == 4
    area = _hull_area(pts[keep, :2]) / _hull_area(pts[:, :2])
    support = _support_ratio(pts[keep, :2], pts[:, :2])
    assert _hull_area(pts[keep, :2]) == pytest.approx(_best4(pts), rel=1e-6)
    assert area == pytest.approx(ROWS[yaw_deg], abs=2e-3) and support == pytest.approx(ROWS[yaw_deg], abs=2e-3)
    assert min(area, support) >= LIMIT


def test_a_tilted_pair_keeps_its_deepest_point() -> None:
    pts = _overlap(math.radians(20.0))
    depth = -1.0e-3 - 2.0e-4 * pts[:, 0]          # tilted about y: 0.2 mm deeper at +x
    keep = _run_pick(pts, depth)
    assert depth.argmin() in keep and len(set(keep.tolist())) == 4


def test_the_route_sets_the_key_under_the_fast_effect_only() -> None:
    assert mujoco_overrides({RIGID_FAST_EFFECT}, False, None)[R.REDUCE_KEY][0] is True
    assert "MuJoCo Warp" in mujoco_overrides({RIGID_FAST_EFFECT}, False, None)[R.REDUCE_KEY][1]
    assert R.REDUCE_KEY not in mujoco_overrides(set(), False, None)
    assert R.REDUCE_KEY not in mujoco_overrides(set(), True, None)


def test_a_second_install_cannot_switch_the_pick() -> None:
    from alloy.contact.mjw_boxbox_line import enable_line_boxbox  # noqa: PLC0415

    class _Solver:
        def _mujoco_warp_step(self, *a, **k):
            return None

    s = _Solver()
    assert enable_line_boxbox(s, reduce4=True) and enable_line_boxbox(s, reduce4=True)
    with pytest.raises(RuntimeError, match="reduce4"):
        enable_line_boxbox(s, reduce4=False)


def _sim(reduce4: bool, yaw: float, height: float, friction: float | None = None, step: float | None = None):
    """A cube of ``height`` yawed ``yaw`` on a static 0.4 x 0.4 x 0.1 m box, its entry under the fast effect (the
    4-point pick) or not, settled 0.5 s; CPU."""
    import alloy  # noqa: PLC0415
    from alloy.scene import entries as EN  # noqa: PLC0415

    inner = EN.mujoco_overrides

    def over(effects, *a, **k):
        return inner(set(effects) | {RIGID_FAST_EFFECT} if reduce4 else effects, *a, **k)

    EN.mujoco_overrides = over
    try:
        sc = alloy.Scene(dt=0.01)
        sc.add_box("base", size=(0.4, 0.4, 0.1), pos=(0.0, 0.0, 0.05), static=True, friction=friction)
        sc.add_box("top", size=(0.4, 0.4, height), pos=(0.0, 0.0, 0.1 + 0.5 * height + 1e-4), mass=2.0,
                   friction=friction, rpy=(0.0, 0.0, yaw))
        if step is not None:
            sc.rigid_step = step
        sim = alloy.build(sc, device="cpu")
        sim.settle(0.5)
        return sim
    finally:
        EN.mujoco_overrides = inner


def _rest(reduce4: bool, yaw: float = math.radians(30.0)) -> tuple[int, np.ndarray]:
    sim = _sim(reduce4, yaw, 0.1)
    nacon = int(np.asarray(sim.engine.stepper.solver.mjw_data.nacon.numpy()).ravel()[0])
    st = sim.state(["top"], world=0)
    out = np.concatenate([st["top.pos"], st["top.quat_wxyz"]])
    sim.close()
    return nacon, out


def _tip_onset(reduce4: bool, yaw_deg: float, push_deg: float) -> float:
    """The push (in m g) at the cube's top height at which it tilts past 2 degrees."""
    sim = _sim(reduce4, math.radians(yaw_deg), 0.4, friction=1.0, step=5.0e-3)
    u = (math.cos(math.radians(push_deg)), math.sin(math.radians(push_deg)))
    try:
        for k in range(20, 121):
            f = 0.01 * k * 2.0 * 9.81
            fx, fy = f * u[0], f * u[1]
            sim.apply_wrench("top", force=(fx, fy, 0.0), torque=(-0.2 * fy, 0.2 * fx, 0.0))   # at the top's height
            sim.step(10)
            w, x, y, z = (float(v) for v in sim.state(["top"], world=0)["top.quat_wxyz"])
            if math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))) > 2.0:
                return 0.01 * k
        return math.inf
    finally:
        sim.close()


def test_a_yawed_box_rests_on_4_contacts_where_it_had_8() -> None:
    n8, full = _rest(False)
    n4, reduced = _rest(True)
    assert n8 == 8 and n4 == 4, (n8, n4)
    assert np.allclose(reduced[:3], full[:3], atol=1e-5), (reduced, full)          # the same rest height and place
    assert abs(abs(reduced[3]) - abs(full[3])) < 1e-5                                # and level, same yaw


@pytest.mark.law
@pytest.mark.timeout(600)
@pytest.mark.parametrize("push_deg", [22.5, 90.0])
def test_a_yawed_cube_tips_where_it_tips_on_8_contacts(push_deg: float) -> None:
    on, off = _tip_onset(True, 20.0, push_deg), _tip_onset(False, 20.0, push_deg)
    assert 0.45 < off < 0.65 and abs(on - off) <= TIP_TOL, (on, off)
