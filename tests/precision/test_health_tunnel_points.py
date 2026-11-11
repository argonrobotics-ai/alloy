# SPDX-License-Identifier: Apache-2.0
"""Matter points of the tunnel check (audit ERR2R-5, ERR2R-6): a body is tracked at several points inside its own shapes.

P1  the points of each primitive (centre, axis points), the caps and the dedupe, in ``health_tunnel_setup``.
P2  a point that leaves by another face than it entered by, after 2 mm or more inside, is TUNNEL; a shallower graze is not.
P3  a robot link whose outer part meets a thin static plate (its COM never does): TUNNEL / INVALID, named; slow ramps
    that stop short of the plate or rest on it: never.
P4  a cable dropped on a plate edge: no flag when nothing passes (it slides off the edge); laid gently or a ball rolling
    off a table edge: no flag at any period.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.precision import health_tunnel as HT
from alloy.precision import health_tunnel_setup as S

ROUTES = ["mjc", "warp"]
G = S.G


def _shape(gtype, half, pos=(0, 0, 0), rot=(0, 0, 0, 1), from_mesh=False) -> S.BodyShape:
    return S.BodyShape(int(gtype), np.array(half, float), np.array([*pos, *rot], float), 1.0, from_mesh)


def test_p1_each_primitive_gives_its_centre_and_axis_points() -> None:
    cap, _ = S.matter_points([_shape(G.CAPSULE, (0.02, 0.15, 0))], None)
    assert np.allclose(sorted(map(tuple, cap.round(6))), [(0, 0, -0.15), (0, 0, 0), (0, 0, 0.15)])
    box, _ = S.matter_points([_shape(G.BOX, (0.1, 0.02, 0.03))], None)
    assert np.allclose(sorted(map(tuple, box.round(6))), [(-0.07, 0, 0), (0, 0, 0), (0.07, 0, 0)])
    cube, _ = S.matter_points([_shape(G.BOX, (0.05, 0.05, 0.05))], None)
    ball, _ = S.matter_points([_shape(G.SPHERE, (0.05, 0, 0))], None)
    assert len(cube) == 1 and len(ball) == 1
    cyl, _ = S.matter_points([_shape(G.CYLINDER, (0.05, 0.01, 0))], None)
    assert np.abs(cyl[:, 2]).max() == pytest.approx(0.009)                 # inset: inside the solid


def test_p1_points_are_in_the_body_frame_and_a_mesh_box_gives_none() -> None:
    pts, _ = S.matter_points([_shape(G.CAPSULE, (0.02, 0.15, 0), pos=(0.3, 0, 0))], None)
    assert pts[:, 2].max() == pytest.approx(0.15) and pts[:, 0].min() == pytest.approx(0.3)
    assert len(S.matter_points([_shape(G.BOX, (0.1, 0.02, 0.03), from_mesh=True)], None)[0]) == 0


def test_p1_caps_and_dedupe() -> None:
    many = [_shape(G.CAPSULE, (0.02, 0.1, 0), pos=(0.1 * i, 0, 0)) for i in range(20)]
    pts, dropped = S.matter_points(many, None)
    assert len(pts) == S.BODY_POINT_CAP and dropped > 0
    pts, _ = S.matter_points([_shape(G.SPHERE, (0.05, 0, 0))], np.zeros(3))      # the COM is the centre: no copy
    assert len(pts) == 0


@pytest.fixture(scope="module")
def cube_world():
    scene = alloy.Scene(dt=0.01)
    scene.add_box("block", size=(0.2, 0.2, 0.2), pos=(0, 0, 0.3), static=True)             # top at z = 0.4, side x = 0.1
    scene.add_sphere("ball", 0.01, pos=(0, 0, 1.0), density=1000.0)
    return alloy.build(scene, device="cpu").engine.model


def _hop(model, tw, a, b) -> None:
    names = list(model.body_label)
    ball = [i for i, n in enumerate(names) if n.endswith("ball")][0]
    s0, s1 = model.state(), model.state()
    for st, p in ((s0, a), (s1, b)):
        q = st.body_q.numpy().copy()
        q[ball, :3] = p
        st.body_q.assign(q)
    tw.after_step(s0, s1, 0.00125)


def _watch(model):
    return HT.TunnelWatch(model, S.judged_bodies(model), S.static_shapes(model))


def test_p2_leaving_by_another_face_after_2_mm_is_tunnel_and_a_graze_is_not(cube_world) -> None:
    deep = _watch(cube_world)
    _hop(cube_world, deep, (0.0, 0, 0.41), (0.0, 0, 0.395))                # in by the top, 5 mm deep
    assert deep.tunnel.numpy().tolist() == [0]
    _hop(cube_world, deep, (0.0, 0, 0.395), (0.12, 0, 0.395))              # out by the side face
    assert deep.tunnel.numpy().tolist() == [1]
    shallow = _watch(cube_world)
    _hop(cube_world, shallow, (0.0, 0, 0.41), (0.0, 0, 0.399))             # 1 mm deep
    _hop(cube_world, shallow, (0.0, 0, 0.399), (0.12, 0, 0.399))
    assert shallow.tunnel.numpy().tolist() == [0]
    back = _watch(cube_world)
    _hop(cube_world, back, (0.0, 0, 0.41), (0.0, 0, 0.39))                 # in 10 mm and out by the same face
    _hop(cube_world, back, (0.0, 0, 0.39), (0.0, 0, 0.42))
    assert back.tunnel.numpy().tolist() == [0]


def _arm(kp: float, kv: float, route: str, monkeypatch) -> alloy.Simulation:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    xml = f"""<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position joint="shoulder" kp="{kp}" kv="{kv}"/></actuator></mujoco>"""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=xml, pos=(0, 0, 0.5))
    scene.add_box("plate", size=(0.4, 0.4, 0.005), pos=(0.25, 0, 0.42), static=True)       # met at about q = 0.64
    return alloy.build(scene, device="cpu")


def _severities(sim, steps: int) -> set[str]:
    out: set[str] = set()
    for _ in range(steps):
        sim.step(1)
        out |= {w["severity"] for w in sim.health()["worlds"]}
    return out


@pytest.mark.parametrize("route", ROUTES)
def test_p3_a_link_jumping_through_a_plate_by_its_outer_part_is_tunnel(route, monkeypatch) -> None:
    sim = _arm(5000, 100, route, monkeypatch)
    sim.set_targets({"shoulder": 1.8})
    assert "INVALID" in _severities(sim, 60)
    check = sim.check()
    text = str(check["problems"])
    assert not check["ok"] and "link1" in text and "plate" in text


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("target, seconds", [(0.5, 3.0), (0.8, 4.0)])
def test_p3_slow_ramps_short_of_or_resting_on_the_plate_are_never_tunnel(target, seconds, route, monkeypatch) -> None:
    sim = _arm(200, 10, route, monkeypatch)
    traj = sim.run(seconds, record=["shoulder"], targets={"shoulder": (0.0, target)})
    assert "INVALID" not in _severities(sim, 20) and sim.check()["ok"] and traj.meta["health"]["ok"]
    q = float(np.array(sim.state(["shoulder"], world=0)["shoulder.q"]))
    assert (q == pytest.approx(0.5, abs=0.05)) if target == 0.5 else (0.5 < q < 0.7)       # blocked by the plate


def _rod_sim(drop: float, route: str, monkeypatch) -> alloy.Simulation:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.5, 0.5, 0.01), pos=(-0.25, 0, 0.3), static=True)        # edge at x = 0
    scene.add_rod("rod", length=0.4, radius=0.004, pos=(-0.2, 0, 0.305 + 0.004 + drop), direction=(1, 0, 0))
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("drop, steps", [(1.0, 220), (0.02, 150)])
def test_p4_a_cable_on_a_plate_edge_is_not_tunnel_when_it_slides_off_or_lies(drop, steps, route, monkeypatch) -> None:
    """At 1 m the cable is dragged over the edge by its hanging half and swings round under the plate: no point is
    ever inside the plate (round_rod trace in notes/gap-e2/round3.md); laid gently it simply rests."""
    sim = _rod_sim(drop, route, monkeypatch)
    for _ in range(steps):
        sim.step(1)
        assert all("TUNNEL" not in w["flags"] for w in sim.health()["worlds"])
    assert sim.check()["ok"]


@pytest.mark.parametrize("route", ROUTES)
def test_p4_a_ball_rolling_off_a_table_edge_is_not_tunnel(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_sphere("ball", 0.02, pos=(0.0, 0, 0.42), density=1000.0)
    sim = alloy.build(scene, device="cpu")
    mass = 4 / 3 * np.pi * 0.02**3 * 1000.0
    for k in range(120):
        sim.apply_wrench("ball", force=(mass * 1.0 / 0.05 if k < 5 else 0.0, 0.0, mass * 9.81 if k < 5 else 0.0))
        sim.step(1)
        assert all("TUNNEL" not in w["flags"] for w in sim.health()["worlds"])
    assert sim.state(world=0)["ball.pos"][0] > 0.3                          # it left the table


def _mesh_plate_sim(heights, route, monkeypatch, thick: float = 0.005, hull_box: bool = True) -> alloy.Simulation:
    """A static convex-mesh plate (a box's 8 vertices; ``hull_box=False``: its top shrunk, a wedge) and balls above it."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    x, y, z = 0.25, 0.25, thick
    v = np.array([[a, b, c] for a in (-x, x) for b in (-y, y) for c in (-z, z)], float)
    if not hull_box:
        v[v[:, 2] > 0, 0] *= 0.2                                           # the top face narrows: a hull under 98 % of its box
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
                  [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    scene = alloy.Scene(worlds=len(heights), dt=0.01)
    scene.add_ground()
    scene.add_mesh("plate", vertices=v, faces=f, collision="convex", pos=(0, 0, 0.3), static=True)
    scene.add_sphere("ball", 0.01, pos=np.array([[0, 0, 0.3 + z + 0.01 + h] for h in heights]), density=7800.0)
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("route", ROUTES)
def test_p5_balls_passing_a_static_convex_mesh_plate_are_tunnel_and_a_ball_resting_on_it_is_not(route, monkeypatch) -> None:
    heights = [0.05, 5.0, 6.0, 8.0, 12.0]

    def run(watched: bool):
        with monkeypatch.context() as m:
            if not watched:
                m.setattr(HT, "watch", lambda solver: None)
            sim = _mesh_plate_sim(heights, route, monkeypatch)
            z = sim.run(2.0, record=["ball"])["ball.pos"][:, :, 2].min(axis=0)
            return sim, z
    _, truth = run(False)
    sim, _ = run(True)
    passed = truth < 0.29
    assert passed[1:].any() and not passed[0]                              # the oracle: the high drops really pass
    sev = [w["severity"] for w in sim.health()["worlds"]]
    assert [s == "INVALID" for s in sev] == passed.tolist()


def test_p5_a_wedge_mesh_is_not_judged_as_a_box(monkeypatch) -> None:
    sim = _mesh_plate_sim([0.5], "warp", monkeypatch, thick=0.05, hull_box=False)
    model = sim.engine.model
    assert S.static_shapes(model).tolist() == [int(i) for i in S.static_shapes(model)]
    labels = [str(model.shape_label[int(i)]) for i in S.static_shapes(model)]
    assert not any("plate" in n for n in labels)
