# SPDX-License-Identifier: Apache-2.0
"""Convex-mesh hulls capped by geometric error (alloy.contact.hull_cap; physics ruling 2026-11-14 on finding C5).

Laws:
- contract: the default is Newton's cap; the hull is MuJoCo's own, mapped back onto the input vertices;
- the rule: a brass top's lathe (a 0.3 mm tip on a 5 cm top) is out of tolerance at 64 vertices; every chosen cap is
  within tolerance, or at the ceiling with a warning (once per mesh) whose fix suits the mesh (convex or not);
- the large-mesh path (more than 50,000 vertices): its collider is MuJoCo's capped hull, on the input vertices, with
  the whole mesh's mass and inertia (as with the rule off);
- the fix a warning names: a primitive or fewer vertices for a convex mesh (a scanned ball, noise and all),
  decompose for a concave one (a cup);
- the distance is exact: point-to-triangle against an independent reference; a point dropped 0.39 mm beyond a sharp
  apex reads 0.39 mm (its face planes say 0.10); a 60,000-vertex round scan stays under 600 MB;
- degenerate input (flat, 3 points, collinear) keeps Newton's cap, and a flat convex mesh fails its build with the same
  error as without the rule;
- the opt-out: a card's ``hull_vertices`` is checked (an int, 4 to 2,048, convex only); 64 runs bit for
  bit as with the rule off (MuJoCo C, solver asserted; the large path too), a user cap never warns, and the field
  survives the scene JSON and save / load;
- bit for bit: a mesh whose full hull has at most 64 vertices keeps Newton's cap, and whole runs with it are the same
  with the rule and without (``ALLOY_HULL_TOL=0``); a robot pack's run is the same too (MJCF meshes keep the importer);
- the top: spun to 250 rad/s on its tip, its spin and energy decay (no gain) on MuJoCo C, the CPU Warp route and cuda;
  with Newton's 64 (planted) it gains both on MuJoCo C.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

import alloy
from alloy._vendor import newton
from alloy.contact.hull_limits import HULL_VERTICES_MAX
from alloy.contact.hull_limits import hull_vertices_problem
from alloy.contact import hull_cap as H

pytestmark = pytest.mark.timeout(1800)
PROFILE_MM = [(0, 0.3), (0.2, 0.9), (0.6, 1.5), (1.2, 2.2), (4, 5.4), (8, 9.8), (12, 13.6), (16, 17.0), (20, 19.9),
              (24, 22.3), (28, 24.0), (31, 24.8), (33, 25.0), (35, 24.6), (37, 23.0), (39, 20.0), (40.5, 15.0),
              (41.3, 9.0), (41.6, 3.0), (41.7, 0.3)]           # (z, r) bottom to top: a brass peg top, 5 cm across
TILT = math.radians(15.0)
SPIN = 250.0


def lathe(profile=PROFILE_MM, segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A closed solid of revolution about +z from (z, r) mm points, bottom to top (flat caps at both ends)."""
    ang = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    rings = [np.stack([r / 1e3 * np.cos(ang), r / 1e3 * np.sin(ang), np.full(segments, z / 1e3)], 1)
             for z, r in profile]
    verts = np.concatenate(rings + [np.array([[0.0, 0.0, profile[0][0] / 1e3], [0.0, 0.0, profile[-1][0] / 1e3]])])
    bottom, top = len(verts) - 2, len(verts) - 1
    faces = []
    for i in range(len(profile) - 1):
        a, b = i * segments, (i + 1) * segments
        for k in range(segments):
            k2 = (k + 1) % segments
            faces += [(a + k, a + k2, b + k2), (a + k, b + k2, b + k)]
    last = (len(profile) - 1) * segments
    for k in range(segments):
        k2 = (k + 1) % segments
        faces += [(bottom, k2, k), (top, last + k, last + k2)]
    return verts, np.array(faces, dtype=np.int64)


def _sphere_cloud(n: int, radius: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    return radius * p / np.linalg.norm(p, axis=1)[:, None]


def test_the_default_is_newtons_cap_and_the_hull_is_mujocos_own():
    assert H.DEFAULT == newton.Mesh.MAX_HULL_VERTICES
    assert H.LADDER[0] == H.DEFAULT and H.LADDER[-1] == H.CEILING
    v, _ = lathe()
    v = v.astype(np.float32).astype(np.float64)
    tree = cKDTree(v)
    for cap in (64, 256, -1):
        hull = H.mujoco_hull(v, cap)
        assert len(hull) == (cap if cap > 0 else len(ConvexHull(v).vertices))
        assert tree.query(hull)[0].max() < 1e-8                 # every hull vertex is an input vertex


def test_the_tops_tip_fails_at_64_and_each_choice_is_within_tolerance_or_warns(caplog):
    v, _ = lathe()
    full = v[ConvexHull(v).vertices]
    assert H.deviation(full, H.mujoco_hull(v, 64)) > 10 * H.tolerance(v)          # 1.3 mm against 0.1 mm
    cases = {"top": v, "cloud_2k": _sphere_cloud(2000, 0.03), "cloud_20k": _sphere_cloud(20000, 0.05, seed=1),
             "box_8": np.array([[x, y, z] for x in (0, 0.1) for y in (0, 0.2) for z in (0, 0.05)], float)}
    caplog.set_level(logging.WARNING, logger="alloy.api")
    for name, pts in cases.items():
        c = H.choose(pts)
        p = np.asarray(pts, np.float32).astype(np.float64)
        dev = 0.0 if c.cap >= c.full or c.full <= H.DEFAULT else H.deviation(p[ConvexHull(p).vertices],
                                                                               H.mujoco_hull(p, c.cap))
        assert dev == pytest.approx(c.deviation, abs=1e-12), name
        assert c.within or c.cap == H.CEILING, (name, c)
        for _ in range(2):
            assert H.maxhullvert(pts, name) == c.cap
        said = [r for r in caplog.records if r.getMessage().startswith(name + ":")]
        assert len(said) == (0 if c.within else 1), (name, said)
        if said:
            msg = said[0].getMessage()
            assert (H.FIX_CONVEX if c.convex else H.FIX) in msg and f"{c.deviation * 1e3:.2f} mm" in msg, msg
    assert H.choose(cases["box_8"]).cap == H.DEFAULT


def _reference_distance(p, a, b, c) -> np.ndarray:
    """Point-to-triangle distance by another route: the plane distance when the projection falls inside, else the
    nearest edge."""
    out = []
    for q, x, y, z in zip(p, a, b, c):
        n = np.cross(y - x, z - x)
        n /= np.linalg.norm(n)
        proj = q - n * np.dot(q - x, n)
        if all(np.dot(np.cross(v - u, proj - u), n) >= 0 for u, v in ((x, y), (y, z), (z, x))):
            out.append(abs(np.dot(q - x, n)))
            continue
        seg = [np.linalg.norm(q - (u + np.clip(np.dot(q - u, v - u) / np.dot(v - u, v - u), 0, 1) * (v - u)))
               for u, v in ((x, y), (y, z), (z, x))]
        out.append(min(seg))
    return np.array(out)


def _uv_sphere(r: float, n_lat: int, n_lon: int) -> np.ndarray:
    th = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1).reshape(-1, 3)
    return r * np.vstack([[0.0, 0.0, 1.0], ring, [0.0, 0.0, -1.0]])


def test_the_distance_is_exact_and_bounded_in_memory():
    import tracemalloc

    rng = np.random.default_rng(3)
    p, a, b, c = (rng.normal(size=(2000, 3)) for _ in range(4))
    assert np.abs(H.point_triangle_distance(p, a, b, c) - _reference_distance(p, a, b, c)).max() < 1e-12
    alpha, height, h = math.radians(15.0), 0.02, 0.39e-3          # a sharp square pyramid, a point h past its apex
    r = height * math.tan(alpha)
    hull = np.array([[r, r, 0], [-r, r, 0], [-r, -r, 0], [r, -r, 0], [0, 0, height]])
    full = np.vstack([hull, [[0.0, 0.0, height + h]]])
    eq = ConvexHull(hull).equations
    assert float((full[-1] @ eq[:, :3].T + eq[:, 3]).max()) < 0.3 * h        # what the face planes say
    assert H.deviation(full, hull) == pytest.approx(h, rel=1e-9)
    ball = np.asarray(_uv_sphere(0.04, 152, 400), np.float32).astype(np.float64)   # 60,402 vertices, all on its hull
    tracemalloc.start()
    try:
        H.choose(ball)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 600e6, peak


def _uv_sphere_faces(n_lat: int, n_lon: int) -> np.ndarray:
    """Outward faces of ``_uv_sphere(r, n_lat, n_lon)``."""
    rings, bottom = n_lat - 1, 1 + (n_lat - 1) * n_lon
    idx = lambda i, k: 1 + i * n_lon + (k % n_lon)                                       # noqa: E731
    faces = []
    for k in range(n_lon):
        faces += [(0, idx(0, k), idx(0, k + 1)), (bottom, idx(rings - 1, k + 1), idx(rings - 1, k))]
        for i in range(rings - 1):
            faces += [(idx(i, k), idx(i + 1, k), idx(i + 1, k + 1)), (idx(i, k), idx(i + 1, k + 1), idx(i, k + 1))]
    return np.array(faces, dtype=np.int64)


def _large_sphere_collider(monkeypatch, env: str, hull_vertices: int | None = None) -> tuple[int, float, np.ndarray]:
    """Build a 60,402-vertex convex sphere (the emitter's large path): its colliding mesh's hull vertex count in
    MuJoCo, and the body's mass and principal inertia."""
    import mujoco

    monkeypatch.setenv(H.ENV, env)
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_mesh("ball", vertices=_uv_sphere(0.04, 152, 400), faces=_uv_sphere_faces(152, 400), collision="convex",
               pos=(0, 0, 0.05), density=500.0)
    sim = alloy.build(_with_hull_vertices(s, "ball", hull_vertices), device="cpu")
    try:
        st = sim.engine.stepper
        solver = next(x for x in (getattr(e, "solver", e) for e in (getattr(st, "entries", None) or [st]))
                      if hasattr(x, "mj_model"))
        m = solver.mj_model
        geom = next(i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                    and (m.geom_contype[i] or m.geom_conaffinity[i]))
        k = int(m.geom_dataid[geom])
        b = int(m.geom_bodyid[geom])
        return int(m.mesh_graph[m.mesh_graphadr[k]]), float(m.body_mass[b]), np.sort(np.asarray(m.body_inertia[b]))
    finally:
        sim.close()


def test_the_large_mesh_path_collides_with_mujocos_capped_hull(monkeypatch):
    v = np.asarray(_uv_sphere(0.04, 152, 400), np.float32).astype(np.float64)
    c = H.choose(v)
    assert c.cap == H.CEILING and c.deviation < 0.11e-3, c                 # MuJoCo's hull: 0.103 mm at 2,048
    n_rule, mass_rule, inertia_rule = _large_sphere_collider(monkeypatch, "1")
    n_off, mass_off, inertia_off = _large_sphere_collider(monkeypatch, "0")
    assert (n_rule, n_off) == (c.cap, H.DEFAULT)
    assert mass_rule == pytest.approx(mass_off, rel=1e-6) and np.allclose(inertia_rule, inertia_off, rtol=1e-6)
    mesh = newton.Mesh(v.astype(np.float32), _uv_sphere_faces(152, 400).reshape(-1).astype(np.int32),
                       compute_inertia=False)
    monkeypatch.setenv(H.ENV, "1")
    mesh.maxhullvert = H.maxhullvert(v, "ball")                            # as the emitter does, before capped_mesh
    hull = np.asarray(H.capped_mesh(mesh).vertices, float)
    assert cKDTree(v).query(hull)[0].max() == 0.0                           # input vertices themselves
    assert H.deviation(v[ConvexHull(v).vertices], hull) == pytest.approx(c.deviation, abs=1e-9)


def test_the_fix_named_suits_the_shape():
    rng = np.random.default_rng(5)
    ball = _uv_sphere(0.04, 80, 150)
    for noise in (0.0, 1e-5, 4e-5):                                # a scanned ball: radial noise up to 0.04 mm
        pts = ball * (1 + noise / 0.04 * rng.uniform(-1, 1, size=(len(ball), 1)))
        assert H.convex(pts, H.mujoco_hull(pts, 1024), 1e-4), noise
    outer = [(0, 30), (2, 40), (80, 40)]
    inner = [(80, 37), (4, 37), (4, 0.5)]                          # a cup: 3 mm walls, the inside goes down to a floor
    cup = np.concatenate([lathe(outer)[0], lathe(inner)[0]])
    assert not H.convex(cup, H.mujoco_hull(cup, 256), 1e-4)


def test_degenerate_input_keeps_newtons_cap_and_its_error(monkeypatch):
    grid = np.array([[x, y, 0.0] for x in np.linspace(0, 0.1, 10) for y in np.linspace(0, 0.1, 10)])
    line = np.array([[t, 2 * t, 3 * t] for t in np.linspace(0, 0.1, 20)])
    for pts in (grid, grid[:3], line):
        assert H.choose(pts) == H.HullChoice(H.DEFAULT, 0.0, H.tolerance(pts), 0)
    faces = np.array([[i * 10 + j, (i + 1) * 10 + j, (i + 1) * 10 + j + 1] for i in range(9) for j in range(9)]
                     + [[i * 10 + j, (i + 1) * 10 + j + 1, i * 10 + j + 1] for i in range(9) for j in range(9)])
    errors = []
    for env in ("1", "0"):
        monkeypatch.setenv(H.ENV, env)
        s = alloy.Scene(dt=0.01)
        s.add_ground()
        s.add_mesh("plate", vertices=grid, faces=faces, collision="convex", pos=(0, 0, 0.1), static=True)
        s.add_sphere("ball", 0.02, pos=(0, 0, 0.3))
        try:
            alloy.build(s, device="cpu").close()
            errors.append(None)
        except Exception as e:                                     # noqa: BLE001 - compared, not judged
            errors.append((type(e).__name__, str(e)))
    assert errors[0] == errors[1], errors
    assert errors[0] is None or "Qhull" not in errors[0][1], errors


def _mesh_scene(verts: np.ndarray, faces: np.ndarray) -> alloy.Scene:
    s = alloy.Scene(worlds=2, dt=0.01)
    s.add_ground()
    s.add_mesh("part", vertices=verts, faces=faces, collision="convex", pos=(0.0, 0.0, 0.1),
               rpy=(0.3, 0.2, 0.0), material="steel_carbon")
    s.add_box("brick", size=(0.04, 0.04, 0.04), pos=(0.2, 0.0, 0.05), material="steel_carbon")
    return s


def _digest(scene: alloy.Scene, steps: int) -> str:
    sim = alloy.build(scene, device="cpu", seed=0)
    try:
        h = hashlib.sha256()
        for _ in range(steps):
            sim.step(1)
            st = sim.state()
            for k in sorted(st):
                h.update(np.ascontiguousarray(st[k]).tobytes())
        assert sim.check()["ok"], sim.check()["problems"]
        return h.hexdigest()
    finally:
        sim.close()


def test_a_mesh_with_a_hull_of_64_or_fewer_runs_bit_for_bit(monkeypatch):
    v, f = lathe([(0, 10), (10, 25), (30, 25), (40, 10)], segments=16)        # 64 ring vertices + 2 cap centres
    assert len(ConvexHull(v).vertices) <= H.DEFAULT and H.choose(v).cap == H.DEFAULT
    with_rule = _digest(_mesh_scene(v, f), 30)
    monkeypatch.setenv(H.ENV, "0")
    assert _digest(_mesh_scene(v, f), 30) == with_rule


def test_a_robot_pack_runs_bit_for_bit(monkeypatch):
    from alloy.cards.library import default_root

    def scene() -> alloy.Scene:
        s = alloy.Scene(dt=0.01)
        s.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")
        return s

    with_rule = _digest(scene(), 10)
    monkeypatch.setenv(H.ENV, "0")
    assert _digest(scene(), 10) == with_rule


def _top_scene() -> alloy.Scene:
    s = alloy.Scene(dt=1.0 / 300)
    s.rigid_step = 1.0 / 1200
    s.add_ground(z=-0.01)
    s.add_box("worktop", size=(1.2, 1.2, 0.02), pos=(0.0, 0.0, -0.01), static=True)
    v, f = lathe()
    rx = np.array([[1, 0, 0], [0, math.cos(TILT), -math.sin(TILT)], [0, math.sin(TILT), math.cos(TILT)]])
    s.add_mesh("top", vertices=v, faces=f, collision="convex", pos=(0.0, 0.0, -float((v @ rx.T)[:, 2].min()) + 2e-4),
               rpy=(TILT, 0.0, 0.0), material="brass_c360")
    return s


def _spin_energy(sim) -> tuple[float, float]:
    """The top's spin about its own axis (rad/s) and its energy (J: kinetic + potential) from the engine state."""
    eng, model = sim.engine, sim.engine.model
    labels = list(getattr(model, "body_label", None) or model.body_key)
    b = next(i for i, n in enumerate(labels) if str(n).endswith("top"))
    q = eng.state.body_q.numpy()[b].astype(float)
    qd = eng.state.body_qd.numpy()[b].astype(float)
    x, y, z, w = q[3:7]
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = float(model.body_mass.numpy()[b])
    inertia = np.asarray(model.body_inertia.numpy()[b], float).reshape(3, 3)
    com = q[:3] + rot @ np.asarray(model.body_com.numpy()[b], float)
    v, om = qd[:3], qd[3:6]
    ke = 0.5 * m * v @ v + 0.5 * om @ (rot @ inertia @ rot.T) @ om
    return float(om @ rot[:, 2]), float(ke + m * 9.81 * com[2])


def _top_run(device: str, seconds: float) -> tuple[float, float, float, float]:
    sim = alloy.build(_top_scene(), device=device, seed=0)
    try:
        i3 = float(np.asarray(sim.engine.model.body_inertia.numpy()[-1], float).reshape(3, 3)[2, 2])
        axis = (0.0, -math.sin(TILT), math.cos(TILT))
        sim.apply_wrench("top", torque=tuple(i3 * SPIN * 30 * c for c in axis))   # 1/30 s: about 250 rad/s
        sim.step(10)
        sim.apply_wrench("top")
        s0, e0 = _spin_energy(sim)
        sim.step(round(seconds / sim.dt))
        s1, e1 = _spin_energy(sim)
        assert sim.check()["ok"], sim.check()["problems"]
        return s0, e0, s1, e1
    finally:
        sim.close()


@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_the_top_loses_spin_and_energy_on_the_cpu(route, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    s0, e0, s1, e1 = _top_run("cpu", 1.5 if route == "mjc" else 0.75)
    assert s0 > 240.0, s0
    assert s1 < s0 and e1 < e0, (route, s0, s1, e0, e1)


def test_the_top_gains_with_newtons_64_on_mujoco_c(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    monkeypatch.setenv(H.ENV, "0")
    s0, e0, s1, e1 = _top_run("cpu", 1.5)
    assert s1 > s0 and e1 > e0, (s0, s1, e0, e1)                # the finding this rule fixes (C5)


@pytest.mark.gpu
def test_the_top_loses_spin_and_energy_on_cuda():
    s0, e0, s1, e1 = _top_run("cuda:0", 1.5)
    assert s0 > 240.0 and s1 < s0 and e1 < e0, (s0, s1, e0, e1)


def _with_hull_vertices(scene: alloy.Scene, name: str, value: Any) -> alloy.Scene:
    """``scene`` with ``shape.mesh.hull_vertices = value`` on mesh ``name`` (through the scene JSON; None: as is)."""
    if value is None:
        return scene
    d = json.loads(scene.to_json())
    d["items"][name]["shape"]["mesh"]["hull_vertices"] = value
    return alloy.Scene.from_dict(d)


def test_hull_vertices_is_checked():
    assert HULL_VERTICES_MAX == H.CEILING
    assert "collision='convex' only" in hull_vertices_problem(64, "decompose")
    v, f = lathe()
    for bad in (3, 2049, True, "64"):
        s = alloy.Scene(dt=0.01)
        s.add_mesh("part", vertices=v, faces=f, collision="convex", pos=(0, 0, 0.1))
        with pytest.raises(Exception, match="hull_vertices") as err:
            alloy.build(_with_hull_vertices(s, "part", bad), device="cpu")
        if isinstance(bad, int) and not isinstance(bad, bool):               # non-ints stop at the codec's type check
            assert hull_vertices_problem(bad, "convex") in str(err.value), (bad, str(err.value)[:300])
    for collision, value in (("mesh", 64), ("sdf", 64), ("mesh", 1)):     # set on a mesh that is not convex: refused
        s = alloy.Scene(dt=0.01)
        s.add_ground()
        s.add_mesh("part", vertices=v, faces=f, collision=collision, pos=(0, 0, 0.1), static=True)
        with pytest.raises(Exception, match="hull_vertices") as err:
            alloy.build(_with_hull_vertices(s, "part", value), device="cpu")
        assert hull_vertices_problem(value, collision) in str(err.value), (collision, value, str(err.value)[:300])


def test_a_user_cap_of_64_is_newtons_hull_bit_for_bit(monkeypatch, caplog):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")

    def digest(scene: alloy.Scene) -> str:
        sim = alloy.build(scene, device="cpu", seed=0)
        try:
            assert type(_solver_of(sim)).__name__ == "MjcCoreSolver"
            h = hashlib.sha256()
            for _ in range(40):
                sim.step(1)
                h.update(np.ascontiguousarray(sim.state(["top"])["top.quat_wxyz"]).tobytes())
            return h.hexdigest()
        finally:
            sim.close()

    capped = digest(_with_hull_vertices(_top_scene(), "top", 64))
    monkeypatch.setenv(H.ENV, "0")
    assert digest(_top_scene()) == capped                                # the pre-rule code path
    n_off, mass_off, inertia_off = _large_sphere_collider(monkeypatch, "0")
    n_cap, mass_cap, inertia_cap = _large_sphere_collider(monkeypatch, "1", hull_vertices=64)
    assert n_cap == n_off == H.DEFAULT and mass_cap == mass_off and np.array_equal(inertia_cap, inertia_off)
    caplog.set_level(logging.WARNING, logger="alloy.api")
    ball = np.asarray(_uv_sphere(0.04, 152, 400), np.float32).astype(np.float64)    # out of tolerance under the rule
    assert H.maxhullvert(ball, "ball_capped", fixed=1024) == 1024
    assert not [r for r in caplog.records if r.getMessage().startswith("ball_capped:")]


def _solver_of(sim):
    st = sim.engine.stepper
    return next(x for x in (getattr(e, "solver", e) for e in (getattr(st, "entries", None) or [st]))
                if hasattr(x, "mj_model") or hasattr(x, "mjw_model"))


def test_hull_vertices_survives_json_and_save(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    v, f = lathe()
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_mesh("part", vertices=v, faces=f, collision="convex", pos=(0, 0, 0.1))
    s = _with_hull_vertices(s, "part", 128)
    again = alloy.Scene.from_dict(json.loads(s.to_json()))
    assert json.loads(again.to_json())["items"]["part"]["shape"]["mesh"]["hull_vertices"] == 128
    scene_file = s.save(tmp_path / "saved")
    cards = [p.read_text() for p in (tmp_path / "saved" / "objects").rglob("*.yaml")]
    assert sum("hull_vertices: 128" in c for c in cards) == 1, cards
    sim = alloy.build(alloy.Scene.load(scene_file), device="cpu")
    try:
        assert _collider_hull_vertices(sim) == 128
    finally:
        sim.close()


def _collider_hull_vertices(sim) -> int:
    import mujoco

    m = getattr(_solver_of(sim), "mj_model", None)
    geom = next(i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                and (m.geom_contype[i] or m.geom_conaffinity[i]))
    return int(m.mesh_graph[m.mesh_graphadr[int(m.geom_dataid[geom])]])
