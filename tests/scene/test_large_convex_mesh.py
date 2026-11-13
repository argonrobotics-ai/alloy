# SPDX-License-Identifier: Apache-2.0
"""A high-poly mesh with the default collision="convex" builds (AUDIT S3-004, Gemini 0.1.2): a 1.3 M-face scan
crashed ``alloy.build`` with an uncaught MemoryError (6.25 TiB) in the rigid step's width probe
(``alloy.scene.rigid_step_bodies.min_width``: every vertex against every face normal), and past it Newton's convex
source setup ran vertex by vertex. The laws: the probe stays bounded (hull vertices, binned directions, passes) and
gives the one-pass answer where both run; a convex mesh of more than ``CONVEX_SOURCE_MAX`` vertices collides as its
capped hull with the whole mesh's mass, and lands on the ground."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.scene import rigid_step_bodies as RSB
from alloy.scene.emitters import rigid as RE


def uv_sphere(r: float, n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed latitude-longitude sphere: (n_lat - 1) * n_lon + 2 vertices, 2 * (n_lat - 1) * n_lon faces, wound
    outward."""
    th = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1).reshape(-1, 3)
    v = r * np.vstack([[0.0, 0.0, 1.0], ring, [0.0, 0.0, -1.0]])
    rings, bottom = n_lat - 1, len(v) - 1
    j = np.arange(n_lon)
    idx = lambda i, k: 1 + i * n_lon + (k % n_lon)                          # noqa: E731
    faces = [np.stack([np.zeros(n_lon, int), idx(0, j), idx(0, j + 1)], axis=1)]
    for i in range(rings - 1):
        a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
        faces += [np.stack([a, d, c], axis=1), np.stack([a, c, b], axis=1)]
    faces.append(np.stack([np.full(n_lon, bottom), idx(rings - 1, j + 1), idx(rings - 1, j)], axis=1))
    return v, np.vstack(faces)


def plate(t: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two n x n grids t apart (a 0.2 m x 0.1 m plate's faces), turned off the axes."""
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, n), np.linspace(-0.05, 0.05, n), indexing="ij")
    top = np.stack([x.ravel(), y.ravel(), np.full(x.size, 0.5 * t)], axis=1)
    v = np.vstack([top, top * [1.0, 1.0, -1.0]])
    q = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    tri = np.vstack([np.stack([q, q + n, q + n + 1], 1), np.stack([q, q + n + 1, q + 1], 1)])
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return v @ R.T, np.vstack([tri, tri[:, ::-1] + n * n])


def test_a_million_face_width_probe_is_bounded():
    """640,000 faces (a one-pass probe: 1.6 TB): the sphere's diameter within 1 %, under 1 GB. ~10 s."""
    v, f = uv_sphere(0.08, 400, 800)
    assert len(v) * len(f) * 8 > 1e12
    tracemalloc.start()
    width = RSB.min_width(v, f)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert width == pytest.approx(0.16, rel=0.01) and peak < 1.0e9, (width, peak)


def test_the_bounded_probe_matches_one_pass(monkeypatch):
    """On a mesh small enough for one pass, the hull path with every direction gives the same width; with the
    directions binned to 64, a turned thin plate keeps its exact thickness (its flat sides' normals survive)."""
    v, f = uv_sphere(0.05, 30, 60)
    one = RSB.min_width(v, f)
    monkeypatch.setattr(RSB, "WIDTH_ONE_PASS", 0)
    monkeypatch.setattr(RSB, "WIDTH_WORK", 10**12)
    assert RSB.min_width(v, f) == pytest.approx(one, rel=1e-12)
    monkeypatch.setattr(RSB, "WIDTH_WORK", 1)
    pv, pf = plate(0.002, 60)
    assert RSB.min_width(pv, pf) == pytest.approx(0.002, rel=1e-9)


def test_a_convex_mesh_above_the_cap_builds_with_its_mass_and_lands():
    """A 60,402-vertex sphere (r 4 cm, default convex): one collision shape of at most 64 vertices, a display-only
    mesh, the full mesh's mass (within 0.5 % of the solid sphere's: the hull would be lighter), and it comes to rest
    on the ground at its radius. ~40 s on a CPU."""
    v, f = uv_sphere(0.04, 152, 400)
    assert len(v) > RE.CONVEX_SOURCE_MAX
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_mesh("scan", vertices=v, faces=f, pos=(0.0, 0.0, 0.06), density=1000.0)
    sim = alloy.build(sc, device="cpu")
    m = sim.engine.model
    body = m.body_label.index("scan")
    shapes = np.flatnonzero(m.shape_body.numpy() == body)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    hull = [s for s in shapes if int(m.shape_flags.numpy()[s]) & collide]
    assert len(hull) == 1 and len(shapes) == 2
    assert float(m.body_mass.numpy()[body]) == pytest.approx(1000.0 * 4.0 / 3.0 * np.pi * 0.04**3, rel=5e-3)
    sim.step(50)
    assert float(sim.state(world=0)["scan.pos"][2]) == pytest.approx(0.04, abs=2e-3) and sim.check()["ok"]
