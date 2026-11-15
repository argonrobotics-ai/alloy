# SPDX-License-Identifier: Apache-2.0
"""``add_mesh(..., hull_vertices=n)``: a convex mesh's collision hull capped at ``n`` vertices, the opt-out from the rule
that gives a hull as many vertices as it needs (up to 2,048) to stay within 0.1 mm of the mesh's own.

Laws (CPU, on the MuJoCo C route, solver asserted): a sphere of 3,122 vertices collides as a hull of more than 64
vertices by default and of exactly 64 with ``hull_vertices=64``; the value survives ``to_json`` / ``from_dict`` and
``save`` (the loaded card builds a hull of 128 vertices from 128), and is not written when unset; a value that is not
an int from 4 to 2,048, or one on a mesh that is not ``collision="convex"``, raises ``SceneError`` naming
``hull_vertices`` when it is added; a saved card edited to such a mesh is refused when the library reads it
(``CardError`` on ``shape.hull_vertices``), before any build.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

import alloy.api as alloy


def _sphere(radius: float = 0.04, n_lat: int = 40, n_lon: int = 78) -> tuple[np.ndarray, np.ndarray]:
    """A closed UV sphere: two poles and n_lat - 1 rings of n_lon vertices, faces wound outward."""
    rings = [(radius * np.sin(t) * np.cos(p), radius * np.sin(t) * np.sin(p), radius * np.cos(t))
             for t in np.linspace(0, np.pi, n_lat + 1)[1:-1] for p in np.linspace(0, 2 * np.pi, n_lon, endpoint=False)]
    v = np.array([(0, 0, radius), *rings, (0, 0, -radius)])
    ring = lambda i, j: 1 + i * n_lon + j % n_lon  # noqa: E731
    f = [(0, ring(0, j), ring(0, j + 1)) for j in range(n_lon)]
    for i in range(n_lat - 2):
        for j in range(n_lon):
            f += [(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)), (ring(i, j), ring(i + 1, j + 1), ring(i, j + 1))]
    last = len(v) - 1
    f += [(last, ring(n_lat - 2, j + 1), ring(n_lat - 2, j)) for j in range(n_lon)]
    return v, np.array(f)


def _scene(**kw) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    v, f = _sphere()
    scene.add_mesh("ball", vertices=v, faces=f, pos=(0, 0, 0.05), density=500.0, **kw)
    return scene


def _hull_vertices(scene: alloy.Scene) -> int:
    """The built ball's colliding hull: its vertex count in MuJoCo."""
    sim = alloy.build(scene, device="cpu")
    try:
        solvers = [s for s in sim.engine.solvers if hasattr(s, "mj_model")]
        assert [type(s).__name__ for s in solvers] == ["MjcCoreSolver"], solvers
        m = solvers[0].mj_model
        geom = next(i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                    and (m.geom_contype[i] or m.geom_conaffinity[i]))
        k = int(m.geom_dataid[geom])
        return int(m.mesh_graph[m.mesh_graphadr[k]])
    finally:
        sim.close()


@pytest.mark.compile_heavy
def test_hull_vertices_caps_the_built_hull(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    assert _hull_vertices(_scene()) > 64                       # the rule: within 0.1 mm of the sphere's hull
    assert _hull_vertices(_scene(hull_vertices=64)) == 64      # the opt-out: the earlier fast hull
    saved = _scene(hull_vertices=128).save(tmp_path / "saved")
    assert _hull_vertices(alloy.Scene.load(saved)) == 128      # the saved card keeps it


def test_hull_vertices_round_trips_and_is_not_written_unset(tmp_path):
    assert "hull_vertices" not in _scene().to_json()
    scene = _scene(hull_vertices=128)
    again = alloy.Scene.from_dict(json.loads(scene.to_json()))
    assert '"hull_vertices": 128' in again.to_json()
    scene.save(tmp_path / "saved")
    assert [p.name for p in (tmp_path / "saved").rglob("*.yaml") if "hull_vertices: 128" in p.read_text()]


@pytest.mark.parametrize(("value", "collision", "says"), [
    ("64", "convex", "int"), (True, "convex", "int"), (64.0, "convex", "int"), (3, "convex", "4 to 2048"),
    (4096, "convex", "4 to 2048"), (64, "sdf", "convex"), (64, "decompose", "convex")])
def test_a_bad_hull_vertices_raises_naming_it(value, collision, says):
    with pytest.raises(alloy.SceneError) as err:
        _scene(hull_vertices=value, collision=collision)
    assert err.value.field == "hull_vertices" and says in err.value.message


def test_a_card_edited_to_a_bad_hull_vertices_is_refused_when_read(tmp_path):
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.core.errors import CardError  # noqa: PLC0415

    _scene(hull_vertices=64).save(tmp_path / "saved")
    card = tmp_path / "saved" / "objects" / "ball.yaml"
    card.write_text(card.read_text().replace("collision: convex", "collision: sdf"))   # hand-edited: not convex
    with pytest.raises(CardError) as err:
        CardLibrary([tmp_path / "saved"]).object("ball")
    assert err.value.field == "shape.hull_vertices" and "convex" in err.value.message
