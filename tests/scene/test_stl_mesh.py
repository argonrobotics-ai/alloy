# SPDX-License-Identifier: Apache-2.0
"""STL meshes build (agent friction F6, CPU).

``alloy.Scene.add_mesh`` and the card checks accept STL, but assembly refused it ("unsupported format .stl (use .obj
or .npz)"). The emitter now reads binary and ASCII STL with the card layer's reader (coincident corners merged), so a
cube from STL builds and lands exactly as the same cube from OBJ.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import alloy
from alloy.cards.types import MeshSource
from alloy.scene.emitters.common import load_mesh

H = 0.02          # [m] half edge of the cube
V = np.array([[x, y, z] for x in (-H, H) for y in (-H, H) for z in (-H, H)])
F = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
              (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])


def _binary_stl(path):
    rows = [struct.pack("<3f", 0.0, 0.0, 0.0) + struct.pack("<9f", *V[f].reshape(-1)) + b"\0\0" for f in F]
    path.write_bytes(b"cube".ljust(80, b" ") + struct.pack("<I", len(F)) + b"".join(rows))
    return path


def _ascii_stl(path):
    body = ["solid cube"]
    for f in F:
        corners = [f"  vertex {x!r} {y!r} {z!r}" for x, y, z in V[f].tolist()]
        body += ["facet normal 0 0 0", " outer loop", *corners, " endloop", "endfacet"]
    path.write_text("\n".join(body + ["endsolid cube"]) + "\n")
    return path


def _obj(path):
    lines = [f"v {x!r} {y!r} {z!r}" for x, y, z in V.tolist()] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("write", [_binary_stl, _ascii_stl], ids=["binary", "ascii"])
def test_load_mesh_reads_stl_as_the_same_mesh_as_obj(tmp_path, write):
    stl = load_mesh(MeshSource(str(write(tmp_path / "cube.stl")), ""))
    obj = load_mesh(MeshSource(str(_obj(tmp_path / "cube.obj")), ""))
    assert stl.vertices.shape == (8, 3) and stl.indices.shape == (12, 3)          # corners merged: watertight
    assert {tuple(v) for v in np.round(stl.vertices, 9)} == {tuple(v) for v in np.round(obj.vertices, 9)}


def test_an_stl_cube_builds_and_lands_like_the_obj_cube(tmp_path):
    def land(path) -> tuple[float, float]:
        s = alloy.Scene(dt=0.01)
        s.add_ground()
        s.add_mesh("cube", path, pos=(0.0, 0.0, 0.05), collision="convex", density=500.0)
        sim = alloy.build(s, device="cpu")
        sim.run(0.5)
        assert sim.check()["ok"], sim.check()["problems"]
        return float(sim.engine.model.body_mass.numpy()[0]), float(sim.state(world=0)["cube.pos"][2])

    m_stl, z_stl = land(_binary_stl(tmp_path / "cube.stl"))
    m_obj, z_obj = land(_obj(tmp_path / "cube.obj"))
    assert m_stl == pytest.approx(500.0 * (2 * H) ** 3, rel=1e-5) and m_stl == pytest.approx(m_obj, rel=1e-6)
    assert z_stl == pytest.approx(H, abs=1e-3) and z_stl == pytest.approx(z_obj, abs=1e-6)
