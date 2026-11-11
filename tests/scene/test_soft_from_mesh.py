# SPDX-License-Identifier: Apache-2.0
"""SCR OSS-9-5: a soft card whose shape is a mesh builds fTetWild's tets (CPU).

Before the fix ``_tet_mesh`` reshaped the surface triangles Newton's ``remesh_ftetwild`` returns into "tets": a
reshape error, or nonsense tets that Newton dropped as inverted and a particle-radius ReadbackError. The radius
readback recomputed from the Model's float32 world positions is 3.6e-6 off at the ball's placement here (rtol 1e-6);
it now reads the cached mesh.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("pytetwild")
trimesh = pytest.importorskip("trimesh")

import alloy  # noqa: E402
from alloy.scene.emitters import deformable as D  # noqa: E402

R = 0.03
EDGE = {"edge_length_fac": D.TETWILD_EDGE_FAC}


@pytest.fixture
def tet_cache(tmp_path, monkeypatch):
    """A fresh in-process and on-disk tet cache."""
    monkeypatch.setenv("ALLOY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "_TETS", {})
    return tmp_path / "cache" / "tets"


def _ball() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=3, radius=R)


def _ball_card(root) -> trimesh.Trimesh:
    """A card root with a silicone ball whose shape is a mesh (silicone_40a routes SOFT_VBD)."""
    m = _ball()
    (root / "meshes").mkdir(parents=True)
    (root / "objects").mkdir()
    obj = root / "meshes" / "ball.obj"
    m.export(obj)
    sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    (root / "objects" / "silicone_ball_test.yaml").write_text(
        "schema: alloy.object/1\nname: silicone_ball_test\nshape:\n  mesh:\n    path: meshes/ball.obj\n"
        f"    sha256: {sha}\nmaterial: silicone_40a\n")
    return m


def _signed_volumes(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, c, d = (v[t[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def test_tetrahedralize_gives_oriented_tets_and_repeats(tmp_path, tet_cache, monkeypatch):
    m = _ball()
    monkeypatch.chdir(tmp_path)
    v, t = D.tetrahedralize(m.vertices, m.faces, EDGE)
    assert t.ndim == 2 and t.shape[1] == 4 and t.shape[0] > 100
    vol = _signed_volumes(v, t)
    assert (vol > 0).all()                                    # Newton's orientation: none dropped as inverted
    assert vol.sum() == pytest.approx(m.volume, rel=0.01)
    assert np.unique(t).size == len(v)                        # every vertex is in a tet
    assert not list(tmp_path.glob("__tracked_surface*"))      # fTetWild's scratch file stays out of the cwd
    assert len(list(tet_cache.glob("*.npz"))) == 1
    monkeypatch.setattr(D, "_TETS", {})                       # a new process: the disk cache gives the same tets
    v2, t2 = D.tetrahedralize(m.vertices, m.faces, EDGE)
    assert np.array_equal(v, v2) and np.array_equal(t, t2)


def test_soft_mesh_card_builds_and_rests(tmp_path, tet_cache):
    m = _ball_card(tmp_path / "cards")
    scene = alloy.Scene(dt=0.01, roots=[tmp_path / "cards"])
    scene.add_ground()
    scene.add_object("ball", "silicone_ball_test", pos=(3.7, -2.1, R + 0.002))   # far out: float32 rounding
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["ball"]["route"] == "soft_vbd"
    model = sim.engine.model
    t = model.tet_indices.numpy()
    (cached,) = tet_cache.glob("*.npz")
    with np.load(cached) as data:
        assert np.array_equal(t, data["tets"])                # every fTetWild tet emitted, none dropped as inverted
    assert model.particle_mass.numpy().sum() == pytest.approx(1150.0 * m.volume, rel=0.01)   # silicone_40a density
    traj = sim.run(0.2, record=["ball"])
    pts = traj.world(0)["ball.points"]
    assert sim.check()["ok"]
    assert pts[-1][:, 2].min() > -2e-3                       # rests on the ground, not through it
    assert (_signed_volumes(pts[-1], t) > 0).all()


class _ShiftingTetwild:
    """A stand-in for pytetwild whose every run gives different tets, as fTetWild's do."""

    def __init__(self):
        self.calls = 0

    def tetrahedralize(self, v, f, **kw):
        self.calls += 1
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64) * 0.01 * self.calls
        return verts, np.array([[0, 2, 1, 3]], np.int32)               # negative, as fTetWild's


def test_first_cache_writer_wins(tet_cache):
    m, tw = _ball(), _ShiftingTetwild()
    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)
    key = D._tet_key(v, f, EDGE, "test")
    v1, t1 = D._run_tetwild(tw, v, f, EDGE, key)
    v2, t2 = D._run_tetwild(tw, v, f, EDGE, key)     # a concurrent first build: it finds the key written, uses that
    assert tw.calls == 2 and np.array_equal(v1, v2) and np.array_equal(t1, t2)
    assert t1.tolist() == [[0, 1, 2, 3]] and (_signed_volumes(v1, t1) > 0).all()
    assert [p.name for p in tet_cache.iterdir()] == [f"{key}.npz"]          # no temp files left


def test_tiny_edge_length_raises_before_tetwild(tet_cache):
    from alloy.core.errors import AssemblyError

    box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    with pytest.raises(AssemblyError, match="edge_length"):
        D.tetrahedralize(box.vertices, box.faces, {"edge_length_abs": 1e-3})
