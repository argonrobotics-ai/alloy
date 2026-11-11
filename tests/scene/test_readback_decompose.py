# SPDX-License-Identifier: Apache-2.0
"""Readback of a collision="decompose" mesh (SCR-OSS-10, WP2 review): the body mass is checked against density x the
card mesh's volume (the emitter's rule in both branches), and the inertia source is listed: the card mesh when it is
watertight, the hulls when it is open."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from alloy.cards.types import MeshSource, ObjectCard, ResolvedObject
from alloy.core.ids import digest_of
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.scene.readback_checks import decompose_mass_source
from alloy.testing import route as T
from alloy.testing import scene as S

EDGE = 0.05
FACES = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _cube_text(faces, header: str = "") -> str:
    v = [(x, y, z) for x in (0.0, EDGE) for y in (0.0, EDGE) for z in (0.0, EDGE)]
    lines = ([header] if header else []) + ["o hull_0"] + [f"v {x!r} {y!r} {z!r}" for x, y, z in v]
    return "\n".join(lines + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]) + "\n"


def _cube(tmp_path, open_mesh: bool) -> ResolvedObject:
    path = tmp_path / "cube.obj"
    path.write_text(_cube_text(FACES[:-2] if open_mesh else FACES).replace("o hull_0\n", ""))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "cube.hulls.obj").write_text(_cube_text(FACES, f"# source_sha256: {sha}"))
    card = ObjectCard("cube", MeshSource(str(path), sha, collision="decompose", max_parts=2), "wood")
    mat = T.material("wood", **T.WOOD)
    return ResolvedObject(card, mat, {}, digest_of((card, mat)), (EDGE, EDGE, EDGE), {}, ())


class _Lib(S.TestLibrary):
    __test__ = False

    def resolve(self, card, placement=None):
        if isinstance(card.shape, MeshSource):                   # the test library has no mesh extents
            return self.cube
        return super().resolve(card, placement)


@pytest.mark.parametrize(("open_mesh", "source"), [(False, "card mesh"), (True, "hulls")])
def test_decompose_body_mass_is_checked_and_inertia_source_listed(tmp_path, open_mesh, source):
    cube = _cube(tmp_path, open_mesh)
    volume, got = decompose_mass_source(cube.card.shape)
    assert got == source
    if not open_mesh:
        assert volume == pytest.approx(EDGE**3, rel=1e-5)
    objs = [S.table(), cube]
    card, _ = S.scene(objs, [S.place("table"), S.place("cube", z=0.011)], worlds=2)
    lib = _Lib(materials={o.material.name: o.material for o in objs}, objects={o.card.name: o.card for o in objs},
               pairs={})
    lib.cube = cube
    built = SceneAssembler(lib, Router(entries={Route.RIGID: "mujoco"})).assemble(card, "cpu")   # readback inside
    notes = dict(built.report.unsupported)
    assert source in notes["cube.inertia"]
    assert "cube.material.density" not in notes                  # checked now, not listed
    rho = float(cube.material.density.value)
    mass = built.model.body_mass.numpy()
    assert np.allclose(mass[mass > 0], rho * volume, rtol=1e-6)
