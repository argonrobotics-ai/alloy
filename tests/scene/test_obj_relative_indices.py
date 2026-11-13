# SPDX-License-Identifier: Apache-2.0
"""A Wavefront OBJ with relative (negative) face indices builds (AUDIT S3-022, Gemini 0.1.2): the card reader and
validation accepted ``f -8 -5 -6``, then the assembler's own OBJ reader took the indices as absolute and every
collision mode raised a raw ValueError ("indices contains negative index -9"). The assembler now reads OBJ with the
card layer's reader (``alloy.cards.geometry.read_obj``): the same mesh in relative and absolute indices builds the
same body, in every collision mode that needs no extra."""

from __future__ import annotations

import numpy as np
import pytest

import alloy

CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
FACES = [(1, 4, 3), (1, 3, 2), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]       # 1-based, wound outward


def cube_obj(path, relative: bool) -> None:
    lines = [f"v {0.05 * x} {0.05 * y} {0.05 * z}" for x, y, z in CORNERS]
    lines += ["f " + " ".join(str(i - 9 if relative else i) for i in f) for f in FACES]   # -8 .. -1 after 8 v
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("collision", ["convex", "mesh", "sdf"])
def test_relative_indices_build_the_same_body(tmp_path, collision):
    built = {}
    for relative in (False, True):
        path = tmp_path / f"cube_{relative}.obj"
        cube_obj(path, relative)
        sc = alloy.Scene(dt=0.01)
        sc.add_ground()
        sc.add_mesh("cube", path, pos=(0.0, 0.0, 0.08), collision=collision, density=500.0)
        sim = alloy.build(sc, device="cpu")
        m = sim.engine.model
        built[relative] = (float(m.body_mass.numpy()[0]), sim.state(world=0)["cube.pos"].copy())
    assert built[True][0] == pytest.approx(500.0 * 0.1**3, rel=1e-6)
    assert built[True][0] == built[False][0] and np.array_equal(built[True][1], built[False][1])
