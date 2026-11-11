# SPDX-License-Identifier: Apache-2.0
"""Soft boxes' mirrored Kuhn split (alloy.scene.emitters.deformable.box_tets; coordinator GO 2026-11-08 20:58Z): the
mesh is conforming, fills the box and is mirror-symmetric, and a resting soft box stays where it is put. With the
uniform split (one diagonal in every cell) a resting 30 mm silicone box crept 3.0 mm along that diagonal in 0.4 s."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from alloy.scene.emitters.deformable import box_mesh, box_tets

FULL = np.array([0.03, 0.02, 0.04])


def _volumes(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = (verts[tets[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


@pytest.mark.parametrize("dims", [(4, 4, 4), (3, 5, 2), (9, 9, 9)])
def test_the_split_is_conforming_and_fills_the_box(dims):
    verts, tets = box_tets(FULL, np.array(dims))
    vol = _volumes(verts, tets)
    assert tets.shape[0] == 6 * int(np.prod(dims)) and np.all(vol > 0.0)
    assert vol.sum() == pytest.approx(float(np.prod(FULL)), rel=1e-12, abs=0)
    faces = collections.Counter(tuple(sorted(t[list(f)])) for t in tets for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3),
                                                                                     (1, 2, 3)))
    assert max(faces.values()) == 2                                # no face is shared by three tets
    nx, ny, nz = dims
    boundary = 2 * 2 * (nx * ny + ny * nz + nx * nz)               # two triangles per boundary cell face
    assert sum(1 for n in faces.values() if n == 1) == boundary    # every interior face matches its neighbour's


def test_the_split_is_mirror_symmetric():
    """With an even cell count along an axis, reflecting the box across its mid-plane maps the tet set onto itself
    (the uniform split maps onto its mirror image instead, and that is the bias that sheared a resting box)."""
    verts, tets = box_tets(FULL, np.array((4, 4, 4)))
    key = {tuple(np.round(v, 12)): i for i, v in enumerate(verts)}
    for axis in range(3):
        flip = verts.copy()
        flip[:, axis] *= -1.0
        image = np.array([key[tuple(np.round(v, 12))] for v in flip])
        mapped = {tuple(sorted(image[t])) for t in tets}
        assert mapped == {tuple(sorted(t)) for t in tets}


def _dims(size, budget) -> list[int]:
    from types import SimpleNamespace

    verts, tets = box_mesh(SimpleNamespace(size=size), budget)
    assert len(tets) <= budget
    return [len(np.unique(np.round(verts[:, k], 12))) - 1 for k in range(3)]


def test_box_mesh_rounds_cell_counts_to_even():
    """A body slides along an axis with an odd cell count: a 30 mm silicone box crept 93 µm in 0.4 s on 9 x 8 x 8 cells
    (along its 9-cell axis), 13 µm per 0.4 s on 9 x 9 x 9 after landing, and 0 on 8 x 8 x 9 (vertical odd) or 8 x 8 x 8.
    So odd counts above 3 round down to even. A 3 becomes 4 when the budget allows, else stays 3, so a thin pad keeps 3
    layers (test_dc_show_pad). 1 and 2 stay."""
    assert _dims((0.015,) * 3, 5000) == [8, 8, 8]                      # 9 x 9 x 9 by the budget
    assert _dims((0.025, 0.025, 0.006), 5000) == [14, 14, 4]           # 15 x 15 x 3: 4 fits (4,704 tets)
    assert _dims((0.025, 0.025, 0.006), 4000) == [14, 14, 3]           # 14 x 14 x 3: 4 would not fit
    for size, budget in (((0.05, 0.02, 0.002), 5000), ((0.03, 0.03, 0.03), 40000), ((0.04, 0.003, 0.003), 500)):
        assert all(d <= 3 or d % 2 == 0 for d in _dims(size, budget)), size


@pytest.mark.law
@pytest.mark.timeout(600)
def test_a_resting_soft_box_stays_put():
    """The GO's law: a 30 mm silicone box set on the ground moves at most 0.01 mm sideways over 0.4 s (CPU; measured
    under 0.5 µm; the uniform split crept 3 mm)."""
    import alloy

    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.03,) * 3, pos=(0.0, 0.0, 0.015), material="silicone_40a")
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["block"]["route"] == "soft_vbd"
    start = sim.state(world=0)["block.points"][:, :2].mean(axis=0)
    sim.step(40)
    drift = sim.state(world=0)["block.points"][:, :2].mean(axis=0) - start
    assert float(np.hypot(*drift)) <= 1.0e-5, drift
    assert sim.check()["ok"]
