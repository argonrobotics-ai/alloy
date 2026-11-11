# SPDX-License-Identifier: Apache-2.0
"""Liquid surfaces (SPH particles -> water mesh) and the see-through data-driven material, on the CPU."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import Camera, CameraError
from alloy.camera.liquid import liquid_meshes, surface_mesh
from alloy.camera.meshes import TexturedMesh, mesh_scene
from alloy.camera.scene import Lighting
from tests.camera.glass_packs import box_mesh, slab_brightness

R = 0.002  # particle radius: spacing 4 mm
SKY = 0.5


def _ball(centre=(0.1, 0.2, 0.3), radius=0.02) -> np.ndarray:
    g = np.arange(-radius, radius + 1e-6, 2 * R)
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return p[np.linalg.norm(p, axis=1) <= radius] + np.asarray(centre)


def test_surface_of_a_ball_of_particles_is_closed_outward_and_keeps_the_volume():
    pts = _ball()
    v, t = surface_mesh(pts, R)
    p = v[t]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (np.einsum("ij,ij->i", n, p.mean(1) - (0.1, 0.2, 0.3)) > 0).all()  # wound outward
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()  # closed: every edge shared by 2 triangles
    vol = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0
    assert vol == pytest.approx(len(pts) * (2 * R) ** 3, rel=0.06)


def test_surface_mesh_edge_cases():
    v, t = surface_mesh(np.zeros((0, 3)), R)
    assert v.shape == (0, 3) and t.shape == (0, 3)
    with pytest.raises(CameraError, match="radius"):
        surface_mesh(_ball(), 0.0)


def _water(**kw) -> TexturedMesh:
    v, tris, _ = box_mesh((1.0, 1.0, 0.05))
    return TexturedMesh("water", v, tris, color=(1, 1, 1), roughness=0.0, transmission=1.0, ior=1.33, **kw)


def _centre(obj: TexturedMesh) -> np.ndarray:
    scene = mesh_scene([obj], lighting=Lighting([], (SKY, SKY, SKY)), device="cpu")
    cam = Camera("c", look_at=((0.001, 0.0, 0.6), (0.0, 0.0, 0.0)), width=16, height=16, spp=128)
    return cam.render(scene, output="linear").numpy()[0][6:10, 6:10].mean(axis=(0, 1))


def test_clear_water_mesh_is_invisible_and_absorbing_water_follows_beer_lambert():
    assert np.allclose(_centre(_water()), SKY, atol=2e-3)
    c = _centre(_water(absorption=(5.0, 5.0, 5.0)))
    assert np.allclose(c, SKY * slab_brightness(1.33, 5.0, 0.1), rtol=0.01)


class _FakeSim:
    """The part of alloy.Sim that liquid_meshes reads."""

    def routes(self):
        return {"water": {"route": "liquid_sph"}, "cup": {"route": "rigid_mujoco"}}

    def names(self):
        return {"particles": ["water"], "bodies": ["cup"]}

    def state(self, names, world=0):
        pts = _ball()
        active = np.ones(len(pts), bool)
        active[:3] = False  # not poured yet
        pts[:3] = 50.0
        return {"water.points": pts, "water.active": active}

    def particle_volume(self, name):
        return (2 * R) ** 3


def test_liquid_meshes_of_a_sim_skip_unpoured_particles():
    (m,) = liquid_meshes(_FakeSim())
    assert m.name == "water" and m.transmission == 1.0 and m.ior == pytest.approx(1.33)
    assert np.abs(np.asarray(m.points) - (0.1, 0.2, 0.3)).max() < 0.03  # the parked particles are not drawn
