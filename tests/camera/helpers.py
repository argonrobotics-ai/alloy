# SPDX-License-Identifier: Apache-2.0
"""Small scenes for the camera tests (CPU-sized)."""
from __future__ import annotations

import numpy as np

from alloy.camera.scene import IDENTITY_XF, Lighting, MeshGeom, SceneData


def quad(size: float = 20.0, z: float = 0.0) -> MeshGeom:
    """A square in the z plane, facing +z."""
    h = 0.5 * size
    pts = np.array([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]], np.float32)
    return MeshGeom("quad", pts, np.array([[0, 1, 2], [0, 2, 3]], np.int32))


def cube(half: float = 0.1) -> MeshGeom:
    """An axis-aligned cube centred at the origin, triangles counter-clockwise from outside."""
    p = np.array([[x, y, z] for z in (-half, half) for y in (-half, half) for x in (-half, half)], np.float32)
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = [t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))]
    return MeshGeom("cube", p, np.array(tris, np.int32))


def plane_scene(albedo: float = 0.8, lighting: Lighting | None = None) -> SceneData:
    """One large white-ish plane at z = 0 in one world."""
    return SceneData.single_world([quad()], [(albedo, albedo, albedo)], lighting)


def body_scene(worlds: int = 2, lighting: Lighting | None = None) -> SceneData:
    """A cube on body w in world w (no ground), so worlds can be told apart."""
    return SceneData([cube()], np.zeros(worlds), np.arange(worlds), np.arange(worlds),
                     np.tile(IDENTITY_XF, (worlds, 1)), np.tile([0.8, 0.3, 0.2], (worlds, 1)), worlds,
                     lighting=lighting or Lighting.studio())


class FakeState:
    """Stands in for a Newton state: only body_q is read."""

    def __init__(self, body_q: np.ndarray, device: str = "cpu"):
        import warp as wp
        self.body_q = wp.array(np.asarray(body_q, np.float32), dtype=wp.transform, device=device)
