# SPDX-License-Identifier: Apache-2.0
"""Render textured parts with Alloy's path-traced camera (``alloy.camera.mesh_scene`` + ``Camera``), on the GPU.

One generic data-driven material (base colour, roughness, metallic, UV base-colour texture) serves every object, so
the shading module compiles once per process; each thumbnail then costs one render.

Example::

    r = CameraRenderer(size=512, spp=128)
    img = r.render(parts, view)          # (512, 512, 3) uint8
"""
from __future__ import annotations

import logging

import numpy as np

from framing import GROUND_RGB, View, ground_plane
from visuals import Part, bounds

log = logging.getLogger("thumbnails")

EXPOSURE_EV = -1.0      # the camera's studio light is set for scenes; one stop down keeps light objects off white


def textured_mesh(cls, part: Part):
    """``alloy.camera.TexturedMesh`` of a Part: per-corner points and UVs (bottom-left origin), sRGB colour."""
    tex = part.texture is not None and part.uv is not None
    return cls(part.name, part.points, part.tris, uv=part.uv if tex else None, texture=part.texture if tex else None,
               color=tuple(part.base_color), roughness=float(part.roughness), metallic=float(part.metallic))


class CameraRenderer:
    """Thumbnail renderer: object parts + a light ground plane under the camera's studio lighting."""

    def __init__(self, size: int = 512, spp: int = 128, device: str | None = None, denoise: str | None = None,
                 exposure: float = EXPOSURE_EV) -> None:
        from alloy import camera  # noqa: PLC0415 - needs Warp + a GPU for real speed

        self.camera, self.size, self.spp, self.denoise, self.exposure = camera, size, spp, denoise, float(exposure)
        self.TexturedMesh, self.mesh_scene = camera.TexturedMesh, camera.mesh_scene
        self.device = device or camera.default_device()

    def render(self, parts: list[Part], view: View) -> np.ndarray:
        """One (size, size, 3) uint8 image of ``parts`` seen from ``view``."""
        lo, hi = bounds(parts)
        gp, gt = ground_plane(lo, hi)
        ground = Part("ground", gp, gt, None, None, GROUND_RGB, roughness=0.9)
        meshes = [textured_mesh(self.TexturedMesh, p) for p in [*parts, ground]]
        scene = self.mesh_scene(meshes, device=self.device)
        cam = self.camera.Camera("thumb", look_at=view.look_at, width=self.size, height=self.size, fov=view.fov_deg,
                                 spp=self.spp, denoise=self.denoise)
        img = cam.render(scene, exposure=self.exposure).numpy()        # (worlds, H, W, 3) uint8, display view
        return np.asarray(img[0], np.uint8)
