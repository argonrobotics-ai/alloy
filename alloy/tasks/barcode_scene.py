# SPDX-License-Identifier: Apache-2.0
"""Printed labels on simulated bodies for ``alloy.camera``: a model's full-material scene (its look and lights) plus
one textured mesh per label riding on the labelled body in every world (``alloy.camera.model_scene``, OSS-4), so the
labels move with the bodies at every render.

The model path of the camera has no texture coordinates on its shapes (a box is just a box), so a label is its own
thin mesh: a flat card on a face, or a strip wrapped round a cylinder (a bottle or a can), 0.2 mm proud of the
surface, with the label image (``barcode_symbols.label_image``) as a generic material's base-colour texture (the
same material ``alloy.camera.mesh_scene`` gives textured meshes).

Example::

    scene = labelled_scene(model, [Decal("item", label_image(Symbol("590123412345"), (0.04, 0.03)), (0.04, 0.03),
                                         center=(0.04, 0.0, 0.0), normal=(1, 0, 0))])
    rgb = Camera("scanner", look_at=((0.2, 0, 0.1), (0, 0, 0.1)), width=320, height=240).render(scene, state)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROUD = 2.0e-4                    # m: a label stands this far off the surface it is on
ARC_SEGMENTS = 24                 # columns of a wrapped label


@dataclass(frozen=True)
class Decal:
    """A label on body ``body`` (a body label, or its last path part; "" = fixed in the world, the same place in
    every world, so only for models whose worlds share one frame, as ``replicate`` without spacing): ``image`` (H, W, 3)
    sRGB uint8, row 0 = top; ``size`` (width, height) in m; centred at ``center`` in the body (or world) frame,
    facing ``normal``, its top towards ``up``. ``radius`` > 0 wraps it round a cylinder of that radius whose axis
    runs along ``up`` through ``center - radius * normal`` (the label's middle touches ``center``)."""

    body: str
    image: np.ndarray
    size: tuple[float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0
    roughness: float = 0.45

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(normal, right, up) unit vectors; right = up x normal (the label reads left to right from outside)."""
        n = np.asarray(self.normal, np.float64)
        n /= np.linalg.norm(n)
        v = np.asarray(self.up, np.float64)
        v = v - n * (v @ n)
        if np.linalg.norm(v) < 1e-9:
            raise ValueError(f"decal on {self.body!r}: up {self.up} is parallel to normal {self.normal}")
        v /= np.linalg.norm(v)
        return n, np.cross(v, n), v

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(points (V, 3) in the body frame, tris (T, 3), uv per corner (T, 3, 2), origin bottom-left)."""
        n, u, v = self.frame()
        w, h = float(self.size[0]), float(self.size[1])
        cols = ARC_SEGMENTS if self.radius > 0.0 else 1
        s = np.linspace(0.0, 1.0, cols + 1)
        c = np.asarray(self.center, np.float64)
        if self.radius > 0.0:
            ang = (s - 0.5) * w / self.radius
            r = self.radius + PROUD
            axis = c - self.radius * n
            ring = axis + r * (np.cos(ang)[:, None] * n + np.sin(ang)[:, None] * u)
        else:
            ring = c + PROUD * n + ((s - 0.5) * w)[:, None] * u
        pts = np.concatenate([ring - 0.5 * h * v, ring + 0.5 * h * v]).astype(np.float32)
        uvs = np.concatenate([np.stack([s, np.zeros_like(s)], 1), np.stack([s, np.ones_like(s)], 1)])
        k = np.arange(cols)
        top = k + cols + 1
        tris = np.concatenate([np.stack([k, k + 1, top + 1], 1), np.stack([k, top + 1, top], 1)]).astype(np.int32)
        return pts, tris, uvs.astype(np.float32)[tris]


def scanner_lighting(toward_light=(1.0, 0.0, 1.0), irradiance: float = 4.0):
    """A sun from ``toward_light`` (world direction from the scene to the light) and no sky: the scanner's own
    illumination outshines the room, and a sun's direct light carries no Monte Carlo noise (``barcode_imager``)."""
    from alloy.camera.scene import Lighting, Sun

    d = np.asarray(toward_light, np.float64)
    return Lighting([Sun(d / np.linalg.norm(d), (irradiance,) * 3, 0.0)], (0.0, 0.0, 0.0), 0.0)


def labelled_scene(model, decals: list[Decal], look=None, lighting=None, device: str | None = None):
    """``model``'s full-material camera scene (``look`` / ``lighting`` as ``alloy.camera``) with ``decals`` added,
    each moving with its body in every world (``alloy.camera.model_scene``): for ``Camera.render(scene, state)``."""
    from alloy.camera import TexturedMesh, model_scene

    meshes = []
    for i, d in enumerate(decals):
        pts, tris, uv = d.mesh()
        meshes.append(TexturedMesh(f"label{i}", pts, tris, uv=uv, texture=np.ascontiguousarray(d.image),
                                   color=(1.0, 1.0, 1.0), roughness=d.roughness, body=d.body))
    return model_scene(model, meshes, look=look, lighting=lighting, device=device)


__all__ = ["Decal", "labelled_scene", "scanner_lighting"]
