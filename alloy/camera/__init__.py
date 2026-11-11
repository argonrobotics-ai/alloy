# SPDX-License-Identifier: Apache-2.0
"""Alloy camera: a batched path-traced camera sensor that renders like Blender Cycles, on the GPU.

Quick look at a model (one line)::

    from alloy.camera import snapshot
    img = snapshot(model, state, path="snap.png")          # (480, 640, 3) uint8, auto-framed

A camera in the training loop, over every world::

    from alloy.camera import Camera
    cam = Camera("front", look_at=((1.2, -1.2, 0.9), (0, 0, 0.2)), width=128, height=128, fov=50, spp=16)
    for _ in range(steps):
        solver.step(state_0, state_1, control, contacts, dt)
        img = cam.render_torch(model, state_1)              # torch uint8 (worlds, 128, 128, 3) on the GPU

Blender-quality scenes (materials, area lights, HDRI world, lens) come from scene packs::

    scene = load_pack("c3_pack", frame=420, num_worlds=64)
    img = Camera("pack", width=128, height=128, spp=16).render(scene)   # the pack's own camera

``attach_to="hand"`` mounts the camera on a body (its pose is then in the body frame). Lights default to a studio
sun and sky (``Lighting.studio()``); ``camera_scene(model, lighting=...)`` changes them.
"""
from __future__ import annotations

import importlib

# Names resolve on first use (PEP 562), so subpackages such as alloy.camera.denoiser import without Warp.
_EXPORTS = {
    "Camera": "camera", "CameraOutput": "camera", "camera_scene": "camera", "default_device": "camera",
    "load_pack": "camera", "CameraError": "errors", "DenoiseSpec": "denoise", "image_extent": "pose",
    "look_at": "pose", "Lighting": "scene", "MeshGeom": "scene", "SceneData": "scene", "Sun": "scene",
    "TactileRenderer": "tactile", "TactileSensor": "tactile", "TexturedMesh": "meshes", "mesh_scene": "meshes",
    "model_scene": "meshes", "RenderSettings": "settings", "render": "still", "snapshot": "still",
    "to_display": "view", "write_png": "view",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'alloy.camera' has no attribute {name!r}; public names: {', '.join(__all__)}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
