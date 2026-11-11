# SPDX-License-Identifier: Apache-2.0
"""Vision-based tactile sensors (GelSight / DIGIT style): what the camera behind the gel sees, for every world.

A sensor on a finger, rendered with the physics (images like ``alloy.camera.Camera``)::

    from alloy.camera.tactile import TactileSensor
    sensor = TactileSensor("left_tip", attach_to="left_finger", pose=((0, 0, 0.012), (0, 0, 0, 1)), preset="gelsight")
    img = sensor.render(model, state)              # wp.array uint8 (worlds, 240, 320, 3)
    out = sensor.render(model, state, output="all")   # + press map [mm], gel surface, marker centres, slip state

Any press field to images, no simulation (e.g. an engine's gel penetration map, worlds x H x W in mm)::

    from alloy.camera.tactile import TactileRenderer
    imgs = TactileRenderer("digit", width=160, num_worlds=64).render(press_mm)

Presets: "gelsight" and "digit", from the MIT-licensed Taxim calibrations (``list_presets()``). Images are what the
camera behind the gel sees: sensor frame +Z out of the gel, +X image right, +Y image down. ``Augment.strong()`` adds
per-world lighting, colour, noise and gel-wear randomisation for sim-to-real training.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "TactileSensor": "sensor", "TactileRenderer": "renderer", "TactileOutput": "renderer", "Augment": "augment",
    "TactilePreset": "presets", "MarkerSpec": "presets", "load_preset": "presets", "list_presets": "presets",
    "MarkerMotion": "markers",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'alloy.camera.tactile' has no attribute {name!r}; public names: {', '.join(__all__)}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
