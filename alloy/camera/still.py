# SPDX-License-Identifier: Apache-2.0
"""One-line quick snapshot of a model (world 0): auto-framed unless a view is given, written as a PNG if asked.

Example::

    from alloy.camera import snapshot
    img = snapshot(model, state, path="snap.png")              # (480, 640, 3) uint8 numpy, auto-framed
    img = snapshot(model, look_at=((2, -2, 1.5), (0, 0, 0.3)), width=256, height=256, spp=64)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import Camera, camera_scene
from .tracer.renderer import Renderer
from .view import write_png


def snapshot(model, state=None, path: str | Path | None = None, look_at=None, pose=None, width: int = 640,
             height: int = 480, fov: float = 50.0, spp: int = 64, world: int = 0, device: str | None = None,
             look=None, quality: str = "auto", items="auto", exposure: float = 0.0) -> np.ndarray:
    """Render world ``world`` of ``model`` once and return an (H, W, 3) uint8 image (also saved to ``path``).
    ``look`` / ``quality="full"``: Blender materials and lights, shown through the look's view (AgX by default) and
    exposure; ``items``: scene item looks (see :func:`camera_scene`; an ``alloy.Sim`` as ``model`` brings its own).
    ``exposure``: extra stops. A warning is logged when more than 20 % of the pixels clip to white."""
    sc = camera_scene(model, device=device, look=look, quality=quality, items=items)
    cam = Camera("snapshot", pose=pose, look_at=look_at, width=width, height=height, fov=fov, spp=spp)
    if hasattr(sc, "ps") or type(sc).__name__ == "PbrScene":
        from .pbr import PbrRenderer
        r = PbrRenderer(sc, cam.width, cam.height, None, worlds=[world])
    else:
        r = Renderer(sc, cam.width, cam.height, cam.settings, worlds=[world])
    cam._bound = (sc, r)  # one image for the one world, not one per world
    cam._resolve_attach(sc)
    img = cam.render(sc, state, exposure=exposure).numpy()[0]
    if path is not None:
        write_png(path, img)
    return img


def render(sim, camera=None, world: int = 0, width: int = 640, height: int = 480, liquids=True, **kw) -> np.ndarray:
    """Images of an ``alloy.Sim`` (or anything with ``.engine.model`` / ``.engine.state``, or a newton.Model).

    ``camera=None``: a framed snapshot of ``world`` as an (H, W, 3) uint8 numpy array (``kw`` go to
    :func:`snapshot`, e.g. ``spp=`` (64 by default, so ``spp=64`` repeats the default image), ``look_at=``, ``path=``). With a :class:`Camera`: its (worlds, H, W, 3) batch.
    Scene items with their own look (``alloy.scene.variants.vary``'s texture swaps and tints) render with it, with the
    full-material tracer (``item_looks``; ``items=None`` leaves them out). Cloth, soft bodies and MPM materials are
    drawn from the sim's state (``alloy.camera.deform``), and so are the sim's SPH liquids (``alloy.camera.liquid_scene``:
    a dielectric with absorption on the full-material tracer, which snapshots of a liquid use unless ``quality="fast"``,
    a tinted colour on the fast one). ``liquids=False`` leaves them out of a snapshot (of a sim without deformables);
    a dict draws them as one-off snapshot surfaces with those fields of ``liquid.WATER``.

    Example: ``img = alloy.camera.render(sim, width=320, height=240)``.
    """
    from .item_looks import items_look
    from .tracer.renderer import image_worlds
    if camera is None and isinstance(getattr(sim, "worlds", None), int):
        image_worlds([world], sim.worlds)  # before any per-world read (liquid meshes), as sim.state(world=) checks
    items = kw.pop("items", "auto")
    items = items_look(sim) if items == "auto" else items  # scene variants: each item's own look (swap, tint)
    engine = getattr(sim, "engine", None)
    model = getattr(engine, "model", None) or getattr(sim, "model", None) or sim
    raw = model
    state = getattr(engine, "state", None) if engine is not None else getattr(sim, "state", None)
    state = state() if callable(state) else state
    live = _has_deform(sim) or _has_liquid(sim)
    if camera is None:
        if kw.get("output", "uint8") != "uint8":
            from .errors import CameraError
            raise CameraError(f"output={kw['output']!r} without a camera",
                              "pass a Camera: sim.render(alloy.Camera('front', look_at=(eye, target)), output=...)")
        kw.pop("output", None)
        if isinstance(liquids, dict) and kw.get("quality", "auto") != "fast" and _has_liquid(sim):
            from .liquid import liquid_meshes
            from .meshes import model_scene
            water = liquid_meshes(sim, world, material=liquids)
            model = model_scene(model, water, look=kw.pop("look", None), device=kw.pop("device", None), items=items)
            live = _has_deform(sim)
        elif liquids is False and not _has_deform(sim):
            live = False
        elif _has_liquid(sim) and kw.get("quality", "auto") == "auto":
            kw["quality"] = "full"  # snapshots draw liquids as liquids (the fast tracer tints them)
        plain = model is raw and not _has_liquid(sim)  # the sim itself draws the same, plus its scans' textures
        return snapshot(sim if live or plain else model, state, width=width, height=height, world=world,
                        items=items, **kw)
    from .aov import as_host
    return as_host(camera.render(sim, state, output=kw.get("output", "uint8")))


def _has_deform(sim) -> bool:
    """True for an ``alloy.Sim`` with cloth, soft bodies or MPM materials (drawn from its state)."""
    from .deform import has_deformables
    return has_deformables(sim)


def _has_liquid(sim) -> bool:
    """True for an ``alloy.Sim`` with an SPH liquid."""
    from .liquid_scene import has_liquids
    return has_liquids(sim)
