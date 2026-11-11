# SPDX-License-Identifier: Apache-2.0
"""What a port's camera draws beyond its rigid shapes: its deformables, handed to the camera's own mesh path
(``alloy.camera.deform``: a sheet's triangles, one small sphere per particle), refit from the state every render.

A port spec names a ``display(sim)`` hook that returns a ``DeformLayout``, a ``LiquidLayout`` or None:
- :func:`sim_deformables`: an ``alloy.build`` port (the gift bags' paper) draws what its ``alloy.Sim`` lists (cloth,
  film and soft-body triangles; MPM grains);
- :func:`particle_liquid`: a harness port's SPH fluid (the porridge) as one smooth surface per world, rebuilt on the
  device each render (``alloy.camera.liquid_scene``), in a colour of its own;
- :func:`particle_spheres`: a harness port's particles as spheres of their radius.

Rendering only reads the state, between control periods (never inside a captured graph), so the simulation steps
the same with the camera on or off.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PORRIDGE = (0.86, 0.78, 0.60)          # oat porridge, linear albedo (an estimate)


def sim_deformables(sim: Any) -> Any:
    """The deformables of the ``alloy.Sim`` the port built (None for a harness port or a sim without any)."""
    from alloy.camera.deform import deform_objects  # noqa: PLC0415

    api = getattr(sim, "_port_api_sim", None)
    return deform_objects(api) if api is not None else None


def particle_spheres(name: str, color: tuple[float, float, float]) -> Callable[[Any], Any]:
    """A ``display`` hook drawing every particle of each world (one block per world) as a sphere of its radius."""

    def layout(sim: Any) -> Any:
        from alloy.camera.deform import DeformLayout, DeformObject, grain_mesh  # noqa: PLC0415

        m = sim.model
        worlds = int(getattr(m, "world_count", 1) or 1)
        n = int(m.particle_count) // worlds
        if n == 0:
            return None
        verts, tris = grain_mesh(m.particle_q.numpy()[:n], m.particle_radius.numpy()[:n])
        return DeformLayout(worlds, n, (DeformObject(name, "grains", 0, n, tris, verts, color),))

    return layout


class Liquid:
    """A ``LiquidLayout`` with the colour the fast tracer gives its surfaces (an opaque food, not a liquid preset)."""

    def __init__(self, layout: Any, color: tuple[float, float, float]) -> None:
        self.layout, self.color = layout, color


def particle_liquid(name: str, color: tuple[float, float, float]) -> Callable[[Any], Any]:
    """A ``display`` hook drawing each world's particles (one block per world, ``2 x radius`` apart) as one smooth
    surface (``alloy.camera.liquid_gpu``: anisotropic smoothing, implicit surface, marching cubes) of ``color``."""

    def layout(sim: Any) -> Any:
        from alloy.camera.liquid_gpu import LiquidObject  # noqa: PLC0415
        from alloy.camera.liquid_scene import LiquidLayout  # noqa: PLC0415

        m = sim.model
        worlds = int(getattr(m, "world_count", 1) or 1)
        n = int(m.particle_count) // worlds
        if n == 0:
            return None
        spacing = 2.0 * float(m.particle_radius.numpy()[:n].mean())
        return Liquid(LiquidLayout(worlds, n, (LiquidObject(name, 0, n, spacing),)), color)

    return layout


def deform_scene(sim: Any, layout: Any, rigid: bool = True) -> Any:
    """The camera's diffuse-tracer scene of ``sim``'s model with ``layout`` (a ``DeformLayout``, or a :class:`Liquid`):
    ``alloy.camera``'s own construction for an ``alloy.Sim`` with deformables or liquids. ``rigid=False`` parks the
    model's shapes 10 km away: the deformables alone (for masks)."""
    from alloy.camera.deform_gpu import DeformSet  # noqa: PLC0415
    from alloy.camera.from_newton import scene_from_model  # noqa: PLC0415
    from alloy.camera.liquid_scene import LiquidSet  # noqa: PLC0415
    from alloy.camera.tracer.gpu_scene import GpuScene  # noqa: PLC0415

    model = sim.model
    sd, dev = scene_from_model(model), str(model.device)
    if not rigid:
        sd.inst_body[:] = -1
        sd.inst_local[:, :3] = 1.0e4
    if isinstance(layout, Liquid):
        ls = LiquidSet(layout.layout, len(sd.meshes), dev, sim)
        sd = ls.scene_data(sd)
        n = layout.layout.worlds * len(layout.layout.objects)
        sd.inst_color[-n:] = layout.color                   # the liquid rows come last
        gs = GpuScene(sd, device=dev, liquid=ls.buffers())
        ls.attach(gs)
        return gs
    ds = DeformSet(layout, len(sd.meshes), dev, model)
    gs = GpuScene(ds.scene_data(sd), device=dev, points=ds.points())
    ds.attach(gs)
    return gs


__all__ = ["PORRIDGE", "Liquid", "deform_scene", "particle_liquid", "particle_spheres", "sim_deformables"]
