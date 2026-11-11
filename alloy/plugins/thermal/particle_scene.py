# SPDX-License-Identifier: Apache-2.0
"""``ParticleThermal`` by object name: heat v2 on a scene built with the public API (``alloy.Scene``, ``build``).

The Scene API wraps it: ``add_liquid(..., temperature=K)``, ``scene.set_heater(name, temperature=K)``, then
``alloy.build`` calls ``for_scene``, and ``sim.heat(name)`` / ``sim.temperature(name)`` read it back
(docs/guide/heat.md). Use this module directly on an assembled scene with your own plug-in list.

``for_scene`` turns object names into the plug-in's template particle ranges and heater names into its heater lists;
``object_temperature`` and ``object_summary`` read one object's columns back out of the plug-in's [W, P] read-outs.

Example (warm water in a pan whose floor is held at 80 °C; docs/guide/heat.md)::

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0, 0, 0.02))
    scene.add_liquid("water", size=(0.07, 0.07, 0.02), pos=(0, 0, 0.0355))
    built = alloy.build(scene, device="cpu").built
    therm = for_scene(built, objects={"water": ("water", 293.15)}, heaters={"pan_floor": 353.15}, time_scale=60.0)
    heat_sim = alloy.plugins.simulation(built, [therm])
    for _ in range(100):
        heat_sim.step_period()
    print(object_summary(heat_sim, "water"))   # per world: mean, min, max [K], melted, done
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from alloy.plugins.errors import PluginError
from alloy.plugins.thermal.particle import ParticleThermal

log = logging.getLogger("alloy.plugins")

HeaterSpec = float | tuple[str, float]   # K (a thermostat), or ("temperature" | "power" | "body", value)


def for_scene(built: Any, objects: Mapping[str, tuple[str, float]], heaters: Mapping[str, HeaterSpec] | None = None,
              **config: Any) -> ParticleThermal:
    """A ``ParticleThermal`` for named particle objects of ``built`` (an assembled scene: ``sim.built``).

    ``objects``: object name -> (phase material from ``phase_materials.MATERIALS``, start temperature [K]).
    ``heaters``: static box name -> its temperature [K], or (mode, value) with mode "temperature" (K), "power" (W)
    or "body". Other keywords go to ``ParticleThermal.Config`` (``time_scale``, ``convection``, ``ambient``, ...).
    Without ``heater_gap``, a heated object on an MPM route sets it to its MPM entry's voxel (``mpm_voxel``).
    """
    if not objects:
        raise PluginError("for_scene: name at least one particle object (objects={'water': ('water', 293.15)})")
    ranges = tuple(_particle_range(built, name) for name in objects)
    modes, values = zip(*(_heater(v) for v in (heaters or {}).values())) if heaters else ((), ())
    voxel = mpm_voxel(built, objects)
    if "heater_gap" not in config and voxel is not None:
        config["heater_gap"] = voxel
        log.info("for_scene: heater_gap = %.3g m, the MPM voxel of the heated objects (MPM rests particles anywhere "
                 "within a voxel of a collider)", voxel)
    from alloy.plugins.thermal.buoyancy import BuoyantParticleThermal  # noqa: PLC0415 (gaps-now #8)
    cls = BuoyantParticleThermal if config.get("buoyancy") else ParticleThermal
    return cls(materials=tuple(m for m, _ in objects.values()), ranges=ranges,
               initial=tuple(float(t) for _, t in objects.values()), heaters=tuple(heaters or ()),
               heater_modes=tuple(modes), heater_values=tuple(values), **config)


def mpm_voxel(built: Any, names: Any) -> float | None:
    """The largest MPM voxel [m] among the named objects on an MPM route (their entry's ``voxel_size``, else the
    router's), or None when none is. MPM holds particles anywhere within a voxel of a collider: a paste beside a
    coupled robot rested 1.5 mm higher, past a quarter-spacing gap, and took no heat (WP13 combo sweep, F5)."""
    voxels = []
    for name in names:
        handle = built.handles.get(name)
        if handle is None or handle.decision.entry != "mpm":
            continue
        first = handle.template.particle.start
        entry = [e.solver_kwargs.get("voxel_size") for e in built.entries if e.capability == "mpm" and
                 first in e.particles]
        voxel = next((v for v in entry if v), None) or handle.decision.numbers.get("voxel")
        if voxel:
            voxels.append(float(voxel))
    return max(voxels) if voxels else None


def _particle_range(built: Any, name: str) -> tuple[int, int]:
    handle = built.handles.get(name)
    if handle is None:
        raise PluginError(f"for_scene: no object {name!r}; scene objects: {sorted(built.handles)}")
    r = handle.template.particle
    if len(r) == 0:
        raise PluginError(f"for_scene: {name!r} has no particles (route {handle.decision.route!r}); heat v2 runs on "
                          "SPH and MPM particles (add_liquid, add_pour, particle object cards)")
    return (r.start, r.stop)


def _heater(spec: HeaterSpec) -> tuple[str, float]:
    if isinstance(spec, tuple):
        mode, value = spec
        return str(mode), float(value)
    return "temperature", float(spec)


def columns(plugin: ParticleThermal, built: Any, name: str) -> slice:
    """The columns of object ``name`` in ``plugin``'s [W, P] read-outs (groups are stored in ``ranges`` order)."""
    want = _particle_range(built, name)
    start = 0
    for a, b in plugin.config.ranges:
        if (a, b) == want:
            return slice(start, start + b - a)
        if b < 0:
            break
        start += b - a
    raise PluginError(f"columns: {name!r} is not one of {plugin.label}'s particle groups {plugin.config.ranges}")


def _plugin(sim: Any) -> ParticleThermal:
    host = getattr(sim, "plugins", None)
    found = [p for p in getattr(host, "plugins", {}).values() if isinstance(p, ParticleThermal)]
    if len(found) != 1:
        raise PluginError(f"expected one particle_thermal plug-in on this simulation, found {len(found)}: build it "
                          "with alloy.plugins.simulation(built, [for_scene(built, ...)])")
    return found[0]


def object_temperature(sim: Any, name: str) -> np.ndarray:
    """[W, n] temperatures [K] of object ``name``'s particles (a host copy; spares and evaporated ones included)."""
    plug = _plugin(sim)
    return plug.temperature.numpy()[:, columns(plug, sim.built, name)].copy()


def object_summary(sim: Any, name: str) -> dict[str, np.ndarray]:
    """Per world [W], over object ``name``'s particles that take heat (active, not evaporated; ``particles`` counts
    them): mean, min and max temperature [K] (nan when there are none), melted (mean liquid fraction), done (mean
    doneness) and evaporated (share of the object's mass)."""
    plug = _plugin(sim)
    cols = columns(plug, sim.built, name)
    r = plug.read()
    live = r["taking_heat"][:, cols]
    n = live.sum(axis=1)

    def mean(a: np.ndarray) -> np.ndarray:
        return np.where(n > 0, (a[:, cols] * live).sum(axis=1) / np.maximum(n, 1), np.nan)

    t = r["temperature"][:, cols]
    low, high = np.where(live, t, np.inf).min(axis=1), np.where(live, t, -np.inf).max(axis=1)
    return {"particles": n, "mean_K": mean(r["temperature"]), "min_K": np.where(n > 0, low, np.nan),
            "max_K": np.where(n > 0, high, np.nan), "melted": mean(r["liquid"]), "done": mean(r["done"]),
            "evaporated": np.minimum(r["vapor"][:, cols], 1.0).mean(axis=1)}


__all__ = ["columns", "for_scene", "mpm_voxel", "object_summary", "object_temperature"]
