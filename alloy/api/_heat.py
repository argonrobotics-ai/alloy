# SPDX-License-Identifier: Apache-2.0
"""Heat in the Scene API: heaters, start temperatures, and temperatures read back from a ``Sim``.

The physics is the heat v2 plug-in (``alloy.plugins.thermal.particle.ParticleThermal``; guide
``docs/notes/heat-v2-guide.md``): every particle of a liquid (SPH) or particle object (MPM) gets a temperature,
conducts heat to its neighbours, takes heat from heaters and loses it to the air; materials melt, boil and cook.
``Scene`` records what is hot in ``scene.thermal``; ``alloy.build`` maps names with
``particle_scene.for_scene`` and installs the plug-in in the API's own simulation, so ``run``, ``record``,
``state`` and ``reset`` keep working. Temperatures are in kelvin (20 °C = 293.15 K).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from alloy.api.errors import BuildError, SceneError, SimulationError

KELVIN = 273.15                 # 0 °C in K
COLDEST = 150.0                 # K: below this a value is almost surely °C (no phase material is fluid there)
MANAGED = frozenset({"materials", "ranges", "initial", "heaters", "heater_modes", "heater_values"})
_SETTINGS_FIX = "change the setting the message names (scene.set_heat / set_heater / set_temperature)"


def empty() -> dict[str, Any]:
    """A scene's empty heat record. Example: ``scene.thermal = empty()``."""
    return {"objects": {}, "heaters": {}, "config": {}}


def _kelvin(value: Any, field: str) -> float:
    t = float(value)
    if not math.isfinite(t) or t <= 0.0:
        raise SceneError(f"{field} must be a temperature in kelvin > 0, got {value!r}", "pass kelvin: 20 °C = 293.15",
                         field=field)
    if t < COLDEST:
        raise SceneError(f"{field}={t:g} K is colder than {COLDEST:g} K; temperatures are in kelvin",
                         f"for {t:g} °C pass {t + KELVIN:.2f}", field=field)
    return t


def _phase_material(name: str, field: str = "heat_material") -> str:
    from alloy.plugins.thermal.phase_materials import MATERIALS  # noqa: PLC0415

    if name not in MATERIALS:
        usable = sorted(m for m in MATERIALS if not m.startswith("law_"))
        raise SceneError(f"no heat data for material {name!r}", f"pass heat_material= one of {usable}",
                         field=field)
    return name


class HeatMixin:
    """``Scene`` methods for heat (mixed into ``alloy.Scene``)."""

    thermal: dict[str, Any]
    items: dict[str, dict[str, Any]]
    roots: list[str]

    def add_heater(self, name: str, size: Any, *, pos: Any, temperature: float | None = None,
                   power: float | None = None, **kw: Any) -> str:
        """A static box (full ``size`` (x, y, z) m, centred at ``pos``) that heats the liquid or particles touching
        it: held at ``temperature`` K (a thermostat), or putting ``power`` W into them, shared by contact area.
        Other keywords as ``add_box`` (``material``, orientation). A rigid object on it is not heated (heat v2
        warms particles only).

        Example: ``scene.add_heater("hob", size=(0.2, 0.2, 0.02), pos=(0, 0, 0.01), temperature=353.15)``."""
        spec = heater_spec(temperature, power)                               # checked before the box is added
        name = self.add_box(name, size, pos=pos, static=True, **kw)        # type: ignore[attr-defined]
        self.thermal["heaters"][name] = list(spec)
        return name

    def set_heater(self, name: str, *, temperature: float | None = None, power: float | None = None) -> None:
        """Make a static box a heater, for example an open box's floor: ``temperature`` K or ``power`` W (one of
        the two). ``sim.set_heater`` changes the value while running.

        Example: ``scene.add_open_box("pan", ...); scene.set_heater("pan_floor", temperature=353.15)``."""
        item = self._item(name)                                              # type: ignore[attr-defined]
        if item["kind"] != "box" or not item["static"]:
            raise SceneError(f"{name!r} is not a static box; heaters are static boxes",
                             "use scene.add_heater(...), or name a static box such as an open box's floor "
                             "('<name>_floor')", field="name")
        self.thermal["heaters"][name] = list(heater_spec(temperature, power))

    def set_temperature(self, name: str, temperature: float, *, heat_material: str | None = None) -> None:
        """Start ``name`` (a liquid, pour or particle object card) at ``temperature`` K and let it take heat.
        ``heat_material``: the heat data to use (melting, boiling, conductivity) when it differs from the object's
        own material; ``alloy.plugins.thermal.phase_materials.MATERIALS`` lists them (water, dark_chocolate,
        beef_lean, porridge_oat).

        Example: ``scene.set_temperature("water", 293.15)``."""
        item = self._item(name)                                              # type: ignore[attr-defined]
        if item["kind"] not in ("liquid", "card"):
            raise SceneError(f"{name!r} is a {item['kind']}; heat v2 warms liquids and particle objects only",
                             "give the temperature to a liquid (add_liquid / add_pour) or a particle object card; "
                             "rigid bodies take heat through the Thermal plug-in (alloy.plugins.thermal)", field="name")
        own = item.get("material") or self._card_material(item)
        mat = _phase_material(heat_material or own, "heat_material" if heat_material else "material")
        self.thermal["objects"][name] = {"temperature": _kelvin(temperature, "temperature"), "material": mat}

    def set_heat(self, *, time_scale: float | None = None, ambient: float | None = None,
                 convection: float | None = None, **config: Any) -> None:
        """Settings for all heat in the scene: ``time_scale`` (heat runs that many times faster than motion: with
        ``dt`` 0.01 and 60, each step heats for 0.6 s), ``ambient`` air K, ``convection`` W/(m²·K) to the air (0:
        off), and other ``ParticleThermal.Config`` fields (``heater_conductance``, ``heater_gap``, ...).

        Example: ``scene.set_heat(time_scale=60, ambient=293.15, convection=10)``."""
        from alloy.plugins.thermal.particle import ParticleThermal  # noqa: PLC0415

        fields = set(ParticleThermal.Config.__dataclass_fields__) - MANAGED
        bad = sorted(set(config) - fields)
        if bad:
            raise SceneError(f"unknown heat setting(s) {bad}", f"use {sorted(fields)}", field=bad[0])
        named = {"time_scale": time_scale, "convection": convection,
                 "ambient": None if ambient is None else _kelvin(ambient, "ambient")}
        self.thermal["config"].update({k: v for k, v in {**named, **config}.items() if v is not None})

    def _card_material(self, item: dict[str, Any]) -> str:
        import yaml  # noqa: PLC0415

        from alloy.api import catalog  # noqa: PLC0415

        path = catalog.object_card(item["card_name"], self.roots)     # a Hub object is fetched to read it
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return str((data or {}).get("material", ""))


def heater_spec(temperature: float | None, power: float | None) -> tuple[str, float]:
    """``("temperature", K)`` or ``("power", W)``; exactly one given. Example: ``heater_spec(353.15, None)``."""
    if (temperature is None) == (power is None):
        raise SceneError("a heater needs exactly one of temperature= (K) or power= (W)",
                         "pass temperature=353.15 (80 °C) or power=500", field="temperature")
    if temperature is not None:
        return "temperature", _kelvin(temperature, "temperature")
    w = float(power)                                                         # type: ignore[arg-type]
    if not math.isfinite(w) or w < 0.0:
        raise SceneError(f"power must be finite and >= 0 W, got {power!r}", "pass power=500 (W)", field="power")
    return "power", w


def plugin_for(scene: Any) -> Callable[[Any], Any] | None:
    """A function of the built scene that makes its heat plug-in, or None when nothing in it is hot.

    Example: ``make = plugin_for(scene); therm = make(built) if make else None``."""
    th = scene.thermal
    if not th["objects"]:
        if th["heaters"]:
            raise SceneError(f"heaters {sorted(th['heaters'])} but nothing to heat", "give a liquid a start "
                             "temperature: add_liquid(..., temperature=293.15) or scene.set_temperature(name, K)",
                             field="temperature")
        return None
    objects = {n: (o["material"], o["temperature"]) for n, o in th["objects"].items()}
    heaters = {n: (v[0], float(v[1])) for n, v in th["heaters"].items()}
    config = dict(th["config"])

    def make(built: Any) -> Any:
        from alloy.plugins.errors import PluginError  # noqa: PLC0415
        from alloy.plugins.thermal.particle_scene import for_scene  # noqa: PLC0415

        try:
            return for_scene(built, objects, heaters, **config)
        except (PluginError, ValueError) as exc:
            raise BuildError(f"heat: {exc}", _SETTINGS_FIX, field="thermal") from exc

    return make


def thermal_of(engine: Any, *, need: bool = True) -> Any:
    """The heat plug-in installed in ``engine`` (``Sim.engine``), or None when ``need`` is False and there is none.

    Example: ``thermal_of(sim.engine).read()["heater_area_m2"]``."""
    from alloy.plugins.thermal.particle import ParticleThermal  # noqa: PLC0415

    host = getattr(engine, "plugins", None)
    found = [p for p in getattr(host, "plugins", {}).values() if isinstance(p, ParticleThermal)]
    if found:
        return found[0]
    if need:
        raise SimulationError("this simulation has no heat", "give an object a temperature before building: "
                              "add_liquid(..., temperature=293.15) or scene.set_temperature(name, K)",
                              field="temperature")
    return None


def is_heated(engine: Any, name: str) -> bool:
    """True when ``name``'s particles carry temperatures. Example: ``is_heated(sim.engine, "water")``."""
    plug = thermal_of(engine, need=False)
    return plug is not None and name in getattr(engine, "_alloy_heated", ())


def temperature(engine: Any, name: str) -> Any:
    """(worlds, particles) K of ``name``, in the order of ``"<name>.points"``. Example: ``temperature(e, "water")``."""
    from alloy.plugins.thermal.particle_scene import object_temperature  # noqa: PLC0415

    _check_heated(engine, name)
    return object_temperature(engine, name)


def summary(engine: Any, name: str) -> dict[str, Any]:
    """Per world: ``mean_K``, ``min_K``, ``max_K``, ``melted``, ``done``, ``evaporated``, ``particles``.

    Example: ``summary(sim.engine, "water")["mean_K"]``."""
    from alloy.plugins.thermal.particle_scene import object_summary  # noqa: PLC0415

    _check_heated(engine, name)
    return object_summary(engine, name)


def set_heater(engine: Any, name: str, temperature: Any = None, power: Any = None) -> None:
    """Change a heater's value now: a number for every world or a (worlds,) array. Its mode stays as built.

    Example: ``set_heater(sim.engine, "pan_floor", temperature=373.15)``."""
    import numpy as np  # noqa: PLC0415 - keeps `alloy.Scene` free of numpy until a Sim exists

    plug = thermal_of(engine)
    heaters = list(plug.config.heaters)
    if name not in heaters:
        raise SimulationError(f"{name!r} is not a heater", f"use one of {heaters}", field="name")
    k, want = heaters.index(name), plug.config.heater_modes[heaters.index(name)]
    if (temperature is None) == (power is None) or (want == "temperature") != (temperature is not None):
        raise SimulationError(f"heater {name!r} was built as a {want} heater", f"pass {want}= (one value, or one "
                              "per world); the mode is fixed at build", field=want)
    value = np.asarray(temperature if temperature is not None else power, dtype=np.float64)
    low = COLDEST if temperature is not None else 0.0
    if not np.all(np.isfinite(value)) or value.min() < low:
        raise SimulationError(f"heater {name!r}: {want} must be finite and >= {low:g}, got {value}",
                              "temperatures are in kelvin (80 °C = 353.15), power in W", field=want)
    values = plug.inputs()["heater_values"].numpy().astype(np.float64)
    values[:, k] = np.broadcast_to(value, values.shape[:1])
    plug.set_input("heater_values", values)


def _check_heated(engine: Any, name: str) -> None:
    thermal_of(engine)
    heated = sorted(getattr(engine, "_alloy_heated", ()))
    if name not in heated:
        raise SimulationError(f"{name!r} has no temperature", f"use one of {heated}, or give it one before building "
                              "(scene.set_temperature)", field="name")
