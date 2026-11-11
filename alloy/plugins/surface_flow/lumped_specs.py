# SPDX-License-Identifier: Apache-2.0
"""Specs of the lumped object films (design note §4.9): one well-mixed film per small body (WP9; WP15 reviews).

``Lumped`` names a body whose wetting is one number (water held), not a chart: tomatoes, a sponge, a pan's handle.
``Pores`` makes it a sponge (WP8's G5; Gym-Anything ``surface_rinsing/water/sponge.py`` and ``params.SPONGE``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.process_specs import Soil


def _positive(owner: str, obj: Any, names: tuple[str, ...], non_negative: tuple[str, ...] = ()) -> None:
    for name in names + non_negative:
        v = float(getattr(obj, name))
        if not math.isfinite(v) or (v <= 0.0 if name in names else v < 0.0):
            raise PluginError(f"{owner}: {name} must be finite and {'> 0' if name in names else '>= 0'}, got {v}")


@dataclass(frozen=True)
class Pores:
    """A sponge's pores. Free volume ``phi·area·height·(1 - ε)`` (GA: 0.9 of a 111 x 66 x 12 mm cellulose block, 79
    mL); the strain ε solves ``p/E = ε + 3ε³`` (the foam spring, capped at ``q_stop/height``) for the pressure ``p``
    its pad carries (``tool``: the Applicator the sponge also is, by index or body name; WP15: its stiffness E/H).
    Pressed on a film under capacity it takes it up at ``tau_cap`` (GA's estimate)."""

    phi: float = 0.9
    height: float = 0.012
    area: float = 0.111 * 0.066
    modulus: float = 5.0e4
    q_stop: float = 0.0105
    tau_cap: float = 0.1
    tool: int | str | None = None

    def __post_init__(self) -> None:
        _positive("pores", self, ("phi", "height", "area", "modulus", "q_stop", "tau_cap"))
        if self.phi > 1.0 or self.q_stop >= self.height:
            raise PluginError(f"pores: phi <= 1 and q_stop < height, got phi {self.phi}, q_stop {self.q_stop}")

    @property
    def free(self) -> float:
        """The free pore volume at rest [m³]."""
        return self.phi * self.area * self.height

    def strain(self, p: float | np.ndarray) -> float | np.ndarray:
        """ε for a pad pressure ``p`` [Pa] (host reference of the kernel's Newton solve; 0 for p <= 0)."""
        x = np.asarray(p, np.float64) / self.modulus
        if not np.all(np.isfinite(x)):
            raise PluginError(f"pores: the pad pressure must be finite, got {p}")
        x = np.maximum(x, 0.0)
        e = np.minimum(x, 1.0)
        for _ in range(30):
            e = e - (e + 3.0 * e ** 3 - x) / (1.0 + 9.0 * e ** 2)
        return np.clip(e, 0.0, self.q_stop / self.height)


@dataclass(frozen=True)
class Lumped:
    """A body with one well-mixed film. ``shapes``: its collision shapes by label (empty: all of the body's), which
    the supply's jets hit (DC5's LUMP kind); ``area`` [m²] its wetted area (None: the shapes' surface area, for
    boxes, spheres, capsules and cylinders); ``hold`` [m] the film a wet surface clings to (GA's "clings": a glass
    2 mL over its ~0.04 m², ~50 µm); ``soil`` its soil (a process_specs.Soil: its laws and ``initial`` load), None
    for none; ``pores`` makes it a sponge. In a pool (its centre of mass ``lumped_pool.DEPTH`` under a pool's level,
    DC5's pools) it fills toward capacity at ``k_fill`` of it per second and swaps water with the pool at
    ``k_mix0 + k_mix1·|v|`` per second (v its speed through the pool; GA's sponge estimates, ``pools.py``).
    ``built_water`` [m³]: the water the body's built mass already holds (a sponge built at its typical wet mass); the
    body carries the weight of its water beyond that, ``ρ·(water - built_water)·g`` (negative when it holds less),
    while the water's inertia stays the built one (WP15's ruling: no live mass changes). A body built dry falls fast
    once its water weighs on it (2 mL on a 2.1 g body: 19 m/s²; GA's 79 mL sponge of a few grams: 15-25 g), so the
    build warns when its full water outweighs its mass beyond ``built_water``."""

    body: str
    shapes: tuple[str, ...] = ()
    area: float | None = None
    hold: float = 5.0e-5
    soil: Soil | None = None
    pores: Pores | None = None
    k_fill: float = 2.0
    k_mix0: float = 0.2
    k_mix1: float = 5.0
    built_water: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shapes", tuple(self.shapes))
        if isinstance(self.soil, dict):
            object.__setattr__(self, "soil", Soil(**self.soil))
        if isinstance(self.pores, dict):
            object.__setattr__(self, "pores", Pores(**self.pores))
        _positive(f"lumped {self.body!r}", self, (), ("hold", "k_fill", "k_mix0", "k_mix1", "built_water"))
        if self.area is not None:
            _positive(f"lumped {self.body!r}", self, ("area",))


def lumped_spec(spec: Any) -> Lumped:
    """A ``Lumped`` from a spec or a dict (JSON, cards, ``dataclasses.asdict``)."""
    if isinstance(spec, Lumped):
        return spec
    if isinstance(spec, dict):
        return Lumped(**spec)
    raise PluginError(f"a lumped body is a Lumped or a dict, got {spec!r}")


def shape_area(model: newton.Model, shape: int) -> float:
    """Surface area [m²] of a primitive collision shape (box, sphere, capsule, cylinder)."""
    kind = int(model.shape_type.numpy()[shape])
    s = model.shape_scale.numpy()[shape].astype(np.float64)
    G = newton.GeoType
    if kind == int(G.BOX):
        return 8.0 * (s[0] * s[1] + s[1] * s[2] + s[2] * s[0])
    if kind == int(G.SPHERE):
        return 4.0 * math.pi * s[0] ** 2
    if kind == int(G.CAPSULE):
        return 2.0 * math.pi * s[0] * 2.0 * s[1] + 4.0 * math.pi * s[0] ** 2
    if kind == int(G.CYLINDER):
        return 2.0 * math.pi * s[0] * 2.0 * s[1] + 2.0 * math.pi * s[0] ** 2
    raise PluginError(f"shape {shape}: give the lumped body's area= (only primitive shapes are measured)")


__all__ = ["Lumped", "Pores", "lumped_spec", "shape_area"]
