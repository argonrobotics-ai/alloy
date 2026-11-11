# SPDX-License-Identifier: Apache-2.0
"""The surface-flow supply's user-facing specs (``Nozzle``, ``Drain``, ``Pool``) and a nozzle's ray table; ``Supply``
(supply.py) compiles them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.plugins.errors import PluginError

PATTERNS = {"jet": 0, "fan": 1, "cone": 2, "spray": 3}
STAGE = ":stage"                  # drip_of of the supply's stage-drip nozzles (supply_stage)
WATER = (998.0, 1.0e-6, 0.0728)                     # ρ [kg/m³], ν [m²/s], σ [N/m] at 20 °C (CRC Handbook)
JUMP_COEFF = 0.289                                  # Bhagat, Jha, Linden & Wilson (2018), J. Fluid Mech. 851 R5


@dataclass(frozen=True)
class Nozzle:
    """A nozzle on a body (``body`` None: the world) emitting the outflow of network edge ``edge`` (a node to the
    atmosphere: Q = -discharge). It points along ``axis`` from ``pos`` (body frame); exit speed Q/``exit_area``.

    ``pattern``: "jet" (solid; enters the film at its jump radius), "fan" (``rays`` across ``angle`` in the plane of
    ``axis`` and ``fan``), "cone" (``rays`` over the cone of full ``angle``), "spray" (Gaussian over the cone, σ =
    angle/4; ``efficiency`` lands, the rest is mist; rays that hit nothing are overspray)."""

    name: str
    edge: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    fan: tuple[float, float, float] = (1.0, 0.0, 0.0)
    body: str | None = None
    pattern: str = "jet"
    exit_area: float = 1.0e-4                       # m²
    angle: float = 0.0                              # rad, full angle
    rays: int = 1
    efficiency: float = 1.0
    segments: int = 4
    reach: float = 1.5                              # m along a straight-down path
    density: float = WATER[0]
    viscosity: float = WATER[1]                     # kinematic, m²/s
    surface_tension: float = WATER[2]
    jump_coeff: float = JUMP_COEFF
    h_min: float = 1.0e-3                           # m, floor of the post-jump depth
    froude: float = 0.3                             # post-jump Froude number (sets the ring depth's floor)
    u_max: float = 0.5                              # m/s, cap of the post-jump speed (film u_cap 1.5)
    rim_of: str | None = None                       # a cavity pool's tank: pour from its rim's lowest point
    species: Any = None                             # {film species: kg/m³}; else its node's (network species by name)
    drip_of: str | None = None                      # set by the supply: a holed vessel's drips (supply_holes)

    def __post_init__(self) -> None:
        if self.pattern not in PATTERNS:
            raise PluginError(f"nozzle {self.name!r}: pattern must be one of {sorted(PATTERNS)}")
        if self.pattern == "jet" and self.rays != 1 and self.drip_of is None:
            raise PluginError(f"nozzle {self.name!r}: a jet is one ray")
        if min(self.exit_area, self.reach, self.density, self.viscosity, self.surface_tension, self.rays,
               self.segments, self.u_max, self.h_min, self.froude) <= 0 or not 0.0 < self.efficiency <= 1.0:
            raise PluginError(f"nozzle {self.name!r}: sizes, fluid and counts must be > 0, efficiency in (0, 1]")
        if self.pattern != "jet" and not 0.0 < self.angle < math.pi:
            raise PluginError(f"nozzle {self.name!r}: a {self.pattern} needs 0 < angle < pi")
        if (self.rim_of is not None or self.drip_of is not None) and self.pattern != "jet":
            raise PluginError(f"nozzle {self.name!r}: a rim pour or a drip is a jet")


@dataclass(frozen=True)
class Drain:
    """Film cells of chart ``chart`` that drain to tank ``to`` (None: away). Either the cells within ``radius`` of
    ``center`` (chart frame), free (``cd_area`` inf) or an orifice of ``cd_area`` [m²], or the chart's open edges
    (``open_edges``: their outflux, nothing more). Input ``drain_open`` [W, drains] scales it (0: a stopper)."""

    name: str
    chart: int
    to: str | None = None
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    open_edges: bool = False
    cd_area: float = math.inf


@dataclass(frozen=True)
class Pool:
    """Where tank ``tank``'s free surface is: over the footprint ``floor`` ± ``half`` (x, y; z up) on ``body``.

    ``flood``: the chart cells under its water are flooded (their film joins the pool; a cell leaving the water keeps
    a withdrawal film of the pool's liquid, the tank node's density: Landau-Levich on a face drawn along itself, at
    most ``withdraw_max`` thick, 0: none; a face within ``puddle_tilt`` of level keeps the static puddle of the chart's
    contact angle; ``supply_pool.py``).

    ``cavity``: the pool is the inside of a vessel on ``body`` (``supply_cavity.py``): the inner wall's radius r at
    rising heights z, as (r, z) nodes from the floor's edge (the floor is the disc at the first z) up to the rim,
    about the body's z axis through ``floor`` (z up from it). Its surface lies normal to gravity at any tilt.
    ``spill`` names a ``source`` edge out of the tank that carries what stands above the rim's lowest point (a weir
    over the wetted rim; a ``Nozzle(rim_of=tank)`` on it pours it as a stream). ``weir`` names the pool it exchanges
    water with over the rim when the vessel is in it (a dunked glass fills over its rim; air trapped above the rim's
    highest point keeps an inverted one dry); several (up to 4, e.g. wash, rinse and sanitize bowls) are tried in
    order, the first whose water holds the rim's lowest point. ``holes`` are rings of holes (a colander's) that drain
    its water into ``holes_to``, or (None) drip as ``drip_rays`` jets onto whatever is below (``supply_holes``).
    Cells of a cavity flood only from its own water. A cavity catches jets and sprays over its round mouth (``half``
    is not used), never its own rim nozzle's pour or drips. ``carry``: its water rides on its body as a weight force
    (``supply_float``), or with ``"mass"`` as mass, centre and inertia (``supply_carry``)."""

    tank: str
    floor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half: tuple[float, float] = (0.1, 0.1)
    body: str | None = None
    flood: bool = False
    viscosity: float = WATER[1]                     # kinematic, m²/s
    surface_tension: float = WATER[2]
    withdraw_max: float = 1.0e-3                    # m
    puddle_tilt: float = 10.0                       # degrees (ESTIMATE: contact-angle hysteresis is not modelled)
    cavity: tuple[tuple[float, float], ...] | None = None
    slices: int = 48                                # cavity volume quadrature along its axis
    cavity_tol: float = 1.0e-3                      # m: a cell this far past the inner wall, facing in, is inside
    spill: str | None = None
    weir: str | tuple[str, ...] | None = None         # one pool's tank, or up to 4 tried in order
    weir_cd: float = 0.62                           # sharp-crested weir discharge coefficient (Rehbock)
    holes: tuple[tuple[float, ...], ...] = ()      # rings (z, cd_area[, r]) draining the cavity (supply_holes)
    holes_to: str | None = None                     # the tank they drain into (None: they drip, as jets)
    drip_rays: int = 12                             # jets the drips leave as (holes_to None)
    carry: str = "weight"                           # a cavity's water on its body: "weight" (a force) or "mass"

    def __post_init__(self) -> None:
        if self.weir is not None and not isinstance(self.weir, str):
            object.__setattr__(self, "weir", tuple(self.weir))
            if not self.weir:
                raise PluginError(f"pool of {self.tank!r}: weir names a pool's tank or up to 4 (None: no weir)")
        object.__setattr__(self, "holes", tuple(tuple(float(x) for x in h) for h in (self.holes or ())))
        if min(*self.half, self.viscosity, self.surface_tension) <= 0 or self.withdraw_max < 0 or \
                not 0.0 <= self.puddle_tilt < 90.0:
            raise PluginError(f"pool of {self.tank!r}: half extents, viscosity and surface tension must be > 0, "
                              f"withdraw_max >= 0, puddle_tilt in [0, 90)")
        if self.carry not in ("weight", "mass"):
            raise PluginError(f"pool of {self.tank!r}: carry is 'weight' or 'mass', got {self.carry!r}")
        if self.carry == "mass" and (self.cavity is None or self.body is None):
            raise PluginError(f"pool of {self.tank!r}: carry='mass' needs a cavity on a body (supply_carry)")
        if self.cavity is None:
            if self.spill is not None or self.weir is not None:
                raise PluginError(f"pool of {self.tank!r}: spill and weir need a cavity")
            return
        z = [float(n[1]) for n in self.cavity]
        if len(self.cavity) < 2 or any(b <= a for a, b in zip(z[:-1], z[1:])) or min(n[0] for n in self.cavity) < 0:
            raise PluginError(f"pool of {self.tank!r}: a cavity is >= 2 (r >= 0, z) nodes with z rising to the rim")
        if self.slices < 4 or self.cavity_tol < 0 or not 0.0 < self.weir_cd <= 1.0 or not 1 <= self.drip_rays <= 64:
            raise PluginError(f"pool of {self.tank!r}: slices >= 4, cavity_tol >= 0, weir_cd in (0, 1], drip_rays in "
                              f"[1, 64]")


def coerce(cls: type, items: Any) -> tuple[Any, ...]:
    return tuple(x if isinstance(x, cls) else cls(**x) for x in (items or ()))


def unit(v: Any) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


OVERLAP = 1.5     # a cone's ray footprint radius in ray spacings: hat footprints that only tile the cone leave stripes
FAN_OVERLAP = 2.0  # along a fan's line a whole number of spacings: 1.5 left 13 % stripes, 2 leaves 1.6 % (WP15)


def ray_table(noz: Nozzle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-frame unit directions [R, 3], flow fractions [R] and footprint half-angles [R] of a nozzle's rays. A
    footprint spans ``OVERLAP`` ray spacings on a cone (WP15: tiling hats striped a 64-ray coat by 29 %), and
    ``FAN_OVERLAP`` along a fan's line; each ray's flow is normalised over its own footprint (``supply_spray``), so the
    overlap moves no water."""
    axis = unit(noz.axis)
    if noz.pattern == "jet":                                             # one ray (a drip nozzle: one per drip)
        n = int(noz.rays)
        return np.tile(axis, (n, 1)), np.full(n, 1.0 / n), np.zeros(n)
    side = unit(np.asarray(noz.fan) - (np.asarray(noz.fan) @ axis) * axis)
    n = int(noz.rays)
    if noz.pattern == "fan":
        th = -0.5 * noz.angle + (np.arange(n) + 0.5) * noz.angle / n
        dirs = np.cos(th)[:, None] * axis + np.sin(th)[:, None] * side
        return dirs, np.full(n, 1.0 / n), np.full(n, FAN_OVERLAP * noz.angle / n)
    third = np.cross(axis, side)
    cos_cap = math.cos(0.5 * noz.angle)
    z = 1.0 - (1.0 - cos_cap) * (np.arange(n) + 0.5) / n                 # uniform per solid angle (sunflower)
    phi = np.arange(n) * math.pi * (3.0 - math.sqrt(5.0))
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    dirs = z[:, None] * axis + (s * np.cos(phi))[:, None] * side + (s * np.sin(phi))[:, None] * third
    fp = OVERLAP * math.sqrt(2.0 * math.pi * (1.0 - cos_cap) / n)      # the rays' spacing: √(cap solid angle / n)
    wt = np.ones(n)
    if noz.pattern == "spray":
        sigma = 0.25 * noz.angle
        wt = np.exp(-0.5 * (np.arccos(np.clip(z, -1.0, 1.0)) / sigma) ** 2)
    return dirs, wt / wt.sum(), np.full(n, fp)


__all__ = ["FAN_OVERLAP", "JUMP_COEFF", "OVERLAP", "PATTERNS", "STAGE", "WATER", "Drain", "Nozzle", "Pool", "coerce",
           "ray_table", "unit"]
