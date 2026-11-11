# SPDX-License-Identifier: Apache-2.0
"""ModelBuilder helpers for SPH scenes: liquid blocks on the 2r lattice, spare pools and box tanks.

Every helper registers the ``sph:*`` attributes first, so scenes need no other setup. Particle mass is
``rho0 / S`` (config.particle_mass), which makes an interior lattice particle have rho = rho0 exactly.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import ParticleFlags

from .config import (BOUNDARY_SDF, PHASE_HELD, PHASE_LIQUID, PHASE_SPARE, DFSPHConfig, particle_mass,
                     register_custom_attributes)

log = logging.getLogger("alloy.sph")

Vec3 = tuple[float, float, float]


def lattice_points(lo: Vec3, counts: tuple[int, int, int], spacing: float) -> np.ndarray:
    """Cell centres of an nx x ny x nz lattice whose cells (side ``spacing``) start at corner ``lo``."""
    nx, ny, nz = counts
    ax = [lo[k] + (np.arange(n) + 0.5) * spacing for k, n in enumerate((nx, ny, nz))]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def _attrs(n: int, phase: int, rho0: float, mu: float, sigma: float = 0.0, adhesion: float = 0.0) -> dict[str, list]:
    out = {"sph:phase": [phase] * n, "sph:rest_density": [rho0] * n, "sph:viscosity": [mu] * n}
    if sigma != 0.0 or adhesion != 0.0:  # tension-free scenes keep the attribute defaults (same model)
        out.update({"sph:surface_tension": [sigma] * n, "sph:adhesion": [adhesion] * n})
    return out


def add_fluid_points(builder: newton.ModelBuilder, cfg: DFSPHConfig, points: np.ndarray,
                     rest_density: float = 1000.0, viscosity: float = 1.0e-3,
                     velocity: Vec3 = (0.0, 0.0, 0.0), surface_tension: float = 0.0, adhesion: float = 0.0) -> range:
    """Active liquid particles at ``points`` (N, 3) in the builder's current world; returns their index range.
    ``surface_tension`` [N/m] and ``adhesion`` (W_adh / sigma) act with ``surface_tension="akinci2013"``."""
    register_custom_attributes(builder)
    n = len(points)
    if n == 0:
        return range(builder.particle_count, builder.particle_count)
    m = particle_mass(cfg, rest_density)
    first = builder.particle_count
    builder.add_particles(
        pos=[tuple(map(float, p)) for p in points], vel=[tuple(velocity)] * n, mass=[m] * n,
        radius=[cfg.particle_radius] * n, flags=[int(ParticleFlags.ACTIVE)] * n,
        custom_attributes=_attrs(n, PHASE_LIQUID, rest_density, viscosity, surface_tension, adhesion),
    )
    return range(first, first + n)


def add_fluid_block(builder: newton.ModelBuilder, cfg: DFSPHConfig, lo: Vec3, counts: tuple[int, int, int],
                    rest_density: float = 1000.0, viscosity: float = 1.0e-3,
                    velocity: Vec3 = (0.0, 0.0, 0.0)) -> range:
    """A box of liquid on the 2r lattice with its lower corner at ``lo`` (first layer centred r above lo)."""
    return add_fluid_points(builder, cfg, lattice_points(lo, counts, cfg.spacing), rest_density, viscosity, velocity)


def add_spares(builder: newton.ModelBuilder, cfg: DFSPHConfig, count: int, rest_density: float = 1000.0,
               viscosity: float = 1.0e-3, park: Vec3 = (0.0, 0.0, 0.0), phase: int = PHASE_SPARE) -> range:
    """Inactive spare particles (ACTIVE cleared) parked at ``park``: phase 0 for emitters, or ``PHASE_HELD`` for an
    SPH port's reservoir (``add_held_spares``)."""
    register_custom_attributes(builder)
    first = builder.particle_count
    if count <= 0:
        return range(first, first)
    m = particle_mass(cfg, rest_density)
    builder.add_particles(
        pos=[tuple(park)] * count, vel=[(0.0, 0.0, 0.0)] * count, mass=[m] * count,
        radius=[cfg.particle_radius] * count, flags=[0] * count,
        custom_attributes=_attrs(count, phase, rest_density, viscosity),
    )
    return range(first, first + count)


def add_held_spares(builder: newton.ModelBuilder, cfg: DFSPHConfig, count: int, rest_density: float = 1000.0,
                    viscosity: float = 1.0e-3, park: Vec3 = (0.0, 0.0, 0.0)) -> range:
    """An SPH port's reservoir (ports.py): ``count`` inactive particles that only ``SPHPort.emit`` turns into liquid
    (a film shedding drops, design surface-flow §4.4); emitters never take them."""
    return add_spares(builder, cfg, count, rest_density, viscosity, park, phase=PHASE_HELD)


def _box(builder: newton.ModelBuilder, center: Vec3, half: Vec3, body: int, mode: int,
         density: float | None = None, contact_angle: float | None = None, seal: bool = False) -> int:
    cfg = newton.ModelBuilder.ShapeConfig() if density is None else newton.ModelBuilder.ShapeConfig(density=density)
    attrs = {"sph:shape_boundary": mode}
    if contact_angle is not None:
        attrs["sph:shape_contact_angle"] = float(contact_angle)
    if seal:
        attrs["sph:shape_seal"] = 1
    s = builder.add_shape_box(body, xform=wp.transform(center, wp.quat_identity()), hx=half[0], hy=half[1],
                              hz=half[2], cfg=cfg, custom_attributes=attrs)
    return s


def wall_box(builder: newton.ModelBuilder, center: Vec3, half: Vec3, body: int = -1,
             contact_angle: float | None = None, seal: bool = False) -> int:
    """One SPH wall: a box shape (center, half extents) the liquid sees through its SDF; ``contact_angle`` [rad]
    sets its sph:shape_contact_angle (surface_tension="csf"; None: a free contact line); ``seal`` its
    sph:shape_seal (Config.ambient_pressure, ambient.py)."""
    register_custom_attributes(builder)
    return _box(builder, center, half, body, BOUNDARY_SDF, contact_angle=contact_angle, seal=seal)


def add_box_tank(builder: newton.ModelBuilder, inner_lo: Vec3, inner_hi: Vec3, wall: float, *,
                 open_top: bool = True, body: int = -1, density: float | None = None,
                 floor: bool = True) -> dict[str, int]:
    """An axis-aligned tank of box walls (thickness ``wall``) around the inner box; returns shape ids by name.

    Walls tile the solid without overlap (floor spans the full footprint, x walls span the full y extent, y walls
    sit between them), so each solid point belongs to exactly one shape. ``body`` >= 0 attaches the walls to a body
    (frame of ``inner_lo``/``inner_hi``); ``density`` sets their shape density (0 for a kinematic container).
    """
    register_custom_attributes(builder)
    lo, hi = np.asarray(inner_lo, float), np.asarray(inner_hi, float)
    c = 0.5 * (lo + hi)
    hx, hy, hz = 0.5 * (hi - lo)
    t = 0.5 * wall
    ids = {}
    if floor:
        ids["floor"] = _box(builder, (c[0], c[1], lo[2] - t), (hx + wall, hy + wall, t), body, BOUNDARY_SDF, density)
    for name, sx in (("x_lo", -1.0), ("x_hi", 1.0)):
        ids[name] = _box(builder, (c[0] + sx * (hx + t), c[1], c[2]), (t, hy + wall, hz), body, BOUNDARY_SDF,
                         density)
    for name, sy in (("y_lo", -1.0), ("y_hi", 1.0)):
        ids[name] = _box(builder, (c[0], c[1] + sy * (hy + t), c[2]), (hx, t, hz), body, BOUNDARY_SDF, density)
    if not open_top:
        ids["lid"] = _box(builder, (c[0], c[1], hi[2] + t), (hx + wall, hy + wall, t), body, BOUNDARY_SDF, density)
    return ids


def add_orifice_tank(builder: newton.ModelBuilder, inner_lo: Vec3, inner_hi: Vec3, wall: float,
                     hole: float) -> dict[str, int]:
    """A box tank whose floor (thickness ``wall``) has a centred square orifice of side ``hole`` (a thin-plate drain);
    the floor is four boxes around the hole. Returns shape ids (walls as ``add_box_tank``, floor parts f0..f3)."""
    ids = add_box_tank(builder, inner_lo, inner_hi, wall, floor=False)
    lo, hi = np.asarray(inner_lo, float), np.asarray(inner_hi, float)
    c, t, a = 0.5 * (lo + hi), 0.5 * wall, 0.5 * hole
    hx, hy = 0.5 * (hi[0] - lo[0]) + wall, 0.5 * (hi[1] - lo[1]) + wall
    z = lo[2] - t
    sx = 0.5 * (hx - a)  # half extent of the x strips (full y), y strips sit between them
    parts = [((c[0] - a - sx, c[1], z), (sx, hy, t)), ((c[0] + a + sx, c[1], z), (sx, hy, t)),
             ((c[0], c[1] - a - 0.5 * (hy - a), z), (a, 0.5 * (hy - a), t)),
             ((c[0], c[1] + a + 0.5 * (hy - a), z), (a, 0.5 * (hy - a), t))]
    for k, (center, half) in enumerate(parts):
        ids[f"f{k}"] = _box(builder, center, half, -1, BOUNDARY_SDF)
    return ids


def add_floating_box(builder: newton.ModelBuilder, cfg: DFSPHConfig, fill: tuple[int, int, int],
                     box: tuple[float, float, float], density: float, rest_density: float = 1000.0) -> dict:
    """A pool ``fill`` lattice cells deep in a box tank with a free box (``box`` = side lengths in cells) floating at
    its expected draft m / (rho0 A) in the middle; lattice cells inside the submerged part are left out. Returns
    ``body``, ``draft`` [m], ``half`` (box half extents [m]), ``tank`` shape ids and ``area`` / ``box_area`` [m^2]."""
    s = cfg.spacing
    nx, ny, nz = fill
    half = tuple(0.5 * b * s for b in box)
    inner_hi = (nx * s, ny * s, (nz + 2.0 * box[2]) * s)
    tank = add_box_tank(builder, (0.0, 0.0, 0.0), inner_hi, wall=cfg.h)
    cx, cy, surface = 0.5 * nx * s, 0.5 * ny * s, nz * s
    draft = 2.0 * half[2] * density / rest_density
    cz = max(surface - draft + half[2], half[2] + 0.25 * s)  # a box denser than the liquid rests on the floor
    pts = lattice_points((0.0, 0.0, 0.0), fill, s)
    inside = ((np.abs(pts[:, 0] - cx) < half[0] + 0.5 * s) & (np.abs(pts[:, 1] - cy) < half[1] + 0.5 * s)
              & (pts[:, 2] > cz - half[2] - 0.5 * s))
    add_fluid_points(builder, cfg, pts[~inside], rest_density)
    body = builder.add_body(xform=wp.transform((cx, cy, cz), wp.quat_identity()))
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=density),
                          custom_attributes={"sph:shape_boundary": BOUNDARY_SDF})
    return {"body": body, "draft": draft, "half": half, "tank": tank, "area": nx * ny * s * s,
            "box_area": 4.0 * half[0] * half[1]}


def _inside_primitive(pts: np.ndarray, kind: str, size: tuple[float, ...]) -> np.ndarray:
    """Mask of points inside a primitive centred at the origin (Newton conventions: cylinder/capsule along z)."""
    r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    if kind == "box":
        return np.all(np.abs(pts) <= np.asarray(size[:3]), axis=1)
    if kind == "sphere":
        return r2 + pts[:, 2] ** 2 <= size[0] ** 2
    if kind == "cylinder":
        return (r2 <= size[0] ** 2) & (np.abs(pts[:, 2]) <= size[1])
    if kind == "capsule":
        dz = np.maximum(np.abs(pts[:, 2]) - size[1], 0.0)
        return r2 + dz ** 2 <= size[0] ** 2
    if kind == "ellipsoid":
        return np.sum((pts / np.asarray(size[:3])) ** 2, axis=1) <= 1.0
    raise ValueError(f"SPH emit: no liquid volume for primitive {kind!r}")


def liquid_points(shape, spacing: float) -> np.ndarray:
    """2r lattice points filling a ``ParticleSource`` region or a closed ``Primitive`` (object frame), centred so the
    lattice sits symmetrically in the volume."""
    from alloy.cards.types import ParticleSource, Primitive

    if isinstance(shape, ParticleSource):
        lo, hi = np.asarray(shape.region[0], float), np.asarray(shape.region[1], float)
    elif isinstance(shape, Primitive) and shape.kind != "plane":
        ext = {"box": shape.size[:3], "sphere": (shape.size[0],) * 3, "ellipsoid": shape.size[:3],
               "cylinder": (shape.size[0], shape.size[0], shape.size[1]),
               "capsule": (shape.size[0], shape.size[0], shape.size[1] + shape.size[0])}.get(shape.kind)
        if ext is None:
            raise ValueError(f"SPH emit: no liquid volume for primitive {shape.kind!r}")
        lo, hi = -np.asarray(ext, float), np.asarray(ext, float)
    else:
        raise ValueError(f"SPH emit: liquid needs a ParticleSource region or a closed primitive, got {type(shape)}")
    counts = np.maximum(np.floor((hi - lo) / spacing + 1.0e-9).astype(int), 1)
    start = 0.5 * (lo + hi) - 0.5 * counts * spacing
    pts = lattice_points(tuple(start), tuple(int(c) for c in counts), spacing)
    if isinstance(shape, Primitive):
        pts = pts[_inside_primitive(pts, shape.kind, shape.size)]
    return pts


def emit_object(builder: newton.ModelBuilder, obj, decision) -> None:
    """AlloySolverPlugin.emit: a liquid ``ResolvedObject`` -> active lattice particles at the router's radius
    (``decision.numbers["particle_radius"]``), mass rho0 / S, the card's viscosity on ``sph:viscosity`` (object
    frame; the assembler places the sub-builder at the object's pose). Raises CapacityError over the particle budget."""
    from alloy.core.errors import CapacityError

    r = float(decision.numbers.get("particle_radius", 0.0))
    if not r > 0.0:
        raise ValueError(f"{obj.card.name}: liquid_sph decision without particle_radius")
    cfg = DFSPHConfig(particle_radius=r)
    pts = liquid_points(obj.card.shape, cfg.spacing)
    budget = obj.card.budget.particles
    if len(pts) > budget:
        raise CapacityError(f"{obj.card.name}: {len(pts)} SPH particles at r = {r:.3g} m > budget {budget}")
    fluid = obj.material.fluid
    mu = float(fluid.viscosity.value) if fluid is not None else 1.0e-3
    add_fluid_points(builder, cfg, pts, rest_density=float(obj.material.density.value), viscosity=mu)
    log.debug("SPH emit %s: %d particles, r=%g m, rho0=%g, mu=%g", obj.card.name, len(pts), r,
              float(obj.material.density.value), mu)
