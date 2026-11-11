# SPDX-License-Identifier: Apache-2.0
"""Read a Newton Model (or a SolverCoupled ModelView) into the device tables SolverDFSPH uses."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp
from alloy._vendor.newton import BodyFlags, GeoType, Model
from alloy._vendor.newton._src.geometry.sdf_texture import TextureSDFData

from .boundary import ShapeArrays, world_shape_table
from .kernels import boundary_fit
from .sdf_band import bvh_flags
from .particle_walls import empty_walls
from .soft_walls import empty_soft
from .config import BOUNDARY_IGNORE, PHASE_LIQUID, PHASE_SPARE, DFSPHConfig, particle_mass, sph_phase

log = logging.getLogger("alloy.sph")


@dataclass
class ParticleLayout:
    """Particle ranges per world and the SPH attribute arrays."""

    n: int
    world_count: int
    world_start: np.ndarray  # [W + 1] contiguous ranges
    phase: wp.array  # int8 (Model attribute; emitters and sinks write it)
    rest_density: wp.array  # float32
    viscosity_max: float
    sph_count: int  # liquid, spares and held (config.sph_phase)
    uniform: bool = True  # one particle mass and one rest density (the pressure operator is symmetric)


def _require_sph_namespace(model: Model) -> None:
    ns = getattr(model, "sph", None)
    if ns is None or getattr(ns, "phase", None) is None or getattr(ns, "rest_density", None) is None:
        raise ValueError(
            "SolverDFSPH needs the sph:* custom attributes; call SolverDFSPH.register_custom_attributes(builder) "
            "before adding particles"
        )


def particle_layout(model: Model, cfg: DFSPHConfig) -> ParticleLayout:
    """Validate SPH particles (world >= 0, contiguous per world) and check masses against rho0 / S."""
    _require_sph_namespace(model)
    n = int(model.particle_count)
    W = int(model.world_count)
    phase = model.sph.phase.numpy() if n else np.zeros(0, np.int8)
    world = model.particle_world.numpy() if n else np.zeros(0, np.int32)
    sph = sph_phase(phase)
    if np.any(world[sph] < 0):
        raise ValueError("SPH particles must belong to a world (particle_world >= 0)")
    if n and np.any(np.diff(world[world >= 0]) < 0):
        raise ValueError("particles must be contiguous per world (Model.particle_world sorted)")
    start = np.zeros(W + 1, dtype=np.int64)
    if n:
        ws = model.particle_world_start.numpy()
        start[:W] = ws[:W]
        start[W] = ws[W]
    rho0 = model.sph.rest_density.numpy() if n else np.zeros(0, np.float32)
    _check_masses(model, cfg, phase, rho0)
    _reject_tension_attributes(model, sph, cfg)
    uniform = _is_uniform(model.particle_mass.numpy()[sph] if n else np.zeros(0), rho0[sph])
    mu = model.sph.viscosity.numpy()[sph] if n and getattr(model.sph, "viscosity", None) is not None else []
    return ParticleLayout(
        n=n, world_count=W, world_start=start, phase=model.sph.phase, rest_density=model.sph.rest_density,
        viscosity_max=float(np.max(mu)) if len(mu) else 0.0, sph_count=int(np.count_nonzero(sph)), uniform=uniform,
    )


# tension inputs each model reads (SCR-SC1-2 ruling, CARD-3; SCR-WP9-7)
TENSION_INPUTS = {"none": (), "akinci2013": ("surface_tension", "adhesion"), "csf": ("surface_tension",
                                                                                    "shape_contact_angle")}


def _reject_tension_attributes(model: Model, sph: np.ndarray, cfg: DFSPHConfig) -> None:
    """Refuse tension inputs the configured model would silently drop: nonzero sph:surface_tension / sph:adhesion
    per particle, or a set sph:shape_contact_angle (>= 0) per shape, that Config.surface_tension does not read."""
    used = TENSION_INPUTS[cfg.surface_tension]
    for name, unset in (("surface_tension", 0.0), ("adhesion", 0.0), ("shape_contact_angle", -1.0)):
        arr = getattr(model.sph, name, None)
        per_particle = name != "shape_contact_angle"
        if name in used or arr is None or (per_particle and not np.any(sph)):
            continue
        vals = arr.numpy()[sph] if per_particle else arr.numpy()
        if np.any(vals != unset):
            what = "particles" if per_particle else "shapes"
            raise ValueError(f"SolverDFSPH: {np.count_nonzero(vals != unset)} {what} set sph:{name} but "
                             f"Config.surface_tension={cfg.surface_tension!r} does not read it "
                             f"(csf reads {TENSION_INPUTS['csf']}, akinci2013 {TENSION_INPUTS['akinci2013']})")


def _is_uniform(mass: np.ndarray, rho0: np.ndarray, rel: float = 1.0e-6) -> bool:
    if len(mass) == 0:
        return True
    spread = lambda a: float(np.ptp(a)) / max(float(np.max(np.abs(a))), 1.0e-30)  # noqa: E731
    return spread(mass.astype(np.float64)) <= rel and spread(rho0.astype(np.float64)) <= rel


def _check_masses(model: Model, cfg: DFSPHConfig, phase: np.ndarray, rho0: np.ndarray) -> None:
    """Warn when particle mass is not rho0 / S for the configured radius (the rest state would be off)."""
    liquid = sph_phase(phase)
    if not np.any(liquid):
        return
    mass = model.particle_mass.numpy()[liquid].astype(np.float64)
    r0 = rho0[liquid]
    for rho in np.unique(r0):
        expect = particle_mass(cfg, float(rho))
        rel = float(np.max(np.abs(mass[r0 == rho] - expect)) / expect)
        if rel > 1.0e-4:
            log.warning("SPH particle masses differ from rho0/S (rho0=%g, r=%g m) by up to %.3g relative; the "
                        "rest density will be off", rho, cfg.particle_radius, rel)


def _dummy(device: wp.context.Device, dtype: type, n: int = 1) -> wp.array:
    return wp.zeros(n, dtype=dtype, device=device)


@dataclass
class ShapeTable:
    arrays: ShapeArrays
    keep: list  # device arrays referenced by the struct
    count: int


def origin_radius(model: Model) -> np.ndarray:
    """Per shape, the radius about the shape's own origin that encloses it. The boundary cull (boundary.py), the
    sweep (integrate.py) and the leak monitor measure from the shape origin, but Newton's ``shape_collision_radius``
    of a mesh is half its AABB diagonal, i.e. about the AABB centre (newton@ac7f7830 geometry/utils.py:89-100): a
    bottle mesh with its origin at the base lost every wall beyond that radius from the base (SC1, 2026-11-03)."""
    rad = model.shape_collision_radius.numpy().astype(np.float32).copy()
    types, scale = model.shape_type.numpy(), model.shape_scale.numpy()
    for s in np.nonzero(np.isin(types, (int(GeoType.MESH), int(GeoType.CONVEX_MESH))))[0]:
        src = model.shape_source[s] if s < len(model.shape_source) else None
        if src is None or getattr(src, "vertices", None) is None:
            continue
        v = np.asarray(src.vertices, dtype=np.float64) * np.asarray(scale[s], dtype=np.float64)
        rad[s] = max(float(rad[s]), float(np.linalg.norm(v, axis=1).max()))
    return rad


def _bvh_arrays(model: Model, ids: np.ndarray, reach: float, device: wp.context.Device, keep: list) -> tuple:
    """(Model.shape_source_ptr, per-shape BVH flags) for boundary.shape_sdf (SCR-WP9-9, SCR-JOB2-2; sdf_band.py); a
    table mesh that needs the BVH but has no wp.Mesh is refused, naming it."""
    S = int(model.shape_count)
    ptr = getattr(model, "shape_source_ptr", None)
    host = ptr.numpy() if ptr is not None else np.zeros(S, np.uint64)
    flags, why = bvh_flags(model, ids, reach, device)
    bvh = np.nonzero(flags)[0]
    if len(bvh):
        labels = list(getattr(model, "shape_label", []) or [])
        name = lambda s: labels[s] if s < len(labels) and labels[s] else f"shape {s}"  # noqa: E731
        missing = [int(s) for s in bvh if host[s] == 0]
        if missing:
            raise ValueError(f"SolverDFSPH: {len(missing)} particle-colliding mesh shapes need the mesh BVH (no volume "
                             f"SDF, or one whose band misses the reach) but have no mesh: "
                             f"{', '.join(name(s) for s in missing[:8])}")
        log.info("SolverDFSPH: %d particle-colliding mesh shapes read their walls from the mesh BVH: %d without a "
                 "volume SDF (e.g. %s), %d whose SDF band is narrower than the %.2f mm reach (e.g. %s)", len(bvh),
                 len(why["no_sdf"]), name(why["no_sdf"][0]) if why["no_sdf"] else "-", len(why["narrow_band"]),
                 reach * 1e3, name(why["narrow_band"][0]) if why["narrow_band"] else "-")
    if ptr is None:
        ptr = wp.array(host if S else np.zeros(1, np.uint64), dtype=wp.uint64, device=device)
        keep.append(ptr)
    flag_arr = wp.array(flags if S else np.zeros(1, np.int32), dtype=wp.int32, device=device)
    keep.append(flag_arr)
    return ptr, flag_arr


def _reach(cfg: DFSPHConfig) -> float:
    """Slots are kept for phi < reach: 1.5 spacings, or h when the wall viscosity needs the whole support (pressure
    psi is 0 past 1.5 spacings, SCR-SC1-3)."""
    return float(cfg.h) if cfg.boundary_viscosity else 1.5 * cfg.spacing


def shape_table(model: Model, cfg: DFSPHConfig, device: wp.context.Device) -> ShapeTable:
    """Particle-colliding SDF shapes per world (+ global), excluding sph.shape_boundary == ignore."""
    S = int(model.shape_count)
    W = int(model.world_count)
    if S == 0:
        start, ids = np.zeros(W + 1, np.int32), np.zeros(0, np.int32)
    else:
        excl = None
        sb = getattr(getattr(model, "sph", None), "shape_boundary", None)
        if sb is not None:
            excl = sb.numpy() == BOUNDARY_IGNORE
        start, ids = world_shape_table(
            model.shape_world.numpy(), model.shape_flags.numpy(), model.shape_type.numpy(), W, excl
        )
    keep: list = []

    def up(x: np.ndarray, dtype: type) -> wp.array:
        a = wp.array(x if len(x) else np.zeros(1, x.dtype), dtype=dtype, device=device)
        keep.append(a)
        return a

    a = ShapeArrays()
    a.start = up(start, wp.int32)
    a.ids = up(ids, wp.int32)
    if S:
        a.geo, a.scale, a.xform = model.shape_type, model.shape_scale, model.shape_transform
        a.body, a.radius = model.shape_body, up(origin_radius(model), wp.float32)
        a.mu = model.shape_material_mu
        a.sdf_index = model._shape_sdf_index
        a.mesh, a.bvh = _bvh_arrays(model, ids, _reach(cfg), device, keep)
    else:
        a.geo, a.body, a.sdf_index, a.bvh = (_dummy(device, wp.int32) for _ in range(4))
        a.mesh = _dummy(device, wp.uint64)
        a.scale, a.xform, a.radius = _dummy(device, wp.vec3), _dummy(device, wp.transform), _dummy(device, wp.float32)
        a.mu = _dummy(device, wp.float32)
    tex = getattr(model, "_texture_sdf_data", None)
    if tex is None or len(tex) == 0:
        tex = _dummy(device, TextureSDFData)
        keep.append(tex)
    a.tex = tex
    a.curvature = int(cfg.boundary_curvature_order)
    a.offset, a.amplitude = (float(v) for v in boundary_fit(cfg.spacing, cfg.h, cfg.kernel))
    a.taper0, a.taper1 = 0.5 * cfg.spacing, 1.5 * cfg.spacing
    a.reach = _reach(cfg)
    a.bvh_dist = 2.0 * float(cfg.h)  # BVH search radius (boundary._mesh_bvh_sdf)
    a.soft = empty_soft(S, device)  # SolverDFSPH installs its SoftWalls (soft_walls.py)
    a.pw = empty_walls(S, cfg.h, cfg.kernel_id, device)  # SolverDFSPH installs its ParticleWalls
    a.friction = int(bool(cfg.wall_friction))
    return ShapeTable(arrays=a, keep=keep, count=len(ids))


@dataclass
class BodyArrays:
    """Model body arrays, or 1-element dummies when the model has no bodies."""

    count: int
    mass: wp.array
    inv_mass: wp.array
    inertia: wp.array
    inv_inertia: wp.array
    com: wp.array
    world: wp.array
    flags: wp.array


def body_arrays(model: Model, device: wp.context.Device) -> BodyArrays:
    B = int(model.body_count)
    if B == 0:
        return BodyArrays(0, _dummy(device, wp.float32), _dummy(device, wp.float32), _dummy(device, wp.mat33),
                          _dummy(device, wp.mat33), _dummy(device, wp.vec3), _dummy(device, wp.int32),
                          _dummy(device, wp.int32))
    flags = getattr(model, "body_flags", None)
    if flags is None:
        flags = wp.full(B, int(BodyFlags.DYNAMIC), dtype=wp.int32, device=device)
    return BodyArrays(B, model.body_mass, model.body_inv_mass, model.body_inertia, model.body_inv_inertia,
                      model.body_com, model.body_world, flags)


def spare_ids(phase: np.ndarray, world: np.ndarray, flags: np.ndarray, w: int, active_flag: int) -> np.ndarray:
    """Inactive SPH spares of world w, ascending ids."""
    mask = (phase == PHASE_SPARE) & (world == w) & ((flags & active_flag) == 0)
    return np.nonzero(mask)[0].astype(np.int32)


def liquid_ids(phase: np.ndarray) -> np.ndarray:
    return np.nonzero(phase == PHASE_LIQUID)[0]
