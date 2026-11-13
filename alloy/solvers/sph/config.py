# SPDX-License-Identifier: Apache-2.0
"""SolverDFSPH configuration (SPEC §3.8), custom attributes and time-step helpers.

Fields up to ``world_bounds`` are the SPEC §3.8 ``SolverDFSPH.Config``; the fields below them are WP9 additions with
defaults (allowed without an SCR, D17).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from alloy._vendor import newton

from .kernels import KERNEL_IDS, kernel_w_np, lattice_sum

log = logging.getLogger("alloy.sph")

Vec3 = tuple[float, float, float]

# Router rule: viscosity above this uses weiler2018 (SPEC §3.3 RouterPolicy.sph_viscous_above, §4.2).
VISCOUS_ABOVE = 1.0e-2  # Pa·s
# Phase values of the sph:phase attribute: -1 = not an SPH particle (default for foreign particles).
PHASE_NONE = -1
PHASE_SPARE = 0
PHASE_LIQUID = 1
PHASE_RETIRED = -2  # sunk or escaped; never reused
PHASE_HELD = -3  # a port's reservoir (ports.py): inactive, reused only by SPHPort.emit, never by emitters


def sph_phase(phase: np.ndarray) -> np.ndarray:
    """Host mask of the SPH solver's particles at build: liquid, spares and a port's held reservoir."""
    return (phase >= PHASE_SPARE) | (phase == PHASE_HELD)
# sph:shape_boundary values (per shape).
BOUNDARY_AUTO = 0
BOUNDARY_SDF = 1
BOUNDARY_AKINCI = 2
BOUNDARY_IGNORE = 3


@dataclass
class DFSPHConfig:
    particle_radius: float = 2.0e-3  # r [m]; spacing 2r; the router sets min(2 mm, passage/16)
    support_factor: float = 4.0  # h = support_factor * r
    kernel: Literal["cubic", "wendland_c2"] = "cubic"
    max_density_error: float = 1.0e-3  # mean max(rho - rho0, 0)/rho0 per world (unilateral, SCR-WP9-3)
    rest_jitter_speed: float | None = 1.5e-3
    """[m/s] Resting-jitter rule (SCR-SC-C2, solve_driver.py): per world and step the density tolerance is
    min(max_density_error, v·dt/H), H the liquid's pressure-weighted hydrostatic depth from the density warm start,
    v = max(rest_jitter_speed, rest_jitter_flow_ratio·U), U = |mean velocity| of the world's liquid at the start of
    the step, floored at min_density_error; a settled column jitters at about tol·H/dt. None: max_density_error
    alone."""
    rest_jitter_flow_ratio: float = 0.5
    """[1] Bulk flow hides jitter below a fraction of its speed, so the rule allows jitter up to this fraction of U and
    flowing scenes (a dam break at its CFL dt) keep about the base tolerance's cost. In [0, 1]; 0: no allowance."""
    min_density_error: float = 1.0e-5
    max_divergence_error: float = 1.0e-3  # per step, mean max(Drho/Dt, 0)·dt/rho0 per world
    min_iterations: int = 2
    max_iterations: int = 50
    cfl: float = 0.2  # dt <= cfl·2r/|v|max, snapped down to the D4 grid at period boundaries
    speed_margin: float = 0.25  # coupled clock's estimate: this x the CFL step (a pour beside cloth splashed at 1)
    dt_min: float = 1.953125e-5
    dt_max: float = 1.25e-3
    viscosity: Literal["auto", "none", "xsph", "weiler2018"] = "auto"  # auto: weiler2018 if mu > 1e-2 Pa·s
    xsph_c: float = 0.01
    surface_tension: Literal["none", "akinci2013", "csf"] = "none"
    """csf: continuum surface force with the Breinlinger contact angle (csf.py; sph:surface_tension per particle,
    sph:shape_contact_angle per shape; laws in wp9-results.md); akinci2013: Akinci 2013 cohesion and adhesion
    (tension.py; sph:surface_tension, sph:adhesion), qualitative."""
    boundary: Literal["sdf_tube", "akinci_particles", "auto"] = "auto"  # auto: particles for walls < h
    boundary_curvature_order: int = 1
    neighbor_capacity: int = 64
    # x h: lists reused until a particle moves > skin/2 (L4a, neighbors_skin.py; results bitwise the same). Off by
    # default: on a T4 at 0.1, 64 x 4k at rest 1.27x faster, a dam break 14 % slower, +13 % memory per world (r67)
    reneighbor_skin: float | None = None
    coupling: Literal["two_way", "one_way"] = "two_way"
    world_bounds: tuple[Vec3, Vec3] | None = None  # escape box (per world, origin frame); the sph capability's
    # factory sets it from the scene's content when the caller gives none (escape_box.py, SCR-OSS3-2)
    # ---- WP9 additions ----
    integrate_bodies: Literal["proxies", "dynamic", "none"] = "proxies"
    """Which bodies SolverDFSPH integrates with the fluid wrench: proxy bodies at the source's effective mass
    ("proxies"), also non-proxy dynamic bodies it owns ("dynamic", D5 monolithic free bodies), or none (the fluid
    wrench is reported directly by the harvest hook)."""
    min_neighbors_divergence: float = 20.0  # SPlisHSPlasH TimeStepDFSPH.cpp:417-424 (3D)
    divergence_solve: bool = True
    divergence_two_sided: bool = False  # no expansion where the liquid is under pressure (pressure.py)
    jacobi_omega: float = 0.5  # relaxation of the Jacobi update (SPlisHSPlasH: 0.5)
    max_error_factor: float = 10.0  # also stop only when the per-particle max error <= factor x tolerance (LIQ-3 max)
    warm_start_density: float = 0.9  # x previous k (SPlisHSPlasH: 0.5); measured trade-off in wp9-results.md
    warm_start_divergence: float = 0.5  # SPlisHSPlasH TimeStepDFSPH.cpp:428-431
    grid_dim: int = 128  # hash grid cells per axis
    viscosity_tolerance: float = 1.0e-3  # weiler2018: per world |r| <= tol |v*|
    boundary_viscosity: float | None = None
    """[Pa·s] Wall viscosity mu_b of the weiler2018 solve (SCR-SC1-1, wall_viscosity.py): None or 0 = free-slip walls
    (bitwise the pre-SCR solver); the liquid's own mu gives no-slip walls (honey on a dipper). Needs weiler2018."""
    viscosity_max_iterations: int = 100
    diagonal_floor: float = 0.25
    """Floor of the DFSPH diagonal 1/alpha as a fraction of its interior rest-lattice value: bounds the pressure step
    where the density gradient degenerates (particles nearly on top of each other, pressure.py, LIQ-8)."""
    contact_separation_speed: float = 0.5
    """[m/s] Pressure on a particle without fluid neighbours (wall contact only) may stop its approach to the walls and
    separate it at most this fast relative to them (pressure.py, lone particles, LIQ-8)."""
    compression_relief_speed: float | None = 0.5
    """[m/s] The density solve removes a particle's existing compression r_i - 1 at most about this fast: the
    positional part of its source is capped at v·dt·sqrt(D), D the interior rest-lattice diagonal (pressure.py,
    compression relief). A flat impact leaves corner and floor particles in the walls' 1 um projection skin (r_i
    1.5-2.5), and the uncapped solve shot them and their neighbours out at 3-500 m/s. Inactive in normal flow (the cap
    is 8-20 % at dt 0.5-1.25 ms, r = 1 mm). None: uncapped (SPlisHSPlasH)."""
    wall_friction: bool = True
    """Coulomb friction on the wall projection (SCR-OSS3-2): a particle pushed out of a wall loses tangential speed
    relative to it by up to mu |v_n|, mu the shape's ``shape_material_mu``, v_n the inward speed removed. A lone particle
    resting on a floor (no neighbours, no pressure) is projected every step and slows at mu g; before it slid forever.
    Liquid held off walls by its pressure is never projected and is unchanged. False: bitwise the solver before."""
    ambient_pressure: float | None = None
    """[Pa] Model ambient pressure for sealed liquid (SCR-SC10-1, ambient.py): liquid connected to ``sph:shape_seal``
    walls may carry gauge pressure down to -ambient_pressure, so a column hangs under a sealing piston; the free
    surface stays at 0 gauge and open walls one-sided. It is the largest tension sealed liquid carries, not the
    atmosphere: set a few times the scene's need (C18: rho g H = 160 Pa, 500 Pa; 101 kPa blows a C18 column apart).
    Flag every wall the sealed liquid wets (the piston and the bore; a wall a liquid bridge spans). Start the liquid
    in contact: the seal wall above a column half a spacing over its top layer, as a lattice fill leaves it; a wider
    gap reads as air at ambient, not vacuum (SC10's 32 uL column started 1.2 spacings under its piston and slid
    0.47 mm). Sealed liquid at gauge >= 0 behaves as open. None: unilateral pressure, bitwise the solver without it."""
    pressure_solver: Literal["cg", "jacobi"] = "cg"
    """Pressure iteration: per-world projected preconditioned CG (krylov.py; needs one particle mass and rest density,
    else Jacobi is used with a warning) or relaxed Jacobi (SPlisHSPlasH). Measured on a 50-layer resting column: CG
    8.2 density iterations per step, Jacobi 21 (O(N^2) in the layer count), same wall time on CPU."""

    @property
    def h(self) -> float:
        """Support radius [m]."""
        return self.support_factor * self.particle_radius

    @property
    def spacing(self) -> float:
        return 2.0 * self.particle_radius

    @property
    def kernel_id(self) -> int:
        return KERNEL_IDS[self.kernel]

    @classmethod
    def fast(cls, **overrides) -> DFSPHConfig:
        """The 'fast' preset (SCR-WP8-7): ``rest_jitter_speed=None`` (the SC-C2 resting-jitter rule off) and
        ``cfl=0.4``, plus ``overrides``.

        - Liquid at rest takes 2-3x fewer density iterations: LIQ-9's 30k-particle column on an L4 went 10.6 to
          4.6 ms/step. A settled column then jitters at max_density_error·H/dt, not at rest_jitter_speed, so
          A13.resting_jitter (LIQ-10) is not met.
        - Flowing scenes take half the steps: A14's dam break passes at CFL 0.4.
        - Not for capillary or sealed liquids (``surface_tension="csf"``, ``ambient_pressure``): A13.sealed_column
          fails at twice its dt (pressure error 0.69, bound 0.1), so keep the default CFL 0.2 there."""
        return cls(**{"rest_jitter_speed": None, "cfl": 0.4, **overrides})

    def validate(self) -> None:
        """Raise ValueError on an inconsistent configuration."""
        checks = [
            (self.particle_radius > 0.0, "particle_radius must be > 0"),
            (self.support_factor >= 2.0, "support_factor must be >= 2 (h >= spacing)"),
            (self.kernel in KERNEL_IDS, f"unknown kernel {self.kernel!r}"),
            (0 < self.min_iterations <= self.max_iterations, "need 0 < min_iterations <= max_iterations"),
            (self.max_density_error > 0.0 and self.max_divergence_error > 0.0, "error tolerances must be > 0"),
            (self.rest_jitter_speed is None or self.rest_jitter_speed > 0.0, "rest_jitter_speed must be > 0 or None"),
            (self.compression_relief_speed is None or self.compression_relief_speed > 0.0,
             "compression_relief_speed must be > 0 or None"),
            (self.ambient_pressure is None or self.ambient_pressure > 0.0, "ambient_pressure must be > 0 or None"),
            (0.0 <= self.rest_jitter_flow_ratio <= 1.0, "rest_jitter_flow_ratio must be in [0, 1]"),
            (0.0 < self.min_density_error <= self.max_density_error,
             "need 0 < min_density_error <= max_density_error"),
            (0.0 < self.cfl <= 1.0, "cfl must be in (0, 1]"),
            (0.0 < self.speed_margin <= 1.0, "speed_margin must be in (0, 1]"),
            (0.0 < self.dt_min <= self.dt_max, "need 0 < dt_min <= dt_max"),
            (self.viscosity in ("auto", "none", "xsph", "weiler2018"), f"unknown viscosity {self.viscosity!r}"),
            (self.surface_tension in ("none", "akinci2013", "csf"),
             f"unknown surface_tension {self.surface_tension!r}"),
            (self.boundary in ("sdf_tube", "akinci_particles", "auto"), f"unknown boundary {self.boundary!r}"),
            (self.boundary_curvature_order in (0, 1), "boundary_curvature_order must be 0 or 1"),
            (self.coupling in ("two_way", "one_way"), f"unknown coupling {self.coupling!r}"),
            (self.integrate_bodies in ("proxies", "dynamic", "none"), "unknown integrate_bodies"),
            (0.0 <= self.warm_start_density <= 1.0 and 0.0 <= self.warm_start_divergence <= 1.0,
             "warm start factors must be in [0, 1]"),
            (0.0 < self.jacobi_omega <= 0.6, "jacobi_omega must be in (0, 0.6] (0.8 and 1.0 diverge, measured)"),
            (self.max_error_factor >= 1.0, "max_error_factor must be >= 1"),
            (self.reneighbor_skin is None or 0.0 <= self.reneighbor_skin <= 1.0, "reneighbor_skin must be in [0, 1]"),
            (0.0 <= self.diagonal_floor < 1.0, "diagonal_floor must be in [0, 1)"),
            (self.contact_separation_speed >= 0.0, "contact_separation_speed must be >= 0"),
            (self.pressure_solver in ("jacobi", "cg"), f"unknown pressure_solver {self.pressure_solver!r}"),
            (self.viscosity_tolerance > 0.0 and self.viscosity_max_iterations > 0, "viscosity solve limits must be > 0"),
            (self.boundary_viscosity is None or self.boundary_viscosity >= 0.0, "boundary_viscosity must be >= 0"),
            (not self.boundary_viscosity or self.viscosity in ("auto", "weiler2018"),
             "boundary_viscosity needs viscosity 'weiler2018' (or 'auto' resolving to it)"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(f"SolverDFSPH.Config: {msg}")


def resolve_viscosity(cfg: DFSPHConfig, mu_max: float) -> str:
    """'auto' -> weiler2018 above VISCOUS_ABOVE Pa·s, else xsph (SPEC §3.8, §4.2)."""
    if cfg.viscosity != "auto":
        return cfg.viscosity
    return "weiler2018" if mu_max > VISCOUS_ABOVE else "xsph"


def stable_dt(cfg: DFSPHConfig, v_max: float) -> float:
    """DFSPH CFL step clamp(cfl·2r/|v|max, dt_min, dt_max) [s] (D6)."""
    if v_max <= 0.0 or not math.isfinite(v_max):
        return cfg.dt_max if v_max <= 0.0 else cfg.dt_min
    return float(min(max(cfg.cfl * cfg.spacing / v_max, cfg.dt_min), cfg.dt_max))


def snap_down(dt: float, control_dt: float, j_max: int = 10) -> float:
    """Largest control_dt / 2^j <= dt with j <= j_max (D4 grid); the finest rung if dt is below it."""
    for j in range(j_max + 1):
        g = control_dt / float(2**j)
        if g <= dt:
            return g
    return control_dt / float(2**j_max)


def particle_mass(cfg: DFSPHConfig, rest_density: float) -> float:
    """m = rho0 / S, S = lattice kernel sum at spacing 2r, so an interior lattice has rho = rho0 exactly [kg]."""
    return float(rest_density / lattice_sum(cfg.spacing, cfg.h, cfg.kernel))


def full_neighbor_count(cfg: DFSPHConfig) -> float:
    """Neighbours (self excluded) of an interior lattice particle within h; scales boundary volume to a count."""
    n = int(math.ceil(cfg.h / cfg.spacing)) + 1
    ax = np.arange(-n, n + 1) * cfg.spacing
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(gx**2 + gy**2 + gz**2)
    return float(np.count_nonzero(kernel_w_np(r, cfg.h, cfg.kernel) > 0.0) - 1)


def _attr(name: str, dtype: type, freq: newton.Model.AttributeFrequency, default: float | int):
    return newton.ModelBuilder.CustomAttribute(name=name, dtype=dtype, frequency=freq, namespace="sph",
                                              default=default)


def register_custom_attributes(builder: newton.ModelBuilder) -> None:
    """model.sph.{rest_density kg/m^3, viscosity Pa·s, surface_tension N/m, adhesion -, phase int8} per particle
    (phase -1 not SPH, 0 spare, 1 liquid) and per shape sph.shape_boundary int8 (0 auto, 1 sdf, 2 akinci, 3 none),
    sph.shape_contact_angle rad (-1 free) and sph.shape_seal int8 (1: may hold liquid under tension, ambient.py).
    Idempotent (Newton accepts a re-registration with the same spec)."""
    P = newton.Model.AttributeFrequency.PARTICLE
    S = newton.Model.AttributeFrequency.SHAPE
    for a in (
        _attr("rest_density", wp.float32, P, 1000.0),
        _attr("viscosity", wp.float32, P, 1.0e-3),
        _attr("surface_tension", wp.float32, P, 0.0),
        _attr("adhesion", wp.float32, P, 0.0),
        _attr("phase", wp.int8, P, PHASE_NONE),
        _attr("shape_boundary", wp.int8, S, BOUNDARY_AUTO),
        _attr("shape_contact_angle", wp.float32, S, -1.0),
        _attr("shape_seal", wp.int8, S, 0),
    ):
        builder.add_custom_attribute(a)
