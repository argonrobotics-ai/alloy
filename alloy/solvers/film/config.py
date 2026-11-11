# SPDX-License-Identifier: Apache-2.0
"""SolverFilm configuration and its custom attributes (SC8, docs/notes/sc8-film-design.md).

Per-object values live on the model as custom attributes in the ``film`` namespace, written by the film emitter
(``film.emit``) or by ``add_cloth_mesh(custom_attributes_*=...)``; every default is "off", so a model without film
items steps exactly as ``SolverVBD`` (bitwise, tests/solvers/film/test_film_default_off.py):

- ``film:strain_cap`` per triangle: largest principal stretch minus 1 (0 = not limited);
- ``film:compress_cap`` per triangle: largest principal compression 1 - s_min (>= 1 = not limited);
- ``film:air`` per triangle: aerodynamic force scale (0 = no air force, 1 = full);
- ``film:yield_angle`` per hinge edge: elastic range of the hinge angle about its rest angle [rad] (<= 0 = elastic);
- ``film:j2_yield``, ``film:j2_hardening``, ``film:j2_modulus``, ``film:j2_thickness``, ``film:hinge_length`` per hinge
  edge: a J2 metal sheet's plane-strain section (σ_y' > 0 turns it on; E', H' [Pa], t [m]) and the hinge length
  l_h = (A1 + A2)/|e| [m] (sheet_j2.py).

Scene-wide numbers (air, iteration counts) are ``FilmConfig`` fields.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton

NAMESPACE = "film"
STRAIN_CAP = f"{NAMESPACE}:strain_cap"
COMPRESS_CAP = f"{NAMESPACE}:compress_cap"
LIMIT_KE = f"{NAMESPACE}:limit_ke"          # N/m, the film's full in-plane stiffness E h / (1 - nu^2) (stress.py)
AIR = f"{NAMESPACE}:air"
YIELD_ANGLE = f"{NAMESPACE}:yield_angle"
J2_YIELD = f"{NAMESPACE}:j2_yield"           # Pa, the section's plane-strain yield σ_y' (0 = no J2 section)
J2_HARDENING = f"{NAMESPACE}:j2_hardening"   # Pa, H'
J2_MODULUS = f"{NAMESPACE}:j2_modulus"       # Pa, E' (E' t³/12 = the hinge's D)
J2_THICKNESS = f"{NAMESPACE}:j2_thickness"   # m
HINGE_LENGTH = f"{NAMESPACE}:hinge_length"   # m, l_h = (A1 + A2)/|e| (κ = θ / l_h; emit.hinge_length)
TEAR_FORCE = f"{NAMESPACE}:tear_force"     # N per particle: Γ·h/2, a seam's break force at a tip (seams.py)
# A seam also tears where the sheet next to it is stretched this far (the topology controller's strain test on its
# triangles). The film is inextensible at its strain cap (~yield strain, 1.4 % for kraft), so stretch past the cap
# only appears where the projection cannot meet a demand: a sheet pulled apart past its extent (kinematic grips), or a
# fast fold (12 % at a trouser tip at 16 sweeps). Paper's own elongation at break (1.6-2.8 %, failure.fracture_strain)
# is inside that noise, so it is not used on this route. Measured: docs/notes/paper-pack.md (tear lines).
SEAM_OVERLOAD_STRAIN = 0.25

# Air at 20 C, 1 atm (Engineering ToolBox, sources/clips/sc8/film_aero.md).
AIR_DENSITY = 1.204          # kg/m^3
AIR_KINEMATIC_VISCOSITY = 1.506e-5   # m^2/s


@dataclass(frozen=True)
class FilmConfig:
    """Scene-wide film numbers. Air: normal pressure drag 0.5·rho·Cd·A·|v_n|·v_n per triangle (Cd of a flat plate
    broadside, 1.28 for a finite plate, NASA Glenn) plus tangential skin friction 0.5·rho·Ct·A·|v_t|·v_t (Ct =
    2 sides x Blasius 1.328/sqrt(Re) at Re ~ 1e4: 0.027; see the design note)."""

    strain_iterations: int = 16      # Jacobi sweeps of the strain-limit projection per step
    strain_relaxation: float = 1.0   # over-relaxation of the averaged per-vertex correction
    mass_ratio_cap: float = 0.0      # > 0: projection weights clamped to this max/min mass ratio (not momentum-safe)
    jacobi: str = "momentum"         # "momentum" (conserving, strain.constraint_scale) or "vertex" (averaging)
    air_density: float = AIR_DENSITY
    drag_normal: float = 1.28        # Cd, flat plate normal to the flow
    drag_tangent: float = 0.027      # Ct, both faces
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)   # m/s, uniform
    drag_cap: bool = True            # clamp each vertex's drag impulse to its relative speed (never reverses it)
    ledger: bool = True              # per-world drag work and projection energy (float64, see ledger.py)
    # pinned film pieces start VBD's iterations at x + h·v instead of the inertial target (Newton's
    # particle_pinned_piece_start, WP11): with a finite iteration count the inertial start leaves an h² sag bias, so
    # FILM-1's cantilever read 62 deg at 0.625 ms (target 41.5); with it, 42 deg at any dt (item 8, scr/WP4.md)
    pinned_piece_start: bool = True
    # free film components take A17's exact per-component translation step (SolverVBDMomentum, WP11): a falling film
    # keeps its momentum (FILM-4b 0.65 -> 1.05 m/s); pinned or contacting components keep the stock iteration
    momentum: bool = True
    # a seam's force is averaged over this time before the topology controller's tear test (seams.seam_force): a
    # landing sheet's twins touch down a fraction of a micron apart, and one substep's correction between them read
    # 0.58 N (a 60 g/m2 sheet dropped 2 cm on a table tore 3 edge seams); a pull that tears paper lasts far longer
    seam_force_tau: float = 0.01     # s; 0 = no averaging
    # film vertices held by rigid contacts (SCR-WP13-GRIP-1, grip.py; C16's strap slip): the projection's tangential
    # shift of a contact vertex is limited by its friction (holds below mu·N, slips by the excess), and contacts at
    # exactly zero slip get the friction stiffness Newton's VBD drops there. Free film is unaffected (bitwise)
    projection_friction: bool = True
    static_friction_hessian: bool = True
    # film vertices at a static shape (skin.py; docs/known_gaps/film-thin-sheet-box-edge.md): a vertex outside its
    # contact skin may enter it by at most this fraction of its contact radius per VBD iteration (Newton's step from
    # outside overshot), and the strain projection may press a lightly touching vertex at most 10 % of the skin deep
    # (it pulled a taut sheet over a box edge). 50 um LDPE over a box went INVALID without both. 0.05 left a vertex
    # 25 um in the skin, ~50 N at the box's gains, and the friction on it tore the sheet at a corner. None = off.
    # Opt-in (0.002, the measured value) for the 2026-11-09 release: on, a label rolled onto a tube at 1.25 mm cells
    # read no roller load (Bonds C-1); it goes on by default once that law re-passes (physics ruling 2026-11-08)
    contact_skin_entry: float | None = None
    # the coupler's exchange caps (alloy.coupling.stiffness_cap, SCR-SC4-4d) on the rigid bodies coupled to this film,
    # when set: (c, c_d) = the near-contact ke sum <= c·m/dt_x², kd sum <= c_d·m/dt_x per body (None: that cap off;
    # None for both: none). A SchedulerPolicy that sets either cap wins. (4.0, 0.5) stops most launches off the
    # open-carton peel (docs/known_gaps/paper-open-carton-peel.md: 1 in 8 worlds without, 1 in 40 with), and that task
    # sets them itself. Opt-in again since 2026-11-09: on by default (22248a172) the bag task's carton-off-the-mouth
    # GPU test went from 24 min to over 3 h on an L4 (tests/tasks/test_gift_bag_gpu.py)
    exchange_caps: tuple[float | None, float | None] | None = None
    # how the stretch limit acts. "projection" (the default): vertices are moved after the VBD solve (strain.py); the
    # tension is not a force, and a load much heavier than a film vertex is not held (a strip with 2-500 g on it
    # stood 11.5-34x past its cap). "stress": past its cap a triangle is an elastic membrane of the film's own
    # stiffness inside VBD's solve (stress.py): the tension is a force there, a load is held at the film's elastic
    # stretch, and no projection runs (no tear seams or compress caps yet). It has no yield: an overload keeps
    # stretching and is not flagged (5 kg on that strip, 2.2x its yield force: 3.8x past the cap after 2 s, still
    # stretching at 36 mm/s). Opt-in until every film law passes with it
    limiter: str = "projection"
    # "stress" only: a VBD step that takes a slack triangle past its cap is cut at this share of the cap past it
    # (VBD's Hessian has no stiffness for a slack triangle; uncut, a 2 g load rang 9x past the cap). 0 = no cut
    stress_entry: float = 0.05

    def __post_init__(self) -> None:
        if self.strain_iterations < 0:
            raise ValueError(f"FilmConfig.strain_iterations must be >= 0, got {self.strain_iterations}")
        if not 0.0 < self.strain_relaxation < 2.0:
            raise ValueError(f"FilmConfig.strain_relaxation must be in (0, 2), got {self.strain_relaxation}")
        if self.jacobi not in ("momentum", "vertex"):
            raise ValueError(f"FilmConfig.jacobi must be 'momentum' or 'vertex', got {self.jacobi!r}")
        if self.seam_force_tau < 0.0:
            raise ValueError(f"FilmConfig.seam_force_tau must be >= 0, got {self.seam_force_tau}")
        if self.contact_skin_entry is not None and not 0.0 < self.contact_skin_entry < 1.0:
            raise ValueError(f"FilmConfig.contact_skin_entry must be in (0, 1) or None, got {self.contact_skin_entry}")
        if self.air_density < 0.0 or self.drag_normal < 0.0 or self.drag_tangent < 0.0:
            raise ValueError("FilmConfig: air density and drag coefficients must be >= 0")
        if self.limiter not in ("projection", "stress"):
            raise ValueError(f"FilmConfig.limiter must be 'projection' or 'stress', got {self.limiter!r}")
        if not 0.0 <= self.stress_entry < 1.0:
            raise ValueError(f"FilmConfig.stress_entry must be in [0, 1), got {self.stress_entry}")
        if self.exchange_caps is not None and any(v is not None and v <= 0.0 for v in self.exchange_caps):
            raise ValueError(f"FilmConfig.exchange_caps must be > 0 or None, got {self.exchange_caps}")


def _attr(name: str, dtype: type, freq: newton.Model.AttributeFrequency, default: float):
    return newton.ModelBuilder.CustomAttribute(name=name, dtype=dtype, frequency=freq, namespace=NAMESPACE,
                                              default=default)


def register_custom_attributes(builder: newton.ModelBuilder) -> None:
    """model.film.{strain_cap, compress_cap, limit_ke, air} per triangle, model.film.{yield_angle, j2_*,
    hinge_length} per edge and model.film.tear_force per particle (module doc, seams.py). Idempotent (Newton accepts
    a re-registration with the same spec)."""
    T = newton.Model.AttributeFrequency.TRIANGLE
    E = newton.Model.AttributeFrequency.EDGE
    P = newton.Model.AttributeFrequency.PARTICLE
    for a in (
        _attr("strain_cap", wp.float32, T, 0.0),
        _attr("compress_cap", wp.float32, T, 1.0),
        _attr("limit_ke", wp.float32, T, 0.0),
        _attr("air", wp.float32, T, 0.0),
        _attr("yield_angle", wp.float32, E, 0.0),
        _attr("j2_yield", wp.float32, E, 0.0),
        _attr("j2_hardening", wp.float32, E, 0.0),
        _attr("j2_modulus", wp.float32, E, 0.0),
        _attr("j2_thickness", wp.float32, E, 0.0),
        _attr("hinge_length", wp.float32, E, 0.0),
        _attr("tear_force", wp.float32, P, math.inf),
    ):
        builder.add_custom_attribute(a)
