# SPDX-License-Identifier: Apache-2.0
"""Card types (SPEC §3.2; parsing, validation and the library are WP1).

Cards describe any object once in SI with sourced values (``Q``). Unknown keys are errors (WP1); every leaf left at
source "default" that changes routing or contact is listed in ``ResolvedObject.default_leaves``. Types here are
plain frozen dataclasses: no validation beyond ``Q``'s own rules, so WP1 can report every issue with its field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from alloy.core.ids import OMIT_IF_NONE
from alloy.core.quantity import Q, Vec3
from alloy.topology.types import TopologyBudget

# --- material ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    mu_static: Q
    mu_dynamic: Q
    restitution: Q
    torsional_lever: Q | None = None        # m; None -> Hertz estimate (derive.torsional_lever)
    rolling_lever: Q | None = None          # m; None -> 0 unless the shape rolls
    contact_stiffness: Q | None = None      # N/m, per MATERIAL, mass-independent; pairs combine in series (D22).
                                            # default 2e6 (series 1e6); precision pairs size K from need (§3.5)
    hydro_modulus: Q | None = None          # Pa (Elandt 2019)
    contact_damping_ratio: Q | None = None  # -, None -> derived from restitution: ζ = −ln e/√(π² + ln²e)


@dataclass(frozen=True)
class ClothAniso:
    stretch_warp: Q
    stretch_weft: Q
    shear: Q                                # N/m
    bend_warp: Q
    bend_weft: Q                            # N·m


@dataclass(frozen=True)
class Elastic:
    youngs: Q
    poisson: Q
    model: Literal["linear", "corotated", "snh"] = "snh"
    rayleigh_beta: Q | None = None          # s
    cloth: ClothAniso | None = None
    # N·m: a cloth's bending rigidity (KES B where measured, else derived from E at the card's fitted thickness). The
    # cloth route takes D from it, not E·h³ at the sheet's thickness (then the contact thickness only); left out of
    # digests while None (R2-E1)
    bending: Q | None = field(default=None, metadata=OMIT_IF_NONE)
    # N (N·m/m): a cloth's bending hysteresis, KES 2HB, the width of its bending loop. The cloth route's hinges then
    # hold a fold by fibre friction of HB = 2HB/2 (alloy/solvers/vbd/hysteresis.py); left out of digests while None
    bending_hysteresis: Q | None = field(default=None, metadata=OMIT_IF_NONE)


@dataclass(frozen=True)
class Plastic:
    yield_stress: Q | None = None
    yield_pressure: Q | None = None         # Pa
    friction_angle: Q | None = None         # rad (present => granular)
    cohesion: Q | None = None               # Pa
    hardening: Q | None = None              # -, Newton MPM's ξ (paste / granular routes)
    dilatancy: Q | None = None              # -
    # "j2": a solid that yields (von Mises, linear isotropic hardening; alloy/solvers/plastic/j2.py). It keeps its
    # solid route (rigid, soft, sheet, rod) and yield_stress is its uniaxial initial yield σ_y0; without it a
    # yield_stress makes a paste (SCR-WP7-15)
    model: Literal["j2"] | None = field(default=None, metadata=OMIT_IF_NONE)
    hardening_modulus: Q | None = field(default=None, metadata=OMIT_IF_NONE)  # Pa, H in σ_y = σ_y0 + H ε̄p (j2)


@dataclass(frozen=True)
class Fluid:
    """A liquid or paste. ``viscosity`` is the Newtonian viscosity; for a material with ``plastic.yield_stress`` it is
    the Bingham plastic viscosity, so that yield stress + viscosity x rate gives the measured stress at the card's
    stated reference rate (an apparent viscosity there would count the yield stress twice). ``consistency`` K
    (Pa·s^n; a unit string converts as a viscosity, time in s) and ``flow_index`` n add Herschel-Bulkley
    tau = tau_y + K gdot^n (n < 1 thins, n > 1 thickens) on the MPM paste route; ``viscosity`` stays required as the
    starting value and the fallback for solvers without it. Known gap: the thinning is too weak today (n 0.4 behaves
    as about 0.7; docs/known_gaps/paste-shear-thinning.md)."""

    viscosity: Q                            # Pa·s
    surface_tension: Q | None = None        # N/m
    adhesion: Q | None = None               # - (Akinci 2013 coefficient)
    sound_speed: Q | None = None            # m/s (WCSPH only; unused by DFSPH)
    consistency: Q | None = None            # Pa·s^n, Herschel-Bulkley K (paste route; with flow_index)
    flow_index: Q | None = None             # -, Herschel-Bulkley n (n < 1 shear-thinning)


@dataclass(frozen=True)
class Failure:
    fracture_strain: Q | None = None        # -
    fracture_stress: Q | None = None        # Pa
    tear_energy: Q | None = None            # J/m^2
    cut_min_force: Q | None = None          # N
    cut_min_strain: Q | None = None         # - (depth gate, GOTCHAS:564)


@dataclass(frozen=True)
class Film:
    """Film-only values for route ``cloth_film`` (alloy/solvers/film): each field optional, the film plug-in's
    ``FILM_DEFAULTS`` (alloy/solvers/film/materials.py) fills the rest."""
    dead_fold_yield: Q | None = None        # rad, hinge elastic range about rest: the dead-fold return angle
    contact_radius: Q | None = None         # m, collision radius of film particles
    air_scale: Q | None = None              # -, aerodynamic force scale (0 = no air force)
    crease_ratio: Q | None = None           # -, L*/h: a fold line (mesh OBJ `l` lines) bends as D/(ratio·h) per m
    crease_yield: Q | None = None           # rad, elastic range of a crease hinge about its rest angle


@dataclass(frozen=True)
class MaterialCard:
    name: str
    density: Q                              # kg/m^3, volumetric
    surface: Surface
    elastic: Elastic | None = None          # None = rigid-only
    plastic: Plastic | None = None
    fluid: Fluid | None = None
    failure: Failure | None = None
    # route cloth_film only (alloy/solvers/film); a None block is left out of digests, so cards without it keep
    # their card_hash (WP1 ruling, 2026-11-06)
    film: Film | None = field(default=None, metadata=OMIT_IF_NONE)
    # the material whose sourced pair cards this one takes where it has none of its own (alloy.Scene's per-object
    # materials: mass, density or restitution set on one object; alloy.scene.pairs.pair_card); None is left out
    pairs_from: str | None = field(default=None, metadata=OMIT_IF_NONE)
    schema: Literal["alloy.material/1"] = "alloy.material/1"


@dataclass(frozen=True)
class PairCard:
    a: str
    b: str                                  # material names, unordered
    mu_static: Q
    mu_dynamic: Q
    restitution: Q | None = None
    torsional_lever: Q | None = None
    single_mu: Literal["dynamic", "static"] = "dynamic"
    schema: Literal["alloy.pair/1"] = "alloy.pair/1"

    @property
    def key(self) -> tuple[str, str]:
        """Order-independent identity ``(min(a, b), max(a, b))``; library file ``<key[0]>__<key[1]>.yaml``."""
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


# --- shape sources ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Primitive:
    kind: Literal["box", "sphere", "capsule", "cylinder", "cone", "plane", "ellipsoid"]
    size: tuple[float, ...]                 # half-extents / radius / (radius, half-height), m


@dataclass(frozen=True)
class MeshSource:
    path: str
    sha256: str
    scale: Vec3 = (1.0, 1.0, 1.0)
    collision: Literal["convex", "decompose", "sdf", "mesh"] = "convex"
    max_parts: int = 32


@dataclass(frozen=True)
class SDFSource:
    path: str | None = None
    analytic_id: str | None = None
    voxel: float = 1e-4


@dataclass(frozen=True)
class TetSource:
    path: str | None = None
    from_mesh: str | None = None
    edge_length: float = 5e-3


@dataclass(frozen=True)
class CurveSource:
    points: tuple[Vec3, ...]
    radius: float


@dataclass(frozen=True)
class ParticleSource:
    region: tuple[Vec3, Vec3]
    spacing: float
    grain_diameter: float | None = None


@dataclass(frozen=True)
class USDSource:
    path: str
    prim: str


@dataclass(frozen=True)
class MJCFSource:
    path: str
    body: str | None = None                 # MJCF flex is converted by Alloy (Newton skips it)


ShapeSource = Primitive | MeshSource | SDFSource | TetSource | CurveSource | ParticleSource | USDSource | MJCFSource
SHAPE_SOURCES: tuple[type, ...] = (Primitive, MeshSource, SDFSource, TetSource, CurveSource, ParticleSource,
                                   USDSource, MJCFSource)

# --- object -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadEnvelope:
    grip_force: Q = Q(20.0, "default")      # N, local squeeze (flat-punch check) only
    bend_force: Q = Q(2.0, "default")       # N, whole-body bending load for the rod and sheet checks
    contact_speed: Q = Q(0.5, "default")    # m/s
    tolerance: Q = Q(1e-4, "default")       # m
    design_force: Q = Q(50.0, "default")    # N, precision-pair peak force for K sizing (§3.5 L2)


@dataclass(frozen=True)
class Crease:
    a: Vec3
    b: Vec3                                 # crease line endpoints, body frame, m
    stiffness: Q                            # N·m/rad


@dataclass(frozen=True)
class Features:
    thickness: Q | None = None
    creases: tuple[Crease, ...] = ()
    self_contact: bool = False
    passage_width: Q | None = None          # m
    precision_anchor: tuple[float, float, float] | None = None   # body-frame task origin (F2)
    min_wall: Q | None = None               # m, thinnest wall (CCD, SPH thin-wall flag)
    bend_stiffness: Q | None = None         # N·m², measured composite EI (cables); overrides E·πr⁴/4 (SCR-WP1-7)
    twist_stiffness: Q | None = None        # N·m², measured composite GJ; overrides G·πr⁴/2 (SCR-WP1-7)
    chain_links: int | None = None          # CHAIN route link-count override, >= 3 (SCR-DC3-7; None = sag rule)


@dataclass(frozen=True)
class Budget:
    collision_parts: int = 32
    tets: int = 5000
    particles: int = 200_000


@dataclass(frozen=True)
class BladeSpec:                            # marks a cutting edge on this object (topology, D19)
    edge_a: Vec3
    edge_b: Vec3                            # body frame, m
    edge_radius: float = 1.0e-4
    softness: float = 3.0
    cut_spring_ke: float = 500.0


@dataclass(frozen=True)
class PrecisionFeature:                     # a feature of this object that forms precision pairs
    name: str                               # e.g. "bore", "thread", "plug_shell"
    crop: tuple[Vec3, Vec3]                 # shape frame, m
    mode: Literal["hydroelastic", "sdf"] = "sdf"
    voxel: float | None = None              # None -> validity rule (§3.5)
    analytic_id: str | None = None          # e.g. "iso_thread:M10x1.5:6g" -> analytic SDF (threads)
    clearance: Q | None = None              # diametral clearance to its partner feature, m


@dataclass(frozen=True)
class RouteOverride:
    route: str
    reason: str


@dataclass(frozen=True)
class ObjectCard:
    name: str
    shape: ShapeSource
    material: str
    part_materials: Mapping[str, str] = field(default_factory=dict)
    motion: Literal["dynamic", "kinematic", "static"] = "dynamic"
    loads: LoadEnvelope = LoadEnvelope()
    features: Features = Features()
    precision_partners: tuple[str, ...] = ()            # "object" or "object:feature"
    precision_features: tuple[PrecisionFeature, ...] = ()
    budget: Budget = Budget()
    topology: TopologyBudget | None = None              # spares reserved by the assembler (alloy.topology.types)
    blade: BladeSpec | None = None
    route: RouteOverride | None = None
    schema: Literal["alloy.object/1"] = "alloy.object/1"


# --- scene ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Randomization:
    field: str                              # dotted card path
    lo: float
    hi: float                               # SI


@dataclass(frozen=True)
class Placement:
    object: str
    pose: tuple[float, ...]                 # (x, y, z, qx, qy, qz, qw)
    randomize: tuple[Randomization, ...] = ()


@dataclass(frozen=True)
class RobotRef:
    path: str
    kind: Literal["urdf", "mjcf", "usd"]
    base_pose: tuple[float, ...]
    name: str
    contact_gap: float | None = None        # m, contact gap of the file's shapes; None = RIGID_GAP (5 mm) as every
                                            # scene object (a geom's own MJCF gap is kept); Newton's default is 0.1 m


@dataclass(frozen=True)
class Attachment:                           # weld/joint/spring between two placed objects (source for bonds)
    a: str
    b: str                                  # "object" or "object:link"
    kind: Literal["weld", "joint", "spring"] = "weld"
    break_force: Q | None = None            # N or N·m; None = unbreakable
    break_strain: Q | None = None           # -
    anchor: tuple[float, ...] | None = None  # (x, y, z, qx, qy, qz, qw) in a's frame
    timeconst: Q | None = None              # s; weld/joint only: equality solref time constant (None = WP2 sizes it)
    dampratio: Q | None = None              # -; weld/joint only: equality solref damping ratio (SCR-DC3-3)


@dataclass(frozen=True)
class EmitterSpec:                          # SPH emitter or sink
    kind: Literal["emitter", "sink"]
    material: str                           # fluid material (emitter)
    pose: tuple[float, ...]
    half_extents: Vec3
    rate: float = 0.0                       # m^3/s (emitter)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    start_s: float = 0.0
    stop_s: float | None = None


@dataclass(frozen=True)
class SceneCard:
    robots: tuple[RobotRef, ...]
    objects: tuple[Placement, ...]
    pairs: tuple[PairCard, ...] = ()
    attachments: tuple[Attachment, ...] = ()
    emitters: tuple[EmitterSpec, ...] = ()
    worlds: int = 1
    control_dt: float = 0.02                # s, policy period; all dts are control_dt / 2^j (D4)
    combine: Literal["harmonic", "max"] = "harmonic"
    # light slender free bodies get armature up to m·r0² (alloy.contact.inertia_floor); False turns it off. None
    # (unset) = on, and is left out of digests, so scenes without it keep their hashes (coordinator 2026-11-08)
    inertia_floor: bool | None = field(default=None, metadata=OMIT_IF_NONE)
    schema: Literal["alloy.scene/1"] = "alloy.scene/1"


@dataclass(frozen=True)
class ResolvedObject:                       # what the router and emitters consume
    card: ObjectCard
    material: MaterialCard
    part_materials: Mapping[str, MaterialCard]
    card_hash: str                          # digest_of(resolved card with defaults filled)
    extents: tuple[float, float, float]     # OBB l1 >= l2 >= l3 [m]
    worst_case: Mapping[str, float]         # randomized fields at the routing-worst end of their range
    default_leaves: tuple[str, ...]         # dotted paths still at source "default"


@dataclass(frozen=True)
class CardIssue:
    """One validation finding (WP1 ``validate_*``); ``field`` is the dotted path of the leaf."""

    level: Literal["error", "warn"]
    card: str
    field: str
    message: str


Card = MaterialCard | PairCard | ObjectCard | SceneCard
SCHEMAS: Mapping[str, type] = {
    "alloy.material/1": MaterialCard, "alloy.pair/1": PairCard,
    "alloy.object/1": ObjectCard, "alloy.scene/1": SceneCard,
}
