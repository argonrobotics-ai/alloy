# SPDX-License-Identifier: Apache-2.0
"""Scene-assembly types (SPEC §3.4; ``SceneAssembler``, emitters, the pair-table kernel and ``Simulation`` are WP2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, State

from alloy.cards.types import BladeSpec, EmitterSpec
from alloy.contact.types import PrecisionPairSpec
from alloy.coupling.types import PairPolicy
from alloy.route.types import RouteDecision
from alloy.topology.types import BondSet


@dataclass(frozen=True)
class IndexRanges:
    body: range
    shape: range
    joint: range
    particle: range
    tri: range
    tet: range
    edge: range

    @classmethod
    def empty(cls) -> IndexRanges:
        r = range(0)
        return cls(r, r, r, r, r, r, r)


@dataclass(frozen=True)
class ObjectHandle:
    name: str
    card_hash: str
    decision: RouteDecision
    template: IndexRanges                   # in the world template
    world_stride: IndexRanges               # per-world offset after replicate
    blade: BladeSpec | None = None          # the card's cutting edge, in its one body's frame (SCR-SC5-1)


@dataclass(frozen=True)
class EntrySpec:
    name: str                               # entry name in SolverCoupled
    capability: str                         # solver capability name
    bodies: tuple[int, ...]
    particles: tuple[int, ...]
    joints: tuple[int, ...]
    shapes: tuple[int, ...]
    substeps: int                           # initial rung (power of two)
    target_dt: float                        # [s] D4 target
    max_substeps: int = 1                   # construction rung (D21); adaptive entries use the ladder max
    solver_kwargs: Mapping[str, Any] = field(default_factory=dict)   # njmax/nconmax (PackSpec), VBD iterations,
                                                                     # MPM voxel, integrator, solimp, impratio, ...


@dataclass(frozen=True)
class CouplerChoice:
    kind: Literal["none", "proxy", "admm"]
    pairs: tuple[PairPolicy, ...]           # alloy.coupling.types
    strong_unrouted: tuple[tuple[str, str], ...] = ()


@dataclass
class RouteReport:
    decisions: dict[str, RouteDecision]
    default_leaves: dict[str, tuple[str, ...]]
    unsupported: list[tuple[str, str]]      # (card field, reason) values that cannot reach the Model
    unsourced_pairs: list[tuple[str, str]]
    scene_effects: list[str]
    warnings: list[str]
    k_real: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)  # pair -> (declared, realised) N/m
    ladder: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)      # pair -> available rungs (§3.5)
    entry_options: list[str] = field(default_factory=list)   # MuJoCo options from MJCF <option> and route overrides
    # body -> (smallest principal inertia, its floor m·r0²) kg·m², the bodies alloy.contact.inertia_floor raised
    inertia_floor: dict[str, tuple[float, float]] = field(default_factory=dict)
    rigid_step: Any = None      # alloy.scene.rigid_step.RigidStep: the MuJoCo entry's step and why (SCR-WP0-RIGIDDT)

    @classmethod
    def empty(cls) -> RouteReport:
        return cls({}, {}, [], [], [], [])


@dataclass(eq=False)
class PairTable:
    """Per-material-pair contact values, written into ``Contacts`` every step by ``apply`` (D13, D22).

    kernel: rigid_contact_friction = μ_pair / μ_mix(a, b) (a scale, D13); rigid_contact_stiffness = K;
    rigid_contact_damping = kd = 2ζ(e)·√(K·m_eff), ζ = −ln e/√(π² + ln²e), e corrected by e_correction (m_eff from
    current-pose body inverse mass along the normal). Solvers without restitution_path list e as unsupported in
    readback. ``apply`` is implemented by WP2 (``alloy.scene.pairs.apply_pair_table``).

    SolverMuJoCo rows use kd = 2ζ(e)·√(K·m_eff)/√(1−imp) with the row's flat solimp ``shape_imp`` (SCR-WP2-3).
    Zero means "unset" in Newton's per-contact arrays: μ_pair = 0 writes scale 1e-12, e = 1 writes kd = 1e-12.
    Optional arrays (SCR-WP2-1): ``world_table``/``world_stiffness`` hold per-world randomized values (else
    ``table``/``stiffness``); ``override_slot`` [S] (−1 or the precision-shape slot) + ``override_k`` [P, P] hold
    the per-shape-pair precision K (D22, 0 = no override).
    """

    materials: tuple[str, ...]              # index = alloy:*_material id
    table: wp.array                         # [M, M] vec4 (mu_s, mu_d, e, torsional_lever), symmetric
    stiffness: wp.array                     # [M, M] float32 series K (D22), N/m; precision pairs overridden per shape pair
    shape_material: wp.array                # [S] int32
    particle_material: wp.array             # [P] int32
    shape_mu_mix: wp.array                  # [S] float32, the consuming solver's own μ input (D13)
    mu_mix_rule: int                        # from SolverCapability.friction_mix_code
    e_correction: wp.array                  # [n_e bins] float32, implicitfast numerical-damping correction (A6)
    model: newton.Model | None = None       # body inverse mass/inertia for m_eff, shape_world for per-world rows
    shape_imp: wp.array | None = None       # [S] float32 flat solimp per shape (MuJoCo damping correction)
    world_table: wp.array | None = None     # [W, M, M] vec4 per-world randomized surface values
    world_stiffness: wp.array | None = None  # [W, M, M] float32
    override_slot: wp.array | None = None   # [S] int32, -1 or the precision-shape slot of the template shape
    override_k: wp.array | None = None      # [P, P] float32 precision-pair K, 0 = no override

    def apply(self, contacts: Contacts, state: State | None = None) -> None:
        """Write friction scale, stiffness and damping per contact (WP2 kernel). With ``state`` m_eff includes the
        current-pose rotational terms; without it, translational 1/(inv_m_a + inv_m_b) only (SCR-WP2-1)."""
        from alloy.scene.pairs import apply_pair_table  # WP2

        apply_pair_table(self, contacts, state=state)

    def material_index(self, name: str) -> int:
        """Index of material ``name`` in ``materials`` (``alloy:*_material`` id)."""
        return self.materials.index(name)


@dataclass(eq=False)
class BuiltScene:
    model: newton.Model
    entries: list[EntrySpec]
    coupler: CouplerChoice
    handles: dict[str, ObjectHandle]
    pair_table: PairTable
    precision_pairs: tuple[PrecisionPairSpec, ...]
    control_dt: float
    world_count: int
    world_id_offset: int                    # global id of local world 0 (RNG keys, D9)
    world_origin: wp.array                  # [W] vec3d, render/export only (F1)
    bonds: BondSet | None                   # from SceneCard.attachments + pre-split lines (D19)
    spares: Mapping[str, int]               # reserved spare particles/vertices/tets per world (TopologyBudget)
    emitters: tuple[EmitterSpec, ...]
    report: RouteReport
    model_digest: str
    world_ids: wp.array | None = None       # [W] int32 global ids when not consecutive (repacked packs, SCR-WP8-2)

    def global_world_id(self, local: int) -> int:
        """Global world id of local world ``local`` (D9: RNG keys and digests use global ids): ``world_ids[local]``
        when set, else ``world_id_offset + local``."""
        if not 0 <= local < self.world_count:
            raise IndexError(f"local world {local} outside [0, {self.world_count})")
        if self.world_ids is not None:
            return int(self.world_ids.numpy()[local])
        return self.world_id_offset + local

    @property
    def rigid_dt(self) -> float:
        """The MuJoCo entry's rigid step [s] (``alloy.scene.rigid_step``, SCR-WP0-RIGIDDT): chosen at assembly, or
        today's step (snap_to_grid(1/600, control_dt)) for a scene assembled without a choice."""
        from alloy.scene.rigid_step import today_step  # noqa: PLC0415

        step = getattr(self.report, "rigid_step", None)
        return float(step.dt) if step is not None else today_step(self.control_dt)
