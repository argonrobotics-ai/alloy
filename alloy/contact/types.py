# SPDX-License-Identifier: Apache-2.0
"""Contact-layer types (SPEC §3.5; the layer, precision pipeline, sidecar logic and GapMonitor are WP3)."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.defaults import PAIR_MAX_PEN_M, PAIR_S1_CAP_M
from alloy.core.quantity import Vec3



class ContactClass(enum.IntEnum):
    NATIVE_MJ = 0
    RIGID = 1
    PRECISION_SDF = 2
    PRECISION_HYDRO = 3
    SOFT_PARTICLE = 4
    SOFT_FEATURE = 5
    CROSS_PARTICLE = 6


class ContactFlags(enum.IntFlag):
    SPECULATIVE = 1
    SWEPT = 2
    PATCH_ANCHOR = 4
    NEW = 8
    PERSISTENT = 16
    GAP_VIOLATION = 32


@dataclass(frozen=True)
class PrecisionPairSpec:
    shape_a: int
    shape_b: int                                # template shape ids (labels resolved by the assembler)
    mode: Literal["hydroelastic", "sdf"] = "hydroelastic"
    voxel_size: float = 5.0e-5                  # m; 1e-4 allowed when feature extent > 40 mm
    narrow_band: tuple[float, float] = (-5.0e-4, 5.0e-4)
    crop_a: tuple[Vec3, Vec3] | None = None     # shape frame, m
    crop_b: tuple[Vec3, Vec3] | None = None
    texture_format: Literal["uint16", "float32"] = "uint16"
    sampler: Literal["software", "hardware"] = "software"
    margin: float = 0.0
    gap: float = 5.0e-4
    speculative_gap_max: float = 2.0e-3
    max_penetration: float | None = None        # None -> min(5e-5, 0.5·clearance/2) (D2); S2 = 2x, S1 = min(5e-4, clearance/2)
    kh: float = 1.0e11                          # Pa/m (hydroelastic; checked against the pressure-law closed form)
    pair_stiffness: float | None = None         # N/m, sdf mode, force space; None -> 2·design_force/max_penetration (D22)
    mj_imp: float = 0.99                        # flat solimp imp = d0 = dmax on precision rows (D22)
    mj_cone: Literal["elliptic", "pyramidal"] = "elliptic"
    mj_impratio: float = 300.0                  # static hold / creep (§3.5 creep rule)
    ccd: Literal["speculative", "sweep", "off"] = "speculative"
    reduce: Literal["moment_matched", "off"] = "moment_matched"
    match_threshold: float | None = None        # None = voxel_size
    mesh_chord_tol: float = 5.0e-6
    clearance: float | None = None              # m, diametral, from the cards (validity rules)
    wall_thickness: float | None = None
    analytic_id: str | None = None              # analytic SDF (threads: helical ISO 68-1 profile) instead of a texture
    escalation: Literal["flag", "substep", "halt"] = "flag"

    @property
    def radial_clearance(self) -> float | None:
        """Half the diametral clearance [m], or None."""
        return None if self.clearance is None else 0.5 * self.clearance

    def resolved_max_penetration(self) -> float:
        """D2: ``max_penetration`` if set, else ``min(5e-5, 0.5·radial clearance)`` (5e-5 without a clearance)."""
        if self.max_penetration is not None:
            return self.max_penetration
        radial = self.radial_clearance
        return PAIR_MAX_PEN_M if radial is None else min(PAIR_MAX_PEN_M, 0.5 * radial)

    def s2_limit(self) -> float:
        """D3: DEGRADED / bully S2 penetration = the max gate ``2·max_penetration`` [m]."""
        return 2.0 * self.resolved_max_penetration()

    def s1_limit(self) -> float:
        """D3: INVALID / bully S1 penetration = ``min(5e-4, radial clearance)`` [m]."""
        radial = self.radial_clearance
        return PAIR_S1_CAP_M if radial is None else min(PAIR_S1_CAP_M, radial)

    def resolved_match_threshold(self) -> float:
        return self.voxel_size if self.match_threshold is None else self.match_threshold


@dataclass(frozen=True, eq=False)
class GapReport:
    pair_p99_m: np.ndarray                                  # [pairs]
    pair_max_m: np.ndarray                                  # [pairs]
    pair_samples: np.ndarray                                # [pairs] int, samples in the p99 population
    world_max_m: np.ndarray                                 # [worlds]
    tunnel_events: np.ndarray                               # [pairs, worlds] int (swept test)
    histogram_um: np.ndarray                                # [pairs, 2000] 1-µm bins to 2 mm
    reference: tuple[str, ...]                              # per pair: analytic or winding_number

    HISTOGRAM_BINS = 2000

    @classmethod
    def empty(cls, pairs: int, worlds: int) -> GapReport:
        """A report with no samples (fakes, scenes without precision pairs)."""
        return cls(np.zeros(pairs), np.zeros(pairs), np.zeros(pairs, np.int64), np.zeros(worlds),
                   np.zeros((pairs, worlds), np.int64), np.zeros((pairs, cls.HISTOGRAM_BINS), np.int64),
                   tuple("analytic" for _ in range(pairs)))


@dataclass(eq=False)
class AlloyContactSidecar:
    """Alloy extras indexed by the rigid contact slot of the layer's ``newton.Contacts`` (length rigid_contact_max
    unless noted): cls int8, pair int32 (-1), flags uint8, world int32, area float32 [m^2], toi float32 [0,1]
    (1 = none), weight float32 (sdf-mode area weight, D22), stream_count int32[8]; xp_particles vec2i, xp_normal
    vec3, xp_distance float32, xp_world int32 (length xp_contact_max). Forces use Newton's ``rigid_contact_force``
    (requested_attributes={"force"}), filled by MixedContactBinding.precision_forces. Plug-ins read the sidecar
    through the layer handle (ContactLayerLike.sidecar), never through the Contacts object.

    WP0 owns the type and ``allocate``; ``from_pipeline`` and ``assemble`` delegate to ``alloy.contact.sidecar``
    (WP3) so the logic lives in WP3's file.
    """

    cls: wp.array            # [C] int8 (ContactClass)
    pair: wp.array           # [C] int32, -1 = not a precision pair
    flags: wp.array          # [C] uint8 (ContactFlags)
    world: wp.array          # [C] int32
    area: wp.array           # [C] float32, m^2
    toi: wp.array            # [C] float32 in [0, 1], 1 = none
    weight: wp.array         # [C] float32
    stream_count: wp.array   # [8] int32
    xp_particles: wp.array   # [X] vec2i
    xp_normal: wp.array      # [X] vec3
    xp_distance: wp.array    # [X] float32
    xp_world: wp.array       # [X] int32

    STREAMS = 8

    @classmethod
    def allocate(cls, rigid_contact_max: int, *, xp_contact_max: int = 0,
                 device: wp.context.Device | str | None = None) -> AlloyContactSidecar:
        """Zeroed sidecar arrays (pair = -1, toi = 1) for ``rigid_contact_max`` slots."""
        c, x = int(rigid_contact_max), int(xp_contact_max)
        return cls(
            cls=wp.zeros(c, dtype=wp.int8, device=device), pair=wp.full(c, -1, dtype=wp.int32, device=device),
            flags=wp.zeros(c, dtype=wp.uint8, device=device), world=wp.zeros(c, dtype=wp.int32, device=device),
            area=wp.zeros(c, dtype=wp.float32, device=device), toi=wp.full(c, 1.0, dtype=wp.float32, device=device),
            weight=wp.zeros(c, dtype=wp.float32, device=device),
            stream_count=wp.zeros(cls.STREAMS, dtype=wp.int32, device=device),
            xp_particles=wp.zeros(x, dtype=wp.vec2i, device=device), xp_normal=wp.zeros(x, dtype=wp.vec3, device=device),
            xp_distance=wp.zeros(x, dtype=wp.float32, device=device), xp_world=wp.zeros(x, dtype=wp.int32, device=device),
        )

    @property
    def capacity(self) -> int:
        return int(self.pair.shape[0])

    @classmethod
    def from_pipeline(cls, pipeline: newton.CollisionPipeline, *, xp_contact_max: int = 0, world_count: int = 1,
                      per_world_capacity: int | None = None) -> tuple[newton.Contacts, AlloyContactSidecar]:
        """Contacts with the pipeline's own flags (per_contact_shape_properties=True, contact_matching,
        requested_attributes ∪ {"force"}) plus a matching sidecar. Implemented by WP3 (``alloy.contact.sidecar``)."""
        from alloy.contact.sidecar import sidecar_from_pipeline  # WP3

        return sidecar_from_pipeline(cls, pipeline, xp_contact_max=xp_contact_max, world_count=world_count,
                                     per_world_capacity=per_world_capacity)

    def assemble(self, contacts: newton.Contacts) -> None:
        """Merge streams in fixed order NATIVE, RIGID, PRECISION, CROSS. Implemented by WP3."""
        from alloy.contact.sidecar import assemble_streams  # WP3

        assemble_streams(self, contacts)


AlloyContacts = newton.Contacts            # alias kept for readability in signatures
