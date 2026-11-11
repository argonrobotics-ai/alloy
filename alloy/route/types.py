# SPDX-License-Identifier: Apache-2.0
"""Router types (SPEC §3.3; the ``Router`` and physics checks are WP2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

from alloy.cards.types import ResolvedObject


class Route(StrEnum):
    STATIC = "static"
    RIGID = "rigid"
    RIGID_COMPLIANT = "rigid_compliant"
    RIGID_PILE = "rigid_pile"
    CHAIN = "chain"
    ROD_VBD = "rod_vbd"
    PANELS = "panels"
    CLOTH_VBD = "cloth_vbd"
    CLOTH_STYLE3D = "cloth_style3d"
    CLOTH_FILM = "cloth_film"           # SC8 thin-film plug-in (alloy.solvers.film); opt-in by a card route override
    SOFT_VBD = "soft_vbd"
    GRAIN_RIGID = "grain_rigid"
    GRAIN_MPM = "grain_mpm"
    PASTE_MPM = "paste_mpm"
    LIQUID_SPH = "liquid_sph"
    LIQUID_MPM = "liquid_mpm"


APPROXIMATE_ROUTES: Final[frozenset[Route]] = frozenset({Route.LIQUID_MPM})   # D15: fails LIQ-2 by construction
NO_ENTRY_ROUTES: Final[frozenset[Route]] = frozenset({Route.STATIC})          # shapes in the shared pipeline only
#: SPH particle spacing [m] per ``liquid_accuracy`` (user decision 2026-11-08, SCR-WP9-LIQRES): 4 mm by default,
#: 2 mm "super" (8x the particles, about 16x the cost); the particle radius is half the spacing.
LIQUID_SPACING: Final[Mapping[str, float]] = {"default": 4.0e-3, "super": 2.0e-3}


@dataclass(frozen=True)
class RouterConfig:
    rigid_ratio: float = 0.1               # delta <= rigid_ratio * tolerance -> RIGID
    rod_sag_rigid: float = 0.01            # gamma <= 0.01 also required for rigid rods
    chain_links_min: int = 3              # fewest links a CHAIN gets (SCR-DC3-7 dropped the old floor of 30)
    chain_links_max: int = 40             # required links above this -> ROD_VBD (D8)
    grain_rigid_max: int = 2000            # grains per world
    mpm_voxel_grain_min: float = 3.0       # MPM voxel must be >= 3 grain diameters (continuum validity)
    mpm_voxel_max: float = 5e-3            # m
    liquid_viscosity_max: float = 1.0      # Pa·s
    sph_radius_max: float = 2e-3           # m: unused by the router since SCR-WP9-LIQRES (r = LIQUID_SPACING / 2)
    sph_passage_diameters: float = 8.0     # passage_width / (2r) >= 8 (LIQ-6): refined below it, or the build warns
    liquid_accuracy: Literal["default", "super"] = "default"   # SPH spacing LIQUID_SPACING[...] (SCR-WP9-LIQRES)
    cloth_accuracy: Literal["default", "super", "fast"] = "default"   # CLOTH_VBD tagged (scene.cloth_accuracy)
    sph_refine_passage: bool = False       # True: LIQ-6's old r = min(r, passage/16) (the DC show pins it)
    sph_viscous_above: float = 1e-2        # Pa·s; above -> weiler2018, else xsph
    pile_threshold: int = 50               # free bodies per world
    pile_backend: Literal["mujoco", "kamino", "steel"] = "mujoco"
    near_boundary_log10: float = 0.3       # within 2x of a threshold


@dataclass(frozen=True)
class SceneContext:
    rigid_partners: Mapping[str, tuple[str, ...]]   # object -> names of objects it may touch
    free_bodies_per_world: int
    registered_routes: frozenset[str]               # from alloy.core.registry.registered_routes()
    partners: Mapping[str, ResolvedObject] = field(default_factory=dict)   # cards of rigid_partners (SCR-WP2-4)
    needs_l1: bool = False                          # a precision pair in the scene needs gap-ladder L1 (D7)
    other_dynamic_entries: int = 0                  # D16: Style3D only with no other dynamic entry


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    entry: str                              # solver capability name ("mujoco", "vbd", "mpm", "sph", ...)
    numbers: Mapping[str, float]            # delta_task, gamma, bending_length, n_grains, k_compliant, ...
    margin: float                           # log10 distance to nearest threshold
    near_boundary: bool
    reasons: tuple[str, ...]
    scene_effects: tuple[str, ...] = ()     # e.g. "mujoco:integrator=discrete"
    overridden: bool = False
    self_test: Mapping[str, float] | None = None    # rod cantilever check results

    @property
    def approximate(self) -> bool:
        """True for routes flagged approximate (D15)."""
        return self.route in APPROXIMATE_ROUTES


@dataclass(frozen=True)
class RodCard:                              # derived by the router, never authored
    length: float
    radius: float
    density: float                          # m, m, kg/m^3
    EI: float
    GJ: float
    EA: float                               # N·m^2, N·m^2, N
    links: int
    damping_ratio: float = 0.02
    gamma: float = 0.0                      # ρAgL³/(EI) = 8·(tip sag / L) under self-weight
    needs_rod_rod_contact: bool = False
    self_contact: bool = False
    couples_to_vbd: bool = False
    integrator: Literal["discrete", "implicitfast"] = "discrete"
