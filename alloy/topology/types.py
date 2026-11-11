# SPDX-License-Identifier: Apache-2.0
"""Topology-change types (SPEC §3.8, D19; logic WP10). Topology is fixed at assembly: tears and cuts break bonds."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from alloy.core.quantity import Vec3


class BondKind(enum.IntEnum):
    RIGID_WELD = 0
    JOINT = 1
    SPRING = 2
    CLOTH_EDGE = 3
    TET_CUT = 4


@dataclass(frozen=True)
class TopologyBudget:
    """Spares the assembler reserves per world for an object (``ObjectCard.topology``)."""

    spare_particles: int = 0
    spare_vertices: int = 0
    spare_tets: int = 0
    max_events_per_step: int = 64
    split_spacing: float | None = None   # [m] regular pre-split pattern; None = finest that fits spare_vertices (WP10)
    cut_model: Literal["seams", "copies"] = "seams"   # soft (tet) objects: "seams" = pre-split seams with
    # twin bonds (the plan above); "copies" = virtual-node copies, cuttable along any blade path, no spares
    # (alloy.topology.copies_model; coordinator GO 2026-11-06T19:36Z). Cloth supports "seams" only.


@dataclass
class BondSet:
    """Device arrays, length B over all worlds. ``a``/``b`` index bodies, joints or particles by ``kind``."""

    a: wp.array                  # [B] int32
    b: wp.array                  # [B] int32
    kind: wp.array               # [B] int8 (BondKind)
    world: wp.array              # [B] int32
    break_force: wp.array        # [B] float32, N or N·m (inf = unbreakable)
    break_strain: wp.array       # [B] float32, - (inf = none)
    damage: wp.array             # [B] float32 in [0, 1]
    intact: wp.array             # [B] int8
    cut_strain: wp.array | None = None   # [B] float32 TET_CUT blade gate, Failure.cut_min_strain (inf = none; WP10)
    cut_force: wp.array | None = None    # [B] float32 TET_CUT blade gate, Failure.cut_min_force [N] (inf = none)

    @property
    def count(self) -> int:
        return int(self.a.shape[0])


@dataclass(frozen=True)
class Blade:
    body: int
    shape: int
    edge_a: Vec3
    edge_b: Vec3                  # body frame, m
    edge_radius: float = 1.0e-4
    softness: float = 3.0
    cut_spring_ke: float = 500.0


@dataclass(frozen=True, eq=False)
class TopologyLedger:
    """Host copy, one entry per world, for the last period."""

    events: np.ndarray            # [W] count this period
    mass_before: np.ndarray       # [W] kg
    mass_after: np.ndarray        # [W] kg
    momentum_before: np.ndarray   # [W, 6]
    momentum_after: np.ndarray    # [W, 6]
    overflow: np.ndarray          # [W]
    reset_required: np.ndarray    # [W] bool; Scheduler treats as a per-world reset

    @classmethod
    def empty(cls, world_count: int) -> TopologyLedger:
        """A ledger with no events (used by fakes and by scenes without a controller)."""
        z = np.zeros(world_count)
        z6 = np.zeros((world_count, 6))
        return cls(np.zeros(world_count, np.int64), z.copy(), z.copy(), z6.copy(), z6.copy(),
                   np.zeros(world_count, np.int64), np.zeros(world_count, bool))
