# SPDX-License-Identifier: Apache-2.0
"""Packing, structure and speed-ladder types (SPEC §3.9; logic WP8)."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import warp as wp

from alloy.cards.types import SceneCard
from alloy.core.protocols import SimulationLike
from alloy.scene.types import BuiltScene


@dataclass(frozen=True)
class WorldProfile:
    scene_hash: str
    solver_config_hash: str
    newton_commit: str
    gpu_model: str
    dt: float
    nv: int
    n_particles: int
    island_nv_max: int
    chain_len_max: int
    rows_p99: int
    contacts_p99: int
    pairs_peak: int
    t_floor_s: float
    c_w_s: float
    iters_p50: float
    iters_p99: float
    bytes_fixed: int
    bytes_per_world: int


@dataclass(frozen=True)
class PackPolicy:
    plateau_fraction: float = 0.90
    mem_reserve_fraction: float = 0.10
    max_dense_h_fraction: float = 0.25
    latency_lane: bool = True
    latency_fill_occupancy: float = 0.70
    slow_lane_iters_ratio: float = 3.0
    big_world_bytes_fraction: float = 0.8
    capacity_headroom: float = 1.5
    pairs_headroom: float = 1.25


@dataclass(frozen=True)
class WorldRequest:
    scene: SceneCard                        # packs are built through SceneAssembler, never a raw builder_fn
    count: int
    mode: Literal["throughput", "latency"]
    profile: WorldProfile


@dataclass(frozen=True)
class PackSpec:
    device: str
    world_ids: tuple[int, ...]
    n: int
    stream_priority: int
    kind: Literal["throughput", "latency", "big_world"]
    rigid_contact_max: int
    njmax: int
    nconmax: int
    shape_pairs_max: int
    predicted_bytes: int


@dataclass(frozen=True)
class PackPlan:
    packs: tuple[PackSpec, ...]
    predicted_throughput: float             # sim-s per GPU-s, all devices


class StructureClass(enum.Enum):
    DENSE_SMALL = "dense_small"
    DENSE_BLOCKED = "dense_blocked"
    TREE_SPARSE = "tree_sparse"
    CHAIN_PCR = "chain_pcr"
    MATRIX_FREE_COLORED = "matrix_free_colored"


@dataclass(frozen=True)
class IslandPlan:
    island: int
    nv: int
    depth: int
    chain_len: int
    cls: StructureClass
    available: bool                         # a shipped solve exists for this class


@dataclass(frozen=True)
class StructurePlan:
    islands: tuple[IslandPlan, ...]
    parallel_unit: Literal["world", "island", "color"]
    recommended_solver: str
    unavailable: tuple[str, ...]            # classes with no shipped solve yet (upstream)


@dataclass(eq=False)
class Pack:
    spec: PackSpec
    built: BuiltScene
    sim: SimulationLike
    world_ids: wp.array                     # [n] int32 global ids (BuiltScene.world_id_offset + local)


@dataclass(frozen=True, eq=False)
class PackStats:                            # host, per pack, over a window
    pack: int
    wall_s: float
    sim_s: float
    awake_fraction: np.ndarray              # [W]
    iters_p50: float
    iters_p99: float
    contacts_p99: int
    rows_p99: int
    overflow: int
    world_iters_p99: np.ndarray | None = None  # [W] per-world iterations p99 (repack slow lane, parallel §4.5); WP8


@dataclass(frozen=True, eq=False)
class Trace:                                # a replayable control input (WP13 converts recorded traces)
    scene_hash: str
    control_dt: float
    inputs: Mapping[str, np.ndarray]        # control field -> [T, W?, n] SI, one row per control period
    source: str                             # e.g. "bench/traces/r2_t12 (mjb → MJCF → SceneCard)"
    sha256: str
    columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # input key -> column names (actuators,
                                                                          # joints): replays bind by name (SCR-WP13-1)
    initial: Mapping[str, np.ndarray] = field(default_factory=dict)       # start state, keyed like inputs ("qpos",
                                                                          # "qvel", "mujoco.qacc_warmstart")

    @property
    def periods(self) -> int:
        """Number of control periods T (0 for an empty trace)."""
        return min((int(v.shape[0]) for v in self.inputs.values()), default=0)


LadderId = Literal["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]


@dataclass(frozen=True)
class LadderRung:
    id: LadderId
    scene: str
    sweep: Mapping[str, tuple[int, ...]]                       # e.g. {"objects": (1,4,16,64,100)}
    worlds: tuple[int, ...] = (1, 4, 16, 64, 256, 1024, 4096)
