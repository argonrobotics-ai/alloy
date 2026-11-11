# SPDX-License-Identifier: Apache-2.0
"""Bully-agent types (SPEC §3.11; B0-B2, monitors, corpus and replay are WP7)."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from alloy.cards.types import SceneCard
from alloy.core.defaults import (ENERGY_JUMP_REL, ENERGY_S1_REL, PEN_LIMIT_M, PEN_S1_M, REST_DISP_MAX_M,
                                 SPEED_FAIL_MPS, V_MAX_FREE_MPS)
from alloy.core.protocols import SimulationLike
from alloy.precision.types import HealthReport


@dataclass(frozen=True, eq=False)
class ActionSpace:
    low: np.ndarray                         # per actuator, SI (rad, m, N)
    high: np.ndarray
    rate_hz: float
    max_delta_per_tick: np.ndarray
    initial: np.ndarray | None = None       # WP7: command at reset; the rate limit starts here (None: first tick free)

    @property
    def dim(self) -> int:
        return int(np.asarray(self.low).shape[0])


class BullySeverity(enum.IntEnum):
    S1 = 1
    S2 = 2
    S3 = 3


@dataclass(frozen=True)
class BullyLimits:
    """Identical numbers to ``CheckPolicy`` (both default from ``alloy.core.defaults``); §4.3 is the single table."""

    v_max: float = V_MAX_FREE_MPS
    pen_limit: float = PEN_LIMIT_M          # precision pairs: per pair (D3)
    e_gain_s2_rel: float = ENERGY_JUMP_REL  # of E_scale, net of W_ext: per tick (D20)
    e_gain_s1_rel: float = ENERGY_S1_REL    # cumulative (D20)
    rest_disp_max: float = REST_DISP_MAX_M
    joint_sep_s2: float = 1e-3
    pen_s1: float = PEN_S1_M                # WP7: S1 penetration (PEN_FAIL), = CheckPolicy.pen_s1_m
    speed_fail: float = SPEED_FAIL_MPS      # WP7: any body / active particle (SPEED_FAIL), = CheckPolicy.speed_fail_mps
    flight_margin: float = 0.5              # m/s, slack of the ballistic envelope for contact-free flight (SCR-WP7-7)


@dataclass(frozen=True, eq=False)
class EpisodeResult:                           # used by run_episode
    severity: np.ndarray                       # [W] 0 none, else BullySeverity
    first_step: np.ndarray                     # [W] int, -1 none
    monitors: Mapping[str, np.ndarray]         # monitor -> [W] worst value
    health: HealthReport
    badness: np.ndarray                        # [W] b (§3.11)


@dataclass(frozen=True)
class BullyTarget:
    name: str
    scene: SceneCard | None
    make_sim: Callable[[int, str], SimulationLike]            # (world_count, device)
    actions: ActionSpace
    non_target_bodies: tuple[int, ...] = ()
    thin_parts: Mapping[int, float] = field(default_factory=dict)
    action_kind: str = "joint_target_q"     # WP7: joint_target_q | joint_target_qd | joint_f | body_f | body_pose | passive
    action_index: tuple[int, ...] = ()      # WP7: template coord/dof ids (body_f: 6*body + component); () = 0..A-1
    stress_extra: Callable[[BullyTarget], list[np.ndarray]] | None = None   # WP7: target-specific B0 traces
    robot_bodies: tuple[int, ...] | None = None   # WP7: template bodies the actions drive (None: inferred from gains)


@dataclass(frozen=True)
class SearchConfig:
    worlds: int = 4096
    generations: int = 20
    knots: int = 8
    horizon_s: float = 2.0
    sigma0: float = 0.3                       # fraction of the action range
    budget_gpu_h: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class Finding:
    id: str                                   # "BUL-0001"
    severity: BullySeverity
    monitor: str
    value: float
    threshold: float
    first_step: int
    scene_hash: str
    alloy_sha: str
    newton_sha: str
    repro: str
    repeat_of: str | None = None
    target: str = ""                        # WP7: registered bully target name (Corpus.replay rebuilds it)


@dataclass(frozen=True)
class ReplayResult:
    finding: str
    trips: int
    runs: int
    device: str
    counts: bool                            # 10/10 on CPU, >= 9/10 on the same GPU type
