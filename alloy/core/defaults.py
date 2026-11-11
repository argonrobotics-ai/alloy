# SPDX-License-Identifier: Apache-2.0
"""Shared numeric defaults (SPEC §4.2, §4.3). Policies in several packages take their defaults from here, so the
numbers that must agree (``CheckPolicy`` vs ``BullyLimits``, D3) agree by construction."""

from __future__ import annotations

from typing import Final

GRAVITY: Final[float] = 9.81                    # m/s^2, Newton default (0, 0, -9.81), Z up
CONTROL_DT: Final[float] = 0.02                 # s, SceneCard.control_dt default (D4)
DT_X_MAX: Final[float] = 2.0e-3                 # s, exchange-step bound (coupling-time §4.5)

PEN_LIMIT_M: Final[float] = 1.0e-3              # S2 / DEGRADED penetration (D3)
PEN_S1_M: Final[float] = 1.0e-2                 # S1 / INVALID penetration (D3)
PAIR_MAX_PEN_M: Final[float] = 5.0e-5           # precision-pair max_penetration cap (D2)
PAIR_S1_CAP_M: Final[float] = 5.0e-4            # precision-pair S1 cap: min(5e-4, radial clearance) (D3)
JOINT_LIMIT_FLOOR_M: Final[float] = 1.0e-3      # JOINT_FAIL floor (§4.3)
JOINT_LIMIT_FLOOR_RAD: Final[float] = 1.0e-3

SPEED_FAIL_MPS: Final[float] = 20.0             # any body or active particle (D12)
V_MAX_FREE_MPS: Final[float] = 10.0             # free body without robot contact, bully S1 (D12)
ENERGY_JUMP_REL: Final[float] = 1.0e-2          # per step, of E_scale, net of W_ext (D20)
ENERGY_S1_REL: Final[float] = 0.5               # cumulative unexplained gain, of E_scale (D20)
REST_DISP_MAX_M: Final[float] = 5.0e-5          # bully non-target displacement (S2)
E_SCALE_FLOOR_J: Final[float] = 1.0e-3          # E_scale = max(E0 - PE_min, m g L_char, 1e-3 J) (D20)

CREEP_PRECISION: Final[float] = 1.0e-6          # m/s, 0.01 mm per 10 s (§4.2)
CREEP_GENERAL: Final[float] = 1.0e-5            # m/s, 0.1 mm per 10 s (D5)

RIGID_GAP_M: Final[float] = 5.0e-3              # builder gap, never Newton's 0.1 m (contact §1.2)
PRECISION_GAP_M: Final[float] = 5.0e-4
MATERIAL_CONTACT_STIFFNESS: Final[float] = 2.0e6  # N/m per material; two in series = 1e6 (D22)

FRAME_WARN_M: Final[float] = 4.0                # |p| warn (precision §4.1)
FRAME_FAIL_M: Final[float] = 16.0               # |p| fail under the precision preset
