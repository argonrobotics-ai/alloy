# SPDX-License-Identifier: Apache-2.0
"""Offline cause attribution for replay deviations (CPU MuJoCo, offline only; README ground rule).

The Newton replay differs from the recording through several known gaps; each one can be reproduced on the
converted model in CPU MuJoCo by changing one setting, which measures how much of the deviation that gap explains:

- ``noslip_off``: pinned mujoco-warp 3.16.0 has no noslip solver (its ``types.Option`` has no ``noslip_*`` field)
  and Newton's ``SolverMuJoCo`` exposes none, while every robot recording ran ``noslip_iterations`` = 50.
- ``margins_zeroed``: Newton zeroes geom/pair margins for NATIVECCD/MULTICCD (MJWarp #2106,
  newton@ac7f7830:newton/_src/solvers/mujoco/solver_mujoco.py:3499-3511, 4679-4688).
- ``multiccd_off``: Newton disables MULTICCD unless asked (solver_mujoco.py:776).
- ``nudge``: a 1e-9 rad offset on the first hinge (1e-9 m on the first free body without hinges), the chaos floor
  of findings/timestep.md:324-347.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from alloy.tasks.mjb_convert import RecordedTrace
from alloy.tasks.mjb_verify import replay_cpu
from alloy.tasks.mjmodel_info import body_radius, free_bodies


def rotation_angle(q: np.ndarray, q_ref: np.ndarray, scalar_last: bool) -> np.ndarray:
    """Angle of ``q_ref⁻¹·q`` [rad] as ``2·atan2(|r_xyz|, |r_w|)`` on float64-normalised quaternions.

    ``2·acos(|q·q_ref|)`` is ill-conditioned at small angles: float32 quaternion rounding (~6e-8) alone reads as
    ~1e-3 rad (r1_t14 long spatula, a 0.31 mm "error" at r = 0.3 m)."""
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    if not scalar_last:
        q, q_ref = np.roll(q, -1, axis=-1), np.roll(q_ref, -1, axis=-1)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q_ref = q_ref / np.linalg.norm(q_ref, axis=-1, keepdims=True)
    w = np.sum(q * q_ref, axis=-1)                               # scalar part of conj(q_ref)·q
    v = (q_ref[..., 3:4] * q[..., :3] - q[..., 3:4] * q_ref[..., :3]
         - np.cross(q_ref[..., :3], q[..., :3]))  # fmt: skip
    return 2.0 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w))


def pose_errors(p: np.ndarray, q: np.ndarray, p_ref: np.ndarray, q_ref: np.ndarray, radius: np.ndarray,
                scalar_last: bool = False) -> np.ndarray:  # fmt: skip
    """Per object ``(|Δp|, θ, |Δp| + r·θ)`` for poses [..., n, 3] / quaternions [..., n, 4] (same convention)."""
    dp = np.linalg.norm(np.asarray(p, dtype=np.float64) - p_ref, axis=-1)
    ang = rotation_angle(q, q_ref, scalar_last)
    return np.stack([dp, ang, dp + radius * ang])


def qpos_object_errors(mjm: mujoco.MjModel, qpos: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Object errors [3, S, n_obj] between MuJoCo qpos samples [S, nq] and reference rows [S, nq]."""
    objs = free_bodies(mjm)
    radius = np.array([body_radius(mjm, b) for b, _, _ in objs])
    cols = np.array([qa for _, qa, _ in objs], dtype=np.int64)
    pos = cols[:, None] + np.arange(3)
    quat = cols[:, None] + 3 + np.arange(4)
    return pose_errors(qpos[:, pos], qpos[:, quat], ref[:, pos], ref[:, quat], radius)


def _noslip_off(m: mujoco.MjModel) -> None:
    m.opt.noslip_iterations = 0


def _margins_zeroed(m: mujoco.MjModel) -> None:
    m.geom_margin[:] = 0.0
    m.pair_margin[:] = 0.0


def _multiccd_off(m: mujoco.MjModel) -> None:
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)


VARIANTS: dict[str, Callable[[mujoco.MjModel], None]] = {
    "noslip_off": _noslip_off,
    "margins_zeroed": _margins_zeroed,
    "multiccd_off": _multiccd_off,
}


@dataclass(frozen=True)
class VariantResult:
    variant: str
    max_point_err: float                  # m, worst object over the window
    first_over_1mm: float | None          # s


def _nudged(rec: RecordedTrace, eps: float) -> RecordedTrace:
    """``eps`` added to the first hinge angle (rad), or to the first free body's x (m) in scenes without hinges."""
    state0 = rec.state0.copy()
    jt = rec.model.jnt_type
    hinges = np.nonzero(jt == int(mujoco.mjtJoint.mjJNT_HINGE))[0]
    k = int(hinges[0]) if len(hinges) else int(np.nonzero(jt == int(mujoco.mjtJoint.mjJNT_FREE))[0][0])
    state0[1 + int(rec.model.jnt_qposadr[k])] += eps  # state0[0] is time, then qpos
    return RecordedTrace(rec.name, rec.root, rec.model, rec.inputs, state0, rec.ref_qpos, rec.meta, rec.ref_every)


def run_variant(m: mujoco.MjModel, rec: RecordedTrace, steps: int, variant: str) -> VariantResult:
    """Replay the converted model ``m`` with one gap applied and score it against the recording."""
    work = copy.copy(m)
    run_rec = rec
    if variant == "nudge":
        run_rec = _nudged(rec, 1e-9)
    elif variant != "baseline":
        VARIANTS[variant](work)
    q = replay_cpu(work, run_rec, steps)
    err = qpos_object_errors(m, q, rec.ref_qpos[: len(q)])[2]
    worst = err.max(axis=1, initial=0.0)
    over = np.nonzero(worst > 1e-3)[0]
    first = float((over[0] + 1) * rec.ref_every * rec.dt) if len(over) else None
    return VariantResult(variant, float(worst.max(initial=0.0)), first)


def attribute(m: mujoco.MjModel, rec: RecordedTrace, steps: int) -> list[VariantResult]:
    """Baseline, chaos nudge and every single-gap variant."""
    return [run_variant(m, rec, steps, v) for v in ("baseline", "nudge", *VARIANTS)]
