# SPDX-License-Identifier: Apache-2.0
"""One world's tick on the CPU route in numpy: the fallback tick driver when WP2's C extension is missing (SCR-WP8-6).

The same sequence as ``alloy_mjc_tick`` (``alloy/route/_mjc/step.c``, ABI ``alloy_mjc.h``), per physics step:
``mj_step1`` | box-box line contacts (:mod:`alloy.route.mjc_core_line`, after collision here, inside it in C) |
``mj_fwdActuation`` | ``mj_fwdAcceleration`` | ``mj_fwdConstraint`` | scoped noslip
(:mod:`alloy.route.mjc_core_noslip`) | ``mj_sensorAcc`` | ``mj_checkAcc`` | ``mj_Euler`` or ``mj_implicit``, then the
per-step counters (non-finite or MuJoCo-reset steps, deepest contact, fastest dof) and the optional trajectory row.
After the tick: the contact export (``mj_contactForce``, contact frame) and the state pull (``mj_kinematics`` /
``mj_comPos`` / ``mj_comVel``, then body_q ``x y z qx qy qz qw`` and body_qd ``[COM linear, angular]`` in world frame,
in ``body_map`` order). With every plug-in off a tick equals ``mj_step`` bit for bit.

:class:`WorldOut` holds one world's buffers and counters with the field names of WP2's ``MjcOut``, so the route reads
either the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from alloy.route.mjc_core_line import LineRule
from alloy.route.mjc_core_noslip import ScopedNoslipNumpy

_BAD = (int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL),
        int(mujoco.mjtWarning.mjWARN_BADQACC))
_FULL = (int(mujoco.mjtWarning.mjWARN_CONTACTFULL), int(mujoco.mjtWarning.mjWARN_CNSTRFULL))
_EULER = int(mujoco.mjtIntegrator.mjINT_EULER)
COUNTERS = ("steps", "nonfinite", "contact_overflow", "pen_max", "qvel_max", "noslip_slots", "noslip_sweeps",
            "noslip_overflow", "line_fired")


@dataclass
class WorldOut:
    """One world's export buffers (``ncon_max`` contacts, ``nbody`` body rows, ``traj_cap`` trajectory rows) and
    counters, as WP2's ``alloy.route._mjc.abi.MjcOut``."""

    ncon_max: int = 0
    nbody: int = 0
    traj_cap: int = 0
    ncon: int = 0
    counts: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))

    def __post_init__(self) -> None:
        n, b, t = self.ncon_max, self.nbody, self.traj_cap
        self.geom = np.zeros((n, 2), np.int32)
        self.pos, self.frame, self.dist, self.force = np.zeros((n, 3)), np.zeros((n, 9)), np.zeros(n), np.zeros((n, 6))
        self.body_q, self.body_qd = np.zeros((b, 7)), np.zeros((b, 6))
        self.traj_body_q, self.traj_body_qd = np.zeros((t, b, 7)), np.zeros((t, b, 6))

    def counters(self) -> dict[str, float]:
        return dict(self.counts)

    def zero(self) -> None:
        self.counts = dict.fromkeys(COUNTERS, 0)


@dataclass
class NumpyPlugins:
    """One model's numpy plug-ins: scoped noslip (None = off) and the line rule (None = off)."""

    noslip: ScopedNoslipNumpy | None
    line: LineRule | None
    body_map: np.ndarray          # [nbody_out] MuJoCo body id per Newton body row (-1: unwritten)
    export_contacts: bool = True
    pull_state: bool = True


def _bad_warnings(d: mujoco.MjData) -> int:
    return sum(int(d.warning[w].number) for w in _BAD)


def arena_full(d: mujoco.MjData) -> tuple[int, int]:
    """MuJoCo C's arena-full warning counts (contacts, constraint rows): what it dropped since the data's reset."""
    return int(d.warning[_FULL[0]].number), int(d.warning[_FULL[1]].number)


def _step(m: mujoco.MjModel, d: mujoco.MjData, p: NumpyPlugins, out: WorldOut) -> None:
    bad_before = _bad_warnings(d)
    mujoco.mj_step1(m, d)
    if p.line is not None:
        out.counts["line_fired"] += p.line.apply(m, d)
    mujoco.mj_fwdActuation(m, d)
    mujoco.mj_fwdAcceleration(m, d)
    mujoco.mj_fwdConstraint(m, d)
    if p.noslip is not None:
        c = p.noslip.counters
        before = (c.slots, c.sweeps, c.overflow)
        p.noslip.run(m, d)
        out.counts["noslip_slots"] += c.slots - before[0]
        out.counts["noslip_sweeps"] += c.sweeps - before[1]
        out.counts["noslip_overflow"] += c.overflow - before[2]
    d.flg_rnepost = 0
    mujoco.mj_sensorAcc(m, d)
    mujoco.mj_checkAcc(m, d)
    if int(m.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_FWDINV):
        mujoco.mj_compareFwdInv(m, d)
    if int(m.opt.integrator) == _EULER:
        mujoco.mj_Euler(m, d)
    else:
        mujoco.mj_implicit(m, d)
    _count(d, bad_before, out)


def _count(d: mujoco.MjData, bad_before: int, out: WorldOut) -> None:
    qpos, qvel, qacc = np.asarray(d.qpos), np.asarray(d.qvel), np.asarray(d.qacc)
    bad = _bad_warnings(d) != bad_before or not (np.isfinite(qpos).all() and np.isfinite(qvel).all()
                                                 and np.isfinite(qacc).all())
    c = out.counts
    c["steps"] += 1
    c["nonfinite"] += int(bad)
    if qvel.size:
        c["qvel_max"] = max(c["qvel_max"], float(np.max(np.abs(qvel))))
    if d.ncon:
        c["pen_max"] = max(c["pen_max"], float(-np.min(d.contact.dist[: d.ncon])), 0.0)


def tick(m: mujoco.MjModel, d: mujoco.MjData, p: NumpyPlugins, nstep: int, out: WorldOut) -> None:
    """``nstep`` physics steps of one world with the plug-ins, then export and pull (module docstring)."""
    traj = out.traj_cap > 0
    if traj and out.traj_cap < nstep:
        raise ValueError(f"trajectory buffer holds {out.traj_cap} steps, the tick runs {nstep}")
    for k in range(nstep):
        _step(m, d, p, out)
        if traj:
            pull_state(m, d, p.body_map, out.traj_body_q[k], out.traj_body_qd[k])
    if p.export_contacts and out.ncon_max > 0:
        export_contacts(m, d, out)
    if p.pull_state:
        pull_state(m, d, p.body_map, out.body_q, out.body_qd)


def plain_tick(m: mujoco.MjModel, d: mujoco.MjData, p: NumpyPlugins, nstep: int, out: WorldOut) -> None:
    """rk4 / discrete entries: ``mj_step`` (no mid-step plug-ins), then the same counters, export and pull."""
    for k in range(nstep):
        bad_before = _bad_warnings(d)
        mujoco.mj_step(m, d)
        _count(d, bad_before, out)
        if out.traj_cap > 0:
            pull_state(m, d, p.body_map, out.traj_body_q[k], out.traj_body_qd[k])
    if p.export_contacts and out.ncon_max > 0:
        export_contacts(m, d, out)
    if p.pull_state:
        pull_state(m, d, p.body_map, out.body_q, out.body_qd)


def export_contacts(m: mujoco.MjModel, d: mujoco.MjData, out: WorldOut) -> None:
    """The world's contacts after the tick: geoms, pos, frame, dist and ``mj_contactForce`` (contact frame)."""
    n = min(int(d.ncon), out.ncon_max)
    out.counts["contact_overflow"] += int(d.ncon) - n
    out.ncon = n
    if n == 0:
        return
    c = d.contact
    out.geom[:n] = c.geom[:n]
    out.pos[:n] = c.pos[:n]
    out.frame[:n] = c.frame[:n]
    out.dist[:n] = c.dist[:n]
    for i in range(n):
        mujoco.mj_contactForce(m, d, i, out.force[i])


def pull_state(m: mujoco.MjModel, d: mujoco.MjData, body_map: np.ndarray, q: np.ndarray, qd: np.ndarray) -> None:
    """body_q / body_qd rows in ``body_map`` order after FK (the step leaves xpos one step stale)."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    rows = np.flatnonzero(body_map >= 0)
    b = body_map[rows]
    q[rows, :3] = d.xpos[b]
    q[rows, 3:6] = d.xquat[b, 1:]
    q[rows, 6] = d.xquat[b, 0]
    cv = d.cvel[b]
    dif = d.xipos[b] - d.subtree_com[m.body_rootid[b]]
    ang, lin = cv[:, :3], cv[:, 3:]
    cros = np.stack([dif[:, 1] * ang[:, 2] - dif[:, 2] * ang[:, 1], dif[:, 2] * ang[:, 0] - dif[:, 0] * ang[:, 2],
                     dif[:, 0] * ang[:, 1] - dif[:, 1] * ang[:, 0]], axis=1)
    qd[rows, :3] = lin - cros      # mj_objectVelocity: lin - dif x ang at the COM, world orientation
    qd[rows, 3:] = ang


__all__ = ["COUNTERS", "NumpyPlugins", "WorldOut", "arena_full", "export_contacts", "plain_tick", "pull_state", "tick"]
