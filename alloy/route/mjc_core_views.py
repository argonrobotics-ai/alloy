# SPDX-License-Identifier: Apache-2.0
"""Numpy views between Newton's arrays and the CPU route's per-world MjData (SCR-WP8-6).

They replace Newton's per-step Warp push/pull of its C path (8 launches, 0.86 ms per step on T12, PROFILE.md §6):
- **control**: Newton's own control kernels (``_apply_mjc_control``: joint targets, ``mujoco.ctrl``, joint_f,
  body_f -> ``xfrc_applied``) write the Warp data's ``ctrl`` / ``qfrc_applied`` / ``xfrc_applied`` (CPU arrays), and
  each world's MjData takes its row. Same semantics as Newton, once per call (per tick on the fast path).
- **state in**: MuJoCo C keeps float64 ``qpos`` / ``qvel`` between steps. A Newton state is pushed (Newton's
  ``_update_mjc_data``, float32 -> float64) only into worlds whose ``joint_q`` / ``joint_qd`` differ from what the
  route last wrote: the first step, a reset, a snapshot restore, a user edit.
- **state out**: body_q / body_qd from MuJoCo's FK (float64, cast to Newton's float32), joint_q / joint_qd through
  Newton's ``convert_mj_coords_to_warp_kernel`` (free / ball quaternion order, kinematic joints from the input state).
- **contacts**: the tick's exported contacts into a Newton ``Contacts`` (shape ids, normal, body-local points, world
  force on shape 0, as ``SolverMuJoCo.update_contacts``) and into the Warp data's contact arrays (``nacon``, dist,
  geom, frame, worldid) plus ``solver_niter``, which Alloy's health monitor reads.
- **capacity** (fails closed, SCR-WP8-8): each world's ``mjw_data.overflow`` word gets MuJoCo Warp's bits for what the
  call dropped: ``NARROWPHASE`` for contacts (beyond the route's per-world export ``ncon_max``, MuJoCo C's contact
  arena full, beyond ``naconmax`` in the pooled health mirror, beyond a ``Contacts``' ``rigid_contact_max``) and
  ``NEFC`` for constraint rows (MuJoCo C's constraint arena full). Zeroed at the start of every call, as Newton zeroes
  it before every MuJoCo Warp step; HealthMonitor (CONTACT_OVERFLOW / ROW_OVERFLOW) and the bully's
  ``capacity_overflow`` row read it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from alloy._vendor.mujoco_warp import OverflowType


from alloy.core.worldops import joint_coord_world, joint_dof_world

CONTACT_BIT = int(OverflowType.NARROWPHASE)    # MuJoCo Warp: contacts beyond its capacity (health: CONTACT_OVERFLOW)
ROW_BIT = int(OverflowType.NEFC)               # MuJoCo Warp: constraint rows beyond njmax (health: ROW_OVERFLOW)


def _host(arr: Any) -> np.ndarray:
    """Zero-copy numpy view of a CPU Warp array."""
    return arr.numpy()


class NewtonViews:
    """The route's Newton-side views (module docstring) for ``nworld`` per-world MjData."""

    def __init__(self, solver: Any, nworld: int, body_globs: list[np.ndarray]) -> None:
        self.solver, self.model, self.nworld = solver, solver.model, nworld
        self.globs = body_globs
        d = solver.mjw_data
        self.ctrl, self.qfrc, self.xfrc = _host(d.ctrl), _host(d.qfrc_applied), _host(d.xfrc_applied)
        self.qpos, self.qvel = _host(d.qpos), _host(d.qvel)
        self.overflow = _host(d.overflow)
        self.coord_world = joint_coord_world(self.model).numpy() if self.model.joint_coord_count else None
        self.dof_world = joint_dof_world(self.model).numpy() if self.model.joint_dof_count else None
        self.last_q: np.ndarray | None = None
        self.last_qd: np.ndarray | None = None
        self.geom_shape = np.asarray(solver.mjc_geom_to_newton_shape.numpy()).reshape(nworld, -1)
        self.shape_body = self.model.shape_body.numpy().copy() if self.model.shape_count else np.zeros(0, np.int32)
        self._cviews: tuple[Any, dict[str, np.ndarray]] | None = None

    # ---- in ---------------------------------------------------------------------------------------------------------
    def push_control(self, datas: list[Any], state: Any, control: Any) -> None:
        self.solver._apply_mjc_control(self.model, state, control, self.solver.mjw_data)
        for w, d in enumerate(datas):
            d.ctrl[:] = self.ctrl[w]
            d.qfrc_applied[:] = self.qfrc[w]
            d.xfrc_applied[:] = self.xfrc[w]

    def changed_worlds(self, state: Any) -> np.ndarray:
        """Worlds whose joint_q / joint_qd differ from the route's last output (all on the first call)."""
        if self.last_q is None:
            return np.arange(self.nworld)
        bad = np.zeros(self.nworld, dtype=bool)
        for now, last, world in ((_host(state.joint_q), self.last_q, self.coord_world),
                                 (_host(state.joint_qd), self.last_qd, self.dof_world)):
            if now.size and not np.array_equal(now, last, equal_nan=True):
                diff = (now != last) & ~(np.isnan(now) & np.isnan(last))    # a frozen non-finite world is unchanged
                bad[np.unique(world[diff])] = True
        return np.flatnonzero(bad)

    def push_state(self, datas: list[Any], state: Any, worlds: np.ndarray) -> None:
        """Newton joint_q / joint_qd -> qpos / qvel of ``worlds`` (Newton's conversion)."""
        if worlds.size == 0:
            return
        self.solver._update_mjc_data(self.solver.mjw_data, self.model, state)
        for w in worlds.tolist():
            datas[w].qpos[:] = self.qpos[w]
            datas[w].qvel[:] = self.qvel[w]

    # ---- out --------------------------------------------------------------------------------------------------------
    def pull(self, datas: list[Any], outs: list[Any], state_out: Any, state_in: Any) -> None:
        if self.model.body_count:
            bq, bqd = _host(state_out.body_q), _host(state_out.body_qd)
            for w, out in enumerate(outs):
                g = self.globs[w]
                bq[g] = out.body_q
                bqd[g] = out.body_qd
        if self.model.joint_count:
            self.pull_joints(datas, state_out, state_in)

    def pull_joints(self, datas: list[Any], state_out: Any, state_in: Any) -> None:
        from alloy._vendor.newton._src.solvers.mujoco.kernels import convert_mj_coords_to_warp_kernel

        for w, d in enumerate(datas):
            self.qpos[w] = d.qpos
            self.qvel[w] = d.qvel
        s, m = self.solver, self.model
        jpw = m.joint_count // self.nworld
        attrs = getattr(m, "mujoco", None)
        wp.launch(convert_mj_coords_to_warp_kernel, dim=(self.nworld, jpw),
                  inputs=[s.mjw_data.qpos, s.mjw_data.qvel, jpw, m.joint_type, m.joint_q_start, m.joint_qd_start,
                          m.joint_dof_dim, m.joint_child, m.joint_X_p, m.joint_X_c, m.body_com,
                          getattr(attrs, "dof_ref", None) if attrs is not None else None, m.body_flags,
                          state_in.joint_q, state_in.joint_qd, s.mj_q_start, s.mj_qd_start],
                  outputs=[state_out.joint_q, state_out.joint_qd], device=m.device)
        self.remember(state_out)

    def poison(self, state_out: Any, worlds: np.ndarray) -> None:
        """NaN into the frozen ``worlds``' rows of ``state_out``, so health monitors and recorders see the blow-up
        (MuJoCo C may have stopped on a huge but finite value), and remember them as the route's output."""
        if worlds.size == 0:
            return
        if self.model.body_count:
            rows = np.concatenate([self.globs[w] for w in worlds.tolist()])
            _host(state_out.body_q)[rows] = np.nan
            _host(state_out.body_qd)[rows] = np.nan
        if self.coord_world is not None:
            _host(state_out.joint_q)[np.isin(self.coord_world, worlds)] = np.nan
        if self.dof_world is not None:
            _host(state_out.joint_qd)[np.isin(self.dof_world, worlds)] = np.nan
        self.remember(state_out)

    def remember(self, state: Any) -> None:
        self.last_q = _host(state.joint_q).copy()
        self.last_qd = _host(state.joint_qd).copy()

    # ---- contacts and health ----------------------------------------------------------------------------------------
    def gather(self, outs: list[Any], ncons: list[int]) -> dict[str, np.ndarray]:
        """Every world's exported contacts, pooled world after world."""
        sel = [(w, n) for w, n in enumerate(ncons) if n > 0]
        if not sel:
            return {}
        cat = {k: np.concatenate([getattr(outs[w], k)[:n] for w, n in sel]) for k in
               ("geom", "pos", "frame", "dist", "force")}
        cat["world"] = np.concatenate([np.full(n, w, np.int32) for w, n in sel])
        return cat

    def to_contacts(self, pooled: dict[str, np.ndarray], contacts: Any, state: Any, force: Any = None) -> np.ndarray:
        """Newton ``Contacts`` as ``SolverMuJoCo.update_contacts`` fills them (body-local points at ``state``); the
        world-frame force on shape 0 into ``contacts.force``, or into ``force`` (a CONTACT_F observable, zeroed
        first). Returns the worlds whose contacts did not fit ``rigid_contact_max``."""
        v = self._contact_views(contacts)
        f_out = _host(force) if force is not None else v.get("force")
        if force is not None:
            f_out[:] = 0.0
        n = min(int(contacts.rigid_contact_max), int(pooled["dist"].size)) if pooled else 0
        v["count"][0] = n
        contacts.n_contacts = n
        dropped = np.unique(pooled["world"][n:]) if pooled else np.zeros(0, np.int64)
        if n == 0:
            return dropped
        p = {k: a[:n] for k, a in pooled.items()}
        w, g = p["world"], p["geom"]
        s0, s1 = self.geom_shape[w, g[:, 0]], self.geom_shape[w, g[:, 1]]
        normal = p["frame"][:, :3]
        half = 0.5 * p["dist"][:, None] * normal
        v["shape0"][:n], v["shape1"][:n], v["normal"][:n] = s0, s1, normal
        body_q = _host(state.body_q) if state is not None else None
        v["point0"][:n] = self._body_local(s0, p["pos"] - half, body_q)
        v["point1"][:n] = self._body_local(s1, p["pos"] + half, body_q)
        if f_out is not None:
            F = p["frame"].reshape(n, 3, 3)
            top = np.einsum("ni,nij->nj", p["force"][:, :3], F)
            bot = np.einsum("ni,nij->nj", p["force"][:, 3:], F)
            f_out[:n] = -np.concatenate([top, bot], axis=1)
        return dropped

    def _contact_views(self, contacts: Any) -> dict[str, np.ndarray]:
        """Host views of a Contacts' arrays (cached for the last Contacts object, which the cache keeps alive)."""
        if self._cviews is None or self._cviews[0] is not contacts:
            names = {"count": "rigid_contact_count", "shape0": "rigid_contact_shape0",
                     "shape1": "rigid_contact_shape1", "normal": "rigid_contact_normal",
                     "point0": "rigid_contact_point0", "point1": "rigid_contact_point1"}
            views = {k: _host(getattr(contacts, a)) for k, a in names.items()}
            if getattr(contacts, "force", None) is not None:
                views["force"] = _host(contacts.force)
            self._cviews = (contacts, views)
        return self._cviews[1]

    def _body_local(self, shapes: np.ndarray, pts: np.ndarray, body_q: np.ndarray | None) -> np.ndarray:
        body = np.where(shapes >= 0, self.shape_body[np.maximum(shapes, 0)], -1)
        out = pts.copy()
        on = np.flatnonzero(body >= 0)
        if on.size and body_q is not None:
            X = body_q[body[on]].astype(np.float64)
            out[on] = _rotate_inv(X[:, 3:7], pts[on] - X[:, :3])
        return out

    def mirror_health(self, pooled: dict[str, np.ndarray], datas: list[Any]) -> np.ndarray:
        """The pooled contacts into the Warp data's contact arrays and each world's solver iterations into
        ``solver_niter`` (what Alloy's HealthMonitor reads); returns the worlds whose contacts fell beyond naconmax."""
        d = self.solver.mjw_data
        cap = int(d.naconmax)
        total = int(pooled["dist"].size) if pooled else 0
        n = min(cap, total)
        _host(d.nacon)[0] = n
        c = d.contact
        if n:
            _host(c.dist)[:n] = pooled["dist"][:n]
            _host(c.geom)[:n] = pooled["geom"][:n]
            _host(c.worldid)[:n] = pooled["world"][:n]
            _host(c.pos)[:n] = pooled["pos"][:n]
            _host(c.frame)[:n] = pooled["frame"][:n].reshape(n, 3, 3)
        niter = _host(d.solver_niter)
        for w, dd in enumerate(datas):
            niter[w] = int(dd.solver_niter[0])
        return np.unique(pooled["world"][n:]) if total > n else np.zeros(0, np.int64)

    # ---- capacity ---------------------------------------------------------------------------------------------------
    def set_overflow(self, bits: np.ndarray) -> None:
        """This call's per-world overflow words (module docstring, capacity)."""
        self.overflow[:] = bits

    def flag_overflow(self, worlds: np.ndarray, bit: int = CONTACT_BIT) -> None:
        """OR ``bit`` into ``worlds``' overflow words (contacts a later stage dropped)."""
        if worlds.size:
            self.overflow[worlds] |= bit


def overflow_bits(dropped_contacts: int, contact_full: int, cnstr_full: int) -> int:
    """One world's overflow word for a call: contacts dropped (export cap or MuJoCo C's contact arena) and constraint
    rows dropped (MuJoCo C's constraint arena), as counts over the call."""
    return (CONTACT_BIT if dropped_contacts > 0 or contact_full > 0 else 0) | (ROW_BIT if cnstr_full > 0 else 0)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)


def _rotate_inv(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of unit quaternions (x, y, z, w)."""
    u, w = -q_xyzw[:, :3], q_xyzw[:, 3:4]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


__all__ = ["CONTACT_BIT", "ROW_BIT", "NewtonViews", "overflow_bits"]
