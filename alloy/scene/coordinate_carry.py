# SPDX-License-Identifier: Apache-2.0
"""Compensated float32 joint coordinates on a SolverMuJoCo entry (SCR-SC10-3, coordinator ruling 2026-11-05 19:10Z).

**The bug.** MuJoCo moves a joint by q_new = q_old + dt * qvel. Newton keeps ``state.joint_q`` in float32 and writes it
into MuJoCo at the start of every step and back at the end (MuJoCo Warp also keeps its own qpos in float32). An
increment below half the float32 spacing of q is rounded away, so the coordinate does not move while qvel stays. Near
|q| = 2.2 rad that spacing is 2.4e-7 rad. A joint creeping slower than 1.2e-7 rad / dt (1.8 mrad/s at dt 65 us) is
frozen. Smaller steps freeze faster motions: the bound is half-spacing / dt.
- **Servo stall.** A position servo settling with kv / kp = 0.1 s (FR3) stops 0.1 s x 1.8 mrad/s = 180 urad short at
  dt 65 us, and 360 urad short at dt/2, keeping that qvel forever (stock repro: ``quant_stall.py``).
- **The pipette walk.** In C18, FR3 joints 4 and 6 sit at q = -2.2 and 2.6 rad. The hand's frozen joints keep a ghost
  velocity of 0.5 mrad/s (dt) or 1.5 mrad/s (dt/2). The pads' friction carries the gripped pipette at that velocity,
  and the pipette's own coordinates are fine enough to move, so it walks out of the grip at 0.3 / 1.4 mm per 2 s.

**The carry.** After each MuJoCo step, a kernel measures what rounding dropped from each carried coordinate,
``lost = dt * v - (q_new - q_old)`` in float32, where v is the velocity MuJoCo integrated. It adds that to a per-coordinate
residual and puts the residual back into q once it reaches the float32 spacing (Kahan summation). Every operation is
float32 on the model's device, so on MuJoCo Warp it is graph-capturable.
- **Carried coordinates.** Hinge and slide axes (revolute, prismatic and D6 joints: v is ``joint_qd``), and free-joint
  positions whose child anchor sits at the body origin (v is MuJoCo's origin velocity, rotated into the parent frame).
- **Not carried.** Free and ball rotations (quaternions), kinematic bodies (Newton passes their coordinates through)
  and loop joints. A free body turning slower than about 2.4e-7 / dt rad/s still freezes its rotation.
- **Coupled entries.** A coupler replays the MuJoCo entry from the same input several times per exchange (proxy and ADMM
  passes) and keeps only the last pass. The residual is saved when the coupler writes an exchange's input coordinates
  (``coupling_notify_input_state_update`` with ``JOINT_Q``, without ``iteration_restart``), at once, and the first
  step after that or after any restart starts from the saved value, so each exchange advances it by exactly one pass.
  The residual is per-world solver state (``state_arrays`` / ``world_maps``, registered by WP4's
  ``mujoco_state_arrays``), so snapshots, rollback, save/load and the freeze restore it per world. The save must not
  wait for the step: the freeze snapshots a pass's start before the step runs, and a world settling in pass 0 was then
  replayed from the previous exchange's save (one-ulp joint_q, CPL-9 batch invariance, WP3 trace 2026-11-06).
- **Guard.** A change larger than 4 float32 spacings is not rounding (an integrator this module does not model): the
  coordinate is left alone, its residual drops to 0, and ``guard_trips`` counts it. The spacing is taken at
  max(|q_old|, |q_new|, |dt v|) plus the offset between Newton's coordinate and MuJoCo's: an MJCF ``ref`` (Newton keeps
  q = qpos - ref, and MuJoCo rounds qpos) or a free joint's parent translation. Without that offset, plain rounding on
  SFP's ref'd connector slides tripped it (110 trips in 10 periods; WP3 trace 2026-11-05).
- **Body poses.** The carry runs inside the step, between Newton's conversion of MuJoCo's coordinates into
  ``joint_q`` and the forward kinematics that derive ``body_q`` from it (``solver_mujoco.eval_fk``, a private hook in
  ``alloy.core.newton_compat.PRIVATE_HOOKS``). So ``body_q`` stays FK of the carried ``joint_q``, Newton's invariant,
  with no extra kernel. Run after that FK, the body poses lagged q by up to one spacing. That broke the xbench A9
  port check (1.5e-11 m) and any restore that rebuilds ``body_q`` from ``joint_q``. A step path without that FK gets
  the carry after the step instead.
- **Refused** (``CarryError``): rk4 (its position update is not q + dt * qvel), ``update_data_interval != 1`` (no
  per-step round trip, e.g. ``SyncedMuJoCo``), MuJoCo C with more than one world, and a joint count that does not split
  evenly into worlds (shared world -1 joints).

**Default-on** (coordinator ruling 2026-11-05 22:33Z): ``install_default``, called from ``alloy/__init__``, installs it
at the end of every SolverMuJoCo's construction where it can run. Opt out per entry with the entry kwarg
``{CARRY_KEY: False}`` (``alloy.scene.builtin_plugins``), or ``uninstall(solver)`` right after a solver is built.

**Measured** (C18 dry rig, SC10 df7e9c2f world 0, ``[internal note]``). Tip offset from the grasp through the
touch-off (t 3.3 s):
- **MuJoCo C.** 0.22 / 0.66 mm at dt / dt/2 with the round trip, and 0.005 / 0.004 mm with the carry. Without the round
  trip (float64 qpos) it is 0.011 mm at both steps by t 2.2 s.
- **MuJoCo Warp.** The largest dt-vs-dt/2 tip difference drops from 0.678 mm to 0.033 mm.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

log = logging.getLogger("alloy.scene")
_STEPPING = threading.local()   # .carry: the _Carry whose solver step runs on this thread (its FK hook is armed)

CARRY_KEY = "alloy:coordinate_carry"   # MuJoCo entry kwarg: True installs the carry on that entry
AXIS = 0                                # kind of a carried coordinate: a hinge/slide axis; 1, 2, 3: free position x, y, z
GUARD_SPACINGS = 4.0
AXIS_TYPES = (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC), int(newton.JointType.D6))
KINEMATIC = int(newton.BodyFlags.KINEMATIC)
JOINT_Q = int(newton.StateFlags.JOINT_Q)   # notify flag: the coupler wrote the entry's input joint coordinates


class CarryError(ValueError):
    """The carry cannot run on this entry (module docstring, "Refused")."""


@wp.kernel(enable_backward=False)
def _carry_kernel(q_old: wp.array(dtype=wp.float32), joint_qd: wp.array(dtype=wp.float32),
                  qvel: wp.array2d(dtype=wp.float32), coord: wp.array(dtype=wp.int32), dof: wp.array(dtype=wp.int32),
                  kind: wp.array(dtype=wp.int32), joint: wp.array(dtype=wp.int32), world: wp.array(dtype=wp.int32),
                  offset: wp.array(dtype=wp.float32), joint_X_p: wp.array(dtype=wp.transform),
                  joint_child: wp.array(dtype=wp.int32),
                  body_flags: wp.array(dtype=wp.int32), dt: wp.float32, guard: wp.float32,
                  joint_q: wp.array(dtype=wp.float32), carry: wp.array(dtype=wp.float32),
                  trips: wp.array(dtype=wp.int32)):
    t = wp.tid()
    j = joint[t]
    if (body_flags[joint_child[j]] & KINEMATIC) != 0:
        carry[t] = 0.0
        return
    i = coord[t]
    k = kind[t]
    inc = float(0.0)
    if k == AXIS:
        inc = dt * joint_qd[dof[t]]
    else:
        d = dof[t]
        w = world[t]
        v = wp.quat_rotate_inv(wp.transform_get_rotation(joint_X_p[j]),
                               wp.vec3(qvel[w, d], qvel[w, d + 1], qvel[w, d + 2]))
        inc = dt * v[k - 1]
    qn = joint_q[i]
    qo = q_old[i]
    lost = inc - (qn - qo)
    scale = wp.max(wp.max(wp.abs(qn), wp.abs(qo)), wp.abs(inc)) + offset[t]   # MuJoCo rounds at |q + ref|
    if wp.abs(lost) > guard * scale * 1.1920929e-07 + 1.0e-30:
        carry[t] = 0.0
        wp.atomic_add(trips, 0, 1)
        return
    c = carry[t] + lost
    q2 = qn + c
    carry[t] = c - (q2 - qn)
    joint_q[i] = q2


@wp.kernel(enable_backward=False)
def _zero_worlds(world: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool), carry: wp.array(dtype=wp.float32)):
    t = wp.tid()
    if mask[world[t]]:
        carry[t] = 0.0


@dataclass(frozen=True)
class CarryPlan:
    """The carried coordinates (one entry each): joint_q index, the velocity's index (``joint_qd`` for an axis, the
    free joint's first MuJoCo dof for a position), kind, Newton joint, MuJoCo world, the size of the offset between
    Newton's coordinate and MuJoCo's (|dof_ref| for an axis, |X_p translation| for a free position: MuJoCo rounds the
    offset coordinate, so the guard scales by it too); and what was left out."""

    coord: np.ndarray
    dof: np.ndarray
    kind: np.ndarray
    joint: np.ndarray
    world: np.ndarray
    offset: np.ndarray
    skipped: dict[str, int]

    @property
    def counts(self) -> dict[str, int]:
        return {"axes": int(np.sum(self.kind == AXIS)), "free_positions": int(np.sum(self.kind != AXIS))}


def refusal(solver: Any) -> str | None:
    """Why the carry cannot run on ``solver``, or None."""
    import mujoco  # noqa: PLC0415

    if not isinstance(solver, newton.solvers.SolverMuJoCo):
        return f"not a SolverMuJoCo ({type(solver).__name__})"
    own = vars(solver)        # Newton > 1.8 keeps the deprecated option in a private field (its property warns)
    interval = own.get("_update_data_interval", own.get("update_data_interval", 1))
    if (1 if interval is None else int(interval)) != 1:
        return ("update_data_interval != 1: MuJoCo keeps its own float32 qpos between steps (SyncedMuJoCo, rk4 "
                "entries), so there is no per-step joint_q round trip to compensate")
    if int(solver.mj_model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
        return "rk4: its position update is not q + dt * qvel"
    worlds = int(solver.model.world_count)
    if solver.use_mujoco_cpu and worlds != 1:
        return f"MuJoCo C with {worlds} worlds (one MjData)"
    if worlds > 1 and solver.model.joint_count % worlds:
        return f"{solver.model.joint_count} joints do not split evenly into {worlds} worlds (shared joints)"
    return None


def carry_plan(solver: Any) -> CarryPlan:
    """The coordinates the carry covers on a built SolverMuJoCo (module docstring, "Carried coordinates")."""
    model = solver.model
    jt, qs, ds = (a.numpy() for a in (model.joint_type, model.joint_q_start, model.joint_qd_start))
    dim = model.joint_dof_dim.numpy()
    xc = model.joint_X_c.numpy()
    mq, md = solver.mj_q_start.numpy(), solver.mj_qd_start.numpy()
    per_world = model.joint_count // max(1 if solver.use_mujoco_cpu else int(model.world_count), 1)
    rows: list[tuple[int, int, int, int, int]] = []
    skipped = {"loop": 0, "free_offset_anchor": 0, "rotation_only": 0}
    for j in range(model.joint_count):
        w, local = divmod(j, per_world)
        if mq[local] < 0:
            skipped["loop"] += 1
            continue
        if jt[j] in AXIS_TYPES:
            rows += [(qs[j] + a, ds[j] + a, AXIS, j, w) for a in range(int(dim[j, 0] + dim[j, 1]))]
        elif jt[j] == int(newton.JointType.FREE):
            if np.any(xc[j, :3] != 0.0):
                skipped["free_offset_anchor"] += 1
                continue
            rows += [(qs[j] + a, md[local], a + 1, j, w) for a in range(3)]
        elif jt[j] == int(newton.JointType.BALL):
            skipped["rotation_only"] += 1
    arr = np.array(rows, dtype=np.int32).reshape(-1, 5)
    return CarryPlan(*(arr[:, c].copy() for c in range(5)), offset=_offsets(model, arr),
                     skipped={k: v for k, v in skipped.items() if v})


def _offsets(model: Any, rows: np.ndarray) -> np.ndarray:
    """|dof_ref| per carried axis and |joint_X_p translation| per carried free position (``CarryPlan.offset``)."""
    out = np.zeros(len(rows), dtype=np.float32)
    ref = getattr(getattr(model, "mujoco", None), "dof_ref", None)
    axis = rows[:, 2] == AXIS
    if ref is not None and axis.any():
        out[axis] = np.abs(ref.numpy()[rows[axis, 1]])
    if (~axis).any():
        out[~axis] = np.linalg.norm(model.joint_X_p.numpy()[rows[~axis, 3], :3], axis=1)
    return out


_EVAL_FK = None   # Newton's solver_mujoco.eval_fk, once hooked


def _fk_with_carry(model: Any, joint_q: Any, joint_qd: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
    """``solver_mujoco.eval_fk`` with the stepping carry applied first: SolverMuJoCo's ``_update_newton_state`` writes
    MuJoCo's coordinates into ``state.joint_q``, then calls this for ``body_q``. Other FK calls pass through."""
    st = getattr(_STEPPING, "carry", None)
    if st is not None and st.pending is not None and joint_q is st.pending[0].joint_q:
        state_out, dt = st.pending
        st.pending = None
        st.apply(state_out, dt)
    return _EVAL_FK(model, joint_q, joint_qd, state, *args, **kwargs)


def _hook_newton_fk() -> None:
    """Route SolverMuJoCo's in-step forward kinematics through ``_fk_with_carry`` (idempotent)."""
    global _EVAL_FK
    from alloy._vendor.newton._src.solvers.mujoco import solver_mujoco as sm  # noqa: PLC0415  (PRIVATE_HOOKS: solver_mujoco.eval_fk)

    if getattr(sm.eval_fk, "_alloy_carry_fk", False):
        return
    _EVAL_FK = sm.eval_fk
    _fk_with_carry._alloy_carry_fk = True
    sm.eval_fk = _fk_with_carry


class _Carry:
    """Device state of one installed carry: the plan's index arrays, the residuals, the step's q_old and the guard
    counter."""

    def __init__(self, solver: Any, plan: CarryPlan) -> None:
        self.solver, self.plan, dev = solver, plan, solver.model.device
        self.n = len(plan.coord)
        self.idx = [wp.array(getattr(plan, f), dtype=wp.int32, device=dev)
                    for f in ("coord", "dof", "kind", "joint", "world")]
        self.offset = wp.array(plan.offset, dtype=wp.float32, device=dev)
        self.carry = wp.zeros(self.n, dtype=wp.float32, device=dev)
        self.saved = wp.zeros(self.n, dtype=wp.float32, device=dev)   # the residual at the exchange's input
        self.restore_next = False
        self.q_old = wp.zeros(solver.model.joint_coord_count, dtype=wp.float32, device=dev)
        self.trips = wp.zeros(1, dtype=wp.int32, device=dev)
        self._cpu_qvel = None
        self.pending = None     # (state_out, dt) of the running step until its carry is applied

    def before(self, state_in: Any) -> None:
        if self.restore_next:
            wp.copy(self.carry, self.saved)
            self.restore_next = False
        wp.copy(self.q_old, state_in.joint_q)

    def input_updated(self, flags: int, iteration_restart: bool) -> None:
        """A coupler wrote the entry's input. New input coordinates (``JOINT_Q``, not a restart) open an exchange: the
        residual is saved now, before the coupler's freeze snapshots the pass. A restart replays from that input.
        Either way the next step starts from the saved residual, also in a world whose freeze put back older solver
        state. Deferred to the step: the coupler's reset also notifies a restart before ``reset(world_mask)``, which
        must not roll back other worlds."""
        if not iteration_restart and int(flags) & JOINT_Q:
            wp.copy(self.saved, self.carry)
        if iteration_restart or int(flags) & JOINT_Q:
            self.restore_next = True

    def apply(self, state_out: Any, dt: float) -> None:
        """The carry kernel on the step's output coordinates (before their FK: ``_fk_with_carry``)."""
        m, s = self.solver.model, self.solver
        if s.use_mujoco_cpu:
            qvel = np.asarray(s.mj_data.qvel, dtype=np.float32)[None]
            if self._cpu_qvel is None or self._cpu_qvel.shape != qvel.shape:
                self._cpu_qvel = wp.zeros(qvel.shape, dtype=wp.float32, device=m.device)
            self._cpu_qvel.assign(qvel)
            qv = self._cpu_qvel
        else:
            qv = s.mjw_data.qvel
        wp.launch(_carry_kernel, dim=self.n,
                  inputs=[self.q_old, state_out.joint_qd, qv, *self.idx, self.offset, m.joint_X_p, m.joint_child,
                          m.body_flags,
                          float(dt), GUARD_SPACINGS],
                  outputs=[state_out.joint_q, self.carry, self.trips], device=m.device)

    def reset(self, world_mask: Any = None) -> None:
        self.restore_next = False                 # a reset is not a restart (SolverCoupled.reset notifies one first)
        for buf in (self.carry, self.saved):
            if world_mask is None:
                buf.zero_()
            else:
                wp.launch(_zero_worlds, dim=self.n, inputs=[self.idx[4], world_mask], outputs=[buf],
                          device=self.solver.model.device)


def install_coordinate_carry(solver: Any) -> str:
    """Install the carry on a built SolverMuJoCo before its first step (module docstring); returns a summary for the
    route report. Raises CarryError where it cannot run. ``install_default`` already does this at construction; a
    late install on a solver inside a built Simulation or Scheduler is missed by their snapshots and the freeze,
    which list solver state once, at construction (WP4)."""
    why = refusal(solver)
    if why is not None:
        raise CarryError(f"coordinate carry: {why}")
    if hasattr(solver, "_alloy_carry"):
        return "already installed"
    plan = carry_plan(solver)
    path = "MuJoCo C" if solver.use_mujoco_cpu else "MuJoCo Warp"
    if not len(plan.coord):
        return f"{path}: none"
    _hook_newton_fk()
    state = _Carry(solver, plan)
    solver._alloy_carry = state
    step, reset = solver.step, solver.reset

    def carried_step(state_in, state_out, control, contacts, dt, **kw):
        state.before(state_in)
        outer, state.pending = getattr(_STEPPING, "carry", None), (state_out, dt)
        _STEPPING.carry = state
        try:
            step(state_in, state_out, control, contacts, dt, **kw)
        finally:
            _STEPPING.carry = outer
        if state.pending is not None:          # a step path without the in-step FK: carry after it
            state.pending = None
            state.apply(state_out, dt)

    def carried_reset(st, world_mask=None, flags=None):
        reset(st, world_mask=world_mask, flags=flags)
        state.reset(world_mask)

    solver.step, solver.reset = carried_step, carried_reset
    state.wrappers = {"step": carried_step, "reset": carried_reset}
    notify = getattr(solver, "coupling_notify_input_state_update", None)
    if notify is not None:
        def carried_notify(st, flags, *, iteration_restart=False, dt=0.0):
            notify(st, flags, iteration_restart=iteration_restart, dt=dt)
            state.input_updated(flags, bool(iteration_restart))
        solver.coupling_notify_input_state_update = carried_notify
        state.wrappers["coupling_notify_input_state_update"] = carried_notify
    skipped = f", skipped {plan.skipped}" if plan.skipped else ""
    log.info("coordinate carry (%s): %s%s", path, plan.counts, skipped)
    return f"{path}: {plan.counts}{skipped}"


def uninstall(solver: Any) -> bool:
    """Take the carry off a solver (the opt-out, right after construction): False if none was installed. Raises
    CarryError if something wrapped the solver's step since, since removing ours would drop theirs too."""
    state = getattr(solver, "_alloy_carry", None)
    if state is None:
        return False
    if any(vars(solver).get(name) is not fn for name, fn in state.wrappers.items()):
        raise CarryError("coordinate carry: the solver was wrapped again after the carry; opt out before other installs")
    for name in state.wrappers:
        delattr(solver, name)
    del solver._alloy_carry
    return True


def install_default() -> None:
    """Default-on (coordinator ruling 2026-11-05 22:33Z): every SolverMuJoCo built from now on gets the carry at the end
    of its construction, where it can run (``refusal``: SyncedMuJoCo / rk4, ``update_data_interval != 1`` and
    multi-world MuJoCo C keep running without it). Called at this module's import and from ``alloy/__init__``;
    idempotent. Opt out per entry with ``{CARRY_KEY: False}`` or ``uninstall(solver)``."""
    cls = newton.solvers.SolverMuJoCo
    if getattr(cls, "_alloy_carry_default", False):
        return
    init = cls.__init__

    def __init__(self, *a, **k):
        init(self, *a, **k)
        if refusal(self) is None and not hasattr(self, "_alloy_carry"):
            install_coordinate_carry(self)

    __init__.__wrapped__ = init
    cls.__init__ = __init__
    cls._alloy_carry_default = True


def state_arrays(solver: Any) -> dict[str, Any]:
    """The carry's solver state (the residual and its exchange save), for WP4's state adapter: snapshots, rollback,
    save/load and the freeze restore them per world through ``world_maps``. Also the stick anchor's slots
    (`alloy.contact.mjw_stick_anchor.state_arrays`): solver state of the same kind, with a leading world axis."""
    from alloy.contact import mjw_stick_anchor  # noqa: PLC0415  (the stick anchor's slots: (worlds, ...) arrays)

    st = getattr(solver, "_alloy_carry", None)
    out = {} if st is None else {"carry.residual": st.carry, "carry.saved": st.saved}
    return {**out, **mjw_stick_anchor.state_arrays(solver)}


def world_maps(solver: Any) -> dict[str, Any]:
    """Item -> world maps for ``state_arrays`` (one entry per carried coordinate)."""
    st = getattr(solver, "_alloy_carry", None)
    return {} if st is None else {"carry.residual": st.idx[4], "carry.saved": st.idx[4]}


def guard_trips(solver: Any) -> int:
    """Changes the guard refused since install (0 without a carry)."""
    state = getattr(solver, "_alloy_carry", None)
    return int(state.trips.numpy()[0]) if state is not None else 0


__all__ = ["CARRY_KEY", "CarryError", "CarryPlan", "carry_plan", "guard_trips", "install_coordinate_carry",
           "install_default", "refusal", "state_arrays", "uninstall", "world_maps"]

# Default-on from this module's own import as well (WP0's SCR-OSS3-1 lazy alloy/__init__: its post-import hook on newton
# skips while this module is mid-import); idempotent.
install_default()
