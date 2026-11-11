# SPDX-License-Identifier: Apache-2.0
"""Joint springs that yield (SCR-WP7-15 P2): a hinge's torsion spring (or a slide's spring) that takes a permanent set
past its yield torque (force), as a bent clip, a sprung latch arm or an over-twisted torsion spring does.

**Model.** The joint's MuJoCo spring (MJCF ``stiffness`` k, ``springref`` s0) is the elastic part. Per joint a yield
torque τ_y0 (N·m; a force in N for a slide) and a hardening k_h ≥ 0 (N·m/rad; N/m) make it elasto-plastic with
isotropic hardening, the 1-D J2 return of ``alloy.solvers.plastic.j2.return_1d`` in spring units: strain q - s0,
E = k, σ_y0 = τ_y0, H = k_h. The plastic set q_p moves the spring reference to s0 + q_p, so the spring torque is
-k (q - s0 - q_p) and |k (q - s0 - q_p)| <= τ_y0 + k_h ε̄p, ε̄p the accumulated |Δq_p|. Released after an overload to
q_max, the joint rests at s0 + q_max - (τ_y0 + k_h ε̄p) / k (one-sided loading from rest: q_max - τ_y0/k for k_h = 0).
A reversed load yields at -(τ_y0 + k_h ε̄p): no Bauschinger effect (docs/known_gaps/plasticity-metal.md).

**When.** Once per solver step, after it, on the step's converged joint coordinate (explicit, as MuJoCo's own spring).
Before every step the spring reference s0 + q_p is written into the route's model
(``alloy.plugins.joint_yield.routes``).

**State** (``plugin/joint_yield/...``, ``[W, n]`` in the order of ``config.joints``): ``plastic_set`` q_p [rad or m]
and ``plastic_strain`` ε̄p. Snapshots, health rollbacks and ``reset`` restore both; the next step's spring reference
follows.

**Energy.** Health's energy check sums each joint spring about its build reference, 0.5 k (q - s0)². After a yield
the spring holds 0.5 k (q - s0 - q_p)², so the check read the difference, k q_p (q - s0 - q_p / 2), as energy from
nowhere: a 1 N·m/rad hinge sagging past a 0.02 N·m yield went INVALID with ENERGY_JUMP at 0.56 s (audit AGENTS-6,
2026-11-09). Each step the plug-in books that difference's change with health (``health_work.register_work``; the
booked value is the state ``booked_energy`` [J], restored with the set), so health's net energy is the true one and
the plastic work shows as energy leaving.

No yield set: no plug-in, no arrays, no kernels (off is bitwise the stock step).

**Routes.** MuJoCo Warp (GPU, and the warp CPU route) and Alloy's MuJoCo C route; not coupled steppers (a robot
beside cloth, liquid or a soft body) yet. Example (the API does this for ``scene.set_joint_yield``)::

    JointYield(joints=[{"name": "clip/arm", "label": "clip/arm", "limit": 0.02, "hardening": 0.0}])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.joint_yield.kernels import book_set, yield_step
from alloy.plugins.joint_yield.routes import mujoco_solver, springref_writer
from alloy.precision.health_work import register_work

log = logging.getLogger("alloy.plugins.joint_yield")
ONE_DOF = (0, 1)                    # newton.JointType.PRISMATIC, REVOLUTE


@dataclass
class YieldJoint:
    """One yielding joint: ``name`` (for messages and readouts), ``limit`` τ_y0 [N·m or N] > 0, ``hardening`` k_h
    [N·m/rad or N/m] >= 0, and either ``rows`` (its global joint index in every world) or ``label`` (a model label)."""
    name: str
    limit: float
    hardening: float = 0.0
    rows: list[int] | None = None
    label: str | None = None


@dataclass
class JointYieldConfig:
    joints: list[dict[str, Any]] = field(default_factory=list)     # YieldJoint fields per joint


class JointYield(ForcePlugin):
    name = "joint_yield"
    summary = "joint springs that yield: a permanent set past a yield torque (force), isotropic hardening"
    Config = JointYieldConfig

    def setup(self, ctx: Any) -> None:
        self.items = [YieldJoint(**j) for j in self.config.joints]
        if not self.items:
            raise PluginError("joint_yield: no joints given")
        for it in self.items:
            if not (np.isfinite(it.limit) and it.limit > 0.0) or not (np.isfinite(it.hardening) and it.hardening >= 0):
                raise PluginError(f"joint_yield {it.name!r}: yield {it.limit!r} must be > 0 and hardening "
                                  f"{it.hardening!r} >= 0")
        m, W, n = ctx.model, ctx.world_count, len(self.items)
        self.rows = np.stack([np.asarray(it.rows if it.rows is not None else ctx.joints(it.label).numpy(), np.int64)
                              for it in self.items], axis=1)                       # [W, n] global joint indices
        if self.rows.shape != (W, n):
            raise PluginError(f"joint_yield: rows shape {self.rows.shape}, expected (worlds, joints) = {(W, n)}")
        jtype, q_start = m.joint_type.numpy(), m.joint_q_start.numpy()
        for i, it in enumerate(self.items):
            if int(jtype[self.rows[0, i]]) not in ONE_DOF:
                raise PluginError(f"joint_yield {it.name!r} is not a hinge or slide joint; only one-axis joint "
                                  f"springs yield (a ball or free joint has no 1-D spring)")
        self.q_index = wp.array(q_start[self.rows].astype(np.int32), dtype=wp.int32, device=ctx.device)
        self.columns = {int(j): i for i, j in enumerate(self.rows[0])}      # template joint -> state column
        self.limit = ctx.array([it.limit for it in self.items])
        self.hardening = ctx.array([it.hardening for it in self.items])
        self.q_p = self.add_state("plastic_set", ctx.zeros((W, n)))
        self.ebar = self.add_state("plastic_strain", ctx.zeros((W, n)))
        self.booked = self.add_state("booked_energy", ctx.zeros((W, n)))    # module docstring, Energy
        self.work = wp.zeros(W, dtype=wp.float64, device=ctx.device)        # J since health last booked it
        register_work(m, self.work)
        self.k: wp.array | None = None
        self.s0: wp.array | None = None
        self.writer: Any = None

    def attach(self, stepper: Any) -> None:
        """Find the MuJoCo joints, their stiffness and build spring reference; write s0 + q_p before every step."""
        solver = mujoco_solver(stepper)
        mjw, W = solver.mjw_model, self.ctx.world_count
        mj = self._mujoco_joints(solver)
        adr = mjw.jnt_qposadr.numpy()[mj[0]]
        if (mjw.jnt_qposadr.numpy()[mj] != adr[None]).any():
            raise PluginError("joint_yield: the yielding joints sit at different MuJoCo coordinates per world")
        k_rows, ref_rows = mjw.jnt_stiffness.numpy(), mjw.qpos_spring.numpy()
        worlds = np.arange(W)
        k = k_rows[worlds % k_rows.shape[0]][worlds[:, None], mj]
        s0 = ref_rows[worlds % ref_rows.shape[0]][:, adr]
        self._check_springs(solver, mj, k)
        self.k = wp.array(k.astype(np.float32), dtype=float, device=self.ctx.device)
        self.s0 = wp.array(s0.astype(np.float32), dtype=float, device=self.ctx.device)
        self.writer = springref_writer(solver, adr, self.s0, self.q_p)
        inner = stepper.step

        def step(*args: Any, **kwargs: Any) -> Any:
            self.writer.push()
            return inner(*args, **kwargs)

        stepper.step = step
        log.info("joint_yield on %s (%s): %s", type(solver).__name__, type(self.writer).__name__,
                 [(it.name, it.limit, it.hardening) for it in self.items])

    def update(self, state_in: Any, state_out: Any, dt: float) -> None:
        wp.launch(yield_step, dim=self.q_p.shape, inputs=[state_out.joint_q, self.q_index, self.s0, self.k,
                                                          self.limit, self.hardening],
                  outputs=[self.q_p, self.ebar], device=self.ctx.device)
        wp.launch(book_set, dim=self.q_p.shape[0], inputs=[state_out.joint_q, self.q_index, self.s0, self.k, self.q_p],
                  outputs=[self.booked, self.work], device=self.ctx.device)

    def _mujoco_joints(self, solver: Any) -> np.ndarray:
        """``[W, n]`` MuJoCo joint index of each yielding joint (Newton's ``mjc_jnt_to_newton_jnt``)."""
        to_newton = solver.mjc_jnt_to_newton_jnt.numpy()
        W, n = self.rows.shape
        out = np.full((W, n), -1, np.int64)
        for w in range(W):
            table = to_newton[w % to_newton.shape[0]]
            for i in range(n):
                hit = np.flatnonzero(table == self.rows[w, i])
                if hit.size == 0 and to_newton.shape[0] == 1:          # one MuJoCo world: template joint indices
                    hit = np.flatnonzero(table == self.rows[0, i])
                if hit.size != 1:
                    raise PluginError(f"joint_yield {self.items[i].name!r}: not a MuJoCo joint in world {w}")
                out[w, i] = hit[0]
        return out

    def _check_springs(self, solver: Any, mj: np.ndarray, k: np.ndarray) -> None:
        poly = getattr(solver.mjw_model, "jnt_stiffnesspoly", None)
        for i, it in enumerate(self.items):
            if not (k[:, i] > 0.0).all():
                raise PluginError(f"joint_yield {it.name!r} has no spring (stiffness {float(k[:, i].min()):g}): give "
                                  f"the joint stiffness=\"k\" (and springref) in its MJCF")
            if poly is not None:
                p = poly.numpy()
                if np.abs(p[np.arange(k.shape[0]) % p.shape[0]][:, mj[0, i]]).max() > 0.0:
                    raise PluginError(f"joint_yield {it.name!r}: a nonlinear (stiffnesspoly) spring cannot yield "
                                      f"with the 1-D linear return; use a linear spring")

    def names(self) -> list[str]:
        """The joint names, in state column order."""
        return [it.name for it in self.items]
