# SPDX-License-Identifier: Apache-2.0
"""SolverVBDRod: Newton's SolverVBD with every rod chain solved directly instead of by the colour sweep.

**The problem it fixes.** Stock SolverVBD updates a body with its neighbours held. For a rod, each body's angular
block is dominated by its joint anchors' stretch/shear terms, about (l/r)^2 x the bend stiffness, and bending
couples along the whole chain. So iterations hardly move the bend error: the D8 cantilever sags +88 % vs the
elastica at 10 iterations, +55..94 % at 100-400 ([internal note]). Here each chain
(`alloy.solvers.rod.topology`) takes one exact Gauss-Newton step per iteration on all its bodies
(`alloy.solvers.rod.kernels`). At 1 iteration: sag -0.19 %, release drop -3.3 %, period -0.25 % (D8 bounds 5 %,
5 %, 2 %).

**Where it runs in Newton's loop.**
- The chain bodies leave the colour groups, and an empty colour is appended. SolverVBD calls
  `_rigid_penetration_free_truncation` after every colour, and after the last one this class:
  1. accumulates the chain bodies' contact blocks with Newton's own kernels;
  2. evaluates their non-chain joints with Newton's own `evaluate_joint_force_hessian`;
  3. solves the chains.
- That is before Newton's mimic projection and dual updates, so the duals see the updated chains. Running after them
  let ALM duals see stale poses and fed the dense-pile blow-up.
- `hook_order_ok()` checks that order in the pinned Newton's source (a tripwire); the constructor refuses to run if
  it fails. At run time, an iteration that solves rigid bodies without the pass running raises: chain bodies are in no
  colour, so they would freeze silently.

**The forward step** (SCR-WP9-ROD R1 and R3; WP2 reviews). SolverVBD's forward step builds each body's inertial
target to first order in rotation: q + dt/2 omega q, normalized, with any `body_f` torque in omega. A step that asks
for an angle x then turns 2 atan(x / 2). `_initialize_rigid_bodies` changes two things for chain bodies:
- **Applied torques** (R1). A torque in `state.body_f` on a chain body enters the chain's Gauss-Newton step as a
  potential (kernels, Body terms), not the target.
  - Why: x = dt^2 tau / I_axis, and a 1.5 g segment's axial inertia is about 7e-9 kg m^2. So at 0.01-0.03 N m, x
    is 4.5-13.5, and a clamped rod twisted 0.51-0.21 of T L / (G J). An exp-map target would still wrap above pi.
  - As a potential, the twist is T L / (G J) to 1e-7 at physical inertia ([internal note] notes/wp9-runs/rod), with no
    armature.
  - Each nonzero chain-body torque leaves `body_f` for the forward step and goes back after, so the user's `body_f`
    is unchanged. Forces stay in the target: a linear target does not wrap.
  - With no torque on a chain body, nothing changes bit for bit.
  - `control.joint_f` on a chain joint still shifts the target (Newton maps it into a scratch copy inside the
    forward step).
- **Exact rotation target** (R3, `exact_rotation_target=True`). The target rotates by exp(omega dt) from the start
  pose; the forward step's COM is kept.
  - Why: the first-order turn loses (omega dt)^2 / 12 of a spin per step. A free rod rolling down a 10 deg incline
    (r 5 mm, 1.25 ms) reached 0.948 of its own rolling acceleration, and 0.822 at 20 deg. One stock-VBD capsule does
    the same. Exact: 0.998 and 0.991.
  - WP2's D8 rows hardly move: sag -0.489 -> -0.493 %, release peak -3.235 -> -3.230 %, period -0.291 %.
  - `exact_rotation_target=False` gives the previous step, for bitwise comparisons.
  - It runs after the forward step's own work on the first-order poses (Newton's in-forward truncation and the joint
    C0 snapshot). Those differ by O((omega dt)^3), and a spin moves no capsule geometry.

**Joint semantics in a chain.**
- A ROD joint uses its `joint_target_ke` / `joint_target_kd` for all six slots, as Newton's ROD rows do: stretch,
  shear, bend and twist stiffness and damping.
- A FIXED weld inside a chain is a bare penalty: SolverVBD's structural `rigid_joint_linear_ke` /
  `rigid_joint_angular_ke` (1e5 by default), with no damping and no ALM dual. Newton's own weld adds kd and, when hard,
  a dual. Its residual under load is F / k_lin and M / k_ang. The 20 g plug's 0.2 N gives 2 um (measured 2 um); its
  2e-3 N m root moment gives 2e-8 rad.

**Rod groups.** Chains in one group update at once, so contacts between them are Jacobi: each rod treats the other as
held while both move. Groups run one after the other (Gauss-Seidel), each re-accumulating its contacts.
- In a dense pile of 64 touching rods at 2 iterations ([internal note]), one group went
  non-finite at step 538. One group per rod stayed finite. With rod-rod collisions filtered out, one group was fine.
- So the default group k holds the k-th chain of every world. Chains in different worlds never touch, so one rod per
  world (ROD-4) is a single parallel group, and the chains of one world never share a group.
- `rod_groups` overrides it. A per-step contact colouring, so touching rods never share a group, is the follow-up
  for many rods in one world.

**Not covered:**
- a chain body's contact with a body of the same chain is Jacobi;
- a body welded mid-rod or to two rods stays in the sweep (`topology`);
- rod junctions and closed loops raise.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.solvers.vbd import solver_vbd as _sv

from alloy.solvers.rod.forward import rod_after_forward, rod_before_forward
from alloy.solvers.rod.kernels import (RodArgs, RodChains, mat66, rod_assemble, rod_external_joints, rod_serial,
                                       rod_thomas, rod_update, vec6)
from alloy.solvers.rod.plastic import RodPlastic
from alloy.solvers.rod.topology import ROD, RodTopology

log = logging.getLogger("alloy.solvers.rod")


def hook_order_ok() -> bool:
    """True when SolverVBD's rigid iteration calls the truncation hook inside its colour loop and runs the mimic
    projection and the dual updates after that loop, and `_initialize_rigid_bodies` feeds `state_in.body_f` to the
    forward step, which writes the target. That is the order SolverVBDRod relies on."""
    src = inspect.getsource(_sv.SolverVBD._solve_rigid_body_iteration)
    loop = src.find("for color in range(len(body_color_groups))")
    hook = src.find("self._rigid_penetration_free_truncation(state_in, contacts)", loop)
    duals = src.find("update_duals_body_body_contacts", hook)
    init = inspect.getsource(_sv.SolverVBD._initialize_rigid_bodies)
    reads = init.find("body_f_for_integration = state_in.body_f")
    torque_ok = 0 <= reads < init.find("forward_step_rigid_bodies", reads)   # body_f is read there and only there
    return 0 <= loop < hook < duals and src.find("project_joint_mimics", hook) > hook and torque_ok


def rest_relative(model) -> np.ndarray:
    """A0, the rest rotation of each joint's child frame relative to its parent frame (xyzw).
    - A ROD joint takes the structural rest Newton keeps for it, coordinates 3..6 as `init_rod_rest_bend_twist`
      reads them: `model.joint_rest_q` at `joint_q_start` (Newton 1.8, the bump) or, in Newton before it,
      `model.joint_target_q` at `joint_target_q_start`. So a pre-curved rest gives zero strain.
    - Every other joint takes the identity: a weld's frames coincide at rest."""
    jt = model.joint_type.numpy()
    if getattr(model, "joint_rest_q", None) is not None:
        qs, tq = model.joint_q_start.numpy(), model.joint_rest_q.numpy()
    else:
        qs, tq = model.joint_target_q_start.numpy(), model.joint_target_q.numpy()
    out = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (model.joint_count, 1))
    for j in np.flatnonzero(jt == ROD):
        q = tq[qs[j] + 3:qs[j] + 7].astype(np.float64)
        out[j] = q / np.linalg.norm(q)
    return out


class SolverVBDRod(newton.solvers.SolverVBD):
    """SolverVBD whose rod chains are solved per chain by block Thomas (module docstring).

    `rods`: the chains (default `RodTopology.from_model`). `rod_groups`: lists of chain indices solved one group
    after the other (default one group). `rod_split`: three launches (assemble per slot, Thomas per chain, update
    per slot) instead of one thread per chain; the default is split on CUDA, serial on CPU.
    `exact_rotation_target`: chain bodies' inertial target rotates by exp(omega dt) (module docstring); False gives
    Newton's first-order target. `geometric_stiffness`: tension's geometric stiffness in each chain's Hessian
    (kernels docstring); False gives the plain Gauss-Newton blocks."""

    def __init__(self, model: newton.Model, rods: RodTopology | None = None,
                 rod_groups: Sequence[Sequence[int]] | None = None, rod_split: bool | None = None,
                 exact_rotation_target: bool = True, geometric_stiffness: bool = True, **kw):
        if not hook_order_ok():
            raise RuntimeError("SolverVBDRod: SolverVBD's rigid iteration no longer calls the truncation hook before "
                               "its dual updates (Newton pin changed); the rod pass would run out of order")
        super().__init__(model, **kw)
        self.rods = RodTopology.from_model(model) if rods is None else rods
        dev = self.device
        self.rod_split = wp.get_device(dev).is_cuda if rod_split is None else bool(rod_split)
        self._chains = self._chain_arrays(dev)
        self._groups = self._group_arrays(rod_groups, dev)
        self._other_groups, self._other_dims = self._sweep_groups(model, dev)
        self._rest_rel = wp.array(rest_relative(model), dtype=wp.quat, device=dev)
        self.rod_plastic = RodPlastic.for_model(model, self._rest_rel)   # J2 rods (plastic.py); None: elastic
        if self.rod_plastic is not None:                                 # solver state for snapshots (plastic.py)
            for key, arr in self.rod_plastic.state_arrays().items():
                setattr(self, f"_{key}", arr)
        self._pairs = self._external_joint_pairs(model, dev)
        self._applied_t = wp.zeros(len(self.rods.body), dtype=wp.vec3, device=dev)
        self._pose0 = wp.zeros(len(self.rods.body), dtype=wp.transform, device=dev)
        self.exact_rotation_target = bool(exact_rotation_target)
        self.geometric_stiffness = bool(geometric_stiffness)
        self._rod_pending = None
        self._rod_colour = 0
        log.info("SolverVBDRod: %d chains, %d bodies, %d groups, %s, %d chain bodies with non-chain joints",
                 self.rods.rods, len(self.rods.body), len(self._groups), "split" if self.rod_split else "serial",
                 0 if self._pairs is None else self._pairs[0].size)

    # ---- set-up ----
    def _chain_arrays(self, dev) -> RodChains:
        r = self.rods
        last = np.zeros(len(r.body), np.int32)
        last[r.start + r.count - 1] = 1
        c = RodChains()
        c.body, c.joint, c.start, c.count = (wp.array(x, dtype=int, device=dev) for x in (r.body, r.joint, r.start,
                                                                                          r.count))
        c.last = wp.array(last, dtype=int, device=dev)
        n = len(r.body)
        c.D, c.O, c.F = wp.zeros(n, dtype=mat66, device=dev), wp.zeros(n, dtype=mat66, device=dev), \
            wp.zeros(n, dtype=vec6, device=dev)
        return c

    def _group_arrays(self, rod_groups, dev) -> list[tuple[wp.array, wp.array, wp.array]]:
        """(chain ids, bodies, slots) per group."""
        r = self.rods
        groups = self._per_world_groups() if rod_groups is None else [list(g) for g in rod_groups if len(g)]
        if sorted(i for g in groups for i in g) != list(range(r.rods)):
            raise ValueError("rod_groups must name every chain exactly once")
        out = []
        for g in groups:
            slots = [k for i in g for k in range(int(r.start[i]), int(r.start[i] + r.count[i]))]
            out.append((wp.array(g, dtype=int, device=dev), wp.array(r.body[slots], dtype=int, device=dev),
                        wp.array(slots, dtype=int, device=dev)))
        return out

    def _per_world_groups(self) -> list[list[int]]:
        """Group k = the k-th chain of every world (the default; module docstring)."""
        world = np.asarray(self.model.body_world.numpy())[self.rods.body[self.rods.start]] if self.rods.rods else []
        seen: dict[int, int] = {}
        groups: list[list[int]] = []
        for i, w in enumerate(world):
            k = seen.get(int(w), 0)
            seen[int(w)] = k + 1
            if k == len(groups):
                groups.append([])
            groups[k].append(i)
        return groups or [[]]

    def _sweep_groups(self, model, dev):
        """Newton's colour groups without chain bodies, plus one empty colour after which the rod pass runs."""
        chain = set(int(b) for b in self.rods.body)
        groups, dims = [], []
        for g, dim in zip(model.body_color_groups, self._rigid_color_block_dims, strict=True):
            keep = [int(b) for b in g.numpy() if int(b) not in chain]
            if keep:
                groups.append(wp.array(keep, dtype=int, device=dev))
                dims.append(dim)
        groups.append(wp.zeros(0, dtype=int, device=dev))
        dims.append((_sv._select_rigid_block_dim(1, dev),) * 3)
        return groups, tuple(dims)

    def _external_joint_pairs(self, model, dev):
        """CSR of the enabled non-chain joints touching each chain body: (bodies, start, joints), bodies and each
        body's joints ascending, for `rod_external_joints` (one thread per body, fixed order)."""
        chain = set(int(b) for b in self.rods.body)
        internal = set(int(j) for j in self.rods.joint if j >= 0)
        par, chi, en = model.joint_parent.numpy(), model.joint_child.numpy(), model.joint_enabled.numpy()
        per: dict[int, list[int]] = {}
        for j in range(model.joint_count):
            if not en[j] or j in internal:
                continue
            for b in {int(par[j]), int(chi[j])} & chain:
                per.setdefault(b, []).append(j)
        if not per:
            return None
        bodies = sorted(per)
        start = np.concatenate([[0], np.cumsum([len(per[b]) for b in bodies])]).astype(np.int32)
        joints = [j for b in bodies for j in sorted(per[b])]
        return tuple(wp.array(x, dtype=int, device=dev) for x in (bodies, start, joints))

    def notify_model_changed(self, flags: int) -> None:
        """Newton refreshes the rod rest on a joint-property change (its test_cable pins that); so does the rod pass."""
        super().notify_model_changed(flags)
        if hasattr(self, "_rest_rel"):
            plastic = getattr(self, "rod_plastic", None)
            old = wp.clone(self._rest_rel) if plastic is not None else None
            self._rest_rel.assign(rest_relative(self.model))
            if plastic is not None:                     # a new authored rest; yielded joints keep their set
                plastic.authored_rest_changed(old)

    def step(self, state_in, state_out, control, contacts, dt: float, **kw) -> None:
        """The VBD step, then the J2 return of the yielding rod joints on the converged poses (plastic.py)."""
        super().step(state_in, state_out, control, contacts, dt, **kw)
        if self.rod_plastic is not None:
            self.rod_plastic.after_step(state_out.body_q)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask, flags)
        if self.rod_plastic is not None:
            self.rod_plastic.reset(world_mask)

    # ---- the forward step (module docstring) ----
    def _initialize_rigid_bodies(self, state_in, control, contacts, dt, refresh) -> None:
        n, f = self._chains.body.size, state_in.body_f
        if n == 0:
            return super()._initialize_rigid_bodies(state_in, control, contacts, dt, refresh)
        assert f is not None, "SolverVBDRod: state.body_f is None with rod chains (Newton allocates it per body)"
        C = self._chains
        wp.launch(rod_before_forward, dim=n, inputs=[C.body, f, state_in.body_q, self._applied_t, self._pose0],
                  device=self.device)
        try:
            super()._initialize_rigid_bodies(state_in, control, contacts, dt, refresh)
        finally:
            wp.launch(rod_after_forward, dim=n, device=self.device, inputs=[
                C.body, self._applied_t, self._pose0, self.model.body_com, self.body_inv_mass_effective,
                state_in.body_qd, float(dt), int(self.exact_rotation_target), f, state_in.body_q, self.body_inertia_q])

    # ---- the iteration hook ----
    def _solve_rigid_body_iteration(self, state_in, state_out, control, contacts, dt):
        model = self.model
        groups, dims = model.body_color_groups, self._rigid_color_block_dims
        model.body_color_groups, self._rigid_color_block_dims = self._other_groups, self._other_dims
        self._rod_pending, self._rod_colour = (control, dt), 0
        try:
            super()._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
            missed = self._rod_pending is not None
        finally:
            model.body_color_groups, self._rigid_color_block_dims = groups, dims
            self._rod_pending = None
        if missed and self._integrates_rigid_bodies:   # Newton's skip-rigid-solve return is the one legal miss
            raise RuntimeError("SolverVBDRod: the rod pass did not run in this iteration (SolverVBD no longer calls "
                               "the truncation hook after its last colour); the chains would freeze silently")

    def _rigid_penetration_free_truncation(self, state, contacts):
        if self._rod_pending is not None:
            self._rod_colour += 1
            if self._rod_colour == len(self._other_groups):
                control, dt = self._rod_pending
                self._rod_pending = None
                self._rod_pass(state, control, contacts, dt)
        super()._rigid_penetration_free_truncation(state, contacts)

    def _rod_pass(self, state, control, contacts, dt) -> None:
        if self._pairs is not None:          # the sweep zeroed the scratch and never writes chain bodies
            self._rod_joints(state, control, dt)
        for rods, bodies, slots in self._groups:
            ext = self._rod_contacts(state, contacts, dt, bodies) or self._pairs is not None
            a = self._rod_args(dt, ext)
            if self.rod_split:
                wp.launch(rod_assemble, dim=slots.size, inputs=[a, self._chains, slots, state.body_q], device=self.device)
                wp.launch(rod_thomas, dim=rods.size, inputs=[self._chains, rods], device=self.device)
                wp.launch(rod_update, dim=slots.size, inputs=[a, self._chains, slots, state.body_q], device=self.device)
            else:
                wp.launch(rod_serial, dim=rods.size, inputs=[a, self._chains, rods, state.body_q], device=self.device)

    def _rod_args(self, dt, use_ext) -> RodArgs:
        m = self.model
        a = RodArgs()
        a.dt, a.use_ext, a.use_geo = float(dt), int(bool(use_ext)), int(self.geometric_stiffness)
        a.k_lin, a.k_ang = float(self.rigid_joint_linear_ke), float(self.rigid_joint_angular_ke)
        a.rest_rel, a.body_q_prev, a.body_inertia_q = self._rest_rel, self.body_q_prev, self.body_inertia_q
        a.body_com, a.body_mass, a.body_inertia = m.body_com, m.body_mass, m.body_inertia
        a.body_inv_mass = self.body_inv_mass_effective
        a.joint_type, a.joint_parent, a.joint_X_p, a.joint_X_c = m.joint_type, m.joint_parent, m.joint_X_p, m.joint_X_c
        a.joint_qd_start, a.joint_ke, a.joint_kd = m.joint_qd_start, m.joint_target_ke, m.joint_target_kd
        a.ext_f, a.ext_t = self.body_forces, self.body_torques
        a.ext_ll, a.ext_al, a.ext_aa = self.body_hessian_ll, self.body_hessian_al, self.body_hessian_aa
        a.applied_t = self._applied_t
        return a

    def _rod_contacts(self, state, contacts, dt, bodies) -> bool:
        """The group's body-particle and body-body contact blocks, by Newton's own accumulate kernels with the same
        arguments as SolverVBD's per-colour launches. True when any launch ran."""
        if contacts is None:
            return False
        from alloy._vendor.newton._src.geometry.soft_contacts_mesh import _make_soft_contact_filter_data

        model, dev, n = self.model, self.device, bodies.size
        out = [self.body_forces, self.body_torques, self.body_hessian_ll, self.body_hessian_al, self.body_hessian_aa]
        ran = False
        if model.particle_count > 0:
            cf = _make_soft_contact_filter_data(model, contacts, state.particle_q, state.body_q)
            nt = self._body_particle_contact_threads
            wp.launch(_sv.accumulate_body_particle_contacts_per_body, dim=n * nt, inputs=[
                dt, bodies, nt, state.particle_q, self.particle_q_prev, model.particle_radius, self.body_q_prev,
                state.body_q, state.body_qd, model.body_com, self.body_inv_mass_effective, model.shape_body,
                self.friction_epsilon, self.rigid_soft_contact_use_log_barrier, self.body_particle_contact_penalty_k,
                self.body_particle_contact_material_ke, self.body_particle_contact_material_kd,
                self.body_particle_contact_material_mu, contacts.soft_contact_count, contacts.soft_contact_indices,
                contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_body_vel,
                contacts.soft_contact_normal, contacts.soft_contact_barycentric, model.shape_margin,
                self.body_particle_contact_buffer_pre_alloc, self.body_particle_contact_counts,
                self.body_particle_contact_indices, cf], outputs=out,
                block_dim=_sv._select_rigid_block_dim(n * nt, dev), device=dev)
            ran = True
        if model.shape_count > 0:
            nt = _sv._NUM_CONTACT_THREADS_PER_BODY
            wp.launch(_sv.accumulate_body_body_contacts_per_body, dim=n * nt, inputs=[
                dt, bodies, self.body_q_prev, state.body_q, model.body_com, self.body_inv_mass_effective,
                self.friction_epsilon, self.body_body_contact_penalty_k, self.body_body_contact_normal_rho,
                self.body_body_contact_material_ke, self.body_body_contact_material_kd,
                self.body_body_contact_material_mu, self.body_body_contact_tangent_rho,
                self.body_body_contact_lambda, self.body_body_contact_C0, self.rigid_contact_alpha,
                self.rigid_contact_hard, self.rigid_compliant_alm, contacts.rigid_contact_count,
                contacts.rigid_contact_shape0, contacts.rigid_contact_shape1, contacts.rigid_contact_point0,
                contacts.rigid_contact_point1, contacts.rigid_contact_surface_velocity, contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1, contacts.rigid_contact_normal, contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1, model.shape_body, self.body_body_contact_buffer_pre_alloc,
                self.body_body_contact_counts, self.body_body_contact_indices], outputs=out,
                block_dim=_sv._select_rigid_block_dim(n * nt, dev), device=dev)
            ran = True
        return ran

    def _rod_joints(self, state, control, dt) -> None:
        m = self.model
        wp.launch(rod_external_joints, dim=self._pairs[0].size, device=self.device, inputs=[
            self._pairs[0], self._pairs[1], self._pairs[2], state.body_q, self.body_q_prev, m.body_q, m.body_com, m.joint_type,
            m.joint_enabled, m.joint_parent, m.joint_child, m.joint_X_p, m.joint_X_c, m.joint_axis,
            self.joint_rod_rest_kb_local, self.joint_rod_rest_twist, m.joint_qd_start, m.joint_target_q_start,
            self.joint_constraint_start, self.joint_penalty_k, self.joint_rho, self.joint_material_k,
            self.joint_penalty_kd, self.joint_sigma_start, self.joint_C_fric, m.joint_target_ke, m.joint_target_kd,
            control.joint_target_q, control.joint_target_qd, m.joint_limit_lower, m.joint_limit_upper,
            m.joint_limit_ke, m.joint_limit_kd, self.joint_drive_limit_support, self.joint_drive_lambda,
            self.joint_limit_lambda, self.joint_lambda_lin, self.joint_lambda_ang, self.joint_C0_lin,
            self.joint_C0_ang, self.joint_is_hard, self.rigid_joint_alpha, self.rigid_compliant_alm, m.joint_dof_dim,
            self.joint_rest_angle, self.joint_angle_prev, dt, self.body_forces, self.body_torques,
            self.body_hessian_ll, self.body_hessian_al, self.body_hessian_aa])


__all__ = ["SolverVBDRod", "hook_order_ok", "rest_relative"]
