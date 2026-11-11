# SPDX-License-Identifier: Apache-2.0
"""J2 plasticity for elastic rods (SCR-WP9-ROD Phase 2; WP7 reviews the use of its model): a wire that keeps a bend
or a twist.

**Model.** WP7's shared J2 fibre section (`alloy.solvers.plastic.j2.rod_section`): a polar grid of 12 Gauss-Lobatto
radii x 20 sectors (220 fibres), each in tension and shear, with linear isotropic hardening. First yield at
M_y = σ_y0 π R³/4 and T_y = π R³ τ_y / 2 is exact, and the section saturates at M_p = 4 σ_y0 R³/3.

**Mechanism.** Once per solver step, on the converged positions (explicit in time, as WP7's tets), for every ROD joint
of a yielding rod:
- the joint's strain from its authored rest A0 (`rest_relative`): kap = rot_vec(A0⁻¹ rel), with rel the child frame
  relative to the parent frame (as the chain kernels read it), and the axial strain u_z / h from the anchors;
- the section at curvatures κ = kap_xy / h and twist rate φ' = kap_z / h, with E = EI / I and G = GJ / J taken from
  the joint's own stiffness (kb h), so a composite cable's measured EI / GJ stay what the spring uses;
- the joint's rest becomes A = rel exp(-M ⊘ kb): the spring then sees exactly the elastic rotation M ⊘ kb and carries
  the section's moment M. A joint none of whose fibres has yielded keeps A0, bit for bit.
- The axial strain u_z / h assumes the joint's two anchors (X_p, X_c) coincide at rest, as the ROD_VBD emitter
  builds them; an offset anchor would read an axial strain of order 1.
- The section is round: E comes from the first bend stiffness. SolverVBD refuses a ROD joint whose two bend
  stiffnesses differ at all, at build and on `notify_model_changed`, so the second always equals it.
- A change to the authored rest (`notify_model_changed`) sets rest0 anew and keeps a yielded joint's plastic rest,
  so the joint does not spring toward its original shape for a step (WP7's review). A change of authored geometry on
  a yielded joint keeps its set as is: its fibres' plastic strain stays measured from the old A0.
The rest is derived from the fibres every step, not accumulated, so float32 drift does not build up in it.

**Which joints.** Per-joint attributes `plastic:rod_yield_stress` and `plastic:rod_hardening` (Pa) and the section's
`plastic:rod_radius` and `plastic:rod_length` (m, the joint spacing h), JOINT frequency, written by the ROD_VBD
emitter from a `plastic.model: j2` card (`mark_joints`). Joints with yield stress 0 stay elastic. With none above 0
the hook is never built: no arrays, no kernel, bitwise the elastic solver.

**State** (solver state: snapshots and rollbacks restore it with the rest of the entry): the fibres' `eps_p`, `gam_p`
and `ebar` (joint x fibre), and the joints' rest rotations. `reset` restores the authored state.

**Readout.** `ebar_max` per joint, the largest equivalent plastic strain over its fibres: `"<rod>.plastic_strain"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.solvers.plastic.j2 import Quad, fibres, rod_quad, rod_section
from alloy.solvers.rod.kernels import rot_vec

NAMESPACE = "plastic"
YIELD, HARDENING, RADIUS, LENGTH = "rod_yield_stress", "rod_hardening", "rod_radius", "rod_length"
PI = wp.constant(3.141592653589793)


def register_attributes(builder: Any) -> None:
    """Declare the per-joint rod plasticity attributes on a ModelBuilder (idempotent)."""
    from alloy._vendor import newton  # noqa: PLC0415

    for name in (YIELD, HARDENING, RADIUS, LENGTH):
        if f"{NAMESPACE}:{name}" in builder.custom_attributes:
            continue
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=name, namespace=NAMESPACE, frequency=newton.Model.AttributeFrequency.JOINT, dtype=wp.float32,
            default=0.0))


def mark_joints(builder: Any, joints: Any, yield_stress: float, hardening: float, radius: float,
                spacing: float) -> None:
    """Make the ROD joints `joints` of `builder` J2: σ_y0 = yield_stress, H = hardening [Pa]; section radius and
    joint spacing h [m]."""
    from alloy.solvers.rod.topology import ROD  # noqa: PLC0415

    register_attributes(builder)
    rod = [int(j) for j in joints if int(builder.joint_type[int(j)]) == ROD]      # not add_rod's free root joint
    for name, value in ((YIELD, yield_stress), (HARDENING, hardening), (RADIUS, radius), (LENGTH, spacing)):
        attr = builder.custom_attributes[f"{NAMESPACE}:{name}"]
        if attr.values is None:
            attr.values = {}
        for j in rod:
            attr.values[j] = float(value)


@wp.func
def quat_exp(v: wp.vec3) -> wp.quat:
    """The rotation of rotation vector v."""
    a = wp.length(v)
    if a < 1.0e-12:
        return wp.quat(0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0)
    return wp.quat_from_axis_angle(v / a, a)


@wp.kernel
def _rod_return(joints: wp.array(dtype=int), body_q: wp.array(dtype=wp.transform), parent: wp.array(dtype=int),
                child: wp.array(dtype=int), X_p: wp.array(dtype=wp.transform), X_c: wp.array(dtype=wp.transform),
                qd_start: wp.array(dtype=int), ke: wp.array(dtype=float), rest0: wp.array(dtype=wp.quat),
                sy0: wp.array(dtype=float), hmod: wp.array(dtype=float), radius: wp.array(dtype=float),
                length: wp.array(dtype=float), q: Quad, eps_p: wp.array2d(dtype=float),
                gam_p: wp.array2d(dtype=float), ebar: wp.array2d(dtype=float), rest: wp.array(dtype=wp.quat),
                ebar_max: wp.array(dtype=float)):
    t = wp.tid()
    j = joints[t]
    tp = body_q[parent[j]]
    tc = body_q[child[j]]
    qwp = wp.mul(wp.transform_get_rotation(tp), wp.transform_get_rotation(X_p[j]))
    qwc = wp.mul(wp.transform_get_rotation(tc), wp.transform_get_rotation(X_c[j]))
    rel = wp.mul(wp.quat_inverse(qwp), qwc)
    kap = rot_vec(wp.mul(wp.quat_inverse(rest0[j]), rel))
    d = wp.transform_point(tc, wp.transform_get_translation(X_c[j])) - wp.transform_point(
        tp, wp.transform_get_translation(X_p[j]))
    u = wp.quat_rotate_inv(qwp, d)
    h = length[j]
    R = radius[j]
    o = qd_start[j]
    kb = wp.vec3(ke[o + 3], ke[o + 4], ke[o + 5])
    inertia = 0.25 * PI * R * R * R * R
    E = kb[0] * h / inertia
    G = kb[2] * h / (2.0 * inertia)
    s = rod_section(j, u[2] / h, kap[0] / h, kap[1] / h, kap[2] / h, R, E, G, sy0[j], hmod[j], q, eps_p, gam_p, ebar)
    m = float(0.0)
    for f in range(ebar.shape[1]):
        m = wp.max(m, ebar[j, f])
    ebar_max[j] = m
    if m > 0.0:                                  # never yielded: the authored rest, bit for bit (module doc)
        th_e = wp.vec3(s[1] / kb[0], s[2] / kb[1], s[3] / kb[2])
        rest[j] = wp.mul(rel, quat_exp(-th_e))
    else:
        rest[j] = rest0[j]


@wp.kernel
def _reset(joints: wp.array(dtype=int), joint_world: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool),
           rest0: wp.array(dtype=wp.quat), eps_p: wp.array2d(dtype=float), gam_p: wp.array2d(dtype=float),
           ebar: wp.array2d(dtype=float), rest: wp.array(dtype=wp.quat), ebar_max: wp.array(dtype=float)):
    t = wp.tid()
    j = joints[t]
    w = joint_world[j]
    if w >= 0 and not mask[w]:
        return
    for f in range(ebar.shape[1]):
        eps_p[j, f] = 0.0
        gam_p[j, f] = 0.0
        ebar[j, f] = 0.0
    rest[j] = rest0[j]
    ebar_max[j] = 0.0


@wp.kernel
def _keep_plastic(joints: wp.array(dtype=int), old: wp.array(dtype=wp.quat), ebar_max: wp.array(dtype=float),
                  rest: wp.array(dtype=wp.quat)):
    t = wp.tid()
    j = joints[t]
    if ebar_max[j] > 0.0:
        rest[j] = old[j]


class RodPlastic:
    """The post-step J2 hook for one model's yielding rod joints (module doc). Build it with `RodPlastic.for_model`
    (None when no rod joint yields)."""

    def __init__(self, model: Any, rest: wp.array) -> None:
        attrs = getattr(model, NAMESPACE)
        dev = model.device
        self.model, self.rest = model, rest
        self.sy0, self.hmod = getattr(attrs, YIELD), getattr(attrs, HARDENING)
        self.radius, self.length = getattr(attrs, RADIUS), getattr(attrs, LENGTH)
        marked = np.flatnonzero(self.sy0.numpy() > 0.0).astype(np.int32)
        self.joints = wp.array(marked, dtype=int, device=dev)
        self.rest0 = wp.clone(rest)
        self.quad = rod_quad(dev)
        n, nf = int(model.joint_count), fibres(self.quad)
        self.eps_p = wp.zeros((n, nf), dtype=float, device=dev)
        self.gam_p = wp.zeros((n, nf), dtype=float, device=dev)
        self.ebar = wp.zeros((n, nf), dtype=float, device=dev)
        self.ebar_max = wp.zeros(n, dtype=float, device=dev)

    @classmethod
    def for_model(cls, model: Any, rest: wp.array) -> RodPlastic | None:
        attrs = getattr(model, NAMESPACE, None)
        sy0 = getattr(attrs, YIELD, None) if attrs is not None else None
        if sy0 is None or model.joint_count == 0 or not bool((sy0.numpy() > 0.0).any()):
            return None
        return cls(model, rest)

    def after_step(self, body_q: wp.array) -> None:
        m = self.model
        wp.launch(_rod_return, dim=self.joints.size, device=m.device, inputs=[
            self.joints, body_q, m.joint_parent, m.joint_child, m.joint_X_p, m.joint_X_c, m.joint_qd_start,
            m.joint_target_ke, self.rest0, self.sy0, self.hmod, self.radius, self.length, self.quad, self.eps_p,
            self.gam_p, self.ebar, self.rest, self.ebar_max])

    def authored_rest_changed(self, old_rest: wp.array) -> None:
        """The solver re-read the authored rest into `rest`: make it rest0 and put back the plastic rest of every
        joint that has yielded (`old_rest`, the rest before the re-read)."""
        wp.copy(self.rest0, self.rest)
        wp.launch(_keep_plastic, dim=self.joints.size, inputs=[self.joints, old_rest, self.ebar_max, self.rest],
                  device=self.model.device)

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Authored state for the worlds in `world_mask` (all when None)."""
        if world_mask is None:
            for a in (self.eps_p, self.gam_p, self.ebar, self.ebar_max):
                a.zero_()
            wp.copy(self.rest, self.rest0)
            return
        m = self.model
        wp.launch(_reset, dim=self.joints.size, device=m.device, inputs=[
            self.joints, m.joint_world, world_mask, self.rest0, self.eps_p, self.gam_p, self.ebar, self.rest,
            self.ebar_max])

    def state_arrays(self) -> dict[str, wp.array]:
        """The fibres. The plastic rest rotations are the solver's own `_rest_rel`; coupling/adapters.py snapshots
        it too."""
        return {"plastic_rod_eps_p": self.eps_p, "plastic_rod_gam_p": self.gam_p, "plastic_rod_ebar": self.ebar}


__all__ = ["HARDENING", "LENGTH", "NAMESPACE", "RADIUS", "YIELD", "RodPlastic", "mark_joints", "quat_exp",
           "register_attributes"]
