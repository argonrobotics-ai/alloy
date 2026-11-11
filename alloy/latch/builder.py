# SPDX-License-Identifier: Apache-2.0
"""ModelBuilder back end: add a ``LatchSpec`` to a ``newton.ModelBuilder`` (rigs, bench twins, hand-built scenes).

``add_latch`` mirrors ``latch_mjcf`` (same bodies, inertias, joints, springs, stops, excludes and kept contact
pairs), so a rig built here and a card scene built from the MJCF piece hold the same mechanism. The root is mounted
``free`` (dynamic default), ``fixed`` (welded; static default), on a ``slide`` stage along ``mount_axis``
(prescribed-motion rigs), or ``float``: driven along ``mount_axis`` and free in the two cross directions with the
orientation held (a tester's clamp: the jack, not the stage, takes the contact springs' reaction). The stage dof
carries a PD drive the caller sets through ``joint_target_*``.

Per-world variation (tolerance spreads): build each world from its own spec (``ModelBuilder.add_builder``), or
write the finalized Model arrays through the returned handles (dof ids for ``joint_stiffness``, coordinate ids for
``joint_rest_q``, shape ids for ``shape_material_mu``). Member springs use Newton's core joint spring (``stiffness``,
``rest_q``; Newton 1.8 deprecated the ``mujoco:dof_passive_stiffness`` / ``dof_springref`` inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Literal

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.latch.mass import MassProps, member_props, root_props, stop_solref
from alloy.latch.mjcf import body_name, joint_impedance, joint_name, pair_params
from alloy.latch.spec import ROOT, Box, LatchSpec

Mount = Literal["auto", "free", "fixed", "slide", "float"]
SOLREF_MODE_RAW = 1                     # newton/_src/solvers/mujoco/constants.py:32 (raw MuJoCo solreflimit)


@dataclass(frozen=True)
class LatchHandles:
    """Template indices of one added latch (offset them per world after ``replicate``)."""
    root: int
    root_joint: int
    bodies: dict[str, int]                      # part -> body
    joints: dict[str, int]                      # member -> joint
    dofs: dict[str, int]                        # member -> qd index (one dof each)
    coords: dict[str, int]                      # member -> q index
    shapes: dict[str, int]                      # box name -> shape
    roles: dict[str, tuple[int, ...]] = field(default_factory=dict)
    stage_dof: int | None = None                # the slide mount's dof
    stage_coord: int | None = None              # the slide mount's q index


def _ensure_mujoco_attributes(b: newton.ModelBuilder) -> None:
    if not b.has_custom_attribute("mujoco:solreflimit"):
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)


def _mat(p: MassProps) -> wp.mat33:
    return wp.mat33(*[float(x) for x in p.inertia.reshape(-1)])


def _tf(pos, quat=(0.0, 0.0, 0.0, 1.0)) -> wp.transform:
    return wp.transform(wp.vec3(*map(float, pos)), wp.quat(*map(float, quat)))


def _shapes(b: newton.ModelBuilder, body: int, boxes: tuple[Box, ...],
            cfg_for: Callable[[Box], newton.ModelBuilder.ShapeConfig], out: dict[str, int]) -> None:
    for box in boxes:
        cfg = cfg_for(box)
        if box.role == "visual" or not box.collide:
            cfg.has_shape_collision = False
            cfg.has_particle_collision = False
        cfg.density = 0.0
        out[box.name] = b.add_shape_box(body, xform=_tf(box.pos, box.quat), hx=box.half[0], hy=box.half[1],
                                        hz=box.half[2], cfg=cfg, label=box.name)


def _default_cfg(_: Box) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=0.0)


def _member_joint(b: newton.ModelBuilder, spec: LatchSpec, name: str, parent: int, child: int,
                  imp: tuple) -> int:
    """``imp`` = (armature, flat solimplimit or None, limit margin) from ``mjcf.joint_impedance`` (as the MJCF)."""
    m = spec.member(name)
    k, d = stop_solref(spec)
    arm, si, margin = imp
    attrs = {"mujoco:solreflimit": wp.vec2(k, d), "mujoco:solreflimit_mode": SOLREF_MODE_RAW,
             "mujoco:limit_margin": float(margin)}
    if si is not None:
        attrs["mujoco:solimplimit"] = b.custom_attributes["mujoco:solimplimit"].dtype(*si)
    add = b.add_joint_revolute if m.kind == "hinge" else b.add_joint_prismatic
    return add(parent, child, parent_xform=_tf(m.anchor), child_xform=wp.transform_identity(),
               axis=wp.vec3(*m.axis), damping=float(m.spring.damping), stiffness=float(m.spring.rate),
               rest_q=float(m.spring.springref), limit_lower=0.0, limit_upper=float(m.travel),
               armature=float(arm), label=joint_name(spec, name), custom_attributes=attrs)


def _root_joint(b: newton.ModelBuilder, spec: LatchSpec, root: int, xform: wp.transform, mount: str,
                mount_axis: tuple[float, float, float], stage: tuple[float, float] | None,
                cross: tuple[float, float] = (5000.0, 25.0)) -> int:
    if mount == "free":
        return b.add_joint_free(root, parent_xform=xform, child_xform=wp.transform_identity())
    if mount == "fixed":
        return b.add_joint_fixed(-1, root, parent_xform=xform, child_xform=wp.transform_identity())
    if mount == "slide":
        drive = {} if stage is None else {"target_ke": float(stage[0]), "target_kd": float(stage[1]),
                                          "actuator_mode": newton.JointTargetMode.POSITION}
        return b.add_joint_prismatic(-1, root, parent_xform=xform, child_xform=wp.transform_identity(),
                                     axis=wp.vec3(*mount_axis), label=f"{spec.name}_stage", **drive)
    if mount == "float":       # driven along mount_axis, free in the two cross directions, orientation held
        J = newton.ModelBuilder.JointDofConfig
        a = np.asarray(mount_axis, np.float64)
        c1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [0.0, 1.0, 0.0])
        c1 /= np.linalg.norm(c1)
        c2 = np.cross(a, c1)
        drive = {} if stage is None else {"target_ke": float(stage[0]), "target_kd": float(stage[1]),
                                          "actuator_mode": newton.JointTargetMode.POSITION}
        soft = {"target_ke": float(cross[0]), "target_kd": float(cross[1]),
                "actuator_mode": newton.JointTargetMode.POSITION}      # weak centring: holds the start pose in air
        lin = [J(axis=wp.vec3(*a), **drive), J(axis=wp.vec3(*c1), **soft), J(axis=wp.vec3(*c2), **soft)]
        return b.add_joint_d6(-1, root, linear_axes=lin, parent_xform=xform, child_xform=wp.transform_identity(),
                              label=f"{spec.name}_stage")
    raise ValueError(f"mount {mount!r}")


def add_latch(b: newton.ModelBuilder, spec: LatchSpec, xform: wp.transform | None = None, *, mount: Mount = "auto",
              mount_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
              cfg_for: Callable[[Box], newton.ModelBuilder.ShapeConfig] | None = None,
              stage: tuple[float, float] | None = None) -> LatchHandles:
    """Add ``spec`` at ``xform`` (root frame in the world). ``cfg_for(box)`` gives each shape's ShapeConfig (a fresh
    one per call; density is forced to 0, mass comes from the spec). ``stage = (ke, kd)`` puts a position drive
    on a ``slide`` mount (set ``control.joint_target_q`` of ``handles.stage_coord``). Returns template indices."""
    _ensure_mujoco_attributes(b)
    xform = xform if xform is not None else wp.transform_identity()
    mount = ("fixed" if spec.static else "free") if mount == "auto" else mount
    cfg_for = cfg_for or _default_cfg
    rp = root_props(spec)
    root = b.add_link(xform=xform, mass=float(rp.mass), com=wp.vec3(*rp.com), inertia=_mat(rp),
                      label=body_name(spec, ROOT))
    bodies, shapes, joints, world = {ROOT: root}, {}, {}, {ROOT: xform}
    _shapes(b, root, spec.root_shapes, cfg_for, shapes)
    jlist = [_root_joint(b, spec, root, xform, mount, mount_axis, stage)]
    imp = joint_impedance(spec)
    for m in spec.members:
        wx = wp.transform_multiply(world[m.parent], _tf(m.anchor))
        p = member_props(m, spec.design_dt)
        body = b.add_link(xform=wx, mass=float(p.mass), com=wp.vec3(*p.com), inertia=_mat(p),
                          label=body_name(spec, m.name))
        _shapes(b, body, m.shapes, cfg_for, shapes)
        joints[m.name] = _member_joint(b, spec, m.name, bodies[m.parent], body, imp[m.name])
        jlist.append(joints[m.name])
        bodies[m.name], world[m.name] = body, wx
    b.add_articulation(jlist, label=spec.name)
    for c in spec.couplings:
        if any(abs(x) > 0.0 for x in c.polycoef[2:]):
            raise ValueError(f"{spec.name}: builder couplings are linear (polycoef[2:] must be 0), got {c.polycoef}")
        b.set_joint_mimic(joints[c.follower], joints[c.leader], coeffs=(c.polycoef[0], c.polycoef[1]))
    _exclude_within(b, spec, shapes)
    _kept_contacts(b, spec, shapes)
    return _handles(b, spec, root, jlist[0], bodies, joints, shapes, mount)


def _exclude_within(b: newton.ModelBuilder, spec: LatchSpec, shapes: dict[str, int]) -> None:
    """No contact between any two parts of one assembly (CON-47; the parent filter misses welded roots)."""
    by_part = {ROOT: [shapes[x.name] for x in spec.root_shapes]}
    by_part.update({m.name: [shapes[x.name] for x in m.shapes] for m in spec.members})
    for pa, pb in combinations(by_part, 2):
        for sa in by_part[pa]:
            for sb in by_part[pb]:
                b.add_shape_collision_filter_pair(sa, sb)


def _kept_contacts(b: newton.ModelBuilder, spec: LatchSpec, shapes: dict[str, int]) -> None:
    """``spec.contacts`` as ``mujoco:pair_*`` rows (an explicit pair bypasses the filter pairs above), as the MJCF."""
    for c in spec.contacts:
        p = pair_params(spec, c)
        b.add_custom_values(**{
            "mujoco:pair_world": b.current_world, "mujoco:pair_geom1": shapes[c.box1],
            "mujoco:pair_geom2": shapes[c.box2], "mujoco:pair_condim": p["condim"], "mujoco:pair_solref": p["solref"],
            "mujoco:pair_solimp": p["solimp"], "mujoco:pair_friction": p["friction"],
            "mujoco:pair_margin": p["margin"], "mujoco:pair_gap": p["gap"]})


def _handles(b: newton.ModelBuilder, spec: LatchSpec, root: int, root_joint: int, bodies: dict, joints: dict,
             shapes: dict, mount: str) -> LatchHandles:
    roles: dict[str, list[int]] = {}
    for part in (ROOT, *[m.name for m in spec.members]):
        for box in spec.part_shapes(part):
            roles.setdefault(box.role, []).append(shapes[box.name])
    dofs = {n: int(b.joint_qd_start[j]) for n, j in joints.items()}
    coords = {n: int(b.joint_q_start[j]) for n, j in joints.items()}
    slide = mount in ("slide", "float")
    return LatchHandles(root, root_joint, bodies, joints, dofs, coords, shapes, {k: tuple(v) for k, v in roles.items()},
                        int(b.joint_qd_start[root_joint]) if slide else None,
                        int(b.joint_q_start[root_joint]) if slide else None)
