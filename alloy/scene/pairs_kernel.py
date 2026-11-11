# SPDX-License-Identifier: Apache-2.0
"""Warp kernel behind `PairTable.apply` (see pairs.py for the formulas). One thread per rigid contact slot; slots
past `rigid_contact_count` are skipped; no atomics, no host sync (capture-safe, deterministic)."""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton import Contacts, State

from alloy.scene.derive import TINY_DAMPING, TINY_SCALE
from alloy.scene.types import PairTable

PI = 3.141592653589793


@wp.func
def _mu_mix(rule: int, a: float, b: float) -> float:
    if rule == 0:
        return wp.max(a, b)
    if rule == 1:
        return 0.5 * (a + b)
    if rule == 2:
        return wp.sqrt(wp.max(a, 0.0) * wp.max(b, 0.0))
    if rule == 3:
        if a + b <= 0.0:
            return 0.0
        return 2.0 * a * b / (a + b)
    return 0.0


@wp.func
def _zeta(e: float) -> float:
    if e <= 0.0:
        return 1.0
    if e >= 1.0:
        return 0.0
    le = wp.log(e)
    return wp.min(1.0, -le / wp.sqrt(PI * PI + le * le))


@wp.func
def _side_inv_mass(body: int, point: wp.vec3, n: wp.vec3, has_state: int, body_q: wp.array[wp.transform],
                   inv_mass: wp.array[float], inv_inertia: wp.array[wp.mat33], com: wp.array[wp.vec3]) -> float:
    """Inverse mass of one body along the world normal n at a body-frame contact point."""
    if body < 0:
        return 0.0
    w = inv_mass[body]
    if has_state == 0 or w == 0.0:
        return w
    q = wp.transform_get_rotation(body_q[body])
    nb = wp.quat_rotate_inv(q, n)
    rn = wp.cross(point - com[body], nb)
    return w + wp.dot(rn, inv_inertia[body] * rn)


@wp.kernel
def pair_kernel(
    count: wp.array[wp.int32], shape0: wp.array[wp.int32], shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3], point1: wp.array[wp.vec3], normal: wp.array[wp.vec3],
    shape_material: wp.array[wp.int32], shape_mu: wp.array[float], shape_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32], shape_imp: wp.array[float], has_imp: int,
    table: wp.array2d[wp.vec4], stiffness: wp.array2d[float],
    world_table: wp.array3d[wp.vec4], world_stiffness: wp.array3d[float], has_world: int,
    override_slot: wp.array[wp.int32], override_k: wp.array2d[float], has_override: int, precision_external: int,
    e_correction: wp.array[float], rule: int,
    has_state: int, body_q: wp.array[wp.transform], inv_mass: wp.array[float], inv_inertia: wp.array[wp.mat33],
    com: wp.array[wp.vec3],
    out_friction: wp.array[float], out_stiffness: wp.array[float], out_damping: wp.array[float],
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    s0, s1 = shape0[tid], shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    m0, m1 = shape_material[s0], shape_material[s1]
    if m0 < 0 or m1 < 0:
        return
    world = shape_world[s0]
    if world < 0:
        world = shape_world[s1]
    vals = table[m0, m1]
    k = stiffness[m0, m1]
    if has_world != 0 and world >= 0:
        vals = world_table[world, m0, m1]
        k = world_stiffness[world, m0, m1]
    precision_row = int(0)
    if has_override != 0:
        a, b = override_slot[s0], override_slot[s1]
        if a >= 0 and b >= 0:
            if override_k[a, b] > 0.0:
                k = override_k[a, b]
                precision_row = 1
    mix = _mu_mix(rule, shape_mu[s0], shape_mu[s1])
    scale = float(0.0)
    if mix > 0.0:
        scale = wp.max(vals[1] / mix, TINY_SCALE)
    out_friction[tid] = scale
    if precision_row != 0 and precision_external != 0:
        return   # the contact layer wrote K and damping for precision rows (sdf weights / hydroelastic), WP3
    out_stiffness[tid] = k
    nb = e_correction.shape[0]
    e = vals[2]
    if nb > 0:
        idx = wp.min(nb - 1, int(e * float(nb - 1) + 0.5))
        e = wp.clamp(e * e_correction[idx], 0.0, 1.0)
    n = normal[tid]
    w = _side_inv_mass(shape_body[s0], point0[tid], n, has_state, body_q, inv_mass, inv_inertia, com)
    w = w + _side_inv_mass(shape_body[s1], point1[tid], n, has_state, body_q, inv_mass, inv_inertia, com)
    kd = float(TINY_DAMPING)
    if w > 0.0 and k > 0.0:
        kd = 2.0 * _zeta(e) * wp.sqrt(k / w)
        if has_imp != 0:
            imp = 0.5 * (shape_imp[s0] + shape_imp[s1])
            kd = kd / wp.sqrt(wp.max(1.0 - imp, 1.0e-6))
        kd = wp.max(kd, TINY_DAMPING)
    out_damping[tid] = kd


def _dummies(device) -> dict[str, wp.array]:
    return {
        "f1": wp.zeros(1, dtype=float, device=device), "i1": wp.zeros(1, dtype=wp.int32, device=device),
        "v4": wp.zeros((1, 1, 1), dtype=wp.vec4, device=device), "f3": wp.zeros((1, 1, 1), dtype=float, device=device),
        "f2": wp.zeros((1, 1), dtype=float, device=device), "tf": wp.zeros(1, dtype=wp.transform, device=device),
        "m3": wp.zeros(1, dtype=wp.mat33, device=device), "v3": wp.zeros(1, dtype=wp.vec3, device=device),
    }


_DUMMY: dict[str, dict[str, wp.array]] = {}


def launch_pair_kernel(table: PairTable, contacts: Contacts, state: State | None,
                       precision_external: bool = True) -> None:
    model = table.model
    dev = model.device
    d = _DUMMY.setdefault(str(dev), _dummies(dev))
    imp = getattr(table, "shape_imp", None)
    wt, ws = getattr(table, "world_table", None), getattr(table, "world_stiffness", None)
    oslot, ok = getattr(table, "override_slot", None), getattr(table, "override_k", None)
    has_world = wt is not None and ws is not None
    has_override = oslot is not None and ok is not None
    bodies = model.body_count > 0
    wp.launch(pair_kernel, dim=contacts.rigid_contact_max, device=dev, inputs=[
        contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
        contacts.rigid_contact_point0, contacts.rigid_contact_point1, contacts.rigid_contact_normal,
        table.shape_material, table.shape_mu_mix, model.shape_world, model.shape_body,
        imp if imp is not None else d["f1"], int(imp is not None),
        table.table, table.stiffness, wt if has_world else d["v4"], ws if has_world else d["f3"], int(has_world),
        oslot if has_override else d["i1"], ok if has_override else d["f2"], int(has_override),
        int(precision_external),
        table.e_correction, int(table.mu_mix_rule),
        int(state is not None and bodies), state.body_q if state is not None and bodies else d["tf"],
        model.body_inv_mass if bodies else d["f1"], model.body_inv_inertia if bodies else d["m3"],
        model.body_com if bodies else d["v3"],
    ], outputs=[contacts.rigid_contact_friction, contacts.rigid_contact_stiffness, contacts.rigid_contact_damping])
