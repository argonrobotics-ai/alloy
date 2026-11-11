# SPDX-License-Identifier: Apache-2.0
"""Crumble runtime: source tables, triggers, per-world commit, particle conversion and body parking.

See crumble.py for the method. All kernels are per source or per particle, with no float atomics. The commit
shares the per-world event cap with bond breaks; crumbles that would exceed it count as overflow and are retried
next step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.bonds import NS
from alloy.topology.crumble import TRIGGERS
from alloy.topology.geometry import skew
from alloy.topology.tables import BODY_KINDS, item_worlds

log = logging.getLogger("alloy.topology")

ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
RELEASE = wp.constant(TRIGGERS["release"])


@dataclass
class CrumbleTables:
    count: int
    body: wp.array            # [S] int32
    park_eq: wp.array         # [S] int32 Newton equality row (−1 = none)
    trigger: wp.array         # [S] int32
    q_start: wp.array         # [S] int32 free-joint coordinate start
    qd_start: wp.array        # [S] int32
    joint: wp.array           # [S] int32
    slot: wp.array            # [S] int32 rank of the source in its world (parking slot)
    part_start: wp.array      # [S+1]
    part: wp.array            # [P] particle ids
    part_src: wp.array        # [P] source of each listed particle
    bond_start: wp.array      # [S+1] body bonds incident to the chunk (release trigger)
    bond: wp.array
    world_start: wp.array     # [W+1]
    world_order: wp.array     # [S]
    done: wp.array            # [S] int8
    request: wp.array         # [S] int8 manual trigger
    want: wp.array            # [S] int8
    commit: wp.array          # [S] int8


def _free_joints(model: newton.Model, bodies: np.ndarray) -> np.ndarray:
    child, jtype = model.joint_child.numpy(), model.joint_type.numpy()
    out = np.full(len(bodies), -1, np.int64)
    for k, b in enumerate(bodies):
        j = np.flatnonzero((child == b) & (jtype == int(newton.JointType.FREE)))
        if len(j) != 1:
            raise AssemblyError(f"crumble source body {int(b)} needs exactly one free joint (found {len(j)})")
        out[k] = j[0]
    return out


def _csr(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    start = np.concatenate([[0], np.cumsum([len(g) for g in groups])]).astype(np.int32)
    flat = np.concatenate(groups).astype(np.int32) if groups and start[-1] else np.zeros(0, np.int32)
    return start, flat


def build_crumble_tables(model: newton.Model, bond_kind: np.ndarray, bond_a: np.ndarray, bond_b: np.ndarray,
                         world_count: int) -> CrumbleTables | None:
    ns = getattr(model, NS, None)
    if ns is None or getattr(ns, "crumble_park_eq", None) is None or not model.body_count:
        return None
    park = ns.crumble_park_eq.numpy()
    bodies = np.flatnonzero(park >= 0)
    if len(bodies) == 0:
        return None
    joints = _free_joints(model, bodies)
    owner = ns.crumble_body.numpy() if model.particle_count else np.zeros(0, np.int64)
    parts = [np.flatnonzero(owner == b) for b in bodies]
    is_body = np.isin(bond_kind, BODY_KINDS)
    bonds = [np.flatnonzero(is_body & ((bond_a == b) | (bond_b == b))) for b in bodies]
    world = item_worlds(model.body_world.numpy()[bodies], world_count, "crumble source")
    order = np.argsort(world, kind="stable")
    slot = np.zeros(len(bodies), np.int32)
    for w in np.unique(world):
        slot[np.flatnonzero(world == w)] = np.arange(int((world == w).sum()))
    _check_mass(model, bodies, parts)
    ps, pf = _csr(parts)
    bs, bf = _csr(bonds)
    src = np.repeat(np.arange(len(bodies)), [len(p) for p in parts]).astype(np.int32)
    ws = np.concatenate([[0], np.cumsum(np.bincount(world, minlength=world_count)[:world_count])]).astype(np.int32)
    dev = model.device
    i32 = lambda x: wp.array(np.asarray(x, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
    i8 = lambda: wp.zeros(len(bodies), dtype=wp.int8, device=dev)  # noqa: E731
    return CrumbleTables(
        count=len(bodies), body=i32(bodies), park_eq=i32(park[bodies]), trigger=i32(ns.crumble_trigger.numpy()[bodies]),
        q_start=i32(model.joint_q_start.numpy()[joints]), qd_start=i32(model.joint_qd_start.numpy()[joints]),
        joint=i32(joints), slot=i32(slot), part_start=i32(ps), part=i32(pf), part_src=i32(src), bond_start=i32(bs),
        bond=i32(bf if len(bf) else np.zeros(1)), world_start=i32(ws), world_order=i32(order), done=i8(),
        request=i8(), want=i8(), commit=i8())


def _check_mass(model: newton.Model, bodies: np.ndarray, parts: list[np.ndarray]) -> None:
    bm, pm = model.body_mass.numpy().astype(np.float64), model.particle_mass.numpy().astype(np.float64)
    for b, p in zip(bodies, parts):
        if len(p) == 0:
            raise AssemblyError(f"crumble source body {int(b)} has no spare particles")
        rel = abs(pm[p].sum() - bm[b]) / bm[b]
        if rel > 1e-6:
            log.warning("crumble body %d: spare mass %.9g kg vs body mass %.9g kg (rel %.2e); the ledger will show it",
                        int(b), pm[p].sum(), bm[b], rel)


@wp.kernel
def crumble_triggers(trigger: wp.array[wp.int32], done: wp.array[wp.int8], request: wp.array[wp.int8],
                     bond_start: wp.array[wp.int32], bond: wp.array[wp.int32], intact: wp.array[wp.int8],
                     want: wp.array[wp.int8]):
    s = wp.tid()
    want[s] = wp.int8(0)
    if done[s] != 0:
        return
    if request[s] != 0:
        want[s] = wp.int8(1)
        return
    n = bond_start[s + 1] - bond_start[s]
    if trigger[s] != RELEASE or n == 0:
        return
    for k in range(bond_start[s], bond_start[s + 1]):
        if intact[bond[k]] != 0:
            return
    want[s] = wp.int8(1)


@wp.kernel
def commit_crumbles(world_start: wp.array[wp.int32], world_order: wp.array[wp.int32], want: wp.array[wp.int8],
                    done: wp.array[wp.int8], request: wp.array[wp.int8], cap: int, commit: wp.array[wp.int8],
                    step_events: wp.array[wp.int32], overflow: wp.array[wp.int32]):
    w = wp.tid()
    n = step_events[w]
    over = int(0)
    for k in range(world_start[w], world_start[w + 1]):
        s = world_order[k]
        commit[s] = wp.int8(0)
        if want[s] != 0:
            if n < cap:
                commit[s] = wp.int8(1)
                done[s] = wp.int8(1)
                request[s] = wp.int8(0)
                n += 1
            else:
                over += 1
    step_events[w] = n
    overflow[w] = overflow[w] + over


@wp.struct
class FlagTargets:
    n: int                          # how many of f0..f3 are real
    f0: wp.array[wp.int32]
    f1: wp.array[wp.int32]
    f2: wp.array[wp.int32]
    f3: wp.array[wp.int32]


@wp.struct
class MpmTargets:
    use: int
    qd_grad: wp.array[wp.mat33]
    elastic_strain: wp.array[wp.mat33]
    transform: wp.array[wp.mat33]
    stress: wp.array[wp.mat33]
    jp: wp.array[float]
    jp0: wp.array[float]
    volume: wp.array[float]           # solver-cached particle volume (source)
    material_volume: wp.array[float]  # solver-cached material volume (zeroed for inactive particles)


@wp.func
def set_active(f: wp.array[wp.int32], i: int):
    f[i] = f[i] | ACTIVE


@wp.kernel
def convert_particles(part: wp.array[wp.int32], part_src: wp.array[wp.int32], commit: wp.array[wp.int8],
                      body: wp.array[wp.int32], local: wp.array[wp.vec3], body_q: wp.array[wp.transform],
                      body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], x: wp.array[wp.vec3],
                      v: wp.array[wp.vec3], flags: FlagTargets, mpm: MpmTargets):
    j = wp.tid()
    s = part_src[j]
    if commit[s] == 0:
        return
    i = part[j]
    b = body[s]
    xf = body_q[b]
    om = wp.spatial_bottom(body_qd[b])
    xi = wp.transform_point(xf, local[i])
    x[i] = xi
    v[i] = wp.spatial_top(body_qd[b]) + wp.cross(om, xi - wp.transform_point(xf, body_com[b]))
    set_active(flags.f0, i)
    if flags.n > 1:
        set_active(flags.f1, i)
    if flags.n > 2:
        set_active(flags.f2, i)
    if flags.n > 3:
        set_active(flags.f3, i)
    if mpm.use != 0:
        mpm.qd_grad[i] = skew(om)
        mpm.elastic_strain[i] = wp.identity(n=3, dtype=float)
        mpm.transform[i] = wp.identity(n=3, dtype=float)
        mpm.stress[i] = wp.mat33(0.0)
        mpm.jp[i] = mpm.jp0[i]
        mpm.material_volume[i] = mpm.volume[i]


@wp.kernel
def park_bodies(commit: wp.array[wp.int8], body: wp.array[wp.int32], joint: wp.array[wp.int32],
                q_start: wp.array[wp.int32], qd_start: wp.array[wp.int32], slot: wp.array[wp.int32],
                park_eq: wp.array[wp.int32], offset: wp.vec3, joint_x_p: wp.array[wp.transform],
                joint_x_c: wp.array[wp.transform], body_q: wp.array[wp.transform],
                body_qd: wp.array[wp.spatial_vector], joint_q: wp.array[float], joint_qd: wp.array[float],
                eq_enabled: wp.array[wp.bool], eq_relpose: wp.array[wp.transform], body_active: wp.array[wp.int8]):
    s = wp.tid()
    if commit[s] == 0:
        return
    b = body[s]
    xf = body_q[b]
    park = wp.transform(wp.transform_get_translation(xf) + offset * float(1 + slot[s]), wp.transform_get_rotation(xf))
    body_q[b] = park
    body_qd[b] = wp.spatial_vector()
    j = joint[s]
    xj = wp.transform_inverse(joint_x_p[j]) * park * joint_x_c[j]
    q0 = q_start[s]
    p = wp.transform_get_translation(xj)
    r = wp.transform_get_rotation(xj)
    for k in range(3):
        joint_q[q0 + k] = p[k]
    for k in range(4):
        joint_q[q0 + 3 + k] = r[k]
    for k in range(6):
        joint_qd[qd_start[s] + k] = 0.0
    e = park_eq[s]
    if e >= 0:
        eq_enabled[e] = True
        eq_relpose[e] = park
    body_active[b] = wp.int8(0)
