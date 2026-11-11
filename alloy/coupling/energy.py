# SPDX-License-Identifier: Apache-2.0
"""Per-world energy bookkeeping on the parent model (SPEC D20, §3.6 "Energies").

- PE datum: ``z_ref[w]`` = lowest dynamic body COM / particle height at reset, along ``up = -g/|g|``;
  ``E = KE + Σ m·|g|·(h − z_ref)``, so E may go negative when things fall below their reset height.
- ``E_scale = max(E₀, m_tot·|g|·L_char, 1e-3 J)``; ``L_char`` = largest AABB extent of the world's dynamic items
  (body COMs padded by their shapes' bounding radius, particles by their radius) at reset.
- dynamic items: particles and bodies with mass, except bodies welded to the world (``kinematic.world_welded_bodies``:
  a fixed-joint chain to the world). Those cannot move, so they carry no energy the ledger could track, and counting
  them let a scene's static set dress dominate the scale (SCR-SC4-4b: C16's 1.8 t, 8.5 m floor made E_scale
  154 kJ, so 37 J the coupler pumped into a finger never warned).
- external work, midpoint/displacement rule: ``W_ext += f·Δx + dt·τ·ω_out`` over applied body wrenches,
  ``f_p·Δx_p`` over particle forces, ``dt·joint_f·(q̇ₙ + q̇ₙ₊₁)/2`` over joint forces.

Per-world sums loop items in index order (float64), so values are bitwise batch-invariant (D9): the energy in
chunks of each world's bodies-then-particles sequence, then over its chunks in order (``alloy.precision.chunk_sums``;
a world with at most CHUNK items sums exactly as one serial loop), the rest one thread per world. Items of the
global world (-1) are ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.precision.chunk_sums import Chunks, chunk_count, chunk_sum, make_chunks

E_SCALE_FLOOR = wp.constant(wp.float64(1.0e-3))


@wp.func
def world_gravity(gravity: wp.array[wp.vec3], w: int) -> wp.vec3:
    if gravity.shape[0] > 1:
        return gravity[w]
    return gravity[0]


@wp.func
def up_axis(g: wp.vec3) -> wp.vec3:
    n = wp.length(g)
    if n > 0.0:
        return -g / n
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def body_com_world(q: wp.transform, com: wp.vec3) -> wp.vec3:
    return wp.transform_point(q, com)


@wp.struct
class EnergyInputs:
    body_start: wp.array[wp.int32]
    particle_start: wp.array[wp.int32]
    body_mass: wp.array[float]
    body_inertia: wp.array[wp.mat33]
    body_com: wp.array[wp.vec3]
    body_radius: wp.array[float]
    body_dynamic: wp.array[wp.int32]        # 0: welded to the world (not a dynamic item), 1 otherwise
    particle_mass: wp.array[float]
    particle_radius: wp.array[float]
    gravity: wp.array[wp.vec3]
    chunks: Chunks                          # each world's bodies-then-particles sequence in chunks (chunk_sums)
    e_part: wp.array[wp.float64]            # [chunks] scratch: one chunk's energy, max speed, non-finite flag
                                            # (one per EnergyInputs: launches in order on one stream)
    v_part: wp.array[float]
    bad_part: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def reference_kernel(
    inp: EnergyInputs,
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    z_ref: wp.array[wp.float64],
    m_tot: wp.array[wp.float64],
    l_char: wp.array[wp.float64],
):
    """Per world: lowest dynamic height (datum), total dynamic mass and largest AABB extent."""
    w = wp.tid()
    up = up_axis(world_gravity(inp.gravity, w))
    lo = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    hi = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    hmin = float(1.0e30)
    mass = wp.float64(0.0)
    for b in range(inp.body_start[w], inp.body_start[w + 1]):
        m = inp.body_mass[b]
        if m > 0.0 and inp.body_dynamic[b] != 0:
            p = body_com_world(body_q[b], inp.body_com[b])
            r = inp.body_radius[b]
            lo = wp.min(lo, p - wp.vec3(r, r, r))
            hi = wp.max(hi, p + wp.vec3(r, r, r))
            hmin = wp.min(hmin, wp.dot(p, up))
            mass = mass + wp.float64(m)
    for i in range(inp.particle_start[w], inp.particle_start[w + 1]):
        m = inp.particle_mass[i]
        if m > 0.0:
            p = particle_q[i]
            r = inp.particle_radius[i]
            lo = wp.min(lo, p - wp.vec3(r, r, r))
            hi = wp.max(hi, p + wp.vec3(r, r, r))
            hmin = wp.min(hmin, wp.dot(p, up))
            mass = mass + wp.float64(m)
    if mass > wp.float64(0.0):
        ext = hi - lo
        z_ref[w] = wp.float64(hmin)
        l_char[w] = wp.float64(wp.max(wp.max(ext[0], ext[1]), ext[2]))
    else:
        z_ref[w] = wp.float64(0.0)
        l_char[w] = wp.float64(0.0)
    m_tot[w] = mass


@wp.kernel(enable_backward=False)
def energy_chunks(
    inp: EnergyInputs,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    z_ref: wp.array[wp.float64],
):
    """One chunk of a world's bodies-then-particles sequence: E = KE + PE about z_ref (float64, item order), the max
    linear speed and a non-finite flag (``chunk_sums``)."""
    c = wp.tid()
    w = inp.chunks.world[c]
    g = world_gravity(inp.gravity, w)
    gmag = wp.float64(wp.length(g))
    up = up_axis(g)
    zr = z_ref[w]
    nb = inp.body_start[w + 1] - inp.body_start[w]
    lo = inp.chunks.lo[c]
    hi = inp.chunks.hi[c]
    e = wp.float64(0.0)
    vmax = float(0.0)
    bad = int(0)
    for p in range(lo, wp.min(hi, nb)):
        b = inp.body_start[w] + p
        m = inp.body_mass[b]
        if m > 0.0 and inp.body_dynamic[b] != 0:
            q = body_q[b]
            v = wp.spatial_top(body_qd[b])
            om = wp.quat_rotate_inv(wp.transform_get_rotation(q), wp.spatial_bottom(body_qd[b]))
            h = wp.float64(wp.dot(body_com_world(q, inp.body_com[b]), up))
            ke = 0.5 * m * wp.dot(v, v) + 0.5 * wp.dot(om, inp.body_inertia[b] * om)
            e = e + wp.float64(ke) + wp.float64(m) * gmag * (h - zr)
            s = wp.length(v)
            vmax = wp.max(vmax, s)
            if not (wp.isfinite(s) and wp.isfinite(ke) and wp.isfinite(h)):
                bad = 1
    for p in range(wp.max(lo, nb), hi):
        i = inp.particle_start[w] + p - nb
        m = inp.particle_mass[i]
        if m > 0.0:
            v = particle_qd[i]
            h = wp.float64(wp.dot(particle_q[i], up))
            e = e + wp.float64(0.5 * m * wp.dot(v, v)) + wp.float64(m) * gmag * (h - zr)
            s = wp.length(v)
            vmax = wp.max(vmax, s)
            if not (wp.isfinite(s) and wp.isfinite(h)):
                bad = 1
    inp.e_part[c] = e
    inp.v_part[c] = vmax
    inp.bad_part[c] = bad


@wp.kernel(enable_backward=False)
def energy_sum(inp: EnergyInputs, energy: wp.array[wp.float64], max_speed: wp.array[float],
               nonfinite: wp.array[wp.int32]):
    """Per world: its chunks' energies added in chunk order, the max of their speeds, any non-finite flag."""
    w = wp.tid()
    vmax = float(0.0)
    bad = int(0)
    for c in range(inp.chunks.start[w], inp.chunks.start[w + 1]):
        vmax = wp.max(vmax, inp.v_part[c])
        bad = wp.max(bad, inp.bad_part[c])
    energy[w] = chunk_sum(inp.chunks, inp.e_part, w)
    max_speed[w] = vmax
    nonfinite[w] = bad


def launch_energy(inp: EnergyInputs, state: Any, z_ref: wp.array, energy: wp.array, max_speed: wp.array,
                  nonfinite: wp.array, device) -> None:
    """Per world [W]: E = KE + PE about z_ref (float64, fixed order), the max linear speed, a non-finite flag."""
    n = chunk_count(inp.chunks)
    if n:
        wp.launch(energy_chunks, dim=n, inputs=[inp, state.body_q, state.body_qd, state.particle_q,
                                                state.particle_qd, z_ref], device=device)
    wp.launch(energy_sum, dim=energy.shape[0], inputs=[inp, energy, max_speed, nonfinite], device=device)


@wp.kernel(enable_backward=False)
def scale_kernel(
    gravity: wp.array[wp.vec3],
    energy: wp.array[wp.float64],
    m_tot: wp.array[wp.float64],
    l_char: wp.array[wp.float64],
    energy_ref: wp.array[wp.float64],
    e_scale: wp.array[wp.float64],
):
    """E_ref = E₀; E_scale = max(E₀, m_tot·|g|·L_char, 1e-3 J) (D20, PE_min = 0 at the datum)."""
    w = wp.tid()
    gmag = wp.float64(wp.length(world_gravity(gravity, w)))
    energy_ref[w] = energy[w]
    e_scale[w] = wp.max(wp.max(energy[w], m_tot[w] * gmag * l_char[w]), E_SCALE_FLOOR)


@wp.struct
class WorkInputs:
    body_start: wp.array[wp.int32]
    particle_start: wp.array[wp.int32]
    dof_start: wp.array[wp.int32]
    body_com: wp.array[wp.vec3]
    chunks: Chunks                          # each world's bodies-then-particles sequence in chunks (chunk_sums)
    w_part: wp.array[wp.float64]            # [chunks] scratch: one chunk's work (one stream, chunk_sums)


@wp.kernel(enable_backward=False)
def work_chunks(
    inp: WorkInputs,
    dt: float,
    body_f: wp.array[wp.spatial_vector],
    body_q_in: wp.array[wp.transform],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
    particle_q_in: wp.array[wp.vec3],
    particle_q_out: wp.array[wp.vec3],
):
    """One chunk's work of applied wrenches and particle forces over one exchange (float64, item order)."""
    c = wp.tid()
    w = inp.chunks.world[c]
    nb = inp.body_start[w + 1] - inp.body_start[w]
    lo = inp.chunks.lo[c]
    hi = inp.chunks.hi[c]
    acc = wp.float64(0.0)
    if body_f.shape[0] > 0:
        for p in range(lo, wp.min(hi, nb)):
            b = inp.body_start[w] + p
            f = body_f[b]
            dx = body_com_world(body_q_out[b], inp.body_com[b]) - body_com_world(body_q_in[b], inp.body_com[b])
            acc = acc + wp.float64(wp.dot(wp.spatial_top(f), dx))
            acc = acc + wp.float64(dt * wp.dot(wp.spatial_bottom(f), wp.spatial_bottom(body_qd_out[b])))
    if particle_f.shape[0] > 0:
        for p in range(wp.max(lo, nb), hi):
            i = inp.particle_start[w] + p - nb
            acc = acc + wp.float64(wp.dot(particle_f[i], particle_q_out[i] - particle_q_in[i]))
    inp.w_part[c] = acc


@wp.kernel(enable_backward=False)
def work_sum(
    inp: WorkInputs,
    dt: float,
    joint_f: wp.array[float],
    joint_qd_in: wp.array[float],
    joint_qd_out: wp.array[float],
    external_work: wp.array[wp.float64],
):
    """W_ext[w] += its chunks' work in chunk order, then its joint forces' work."""
    w = wp.tid()
    acc = chunk_sum(inp.chunks, inp.w_part, w)
    if joint_f.shape[0] > 0:
        for d in range(inp.dof_start[w], inp.dof_start[w + 1]):
            acc = acc + wp.float64(dt * joint_f[d] * 0.5 * (joint_qd_in[d] + joint_qd_out[d]))
    external_work[w] = external_work[w] + acc


def launch_external_work(inp: WorkInputs, dt: float, body: tuple, particle: tuple, joint: tuple,
                         external_work: wp.array, device) -> None:
    """W_ext[w] += work of applied wrenches, particle forces and joint forces over one exchange (module docstring):
    ``body`` = (body_f, body_q_in, body_q_out, body_qd_out), ``particle`` = (particle_f, particle_q_in,
    particle_q_out), ``joint`` = (joint_f, joint_qd_in, joint_qd_out)."""
    n = chunk_count(inp.chunks)
    if n:
        wp.launch(work_chunks, dim=n, inputs=[inp, float(dt), *body, *particle], device=device)
    wp.launch(work_sum, dim=external_work.shape[0], inputs=[inp, float(dt), *joint, external_work], device=device)


def work_inputs(model: Any) -> WorkInputs:
    """The external-work pass's per-model inputs (host, once)."""
    inp = WorkInputs()
    inp.body_start, inp.particle_start = model.body_world_start, model.particle_world_start
    inp.dof_start, inp.body_com = model.joint_dof_world_start, model.body_com
    inp.chunks = make_chunks(_item_counts(model), model.device)
    inp.w_part = wp.zeros(max(chunk_count(inp.chunks), 1), dtype=wp.float64, device=model.device)
    return inp


def _item_counts(model: Any) -> np.ndarray:
    """[W] bodies + particles of each world (global items left out)."""
    W = int(model.world_count)
    starts = (model.body_world_start, model.particle_world_start)
    return sum(np.diff(np.asarray(a.numpy(), np.int64)[: W + 1]) for a in starts)


def energy_inputs(model: Any) -> EnergyInputs:
    """The D20 ledger's per-model inputs (host, once): masses, inertia, COMs, bounding radii, which bodies count."""
    from alloy.coupling.kinematic import world_welded_bodies  # noqa: PLC0415

    dev = model.device
    inp = EnergyInputs()
    inp.body_start, inp.particle_start = model.body_world_start, model.particle_world_start
    inp.body_mass, inp.body_inertia, inp.body_com = model.body_mass, model.body_inertia, model.body_com
    inp.body_radius = wp.array(body_bounding_radius(model), dtype=float, device=dev)
    dynamic = (~world_welded_bodies(model)).astype(np.int32)
    inp.body_dynamic = wp.array(dynamic if dynamic.size else np.ones(1, np.int32), dtype=wp.int32, device=dev)
    inp.particle_mass, inp.particle_radius, inp.gravity = model.particle_mass, model.particle_radius, model.gravity
    _energy_chunks(inp, model)
    return inp


def _energy_chunks(inp: EnergyInputs, model: Any) -> None:
    """Chunks of each world's bodies then particles (global items left out) and their scratch (``chunk_sums``)."""
    dev = model.device
    inp.chunks = make_chunks(_item_counts(model), dev)
    n = max(chunk_count(inp.chunks), 1)
    inp.e_part = wp.zeros(n, dtype=wp.float64, device=dev)
    inp.v_part = wp.zeros(n, dtype=float, device=dev)
    inp.bad_part = wp.zeros(n, dtype=wp.int32, device=dev)


def body_bounding_radius(model: object) -> np.ndarray:
    """Per-body max shape bounding radius [m] (0 for bodies without shapes), host, once at construction."""
    radius = np.zeros(int(model.body_count), dtype=np.float32)
    if model.shape_count == 0 or model.body_count == 0:
        return radius
    shape_body = model.shape_body.numpy()
    shape_r = model.shape_collision_radius.numpy() if model.shape_collision_radius is not None else None
    if shape_r is None:
        return radius
    mask = shape_body >= 0
    np.maximum.at(radius, shape_body[mask], shape_r[mask].astype(np.float32))
    return radius
