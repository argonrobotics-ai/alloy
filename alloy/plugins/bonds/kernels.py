# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the ``Bonds`` plug-in (``field.py`` launches them; the model is in ``plugin.py``'s docstring).

Particle arrays are in the film solver's particle indexing, which equals the model's: the film entry owns every
particle. Shape ids are the model's (entry views keep shape numbering). Body ids go through the caller's
``shape_body``. A film entry's view numbers bodies locally; the host's ``body_f`` uses the model's numbering.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, wrench_at

from .wetout import FORCE_SCALE, M_EST, P_REF, Rollers, roller_pressure

FREE = wp.constant(0)
SUBSTRATE = wp.constant(1)
GRIP = wp.constant(2)
FAR = wp.constant(1.0e30)                 # material distance of a vertex no grip can peel
KEY_SHIFT = wp.constant(wp.int64(1048576))   # 2^20: key = pressure [Pa] * 2^20 + shape


@wp.func
def to_world(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], s: wp.int32, p: wp.vec3) -> wp.vec3:
    """Point ``p`` in shape ``s``'s body frame, in the world (static shapes: already world)."""
    b = shape_body[s]
    if b < 0:
        return p
    return wp.transform_point(body_q[b], p)


@wp.func
def dir_to_world(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], s: wp.int32, v: wp.vec3) -> wp.vec3:
    b = shape_body[s]
    if b < 0:
        return v
    return wp.quat_rotate(wp.transform_get_rotation(body_q[b]), v)


@wp.struct
class Bonded:
    """Per vertex: state, world-local target shape (+ ``base[world]`` = the model's shape id), anchor, stiffness."""
    state: wp.array[wp.int32]
    shape: wp.array[wp.int32]
    world: wp.array[wp.int32]
    base: wp.array[wp.int32]
    anchor: wp.array[wp.vec3]
    k: wp.array[float]


@wp.func
def model_shape(bd: Bonded, i: int) -> int:
    return bd.shape[i] + bd.base[bd.world[i]]


@wp.kernel
def bond_accumulate(colour: int, colors: wp.array[wp.int32], q: wp.array[wp.vec3], body_q: wp.array[wp.transform],
                    shape_body: wp.array[wp.int32], bd: Bonded, forces: wp.array[wp.vec3],
                    hessians: wp.array[wp.mat33]):
    """VBD after_accumulate slot: an implicit isotropic spring k·(x - anchor) on every bonded vertex of ``colour``."""
    i = wp.tid()
    if bd.state[i] == FREE or colors[i] != colour:
        return
    k = bd.k[i]
    a = to_world(body_q, shape_body, model_shape(bd, i), bd.anchor[i])
    forces[i] = forces[i] - k * (q[i] - a)
    hessians[i] = hessians[i] + k * wp.identity(3, float)


@wp.kernel
def pin_sync(state: wp.array[wp.int32], inv_mass0: wp.array[float], inv_mass: wp.array[float]):
    """Bonded vertices are pinned for SolverFilm's strain projection (inverse mass 0; VBD keeps their mass)."""
    i = wp.tid()
    if state[i] != FREE:
        inv_mass[i] = 0.0
    else:
        inv_mass[i] = inv_mass0[i]


@wp.kernel
def release_grips(release: wp.array[wp.int32], world: wp.array[wp.int32], state: wp.array[wp.int32]):
    """Input ``release_grip``: a world's grip bonds let go."""
    i = wp.tid()
    if state[i] == GRIP and release[world[i]] != 0:
        state[i] = FREE


# --- formation: elastic contact pressure above p_form for the dwell --------------------------------------------------

@wp.struct
class Contacts:
    count: wp.array[wp.int32]
    particle: wp.array[wp.int32]
    shape: wp.array[wp.int32]
    body_pos: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]


@wp.struct
class Surface:
    """Per shape: substrate flag, formation pressure and dwell, contact stiffness and margin."""
    substrate: wp.array[wp.int32]
    p_form: wp.array[float]
    dwell: wp.array[float]
    ke: wp.array[float]           # the contact stiffness VBD uses: (soft_contact_ke + shape ke) / 2
    margin: wp.array[float]
    tau_w: wp.array[float]        # wet-out time constant (wetout.wet_tau of the card's p_form)


@wp.struct
class Press:
    """What the formation pressure reads besides the contact: the rollers, the vertices' worlds, shape poses."""
    rollers: Rollers
    world: wp.array[wp.int32]
    shape_xform: wp.array[wp.transform]


@wp.func
def contact_key(c: int, ct: Contacts, sf: Surface, q: wp.array[wp.vec3], radius: wp.array[float],
                body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], cell: wp.array[float],
                pr: Press) -> wp.int64:
    """Contact c's formation key: pressure [Pa] * 2^20 + shape, or 0 if it is not on a substrate at p >= p_form. The
    pressure is k_c·pen/a_cell (a uniform press), or a roller's Hertz pressure under a roller (wetout.py)."""
    s = ct.shape[c]
    if s < 0:
        return wp.int64(0)
    if sf.substrate[s] == 0:
        return wp.int64(0)
    p = ct.particle[c]
    bx = to_world(body_q, shape_body, s, ct.body_pos[c])
    pen = radius[p] + sf.margin[s] - wp.dot(ct.normal[c], q[p] - bx)
    pres = sf.ke[s] * pen / cell[p]
    under = roller_pressure(pr.rollers, pr.world[p], q[p], ct.normal[c], s, radius[p], body_q, shape_body,
                            pr.shape_xform)
    if under >= 0.0:
        pres = under
    if pen <= 0.0 or pres < sf.p_form[s]:
        return wp.int64(0)
    return wp.int64(wp.min(pres, 1.0e12)) * KEY_SHIFT + wp.int64(s)


@wp.kernel
def roller_force(ct: Contacts, sf: Surface, q: wp.array[wp.vec3], radius: wp.array[float],
                 body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], pr: Press):
    """Each roller's film load this step: Σ k_c·pen over its contacts with the film (wetout.py, "measured"), summed
    in fixed point so the sum does not depend on the contacts' order."""
    c = wp.tid()
    if c >= ct.count[0]:
        return
    s = ct.shape[c]
    if s < 0:
        return
    r = pr.rollers.of_shape[s]
    if r < 0:
        return
    p = ct.particle[c]
    bx = to_world(body_q, shape_body, s, ct.body_pos[c])
    pen = radius[p] + sf.margin[s] - wp.dot(ct.normal[c], q[p] - bx)
    if pen > 0.0:
        wp.atomic_add(pr.rollers.force, pr.world[p], r, wp.int64(sf.ke[s] * pen * FORCE_SCALE))


@wp.kernel
def roller_load(rl: Rollers, dt: float):
    """Line load in use per world and roller: the override if > 0, else the measured load over the width, averaged
    over tau (first order)."""
    w, r = wp.tid()
    if r >= rl.count:
        return
    if rl.override[w, r] > 0.0:
        rl.load[w, r] = rl.override[w, r]
    else:
        q = float(rl.force[w, r]) / FORCE_SCALE / (2.0 * rl.half_len[r])
        rl.load[w, r] = rl.load[w, r] + (q - rl.load[w, r]) * wp.min(dt / rl.tau, 1.0)
    rl.force[w, r] = wp.int64(0)


@wp.kernel
def press_keys(ct: Contacts, sf: Surface, q: wp.array[wp.vec3], radius: wp.array[float],
               body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], cell: wp.array[float], pr: Press,
               key: wp.array[wp.int64]):
    c = wp.tid()
    if c >= ct.count[0]:
        return
    kk = contact_key(c, ct, sf, q, radius, body_q, shape_body, cell, pr)
    if kk > wp.int64(0):
        wp.atomic_max(key, ct.particle[c], kk)


@wp.kernel
def press_dwell(key: wp.array[wp.int64], sf: Surface, dt: float, state: wp.array[wp.int32],
                dwell: wp.array[float], fresh: wp.array[wp.int32]):
    """Dwell clock per free vertex: runs while pressed at p >= p_form, resets otherwise; ``fresh`` once it passes."""
    i = wp.tid()
    fresh[i] = 0
    if state[i] != FREE or key[i] == wp.int64(0):
        dwell[i] = 0.0
        return
    dwell[i] = dwell[i] + dt
    s = int(key[i] % KEY_SHIFT)
    if dwell[i] >= sf.dwell[s]:
        fresh[i] = 1


@wp.kernel
def form_bonds(ct: Contacts, sf: Surface, q: wp.array[wp.vec3], radius: wp.array[float],
               body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], cell: wp.array[float], pr: Press,
               key: wp.array[wp.int64], fresh: wp.array[wp.int32], rest_dir: wp.array[wp.vec3], bd: Bonded,
               normal: wp.array[wp.vec3], bond_dir: wp.array[wp.vec3], dwell: wp.array[float]):
    """A fresh vertex bonds to its key's contact: anchored at the surface point plus (r + margin)·n (body frame)."""
    c = wp.tid()
    if c >= ct.count[0]:
        return
    p = ct.particle[c]
    if fresh[p] == 0:
        return
    if contact_key(c, ct, sf, q, radius, body_q, shape_body, cell, pr) != key[p]:
        return
    s = ct.shape[c]
    b = shape_body[s]
    n = ct.normal[c]
    d = rest_dir[p]
    if b >= 0:
        rot = wp.transform_get_rotation(body_q[b])
        n = wp.quat_rotate_inv(rot, n)
        d = wp.quat_rotate_inv(rot, d)
    bd.shape[p] = s - bd.base[bd.world[p]]
    bd.anchor[p] = ct.body_pos[c] + (radius[p] + sf.margin[s]) * n
    normal[p] = n
    bond_dir[p] = d
    bd.state[p] = SUBSTRATE
    dwell[p] = 0.0


@wp.kernel
def wet_update(key: wp.array[wp.int64], sf: Surface, pr: Press, bd: Bonded, q: wp.array[wp.vec3],
               radius: wp.array[float], normal: wp.array[wp.vec3], body_q: wp.array[wp.transform],
               shape_body: wp.array[wp.int32], dt: float, wet: wp.array[float]):
    """Wet-out (wetout.py): w <- 1 - (1 - w)·exp(-(p/P_REF)^M·dt/tau_w) while pressed at p >= p_form. A free vertex
    reads its formation key (contact or roller pressure); a bonded one a roller over it, else its key; a free vertex
    off the press goes back to 0, a bonded one keeps its w."""
    i = wp.tid()
    if bd.state[i] == GRIP:
        return
    k = key[i]
    p = float(0.0)
    s = int(-1)
    if k > wp.int64(0):
        p = float(k // KEY_SHIFT)
        s = int(k % KEY_SHIFT)
    if bd.state[i] == SUBSTRATE:
        s = model_shape(bd, i)
        under = roller_pressure(pr.rollers, pr.world[i], q[i], dir_to_world(body_q, shape_body, s, normal[i]), s,
                                radius[i], body_q, shape_body, pr.shape_xform)
        if under >= 0.0:
            p = under
        if p < sf.p_form[s]:
            return
    elif s < 0:
        wet[i] = 0.0
        return
    wet[i] = 1.0 - (1.0 - wet[i]) * wp.exp(-wp.pow(p / P_REF, M_EST) * dt / sf.tau_w[s])


# --- the pull on the bodies (ForcePlugin.apply, model numbering) ----------------------------------------------------

@wp.kernel
def apply_pull(force: wp.array[wp.vec3], point: wp.array[wp.vec3], front_point: wp.array[wp.vec3],
               grip_shape: wp.array[wp.int32], sub_shape: wp.array[wp.int32], base: wp.array[wp.int32],
               shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3],
               body_f: wp.array[wp.spatial_vector]):
    """-T·u on the grip body at the grip point; +T·u on a dynamic substrate at the front. One thread per world;
    shape ids are world-local (+ ``base[w]``)."""
    w = wp.tid()
    if grip_shape[w] < 0:
        return
    bg = shape_body[grip_shape[w] + base[w]]
    if bg >= 0:
        add_wrench(body_f, bg, wrench_at(force[w], point[w], com_world(body_q, body_com, bg)))
    if sub_shape[w] >= 0:
        bs = shape_body[sub_shape[w] + base[w]]
        if bs >= 0 and bs != bg:
            add_wrench(body_f, bs, wrench_at(-force[w], front_point[w], com_world(body_q, body_com, bs)))
