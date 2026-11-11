# SPDX-License-Identifier: Apache-2.0
"""A vessel's water as mass on its body (G2, JOB3's can tipped and emptied; GA lumps a cavity pool into its container's
mass every tick: surface_rinsing/water/system.py:645-686, world.py:99-108 and painting/world.py:181-198).

``Pool(..., body=, cavity=, carry="mass")``. Each exchange, in ``apply`` before the solver step (``carry_mass``):
- the body's mass becomes its dry mass plus ρ·V of each such pool on it (V from the pool's tank, as the ledgers have
  it; ``carried_kg [W, P]`` in ``read()``), its centre of mass the combined centroid (the water's from the cavity's
  slices below its surface, as ``supply_float``'s weight), and its inertia the dry inertia moved to the new centre plus
  each pool's water as a point mass at its centroid: the water translates with its vessel and does not turn with it
  (its own rotational inertia about its centroid is left out, as GA's; GA also leaves out the water's parallel-axis
  term, Alloy keeps it). The values land in ``model.body_mass``, ``body_inv_mass``, ``body_com``, ``body_inertia`` and
  ``body_inv_inertia``, which plug-ins read in ``apply`` (a hand listed after the flow holds the can with its water),
  and the plug-in host pushes them into the solver after ``apply`` (``PluginHost``, ``changes_inertia``);
- the body's velocities (``body_qd``; a free joint's ``joint_qd``) move from its dry centre to the new one, v += ω ×
  R·(c - c_dry), so every point of the body keeps its velocity (Newton's free-joint velocity is its centre's).
After the solver step (``carry_restore``, first in ``after_film``) the dry values and the dry-centre velocities come
back: between exchanges the model and the state are the dry body's, so a reset, a redraw of its density and every
reader outside the step see the scene as built (MuJoCo's constants from a reset's notify stay dry; the health
monitor's energy is the dry body's). The dry values are read from the model at each ``apply``. A carried pool's weight
force (``supply_float``) is dropped: gravity acts on the mass. Default ``carry="weight"``: the force only.

Limits: MuJoCo's constants (body_invweight0, subtreemass: the contact softness's scale) stay at the dry mass, as GA's
(it writes the masses without mj_setConst); a state read between exchanges is the dry body's (its centre's velocity);
the solvers that take run-time masses are listed in ``PluginHost`` (others and coupled steppers are refused).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.supply_float import Hull, below
from alloy.plugins.surface_flow.supply_kernels import PoolSpec


@wp.func
def pool_kg(w: int, p: int, pools: wp.array[PoolSpec], pressure: wp.array2d[wp.float64]) -> wp.float64:
    """ρ·V of pool p's water in world w, V from its tank (float64, as the ledgers)."""
    pool = pools[p]
    return wp.float64(pool.rho) * pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0))


@wp.func
def pool_water(w: int, p: int, pools: wp.array[PoolSpec], pressure: wp.array2d[wp.float64], surf: wp.array2d[wp.vec4],
               vessels: wp.array[Hull], hsl: wp.array[wp.vec3d]) -> wp.vec4:
    """(ρ·V, centroid in the body's frame) of pool p's water in world w (V from its tank; the floor's centre while
    its surface is not set or the slices hold none of it)."""
    pool = pools[p]
    kg = pool_kg(w, p, pools, pressure)
    if kg <= wp.float64(0.0):
        return wp.vec4(0.0, pool.floor[0], pool.floor[1], pool.floor[2])
    c = pool.floor
    sv = surf[w, p]
    if sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2] >= 0.25:
        vc = below(vessels[p], hsl, wp.vec3d(wp.float64(sv[0]), wp.float64(sv[1]), wp.float64(sv[2])),
                   wp.float64(sv[3]))
        if vc[0] > wp.float64(0.0):
            c = pool.floor + wp.vec3(float(vc[1]), float(vc[2]), float(vc[3]))
    return wp.vec4(float(kg), c[0], c[1], c[2])


@wp.func
def point_inertia(m: float, d: wp.vec3) -> wp.mat33:
    """A point mass m at offset d: m·(|d|²·1 - d·dᵀ)."""
    return m * (wp.dot(d, d) * wp.identity(n=3, dtype=float) - wp.outer(d, d))


@wp.func
def shift_velocity(b: int, j: int, d: wp.vec3, body_qd: wp.array[wp.spatial_vector], joint_qd: wp.array[wp.float32],
                   joint_qd_start: wp.array[wp.int32], joint_X_p: wp.array[wp.transform]):
    """Body b's centre velocity moved by ω × d (world d from the old centre to the new), in ``body_qd`` and, for a
    free joint j (else -1), in its ``joint_qd`` (the centre's velocity in the joint's parent frame)."""
    qd = body_qd[b]
    om = wp.spatial_bottom(qd)
    dv = wp.cross(om, d)
    body_qd[b] = wp.spatial_vector(wp.spatial_top(qd) + dv, om)
    if j >= 0:
        s = joint_qd_start[j]
        dp = wp.quat_rotate_inv(wp.transform_get_rotation(joint_X_p[j]), dv)
        joint_qd[s + 0] = joint_qd[s + 0] + dp[0]
        joint_qd[s + 1] = joint_qd[s + 1] + dp[1]
        joint_qd[s + 2] = joint_qd[s + 2] + dp[2]


@wp.kernel
def carry_mass(pools: wp.array[PoolSpec], pressure: wp.array2d[wp.float64], surf: wp.array2d[wp.vec4],
               vessels: wp.array[Hull], hsl: wp.array[wp.vec3d], cbody: wp.array2d[wp.int32],
               cjoint: wp.array2d[wp.int32], cstart: wp.array[wp.int32], cpool: wp.array[wp.int32],
               body_q: wp.array[wp.transform], joint_qd_start: wp.array[wp.int32], joint_X_p: wp.array[wp.transform],
               body_mass: wp.array[wp.float32], body_inv_mass: wp.array[wp.float32], body_com: wp.array[wp.vec3],
               body_inertia: wp.array[wp.mat33], body_inv_inertia: wp.array[wp.mat33],
               body_qd: wp.array[wp.spatial_vector], joint_qd: wp.array[wp.float32], m0: wp.array2d[wp.float32],
               c0: wp.array2d[wp.vec3], i0: wp.array2d[wp.mat33], cw: wp.array2d[wp.vec3],
               carried: wp.array2d[wp.float64], im0: wp.array2d[wp.float32], ii0: wp.array2d[wp.mat33]):
    """Per world, each carrying body: its dry values saved (the inverses as the model holds them, so the restore is
    bit for bit whatever the builder's inverse was), then its dry mass plus its pools' water (module docstring) and its
    velocities onto the new centre."""
    w = wp.tid()
    for k in range(cbody.shape[1]):
        b = cbody[w, k]
        md = body_mass[b]
        cd = body_com[b]
        idry = body_inertia[b]
        m0[w, k] = md
        c0[w, k] = cd
        i0[w, k] = idry
        im0[w, k] = body_inv_mass[b]
        ii0[w, k] = body_inv_inertia[b]
        m = md
        mc = md * cd
        for n in range(cstart[k], cstart[k + 1]):
            pw = pool_water(w, cpool[n], pools, pressure, surf, vessels, hsl)
            carried[w, cpool[n]] = pool_kg(w, cpool[n], pools, pressure)
            m = m + pw[0]
            mc = mc + pw[0] * wp.vec3(pw[1], pw[2], pw[3])
        com = mc / m
        inertia = idry + point_inertia(md, cd - com)
        for n in range(cstart[k], cstart[k + 1]):
            pw = pool_water(w, cpool[n], pools, pressure, surf, vessels, hsl)
            inertia = inertia + point_inertia(pw[0], wp.vec3(pw[1], pw[2], pw[3]) - com)
        cw[w, k] = com
        body_mass[b] = m
        body_inv_mass[b] = 1.0 / m
        body_com[b] = com
        body_inertia[b] = inertia
        body_inv_inertia[b] = wp.inverse(inertia)
        d = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), com - cd)
        shift_velocity(b, cjoint[w, k], d, body_qd, joint_qd, joint_qd_start, joint_X_p)


@wp.kernel
def carry_restore(cbody: wp.array2d[wp.int32], cjoint: wp.array2d[wp.int32], m0: wp.array2d[wp.float32],
                  c0: wp.array2d[wp.vec3], i0: wp.array2d[wp.mat33], cw: wp.array2d[wp.vec3],
                  body_q: wp.array[wp.transform], joint_qd_start: wp.array[wp.int32],
                  joint_X_p: wp.array[wp.transform], body_mass: wp.array[wp.float32],
                  body_inv_mass: wp.array[wp.float32], body_com: wp.array[wp.vec3], body_inertia: wp.array[wp.mat33],
                  body_inv_inertia: wp.array[wp.mat33], body_qd: wp.array[wp.spatial_vector],
                  joint_qd: wp.array[wp.float32], im0: wp.array2d[wp.float32], ii0: wp.array2d[wp.mat33]):
    """Per world, each carrying body: its dry values (the saved inverses, bit for bit) and its dry centre's velocities
    back (module docstring)."""
    w = wp.tid()
    for k in range(cbody.shape[1]):
        b = cbody[w, k]
        body_mass[b] = m0[w, k]
        body_inv_mass[b] = im0[w, k]
        body_com[b] = c0[w, k]
        body_inertia[b] = i0[w, k]
        body_inv_inertia[b] = ii0[w, k]
        d = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), c0[w, k] - cw[w, k])
        shift_velocity(b, cjoint[w, k], d, body_qd, joint_qd, joint_qd_start, joint_X_p)


def carry_tables(pools: tuple[Any, ...]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The carrying bodies' labels (first appearance), and per body the range of its pools in the pool list."""
    labels: list[str] = []
    for pool in pools:
        if pool.carry == "mass" and pool.body not in labels:
            labels.append(pool.body)
    groups = [[p for p, pool in enumerate(pools) if pool.carry == "mass" and pool.body == lb] for lb in labels]
    start = np.cumsum([0] + [len(g) for g in groups]).astype(np.int32)
    flat = np.asarray([p for g in groups for p in g] or [0], np.int32)
    return labels, start, flat


def free_joints(model: newton.Model, bodies: np.ndarray) -> np.ndarray:
    """[W, K] the free joint whose child each body is (its velocity is its centre's in ``joint_qd``), else -1."""
    child, jtype = model.joint_child.numpy(), model.joint_type.numpy()
    free = {int(c): j for j, (c, t) in enumerate(zip(child, jtype)) if int(t) == int(newton.JointType.FREE)}
    return np.vectorize(lambda b: free.get(int(b), -1))(bodies).astype(np.int32)


class Carrying:
    """``Supply``'s pools carried as mass (a mixin: it uses the supply's pools, cavity surfaces and float tables)."""

    def _setup_carry(self, ctx: Any) -> None:
        """The carrying bodies (``ctx.bodies`` per label), their free joints, the saved values and ``carried_kg``."""
        W, P = ctx.world_count, max(len(self.pools), 1)
        labels, start, flat = carry_tables(self.pools)
        self.carrying = bool(labels)
        self.carried = ctx.zeros((W, P), wp.float64)
        if not labels:
            return
        bodies = np.stack([self._bodies(ctx, lb) for lb in labels], 1).astype(np.int32)     # [W, K]
        if (bodies < 0).any():
            raise PluginError(f"carry='mass': a pool's body is missing in some world ({labels})")
        if (ctx.model.body_mass.numpy()[bodies] <= 0.0).any():
            raise PluginError(f"carry='mass': {labels} must be free bodies with mass (a static vessel carries none)")
        K = len(labels)
        self.cbody, self.cjoint = ctx.array(bodies, wp.int32), ctx.array(free_joints(ctx.model, bodies), wp.int32)
        self.cstart, self.cpool = ctx.array(start, wp.int32), ctx.array(flat, wp.int32)
        self.c_m0, self.c_c0 = ctx.zeros((W, K), wp.float32), ctx.zeros((W, K), wp.vec3)
        self.c_i0, self.c_cw = ctx.zeros((W, K), wp.mat33), ctx.zeros((W, K), wp.vec3)
        self.c_im0, self.c_ii0 = ctx.zeros((W, K), wp.float32), ctx.zeros((W, K), wp.mat33)
        self.film.changes_inertia = True                  # the plug-in host pushes the masses into the solver

    def _carry(self, state: newton.State) -> None:
        """In ``apply``, after the cavities' surfaces and before every force the supply computes about a centre of mass
        (``Floating._float_forces``): the carrying bodies' wet mass properties."""
        if not self.carrying:
            return
        m = self.ctx.model
        wp.launch(carry_mass, dim=self.ctx.world_count,
                  inputs=[self.pool_spec, self.fluid.pressure, self.cav_surf, self.vessels, self.hull_slices,
                          self.cbody, self.cjoint, self.cstart, self.cpool, state.body_q, m.joint_qd_start,
                          m.joint_X_p],
                  outputs=[m.body_mass, m.body_inv_mass, m.body_com, m.body_inertia, m.body_inv_inertia,
                           state.body_qd, state.joint_qd, self.c_m0, self.c_c0, self.c_i0, self.c_cw, self.carried,
                           self.c_im0, self.c_ii0],
                  device=self.ctx.device)

    def _carry_restore(self, state: newton.State) -> None:
        """First in ``after_film`` (``state``: the solver's output): the dry body back."""
        if not self.carrying:
            return
        m = self.ctx.model
        wp.launch(carry_restore, dim=self.ctx.world_count,
                  inputs=[self.cbody, self.cjoint, self.c_m0, self.c_c0, self.c_i0, self.c_cw, state.body_q,
                          m.joint_qd_start, m.joint_X_p],
                  outputs=[m.body_mass, m.body_inv_mass, m.body_com, m.body_inertia, m.body_inv_inertia,
                           state.body_qd, state.joint_qd, self.c_im0, self.c_ii0], device=self.ctx.device)


__all__ = ["Carrying", "carry_mass", "carry_restore", "carry_tables", "free_joints", "point_inertia", "pool_kg",
           "pool_water", "shift_velocity"]
