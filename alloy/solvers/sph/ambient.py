# SPDX-License-Identifier: Apache-2.0
"""Ambient pressure for sealed liquid (SCR-SC10-1, ``DFSPHConfig.ambient_pressure``).

DFSPH's pressure is unilateral (gauge >= 0, pressure.py): liquid leaves a wall at no cost. A sealed volume cannot:
a liquid column in a capillary under a piston would have to drop below ambient to leave the piston face (a vacuum
gap), so it hangs at gauge -rho g H under the piston and moves only with it.

Formulation: no tension in the liquid. The solve runs on P' = p + p_a (p the gauge pressure, p_a =
``ambient_pressure``) with P' >= 0, so it stays unilateral, and sealed liquid is in compression (P' = p_a - rho g H
under the piston). Carrying tension in the liquid instead (a bilateral interior, k down to -p_a) was built and
measured first and fails: the cubic kernel is convex at the nearest-neighbour distance, so under tension the layers
pair up and a column lengthens at rest density (Swegle's tensile instability), and the bilateral constraints at the
edge of the interior drive k to the floor (docs/notes/wp9-results.md). With k_a = p_a / rho0, the offset k_a in
every particle's pressure adds -2 k_a g_i to its pressure acceleration, g_i = sum_j V_j gradW_ij + 1/2 sum_b
gradPsi_b the colour gradient with walls counting as liquid (zero for a complete rest neighbourhood). Per step:
- sealed liquid: a particle is sealed when liquid connects it to a seal wall (``sph:shape_seal`` = 1): hop distance
  ``hop`` to a particle with a seal slot, through neighbours, broken by the free surface and by open walls (a few
  relaxation sweeps per step, ``seal_hop_kernel``).
- air: every particle takes v += dt 2 k_a w g (ghost air at P' = p_a), which cancels the offset's force where w = 1:
  liquid that is not sealed, the free surface of sealed liquid (|g_air| h >= AIR_SURFACE, g_air the fluid colour
  gradient with the slot walls' normal components removed: walls are never air), and sealed liquid at gauge >= 0.
  Sealed liquid below 0 gauge keeps part of the offset's force, w = clamp(1 + p / (GAUGE_RAMP p_a), 0, 1) from the
  last step's solve: there it is the repulsion of positive P' that keeps liquid under tension from pairing. Liquid
  near the free surface is at gauge ~0, so the push cancels the offset there completely and the meniscus sees no
  spurious pressure. (The first design ramped w on |g_air| h instead: the partly cancelled layers under a sealed
  meniscus put a few % of p_a on it, -0.88 rho g H under C18's piston at 500 Pa, x2-5 at 1-2 kPa, and peeled sealed
  pendant drops off their tip.)
- gauge >= 0 where the liquid is not sealed and on the free surface of sealed liquid (density-solve lower bound
  k_a): the free surface stays at 0 gauge, liquid against an open wall leaves it as today, and liquid that is not
  sealed is today's solver shifted by k_a (A13.hydrostatic and A13.resting_jitter pass with p_a set and no seal).
  Sealed liquid may fall to P' = 0, gauge -p_a; particles there count as ``cavitating`` (WorldStats.diagnostics).
- walls: every slot reaction of the density solve gives back m_i k_a dt gradPsi_b, so walls get the gauge pressure
  (a seal wall above a hanging column is pulled: -rho g H A).
- zero net residual (SCR-WP9-12): the offset's force the push leaves, -2 k_a (1 - w) g, is a per-particle weight on
  an antisymmetric sum, so its sum over sealed liquid is not zero: it pulled liquid toward walls and held 3-11 % of
  C18's column with no reaction on any shape, and the piston read 0.89-0.94 rho g H. Each world's mean residual over
  its sealed liquid off the free surface is taken off those particles (deterministic per-world sums), so the local
  repulsion stays and the gauge carries the weight. Sealed bodies of liquid that share a world share the correction
  (the tests and the A13 law scenes have one per world).
- prime (runtime.prime_hydrostatic): its divergence solve is two-sided on sealed liquid off the free surface, down to
  -p_a, so a column hanging under its piston starts at its hydrostatic tension; from the one-sided prime a 110-layer
  column needed > 0.1 s to build it under the 50-iteration cap and sagged 2-5 spacings.
p_a is the most tension sealed liquid carries (the cavitation floor); set a few times the scene's need (C18: 500 Pa;
docs/notes/wp9-results.md has the measured range). The density warm start is stored as its gauge part (k_g dt^2,
the solver's dt-free convention) and takes k_a dt^2 at each step's own dt, so the offset stays exact when a CFL dt
changes (stored whole, every dt change rescaled all of p_a in the warm start and in the push weight's gauge); the
resting-jitter rule reads the hydrostatic head from |k_g|. None: nothing of this runs (bitwise the solver without it).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .kernels import kernel_grad
from .pressure import Particles, Slots, is_fluid, nbr_base, nbr_j
from .reduce import WorldReducer

HOP_FAR = wp.constant(1 << 20)  # hop distance of liquid that is not sealed
HOP_SWEEPS = 8  # relaxation sweeps of the seal hop distance per step
AIR_SURFACE = 0.45  # |g_air| h at or above which sealed liquid is free surface (a flat surface layer reads 0.85-1.0)
GAUGE_RAMP = 0.5  # sealed liquid below 0 gauge: push weight 0 at gauge -GAUGE_RAMP p_a
NORMAL_INDEPENDENT = wp.constant(0.2)  # a slot normal counts if at least this part of it is off the previous ones


@wp.func
def _off(v: wp.vec3, e: wp.vec3) -> wp.vec3:
    return v - wp.dot(v, e) * e


@wp.func
def air_gradient(p: Particles, sl: Slots, i: int, h: float, kind: int) -> wp.vec3:
    """The fluid colour gradient sum_j V_j gradW_ij without its components along the slot walls' normals (walls are
    never air): what is left points from the air into the liquid. Wall volumes do not enter, so a wall whose fitted
    volume is off (a curved bore at R/h < 2) cannot make its liquid look like a free surface."""
    xi = p.q[i]
    inv = 1.0 / p.rho0[i]
    g = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        g += (p.mass[j] * inv) * kernel_grad(xi - p.q[j], h, kind)
    e1 = wp.vec3(0.0, 0.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 0.0)
    ne = int(0)
    for b in range(4):
        if sl.shape[i, b] >= 0 and ne < 3:
            v = sl.gpsi[i, b]
            l0 = wp.length(v)
            if ne >= 1:
                v = _off(v, e1)
            if ne >= 2:
                v = _off(v, e2)
            l = wp.length(v)
            if l0 > 0.0 and l > NORMAL_INDEPENDENT * l0:
                u = v / l
                g = _off(g, u)
                if ne == 0:
                    e1 = u
                elif ne == 1:
                    e2 = u
                ne += 1
    return g


@wp.func
def open_slot(sl: Slots, seal: wp.array[wp.int32], i: int) -> bool:
    """A slot of particle i lies on a shape that is not ``sph:shape_seal``."""
    out = False
    for b in range(4):
        s = sl.shape[i, b]
        if s >= 0:
            if s >= seal.shape[0]:  # virtual (soft) shapes are not seals (soft_walls.py)
                out = True
            elif seal[s] == 0:
                out = True
    return out


@wp.func
def seal_slot(sl: Slots, seal: wp.array[wp.int32], i: int) -> bool:
    out = False
    for b in range(4):
        s = sl.shape[i, b]
        if s >= 0 and s < seal.shape[0]:
            if seal[s] != 0:
                out = True
    return out


@wp.kernel(enable_backward=False)
def seal_hop_kernel(p: Particles, sl: Slots, seal: wp.array[wp.int32], h: float, kind: int, surface: float,
                    hop_in: wp.array[wp.int32], hop_out: wp.array[wp.int32]):
    """One relaxation sweep of the hop distance to a seal wall through liquid (module doc): 0 on a seal slot, HOP_FAR
    on the free surface and against open walls (the air or an open wall breaks the seal), else min(neighbours) + 1."""
    i = wp.tid()
    if not is_fluid(p, i) or p.count[i] == 0:
        hop_out[i] = HOP_FAR
        return
    if seal_slot(sl, seal, i):
        hop_out[i] = 0
        return
    if open_slot(sl, seal, i) or wp.length(air_gradient(p, sl, i, h, kind)) * h >= surface:
        hop_out[i] = HOP_FAR
        return
    best = int(HOP_FAR)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        best = wp.min(best, hop_in[nbr_j(p, b_, i, k)])
    hop_out[i] = wp.min(best + 1, HOP_FAR)


@wp.kernel(enable_backward=False)
def air_push_kernel(p: Particles, sl: Slots, h: float, kind: int, k_air: float, k_ramp: float, surface: float,
                    dt: float, hop: wp.array[wp.int32], kd: wp.array[wp.float32], vel: wp.array[wp.vec3],
                    surf: wp.array[wp.int32], resid: wp.array[wp.vec3]):
    """v += dt w 2 k_a g (module doc): w = 1 in liquid not sealed (hop = HOP_FAR) and on the free surface of sealed
    liquid (|g_air| h >= surface); inside sealed liquid w = clamp(1 + gauge / (rho0 k_ramp), 0, 1), the gauge from
    the last step's solve (kd = k dt^2). ``surf`` 1 where the solve keeps gauge >= 0 (not sealed, or free surface);
    ``resid`` the offset's acceleration the push leaves, -2 k_a (1 - w) g (the viscous solve's guess, viscosity.py)."""
    i = wp.tid()
    surf[i] = 0
    resid[i] = wp.vec3(0.0, 0.0, 0.0)
    if not is_fluid(p, i):
        return
    v = p.qd[i]
    w = float(1.0)
    s = int(1)
    if hop[i] < HOP_FAR and p.count[i] > 0:
        if wp.length(air_gradient(p, sl, i, h, kind)) * h < surface:
            s = 0
            w = wp.clamp(1.0 + (kd[i] / (dt * dt) - k_air) / k_ramp, 0.0, 1.0)
    if w > 0.0 or s == 0:
        xi = p.q[i]
        inv = 1.0 / p.rho0[i]
        g = wp.vec3(0.0, 0.0, 0.0)
        b_ = nbr_base(p, i)
        for k in range(p.count[i]):
            j = nbr_j(p, b_, i, k)
            g += (p.mass[j] * inv) * kernel_grad(xi - p.q[j], h, kind)
        for b in range(4):
            if sl.shape[i, b] >= 0:
                g += 0.5 * sl.gpsi[i, b]
        if w > 0.0:
            v = v + (2.0 * dt * k_air * w) * g
        resid[i] = (-2.0 * k_air * (1.0 - w)) * g
    surf[i] = s
    vel[i] = v


@wp.kernel(enable_backward=False)
def warm_offset_kernel(p: Particles, delta: float, kd: wp.array[wp.float32]):
    """kd += delta on the liquid: the density warm start holds the gauge part between steps (k_g dt^2, the solver's
    dt-free convention) and the offset k_a dt^2 at the step's own dt only during the density solve."""
    i = wp.tid()
    if is_fluid(p, i):
        kd[i] = kd[i] + delta


@wp.kernel(enable_backward=False)
def prime_bound_kernel(p: Particles, sl: Slots, h: float, kind: int, surface: float, hop: wp.array[wp.int32],
                       kd0: float, kd: wp.array[wp.float32]):
    """Prime (runtime.prime_hydrostatic): kd = k_a dt^2 on sealed liquid off its free surface, else 0, so the
    two-sided divergence solve may give it tension down to -p_a (a hanging column starts at its hydrostatic field)."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    v = float(0.0)
    if hop[i] < HOP_FAR and p.count[i] > 0:
        if wp.length(air_gradient(p, sl, i, h, kind)) * h < surface:
            v = kd0
    kd[i] = v


@wp.kernel(enable_backward=False)
def seal_bound_kernel(p: Particles, surf: wp.array[wp.int32], k_air: float, lower: wp.array[wp.float32],
                      kappa: wp.array[wp.float32]):
    """Density-solve lower bound k_a (gauge >= 0) where the background is fully cancelled (``surf``: open walls, free
    surface), else 0; warm start clamped."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    lb = float(0.0)
    if surf[i] != 0:
        lb = k_air
    lower[i] = lb
    kappa[i] = wp.max(kappa[i], lb)


@wp.kernel(enable_backward=False)
def gauge_reaction_kernel(p: Particles, sl: Slots, k_air: float, dt: float, slot_impulse: wp.array2d[wp.vec3]):
    """Take the offset back from the density solve's slot reactions: J_b -= m_i k_a dt gradPsi_b (gauge walls)."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    mk = p.mass[i] * k_air * dt
    for b in range(4):
        if sl.shape[i, b] >= 0:
            slot_impulse[i, b] = slot_impulse[i, b] - mk * sl.gpsi[i, b]


@wp.kernel(enable_backward=False)
def resid_terms_kernel(p: Particles, surf: wp.array[wp.int32], resid: wp.array[wp.vec3], txy: wp.array[wp.vec2],
                       tzm: wp.array[wp.vec2]):
    """Per-particle terms of the per-world residual sums: (m r_x, m r_y), (m r_z, m) on sealed liquid off its free
    surface (fluid with surf = 0), zeros elsewhere."""
    i = wp.tid()
    txy[i] = wp.vec2(0.0, 0.0)
    tzm[i] = wp.vec2(0.0, 0.0)
    if not is_fluid(p, i) or surf[i] != 0:
        return
    m = p.mass[i]
    r = resid[i]
    txy[i] = wp.vec2(m * r[0], m * r[1])
    tzm[i] = wp.vec2(m * r[2], m)


@wp.kernel(enable_backward=False)
def resid_zero_kernel(p: Particles, surf: wp.array[wp.int32], sxy: wp.array[wp.vec2d], szm: wp.array[wp.vec2d],
                      dt: float, resid: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    """resid -= the world's mean residual a, v -= dt a, on the particles the sums ran over."""
    i = wp.tid()
    if not is_fluid(p, i) or surf[i] != 0:
        return
    w = p.world[i]
    m = szm[w][1]
    if m <= wp.float64(0.0):
        return
    a = wp.vec3(float(sxy[w][0] / m), float(sxy[w][1] / m), float(szm[w][0] / m))
    resid[i] = resid[i] - a
    vel[i] = vel[i] - dt * a


@wp.kernel(enable_backward=False)
def cavitation_kernel(p: Particles, lower: wp.array[wp.float32], kappa: wp.array[wp.float32],
                      surf: wp.array[wp.int32], count: wp.array[wp.int32]):
    """Particles at the cavitation floor: liquid neighbours, not pushed by the air, P' = 0 allowed (no open wall) and
    solved to P' <= 0 (gauge -p_a)."""
    i = wp.tid()
    if not is_fluid(p, i) or p.count[i] == 0 or surf[i] != 0 or lower[i] > 0.0:
        return
    if kappa[i] <= 0.0:
        wp.atomic_add(count, p.world[i], 1)


class AmbientPressure:
    """Seal table, air push and gauge bookkeeping of one SolverDFSPH (``ambient_pressure`` set)."""

    def __init__(self, model, cfg, n: int, device, world_start: np.ndarray) -> None:
        self.p_air = float(cfg.ambient_pressure)
        rho0 = model.sph.rest_density.numpy() if n else np.ones(1)
        rho = float(np.max(rho0)) if len(rho0) else 1000.0
        if n and float(np.ptp(rho0)) > 1.0e-6 * rho:
            raise ValueError("SolverDFSPH: ambient_pressure needs one rest density per model (k_a = p_a / rho0)")
        self.k_air = self.p_air / rho
        self.device = device
        self.surf = wp.zeros(max(n, 1), dtype=wp.int32, device=device)  # 1 where gauge >= 0 (last step)
        self.hop = wp.full(max(n, 1), int(HOP_FAR), dtype=wp.int32, device=device)  # hop distance to a seal wall
        self._hop_tmp = wp.full(max(n, 1), int(HOP_FAR), dtype=wp.int32, device=device)
        self.resid = wp.zeros(max(n, 1), dtype=wp.vec3, device=device)  # offset acceleration the push leaves
        self._reducer = WorldReducer(world_start, device)
        W = max(len(world_start) - 1, 1)
        self._txy, self._tzm = (wp.zeros(max(n, 1), dtype=wp.vec2, device=device) for _ in range(2))
        self._sxy, self._szm = (wp.zeros(W, dtype=wp.vec2d, device=device) for _ in range(2))
        self.refresh(model)

    def refresh(self, model) -> None:
        """Re-read ``sph:shape_seal`` (shape properties changed)."""
        flags = seal_flags(model)
        self.seal = wp.array(flags, dtype=wp.int32, device=self.device)
        self.any_seal = bool(flags.any())

    def relax(self, p: Particles, sl: Slots, h: float, kind: int, n: int, sweeps: int = HOP_SWEEPS) -> None:
        """``sweeps`` relaxation sweeps of the seal hop distance (an even count keeps the result in ``hop``)."""
        for _ in range(max(2, sweeps + sweeps % 2) // 2):
            for a, b in ((self.hop, self._hop_tmp), (self._hop_tmp, self.hop)):
                wp.launch(seal_hop_kernel, dim=n, inputs=[p, sl, self.seal, h, kind, AIR_SURFACE, a, b],
                          device=self.device)

    def push(self, p: Particles, sl: Slots, h: float, kind: int, dt: float, kd: wp.array, tmp: wp.array,
             n: int) -> None:
        """The air's push (into ``tmp``, then copied to the velocities); ``kd`` the density solve's warm start, which
        takes the offset k_a dt^2 here (warm_offset_kernel)."""
        wp.launch(warm_offset_kernel, dim=n, inputs=[p, self.k_air * dt * dt, kd], device=self.device)
        if self.any_seal:
            self.relax(p, sl, h, kind, n)
        wp.launch(air_push_kernel, dim=n, inputs=[p, sl, h, kind, self.k_air, GAUGE_RAMP * self.k_air, AIR_SURFACE,
                                                  dt, self.hop, kd, tmp, self.surf, self.resid],
                  device=self.device)
        if self.any_seal:
            self.zero_net_residual(p, dt, tmp, n)
        wp.copy(p.qd, tmp, count=n)

    def zero_net_residual(self, p: Particles, dt: float, vel: wp.array, n: int) -> None:
        """Take each world's mean residual over its sealed liquid off those particles (module doc, SCR-WP9-12)."""
        wp.launch(resid_terms_kernel, dim=n, inputs=[p, self.surf, self.resid, self._txy, self._tzm], device=self.device)
        self._reducer.sum2(self._txy, self._sxy)
        self._reducer.sum2(self._tzm, self._szm)
        wp.launch(resid_zero_kernel, dim=n, inputs=[p, self.surf, self._sxy, self._szm, dt, self.resid, vel],
                  device=self.device)

    def prime_bounds(self, p: Particles, sl: Slots, h: float, kind: int, dt: float, kd: wp.array, n: int,
                     sweeps: int = 64) -> None:
        """Before the prime's solve: the seal hop distance relaxed ``sweeps`` times and kd = k_a dt^2 on sealed
        liquid off its free surface (prime_bound_kernel)."""
        self.relax(p, sl, h, kind, n, sweeps)
        wp.launch(prime_bound_kernel, dim=n, inputs=[p, sl, h, kind, AIR_SURFACE, self.hop, self.k_air * dt * dt, kd],
                  device=self.device)

    def bound(self, p: Particles, sl: Slots, lower: wp.array, kappa: wp.array, n: int) -> None:
        wp.launch(seal_bound_kernel, dim=n, inputs=[p, self.surf, self.k_air, lower, kappa], device=self.device)

    def gauge(self, p: Particles, sl: Slots, dt: float, slot_impulse: wp.array, kd: wp.array, n: int) -> None:
        """Walls get the gauge pressure; the stored warm start drops the offset again (warm_offset_kernel)."""
        wp.launch(gauge_reaction_kernel, dim=n, inputs=[p, sl, self.k_air, dt, slot_impulse], device=self.device)
        wp.launch(warm_offset_kernel, dim=n, inputs=[p, -self.k_air * dt * dt, kd], device=self.device)

    def count_cavitating(self, p: Particles, lower: wp.array, kappa: wp.array, count: wp.array, n: int) -> None:
        count.zero_()
        wp.launch(cavitation_kernel, dim=n, inputs=[p, lower, kappa, self.surf, count], device=self.device)


def make_ambient(model, cfg, n: int, device, world_start: np.ndarray) -> AmbientPressure | None:
    """The solver's AmbientPressure, or None without ``ambient_pressure``; refuses seal flags it would not read (as
    the tension inputs, layout.py)."""
    if cfg.ambient_pressure is not None:
        return AmbientPressure(model, cfg, n, device, world_start)
    flagged = int(np.count_nonzero(seal_flags(model)))
    if flagged:
        raise ValueError(f"SolverDFSPH: {flagged} shapes set sph:shape_seal but Config.ambient_pressure is None "
                         f"(only an ambient pressure reads it)")
    return None


def seal_flags(model) -> np.ndarray:
    """``sph:shape_seal`` per model shape as int32 (0 where unregistered); at least one element."""
    S = int(model.shape_count)
    arr = getattr(getattr(model, "sph", None), "shape_seal", None)
    flags = np.zeros(max(S, 1), np.int32)
    if arr is not None and S:
        flags[:S] = (arr.numpy()[:S] != 0).astype(np.int32)
    return flags


__all__ = ["AmbientPressure", "air_gradient", "air_push_kernel", "cavitation_kernel", "gauge_reaction_kernel",
           "make_ambient", "prime_bound_kernel", "resid_terms_kernel", "resid_zero_kernel", "seal_bound_kernel",
           "seal_flags", "warm_offset_kernel"]
