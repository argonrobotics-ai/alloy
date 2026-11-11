# SPDX-License-Identifier: Apache-2.0
"""Thermal buoyancy for heated liquid particles: a Boussinesq body force from temperature (gaps-now #8, 2026-11-08).

A parcel of liquid lighter than the liquid around it rises: F_i = m_i φ_i (ρ(T_i) - ρ̄) / ρ̄ · g, with m_i the
particle's mass, φ_i its liquid fraction, ρ(T) the liquid's density at temperature T and g the world's gravity
(pointing down, so a lighter parcel, ρ < ρ̄, is pushed up). SPH itself keeps one rest density, so a hot particle
does not expand. This force adds the buoyancy that the expansion would give, the Boussinesq approximation (density
differences of a few percent; water 20 -> 80 C: 2.6 %).

The reference ρ̄ is the liquid-mass-weighted mean density of all the liquid of the same heat material in the same
world (every heat group of it: a hot layer and a cold layer added as two liquids share one), taken every step: ρ̄ = Σ
m φ ρ / Σ m φ. The forces then sum to zero over that liquid (to the 2^-40 fixed point of the sums, about 1e-8 of its
weight), so heating it all evenly changes nothing (water does not weigh less when warm), and only differences of
temperature drive flow. Limit: separate bodies of one material (a hot cup and a cold cup of water) share one ρ̄, so
each feels a small force against its container, a weight error of the order of their density difference (60 K: about
1.5 % of the hot cup's weight).

ρ(T) comes from a sourced table per heat material (``PhaseMaterial.density_table``; water: NIST Chemistry WebBook
SRD 69 at 0.101325 MPa, 0.01-95.01 C in 5 K steps, with its 4 C maximum: phase_materials.WATER_DENSITY). Between rows
it is linear; beyond the first and last rows it follows the end segments, out to 5 K past them. A heat material with
no table gets no buoyancy (``Buoyancy.skipped``). The sums are int64 fixed point, so the force is deterministic.

``Buoyancy.build(...)`` sets it up from the heat plug-in's slots (alloy.plugins.thermal.particle.ParticleThermal);
``apply(...)`` adds the forces into a ``particle_f`` buffer before the solver step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

Table = tuple[tuple[float, ...], tuple[float, ...]]   # (T [K], ρ [kg/m³]) rows, T increasing
PAST = 5.0        # K: the end segments extend this far beyond a table's first and last rows
SCALE = wp.constant(1099511627776.0)  # 2^40 fixed point for the per-object sums (mass and mass x density)
_PAST = wp.constant(wp.float64(PAST))


def density(table: Table, temperature_k: float | np.ndarray) -> np.ndarray:
    """ρ(T) [kg/m³] from ``table`` (host side, the same interpolation as the kernel)."""
    t_k, rho = (np.asarray(a, np.float64) for a in table)
    t = np.clip(np.asarray(temperature_k, np.float64), t_k[0] - PAST, t_k[-1] + PAST)
    k = np.clip(np.searchsorted(t_k, t, side="right") - 1, 0, t_k.size - 2)
    f = (t - t_k[k]) / (t_k[k + 1] - t_k[k])
    return rho[k] + f * (rho[k + 1] - rho[k])


@wp.func
def _rho(t_k: wp.float64, table: wp.array[wp.float64], t_c: wp.array[wp.float64], start: int,
         n: int) -> wp.float64:
    t = wp.clamp(t_k, t_c[start] - _PAST, t_c[start + n - 1] + _PAST)
    k = int(0)
    for i in range(1, n - 1):
        if t >= t_c[start + i]:
            k = i
    a, b = t_c[start + k], t_c[start + k + 1]
    ra, rb = table[start + k], table[start + k + 1]
    return ra + (t - a) / (b - a) * (rb - ra)


@wp.kernel(enable_backward=False)
def _sums_kernel(temperature: wp.array[wp.float64], liquid: wp.array[wp.float64], mass: wp.array[wp.float64],
                 obj: wp.array[wp.int32], tab: wp.array[wp.int32], tab_start: wp.array[wp.int32],
                 tab_n: wp.array[wp.int32], rho_table: wp.array[wp.float64], t_table: wp.array[wp.float64],
                 per_world: int, n_obj: int, rho: wp.array[wp.float64], sum_m: wp.array[wp.int64],
                 sum_mr: wp.array[wp.int64]):
    s = wp.tid()
    p = s % per_world
    k = tab[p]
    if k < 0:
        return
    r = _rho(temperature[s], rho_table, t_table, tab_start[k], tab_n[k])
    rho[s] = r
    wm = mass[s] * wp.clamp(liquid[s], wp.float64(0.0), wp.float64(1.0))
    o = (s // per_world) * n_obj + obj[p]
    wp.atomic_add(sum_m, o, wp.int64(wm * wp.float64(SCALE)))
    wp.atomic_add(sum_mr, o, wp.int64(wm * r * wp.float64(SCALE) / wp.float64(1000.0)))


@wp.kernel(enable_backward=False)
def _force_kernel(liquid: wp.array[wp.float64], mass: wp.array[wp.float64], obj: wp.array[wp.int32],
                  tab: wp.array[wp.int32], gidx: wp.array[wp.int32], gravity: wp.array[wp.vec3], per_world: int,
                  n_obj: int, rho: wp.array[wp.float64], sum_m: wp.array[wp.int64], sum_mr: wp.array[wp.int64],
                  particle_f: wp.array[wp.vec3]):
    s = wp.tid()
    p = s % per_world
    if tab[p] < 0:
        return
    w = s // per_world
    o = w * n_obj + obj[p]
    if sum_m[o] <= wp.int64(0):
        return
    rho_bar = wp.float64(1000.0) * wp.float64(sum_mr[o]) / wp.float64(sum_m[o])
    wm = mass[s] * wp.clamp(liquid[s], wp.float64(0.0), wp.float64(1.0))
    c = wm * (rho[s] - rho_bar) / rho_bar            # x g: up where rho < rho_bar
    g = gravity[w]
    i = gidx[s]
    particle_f[i] = particle_f[i] + wp.vec3(float(c) * g[0], float(c) * g[1], float(c) * g[2])


@dataclass
class Buoyancy:
    """Device tables and per-object sums for one heat plug-in's slots (see the module docstring)."""

    tab: wp.array            # [P] table index per slot of a world (-1: no buoyancy)
    obj: wp.array            # [P] heat material per slot of a world (the reference's liquid)
    tab_start: wp.array
    tab_n: wp.array
    rho_table: wp.array
    t_table: wp.array
    rho: wp.array            # [W * P] ρ(T) of the step (scratch)
    sum_m: wp.array          # [W * n_obj] int64, n_obj = the distinct heat materials
    sum_mr: wp.array
    per_world: int
    n_obj: int
    skipped: tuple[str, ...]  # heat materials with no density table

    @classmethod
    def build(cls, materials: Sequence[str], tables: Mapping[str, Table], slot_group: np.ndarray, worlds: int,
              device: str) -> Buoyancy:
        """``materials``: the heat material of each group; ``tables``: ρ(T) per material name (missing or empty: no
        buoyancy); ``slot_group``: the group of each slot of one world."""
        tables = {n: t for n, t in tables.items() if t and len(t[0]) >= 2}
        names = list(dict.fromkeys(m for m in materials if m in tables))
        t_c = np.concatenate([np.asarray(tables[n][0], np.float64) for n in names]) if names else np.zeros(2)
        rho = np.concatenate([np.asarray(tables[n][1], np.float64) for n in names]) if names else np.zeros(2)
        sizes = [len(tables[n][0]) for n in names]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32) if names else np.zeros(1, np.int32)
        group_tab = np.asarray([names.index(m) if m in tables else -1 for m in materials], np.int32)
        distinct = list(dict.fromkeys(materials))
        group_obj = np.asarray([distinct.index(m) for m in materials], np.int32)
        slot_group = np.asarray(slot_group, np.int32)
        per_world, n_obj = int(slot_group.size), max(len(distinct), 1)

        def arr(a, dt):
            return wp.array(a, dtype=dt, device=device)

        return cls(tab=arr(group_tab[slot_group], wp.int32), obj=arr(group_obj[slot_group], wp.int32),
                   tab_start=arr(starts, wp.int32), tab_n=arr(np.asarray(sizes or [2], np.int32), wp.int32),
                   rho_table=arr(rho, wp.float64), t_table=arr(t_c, wp.float64),
                   rho=wp.zeros(per_world * worlds, dtype=wp.float64, device=device),
                   sum_m=wp.zeros(worlds * n_obj, dtype=wp.int64, device=device),
                   sum_mr=wp.zeros(worlds * n_obj, dtype=wp.int64, device=device), per_world=per_world,
                   n_obj=n_obj, skipped=tuple(sorted({m for m in materials if m not in tables})))

    def apply(self, temperature: wp.array, liquid: wp.array, mass: wp.array, gidx: wp.array, gravity: wp.array,
              particle_f: wp.array) -> None:
        """Add the buoyancy forces [N] into ``particle_f`` (global particle ids ``gidx`` per slot). ``temperature``
        [K], ``liquid`` (fraction 0-1) and ``mass`` [kg] are flat float64 slot arrays, world-major."""
        n = int(temperature.shape[0])
        if n == 0:
            return
        dev = temperature.device
        self.sum_m.zero_()
        self.sum_mr.zero_()
        wp.launch(_sums_kernel, dim=n, device=dev,
                  inputs=[temperature, liquid, mass, self.obj, self.tab, self.tab_start, self.tab_n, self.rho_table,
                          self.t_table, self.per_world, self.n_obj],
                  outputs=[self.rho, self.sum_m, self.sum_mr])
        wp.launch(_force_kernel, dim=n, device=dev,
                  inputs=[liquid, mass, self.obj, self.tab, gidx, gravity, self.per_world, self.n_obj, self.rho,
                          self.sum_m, self.sum_mr],
                  outputs=[particle_f])
