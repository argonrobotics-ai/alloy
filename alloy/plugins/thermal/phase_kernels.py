# SPDX-License-Identifier: Apache-2.0
"""Heat v2 step 3 device kernels: melt -> flow feedback (docs/notes/heat-v2-design.md §6).

After each heat update, every thermal particle whose material flows (``PhaseMaterial.flows``) writes its material
values into the solvers' per-particle arrays, indexed by global particle id (entry ModelViews share them):
- SPH: ``model.sph.viscosity`` (WP9's implicit ``weiler2018`` viscosity reads it every step);
- MPM: ``model.mpm.viscosity``, ``yield_stress``, ``friction`` and ``young_modulus`` (Newton's ``SolverImplicitMPM``
  binds these arrays as its ``material_parameters`` and reads them every step; WP12).

The values are a pure function of the particle's (T, φ, doneness). ``ParticleThermal`` writes them after each update
and again before every solver step (its ``apply``) from its own state, so after a reset or a rollback, which restore T
but not these model arrays, the solver never steps on values from the abandoned run (WP13 step-3 review). The law
and which parts of it are modelling choices: ``phase_materials`` module doc.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.thermal.particle_kernels import R_GAS, Slots

F64 = wp.float64
WRITE_SPH = wp.constant(1)
WRITE_MPM = wp.constant(2)


@wp.struct
class FlowTable:
    mu_l: wp.array[wp.float64]  # Pa·s at the liquidus (0: the material does not flow)
    mu_e: wp.array[wp.float64]  # J/mol
    mu_d: wp.array[wp.float64]  # Pa·s at the liquidus fully cooked (0: doneness leaves μ alone)
    mu_s: wp.array[wp.float64]  # Pa·s at φ = 0 (0: liquid law only)
    tau_s: wp.array[wp.float64]  # Pa
    fric_s: wp.array[wp.float64]  # internal friction at φ = 0
    n: wp.array[wp.float64]
    e_s: wp.array[wp.float64]  # Pa (0: leave Young's modulus alone)
    e_l: wp.array[wp.float64]  # Pa
    t_s: wp.array[wp.float64]  # K
    t_l: wp.array[wp.float64]  # K


@wp.struct
class FlowTargets:
    sph_viscosity: wp.array[float]
    mpm_viscosity: wp.array[float]
    mpm_yield: wp.array[float]
    mpm_friction: wp.array[float]
    mpm_young: wp.array[float]
    write: int  # WRITE_SPH | WRITE_MPM


@wp.func
def viscosity(t: wp.float64, phi: wp.float64, done: wp.float64, m: int, ft: FlowTable) -> wp.float64:
    tc = wp.max(t, ft.t_s[m])
    mu_ref = ft.mu_l[m]
    if ft.mu_d[m] > F64(0.0):
        mu_ref = wp.exp((F64(1.0) - done) * wp.log(ft.mu_l[m]) + done * wp.log(ft.mu_d[m]))
    mu_l = mu_ref * wp.exp(ft.mu_e[m] / F64(R_GAS) * (F64(1.0) / tc - F64(1.0) / ft.t_l[m]))
    mu_s = ft.mu_s[m]
    if mu_s <= F64(0.0):
        return mu_l
    return wp.exp((F64(1.0) - phi) * wp.log(mu_s) + phi * wp.log(mu_l))


@wp.func
def solid_weight(phi: wp.float64, m: int, ft: FlowTable) -> wp.float64:
    """(1 − φ)^n."""
    return wp.pow(wp.max(F64(1.0) - phi, F64(0.0)), ft.n[m])


@wp.kernel(enable_backward=False)
def write_flow(sl: Slots, temp: wp.array[wp.float64], liquid: wp.array[wp.float64], done: wp.array[wp.float64],
               ft: FlowTable, out: FlowTargets):
    s = wp.tid()
    m = sl.mat[s]
    if ft.mu_l[m] <= F64(0.0):
        return
    i = sl.gidx[s]
    phi = liquid[s]
    mu = float(viscosity(temp[s], phi, done[s], m, ft))
    if (out.write & WRITE_SPH) != 0:
        out.sph_viscosity[i] = mu
    if (out.write & WRITE_MPM) != 0:
        w = solid_weight(phi, m, ft)
        out.mpm_viscosity[i] = mu
        out.mpm_yield[i] = float(ft.tau_s[m] * w)
        out.mpm_friction[i] = float(ft.fric_s[m] * w)
        if ft.e_s[m] > F64(0.0):
            out.mpm_young[i] = float(ft.e_l[m] + (ft.e_s[m] - ft.e_l[m]) * w)
