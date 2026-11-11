# SPDX-License-Identifier: Apache-2.0
"""Heat v2 phase-change laws for ``ParticleThermal`` (docs/notes/heat-v2-design.md §9), on still particle lattices.

H-3, H-6, H-6b and H-9 run on uniform blocks, where conduction is exactly zero: they are identity checks of the phase
arithmetic and the ledgers. H-4 is the law with phase change under conduction. A two-phase (Neumann) law with water-like
ratios is not run: α_s ≈ 7 α_l makes the solid's thermal penetration ~10 cm (100 particles) at a 5 mm front, too large
for a CPU law; it is noted in the design note §14.

- H-3 ``latent_plateau``: a uniform block under volumetric power q. Its temperature rises at q/c_s, holds at the melting
  point for L/q while φ rises linearly, then rises at q/c_l. Uniform, so conduction is exactly zero: bound 1e-6 K.
- H-4 ``stefan_front``: a fixed-temperature wall (``law_reservoir``, 1e6 times the heat capacity) against a solid at
  its melting point. The one-phase Stefan solution gives s(t) = 2λ √(α t), with λ e^{λ²} erf λ = St / √π. The front
  is measured from the reservoir's last particle centre, where the wall temperature sits: s = a Σφ + a/2 per core
  column (a the spacing; a Σφ is the melted length, the enthalpy method's own measure). Measured +1.05 %; WP13's
  side-face-free copy gives −0.1 %, so the +1.05 % is this 7 × 7 bar's free side faces (H-1's residual). Bound 2 %
  for t ≥ 1 s: H-1's 7-across residual (+1.8 %) plus the half-cell shift's quasi-steady error δ²/(2s), about 0.1 %
  here (WP13 reviews of steps 2 and 3). The raw melted length alone lags by 0.4-0.5 a: −4.3 % in the bulk, −3.5 % on
  this bar; it is reported in the detail. dt 3e-4 and 5e-4 agree to 4e-6.
- H-6 ``boil_rate``: water just below boiling under volumetric power. T holds at T_b, and the evaporated mass grows at
  q M / L_v. Bound 1e-6 relative.
- H-6b ``boil_off``: water at T_b boils off completely. Evaporated mass = M, the ledger closes (Σ m ΔH + boiled =
  heater heat), and a gone particle takes no more heat. Bound 1e-9 relative.
- H-8a ``flow_arrays`` (an identity check: the kernel against the host copy of the same formula, so it catches index
  and precision errors but not a unit slip made on both sides): a bar with a temperature ramp across melting
  (290-310 K) on a model with both SPH and MPM attributes. After it diffuses, every particle's ``sph.viscosity`` and
  ``mpm.viscosity`` / ``yield_stress`` / ``friction`` / ``young_modulus`` equal the host law at its (T, φ), within
  float32 rounding (1e-6 relative).
- H-8b ``melt_spreads``: in each world two 6 × 6 × 10 paste columns (2 mm particles, 4 mm voxels: 2 per voxel per
  axis), 40 mm apart, on the ground under Newton's implicit MPM: ``law_paste`` and its 10x-viscosity twin. World 0
  stays at 290 K (solid: yield 20 kPa, 50 times ρ g h), world 1 starts at 310 K (melted: 0.34 and 3.4 Pa·s, no yield,
  no friction). After 0.5 s the solid columns keep their height (≤ 0.5 mm sag), the melted one slumps below half the
  solid's, and the melted heights scale with viscosity as a viscous gravity current's: h(10μ)/h(μ) = 10^(1/4) = 1.78
  (Huppert 1982) within 25 %: measured 1.51 on surface heights (top centre + radius; WP13's probe 1.65 on centre
  heights). The film is about one voxel thin at 0.5 s and the early flow is not yet in the viscous similarity regime
  (Re ≈ 13), so the bound is wide; a viscosity that did not reach MPM would read 1.
  Before the voxel was set (WP12, 2026-11-06), Newton's default 0.1 m voxel held each column in one cell.
- H-9 ``doneness``: a block held at 350 K: k = 0.3575/s pinned as a literal, 1e-5 (catches a unit slip that a formula
  on both sides would hide); doneness = 1 − exp(−k t), exact per step, 1e-9. ``doneness_ramp``: a block heated at
  20 K/s from 340 K, against 1 − exp(−∫k dt) by quadrature. The kernel's end-of-step rule errs by (dt/2)(k_end − k_0)
  to leading order at the run's exchange step: the measured error must match that prediction within 1 % (it does to
  0.01 %; WP13 review of step 3).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from scipy.optimize import brentq
from scipy.special import erf

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.thermal.particle_laws import LABEL, SPACING, SUPPORT, _set, bar_scene
from alloy.plugins.thermal.phase_materials import material

# --- H-3 -----------------------------------------------------------------------------------------------------------

Q3, T03 = 1.0e5, 290.0  # W/kg, K


def _plateau(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = material("law_pcm")
    t1 = m.c_solid * (m.t_solidus - T03) / Q3
    t2 = t1 + m.latent / Q3
    temp = np.where(t < t1, T03 + Q3 * t / m.c_solid,
                    np.where(t <= t2, m.t_solidus, m.t_liquidus + Q3 * (t - t2) / m.c_liquid))
    return temp, np.clip((t - t1) / (t2 - t1), 0.0, 1.0)


def _check_plateau(run) -> LawResult:
    t = np.asarray(run.trace.t)
    temp = run.trace.plugin(LABEL, "temperature")[:, 0]
    phi = run.trace.plugin(LABEL, "liquid")[:, 0]
    want_t, want_phi = _plateau(t)
    err = max(np.abs(temp - want_t[:, None]).max(), 300.0 * np.abs(phi - want_phi[:, None]).max())
    return LawResult.within(err, 1.0e-6, "max |T − T(t)| [K] (and 300·|φ − φ(t)|) through solid, plateau and liquid")


# --- H-4 -----------------------------------------------------------------------------------------------------------

RES_LAYERS, PCM_LAYERS, NY4 = 6, 30, 7
T_WALL = 320.0
RES_X = RES_LAYERS * SPACING  # the wall: the reservoir's last cell boundary


def stefan_lambda(stefan: float) -> float:
    """λ of the one-phase Stefan problem: λ e^{λ²} erf λ = St / √π."""
    return brentq(lambda x: x * np.exp(x * x) * erf(x) - stefan / np.sqrt(np.pi), 1e-6, 5.0)


def _melted(phi: np.ndarray, q: np.ndarray) -> float:
    """Melted length per core column (|y|, |z| ≤ 1 spacing): spacing × Σ φ over the column's melting particles,
    i.e. the latent heat absorbed / (ρ L A). The enthalpy method's own front measure, free of the stair-step that a
    φ = 0.5 crossing shows while a kernel-wide band of particles melts together."""
    core = (np.abs(q[:, 1]) < 1.5 * SPACING) & (np.abs(q[:, 2] - 0.02) < 1.5 * SPACING) & (q[:, 0] > RES_X)
    cols = {(round(y / SPACING), round((z - 0.02) / SPACING)) for y, z in q[core][:, 1:]}
    lengths = [SPACING * phi[core & (np.round(q[:, 1] / SPACING) == a) & (np.round((q[:, 2] - 0.02) / SPACING) == b)
                             ].sum() for a, b in cols]
    return float(np.mean(lengths)) if lengths else float("nan")


def _check_stefan(run) -> LawResult:
    m = material("law_stefan_pcm")
    lam = stefan_lambda(m.c_liquid * (T_WALL - m.t_solidus) / m.latent)
    t = np.asarray(run.trace.t)
    phi = run.trace.plugin(LABEL, "liquid")[:, 0]
    q = run.trace.particle_q[0][: phi.shape[1]]
    errs, raws = [], []
    for k in np.nonzero(t >= 1.0 - 1e-9)[0]:
        want = 2.0 * lam * np.sqrt(m.diffusivity(liquid=True) * t[k])
        melted = _melted(phi[k], q)
        errs.append((melted + 0.5 * SPACING) / want - 1.0)
        raws.append(melted / want - 1.0)
    if not errs:
        return LawResult(False, float("nan"), 0.02, "no frame at t >= 1 s", "run the law for at least 1 s")
    worst = max(errs, key=abs)
    return LawResult.within(worst, 0.02, f"front a·Σφ + a/2 vs 2λ√(αt) (λ = {lam:.4f}) for t >= 1 s: worst "
                            f"{worst:+.4f}; raw melted length {min(raws):+.4f} to {max(raws):+.4f}",
                            "check the mushy-zone T(H) and the conduction into the solid at T_m")


# --- H-6 -----------------------------------------------------------------------------------------------------------

Q6, T06 = 1.0e5, 372.15  # W/kg, K (1 K below boiling)
Q6_OFF = 5.0e6  # W/kg


def _check_boil(run) -> LawResult:
    w = material("water")
    t = np.asarray(run.trace.t)
    temp = run.trace.plugin(LABEL, "temperature")[:, 0]
    evap = run.trace.plugin(LABEL, "evaporated_kg")[:, 0]
    mass = float(run.sim.model.particle_mass.numpy().astype(np.float64).sum())
    t_b = w.c_liquid * (w.t_boil - T06) / Q6
    want = Q6 * mass * np.clip(t - t_b, 0.0, None) / w.latent_vapor
    rel = np.abs(evap[-1] - want[-1]) / want[-1]
    hold = np.abs(temp[t > t_b + 0.02] - w.t_boil).max()
    return LawResult.within(max(rel, hold), 1.0e-6, f"evaporated mass vs q·M·(t − t_b)/L_v (rel {rel:.2e}); "
                            f"|T − T_b| after onset {hold:.2e} K")


def _check_boil_off(run) -> LawResult:
    stored = run.trace.plugin(LABEL, "stored_J")[:, 0]
    heated = run.trace.plugin(LABEL, "heat_in_J")[:, 0]
    boiled = run.trace.plugin(LABEL, "boiled_J")[:, 0]
    evap = run.trace.plugin(LABEL, "evaporated_kg")[:, 0]
    mass = float(run.sim.model.particle_mass.numpy().astype(np.float64).sum())
    ledger = abs((stored[-1] - stored[0]) + boiled[-1] - heated[-1]) / heated[-1]
    gone = abs(evap[-1] - mass) / mass
    still = abs(heated[-1] - heated[-3]) / heated[-1]  # no heat into gone particles over the last two periods
    return LawResult.within(max(ledger, gone, still), 1.0e-9, f"ledger {ledger:.1e}, evaporated − M {gone:.1e}, "
                            f"heat after boil-off {still:.1e} (relative)")


# --- H-8 -----------------------------------------------------------------------------------------------------------

PASTE, PASTE_X10 = "law_paste", "law_paste_x10"


def _with_flow_attributes(scene):
    """``scene`` with SPH and MPM per-particle attributes registered on its template (no SPH or MPM solver)."""
    from alloy.solvers.sph.config import register_custom_attributes  # noqa: PLC0415 - SPH package

    inner = scene.build

    def build(b: newton.ModelBuilder) -> None:
        register_custom_attributes(b)
        newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
        inner(b)
    scene.build = build
    return scene


def _ramp(q: np.ndarray) -> np.ndarray:
    return 290.0 + 20.0 * q[..., 0] / q[..., 0].max()


def _check_flow_arrays(run) -> LawResult:
    m, model = material(PASTE), run.sim.model
    temp = run.trace.plugin(LABEL, "temperature")[-1, 0]
    phi = run.trace.plugin(LABEL, "liquid")[-1, 0]
    ids = run.sim.plugins[LABEL]._gidx.numpy()[: temp.size]
    want = {"sph.viscosity": m.viscosity(temp, phi), "mpm.viscosity": m.viscosity(temp, phi),
            "mpm.yield_stress": m.yield_stress(phi), "mpm.friction": m.friction(phi), "mpm.young_modulus": m.young(phi)}
    errs = {}
    for key, value in want.items():
        ns, name = key.split(".")
        got = getattr(getattr(model, ns), name).numpy()[ids].astype(np.float64)
        errs[key] = float(np.max(np.abs(got - value) / np.maximum(np.abs(value), 1e-30)))
    worst = max(errs, key=errs.get)
    return LawResult.within(errs[worst], 1.0e-6, f"max relative error of the written flow arrays vs the host law "
                            f"(worst {worst}); φ spans {phi.min():.2f}-{phi.max():.2f}",
                            "check phase_kernels.write_flow against PhaseMaterial.viscosity / yield_stress / young")


SPREAD_N, SPREAD_H, SPREAD_S = 6, 10, 2.0e-3  # columns, layers, spacing [m]
SPREAD_VOXEL = 2.0 * SPREAD_S  # m: 2 particles per voxel per axis (Alloy's emitters); Newton's default is 0.1 m


SPREAD_X = (-0.02, 0.02)  # m: column A (law_paste) and column B (its 10x-viscosity twin), 40 mm apart


def _columns(b: newton.ModelBuilder) -> None:
    newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
    n, s = SPREAD_N, SPREAD_S
    for x0 in SPREAD_X:
        for i in range(n):
            for j in range(n):
                for k in range(SPREAD_H):
                    pos = wp.vec3(x0 + s * (i - 0.5 * (n - 1)), s * (j - 0.5 * (n - 1)), s * (k + 0.5))
                    b.add_particle(pos, wp.vec3(0.0, 0.0, 0.0), 1000.0 * s ** 3, 0.5 * s)


def _cold_and_hot(sim) -> None:
    therm = sim.plugins[LABEL]
    therm.set_temperature(np.array([290.0, 310.0])[:, None] * np.ones((1, therm.P)))


def _check_spread(run) -> LawResult:
    """Surface heights (top particle centre + its radius) per column: world 0 solid, world 1 melted."""
    q = np.asarray(run.trace.particle_q)
    n = SPREAD_N * SPREAD_N * SPREAD_H
    top = lambda k, w, c: q[k, (2 * w + c) * n: (2 * w + c + 1) * n, 2].max() + 0.5 * SPREAD_S  # noqa: E731
    sag = max(top(0, 0, c) - top(-1, 0, c) for c in (0, 1))
    slump = top(-1, 1, 0) / top(-1, 0, 0)
    scaling = top(-1, 1, 1) / top(-1, 1, 0)
    want = 10.0 ** 0.25
    err = max(abs(scaling / want - 1.0) / 0.25, slump / 0.5, sag / 0.5e-3)
    return LawResult.within(err, 1.0,
                            f"melted height at 10x / 1x viscosity {scaling:.3f} vs Huppert 1982's 10^(1/4) = "
                            f"{want:.3f} (25 %); melted / solid {slump:.3f} (<= 0.5); solid sag {sag * 1e3:.2f} mm "
                            f"(<= 0.5)", "check that write_flow reaches the MPM arrays (has_viscosity is fixed at "
                            "build)")


# --- H-9 -----------------------------------------------------------------------------------------------------------

T8 = 350.0


K8 = 0.3575  # 1/s: 1e10 exp(−7e4 / (8.314462618 × 350)), written out (not the formula the kernel uses)
Q8, T8R = 2.0e4, 340.0  # W/kg (20 K/s at c = 1000), K


def _check_doneness(run) -> LawResult:
    t = np.asarray(run.trace.t)
    done = run.trace.plugin(LABEL, "done")[:, 0]
    k = float(material("law_doneness").doneness_k(T8))
    err = np.abs(done - (1.0 - np.exp(-k * t))[:, None]).max()
    k_rel = abs(k / K8 - 1.0)
    return LawResult.within(max(err / 1.0e-9, k_rel / 1.0e-5), 1.0,
                            f"max |done − (1 − e^{{−k t}})| at 350 K = {err:.1e} (bound 1e-9); table k vs the pinned "
                            f"{K8}/s off by {k_rel:.1e} (bound 1e-5); value = the worse of the two over its bound")


def _check_doneness_ramp(run) -> LawResult:
    from scipy.integrate import quad  # noqa: PLC0415

    m = material("law_doneness")
    t = np.asarray(run.trace.t)
    done = run.trace.plugin(LABEL, "done")[-1, 0]
    rate = Q8 / m.c_liquid
    integral = quad(lambda s: float(m.doneness_k(T8R + rate * s)), 0.0, t[-1], epsabs=1e-14, epsrel=1e-12)[0]
    got = -np.log1p(-done)
    dt = run.scene.control_dt / 2 ** int(run.sim.clock.j)
    want = 0.5 * dt * float(m.doneness_k(T8R + rate * t[-1]) - m.doneness_k(T8R))
    ratio = (got - integral) / want
    worst = float(ratio[np.argmax(np.abs(ratio - 1.0))])
    return LawResult.within(abs(worst - 1.0), 0.01,
                            f"(∫k dt from doneness − quadrature {integral:.6f}) / ((dt/2)(k_end − k_0)) = {worst:.5f} "
                            f"at dt = {dt:.4g} s under a 20 K/s ramp; bound |ratio − 1| <= 1 %")


# --- H-9c / H-8c: the starch sauce --------------------------------------------------------------------------------

SAUCE = "starch_sauce"
RAMP9C, T09C, SCALE9C, PERIODS9C = 5.0 / 60.0, 293.15, 171.0, 200  # K/s (5 K/min), K, time scale, periods
TARGETS9C = ((333.15, 0.18), (350.15, 0.95))  # the reference task's gelatinisation targets (60 °C, 77 °C)


def _check_gelatinise(run) -> LawResult:
    """The card's doneness under a 5 K/min ramp against the exact quadrature at every frame, and at 60 / 77 °C."""
    from scipy.integrate import quad  # noqa: PLC0415

    m = material(SAUCE)
    temp = run.trace.plugin(LABEL, "temperature")[:, 0].mean(axis=1)
    done = run.trace.plugin(LABEL, "done")[:, 0].mean(axis=1)
    exact = np.array([1.0 - np.exp(-quad(lambda t: float(m.doneness_k(t)), T09C, max(T, T09C), epsabs=1e-14)[0]
                                    / RAMP9C) for T in temp])
    track = float(np.abs(done - exact).max())
    at = [float(np.interp(T, temp, done)) for T, _ in TARGETS9C]
    target = max(abs(x - want) for x, (_, want) in zip(at, TARGETS9C))
    return LawResult.within(max(track / 2.0e-3, target / 0.01), 1.0,
                            f"doneness vs quadrature of k(T) under 5 K/min: max |Δ| {track:.1e} (bound 2e-3, the "
                            f"end-point rule's bias); at 60 / 77 °C {at[0]:.4f} / {at[1]:.4f} vs the reference "
                            f"task's 0.18 / 0.95 (0.01)",
                            "starch_sauce's doneness_rate / doneness_energy (phase_materials)")


def _sauce_ramp(q: np.ndarray) -> np.ndarray:
    return 335.0 + 20.0 * q[..., 0] / q[..., 0].max()   # 62-82 °C across x: gelatinising at different rates


def _check_flow_done(run) -> LawResult:
    m, model = material(SAUCE), run.sim.model
    temp, phi, done = (run.trace.plugin(LABEL, key)[-1, 0] for key in ("temperature", "liquid", "done"))
    ids = run.sim.plugins[LABEL]._gidx.numpy()[: temp.size]
    want = m.viscosity(temp, phi, done)
    errs = {key: float(np.max(np.abs(getattr(model, key[:3]).viscosity.numpy()[ids] - want) / want))
            for key in ("sph.viscosity", "mpm.viscosity")}
    worst = max(errs, key=errs.get)
    return LawResult.within(errs[worst], 1.0e-6,
                            f"written viscosity vs μ(T, φ, X) (worst {worst}); doneness spans {done.min():.2f}-"
                            f"{done.max():.2f}, μ {want.min():.3g}-{want.max():.3g} Pa·s",
                            "phase_kernels.viscosity's doneness term against PhaseMaterial.viscosity")


def phase_laws() -> list[Law]:
    base = {"support": SUPPORT}
    res_n = RES_LAYERS * NY4 * NY4
    return [
        Law("latent_plateau", _check_plateau, scene=bar_scene(6, 6, 6, periods=50, worlds=1),
            config={**base, "materials": ("law_pcm",), "initial": (T03,), "power_density": (Q3,)},
            doc="H-3: solid rise q/c_s, plateau L/q at T_m, liquid rise q/c_l; 1e-6 K"),
        Law("stefan_front", _check_stefan, scene=bar_scene(RES_LAYERS + PCM_LAYERS, NY4, NY4, periods=100, worlds=1,
                                                           dt=5.0e-4),
            config={**base, "materials": ("law_reservoir", "law_stefan_pcm"), "ranges": ((0, res_n), (res_n, -1)),
                    "initial": (T_WALL, 300.0)},
            doc="H-4: one-phase Stefan melt front 2λ√(αt), 2 % for t >= 1 s"),
        Law("boil_rate", _check_boil, scene=bar_scene(5, 5, 5, periods=25, worlds=1, rho=997.05),
            config={**base, "materials": ("water",), "initial": (T06,), "power_density": (Q6,)},
            doc="H-6: T holds at T_b, evaporated mass at q·M/L_v; 1e-6"),
        Law("boil_off", _check_boil_off, scene=bar_scene(3, 3, 3, periods=30, worlds=1, rho=997.05),
            config={**base, "materials": ("water",), "initial": (373.15,), "power_density": (Q6_OFF,)},
            doc="H-6b: complete boil-off: evaporated = M, ledger closes, gone particles take no heat; 1e-9"),
        Law("flow_arrays", _check_flow_arrays,
            scene=_with_flow_attributes(bar_scene(20, 4, 4, periods=10, worlds=1, init=_set(_ramp))),
            config={**base, "materials": (PASTE,)},
            doc="H-8a (identity check): written SPH/MPM flow arrays equal the host law at each particle's (T, φ); "
                "1e-6 relative"),
        Law("melt_spreads", _check_spread,
            scene=TestScene(build=_columns, worlds=2, periods=25, control_dt=0.02, dt=1.0e-3, solver="mpm",
                            solver_kwargs={"voxel_size": SPREAD_VOXEL}, init=_cold_and_hot),
            config={"materials": (PASTE, PASTE_X10), "initial": (290.0, 290.0), "support": 2.2 * SPREAD_S,
                    "ranges": ((0, SPREAD_N ** 2 * SPREAD_H), (SPREAD_N ** 2 * SPREAD_H, -1))},
            doc="H-8b: melted paste columns at 1x and 10x viscosity: heights scale as Huppert's μ^(1/4) (25 %), the "
                "melt slumps below half the solid, the solid sags <= 0.5 mm"),
        Law("doneness", _check_doneness, scene=bar_scene(4, 4, 4, periods=50, worlds=1),
            config={**base, "materials": ("law_doneness",), "initial": (T8,)},
            doc="H-9: doneness 1 − exp(−k t) at 350 K, 1e-9, with k pinned at 0.3575/s (1e-5)"),
        Law("doneness_ramp", _check_doneness_ramp, scene=bar_scene(4, 4, 4, periods=50, worlds=1),
            config={**base, "materials": ("law_doneness",), "initial": (T8R,), "power_density": (Q8,)},
            doc="H-9b: doneness under a 20 K/s ramp vs quadrature of k(T(t)): the error is the end-point rule's "
                "(dt/2)(k_end − k_0) within 1 %"),
        Law("gelatinise_ramp", _check_gelatinise, scene=bar_scene(3, 3, 3, periods=PERIODS9C, worlds=1, rho=1083.0),
            config={**base, "materials": (SAUCE,), "initial": (T09C,), "time_scale": SCALE9C,
                    "power_density": (material(SAUCE).c_liquid * RAMP9C,)},
            doc="H-9c: starch_sauce gelatinises under a 5 K/min ramp as the quadrature of its k(T) (2e-3) and as "
                "the reference task's targets, 18 % at 60 °C and 95 % at 77 °C (0.01)"),
        Law("flow_doneness", _check_flow_done,
            scene=_with_flow_attributes(bar_scene(20, 3, 3, periods=10, worlds=1, rho=1083.0,
                                                  init=_set(_sauce_ramp))),
            config={**base, "materials": (SAUCE,), "time_scale": 400.0},
            doc="H-8c (identity check): the written SPH / MPM viscosity equals μ(T, φ, X) for a sauce gelatinising "
                "at 62-82 °C; 1e-6 relative"),
    ]
