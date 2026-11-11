# SPDX-License-Identifier: Apache-2.0
"""Heat v2 laws for ``ParticleThermal`` on still particle lattices (docs/notes/heat-v2-design.md §9).

Lattices: 1 mm spacing, particle radius 0.25 mm (so XPBD sees no contacts), gravity off. The particles do not move, so
the laws judge the heat update alone. The kernel support is 2.2 spacings: on a cubic lattice its discrete Laplacian
is within 0.2 % of the continuum (a 1.3 % excess at SPH's 2 spacings).

- H-1 ``diffusion_mode``: the cosine mode of an insulated 40 × 11 × 11 bar decays at α (π/L)² (kh = 0.09). Measured:
  without the free-surface correction −10.5 % (7 × 7 bar); with it +1.8 % on 7 × 7 and +1.1 % on 11 × 11, the same at
  20 and 40 long. The residual (WP13 review of step 1) is the second layer under each free face, which conducts about
  3.7 % too much along the face: +0.55 % per pair of free side faces at 11 particles across, scaling with 1 / the
  thickness. Edges and corners add about 0.03 %; with the side faces removed the error is −0.03 %. Bound 2 %: the
  predicted +1.1 % for two pairs of side faces at 11 across, with 1.8x margin. Bodies thinner than about 6 particles
  conduct along their surface more than 2 % fast.
- H-2 ``contact_erf``: two half-bars at T0 ± ΔT/2 touch; T = T0 + (ΔT/2) erf(x / (2 √(α t))). Bound: 1 % of ΔT on
  the bar's core columns at t = 1 s (measured 0.19 %).
- H-5 ``energy_closed`` / ``energy_heater`` / ``power_heater``: Σ m H is constant without sources, within 1e-9 of the
  heat exchanged (Σ |m ΔH| per particle). With a fixed-temperature heater it changes by the ledger's heat_in (1e-6
  relative). A power heater puts in exactly η P t, all of it into ACTIVE particles: 32 spares parked inside the plate
  get none (1e-9 relative). Each energy law also needs zero neighbour-row overflow.
- H-5d ``body_heater``: a bar on a floating 360 K pan, a WP13 ``Thermal`` body, beside a far "dummy" Thermal body,
  in 2 worlds (MuJoCo route, convection and friction heat off). In each world the particles' Σ m ΔH equals their
  heater ledger, the pan's booked plus pending external heat is exactly its negative (whole µJ both sides) and the
  dummy body gets nothing: 1e-9 relative.
- H-5f ``body_rate``: the same scene, the water 66.85 K and 3 K below the pan. Heat in matches h_eff·s² Σ (T_pan − T_i)
  over the touching particles (trapezoid per period) within 1 %. Each step's exchange is whole µJ with the sub-µJ
  remainder carried per particle (``body_residual``), so the error stays under 0.5 µJ per particle: 0.4 % of the
  3 K world's heat. Rounding without the carry (WP13 step-3 review) passed no heat below 4.1 K and ran 2.5 % slow at
  66.85 K.
- H-5e ``power_dead``: a 1 W power plate under water at T_b with half the bottom layer already evaporated. Heat in is
  still η P t (evaporated particles take no share), none of it idle or into the evaporated ones: 1e-9 relative.

- H-5g, H-5h: ``particle_area_laws``.

- H-7 ``null``: uniform temperature, no sources: T unchanged (1e-9 K).

Each law states its bound's source in ``doc``.

``busy_scene`` is the plug-in's test scene for the harness repeat checks (WP13 review of step 2): a non-uniform start,
a plate and 8 parked spares; ``BUSY`` adds a power plate, convection and a volumetric source that boils the bar off
near the end. XPBD particle-particle contacts are off (``particle_max_radius = 0``, no particle grid): Newton's XPBD
collides particles across co-located worlds, which would move world 0 when other worlds are perturbed (a scene
artefact, not a leak).
"""

from __future__ import annotations

from math import erf

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.thermal.phase_materials import material

SPACING = 1.0e-3  # m
RADIUS = 0.25e-3  # m
SUPPORT = 2.2 * SPACING
LABEL = "particle_thermal"
CONDUCTOR = "law_conductor"


def _lattice(nx: int, ny: int, nz: int, rho: float):
    """A world template filling an nx × ny × nz lattice with particles of mass ρ·spacing³, centred on y = z = 0."""
    mass = rho * SPACING ** 3

    def build(b: newton.ModelBuilder) -> None:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = wp.vec3((i + 0.5) * SPACING, (j - 0.5 * (ny - 1)) * SPACING,
                                  (k - 0.5 * (nz - 1)) * SPACING + 0.02)
                    b.add_particle(pos, wp.vec3(0.0, 0.0, 0.0), mass, RADIUS)
    return build


def bar_scene(nx: int = 40, ny: int = 7, nz: int = 7, periods: int = 50, worlds: int = 3, rho: float = 1000.0,
              plate: bool = False, init=None, dt: float = 1.0e-3, spares: int = 0) -> TestScene:
    """An insulated particle bar (optionally on a heater box labelled "plate"), still, gravity off. ``spares``
    inactive particles sit at one point inside the plate, as SPH parks its spares."""
    lattice = _lattice(nx, ny, nz, rho)

    def build(b: newton.ModelBuilder) -> None:
        lattice(b)
        if not plate:
            return
        top = 0.02 - 0.5 * (nz - 1) * SPACING - RADIUS - 1.0e-4  # a box under the bottom layer, 0.1 mm clear
        centre = wp.vec3(0.5 * nx * SPACING, 0.0, top - 2.0e-3)
        b.add_shape_box(-1, xform=wp.transform(centre, wp.quat_identity()), hx=0.5 * nx * SPACING + 1e-3,
                        hy=0.5 * ny * SPACING + 1e-3, hz=2.0e-3, label="plate")
        for _ in range(spares):
            b.add_particle(centre, wp.vec3(0.0, 0.0, 0.0), rho * SPACING ** 3, RADIUS, flags=0)
    return TestScene(build=build, worlds=worlds, periods=periods, control_dt=0.02, dt=dt, solver="xpbd",
                     ground=False, gravity=(0.0, 0.0, 0.0), init=init)


def _set(profile):
    """A scene init that sets the start temperature from ``profile(positions [N, 3]) -> [N]``."""
    def init(sim) -> None:
        therm = sim.plugins[LABEL]
        q = sim.model.particle_q.numpy()[therm._gidx.numpy()].reshape(therm._W, therm.P, 3)
        therm.set_temperature(profile(q))
    return init


def _bar_x(q: np.ndarray) -> np.ndarray:
    return q[..., 0]


# --- H-1 -----------------------------------------------------------------------------------------------------------

NX1 = 40
L1 = NX1 * SPACING


def _cos_mode(q: np.ndarray) -> np.ndarray:
    return 300.0 + 10.0 * np.cos(np.pi * _bar_x(q) / L1)


def _check_mode(run) -> LawResult:
    t = np.asarray(run.trace.t)
    temp = run.trace.plugin(LABEL, "temperature")[:, 0]  # [K, P] world 0
    x = _bar_x(run.trace.particle_q[0][: temp.shape[1]])
    c = np.cos(np.pi * x / L1)
    amp = ((temp - temp.mean(axis=1, keepdims=True)) * c).sum(axis=1) / (c * c).sum()
    rate = -np.polyfit(t, np.log(amp), 1)[0]
    want = material(CONDUCTOR).diffusivity() * (np.pi / L1) ** 2
    return LawResult.within(rate / want - 1.0, 0.02, f"cosine-mode decay rate {rate:.5g}/s vs α(π/L)² {want:.5g}/s",
                            "check the kernel support (2.2 spacings) and the conduction kernel's sign and factor 4")


# --- H-2 -----------------------------------------------------------------------------------------------------------

NX2 = 60
X_MID = 0.5 * NX2 * SPACING


def _step(q: np.ndarray) -> np.ndarray:
    return np.where(_bar_x(q) < X_MID, 310.0, 290.0)


def _check_erf(run) -> LawResult:
    t = float(run.trace.t[-1])
    temp = run.trace.plugin(LABEL, "temperature")[-1, 0]
    q = run.trace.particle_q[0][: temp.size]
    core = (np.abs(q[:, 1]) < 1.5 * SPACING) & (np.abs(q[:, 2] - 0.02) < 1.5 * SPACING)
    width = 2.0 * np.sqrt(material(CONDUCTOR).diffusivity() * t)
    want = np.array([300.0 - 10.0 * erf((x - X_MID) / width) for x in q[core, 0]])
    err = np.abs(temp[core] - want).max() / 20.0
    return LawResult.within(err, 0.01, f"max |T - erf| / ΔT on the core columns at t = {t:.2f} s",
                            "semi-infinite contact solution (Carslaw and Jaeger §2.4); check conduction")


# --- H-5, H-7 ------------------------------------------------------------------------------------------------------

def _random(q: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    return 293.15 + 60.0 * rng.random(q.shape[:-1])


def _overflowed(run) -> LawResult | None:
    """A failed result if any neighbour row overflowed (conservation needs complete rows), else None."""
    worst = int(run.trace.plugin(LABEL, "neighbor_overflow").max())
    if worst == 0:
        return None
    return LawResult(False, float(worst), 0.0, f"{worst} particles overflowed their neighbour rows in one exchange",
                     "raise neighbor_capacity")


def _check_closed(run) -> LawResult:
    stored = run.trace.plugin(LABEL, "stored_J")[:, 0]
    h = run.trace.plugin(LABEL, "enthalpy")[:, 0]
    exchanged = float((np.abs(h[-1] - h[0]) * run.sim.plugins[LABEL]._mass[: h.shape[1]]).sum())
    return _overflowed(run) or LawResult.within(
        (stored[-1] - stored[0]) / exchanged, 1.0e-9,
        f"change of Σ m H without sources / heat exchanged ({exchanged:.4g} J; pairwise-antisymmetric conduction)")


def _check_heater(run) -> LawResult:
    stored = run.trace.plugin(LABEL, "stored_J")[:, 0]
    heat_in = run.trace.plugin(LABEL, "heat_in_J")[:, 0]
    gained = stored[-1] - stored[0]
    if heat_in[-1] <= 0.0:
        return LawResult(False, 0.0, 1e-6, "the heater gave no heat", "check the heater gap and the plate position")
    return _overflowed(run) or LawResult.within(
        (gained - heat_in[-1]) / heat_in[-1], 1.0e-6,
        f"(Σ m ΔH − heater heat) / heater heat; heater gave {heat_in[-1]:.4g} J")


P5, ETA5, SPARES5 = 1.0, 0.8, 32  # W, efficiency, spares in the plate


def _check_power(run) -> LawResult:
    t = np.asarray(run.trace.t)
    want = ETA5 * P5 * (t[-1] - t[0])
    stored = run.trace.plugin(LABEL, "stored_J")[:, 0]
    heated = run.trace.plugin(LABEL, "heated")[-1, 0]
    h = run.trace.plugin(LABEL, "enthalpy")[:, 0]
    therm = run.sim.plugins[LABEL]
    spare = run.sim.model.particle_flags.numpy()[therm._gidx.numpy()[: h.shape[1]]] == 0
    mass = therm._mass[: h.shape[1]]
    to_spares = float(np.abs(heated[spare]).sum() + (mass[spare] * np.abs(h[-1, spare] - h[0, spare])).sum())  # J
    err = max(abs(heated.sum() - want), abs(stored[-1] - stored[0] - want), to_spares) / want
    return _overflowed(run) or LawResult.within(
        err, 1.0e-9, f"heat in and Σ m ΔH vs η·P·t = {want:.4g} J; heat into {int(spare.sum())} spares "
        f"{to_spares:.2e}", "power shares must use the same rounded contact area; inactive particles take nothing")


def _check_null(run) -> LawResult:
    temp = run.trace.plugin(LABEL, "temperature")[:, 0]
    return LawResult.within(np.abs(temp - temp[0]).max(), 1.0e-9, "max |T(t) − T(0)| [K] at uniform temperature")


PAN_T, PAN_C = 360.0, 500.0  # K, J/(kg·K)
PAN_WATER = (293.15, 357.0)  # K per world: ΔT 66.85 K and 3 K below the pan (the old rounding's dead band was 4.1 K)
PAN_H = 1000.0  # W/(m²·K): Config.heater_conductance's default


def _pan_scene() -> TestScene:
    """A 12 × 4 × 4 water bar 0.1 mm above a floating pan (a box shape labelled "plate") and a far "dummy" body, in 2
    worlds; both are WP13 ``Thermal`` bodies (the pan is slot w·2 + 1). MuJoCo, gravity off."""
    lattice = _lattice(12, 4, 4, 997.05)

    def build(b: newton.ModelBuilder) -> None:
        lattice(b)
        top = 0.02 - 1.5 * SPACING - RADIUS - 1.0e-4
        dummy = b.add_body(xform=wp.transform((0.0, 0.05, 0.0), wp.quat_identity()), label="dummy")
        b.add_shape_box(dummy, hx=2.0e-3, hy=2.0e-3, hz=2.0e-3)
        pan = b.add_body(xform=wp.transform((6.0 * SPACING, 0.0, top - 2.0e-3), wp.quat_identity()), label="pan")
        b.add_shape_box(pan, hx=7.0 * SPACING, hy=3.0 * SPACING, hz=2.0e-3, label="plate")

    def companions():
        from alloy.plugins.thermal.sensor import Thermal  # noqa: PLC0415 - WP13's heat v1

        return [Thermal(bodies=("dummy", "pan"), initial=(PAN_T, PAN_T), specific_heat=(PAN_C, PAN_C),
                        convection=0.0, friction_heat=False)]

    def init(sim) -> None:
        therm = sim.plugins[LABEL]
        therm.link_thermal(sim.plugins)
        therm.set_temperature(np.asarray(PAN_WATER)[:, None] * np.ones((1, therm.P)))
    return TestScene(build=build, worlds=2, periods=10, control_dt=0.02, dt=1.0e-3, solver="mujoco", ground=False,
                     gravity=(0.0, 0.0, 0.0), companions=companions, init=init)


def _check_body(run) -> LawResult:
    stored = run.trace.plugin(LABEL, "stored_J")
    heat_in = run.trace.plugin(LABEL, "heat_in_J")[-1]
    thermal = run.sim.plugins["thermal"]
    booked = thermal.external_heat_total.numpy()             # [W, B] J
    pending = thermal.external_heat.numpy() * 1.0e-6         # µJ -> J
    if (heat_in <= 0.0).any():
        return LawResult(False, 0.0, 1e-9, f"a pan gave no heat: {heat_in}", "check link_thermal and the heater gap")
    gained = stored[-1] - stored[0]
    err = max(np.abs(gained - heat_in).max(), np.abs(heat_in + booked[:, 1] + pending[:, 1]).max(),
              np.abs(booked[:, 0] + pending[:, 0]).max()) / heat_in.min()
    return _overflowed(run) or LawResult.within(
        err, 1.0e-9, f"per world: particles gained {gained} J, ledger {heat_in} J, pans booked {booked[:, 1]} J + "
        f"pending {pending[:, 1]} J; dummy bodies {booked[:, 0] + pending[:, 0]} J",
        "body heater heat must be whole µJ on both sides, into slot w·B + b")


def _check_body_rate(run) -> LawResult:
    """Heat in vs h_eff·s²·Σ (T_pan − T_i) over the touching particles, trapezoid in time per period."""
    therm = run.sim.plugins[LABEL]
    q = run.sim.model.particle_q.numpy()[therm._gidx.numpy()].reshape(therm._W, therm.P, 3)[0]
    touch = q[:, 2] < q[:, 2].min() + 0.5 * SPACING
    temp = run.trace.plugin(LABEL, "temperature")[:, :, touch]           # [K, W, n]
    pan = run.trace.plugin("thermal", "temperature")[:, :, 1]            # [K, W]
    gap = (pan[:, :, None] - temp).sum(axis=2)
    t = np.asarray(run.trace.t)
    h_eff = 1.0 / (1.0 / PAN_H + SPACING / (2.0 * material("water").k_liquid))  # with the half cell
    want = h_eff * SPACING ** 2 * (0.5 * (gap[1:] + gap[:-1]) * np.diff(t)[:, None]).sum(axis=0)  # footprint s²
    got = run.trace.plugin(LABEL, "heat_in_J")[-1]
    err = np.abs(got / want - 1.0)
    return LawResult.within(err.max(), 0.01, f"heat in / h_eff·s²·∫ΣΔT dt per world {got / want} at start ΔT "
                            f"{[round(PAN_T - w, 2) for w in PAN_WATER]} K over {int(touch.sum())} touching particles",
                            "carry each particle's sub-µJ remainder (body_residual) in body_exchange")


def _kill_half_bottom(sim) -> None:
    """Water at T_b on the plate, with every other bottom-layer particle already evaporated (boiled = m L_v)."""
    therm = sim.plugins[LABEL]
    q = sim.model.particle_q.numpy()[therm._gidx.numpy()].reshape(therm._W, therm.P, 3)
    bottom = np.flatnonzero(q[0, :, 2] < q[0, :, 2].min() + 0.5 * SPACING)
    dead = bottom[::2]
    boiled = therm.boiled.numpy()
    boiled[:, dead] = therm._mass.reshape(therm._W, therm.P)[:, dead] * material("water").latent_vapor
    therm.boiled.assign(boiled)


def _check_dead(run) -> LawResult:
    t = np.asarray(run.trace.t)
    want = ETA5 * P5 * (t[-1] - t[0])
    heated = run.trace.plugin(LABEL, "heated")[-1, 0]
    vapor = run.trace.plugin(LABEL, "vapor")[0, 0]
    idle = float(run.trace.plugin(LABEL, "heater_idle_J")[-1, 0].sum())
    dead = vapor >= 1.0
    into_dead = float(np.abs(heated[dead]).sum())
    err = max(abs(heated.sum() - want), idle, into_dead) / want
    return LawResult.within(err, 1.0e-9, f"heat in vs η·P·t = {want:.4g} J with {int(dead.sum())} evaporated bottom "
                            f"particles; idle {idle:.2e} J, into evaporated {into_dead:.2e} J",
                            "heater_area and rate must share one takes-heat predicate")


BUSY = {"materials": ("water",), "initial": (330.0,), "support": SUPPORT, "heaters": ("plate",),
        "heater_modes": ("power",), "heater_values": (1.0,), "heater_gap": 2.0e-4, "convection": 10.0,
        "power_density": (6.0e6,)}


def busy_scene() -> TestScene:
    """The harness repeat checks' scene (module doc): 3 worlds of a 12 × 4 × 4 water bar on a plate, 8 spares."""
    scene = bar_scene(12, 4, 4, periods=25, worlds=3, rho=997.05, plate=True, spares=8)

    def init(sim) -> None:
        sim.model.particle_max_radius = 0.0  # XPBD skips particle-particle contacts ...
        sim.model.particle_grid = None  # ... and builds no particle grid (its cell width would be 0)
        host = getattr(sim, "plugins", None)
        therm = host.plugins.get(LABEL) if host is not None else None
        if therm is None:  # the harness's baseline run (speed) has no plug-in
            return
        start = 300.0 + 60.0 * np.random.default_rng(11).random(therm.P)
        therm.set_temperature(np.broadcast_to(start, (therm._W, therm.P)))
    scene.init = init
    return scene


def particle_laws() -> list[Law]:
    conductor = {"materials": (CONDUCTOR,), "support": SUPPORT}
    water = {"materials": ("water",), "support": SUPPORT}
    return [
        Law("diffusion_mode", _check_mode, scene=bar_scene(NX1, 11, 11, periods=50, worlds=1, init=_set(_cos_mode)),
            config=conductor, doc="H-1: insulated-bar cosine mode decays at α(π/L)², 2 % (kh = 0.09): predicted "
                                  "second-layer excess +1.1 % at 11 across (+0.55 % per free face pair), 1.8x margin"),
        Law("contact_erf", _check_erf, scene=bar_scene(NX2, periods=50, worlds=1, init=_set(_step)),
            config=conductor, doc="H-2: two touching half-bars follow the erf solution, 1 % of ΔT at 1 s (5x the "
                                  "measured 0.19 %)"),
        Law("energy_closed", _check_closed, scene=bar_scene(12, 12, 12, periods=25, worlds=1, rho=997.05,
                                                            init=_set(_random)),
            config=water, doc="H-5a: Σ m H constant without sources, 1e-9 of the heat exchanged"),
        Law("energy_heater", _check_heater, scene=bar_scene(12, 6, 6, periods=25, worlds=1, rho=997.05, plate=True),
            config={**water, "initial": (293.15,), "heaters": ("plate",), "heater_modes": ("temperature",),
                    "heater_values": (353.15,), "heater_gap": 2.0e-4},
            doc="H-5b: Σ m ΔH equals the heater ledger, 1e-6 relative"),
        Law("power_heater", _check_power, scene=bar_scene(12, 4, 4, periods=5, worlds=1, rho=997.05, plate=True,
                                                          spares=SPARES5),
            config={**water, "initial": (293.15,), "heaters": ("plate",), "heater_modes": ("power",),
                    "heater_values": (P5,), "heater_efficiency": ETA5, "heater_gap": 2.0e-4},
            doc="H-5c: a power heater puts in η·P·t, none of it into spares parked in the plate; 1e-9 relative"),
        Law("power_dead", _check_dead, scene=bar_scene(12, 4, 4, periods=5, worlds=1, rho=997.05, plate=True,
                                                       init=_kill_half_bottom),
            config={**water, "initial": (373.15,), "heaters": ("plate",), "heater_modes": ("power",),
                    "heater_values": (P5,), "heater_efficiency": ETA5, "heater_gap": 2.0e-4},
            doc="H-5e: a power plate under half-evaporated bottom particles still puts in η·P·t; 1e-9 relative"),
        Law("body_heater", _check_body, scene=_pan_scene(),
            config={**water, "initial": (293.15,), "heaters": ("plate",), "heater_modes": ("body",),
                    "heater_values": (0.0,), "heater_gap": 2.0e-4},
            doc="H-5d: heat from a Thermal body (2 worlds × 2 bodies): particle gain = −(pan's booked + pending "
                "heat) in each world, the dummy body untouched; whole µJ; 1e-9"),
        Law("body_rate", _check_body_rate, scene=_pan_scene(),
            config={**water, "initial": (293.15,), "heaters": ("plate",), "heater_modes": ("body",),
                    "heater_values": (0.0,), "heater_gap": 2.0e-4},
            doc="H-5f: a body heater's rate is h_eff·s²·ΔT at ΔT = 66.85 K and 3 K (in the old 4.1 K dead band); 1 %"),
        Law("null", _check_null, scene=bar_scene(10, 5, 5, periods=10, worlds=1, rho=997.05),
            config={**water, "initial": (330.0,)}, doc="H-7: uniform T without sources stays put, 1e-9 K"),
    ]
