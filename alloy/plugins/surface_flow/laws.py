# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws of the surface-flow film core, against closed forms (design note §6).

- ``drainage`` (F-4): a uniform film on a vertical plate drains as ``h = sqrt(ν·x/(g·t))`` below the top edge
  (Jeffreys 1930), within 3 %.
- ``falling_film`` (F-5): a steady supply ``Q'`` per width down a 30° incline gives ``h_N = (3ν·Q'/(g·sin θ))^(1/3)``
  (Nusselt 1916), within 2 %; water at Re = 10, so the inertial terms are on.
- ``spreading`` (F-2): a viscous gravity current started on its similarity profile reaches
  ``R(t) = (16·g·(4V/(3π))³·t/(9ν))^(1/8)`` (Huppert 1982, ``0.894·(g·V³·t/(3ν))^(1/8)``), within 3 %.
- ``bingham_slump`` (F-7): a Bingham cylinder slumps to rest at ``R = (225·ρ·g·V²/(128·π²·τ_y))^(1/5)``
  (Roussel & Coussot 2005), within 3 %.
- ``dam_break`` (F-9): water released on a dry bed spreads as Ritter's (1892) rarefaction
  ``h = (2·c0 - x/t)²/(9g)``, within 3 % of h0 for ``-c0 < x/t < 1.5·c0`` (the thin tip, Ritter's front at
  ``2·c0``, lags in a first-order scheme and is reported).
- ``levelling`` (F-6): a ripple of wavelength λ decays at ``h³/(3μ)·(σ·k⁴ + ρ·g·k²)`` (Orchard 1962), within 5 %.
- ``contact_angle``: a ridge with no gravity, started narrower than its equilibrium, spreads to a parabola whose
  edge slope is ``tan θ`` (the contact-line condition), within 8 %.
- ``runup``: a dam break up a slope of sin α = 0.2 is set back ``½·g·sin α·t²`` from the flat one (Shen & Meyer
  1963's ballistic tip, with the flat run's own tip ratio), within 5 % (WP2 review F1).
- ``drop_2d``: a drop on a 2D grid with no gravity spreads to the cap of angle θ: wetted radius along the axes and
  the diagonals, and its height, within 6 % (WP2 review F2).
- ``ledger`` (F-11): film + spilled + evaporated + drained - supplied stays constant to float64 rounding.

Each scene is a static box ``plate`` (its top face is the plane chart), or a static revolution chart, plus a free
dummy body (the harness needs one). Gravity is tilted instead of the plate where a law needs a slope.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.surface_flow.charts import PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.film import FilmMaterial

G = 9.81
LABEL = "surface_flow"
OIL = FilmMaterial("oil_1000cSt", density=970.0, viscosity=0.97, surface_tension=0.0)   # Jeffreys: no capillarity
VISCOUS = FilmMaterial("viscous_100Pas", density=1000.0, viscosity=100.0, surface_tension=0.0)
PASTE = FilmMaterial("bingham_paste", density=2400.0, viscosity=20.0, yield_stress=100.0, surface_tension=0.0)
INVISCID = FilmMaterial("water_inviscid", viscosity=1.0e-9, surface_tension=0.0)
PAINT = FilmMaterial("paint_newtonian", density=1295.0, viscosity=0.18, surface_tension=0.032)
WETTING = FilmMaterial("water_20deg", surface_tension=0.0728, contact_angle=20.0)


def plate_scene(length: float, width: float):
    """A static box ``plate`` (top face at z = 0, ``length`` along x) and a free dummy body (the harness needs one)."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * length, hy=0.5 * width, hz=0.01,
                        label="plate")
        dummy = b.add_body(xform=body_xform(0.0, 0.0, 1.0), label="dummy")      # free; nothing touches it
        b.add_shape_sphere(dummy, radius=0.01)
    return build


def _scene(build: Any, init: Any, *, gravity: tuple[float, float, float] = (0.0, 0.0, -G), periods: int,
           control_dt: float = 0.02, dt: float = 2.0e-3, worlds: int = 3) -> TestScene:
    return TestScene(build=build, init=init, ground=False, gravity=gravity, periods=periods, control_dt=control_dt,
                     dt=dt, worlds=worlds)


def _flow(sim: Any) -> Any:
    return sim.plugins[LABEL]


def _x(sim: Any) -> np.ndarray:
    return _flow(sim).cell_centres(0)[:, 0, 0].astype(np.float64)


# --- scenes ------------------------------------------------------------------------------------------------------

DRAIN = {"charts": (PlaneChart("plate", cells=(200, 1), edges=("open", "wall", "wall", "wall")),), "material": OIL,
         "max_depth": 1.5e-3}
DRAIN_H0, DRAIN_T = 1.0e-3, 20.0

NUSSELT_Q, NUSSELT_TH = 1.0e-5, math.radians(30.0)
NUSSELT = {"charts": (PlaneChart("plate", cells=(100, 1), edges=("open", "wall", "wall", "wall")),),
           "material": FilmMaterial("water_flat", surface_tension=0.0), "max_depth": 1.0e-3}

HUPPERT_V, HUPPERT_R0 = 1.0e-5, 0.03
SPREAD = {"charts": (RevolutionChart(profile=((1.0e-3, 0.0), (0.06, 0.0)), n_theta=4, ds=1.0e-3),),
          "material": VISCOUS, "inertia": False, "max_depth": 6.0e-3}

SLUMP_V, SLUMP_R0 = 1.0e-5, 0.01
SLUMP = {"charts": (RevolutionChart(profile=((1.0e-3, 0.0), (0.04, 0.0)), n_theta=4, ds=1.0e-3),),
         "material": PASTE, "max_depth": 0.035}

RITTER_H0, RITTER_T = 0.01, 0.3
RITTER = {"charts": (PlaneChart("plate", cells=(600, 1)),), "material": INVISCID, "max_depth": RITTER_H0}

LEVEL_H0, LEVEL_A, LEVEL_L = 1.0e-4, 5.0e-6, 4.0e-3
LEVEL = {"charts": (PlaneChart("plate", cells=(64, 1)),), "material": PAINT, "inertia": False,
         "max_depth": 1.1e-4}

RIDGE_W = 2.0e-3
RIDGE = {"charts": (PlaneChart("plate", cells=(64, 1)),), "material": WETTING, "max_depth": 4.0e-4}


RUNUP_SIN, RUNUP_T = 0.2, 0.3
RUNUP = {"charts": (PlaneChart("plate", cells=(600, 1)), PlaneChart("slope", cells=(600, 1))), "material": INVISCID,
         "max_depth": RITTER_H0}

DROP_R, DROP_N, DROP_L = 2.4e-3, 64, 8.0e-3
DROP = {"charts": (PlaneChart("plate", cells=(DROP_N, DROP_N)),), "material": WETTING, "max_depth": 4.0e-4}


def runup_scene():
    """The flat plate and a second plate tilted up by asin(RUNUP_SIN) along +x (a box turned about y), side by side."""
    angle = math.asin(RUNUP_SIN)

    def build(b: newton.ModelBuilder) -> None:
        plate_scene(0.6, 0.01)(b)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -angle)          # local x climbs
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.5, 0.0), rot), hx=0.3, hy=0.005, hz=0.01,
                        label="slope")
    return build


def _runup_init(sim: Any) -> None:
    flow = _flow(sim)
    for k in (0, 1):
        x = flow.cell_centres(k)[:, 0, 0]
        flow.set_depth(np.where(x < 0.0, RITTER_H0, 0.0)[:, None], chart=k)


def _drop_init(sim: Any) -> None:
    flow = _flow(sim)
    c = flow.cell_centres(0)
    r = np.hypot(c[..., 0], c[..., 1])
    h = np.clip((0.8 * DROP_R) ** 2 - r * r, 0.0, None)                        # starts narrower: it spreads to θ
    area = flow.tables.area_np[flow.tables.cells_of(0)].reshape(r.shape)
    flow.set_depth(h * _drop_volume() / float((h * area).sum()), chart=0)


def _drop_volume() -> float:
    return math.pi * math.tan(math.radians(WETTING.contact_angle)) * DROP_R**3 / 4.0     # small-slope cap


def _drain_init(sim: Any) -> None:
    _flow(sim).set_depth(DRAIN_H0)


def _nusselt_init(sim: Any) -> None:
    flow = _flow(sim)
    rate = np.zeros(flow.tables.shapes[0])
    rate[-1, :] = NUSSELT_Q / (0.2 / 100)                       # Q' enters through the top row of cells
    flow.set_source(rate, chart=0)


def huppert_radius(t: float) -> float:
    return (16.0 * G * (4.0 * HUPPERT_V / (3.0 * math.pi)) ** 3 * t / (9.0 * VISCOUS.nu)) ** 0.125


def huppert_t0() -> float:
    return HUPPERT_R0**8 * 9.0 * VISCOUS.nu / (16.0 * G * (4.0 * HUPPERT_V / (3.0 * math.pi)) ** 3)


def _huppert_init(sim: Any) -> None:
    flow = _flow(sim)
    r = np.hypot(*flow.cell_centres(0)[..., :2].transpose(2, 0, 1))
    t0, rn = huppert_t0(), HUPPERT_R0
    a = (9.0 * VISCOUS.nu * rn * rn / (16.0 * G * t0)) ** (1.0 / 3.0)
    h = a * np.cbrt(np.clip(1.0 - (r / rn) ** 2, 0.0, None))
    area = flow.tables.area_np[flow.tables.cells_of(0)].reshape(r.shape)
    flow.set_depth(h * HUPPERT_V / float((h * area).sum()), chart=0)       # exact volume on the grid


def _slump_init(sim: Any) -> None:
    flow = _flow(sim)
    r = np.hypot(*flow.cell_centres(0)[..., :2].transpose(2, 0, 1))
    area = flow.tables.area_np[flow.tables.cells_of(0)].reshape(r.shape)
    inside = (r < SLUMP_R0).astype(np.float64)
    flow.set_depth(inside * SLUMP_V / float((inside * area).sum()), chart=0)


def _ritter_init(sim: Any) -> None:
    flow = _flow(sim)
    flow.set_depth(np.where(_x(sim) < 0.0, RITTER_H0, 0.0)[:, None], chart=0)


def _level_init(sim: Any) -> None:
    x = _x(sim)                                                 # cos(2πx/λ): zero slope at the walls (x = ±λ)
    _flow(sim).set_depth((LEVEL_H0 + LEVEL_A * np.cos(2.0 * math.pi * x / LEVEL_L))[:, None], chart=0)


def _ridge_init(sim: Any) -> None:
    x = _x(sim)
    theta = math.radians(WETTING.contact_angle)
    w0 = 0.8 * RIDGE_W                                          # start narrower: the edges advance to θ
    area = (2.0 / 3.0) * math.tan(theta) * RIDGE_W**2           # the equilibrium ridge's cross-section
    h = np.clip(w0**2 - x**2, 0.0, None)
    h *= area / (h.sum() * (x[1] - x[0]))
    _flow(sim).set_depth(h[:, None], chart=0)


def harness_scene(config: Any) -> TestScene:
    """A small water film with a bump on a gently tilted plate: everything moves (for the harness checks). A config
    with a supply brings its own test scene (DC5)."""
    def init(sim: Any) -> None:
        if LABEL not in sim.plugins.plugins:                    # the speed check's run without the plug-in
            return
        flow = _flow(sim)
        x, y = flow.cell_centres(0)[..., 0], flow.cell_centres(0)[..., 1]
        flow.set_depth(2.0e-4 + 8.0e-4 * np.exp(-(x * x + y * y) / 4.0e-4), chart=0)
    return _scene(plate_scene(0.2, 0.2), init, gravity=(-1.0, 0.0, -G), periods=10)


NO_SUPPLY = {"nozzles": (), "drains": (), "network": None}        # a law's scene has its own sources (DC5 note)


def surface_flow_laws(config: Any) -> list[Law]:
    """The film's laws. Each runs a copy of the plug-in with its own charts and material and no supply: a config
    with nozzles or drains would put them into scenes that lack their shapes."""
    laws = _film_laws()
    for law in laws:
        law.config = {**NO_SUPPLY, **dict(law.config or {})}
    return laws


def _film_laws() -> list[Law]:
    sx = math.sin(NUSSELT_TH)
    return [
        Law("drainage", check_drainage, _scene(plate_scene(0.2, 0.01), _drain_init, gravity=(-G, 0.0, 0.0),
                                               periods=int(DRAIN_T / 0.2), control_dt=0.2, dt=0.2),
            "Jeffreys: h = sqrt(ν x / (g t)) on a draining vertical plate", config=DRAIN),
        Law("falling_film", check_falling_film,
            _scene(plate_scene(0.2, 0.01), _nusselt_init, gravity=(-G * sx, 0.0, -G * math.cos(NUSSELT_TH)),
                   periods=150),
            "Nusselt: h_N = (3 ν Q' / (g sin θ))^(1/3), water at Re 10 on 30°", config=NUSSELT),
        Law("spreading", check_spreading, _scene(plate_scene(0.02, 0.02), _huppert_init,
                                                 periods=int(7.0 * huppert_t0() / 0.5), control_dt=0.5, dt=0.5),
            "Huppert: R ∝ t^(1/8), a viscous gravity current from its similarity profile", config=SPREAD),
        Law("bingham_slump", check_slump, _scene(plate_scene(0.02, 0.02), _slump_init, periods=400),
            "Roussel & Coussot: a Bingham slump stops at R = (225 ρ g V² / (128 π² τ_y))^(1/5)", config=SLUMP),
        Law("dam_break", check_dam_break, _scene(plate_scene(0.6, 0.01), _ritter_init, periods=int(RITTER_T / 0.02),
                                                 dt=1.0e-3),
            "Ritter: the dam-break rarefaction on a dry bed", config=RITTER),
        Law("levelling", check_levelling, _scene(plate_scene(2.0 * LEVEL_L, 0.002), _level_init, periods=100,
                                                 dt=1.0e-3),
            "Orchard: a ripple decays at h³/(3μ)·(σk⁴ + ρgk²)", config=LEVEL),
        contact_angle_law(),
        Law("runup", check_runup, _scene(runup_scene(), _runup_init, periods=int(RUNUP_T / 0.02), dt=1.0e-3),
            "Shen & Meyer: an incline takes ½·g·sin α·t² off the dam-break front", config=RUNUP),
        Law("drop_2d", check_drop, _scene(plate_scene(DROP_L, DROP_L), _drop_init, gravity=(0.0, 0.0, -1.0e-3),
                                          periods=15, dt=2.5e-4, worlds=1),
            "a 2D drop spreads to the cap of angle θ, round on the grid", config=DROP),
        Law("ledger", check_ledger, _ledger_scene(), "film + spilled + evaporated + drained - supplied is constant",
            config=LEDGER),
    ]


def contact_angle_law() -> Law:
    """A ridge started at 0.8 of its equilibrium width spreads until its edges reach θ (16 cells per half-width;
    at 4-8 cells the apparent angle is 8-22 % low: the contact line sits on the grid)."""
    return Law("contact_angle", check_contact_angle, _scene(plate_scene(4.0 * RIDGE_W, 0.002), _ridge_init,
                                                             gravity=(0.0, 0.0, -1.0e-3), periods=25, dt=2.5e-4),
               "a ridge spreads to the contact angle θ", config=RIDGE)


LEDGER = {"charts": (PlaneChart("plate", cells=(40, 20), edges=("open", "wall", "open", "wall")),),
          "material": FilmMaterial("water_evap", surface_tension=0.0728, evaporation=1.0e-6), "max_depth": 2.0e-3}


def _ledger_scene() -> TestScene:
    def init(sim: Any) -> None:
        flow = _flow(sim)
        flow.set_depth(5.0e-4)
        rate = np.zeros(flow.tables.shapes[0])
        rate[-5:, -5:] = 2.0e-3
        flow.set_source(rate, chart=0)
    return _scene(plate_scene(0.1, 0.05), init, gravity=(-3.0, -1.0, -G), periods=40)


# --- checks ------------------------------------------------------------------------------------------------------

def _final(run: Any) -> np.ndarray:
    """World 0's film on chart 0 after the last period, ``[nx, ny]``."""
    return run.plugin.depth(0)[0].astype(np.float64)


def check_drainage(run: Any) -> LawResult:
    t = float(run.trace.t[-1])
    x_top = 0.1 - _x(run.sim)                                   # distance below the top (wall) edge
    h = _final(run)[:, 0]
    want = np.sqrt(OIL.nu * x_top / (G * t))
    sel = (want < 0.8 * DRAIN_H0) & (x_top > 0.04)         # 40 cells from the singular top corner
    err = float(np.max(np.abs(h[sel] / want[sel] - 1.0)))
    return LawResult.within(err, 0.03, f"max |h/h_J - 1| = {err:.4f} over {int(sel.sum())} cells at t = {t:.1f} s",
                            "flux h³·g/(3ν) along the plate; check the gravity resolved in the chart frame")


def check_falling_film(run: Any) -> LawResult:
    h = _final(run)[30:70, 0]
    want = (3.0 * NUSSELT["material"].nu * NUSSELT_Q / (G * math.sin(NUSSELT_TH))) ** (1.0 / 3.0)
    err = float(h.mean() / want - 1.0)
    ripple = float(h.std() / want)
    return LawResult.within(max(abs(err), ripple), 0.02, f"mean h {h.mean() * 1e6:.2f} vs h_N {want * 1e6:.2f} µm "
                            f"({err:+.2%}), ripple {ripple:.2%}", "steady q = g sin θ h³/(3ν) with inertia on")


def check_spreading(run: Any) -> LawResult:
    r = np.hypot(*run.plugin.cell_centres(0)[..., :2].transpose(2, 0, 1))[:, 0]
    h = _final(run)[:, 0]
    hm = h.max()
    sel = (h > 0.05 * hm) & (h < 0.6 * hm)
    p = np.polyfit(r[sel] ** 2, h[sel] ** 3, 1)
    got = math.sqrt(-p[1] / p[0])
    t = huppert_t0() + float(run.trace.t[-1])
    want = huppert_radius(t)
    err = got / want - 1.0
    return LawResult.within(err, 0.03, f"R {got * 1e3:.3f} vs {want * 1e3:.3f} mm at t = {t:.0f} s ({err:+.2%}; "
                            f"started at {HUPPERT_R0 * 1e3:.0f} mm)", "lubrication flux h³G/(3ν), upwind mobility")


def check_slump(run: Any) -> LawResult:
    r = np.hypot(*run.plugin.cell_centres(0)[..., :2].transpose(2, 0, 1))[:, 0]
    h = _final(run)[:, 0]
    hm = h.max()
    sel = (h > 0.15 * hm) & (h < 0.85 * hm)
    p = np.polyfit(r[sel], h[sel] ** 2, 1)                      # at rest: h² = 2τ(R - r)/(ρg)
    got = -p[1] / p[0]
    want = (225.0 * PASTE.density * G * SLUMP_V**2 / (128.0 * math.pi**2 * PASTE.yield_stress)) ** 0.2
    moving = float(np.abs(np.asarray(run.trace.plugin(LABEL, "h"))[-1, 0] - np.asarray(
        run.trace.plugin(LABEL, "h"))[-2, 0]).max())
    err = got / want - 1.0
    return LawResult.within(max(abs(err), moving / hm / 1.0e-3 * 0.03), 0.03,
                            f"R {got * 1e3:.3f} vs {want * 1e3:.3f} mm ({err:+.2%}); last-period change "
                            f"{moving * 1e6:.3g} µm (bound 1e-3 of the {hm * 1e3:.1f} mm crest)",
                            "Bingham thin-layer flux; the plug must stand exactly")


def check_dam_break(run: Any) -> LawResult:
    """Ritter's rarefaction ``h = (2c0 - x/t)²/(9g)`` for ``-c0 < x/t < 1.5·c0``, as a fraction of h0. The thin tip
    beyond (Ritter's front at 2c0) lags in a first-order scheme; it is reported, not bounded."""
    x, h = _x(run.sim), _final(run)[:, 0]
    t = float(run.trace.t[-1])
    c0 = math.sqrt(G * RITTER_H0)
    want = np.clip(2.0 * c0 - x / t, 0.0, 3.0 * c0) ** 2 / (9.0 * G)
    sel = (x / t > -c0) & (x / t < 1.5 * c0)
    err = float(np.abs(h[sel] - want[sel]).max()) / RITTER_H0
    wet = np.flatnonzero(h > 1.0e-3 * RITTER_H0)
    front = (float(x[wet[-1]]) + 0.5 * (x[1] - x[0])) / t / (2.0 * c0) - 1.0
    return LawResult.within(err, 0.03, f"max |h - h_Ritter| = {err:.2%} of h0 over the rarefaction at t = {t:.2f} s "
                            f"(front speed {front:+.1%} vs 2·sqrt(g·h0): the first-order tip lags)",
                            "momentum-conservative advection (Stelling & Duinmeijer) with h̄ at the face")


def check_levelling(run: Any) -> LawResult:
    x = _x(run.sim)
    mode = np.cos(2.0 * math.pi * x / LEVEL_L)
    hs = np.asarray(run.trace.plugin(LABEL, "h"))[:, 0, :].astype(np.float64)      # [K, C]
    amp = (hs - hs.mean(axis=1, keepdims=True)) @ mode / (mode @ mode)
    t = np.asarray(run.trace.t)
    k = 2.0 * math.pi / LEVEL_L
    want = LEVEL_H0**3 / (3.0 * PAINT.viscosity) * (PAINT.surface_tension * k**4 + PAINT.density * G * k**2)
    got = -float(np.polyfit(t[5:], np.log(amp[5:]), 1)[0])
    err = got / want - 1.0
    return LawResult.within(err, 0.05, f"decay rate {got:.4f} vs {want:.4f} /s ({err:+.2%}; amplitude "
                            f"{amp[0] * 1e6:.2f} -> {amp[-1] * 1e6:.2f} µm)",
                            "capillary pressure -σ∇²h, 32 cells per λ")


def check_contact_angle(run: Any) -> LawResult:
    x, h = _x(run.sim), _final(run)[:, 0]
    hm = h.max()
    sel = h > 0.4 * hm
    p = np.polyfit(x[sel], h[sel], 2)                            # h = c0 + c1 x - c2 x² near the crest
    c2, c0 = -p[0], p[2] - p[1] ** 2 / (4.0 * p[0])
    slope = 2.0 * math.sqrt(c0 * c2)                            # edge slope of the parabola
    want = math.tan(math.radians(WETTING.contact_angle))
    err = slope / want - 1.0
    return LawResult.within(err, 0.08, f"edge slope {slope:.4f} vs tan θ_e {want:.4f} ({err:+.2%})",
                            "the contact-line condition: slope tan θ toward dry cells in the curvature")


def check_runup(run: Any) -> LawResult:
    """The incline's front against the flat plate's in the same run: Δx = ½·g·sin α·t² plus the slower c0 (Shen &
    Meyer 1963's ballistic tip), with the flat run's own tip ratio r0 (the first-order tip lag cancels)."""
    p, t = run.plugin, float(run.trace.t[-1])
    fronts = []
    for k in (0, 1):
        x, h = p.cell_centres(k)[:, 0, 0].astype(np.float64), p.depth(k)[0][:, 0].astype(np.float64)
        fronts.append(float(x[h > 1.0e-6].max()))
    cos_a = math.sqrt(1.0 - RUNUP_SIN**2)
    c0, c0a = math.sqrt(G * RITTER_H0), math.sqrt(G * cos_a * RITTER_H0)
    r0 = fronts[0] / (2.0 * c0 * t)
    want = r0 * 2.0 * t * (c0 - c0a) + 0.5 * G * RUNUP_SIN * t * t
    got = fronts[0] - fronts[1]
    err = got / want - 1.0
    return LawResult.within(err, 0.05, f"front set back {got * 1e3:.2f} vs {want * 1e3:.2f} mm by sin α = {RUNUP_SIN} "
                            f"({err:+.2%}; flat tip ratio {r0:.3f})", "a front face keeps its momentum uphill")


def check_drop(run: Any) -> LawResult:
    """Wetted radius along the axes and the diagonals against the cap's R, and the height against tan θ·R/2."""
    p = run.plugin
    c, h = p.cell_centres(0).astype(np.float64), p.depth(0)[0].astype(np.float64)
    r = np.hypot(c[..., 0], c[..., 1])
    wet = h > 2.0e-6
    ang = np.degrees(np.arctan2(np.abs(c[..., 1]), np.abs(c[..., 0])))
    dx = DROP_L / DROP_N
    r_axis = float(r[wet & ((ang < 10.0) | (ang > 80.0))].max()) + 0.5 * dx
    r_diag = float(r[wet & (np.abs(ang - 45.0) < 10.0)].max()) + 0.5 * dx * math.sqrt(2.0)
    cap = math.tan(math.radians(WETTING.contact_angle)) * DROP_R / 2.0
    errs = (r_axis / DROP_R - 1.0, r_diag / r_axis - 1.0, float(h.max()) / cap - 1.0)
    worst = max(errs, key=abs)
    shown = ", ".join(f"{e:+.2%}" for e in errs)
    return LawResult.within(worst, 0.06, f"R axis {r_axis * 1e3:.3f}, diagonal {r_diag * 1e3:.3f} vs {DROP_R * 1e3:.2f} "
                            f"mm; height {h.max() * 1e3:.4f} vs {cap * 1e3:.4f} mm ({shown})",
                            "the contact line takes tan θ·|n·d| per dry face along its normal")


def check_ledger(run: Any) -> LawResult:
    ledger = np.asarray(run.trace.plugin(LABEL, "ledger"))       # [K, W]
    scale = float(np.asarray(run.trace.plugin(LABEL, "supplied_volume")).max()) + float(ledger[0].max())
    drift = float(np.abs(ledger - ledger[0]).max()) / scale
    spilled = float(np.asarray(run.trace.plugin(LABEL, "spilled_volume"))[-1].min())
    evaporated = float(np.asarray(run.trace.plugin(LABEL, "evaporated_volume"))[-1].min())
    ok = spilled > 0.0 and evaporated > 0.0
    return LawResult.within(drift if ok else 1.0, 1.0e-12, f"relative ledger drift {drift:.3g} (spilled "
                            f"{spilled * 1e6:.3g} mL, evaporated {evaporated * 1e6:.3g} mL)",
                            "book every move as float64 per cell; open edges into spilled")


__all__ = ["harness_scene", "huppert_radius", "plate_scene", "surface_flow_laws"]
