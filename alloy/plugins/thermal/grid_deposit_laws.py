# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 2 laws: deposition (``grid_deposit``). The scenes reuse ``grid_laws``' torch body; the wire feeds GA's
spray point (12.7 m/min of 0.9 mm wire, 98 % deposited, report_welding §5).

- D-1 ``deposit_plate``: a vertical torch at 6 mm/s over an A36 plate (1 mm slices along x, 0.5 mm across), 4 kW of
  which the droplets carry ρ ε A_w w_f (H_drop − H_wire) ≈ 1.9 kW, for 2 s. Mass: nothing missed; Σ slice budgets =
  fed − deposited exactly; every slice's leftover within one cell (the fill keeps up); the steady slices' mean
  deposited cross-section ε A_w w_f / v within 2 % (single slices get 8 or 9 of the 20 ms exchanges: ±6 %). Energy:
  ledger residual <= 1e-6 of the input; beam + droplet heat = arc power x t within 1 %.
- D-2 ``deposit_corner``: the slice-1 T-joint (1 mm cells) with the torch at 45° into the root: nothing missed, the
  root cell of every slice the arc crossed is weld metal, the ledger closes (1e-6).
- D-3 ``cap_shape``: the cap model in a T-joint at 0.25 mm with the parent pre-fused 4 mm along the plate and 3 mm up
  the bracket (peak 1800 K, no arc), a parked torch feeding one slice 12 mm²: the toes read 4 and 3 mm; the bead's
  area is the slice's feed within three cells (the region holds whole cells; the rest waits in the budget); its face
  stands the circular segment's sagitta beyond the chord within a cell and a half (the segment on the 5 mm chord
  holding 12 − 6 mm² beyond the 6 mm² wedge).
- D-4 ``cap_wets``: the same joint fused only 1 mm along each member but hot (1500 K, below the solidus) to 5 mm, a
  parked torch at 4 kW feeding one slice 20 mm²: the liquid pool wets the hot metal (GA's contact temperature), so
  the toes advance past the fused 1 mm to at least the hot zone's 5 mm (the arc heats further: 6.0 / 6.25 mm), the
  liquid follows the wider cap (its face within 1.5 cells of the segment on the toes' chord: 0.141 vs 0.216 mm), and
  the books close (ledger 7e-11 <= 1e-6).
- D-5 ``deposit_lag``: D-1's plate with the metal landing where the arc was 0.5 s ago (``deposit_lag``, the arc
  pressure's pool pushed back): the books stay exact, the ledger closes (1e-6), the bead stops 3 mm (v τ) short of
  the arc's last position within the spread (3 σ + a slice), and the droplets' heat still lands under the arc.
- D-6 ``pool_levels``: D-3's joint, pre-fused the same along its 2 mm, a parked torch (no arc) feeding 12 mm² per
  slice with the wire spread along the weld (σ 0.5 mm, a Gaussian: 1.6x the mean under the torch) and the pool's
  flow on (``arc_pressure`` with the arc out: no dent). The slices' toes are equal, so at one pressure they hold
  equal areas: every slice within two cells of the pool's volume / length, every slice of the joint holding metal,
  and the ledger closed (1e-6). Without the flow the slice under the torch holds 1.6x the mean.
- D-7 ``pool_dent``: D-6's joint 6 mm long, the torch parked near its middle with an arc that carries only the
  droplets' heat (1 W more), so its pressure acts (GA's 0.0214 N over σ 1.785 mm at 241.3 A) and heats nothing.
  Each slice's area is what its toes hold at the pool's pressure less the arc's mean pressure on its toe chord, the
  arc root being the first metal on the torch's axis: within four cells of that rule evaluated here in numpy (erf
  across the chord, bisection on P), the measured dent at least 8 cells, ledger closed (1e-6). The four cells are
  not derived: a slice's face is a staircase of whole cells and the worst slice measured 3.0 (D-6's convex caps:
  under 2); the dent spans about 110 cells, so the bound is 2-6 % of a slice.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.plugins.harness.scene import Law, LawResult
from alloy.plugins.thermal.grid import graded
from alloy.plugins.thermal.grid_laws import LABEL, TORCH, _cfg, _final, _init, _scene, tee
from alloy.plugins.thermal.grid_materials import material

FEED = 12.7 / 60.0  # m/s (GA: 12.7 m/min)
POWER, SPEED, SECONDS = 4000.0, 0.006, 2.0
PLATE = _cfg(x=graded(0.0, 0.03, 1e-3), y=graded(-0.01, 0.01, 5e-4, band=(-0.005, 0.005)),
             z=graded(-0.006, 0.006, 5e-4), boxes=((0.0, -0.01, -0.006, 0.03, 0.01, 0.0),), sink=("z-",),
             torch=TORCH, torch_axis=(0.0, 0.0, -1.0), arc_sigma=3e-3)
CORNER = _cfg(**tee(1e-3, 0.03), torch=TORCH, torch_axis=(0.0, -math.sqrt(0.5), -math.sqrt(0.5)), arc_sigma=3e-3)


def _mass(run, tol_mean: float | None) -> tuple[LawResult | None, str]:
    grid = run.sim.plugins[LABEL]
    _, s = _final(run)
    budget = grid.deposit_budget.numpy()[0]
    vmax = float((grid._mass / material(grid.config.material).density).max())
    gap = s["fed_m3"] - s["missed_m3"] - s["deposited_m3"] - budget.sum()
    detail = (f"fed {s['fed_m3'] * 1e9:.2f} mm³, deposited {s['deposited_m3'] * 1e9:.2f}, missed "
              f"{s['missed_m3'] * 1e9:.2f}, leftover max {np.abs(budget).max() * 1e9:.3f} (cell {vmax * 1e9:.3f})")
    if s["missed_m3"] > 0.0 or abs(gap) > 1e-15 or np.abs(budget).max() > vmax:
        return LawResult(False, float(abs(gap)), 0.0, detail + f"; books off by {gap:.3g} m³"), detail
    if tol_mean is None:
        return None, detail
    cells = grid.fields(0)["cell"] == grid._bead.weld
    dx = np.diff(np.asarray(grid.config.x))
    xs = 0.5 * (np.asarray(grid.config.x)[1:] + np.asarray(grid.config.x)[:-1])
    yz = np.outer(np.diff(np.asarray(grid.config.y)), np.diff(np.asarray(grid.config.z)))
    area = np.array([(cells[i] * yz).sum() for i in range(len(dx))])
    x_end = 0.005 + SPEED * float(run.trace.t[-1])
    steady = (xs > 0.005 + 2e-3) & (xs < x_end - 2e-3)
    want = grid._bead.area * FEED / SPEED
    err = abs(area[steady].mean() / want - 1.0)
    detail += f"; bead area {area[steady].mean() * 1e6:.2f} mm² vs ε A_w w_f / v = {want * 1e6:.2f} ({err:.2%})"
    if err > tol_mean:
        return LawResult(False, err, tol_mean, detail), detail
    return None, detail


def _energy(run, split_tol: float | None) -> LawResult:
    grid = run.sim.plugins[LABEL]
    _, s = _final(run)
    t = float(run.trace.t[-1])
    m = material(grid.config.material)
    h_wire = float(m.enthalpy(grid.config.ambient))
    droplets = s["filler_J"] - m.density * s["deposited_m3"] * h_wire
    rel = abs(s["residual_J"]) / (s["arc_J"] + abs(s["filler_J"]))
    detail = (f"residual / input {rel:.2e}; beam {s['arc_J']:.1f} J + droplets {droplets:.1f} J vs arc x t "
              f"{POWER * t:.1f} J; {int(s['weld_cells'])} weld cells, T max {s['t_max']:.0f} K")
    if split_tol is not None and abs((s["arc_J"] + droplets) / (POWER * t) - 1.0) > split_tol:
        return LawResult(False, rel, 1e-6, detail + " (beam + droplets is not the arc's power)")
    return LawResult.within(rel, 1e-6, detail)


def _check_plate(run) -> LawResult:
    bad, detail = _mass(run, 0.02)
    if bad is not None:
        return bad
    res = _energy(run, 0.01)
    res.detail = detail + "; " + res.detail
    return res


def _check_corner(run) -> LawResult:
    bad, detail = _mass(run, None)
    if bad is not None:
        return bad
    grid = run.sim.plugins[LABEL]
    f = grid.fields(0)
    cy = 0.5 * (np.asarray(grid.config.y)[1:] + np.asarray(grid.config.y)[:-1])
    cz = 0.5 * (np.asarray(grid.config.z)[1:] + np.asarray(grid.config.z)[:-1])
    xs = 0.5 * (np.asarray(grid.config.x)[1:] + np.asarray(grid.config.x)[:-1])
    j, k = int(np.flatnonzero(cy > 0.0)[0]), int(np.flatnonzero(cz > 0.0)[0])    # the air cell at the root
    crossed = (xs > 0.005 + 1e-3) & (xs < 0.005 + SPEED * float(run.trace.t[-1]) - 1e-3)
    root = f["cell"][crossed, j, k] == grid._bead.weld
    if not root.all():
        return LawResult(False, float(root.mean()), 1.0, detail + f"; root filled in {root.mean():.0%} of the slices")
    res = _energy(run, None)
    res.detail = detail + "; " + res.detail
    return res


CAP = _cfg(x=graded(0.0, 0.002, 2.5e-4), y=graded(-0.012, 0.012, 2.5e-4, band=(-0.004, 0.008)),
           z=graded(-0.006, 0.012, 2.5e-4, band=(-0.004, 0.008)),
           boxes=((0.0, -0.012, -0.006, 0.002, 0.012, 0.0), (0.0, -0.006, 0.0, 0.002, 0.0, 0.012)), parts=(0, 1),
           torch=TORCH, torch_axis=(0.0, -math.sqrt(0.5), -math.sqrt(0.5)), convection=0.0, emissivity=0.0,
           bead_model="cap", joint_root=(0.0, 0.0), joint_rays=((1.0, 0.0), (0.0, 1.0)), deposit_spread=1e-6)
CAP_TOES, CAP_AREA, CAP_TIME = (4e-3, 3e-3), 12e-6, 0.4


def _cap_temps(grid) -> np.ndarray:
    cx, cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.x, grid.config.y,
                                                                              grid.config.z))
    _, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    hot = ((Y > 0) & (Y < CAP_TOES[0]) & (Z > -1e-3) & (Z < 0)) | ((Y > -1e-3) & (Y < 0) & (Z > 0) & (Z < CAP_TOES[1]))
    return np.where(hot, 1800.0, 300.0).reshape(-1)


def segment_angle(chord: float, area: float) -> float:
    """θ of the circular segment on ``chord`` of ``area`` (host twin of grid_deposit.cap_angle)."""
    lo, hi = 1e-9, 2 * math.pi - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chord * chord * (mid - math.sin(mid)) / (8 * math.sin(0.5 * mid) ** 2) < area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _check_cap(run) -> LawResult:
    grid = run.sim.plugins[LABEL]
    f = grid.fields(0)
    ax = int(np.argmax((f["cell"] == grid._bead.weld).sum(axis=(1, 2))))
    toes = grid.toes.numpy()[0, ax]
    dx = float(np.diff(np.asarray(grid.config.x))[ax])
    cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.y, grid.config.z))
    dy, dz = np.diff(np.asarray(grid.config.y)), np.diff(np.asarray(grid.config.z))
    weld = f["cell"][ax] == grid._bead.weld
    area = float((weld * np.outer(dy, dz)).sum())
    target = float(grid.slice_fed_m3.numpy()[0, ax]) / dx
    t1, t2 = CAP_TOES
    chord = math.hypot(t1, t2)
    th = segment_angle(chord, target - 0.5 * t1 * t2)
    sagitta = chord / (2 * math.sin(0.5 * th)) * (1 - math.cos(0.5 * th))
    Y, Z = np.meshgrid(cy, cz, indexing="ij")
    beyond = (Y * t2 + Z * t1 - t1 * t2) / chord                     # signed distance beyond the chord
    stand = float(beyond[weld].max())
    cell = 2.5e-4
    detail = (f"toes {toes[0] * 1e3:.3f}, {toes[1] * 1e3:.3f} mm (set 4, 3); area {area * 1e6:.3f} vs feed "
              f"{target * 1e6:.3f} mm²; face {stand * 1e3:.3f} mm beyond the chord vs sagitta {sagitta * 1e3:.3f} mm")
    ok = (abs(toes[0] - t1) <= cell and abs(toes[1] - t2) <= cell and abs(area - target) <= 3 * cell * cell
          and abs(stand - sagitta) <= 1.5 * cell)       # area: the region holds whole cells (the rest waits)
    return LawResult(ok, abs(stand - sagitta), 1.5 * cell, detail)


WET_TOES, WET_AREA, WET_TIME = (5e-3, 5e-3), 20e-6, 0.3


def _wet_temps(grid) -> np.ndarray:
    cx, cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.x, grid.config.y,
                                                                              grid.config.z))
    _, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    plate = (Y > 0) & (Z > -1e-3) & (Z < 0)
    bracket = (Y > -1e-3) & (Y < 0) & (Z > 0)
    fused = (plate & (Y < 1e-3)) | (bracket & (Z < 1e-3))
    hot = (plate & (Y < WET_TOES[0])) | (bracket & (Z < WET_TOES[1]))
    return np.where(fused, 1800.0, np.where(hot, 1500.0, 300.0)).reshape(-1)


def _check_wets(run) -> LawResult:
    grid = run.sim.plugins[LABEL]
    f = grid.fields(0)
    ax = int(np.argmax((f["cell"] == grid._bead.weld).sum(axis=(1, 2))))
    toes = grid.toes.numpy()[0, ax]
    dx = float(np.diff(np.asarray(grid.config.x))[ax])
    cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.y, grid.config.z))
    weld = f["cell"][ax] == grid._bead.weld
    target = float(grid.slice_fed_m3.numpy()[0, ax]) / dx
    t1, t2 = (float(v) for v in toes)
    chord = math.hypot(t1, t2)
    th = segment_angle(chord, target - 0.5 * t1 * t2)
    sagitta = chord / (2 * math.sin(0.5 * th)) * (1 - math.cos(0.5 * th))
    Y, Z = np.meshgrid(cy, cz, indexing="ij")
    stand = float(((Y * t2 + Z * t1 - t1 * t2) / chord)[weld].max())
    _, s = _final(run)
    rel = abs(s["residual_J"]) / (s["arc_J"] + abs(s["filler_J"]))
    cell = 2.5e-4
    far = float(np.hypot(Y, Z)[(f["cell"] == grid._bead.weld).any(axis=0)].max())   # every slice's weld cells
    detail = (f"toes {t1 * 1e3:.3f}, {t2 * 1e3:.3f} mm (fused 1, hot to 5); face {stand * 1e3:.3f} mm beyond the chord "
              f"vs sagitta {sagitta * 1e3:.3f} mm; farthest weld cell {far * 1e3:.3f} mm from the root (bound: the "
              f"toe-to-toe width {chord * 1e3:.3f} mm); ledger {rel:.1e}")
    ok = (min(t1, t2) >= WET_TOES[0] - cell and abs(stand - sagitta) <= 1.5 * cell and rel <= 1e-6
          and far <= chord + cell)
    return LawResult(ok, abs(stand - sagitta), 1.5 * cell, detail)


LAG = 0.5   # s


def _check_lag(run) -> LawResult:
    bad, detail = _mass(run, None)
    if bad is not None:
        return bad
    grid = run.sim.plugins[LABEL]
    weld = (grid.fields(0)["cell"] == grid._bead.weld).any(axis=(1, 2))
    ex = np.asarray(grid.config.x)
    x_arc = 0.005 + SPEED * float(run.trace.t[-1])
    last = float(ex[1:][weld].max()) if weld.any() else 0.0
    want = x_arc - SPEED * LAG
    slack = 3.0 * grid.config.deposit_spread + float(np.diff(ex).max())
    res = _energy(run, None)
    ok = res.passed and abs(last - want) <= slack
    return LawResult(ok, abs(last - want), slack, f"{detail}; bead ends at {last * 1e3:.2f} mm, the arc at "
                     f"{x_arc * 1e3:.2f} mm, v τ behind it {want * 1e3:.2f} mm; {res.detail}")


POOL = {**CAP, "deposit_spread": 5e-4, "arc_pressure": (0.02, 1.8e-3)}


def _check_pool(run) -> LawResult:
    grid = run.sim.plugins[LABEL]
    f, s = _final(run)
    dx = np.diff(np.asarray(grid.config.x))
    yz = np.outer(np.diff(np.asarray(grid.config.y)), np.diff(np.asarray(grid.config.z)))
    area = ((f["cell"] == grid._bead.weld) * yz[None]).sum(axis=(1, 2))
    level = float((area * dx).sum() / dx.sum())
    off = float(np.abs(area - level).max())
    cell = 2.5e-4 ** 2
    rel = abs(s["residual_J"]) / max(abs(s["filler_J"]), 1e-30)
    detail = (f"slice areas {area.min() * 1e6:.3f} to {area.max() * 1e6:.3f} mm² about the pool's level "
              f"{level * 1e6:.3f} (fed {s['fed_m3'] * 1e9:.2f} mm³ over {dx.sum() * 1e3:.2f} mm; the wire alone: "
              f"1.6x the level under the torch); ledger {rel:.1e}")
    ok = bool((area > 0).all()) and off <= 2 * cell and rel <= 1e-6 and s["missed_m3"] == 0.0
    return LawResult(ok, off, 2 * cell, detail)


DENT_LEN, DENT_AREA = 6e-3, 7e-6
DENT_FORCE = (-0.04017 + 2.5533e-4 * 241.3, 1.4875e-3 + 1.233e-6 * 241.3)     # GA's arc force and σ at 241.3 A
DENT = {**POOL, "x": graded(0.0, DENT_LEN, 5e-4), "deposit_spread": 1e-3, "arc_pressure": DENT_FORCE,
        "flow_moves": 512, "boxes": tuple((b[0], *b[1:3], DENT_LEN, *b[4:]) for b in CAP["boxes"])}
DENT_FEED = DENT_AREA * DENT_LEN / (0.98 * math.pi * 0.45e-3 ** 2 * CAP_TIME)
DENT_START = (0.5 * DENT_LEN + 2.5e-4, 0.01, 0.01)        # over a slice's centre: the arc root's slice is not a tie


def _dent_init(sim) -> None:
    """D-6's start, and an arc of the droplets' power plus 1 W: its pressure is on, its beam carries 1 W."""
    _init(feed=DENT_FEED, temps=_cap_temps)(sim)
    if LABEL in sim.plugins.plugins:
        grid = sim.plugins[LABEL]
        grid.set_input("arc_power", np.full(sim.model.world_count, grid._drop_power * DENT_FEED + 1.0, np.float32))


def _holds(t, kappa: float) -> float:
    """``grid_flow.holds`` in numpy."""
    wedge, c = 0.5 * t[0] * t[1], math.hypot(t[0], t[1])
    k = min(abs(kappa), 2.0 / c)
    if k < 1e-9:
        return wedge
    th = 2.0 * math.asin(min(1.0, 0.5 * c * k))
    seg = 0.5 * (th - math.sin(th)) / (k * k)
    return max(0.0, wedge - seg) if kappa < 0.0 else wedge + seg


def _arc_mean(t, x: float, at: np.ndarray) -> float:
    """GA's arc pressure from the arc root ``at`` averaged over the chord between toes t (on y, on z), exactly."""
    force, s = DENT_FORCE
    p1, d = np.array([t[0], 0.0]), np.array([-t[0], t[1]])
    c = float(np.linalg.norm(d))
    q = at[1:] - p1
    a = float(q @ d) / c
    perp2 = float(q @ q) - a * a
    k = 1.0 / (math.sqrt(2.0) * s)
    mean = s * math.sqrt(0.5 * math.pi) * (math.erf(a * k) + math.erf((c - a) * k)) / c
    return force / (2.0 * math.pi * s * s) * math.exp(-0.5 * ((x - at[0]) ** 2 + perp2) / (s * s)) * mean


def _dent_formula(grid, cell: np.ndarray, area: np.ndarray):
    """Each slice's area by the pool's rule, here in numpy: (areas, P, the arc's mean pressures, the arc root)."""
    ex, ey, ez = (np.asarray(e) for e in (grid.config.x, grid.config.y, grid.config.z))
    dx, xc = np.diff(ex), 0.5 * (ex[1:] + ex[:-1])
    toes, gamma = grid.toes.numpy()[0], grid.config.surface_tension
    at = None
    for k in range(400):                                    # the arc root: the first metal on the torch's axis
        p = np.array(DENT_START) + 5e-5 * k * np.array(grid.config.torch_axis)
        i, j, m = (int(np.searchsorted(e, v)) - 1 for e, v in zip((ex, ey, ez), p, strict=True))
        if cell[i, j, m] != 0:
            at = np.array([xc[i], 0.5 * (ey[j] + ey[j + 1]), 0.5 * (ez[m] + ez[m + 1])])
            break
    arc = np.array([_arc_mean(toes[i], xc[i], at) for i in range(len(xc))])
    lo, hi = -1e6, 1e6
    for _ in range(80):                                     # the pool's pressure: it holds what it has
        mid = 0.5 * (lo + hi)
        vol = sum(_holds(toes[i], (mid - arc[i]) / gamma) * dx[i] for i in range(len(xc)))
        lo, hi = (mid, hi) if vol < float((area * dx).sum()) else (lo, mid)
    return np.array([_holds(toes[i], (hi - arc[i]) / gamma) for i in range(len(xc))]), hi, arc, at


def _check_dent(run) -> LawResult:
    grid = run.sim.plugins[LABEL]
    f, s = _final(run)
    yz = np.outer(np.diff(np.asarray(grid.config.y)), np.diff(np.asarray(grid.config.z)))
    area = ((f["cell"] == grid._bead.weld) * yz[None]).sum(axis=(1, 2))
    want, p, arc, at = _dent_formula(grid, f["cell"], area)
    cell = 2.5e-4 ** 2
    off, dent = float(np.abs(area - want).max()), float(area.max() - area.min())
    rel = abs(s["residual_J"]) / max(abs(s["filler_J"]), 1e-30)
    detail = (f"slice areas {area.min() * 1e6:.3f} (under the arc) to {area.max() * 1e6:.3f} mm², the formula "
              f"{want.min() * 1e6:.3f} to {want.max() * 1e6:.3f} (P {p:.0f} Pa, the arc's mean pressure "
              f"{arc.max():.0f} Pa under it, {arc.min():.0f} at the ends; root {at[1] * 1e3:.2f}, {at[2] * 1e3:.2f} mm"
              f"); ledger {rel:.1e}")
    ok = off <= 4 * cell and dent >= 8 * cell and rel <= 1e-6 and s["missed_m3"] == 0.0
    return LawResult(ok, off, 4 * cell, detail)


def deposit_laws() -> list[Law]:
    periods = round(SECONDS / 0.02)
    return [
        Law("deposit_plate", _check_plate, scene=_scene(periods, _init(power=POWER, sigma=3e-3,
            speed=(SPEED, 0.0, 0.0), feed=FEED), start=(0.005, 0.0, 0.01)), config=PLATE,
            doc="D-1: deposited mass and droplet heat on a plate: books exact, bead area 2 %, ledger 1e-6"),
        Law("deposit_corner", _check_corner, scene=_scene(periods, _init(power=POWER, sigma=3e-3,
            speed=(SPEED, 0.0, 0.0), feed=FEED), start=(0.005, 0.01, 0.01)), config=CORNER,
            doc="D-2: deposition into a T-joint's root at 45°: nothing missed, root filled, ledger 1e-6"),
        Law("cap_shape", _check_cap, scene=_scene(round(CAP_TIME / 0.02), _init(feed=CAP_AREA * 2.5e-4 / (
            0.98 * math.pi * 0.45e-3 ** 2 * CAP_TIME), temps=_cap_temps), start=(0.001, 0.01, 0.01)), config=CAP,
            doc="D-3: the cap model: toes from fusion, area = feed, the face's sagitta within a cell and a half"),
        Law("cap_wets", _check_wets, scene=_scene(round(WET_TIME / 0.02), _init(power=POWER, sigma=3e-3,
            feed=WET_AREA * 2.5e-4 / (0.98 * math.pi * 0.45e-3 ** 2 * WET_TIME), temps=_wet_temps),
            start=(0.001, 0.01, 0.01)), config=CAP,
            doc="D-4: the pool wets hot metal past the fused toes, the liquid follows the wider cap, books close"),
        Law("deposit_lag", _check_lag, scene=_scene(periods, _init(power=POWER, sigma=3e-3, speed=(SPEED, 0.0, 0.0),
            feed=FEED), start=(0.005, 0.0, 0.01)), config={**PLATE, "deposit_lag": LAG},
            doc="D-5: metal lands where the arc was lag seconds ago: books exact, bead ends v τ behind the arc"),
        Law("pool_levels", _check_pool, scene=_scene(round(CAP_TIME / 0.02), _init(feed=CAP_AREA * 2e-3 / (
            0.98 * math.pi * 0.45e-3 ** 2 * CAP_TIME), temps=_cap_temps), start=(0.001, 0.01, 0.01)), config=POOL,
            doc="D-6: the liquid pool flows to one pressure: equal toes hold equal areas along the weld"),
        Law("pool_dent", _check_dent, scene=_scene(round(CAP_TIME / 0.02), _dent_init, start=DENT_START), config=DENT,
            doc="D-7: the arc dents the pool: each slice holds what its toes hold at P less the arc's mean pressure"),
    ]


__all__ = ["deposit_laws"]
