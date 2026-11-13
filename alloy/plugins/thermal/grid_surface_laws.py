# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11 laws: the surface model (``bead_model="surface"``, ``grid_bead_surface``) on the grid, against
the numpy reference (``grid_surface_arc``) evaluated on the run's own toes and metal.

- D-8 ``surface_weightless``: D-3's joint and feed (toes pre-fused 4 / 3 mm, 12 mm² into one slice, no arc) with the
  surface model and no gravity: the surface is the circle, so the toes read 4 / 3 mm, the face stands the circular
  segment's sagitta beyond the chord within a cell and a half (D-3's bound), and the cells differ from those under
  the reference surface only along the face (at most one per face cell). Guards the port against itself.
- D-9 ``surface_puddle``: a flat plate at 1700 K that conducts no heat (``law_insulated``: it neither fuses nor
  freezes the liquid's toe, as the review asked), fused 2 mm each side of the root, one slice fed 40 mm² in 0.3 s
  (no arc; the droplets' heat only), the torch moving on at one slice per 0.3 s so the slice then rests 0.3 s. A
  2 mm-toed slice holds 6 mm² at most, so it overflows: its toes advance (GA's overflow, ``joint_down``) until its
  top holds the metal, and its face stands as high as the reference surface on its final toes within a cell and a
  half; that surface is at or near the cap, the 2-D puddle's 88 deg height (5.15 mm at 20 mm wide, the wide limit
  5.83 mm). Fails on the cap model, whose weightless half disc stands half its width high. Ledger 1e-6.
- D-10 ``surface_fillet``: D-3's T-joint on ``law_insulated`` at 1700 K, fused 2 mm along each member, 30 mm² into
  one slice in 0.3 s and 0.3 s of rest (the torch moving on): overflow runs along the plate and not up the bracket
  (GA's climb rule, e . down < -0.3), so the plate's toe ends at least 1 mm (4 cells) farther out than the
  bracket's (both wet a step per exchange on this plate, which never freezes a contact: measured 7.93 vs 5.62 mm);
  the cells' centroid lies where the reference surface's on the run's own toes and metal does, within 0.1 mm (the
  staircase of whole cells moved it 0.010 mm; the cap model's weightless circle sits 0.170 mm off), and its face
  stands within a cell and a half of the reference's. Fails on the cap model (symmetric toes, centroid off).
- D-11a ``surface_dent``: D-7's scene (6 mm of joint, the torch parked near its middle with GA's arc force at 241.3 A
  and 1 W of beam) with the surface model and the liquid's weight: each slice's area is what the reference surface
  on its own toes holds at the pool's pressure P, the arc's pressure node by node about the lagged arc root (GA
  pool.py:846-853 with process.py:206-209), P set so the joint holds its metal, the shares then shared along x with
  GA's 1 mm kernel (pool.py:678-710): within D-7's four cells per slice;
  the dent at least 8 cells; ledger 1e-6. The four cells and the eight are set (D-7's bounds), not derived. On the
  cap model the check fails only because there is no lagged arc root to compare with (a guard, not a physics
  test); D-7's chord-mean weightless rule is the cap model's.
- D-11 ``surface_touchdown``: D-3's T-joint on cold steel (300 K; fused 2 mm along each member), 60 mm² into one
  slice in 0.3 s and 0.3 s of rest: the liquid at the toe freezes on the cold plate (GA's contact test), so the slice
  cannot spread; its face turns past 90 deg and the bulge touches down: toe 1 ends past the fused 2 mm by at least
  two cells with the plate under that stretch unfused (weld metal on unfused plate: GA's cold lap), the bracket's
  toe short of the plate's (it rises only where the bulge leans on it: 2.62 mm), and the ledger closes. The bounds
  (two cells, two lap cells) are set, not derived: they say a touchdown happened, not where. Fails on slice 10b's
  cap model, where the excess stays put. Steel at 300 K is far below the contact rule's threshold: D-11b pins the
  threshold itself.
- D-11b ``surface_contact``: D-11's T-joint and feed on ``law_contact`` (no conduction, so each member keeps its
  temperature; steel's effusivity for the contact test), two worlds: the members (beyond the fused 2 mm) at T* + 10 K
  and at T* - 10 K, T* the plate temperature at which liquid at the droplets' temperature meets it at the solidus
  (GA §6: T* = T_s + (e_l / e_s)(T_s - T_drop), e_l from GA's liquid k, ρ, c, 708 K here). Above T* the toes wet:
  the bracket's toe climbs past the fused zone by at least four cells (measured 4.12 mm). Below it the contact
  freezes: the bracket's toe moves at most three cells (only where the bulge leans on it: 2.62 mm) and the plate's
  toe moves out by touchdowns. (60 mm² is more than wetting a step per exchange can spread, so both worlds end
  touching down; the bracket, where only wetting moves a toe, tells them apart.) A rule whose threshold moves by
  more than 10 K (the wetting ratio 1.12 off by 1 %) fails one world: planted, the threshold x1.2 and x0.8 each fail
  one world.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.plugins.harness.scene import Law, LawResult
from alloy.plugins.thermal import grid_surface_arc as A
from alloy.plugins.thermal.grid import graded
from alloy.plugins.thermal.grid_deposit_laws import (CAP, CAP_AREA, CAP_TIME, CAP_TOES, DENT, DENT_START,
                                                     _cap_temps, _dent_init, segment_angle)
from alloy.plugins.thermal.grid_laws import LABEL, TORCH, _cfg, _final, _init, _scene
from alloy.plugins.thermal.grid_surface_ref import Chord

CELL = 2.5e-4
WIRE = 0.98 * math.pi * 0.45e-3 ** 2          # the deposited wire's cross-section (ε π d² / 4) [m²]
WEIGHTLESS = {**CAP, "bead_model": "surface", "gravity": 0.0}
PUDDLE_AREA, PUDDLE_TIME, PUDDLE_FUSED = 40e-6, 0.3, 2e-3
PUDDLE = _cfg(x=graded(0.0, 0.002, CELL), y=graded(-0.016, 0.016, CELL, band=(-0.012, 0.012)),
              z=graded(-0.006, 0.010, CELL, band=(-0.004, 0.008)), boxes=((0.0, -0.016, -0.006, 0.002, 0.016, 0.0),),
              parts=(0,), torch=TORCH, torch_axis=(0.0, 0.0, -1.0), convection=0.0, emissivity=0.0,
              bead_model="surface", joint_root=(0.0, 0.0), joint_rays=((1.0, 0.0), (-1.0, 0.0)),
              joint_down=(0.0, -1.0), deposit_spread=1e-6, material="law_insulated")


def _slice(run):
    """The fed slice: its index, toes, weld cells, parent cells, cell centres (Y, Z) and widths."""
    grid = run.sim.plugins[LABEL]
    f = grid.fields(0)
    weld_all = f["cell"] == grid._bead.weld
    ax = int(np.argmax(weld_all.sum(axis=(1, 2))))
    cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.y, grid.config.z))
    Y, Z = np.meshgrid(cy, cz, indexing="ij")
    dy, dz = np.diff(np.asarray(grid.config.y)), np.diff(np.asarray(grid.config.z))
    toes = tuple(float(v) for v in grid.toes.numpy()[0, ax])
    return grid, ax, toes, weld_all[ax], f["cell"][ax] != 0, Y, Z, np.outer(dy, dz)


def under(surface: A.Surface, chord: Chord, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Cell centres between the rays and under the reference surface (its radii linear in the polar angle)."""
    fr = A.frame(chord)
    loc = fr.local(np.stack([Y, Z], axis=-1))
    a = np.arctan2(loc[..., 1], loc[..., 0])
    ang = np.linspace(0.0, fr.sweep, len(surface.points))
    rad = np.hypot(*(surface.points - fr.root).T)
    inside = (a >= 0.0) & (a <= fr.sweep)
    return inside & (np.hypot(loc[..., 0], loc[..., 1]) <= np.interp(np.clip(a, 0.0, fr.sweep), ang, rad))


def _ledger(run) -> float:
    _, s = _final(run)
    return abs(s["residual_J"]) / (s["arc_J"] + abs(s["filler_J"]))


def _check_weightless(run) -> LawResult:
    grid, ax, (t1, t2), weld, metal, Y, Z, yz = _slice(run)
    area = float((weld * yz).sum())
    target = float(grid.slice_fed_m3.numpy()[0, ax]) / float(np.diff(np.asarray(grid.config.x))[ax])
    chord = math.hypot(t1, t2)
    th = segment_angle(chord, target - 0.5 * t1 * t2)
    sagitta = chord / (2 * math.sin(0.5 * th)) * (1 - math.cos(0.5 * th))
    stand = float(((Y * t2 + Z * t1 - t1 * t2) / chord)[weld].max())
    ch = Chord(np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), t1, t2)
    ref = A.pressure_for(ch, area, A.Physics(grid.config.surface_tension, 6950.0, np.zeros(2)))
    want = under(ref, ch, Y, Z) & (~metal | weld) & (Y > 0) & (Z > 0)
    differ = int((want ^ weld).sum())
    face = math.pi * chord / 2.0 / CELL                           # cells along a half-disc face, an upper bound
    rel = _ledger(run)
    detail = (f"toes {t1 * 1e3:.3f}, {t2 * 1e3:.3f} mm (set 4, 3); area {area * 1e6:.3f} vs feed {target * 1e6:.3f} "
              f"mm²; face {stand * 1e3:.3f} mm beyond the chord vs sagitta {sagitta * 1e3:.3f} mm; {differ} cells "
              f"differ from the reference surface (P {ref.pressure:.1f} Pa; bound {face:.0f}); ledger {rel:.1e}")
    ok = (abs(t1 - CAP_TOES[0]) <= CELL and abs(t2 - CAP_TOES[1]) <= CELL and abs(area - target) <= 3 * CELL ** 2
          and abs(stand - sagitta) <= 1.5 * CELL and differ <= face and rel <= 1e-6)
    return LawResult(ok, abs(stand - sagitta), 1.5 * CELL, detail)


def _puddle_temps(grid) -> np.ndarray:
    cx, cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.x, grid.config.y,
                                                                              grid.config.z))
    _, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    fused = (np.abs(Y) < PUDDLE_FUSED) & (Z > -1e-3) & (Z < 0)
    return np.where(fused, 1800.0, 1700.0).reshape(-1)


def _check_puddle(run) -> LawResult:
    grid, ax, (t1, t2), weld, metal, Y, Z, yz = _slice(run)
    area = float((weld * yz).sum())
    ch = Chord(np.zeros(2), np.array([1.0, 0.0]), np.array([-1.0, 0.0]), t1, t2)
    ph = A.Physics(grid.config.surface_tension, 6950.0, np.array([0.0, -grid.config.gravity]))
    ref = A.pressure_for(ch, area, ph)
    cap = A.top(ch, ph)
    height = float((Z + 0.5 * CELL)[weld].max())                   # the top face of the highest weld cell
    want_h = float(ref.points[:, 1].max())
    want = under(ref, ch, Y, Z) & (~metal | weld) & (Z > 0)
    differ = int((want ^ weld).sum())
    face = (t1 + t2 + 2 * want_h) / CELL
    rel = _ledger(run)
    detail = (f"toes {t1 * 1e3:.2f}, {t2 * 1e3:.2f} mm (fused {PUDDLE_FUSED * 1e3:.0f}); {area * 1e6:.2f} mm² of "
              f"weld; height {height * 1e3:.3f} mm vs the reference surface's {want_h * 1e3:.3f} ({ref.status}, P "
              f"{ref.pressure:.1f} Pa; the cap on these toes holds {cap.area * 1e6:.2f} mm² and stands "
              f"{cap.points[:, 1].max() * 1e3:.3f} mm); {differ} cells differ (bound {face:.0f}); ledger {rel:.1e}")
    ok = (min(t1, t2) >= PUDDLE_FUSED + 2 * CELL and abs(height - want_h) <= 1.5 * CELL and differ <= face
          and area >= 0.9 * cap.area and rel <= 1e-6)
    return LawResult(ok, abs(height - want_h), 1.5 * CELL, detail)


FILLET_AREA = 30e-6
FILLET = {**CAP, "bead_model": "surface", "material": "law_insulated", "joint_down": (0.0, -1.0),
          "x": graded(0.0, 0.002, CELL)}


def _fillet_temps(grid) -> np.ndarray:
    cx, cy, cz = (0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.x, grid.config.y,
                                                                              grid.config.z))
    _, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    fused = ((Y > 0) & (Y < PUDDLE_FUSED) & (Z > -1e-3) & (Z < 0)) | ((Y > -1e-3) & (Y < 0) & (Z > 0) &
                                                                       (Z < PUDDLE_FUSED))
    return np.where(fused, 1800.0, 1700.0).reshape(-1)


def _check_fillet(run) -> LawResult:
    grid, ax, (t1, t2), weld, metal, Y, Z, yz = _slice(run)
    area = float((weld * yz).sum())
    ch = Chord(np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), t1, t2)
    ref = A.pressure_for(ch, area, A.Physics(grid.config.surface_tension, 6950.0,
                                             np.array([0.0, -grid.config.gravity])))
    want = under(ref, ch, Y, Z) & (~metal | weld) & (Y > 0) & (Z > 0)
    cen = np.array([(Y * weld * yz).sum(), (Z * weld * yz).sum()]) / max(area, 1e-30)
    w_ref = want * yz
    cen_ref = np.array([(Y * w_ref).sum(), (Z * w_ref).sum()]) / max(float(w_ref.sum()), 1e-30)
    off = float(np.hypot(*(cen - cen_ref)))
    out = (Y + Z) / math.sqrt(2.0)                                   # along the bisector
    face, face_ref = float((out + CELL / math.sqrt(2.0))[weld].max()), float(np.max(ref.points.sum(1)) / math.sqrt(2))
    rel = _ledger(run)
    detail = (f"toes {t1 * 1e3:.2f} (plate), {t2 * 1e3:.2f} (bracket) mm, fused {PUDDLE_FUSED * 1e3:.0f}; "
              f"{area * 1e6:.2f} mm²; centroid ({cen[0] * 1e3:.3f}, {cen[1] * 1e3:.3f}) vs the reference's "
              f"({cen_ref[0] * 1e3:.3f}, {cen_ref[1] * 1e3:.3f}) mm ({ref.status}, P {ref.pressure:.1f} Pa), off "
              f"{off * 1e3:.3f} mm; face {face * 1e3:.3f} vs {face_ref * 1e3:.3f} mm out along the bisector; "
              f"ledger {rel:.1e}")
    ok = t1 - t2 >= 4 * CELL and off <= 1e-4 and abs(face - face_ref) <= 1.5 * CELL and rel <= 1e-6
    return LawResult(ok, off, 1e-4, detail)


def _dent_reference(grid, area: np.ndarray):
    """Each slice's area by the surface model's pool rule, in numpy: (areas, P)."""
    ex = np.asarray(grid.config.x)
    dx, xc = np.diff(ex), 0.5 * (ex[1:] + ex[:-1])
    toes, at, force = grid.toes.numpy()[0], grid.arc_at.numpy()[0], float(grid.arc_force.numpy()[0])
    sig = grid.config.arc_pressure[1]
    rows = []
    for i in range(len(xc)):
        ch = Chord(np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), float(toes[i][0]), float(toes[i][1]))
        peak = force / (2 * math.pi * sig * sig) * math.exp(-0.5 * (xc[i] - at[0]) ** 2 / (sig * sig))
        ph = A.Physics(grid.config.surface_tension, 6950.0, np.array([0.0, -grid.config.gravity]), arc_peak=peak,
                       arc_at=np.array(at[1:], float), arc_sigma=sig)
        rows.append((ch, ph, A.top(ch, ph)))

    def held(p: float) -> np.ndarray:
        out = []
        for ch, ph, cap in rows:
            s = A.surface(ch, p, ph, cap, steps=32)
            out.append(0.0 if s is None else s.area)
        return np.array(out)

    total = float((area * dx).sum())
    lo, hi = min(r[2].pressure for r in rows) - 3000.0, max(r[2].pressure for r in rows)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float((held(mid) * dx).sum()) < total else (lo, mid)
        if hi - lo < 1e-3:
            break
    return smoothed(held(hi), xc, dx, np.array([r[2].area for r in rows])), hi


def smoothed(share: np.ndarray, xc: np.ndarray, dx: np.ndarray, top: np.ndarray, sigma: float = 1e-3) -> np.ndarray:
    """GA's ``_longitudinal`` (pool.py:678-694): the shares shared along x with a Gaussian of ``sigma`` cut at 3
    sigma, the metal conserved and each slice held to its top, four rounds."""
    d = xc[:, None] - xc[None, :]
    k = np.where(np.abs(d) <= 3 * sigma, np.exp(-0.5 * (d / sigma) ** 2), 0.0) * dx[None, :]
    tot = float((share * dx).sum())
    sm = (k @ share) / k.sum(1)
    for _ in range(4):
        sm = sm * tot / float((sm * dx).sum())
        over = sm > top
        sm = np.minimum(sm, top)
        if not over.any():
            break
    return sm


def _check_dent(run) -> LawResult:
    grid = run.sim.plugins[LABEL]
    f, s = _final(run)
    if not hasattr(grid, "arc_at"):
        return LawResult(False, math.inf, 4 * CELL ** 2, "not the surface model: no lagged arc root to check against")
    yz = np.outer(np.diff(np.asarray(grid.config.y)), np.diff(np.asarray(grid.config.z)))
    area = ((f["cell"] == grid._bead.weld) * yz[None]).sum(axis=(1, 2))
    want, p = _dent_reference(grid, area)
    cell = CELL ** 2
    off, dent = float(np.abs(area - want).max()), float(area.max() - area.min())
    rel = abs(s["residual_J"]) / max(abs(s["filler_J"]), 1e-30)
    at = grid.arc_at.numpy()[0]
    detail = (f"slice areas {area.min() * 1e6:.3f} (under the arc) to {area.max() * 1e6:.3f} mm², the reference "
              f"{want.min() * 1e6:.3f} to {want.max() * 1e6:.3f} (P {p:.0f} Pa; lagged root {at[1] * 1e3:.2f}, "
              f"{at[2] * 1e3:.2f} mm, force {float(grid.arc_force.numpy()[0]):.4f} N); worst slice off "
              f"{off / cell:.1f} cells; ledger {rel:.1e}")
    ok = off <= 4 * cell and dent >= 8 * cell and rel <= 1e-6 and s["missed_m3"] == 0.0
    return LawResult(ok, off, 4 * cell, detail)


TOUCH_AREA = 60e-6
TOUCH = {**FILLET, "material": "steel_a36"}


def _touch_temps(grid) -> np.ndarray:
    return np.where(_fillet_temps(grid) > 1750.0, 1800.0, 300.0)


def _check_touchdown(run) -> LawResult:
    grid, ax, (t1, t2), weld, metal, Y, Z, yz = _slice(run)
    f = grid.fields(0)
    cy = Y[:, 0]
    k0 = int(np.searchsorted(Z[0], 0.0)) - 1                            # the plate's top row of cells
    lap = (cy > PUDDLE_FUSED + CELL) & (cy < t1 - CELL)
    above = weld[:, k0 + 1]
    unfused = f["peak"][ax][:, k0] < 0.5 * (1745.15 + 1785.15)
    lapped = int((lap & above & unfused).sum())
    status = int(grid.surf_state.numpy()[0, ax][2]) if hasattr(grid, "surf_state") else -1     # -1: no surface model
    rel = _ledger(run)
    area = float((weld * yz).sum())
    detail = (f"toes {t1 * 1e3:.2f} (plate), {t2 * 1e3:.2f} (bracket) mm, fused {PUDDLE_FUSED * 1e3:.0f}; "
              f"{area * 1e6:.2f} mm² of weld; {lapped} plate cells past the fused zone under weld and unfused (the "
              f"lap); last surface status {status} (5 overhang, 6 touch); ledger {rel:.1e}")
    ok = t1 >= PUDDLE_FUSED + 2 * CELL and lapped >= 2 and t1 > t2 and rel <= 1e-6
    return LawResult(ok, float(t1), PUDDLE_FUSED + 2 * CELL, detail)


CONTACT = {**TOUCH, "material": "law_contact"}
E_LIQUID = math.sqrt(31.0 * 6950.0 * 808.0)   # liquid steel's k, ρ, c (GA §6, Mills 2002)
CONTACT_MARGIN = 10.0                          # K either side of T*


def contact_threshold(grid) -> float:
    """T*: the plate temperature at which the droplets' liquid meets it at the solidus (GA §6, _contact_freezes)."""
    m = grid._mat
    return m.t_solidus + E_LIQUID / m.contact_effusivity * (m.t_solidus - grid.config.drop_temperature)


def _contact_temps(grid) -> np.ndarray:
    t = contact_threshold(grid)
    fused = _fillet_temps(grid) > 1750.0
    return np.stack([np.where(fused, 1800.0, t + d) for d in (CONTACT_MARGIN, -CONTACT_MARGIN)])


def _check_contact(run) -> LawResult:
    grid = run.sim.plugins[LABEL]
    ax = _slice(run)[1]
    toes, st = grid.toes.numpy()[:, ax], grid.surf_state.numpy()[:, ax, 2].astype(int)
    rel = _ledger(run)
    (w1, w2), (c1, c2) = toes[0], toes[1]
    warm = w2 >= PUDDLE_FUSED + 4 * CELL
    cold = c2 <= PUDDLE_FUSED + 3 * CELL and c1 >= PUDDLE_FUSED + 2 * CELL
    detail = (f"T* {contact_threshold(grid):.1f} K; members at T* + {CONTACT_MARGIN:.0f} K: toes {w1 * 1e3:.2f} "
              f"(plate), {w2 * 1e3:.2f} (bracket) mm, status {st[0]}; at T* - {CONTACT_MARGIN:.0f} K: toes "
              f"{c1 * 1e3:.2f}, {c2 * 1e3:.2f} mm, status {st[1]} (1 held, 2 wet, 3 full, 5 overhang, 6 touch, 7 "
              f"leaning); fused {PUDDLE_FUSED * 1e3:.0f} mm; ledger {rel:.1e}")
    return LawResult(warm and cold and rel <= 1e-6, float(w2 - c2), 2 * CELL, detail)


def surface_laws() -> list[Law]:
    return [
        Law("surface_weightless", _check_weightless, scene=_scene(round(CAP_TIME / 0.02), _init(
            feed=CAP_AREA * CELL / (WIRE * CAP_TIME), temps=_cap_temps), start=(0.001, 0.01, 0.01)), config=WEIGHTLESS,
            doc="D-8: the weightless surface is D-3's circle: toes, area, sagitta, cells within the face"),
        Law("surface_puddle", _check_puddle, scene=_scene(round(2 * PUDDLE_TIME / 0.02), _init(
            feed=PUDDLE_AREA * CELL / (WIRE * PUDDLE_TIME), temps=_puddle_temps, speed=(CELL / PUDDLE_TIME, 0.0, 0.0)),
            start=(2 * CELL, 0.0, 0.01)),
            config=PUDDLE, doc="D-9: an overfull puddle spreads until its top holds it; it stands as high as the "
                               "reference surface on its toes (the 2-D puddle's 88° height near the cap)"),
        Law("surface_fillet", _check_fillet, scene=_scene(round(2 * PUDDLE_TIME / 0.02), _init(
            feed=FILLET_AREA * CELL / (WIRE * PUDDLE_TIME), temps=_fillet_temps, speed=(CELL / PUDDLE_TIME, 0.0, 0.0)),
            start=(2 * CELL, 0.01, 0.01)), config=FILLET,
            doc="D-10: an overfull fillet sags onto the plate, not up the bracket, its centroid on the reference's"),
        Law("surface_dent", _check_dent, scene=_scene(round(CAP_TIME / 0.02), _dent_init, start=DENT_START),
            config={**DENT, "bead_model": "surface"},
            doc="D-11a: the arc dents the surface pool: each slice holds what its surface holds at P, node by node"),
        Law("surface_touchdown", _check_touchdown, scene=_scene(round(2 * PUDDLE_TIME / 0.02), _init(
            feed=TOUCH_AREA * CELL / (WIRE * PUDDLE_TIME), temps=_touch_temps, speed=(CELL / PUDDLE_TIME, 0.0, 0.0)),
            start=(2 * CELL, 0.01, 0.01)), config=TOUCH,
            doc="D-11: on cold steel an overfull slice overhangs and touches down: toe 1 out over unfused plate"),
        Law("surface_contact", _check_contact, scene=_scene(round(2 * PUDDLE_TIME / 0.02), _init(
            feed=TOUCH_AREA * CELL / (WIRE * PUDDLE_TIME), temps=_contact_temps, speed=(CELL / PUDDLE_TIME, 0.0, 0.0)),
            start=(2 * CELL, 0.01, 0.01), worlds=2), config=CONTACT,
            doc="D-11b: the toe wets 10 K above GA's contact threshold and freezes (a touchdown) 10 K below it"),
    ]


__all__ = ["surface_laws", "under"]
