# SPDX-License-Identifier: Apache-2.0
"""Concrete, repair mortar, masonry mortar and sealant: setting curves, bleeding and per-pass sole-work laws.

A port of Gym-Anything's concrete_finishing material and process laws (CF = [internal note]
benchmarks/roboworld/environments/concrete_finishing; M = CF/materials.py, P = CF/params.py, C = CF/coupling.py,
K = CF/paste.py; every cite is file:line). Constants labelled "estimate" in their parameter ledger stay estimates
here: they are judgement values, not measurements. Units: SI (Pa, s, m, kg); ages in seconds, temperatures in C.

Two layers, one set of formulas: numpy functions (vectorised over arrays; the reference and the test oracle) and
`wp.func` twins (section "Warp") for per-cell kernels. The Warp twins do their arithmetic in float64 and return
float32. Material parameters are passed to the Warp functions as arguments (take them from the dataclasses);
the global constants below are inlined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

# --- global constants (P = params.py) -------------------------------------------------------------------------
RHO_CONCRETE = 2350.0          # kg/m3, estimate (P:65)
SLUMP_M = 0.100                # m, opened: CIP 13 (P:66)
# NISTIR 6094 eq. 19 at slump 100 mm: rho/347*(300 - 100) + 212 = 1566 Pa (derived, P:67)
TAU_B0 = RHO_CONCRETE / 347.0 * (300.0 - 100.0) + 212.0
A_THIX = 0.25                  # Pa/s, estimate: Roussel's ordinary range (P:68)
N_DEEP = 9.0                   # deep punch bearing factor, estimate (P:69)
PR_IS = 3.45e6                 # Pa, ASTM C403 initial set 500 psi, opened (P:70)
PR_FS = 27.6e6                 # Pa, ASTM C403 final set 4000 psi, opened (P:71)
T_IS = 4.0 * 3600              # s, estimate: initial set 4 h at 70 F (P:72)
T_FS = 6.0 * 3600              # s, opened: final set about 6 h at 70 F (P:73)
SET_M = math.log(PR_FS / PR_IS) / math.log(T_FS / T_IS)   # 5.13, derived (P:74)
TAU_IS = PR_IS / N_DEEP        # Pa, yield stress at initial set, 383 kPa, derived (P:75)
# MASCO final set (h) against temperature (F), opened, approximate (P:76-77)
MASCO = {30: 19.0, 40: 14 + 40 / 60, 50: 10 + 20 / 60, 60: 8.0, 70: 6.0, 80: 4.0, 90: 2 + 40 / 60, 100: 1 + 40 / 60}
R_SURF = (0.15, 0.30)          # concrete surface layer / bulk strength, floated -> densified, estimate (P:78)
BLEED_MEAN = 0.15              # kg/m2/h, estimate (P:98)
BLEED_END_TAU = 1.0e4          # Pa, estimate: bleeding ends when the bulk reaches 10 kPa, ~110 min at 21 C (P:99)
SHEEN_W = 2.0e-5               # m, film thicker than this reads as sheen, estimate (P:100)
AIR = {"still": (20.0, 50.0, 0.2), "door": (20.0, 50.0, 1.0), "fan": (20.0, 40.0, 2.5)}   # C, %, m/s (P:101)
BLISTER_Q = 2.0e-4             # m, trapped bleed water that raises a blister, estimate (P:103)
BLISTER_GAIN = 4.0             # lift = 4 (q - BLISTER_Q), phenomenological (K:196)
BLISTER_BLUR_M = 0.0075        # m, box blur half width (K:199, 205)
BLISTER_CAP = 0.003            # m, blister lift cap (K:211)
EMBED_PASS = 8.3               # e-folds of aggregate exposure per pass at 1 kPa, estimate (P:87)
TEX_PASS = 2.2                 # e-folds of texture toward the tool's own per pass, estimate (P:91)
DENS_ENERGY = 112.0            # J/m2, e-fold densification work, estimate (P:94)
NC = 2.0 + math.pi             # Prandtl's strip factor, opened (P:125)
M_INTERFACE = (0.05, 0.30)     # interface shear / surface yield stress, wet film -> stiff, estimate (P:128)
CEMENT = 350.0                 # kg/m3, cement content: the water worked into the top 6 mm, estimate (P:83)
DWC_LAYER = 0.006              # m: the top layer whose w/c a worked-in film raises (C:1060-1063)
SKIN_FACTOR = 3.0              # a skinned sealant's surface / bulk yield, estimate (P:169)
SEALANT = dict(rho=1400.0, tau0=600.0, mu=50.0, skin=2 * 3600.0)   # estimate / opened (P:119)
FAST_SKIN = 20 * 60.0          # s: a fast-skinning grade (their fast-sealant variant)
TORN_A = 5.0e-4                # m: texture of a torn skin (C:1081-1083)
M_INTERFACE_LOG10 = (math.log10(300.0), 2.0)   # log10(300 Pa) and decades to full stiffness (C:1099-1100)
SEAL_PITCH_MAX = 0.6           # pitch cap in the sealing rule (C:1056)
SEAL_LENGTH = 0.10             # m, sealing length scale (C:1056)
MORTAR_COMPACT = 1.0           # compaction saturates at 1 (P:111)
MORTAR_LAYER = 0.002           # m, surface layer rearranged by tooling, estimate (P:113)
MORTAR_STRAIN = 6.0            # characteristic shear strain, estimate (P:115)
MORTAR_HARDEN = 0.30           # maximum surface yield increase from compaction, estimate (P:117)
THUMBPRINT = (5.0e3, 2.0e4)    # Pa, thumbprint-hard window, estimate (P:110)

_MAX_AGE_S = 48 * 3600.0       # bisection bracket (M:46)


def masco_factor(temp_c: float) -> float:
    """Set-time factor against 70 F from MASCO's final-set table: 1.72 at 10 C ... 0.45 at 32 C (M:16-20).

    Linear interpolation of the table in F, divided by the 70 F value (6 h). 21.1 C is 69.98 F, so s is 1.0007."""
    tf = temp_c * 1.8 + 32.0
    ks = sorted(MASCO)
    return float(np.interp(tf, ks, [MASCO[k] for k in ks]) / MASCO[70])


def scaled_start_age_s(age_at_21c_s: float, temp_c: float) -> float:
    """Start ages are given at 21 C and scaled by s(T) so every temperature starts at the same stage
    (spec section 0: layout.py:241-242, M:16-20; layout.py not read here)."""
    return age_at_21c_s * masco_factor(temp_c)


def evaporation_kg_m2_h(concrete_c: float, air_c: float, rh_percent: float, wind_m_s: float) -> float:
    """ACI 308 nomograph equation as printed in FHWA-RD-02-099 figure 6 (M:166-169), kg/m2/h."""
    return 4.88 * (0.1113 + 0.04224 * wind_m_s / 0.447) * 0.0443 * (
        math.exp(0.0302 * (concrete_c * 1.8 + 32)) - rh_percent / 100.0 * math.exp(0.0302 * (air_c * 1.8 + 32)))


def air_evaporation_kg_m2_h(preset: str, concrete_c: float = 21.1) -> float:
    """Evaporation under an air preset ('still', 'door', 'fan'; P:101) with the concrete at ``concrete_c``: the room's
    air is the preset's at a 21.1 C site, shifted with the site's temperature (paste.py:80-81)."""
    air_c, rh, wind = AIR[preset]
    return evaporation_kg_m2_h(concrete_c, air_c + (concrete_c - 21.1), rh, wind)


def _time_to(tau_fn, target: float) -> float:
    """Age (s) at which a monotone tau_fn(age) reaches target, by bisection (M:45-53); 0 if already above."""
    lo, hi = 0.0, _MAX_AGE_S
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(tau_fn(mid)) < target:
            lo = mid
        else:
            hi = mid
    return hi


# --- materials ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Concrete:
    """100 mm slump concrete. tau_b(a) = tau0 + A_THIX a/s + TAU_IS (a/(t_is s))^m, tau_s = (r0 + (r1-r0) D) tau_b
    (M:23-43; R_SURF 0.15-0.30, estimate). Bleeding: triangle, mean `bleed_mean`, until tau_b = 10 kPa (M:55-64)."""

    rho: float = RHO_CONCRETE
    tau0: float = TAU_B0
    a_thix: float = A_THIX
    tau_is: float = TAU_IS
    t_is: float = T_IS
    m: float = SET_M
    surf_r: tuple = R_SURF
    bleed_mean: float = BLEED_MEAN      # kg/m2/h

    def tau_bulk(self, age_s, temp_c: float = 21.1):
        """Bulk yield stress [Pa] at age (s) (M:37-39)."""
        s = masco_factor(temp_c)
        a = np.maximum(np.asarray(age_s, float), 0.0)
        return self.tau0 + self.a_thix * a / s + self.tau_is * (a / (self.t_is * s)) ** self.m

    def tau_surf(self, tau_b, dens=0.0):
        """Surface-layer yield stress [Pa] from the bulk one and densification D (M:41-43)."""
        r0, r1 = self.surf_r
        return (r0 + (r1 - r0) * np.asarray(dens, float)) * np.asarray(tau_b, float)

    def time_to_tau(self, target_pa: float, temp_c: float = 21.1) -> float:
        """Age (s) at which tau_bulk reaches target_pa (bisection, as M:45-53)."""
        return _time_to(lambda a: self.tau_bulk(a, temp_c), target_pa)

    def initial_set_time(self, temp_c: float = 21.1) -> float:
        """Age (s) at which tau_b = PR_IS / N_DEEP (383 kPa; M:35)."""
        return self.time_to_tau(PR_IS / N_DEEP, temp_c)

    def final_set_time(self, temp_c: float = 21.1) -> float:
        """Age (s) at which tau_b = PR_FS / N_DEEP (M:71)."""
        return self.time_to_tau(PR_FS / N_DEEP, temp_c)

    def bleed_end_time(self, temp_c: float = 21.1) -> float:
        """Age (s) at which tau_b = BLEED_END_TAU (M:34)."""
        return self.time_to_tau(BLEED_END_TAU, temp_c)

    def bleed_rate(self, age_s, temp_c: float = 21.1):
        """kg/m2/s of bleed water: triangle over [0, t_end], peak 2 mean / 3600 at t_end/3 (M:55-64)."""
        a = np.asarray(age_s, float)
        te = self.bleed_end_time(temp_c)
        peak = 2.0 * self.bleed_mean / 3600.0
        r = np.where(a < te / 3.0, peak * a / (te / 3.0), peak * (te - a) / (2.0 * te / 3.0))
        return np.clip(r, 0.0, None)


@dataclass(frozen=True)
class RepairMortar(Concrete):
    """Generic repair mortar (scenario blend, estimate, P:106): rho 2051, tau0 2000 Pa, t_is 50 min, bleed 0.05
    kg/m2/h, same setting form as Concrete (M:75-82). tau_s = (0.5 + 0.5 D) tau_b (M:84-86)."""

    rho: float = 2051.0
    tau0: float = 2000.0
    t_is: float = 50 * 60.0
    surf_r: tuple = (0.5, 1.0)
    bleed_mean: float = 0.05
    mu: float = 20.0                    # Pa s (P:106)


@dataclass(frozen=True)
class Sealant:
    """Their joint sealant (M:126-142): the yield is tau0 at every age; the surface is SKIN_FACTOR x the bulk once its
    skin forms at skin_s x MASCO (2 h; a fast grade 20 min); it does not bleed. Tack is not modelled."""

    rho: float = SEALANT["rho"]
    tau0: float = SEALANT["tau0"]
    mu: float = SEALANT["mu"]
    skin_s: float = SEALANT["skin"]

    def skin_time(self, temp_c: float = 21.1) -> float:
        return self.skin_s * masco_factor(temp_c)

    def tau_bulk(self, age_s, temp_c: float = 21.1):
        return np.full(np.shape(age_s), self.tau0, float)

    def tau_surf(self, tau_b, age_s, temp_c: float = 21.1):
        """The surface's yield: SKIN_FACTOR x the bulk's once skinned (M:139-140)."""
        return np.asarray(tau_b, float) * np.where(np.asarray(age_s) >= self.skin_time(temp_c), SKIN_FACTOR, 1.0)

    def initial_set_time(self, temp_c: float = 21.1) -> float:
        return math.inf

    def bleed_rate(self, age_s, temp_c: float = 21.1):
        return np.zeros_like(np.asarray(age_s, float))


def dwc_rise(w):
    """The top 6 mm's w/c rise when a film w [m] is worked in (C:1060-1063): 1000 w / (CEMENT x 0.006)."""
    return 1000.0 * np.asarray(w, float) / (CEMENT * DWC_LAYER)


@dataclass(frozen=True)
class MasonryMortar:
    """Type N masonry mortar stiffening by suction (estimate, P:108-109; M:89-123):
    tau_b = tau0 + tau_ref (a/(s t_ref))^m; tau_s = tau_b (1 + 0.30 C); no bleeding."""

    rho: float = 2000.0
    tau0: float = 1000.0
    tau_ref: float = 8000.0
    t_ref: float = 35 * 60.0
    m: float = 3.0
    mu: float = 10.0
    harden: float = MORTAR_HARDEN

    def tau_bulk(self, age_s, temp_c: float = 21.1):
        a = np.maximum(np.asarray(age_s, float), 0.0) / masco_factor(temp_c)
        return self.tau0 + self.tau_ref * (a / self.t_ref) ** self.m           # M:98-101

    def tau_surf(self, tau_b, compact=0.0):
        """Surface yield stress from bulk and compaction C in [0, 1] (M:103-105)."""
        return np.asarray(tau_b, float) * (1.0 + self.harden * np.clip(compact, 0.0, 1.0))

    def time_to_tau(self, target_pa: float, temp_c: float = 21.1) -> float:
        return _time_to(lambda a: self.tau_bulk(a, temp_c), target_pa)

    def initial_set_time(self, temp_c: float = 21.1) -> float:
        """Masonry mortar has no initial-set mark in their code (no `initial_set`; C:1090 `late` stays 0): inf."""
        return math.inf

    def bleed_rate(self, age_s, temp_c: float = 21.1):
        return np.zeros_like(np.asarray(age_s, float))                        # M:107-108


# --- per-pass sole-work laws (numpy; C:999-1103) ------------------------------------------------------------
def passes_of_slide(ds, bearing_extent_along_motion):
    """Passes n = ds / B: B is the bearing patch's extent along the motion (C:1010-1022)."""
    return np.asarray(ds, float) / np.asarray(bearing_extent_along_motion, float)


def embed_update(E, pressure_pa, passes):
    """Aggregate exposure E <- E exp(-n k_E clip(p/1 kPa, 0.1, 20)), k_E = EMBED_PASS (C:1024-1025)."""
    k = np.asarray(passes, float) * EMBED_PASS * np.clip(np.asarray(pressure_pa, float) / 1000.0, 0.1, 20.0)
    return np.asarray(E, float) * np.exp(-k)


def interface_factor(tau_s, film_w=0.0):
    """m_i: 0.05 on a wet film (w > SHEEN_W), else 0.05 + 0.25 clip((log10 tau_s - log10 300)/2, 0, 1) (C:1096-1103)."""
    lo, hi = M_INTERFACE
    ts = np.maximum(np.asarray(tau_s, float), 1.0)
    x = np.clip((np.log10(ts) - M_INTERFACE_LOG10[0]) / M_INTERFACE_LOG10[1], 0.0, 1.0)
    return np.where(np.asarray(film_w, float) > SHEEN_W, lo, lo + (hi - lo) * x)


def texture_yield(pressure_pa, tau_s, m_i):
    """y = max(clip(p/(N_c tau_s), 0, 1), m_i): how fully the surface yields under the sole (C:1034)."""
    return np.maximum(np.clip(np.asarray(pressure_pa, float) / (NC * np.asarray(tau_s, float)), 0.0, 1.0), m_i)


def texture_update(a, a_tool, passes, y):
    """Texture amplitude a <- a_tool + (a - a_tool) exp(-n k_a y), k_a = TEX_PASS (C:1036)."""
    return a_tool + (np.asarray(a, float) - a_tool) * np.exp(-np.asarray(passes, float) * TEX_PASS * y)


def densify_update(D, tau_s, ds, m_i, tool_dens):
    """D <- 1 - (1 - D) exp(-Wt / 112 J/m2), Wt = m_i tau_s ds * tool dens (C:1039-1042)."""
    wt = np.asarray(m_i, float) * np.asarray(tau_s, float) * ds * tool_dens
    return 1.0 - (1.0 - np.asarray(D, float)) * np.exp(-wt / DENS_ENERGY)


def seal_update(S, ds, tool_seal, pitch, pressure_pa):
    """S <- 1 - (1 - S) exp(-ds seal min(pitch, 0.6) max(p/1 kPa, 0.1) / 0.10) (C:1049-1056)."""
    rate = ds * tool_seal * np.minimum(pitch, SEAL_PITCH_MAX) * np.maximum(np.asarray(pressure_pa, float) / 1000.0, 0.1)
    return 1.0 - (1.0 - np.asarray(S, float)) * np.exp(-rate / SEAL_LENGTH)


def compact_update(C, pressure_pa, ds, tau_bulk, m_i):
    """Mortar compaction C <- 1 - (1-C) exp(-dose), dose = p/(p+tau_b) m_i ds / (2 mm * 6) (M:111-123)."""
    tau = np.maximum(np.asarray(tau_bulk, float), 1.0)
    p = max(float(pressure_pa), 0.0)
    dose = p / (p + tau) * np.asarray(m_i, float) * max(float(ds), 0.0) / (MORTAR_LAYER * MORTAR_STRAIN)
    return 1.0 - (1.0 - np.clip(np.asarray(C, float), 0.0, 1.0)) * np.exp(-dose)


def blister_lift(q, cell_m, previous=None, wrap=True):
    """Blister lift [m] from trapped water q [m] on a 2-D grid (K:192-214): 4 (q - 2e-4)+, box blur +-7.5 mm
    (k = max(1, round(0.0075/cell)) cells, np.roll as theirs, so the grid wraps), capped at 3 mm; `previous` lift
    is kept where larger (a blister does not sink back). `cell_m` may be the (i, j) sizes (k per axis); `wrap`
    False reads zeros past the grid's edges (the paste layer: our charts end at the paste, where their margin holds
    no water)."""
    q = np.asarray(q, float)
    lift = np.where(q > BLISTER_Q, (q - BLISTER_Q) * BLISTER_GAIN, 0.0)
    cells = (cell_m, cell_m) if np.isscalar(cell_m) else tuple(cell_m)
    for ax in (0, 1):
        k = max(1, int(round(BLISTER_BLUR_M / cells[ax])))
        acc = np.zeros_like(lift)
        for s in range(-k, k + 1):
            acc += np.roll(lift, s, axis=ax) * (1.0 if wrap else _inside(lift.shape, ax, s))
        lift = acc / (2 * k + 1)
    prev = 0.0 if previous is None else np.asarray(previous, float)
    return np.minimum(np.maximum(prev, lift), BLISTER_CAP)


def _inside(shape, ax: int, s: int) -> np.ndarray:
    """1 where np.roll(x, s, axis=ax) reads a cell inside the grid (not wrapped round), else 0."""
    idx = np.arange(shape[ax]) - s
    ok = ((idx >= 0) & (idx < shape[ax])).astype(float)
    return ok[:, None] if ax == 0 else ok[None, :]


def bearing_pressure(tau_b, tau_s, width, length=None, depth=0.0, ts=None, mortar_layer=False):
    """Bearing pressure [Pa] of a sole of narrow width B over paste (K:153-172). min of the bulk term
    N_c (1 + 0.2 min(B/L, 1)) (1 + 0.2 min(D/B, 2.5)) tau_b and the surface-layer term tau_s (2 + B/(4 ts)),
    ts floored at 1e-7 m and the layer term used only where ts > 1e-5 m. `mortar_layer` adds the masonry mortar's
    N_c (tau_s - tau_b)+ min(2 mm / B, 1) (K:168-171). ts None means no surface layer (bulk term only)."""
    tb, tsv = np.asarray(tau_b, float), np.asarray(tau_s, float)
    shape = 1.0 + 0.2 * min(width / length, 1.0) if length else 1.0
    deep = NC * shape * (1.0 + 0.2 * min(max(depth, 0.0) / max(width, 1e-6), 2.5)) * tb
    if ts is None:
        pressure = deep
    else:
        thin = np.maximum(np.asarray(ts, float), 1e-7)
        pressure = np.minimum(deep, np.where(np.asarray(ts, float) > 1e-5, tsv * (2.0 + width / (4.0 * thin)), deep))
    if mortar_layer:
        pressure = pressure + NC * np.maximum(tsv - tb, 0.0) * min(MORTAR_LAYER / max(width, 1e-6), 1.0)
    return pressure


# --- Warp: the same per-cell laws for kernels (float64 inside, float32 out) ------------------------------------
@wp.func
def wp_tau_bulk_setting(age: float, s: float, tau0: float, a_thix: float, tau_is: float, t_is: float, m: float):
    """Concrete / repair mortar tau_b (M:37-39); s = masco_factor(T), the rest from the dataclass."""
    a = wp.max(wp.float64(age), wp.float64(0.0))
    ds = wp.float64(s)
    v = wp.float64(tau0) + wp.float64(a_thix) * a / ds
    v = v + wp.float64(tau_is) * wp.pow(a / (wp.float64(t_is) * ds), wp.float64(m))
    return wp.float32(v)


@wp.func
def wp_tau_surf_setting(tau_b: float, dens: float, r0: float, r1: float):
    """(r0 + (r1 - r0) D) tau_b (M:41-43); repair mortar r0 = 0.5, r1 = 1."""
    return wp.float32((wp.float64(r0) + (wp.float64(r1) - wp.float64(r0)) * wp.float64(dens)) * wp.float64(tau_b))


@wp.func
def wp_tau_bulk_mortar(age: float, s: float, tau0: float, tau_ref: float, t_ref: float, m: float):
    """Masonry mortar tau_b (M:98-101)."""
    a = wp.max(wp.float64(age), wp.float64(0.0)) / wp.float64(s)
    return wp.float32(wp.float64(tau0) + wp.float64(tau_ref) * wp.pow(a / wp.float64(t_ref), wp.float64(m)))


@wp.func
def wp_tau_surf_mortar(tau_b: float, compact: float, harden: float):
    """tau_b (1 + harden clip(C, 0, 1)) (M:103-105)."""
    c = wp.min(wp.max(wp.float64(compact), wp.float64(0.0)), wp.float64(1.0))
    return wp.float32(wp.float64(tau_b) * (wp.float64(1.0) + wp.float64(harden) * c))


@wp.func
def wp_bleed_rate(age: float, t_end: float, mean_kg_m2_h: float):
    """Triangle bleed rate kg/m2/s (M:55-64); t_end from Concrete.bleed_end_time."""
    a = wp.float64(age)
    te = wp.float64(t_end)
    peak = wp.float64(2.0) * wp.float64(mean_kg_m2_h) / wp.float64(3600.0)
    r = peak * (te - a) / (wp.float64(2.0) * te / wp.float64(3.0))
    if a < te / wp.float64(3.0):
        r = peak * a / (te / wp.float64(3.0))
    return wp.float32(wp.max(r, wp.float64(0.0)))


@wp.func
def wp_embed_update(E: float, pressure: float, passes: float):
    """C:1024-1025."""
    p = wp.min(wp.max(wp.float64(pressure) / wp.float64(1000.0), wp.float64(0.1)), wp.float64(20.0))
    return wp.float32(wp.float64(E) * wp.exp(-wp.float64(passes) * wp.float64(8.3) * p))


@wp.func
def wp_interface_factor(tau_s: float, film_w: float):
    """C:1096-1103 (0.05 wet, up to 0.30; log10 tau_s from 300 Pa over 2 decades)."""
    if wp.float64(film_w) > wp.float64(2.0e-5):
        return wp.float32(0.05)
    ts = wp.max(wp.float64(tau_s), wp.float64(1.0))
    x = (wp.log(ts) / wp.log(wp.float64(10.0)) - wp.log(wp.float64(300.0)) / wp.log(wp.float64(10.0))) / wp.float64(2.0)
    x = wp.min(wp.max(x, wp.float64(0.0)), wp.float64(1.0))
    return wp.float32(wp.float64(0.05) + wp.float64(0.25) * x)


@wp.func
def wp_texture_update(a: float, a_tool: float, passes: float, pressure: float, tau_s: float, m_i: float):
    """C:1034-1036; y = max(clip(p/(N_c tau_s), 0, 1), m_i)."""
    nc = wp.float64(5.141592653589793)
    y = wp.min(wp.max(wp.float64(pressure) / (nc * wp.float64(tau_s)), wp.float64(0.0)), wp.float64(1.0))
    y = wp.max(y, wp.float64(m_i))
    at = wp.float64(a_tool)
    return wp.float32(at + (wp.float64(a) - at) * wp.exp(-wp.float64(passes) * wp.float64(2.2) * y))


@wp.func
def wp_densify_update(D: float, tau_s: float, ds: float, m_i: float, tool_dens: float):
    """C:1039-1042."""
    wt = wp.float64(m_i) * wp.float64(tau_s) * wp.float64(ds) * wp.float64(tool_dens)
    return wp.float32(wp.float64(1.0) - (wp.float64(1.0) - wp.float64(D)) * wp.exp(-wt / wp.float64(112.0)))


@wp.func
def wp_seal_update(S: float, ds: float, tool_seal: float, pitch: float, pressure: float):
    """C:1049-1056."""
    pm = wp.min(wp.float64(pitch), wp.float64(0.6))
    pr = wp.max(wp.float64(pressure) / wp.float64(1000.0), wp.float64(0.1))
    rate = wp.float64(ds) * wp.float64(tool_seal) * pm * pr / wp.float64(0.10)
    return wp.float32(wp.float64(1.0) - (wp.float64(1.0) - wp.float64(S)) * wp.exp(-rate))


@wp.func
def wp_compact_update(C: float, pressure: float, ds: float, tau_bulk: float, m_i: float):
    """M:111-123 (MORTAR_LAYER 2 mm, MORTAR_STRAIN 6)."""
    tau = wp.max(wp.float64(tau_bulk), wp.float64(1.0))
    p = wp.max(wp.float64(pressure), wp.float64(0.0))
    dose = p / (p + tau) * wp.float64(m_i) * wp.max(wp.float64(ds), wp.float64(0.0)) / wp.float64(0.012)
    c = wp.min(wp.max(wp.float64(C), wp.float64(0.0)), wp.float64(1.0))
    return wp.float32(wp.float64(1.0) - (wp.float64(1.0) - c) * wp.exp(-dose))
