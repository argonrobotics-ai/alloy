# SPDX-License-Identifier: Apache-2.0
"""Heat v2 surface laws for ``ParticleThermal`` (WP13 review of step 4): the convection area and the heater coupling.

- H-5g ``convection_area``: one world holds six bodies, apart: an 8 × 8 × 6 block on a plate (a heater at the start
  temperature), a lone particle, a one-layer 8 × 8 sheet, a 12³ block jittered by ±0.1 spacings, a uniformly random
  cloud of 12³ particles in a 12-spacing box and a uniformly random 16 × 16 × 3 puddle (a spread melt's shape; seeded;
  XPBD particle contacts off). Exposure (``particle_area``) in footprints s²: face columns on every axis sum to 1 per
  face (1 %; the 62 view directions have the cube's symmetry, 0.000); a top edge particle reads 2 within 15 % (2.08)
  and a top corner 3 within 20 % (3.35); the block totals its box within 10 % (1.04), the jittered block too (0.96,
  nothing deep), the random cloud 0.9-1.25 of its box (1.03) with at most 3 % three or more spacings inside (2.0 %),
  the puddle its box within 10 % (1.00); the lone particle reads at least 0.8 of its 6 faces, the sheet 0.8-1.5 of its
  area (1.18); and the plate's contact face is not open to the air (its interior particles exchange at most 5 % of
  what the open top face's do: 0).
- H-5h ``plate_vs_slab``: a 12 × 12 × 6 water bar (1 mm particles) on a 353.15 K plate (h = 1000 W/(m²·K)), time
  scale 100. Heat in per contact area against the exact plane wall with one Robin face and one insulated face (series
  in λ_n tan λ_n = Bi, Carslaw and Jaeger §3.11; Bi = hL/k = 9.9), within 3 % for t ≥ 1.5 s²/α (10 s). The heater's
  reported contact area is the 144 footprints under the bar (1e-9). Without the half cell (h_eff = 1/(1/h + s/(2k)))
  the bar took heat 1.21-1.39x too fast (WP13). Measured: −2.0 % at 12 s, +2.6 % at 100 s (Fo 0.4), the same on a
  20 × 20 bar, so not the side faces: the bar's 6 layers and the free-face correction across its top and plated faces.
- H-5i ``film_contact`` (WP13 re-reviews R1 and r3): on a 333.15 K plate with a 2-spacing gap,
  - a two-layer film, 12 × 12 columns jittered ±0.2 spacings across and each lifted 0-1 spacing (MPM holds particles
    anywhere within a voxel of a collider);
  - a close-packed bilayer;
  - an 8 × 8 layer with 2 rows past the plate's edge;
  - a particle of 8x mass with 4 small ones resting on it.
  Checks: the touching area is the films' base (0.9-1.05; 1.004); the upper layers take at most 6 % / 10 % of the
  first period's plate heat (0.045 / 0.067; a yes/no 35° cone gave the close-packed one 0.43); every bottom particle
  on the plate takes heat, and none past the edge or on the larger particle does (both counted 0.74 a footprint
  before); the bilayers end within 1 % of the plate temperature (0.5 %).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from scipy.optimize import brentq

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.thermal.particle_laws import LABEL, RADIUS, SPACING, SUPPORT, bar_scene
from alloy.plugins.thermal.phase_materials import material

# H-5g layout: (name, lattice counts, x offset in spacings); jitter only for "jittered"
BODIES = (("block", (8, 8, 6), 0.0), ("lone", (1, 1, 1), 20.0), ("sheet", (8, 8, 1), 30.0),
          ("jittered", (12, 12, 12), 50.0), ("random", (12, 12, 12), 75.0), ("puddle", (16, 16, 3), 100.0))
JITTER, SEED = 0.1, 7
T5G, AIR5G = 300.0, 310.0


def _bodies(b: newton.ModelBuilder) -> None:
    mass = 997.05 * SPACING ** 3
    rng = np.random.default_rng(SEED)
    for name, (nx, ny, nz), x0 in BODIES:
        pts = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
        pts = pts.astype(float) + (rng.uniform(-JITTER, JITTER, pts.shape) if name == "jittered" else 0.0)
        if name == "random":                                      # uniform random cloud: a melt's disorder, extreme
            pts = rng.uniform(-0.5, 11.5, pts.shape)
        if name == "puddle":                                      # a thin random layer: a spread melt's shape
            pts = rng.uniform([-0.5, -0.5, -0.5], [15.5, 15.5, 2.5], pts.shape)
        for p in pts:
            b.add_particle(wp.vec3(*((p + [x0, 0.0, 0.0]) * SPACING + [0.0, 0.0, 0.02])), wp.vec3(0.0, 0.0, 0.0),
                           mass, RADIUS)
    top = 0.02 - RADIUS - 1.0e-4                                   # the plate under the block, 0.1 mm clear
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(4.0 * SPACING, 3.5 * SPACING, top - 2.0e-3), wp.quat_identity()),
                    hx=4.0 * SPACING + 1.0e-3, hy=4.0 * SPACING + 1.0e-3, hz=2.0e-3, label="plate")


def _parts(a: np.ndarray, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out, start = {}, 0
    for name, (nx, ny, nz), _ in BODIES:
        n = nx * ny * nz
        out[name] = (a[start: start + n], q[start: start + n] / SPACING)
        start += n
    return out


def _box(q: np.ndarray) -> float:
    n = np.round(q.max(axis=0) - q.min(axis=0)) + 1.0
    return float(2.0 * (n[0] * n[1] + n[1] * n[2] + n[0] * n[2]))


def _block_checks(a: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Relative errors (over their bounds) on the plated block: face columns, a top edge, a top corner, the total."""
    lo, hi = q.min(axis=0), q.max(axis=0)
    cols = []
    for axis in range(3):                                         # face columns along every axis (WP13 re-review b)
        side = [k for k in range(3) if k != axis]
        centre = np.all(np.abs(q[:, side] - 0.5 * (lo + hi)[side]) < 0.6, axis=1)
        for key in np.unique(q[centre][:, side], axis=0):
            cols.append(a[centre & np.all(np.abs(q[:, side] - key) < 0.1, axis=1)].sum() / 2.0)  # 2 faces per column
    edge = a[(np.abs(q[:, 0] - lo[0] - 4.0) < 0.1) & (q[:, 1] == hi[1]) & (q[:, 2] == hi[2])][0]
    corner = a[np.all(q == hi, axis=1)][0]
    return {"face": max(abs(c - 1.0) for c in cols) / 0.01, "edge": abs(edge / 2.0 - 1.0) / 0.15,
            "corner": abs(corner / 3.0 - 1.0) / 0.2, "block": abs(a.sum() / _box(q) - 1.0) / 0.1}


def _cloud(a: np.ndarray, q: np.ndarray, box: float | None = None) -> tuple[float, float]:
    """(share of the exposure three or more spacings inside, total / box area) of a 12³-sized cloud."""
    lo, hi = q.min(axis=0), q.max(axis=0)
    deep = np.all((q > lo + 2.5) & (q < hi - 2.5), axis=1)
    return float(a[deep].sum() / a.sum()), float(a.sum() / (box if box is not None else _box(q)))


def _check_exposure(run) -> LawResult:
    therm = run.sim.plugins[LABEL]
    q = run.sim.model.particle_q.numpy()[therm._gidx.numpy()].reshape(therm._W, therm.P, 3)[0]
    parts = _parts(run.trace.plugin(LABEL, "exposure")[-1, 0] / SPACING ** 2, q)
    err = _block_checks(*parts["block"])
    lone = float(parts["lone"][0][0])
    sheet = float(parts["sheet"][0].sum() / _box(parts["sheet"][1]))
    deep_share, jtotal = _cloud(*parts["jittered"])
    rdeep, rtotal = _cloud(*parts["random"], box=12.0 ** 2 * 6.0)
    ptotal = float(parts["puddle"][0].sum() / (2.0 * (16.0 * 16.0 + 2.0 * 16.0 * 3.0)))
    cooled = run.trace.plugin(LABEL, "cooled")[-1, 0][: parts["block"][0].size]
    qb = parts["block"][1]
    inside = np.all((qb[:, :2] > qb[:, :2].min(0) + 1.5) & (qb[:, :2] < qb[:, :2].max(0) - 1.5), axis=1)
    plated = cooled[inside & (qb[:, 2] == qb[:, 2].min())].mean() / cooled[inside & (qb[:, 2] == qb[:, 2].max())].mean()
    err.update(lone=max(0.0, 0.8 - lone / 6.0) / 0.01, sheet=max(0.0, 0.8 - sheet, sheet - 1.5) / 0.01,
               deep=deep_share / 0.01, jittered=abs(jtotal - 1.0) / 0.1, plate=abs(plated) / 0.05,
               random=max(0.0, 0.9 - rtotal, rtotal - 1.25) / 0.01, random_deep=rdeep / 0.03,
               puddle=abs(ptotal - 1.0) / 0.1)
    worst = max(err, key=err.get)
    return LawResult.within(err[worst], 1.0,
                            f"worst {worst}; lone {lone:.2f} of 6 s², sheet {sheet:.3f} of its area, jittered total "
                            f"{jtotal:.3f} of its box with {deep_share:.4f} deep, random cloud {rtotal:.3f} with "
                            f"{rdeep:.4f} deep, thin random puddle {ptotal:.3f} of its box, plated / open face air "
                            f"exchange {plated:.4f} (5 %); errors / bounds "
                            f"{ {k: round(float(v), 3) for k, v in err.items()} }",
                            "particle_area: exposure = min(fill estimate, cone estimate); on_heater removes the "
                            "contact face")


def _still(sim) -> None:
    """XPBD particle-particle contacts off: the random cloud's overlapping pairs would push it apart."""
    sim.model.particle_max_radius = 0.0
    sim.model.particle_grid = None


def _shapes_scene() -> TestScene:
    return TestScene(build=_bodies, worlds=1, periods=2, control_dt=0.02, dt=1.0e-3, solver="xpbd", ground=False,
                     gravity=(0.0, 0.0, 0.0), init=_still)


# --- H-5h -----------------------------------------------------------------------------------------------------------

NXY5H, NZ5H, T0_5H, TW_5H, SCALE_5H = 12, 6, 293.15, 353.15, 100.0


def slab_heat(t: np.ndarray, h: float, k: float, rho_c: float, length: float, terms: int = 200) -> np.ndarray:
    """Heat in per area [J/m²] of a plane wall (thickness ``length``, insulated far face) whose near face meets a medium
    ΔT = 1 K hotter through h: ρ c L (1 − Σ C_n (sin λ_n / λ_n) e^{−λ_n² Fo}), λ_n tan λ_n = Bi."""
    bi, alpha = h * length / k, k / rho_c
    lam = np.array([brentq(lambda x: x * np.tan(x) - bi, (n - 1) * np.pi + 1e-12, (n - 0.5) * np.pi - 1e-12)
                    for n in range(1, terms + 1)])
    c = 4.0 * np.sin(lam) / (2.0 * lam + np.sin(2.0 * lam))
    fo = alpha * np.asarray(t, np.float64)[:, None] / length ** 2
    return rho_c * length * (1.0 - (c * np.sin(lam) / lam * np.exp(-lam ** 2 * fo)).sum(axis=1))


def _check_slab(run) -> LawResult:
    m = material("water")
    t = np.asarray(run.trace.t) * SCALE_5H
    area_rep = run.trace.plugin(LABEL, "heater_area_m2")[-1, 0, 0]
    area = NXY5H * NXY5H * SPACING ** 2
    want = slab_heat(t, 1000.0, m.k_liquid, m.density * m.c_liquid, NZ5H * SPACING) * (TW_5H - T0_5H) * area
    got = run.trace.plugin(LABEL, "heat_in_J")[:, 0]
    late = t >= 1.5 * SPACING ** 2 / m.diffusivity(liquid=True)
    rel = got[late] / want[late] - 1.0
    worst = float(rel[np.argmax(np.abs(rel))])
    area_err = abs(area_rep / area - 1.0)
    return LawResult.within(max(abs(worst) / 0.03, area_err / 1.0e-9), 1.0,
                            f"heat in / exact Robin slab, t {t[late][0]:.1f}-{t[-1]:.0f} s: worst {worst:+.4f} (3 %); "
                            f"at {t[late][0]:.1f} s {rel[0]:+.4f}, at the end {rel[-1]:+.4f}; reported contact area "
                            f"{area_rep:.6g} m² vs {area:.6g} m² (1e-9)",
                            "heaters act through h_eff = 1/(1/h + s/(2k)) (surface_conductance)")


# --- H-5i -----------------------------------------------------------------------------------------------------------

N5I, LIFT5I, GAP5I, T0_5I, TW_5I, SCALE5I = 12, 1.0, 2.0, 293.15, 333.15, 100.0
X5I, EDGE5I = 2.0 * N5I + 8.0, 14.0  # spacings: the third and fourth films' x; the plate's +y edge


HCP_A = 2.0 ** (1.0 / 6.0)  # close-packed in-plane spacing [spacings]: a³/√2 = s³ per particle


def _hcp_bilayer(x0: float) -> tuple[np.ndarray, np.ndarray]:
    """[n, 3] points (spacings) of two close-packed layers, the upper one in the lower one's hollows; is-upper mask."""
    a, rows = HCP_A, int(N5I / (HCP_A * np.sqrt(3.0) / 2.0))
    pts, upper = [], []
    for k in range(2):
        for j in range(rows):
            for i in range(int(N5I / a)):
                x = i * a + (j % 2) * 0.5 * a + k * 0.5 * a
                y = j * a * np.sqrt(3.0) / 2.0 + k * a / (2.0 * np.sqrt(3.0))
                pts.append([x0 + x, y, k * a * np.sqrt(2.0 / 3.0)])
                upper.append(k == 1)
    return np.asarray(pts), np.asarray(upper)


def _film_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(points [spacings, centres above the plate face], is-upper, kind, mass factor) of the H-5i films. Kind 0: the
    lifted bilayer; 1: the close-packed bilayer; 2: an 8 × 8 layer whose last 2 rows hang past the plate's +y edge;
    3: one particle of 8x mass (a 2-spacing footprint) with 4 small ones resting on it (WP13 re-review r3)."""
    rng = np.random.default_rng(SEED)
    lift = rng.uniform(0.0, LIFT5I, (N5I, N5I))
    lifted = [np.array([ix, iy, iz + lift[ix, iy]]) + rng.uniform(-0.2, 0.2, 3) * [1.0, 1.0, iz]
              for ix in range(N5I) for iy in range(N5I) for iz in range(2)]
    hcp, hcp_upper = _hcp_bilayer(N5I + 4.0)
    over = [[X5I + ix, EDGE5I - 5.5 + iy, 0.0] for ix in range(8) for iy in range(8)]   # rows at EDGE + 0.5, + 1.5
    mixed = [[X5I + 10.5, 2.5, 0.5]] + [[X5I + 10.5 + dx, 2.5 + dy, 2.0] for dx in (-0.35, 0.35)
                                         for dy in (-0.35, 0.35)]
    pts = np.concatenate([np.asarray(lifted), hcp, np.asarray(over), np.asarray(mixed)])
    kind = np.repeat([0, 1, 2, 3], [len(lifted), len(hcp), len(over), len(mixed)])
    upper = np.concatenate([np.arange(len(lifted)) % 2 == 1, hcp_upper, np.zeros(len(over), bool),
                            np.arange(len(mixed)) > 0])
    mass = np.where((kind == 3) & ~upper, 8.0, 1.0)
    return pts, upper, kind, mass


def _film(b: newton.ModelBuilder) -> None:
    """Spread films on one plate (``_film_points``): MPM holds particles anywhere within a voxel of a collider (WP13
    re-review R1), so contact must not count the particles above the bottom ones, nor footprints past the edge."""
    mass, top = 997.05 * SPACING ** 3, 0.02
    pts, _, _, factor = _film_points()
    for p, f in zip(pts, factor):
        r = RADIUS if f == 1.0 else 2.0 * RADIUS
        b.add_particle(wp.vec3(*(p * SPACING + [0.0, 0.0, top + r])), wp.vec3(0.0, 0.0, 0.0), mass * f, r)
    x_hi, y_hi = (X5I + 13.0) * SPACING, EDGE5I * SPACING
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.5 * x_hi - 1.0e-3, 0.5 * y_hi - 1.0e-3, top - 2.0e-3),
                                           wp.quat_identity()),
                    hx=0.5 * x_hi + 1.0e-3, hy=0.5 * y_hi + 1.0e-3, hz=2.0e-3, label="plate")


def _check_film(run) -> LawResult:
    first = run.trace.plugin(LABEL, "heated")[1, 0]  # one period in, all near T0: plate heat ∝ touching area
    heated = run.trace.plugin(LABEL, "heated")[-1, 0]
    t_end = run.trace.plugin(LABEL, "temperature")[-1, 0]
    pts, upper, kind, _ = _film_points()
    packed = kind == 1
    lo, hi = pts[packed, :2].min(axis=0) + 2.0, pts[packed, :2].max(axis=0) - 2.0   # away from the patch's edges, where
    inner = packed & np.all((pts[:, :2] > lo) & (pts[:, :2] < hi), axis=1)          # an upper particle overhangs
    past = (kind == 2) & (pts[:, 1] > EDGE5I)
    base_m2 = (N5I * N5I + (~upper & packed).sum() * np.sqrt(3.0) / 2.0 * HCP_A ** 2 + 48.0 + 4.0) * SPACING ** 2
    up_lat = float(first[upper & (kind == 0)].sum() / first[kind == 0].sum())
    up_hcp = float(first[upper & inner].sum() / first[inner].sum())
    area = float(run.trace.plugin(LABEL, "heater_area_m2")[-1, 0, 0] / base_m2)
    on = ~upper & ~past
    base = float((heated[on] > 0.0).mean())
    wrong = int((first[past] > 0.0).sum() + (first[upper & (kind == 3)] > 0.0).sum())
    end = float(np.abs(t_end[kind < 2] - TW_5I).max() / (TW_5I - T0_5I))  # the bilayers (the rest test contact only)
    err = {"area": max(0.0, 0.9 - area, area - 1.05) / 0.01, "upper": up_lat / 0.06, "upper_packed": up_hcp / 0.1,
           "base": max(0.0, 0.99 - base) / 0.01, "wrong": 2.0 * wrong, "end": end / 0.01}
    worst = max(err, key=err.get)
    return LawResult.within(err[worst], 1.0,
                            f"worst {worst}; touching area {area:.3f} of the films' base (0.9-1.05; a close-packed "
                            f"bottom alone is 0.92 of its base); upper layers' share of the first period's plate heat: "
                            f"lifted film {up_lat:.4f} (6 %), close-packed {up_hcp:.4f} (10 %; a 35° cone: 0.43); "
                            f"bottom particles heated {base:.3f} (0.99); particles past the edge or on a larger one "
                            f"heated by the plate {wrong} (0); end |T - T_plate| <= {end:.2e} of the start difference "
                            f"(1 %; the bilayers)",
                            "particle_kernels.touch_share: the footprint share over the face that no lower particle "
                            "covers")


def area_laws() -> list[Law]:
    water = {"materials": ("water",), "support": SUPPORT}
    return [
        Law("convection_area", _check_exposure, scene=_shapes_scene(),
            config={**water, "initial": (T5G,), "ambient": AIR5G, "convection": 10.0, "heaters": ("plate",),
                    "heater_modes": ("temperature",), "heater_values": (T5G,), "heater_gap": 2.0e-4},
            doc="H-5g: exposure on a plated block, a lone particle, a sheet and a jittered block: faces 1 %, edges "
                "15 %, corners 20 %, totals 10 %, deep ≤ 1 %, no air on the plated face"),
        Law("plate_vs_slab", _check_slab, scene=bar_scene(NXY5H, NXY5H, NZ5H, periods=50, worlds=1, rho=997.05,
                                                          plate=True),
            config={**water, "initial": (T0_5H,), "heaters": ("plate",), "heater_modes": ("temperature",),
                    "heater_values": (TW_5H,), "heater_gap": 2.0e-4, "time_scale": SCALE_5H},
            doc="H-5h: a water bar on a plate takes the exact Robin slab's heat within 3 % for t >= 1.5 s²/α"),
        Law("film_contact", _check_film,
            scene=TestScene(build=_film, worlds=1, periods=50, control_dt=0.02, dt=1.0e-3, solver="xpbd", ground=False,
                            gravity=(0.0, 0.0, 0.0), init=_still),
            config={**water, "initial": (T0_5I,), "heaters": ("plate",), "heater_modes": ("temperature",),
                    "heater_values": (TW_5I,), "heater_gap": GAP5I * SPACING, "time_scale": SCALE5I},
            doc="H-5i: spread two-layer films (lifted columns; close-packed) on a plate with a 2-spacing gap: the "
                "touching area is their base, the upper layers take at most 6 % / 10 % of the plate heat, nothing past "
                "the edge or on a larger particle touches, and they end "
                "at the plate temperature"),
    ]


__all__ = ["area_laws", "slab_heat"]
