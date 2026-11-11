# SPDX-License-Identifier: Apache-2.0
"""Heat v3 laws for ``GridThermal`` (docs/notes/heat-v3-design.md §Laws). Each scene holds one torch body (a free
2 mm sphere, gravity off, no ground) that a law either flies at constant speed (the moving arc) or leaves idle (the
``arc_pose`` input). Each law states its region and the source of its bound.

- G-1 ``slab_flux``: a column (one cell across, insulated sides) under a uniform surface flux q0; the field after 2 s
  against T0 + (2 q0 / k) √(αt) ierfc(z / 2√(αt)) on a graded z grid (0.25 mm growing 1.2x). Bound 1 % of the surface
  rise at every cell centre (the measured flux q0 from the ledger, so the law judges conduction).
- G-2 ``stefan``: Neumann's one-phase melting of a pure substance (law_pure: T_l = T_s, the plateau H ∈ [0, L] is one
  table segment) from a face held at T_s + 500 K (a sink of 1e8 W/(m²·K)); Ste 1.2. The melted thickness Σ φ d
  against 2 λ √(α t) at 0.5, 1 and 2 s. Bound 1 % (GA E1: 0.11 %).
- G-3 ``ledger``: steel A36 T-joint (plate and bracket as two parts, 1 mm cells), a 3 kW arc at 45° into the root
  moving at 10 mm/s for 3 s: melting, fusion across the parts, convection, radiation, a table sink, the beam's
  shadows. |stored − stored0 − (arc − lost − evaporated)| <= 1e-6 of the arc's input (GA: 3.6e-8); the beam put
  95-100 % of its power into the metal (σ 2 mm, the footprint on the plate and the bracket's face); the parts fused.
- G-4 ``rosenthal``: a 1 kW Gaussian (σ 1.5 mm) at 5 mm/s on law_iron (GA E1's case), 0.5 mm cells, after 8 s;
  against the exact transient solution of a moving Gaussian surface source in the adiabatic box (method of images,
  quadrature in √θ). Region: cell centres 5, 8, 12 and 20 mm behind the source on the top row's centreline (GA E1's
  points), 4 mm beside it and 3 mm below it: rise within 3 %; and the 300 K-rise isotherm's length behind the source
  and half-width on the top layer within 3 % or one cell. The source itself (r < 3σ) is not compared.
- G-5 ``contact``: two 5 mm law_fast cubes (k 400, Biot 0.025) as two parts at 300 and 400 K behind a contact
  conductance h_c = 2000; their mean temperatures close with τ = C1 C2 / ((C1 + C2) h_c A) (WP13): measured τ within
  3 % after 4 s (the lumped model's error ~ Bi / 3); stored energy constant within 1e-9 of C ΔT.
- G-6 ``null``: uniform temperature, ambient equal, no arc: T unchanged (1e-9 K).

``busy_scene`` is the test scene for the harness's repeat checks: a small A36 T-joint (2 mm cells, so the speed check
times the hooks, not a fine heat solve), a torch at 45° flying along the root depositing wire, convection, radiation
and a table sink, 3 worlds.
"""

from __future__ import annotations

import math

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.thermal.grid import Config, graded
from alloy.plugins.thermal.grid_materials import material

LABEL = "grid_thermal"
TORCH = "torch"
PERIOD = 0.02  # s, control period = exchange step in every law scene
_DEFAULTS = {f: getattr(Config, f) for f in Config.__dataclass_fields__ if hasattr(Config, f)}


def _cfg(**kw) -> dict:
    """A complete law config: every field at its default unless given (so the plug-in under test's own settings do
    not leak into a law)."""
    out = {k: v for k, v in _DEFAULTS.items()}
    out.update(kw)
    return out


def _scene(periods: int, init, start=(0.0, 0.0, 0.05), worlds: int = 1) -> TestScene:
    def build(b: newton.ModelBuilder) -> None:
        torch = b.add_body(xform=body_xform(*start), mass=0.1, label=TORCH)
        b.add_shape_sphere(torch, radius=1.0e-3, label=TORCH)
    return TestScene(build=build, worlds=worlds, periods=periods, control_dt=PERIOD, dt=PERIOD, ground=False,
                     gravity=(0.0, 0.0, 0.0), init=init)


def _init(power: float = 0.0, sigma: float | None = None, pose=None, speed=(0.0, 0.0, 0.0), temps=None,
          ambient: float | None = None, feed: float = 0.0):
    def init(sim) -> None:
        w = sim.model.world_count
        qd = sim.state.joint_qd.numpy()
        qd.reshape(w, -1)[:, :3] = speed                    # the torch's free joint: linear velocity first
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)   # body_qd from frame 0 on
        if LABEL not in sim.plugins.plugins:                # the harness's timing run without the plug-in
            return
        grid = sim.plugins[LABEL]
        grid.set_input("arc_power", np.full(w, power, np.float32))
        grid.set_input("wire_feed", np.full(w, feed, np.float32))
        if sigma is not None:
            grid.set_input("arc_sigma", np.full(w, sigma, np.float32))
        if pose is not None:
            grid.set_input("arc_pose", np.tile(np.asarray(pose, np.float32), (w, 1)))
        if ambient is not None:
            grid.set_input("ambient", np.full(w, ambient, np.float32))
        if temps is not None:
            grid.set_temperature(temps(grid) if callable(temps) else temps)
    return init


def _final(run) -> tuple[dict, dict]:
    grid = run.sim.plugins[LABEL]
    return grid.fields(0), {k: v[0] for k, v in grid.read().items()}


def _centres(grid) -> list[np.ndarray]:
    return [0.5 * (np.asarray(e)[1:] + np.asarray(e)[:-1]) for e in (grid.config.x, grid.config.y, grid.config.z)]


# --- G-1 -----------------------------------------------------------------------------------------------------------

SLAB = _cfg(material="law_iron", x=(0.0, 2e-3), y=(0.0, 2e-3), z=graded(-0.03, 0.0, 2.5e-4, band=(-2e-3, 0.0)),
            boxes=((0.0, 0.0, -0.03, 2e-3, 2e-3, 0.0),), convection=0.0, emissivity=0.0, initial=300.0)
Q0 = 1.0e6  # W/m²; a beam σ = 1 m is flat to 1e-6 over the 2 mm column


def _ierfc(x: np.ndarray) -> np.ndarray:
    from scipy.special import erfc  # noqa: PLC0415

    return np.exp(-x * x) / math.sqrt(math.pi) - x * erfc(x)


def _check_slab(run) -> LawResult:
    fields, sums = _final(run)
    m, t = material("law_iron"), float(run.trace.t[-1])
    k, alpha = 30.0, 30.0 / (m.density * 600.0)
    q0 = sums["arc_J"] / (4e-6 * t)
    z = _centres(run.sim.plugins[LABEL])[2]
    want = 300.0 + 2.0 * q0 / k * math.sqrt(alpha * t) * _ierfc(-z / (2.0 * math.sqrt(alpha * t)))
    got = fields["temperature"][0, 0, :]
    rise = want[-1] - 300.0
    err = float(np.max(np.abs(got - want)) / rise)
    return LawResult.within(err, 0.01, f"max |T − T_exact| / surface rise ({rise:.1f} K) at t = {t:g} s, q0 measured "
                            f"{q0:.4g} W/m² (set {Q0:.4g})")


# --- G-2 -----------------------------------------------------------------------------------------------------------

STEFAN = _cfg(material="law_pure", x=(0.0, 2e-3), y=(0.0, 2e-3), z=graded(-0.012, 0.0, 2.5e-4),
              boxes=((0.0, 0.0, -0.012, 2e-3, 2e-3, 0.0),), convection=0.0, emissivity=0.0, sink=("z-",),
              sink_conductance=1.0e8, initial=1000.0, ambient=1500.0)


def neumann_lambda(stefan: float) -> float:
    """λ of Neumann's one-phase melting: λ exp(λ²) erf(λ) = Ste / √π (bisection)."""
    lo, hi = 1e-9, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * math.exp(mid * mid) * math.erf(mid) < stefan / math.sqrt(math.pi):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _check_stefan(run) -> LawResult:
    m = material("law_pure")
    alpha, ste = 30.0 / (m.density * 600.0), 600.0 * 500.0 / m.latent
    lam = neumann_lambda(ste)
    t = np.asarray(run.trace.t)
    melted = run.trace.plugin(LABEL, "liquid_m3")[:, 0] / 4e-6
    errs = []
    for when in (0.5, 1.0, 2.0):
        k = int(np.argmin(np.abs(t - when)))
        want = 2.0 * lam * math.sqrt(alpha * t[k])
        errs.append(abs(melted[k] - want) / want)
    return LawResult.within(max(errs), 0.01, f"melted thickness vs 2λ√(αt) (Ste {ste:.2f}, λ {lam:.4f}) at 0.5, 1, "
                            f"2 s: rel. errors {', '.join(f'{e:.2%}' for e in errs)}")


# --- G-3 -----------------------------------------------------------------------------------------------------------

def tee(fine: float, x1: float, sink: bool = True) -> dict:
    """A T-joint in the grid frame: plate z ∈ [−6, 0] mm, y ∈ [−12, 20] mm (part 0); bracket y ∈ [−6, 0] mm (its
    weld face at y = 0), z ∈ [0, 20] mm (part 1); the root along x at y = z = 0."""
    return dict(x=graded(0.0, x1, fine), y=graded(-0.012, 0.02, fine, band=(-0.006, 0.008)),
                z=graded(-0.006, 0.02, fine, band=(-0.006, 0.006)),
                boxes=((0.0, -0.012, -0.006, x1, 0.02, 0.0), (0.0, -0.006, 0.0, x1, 0.0, 0.02)), parts=(0, 1),
                sink=("z-",) if sink else ())


LEDGER = _cfg(**tee(1e-3, 0.04), torch=TORCH, torch_axis=(0.0, -math.sqrt(0.5), -math.sqrt(0.5)))
LEDGER_POWER, LEDGER_SPEED = 3000.0, 0.01


def _check_ledger(run) -> LawResult:
    _, s = _final(run)
    t = float(run.trace.t[-1])
    frac = s["arc_J"] / (LEDGER_POWER * t)
    if not 0.95 <= frac <= 1.0 + 1e-6:
        return LawResult(False, frac, 1.0, f"the beam put {frac:.2%} of η·V·I into the metal (want 95-100 %)")
    if s["fused_cells"] == 0 or s["t_max"] < material("steel_a36").t_liquidus:
        return LawResult(False, float(s["t_max"]), 0.0, "the arc did not melt the root")
    rel = abs(s["residual_J"]) / s["arc_J"]
    return LawResult.within(rel, 1e-6, f"|ledger residual| / arc input ({s['arc_J']:.1f} J; lost {s['lost_J']:.1f} J, "
                            f"evaporated {s['evaporated_J']:.3g} J, {int(s['fused_cells'])} cells fused, T max "
                            f"{s['t_max']:.0f} K)")


# --- G-4 -----------------------------------------------------------------------------------------------------------

ROS_BOX = ((-0.01, 0.05), (-0.015, 0.015), (-0.012, 0.0))
ROSENTHAL = _cfg(material="law_iron", x=graded(*ROS_BOX[0], 5e-4, band=(-0.002, 0.045)),
                 y=graded(*ROS_BOX[1], 5e-4, band=(-0.00625, 0.00625)), z=graded(*ROS_BOX[2], 5e-4, band=(-0.004, 0.0)),
                 boxes=((-0.01, -0.015, -0.012, 0.05, 0.015, 0.0),), convection=0.0, emissivity=0.0, initial=300.0,
                 torch=TORCH, torch_axis=(0.0, 0.0, -1.0))
ROS_POWER, ROS_SPEED, ROS_SIGMA, ROS_TIME = 1000.0, 0.005, 1.5e-3, 8.0


def _green(x: np.ndarray, xs: np.ndarray, var: np.ndarray, box: tuple[float, float], images: int = 3) -> np.ndarray:
    """1-D Green's function of an adiabatic slab [a, b] for a Gaussian source at xs of variance var: [len(x), U]."""
    a, b = box
    span = b - a
    x, xs, var = x[:, None], xs[None, :], var[None, :]
    out = np.zeros(np.broadcast_shapes(x.shape, xs.shape))
    for m in range(-images, images + 1):
        for src in (xs + 2 * m * span, 2 * a - xs + 2 * m * span):
            out += np.exp(-(x - src) ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
    return out


def moving_gaussian(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, t: float, power: float, speed: float,
                    sigma: float, alpha: float, rho_c: float, box=ROS_BOX, x0: float = 0.0, samples: int = 20000):
    """Exact temperature rise [K] on the grid xs x ys x zs at time t of a Gaussian surface source on the top face
    (z = box top) moving along x from x0 at ``speed``, in an adiabatic box, constant properties: ∫ P/(ρc) Gx Gy Gz dτ,
    with θ = t − τ = u² (the surface kernel's θ^-1/2 becomes smooth)."""
    u = (np.arange(samples) + 0.5) / samples * math.sqrt(t)
    theta, du = u * u, math.sqrt(t) / samples
    var_xy, var_z = sigma * sigma + 2.0 * alpha * theta, 2.0 * alpha * theta
    src_x = x0 + speed * (t - theta)
    gx = _green(xs, src_x, var_xy, box[0])
    gy = _green(ys, np.zeros_like(theta), var_xy, box[1])
    gz = _green(zs, np.full_like(theta, box[2][1]), var_z, box[2])
    weight = 2.0 * u * du * power / rho_c
    return np.einsum("iu,ju,ku,u->ijk", gx, gy, gz, weight)


def _iso(pos: np.ndarray, rise: np.ndarray, level: float) -> float:
    """The outermost position along ``pos`` (sorted away from the source) where ``rise`` falls through ``level``."""
    above = np.flatnonzero(rise >= level)
    if above.size == 0 or above[-1] + 1 >= len(pos):
        return float("nan")
    i = above[-1]
    f = (rise[i] - level) / (rise[i] - rise[i + 1])
    return float(pos[i] + f * (pos[i + 1] - pos[i]))


def _check_rosenthal(run) -> LawResult:
    fields, _ = _final(run)
    grid = run.sim.plugins[LABEL]
    cx, cy, cz = _centres(grid)
    t = float(run.trace.t[-1])
    alpha, rho_c = 30.0 / (7850.0 * 600.0), 7850.0 * 600.0
    xs = ROS_SPEED * t
    num = fields["temperature"] - 300.0
    ana = moving_gaussian(cx, cy, cz, t, ROS_POWER, ROS_SPEED, ROS_SIGMA, alpha, rho_c)
    iy0, iz0 = int(np.argmin(np.abs(cy))), len(cz) - 1
    errs = []
    for dx, dy, dz in ((-5e-3, 0, 0), (-8e-3, 0, 0), (-12e-3, 0, 0), (-20e-3, 0, 0), (0, 4e-3, 0), (0, 0, -3e-3)):
        i, j, k = (int(np.argmin(np.abs(c - v))) for c, v in ((cx, xs + dx), (cy, dy), (cz, cz[iz0] + dz)))
        errs.append(abs(num[i, j, k] - ana[i, j, k]) / ana[i, j, k])
    level, cell = 300.0, 5e-4
    behind = np.flatnonzero(cx <= xs)[::-1]
    length = [xs - _iso(cx[behind], f[behind, iy0, iz0], level) for f in (num, ana)]
    side = np.flatnonzero(cy >= 0.0)
    widths = [float(np.nanmax([_iso(cy[side], f[i, side, iz0], level) for i in behind[:40]])) for f in (num, ana)]
    shape_ok = all(abs(a - b) <= max(0.03 * b, cell) for a, b in (length, widths))
    detail = (f"t {t:g} s, source at x = {xs * 1e3:.1f} mm; rise errors behind 5/8/12/20 mm, beside 4 mm, below 3 mm: "
              f"{', '.join(f'{e:.2%}' for e in errs)}; {level:g} K isotherm length {length[0] * 1e3:.2f} vs "
              f"{length[1] * 1e3:.2f} mm, half-width {widths[0] * 1e3:.2f} vs {widths[1] * 1e3:.2f} mm")
    if not shape_ok:
        return LawResult(False, max(errs), 0.03, detail + " (isotherm off by more than 3 % and one cell)")
    return LawResult.within(max(errs), 0.03, detail)


# --- G-5, G-6 ------------------------------------------------------------------------------------------------------

CONTACT = _cfg(material="law_fast", x=graded(0.0, 0.01, 1e-3), y=graded(0.0, 0.005, 1e-3), z=graded(0.0, 0.005, 1e-3),
               boxes=((0.0, 0.0, 0.0, 0.005, 0.005, 0.005), (0.005, 0.0, 0.0, 0.01, 0.005, 0.005)), parts=(0, 1),
               contact=2000.0, convection=0.0, emissivity=0.0, initial=300.0)


def _split_temps(grid) -> np.ndarray:
    return np.where(grid.cell.numpy()[0] == 2, 400.0, 300.0)


def _check_contact(run) -> LawResult:
    fields, sums = _final(run)
    grid = run.sim.plugins[LABEL]
    t = float(run.trace.t[-1])
    mass = grid._mass.reshape(grid._shape)
    means = [float((fields["temperature"] * mass)[fields["cell"] == p].sum() / mass[fields["cell"] == p].sum())
             for p in (1, 2)]
    c1 = 7850.0 * 600.0 * 0.005 ** 3
    tau = c1 * c1 / ((2 * c1) * 2000.0 * 0.005 ** 2)
    tau_meas = -t / math.log((means[1] - means[0]) / 100.0)
    drift = abs(sums["stored_J"] - grid.stored0_J.numpy()[0]) / (c1 * 100.0)
    if drift > 1e-9:
        return LawResult(False, drift, 1e-9, f"stored energy drifted {drift:.2e} of C ΔT without sources")
    return LawResult.within(abs(tau_meas / tau - 1.0), 0.03, f"τ measured {tau_meas:.4f} s vs C1C2/((C1+C2) h_c A) "
                            f"= {tau:.4f} s (means {means[0]:.3f}, {means[1]:.3f} K at {t:g} s; energy drift "
                            f"{drift:.1e})")


NULL = _cfg(material="law_iron", x=graded(0.0, 0.004, 1e-3), y=graded(0.0, 0.003, 1e-3), z=graded(0.0, 0.002, 1e-3),
            boxes=((0.0, 0.0, 0.0, 0.004, 0.003, 0.002),), initial=500.0, ambient=500.0, sink=("z-",))


def _check_null(run) -> LawResult:
    fields, _ = _final(run)
    return LawResult.within(float(np.abs(fields["temperature"] - 500.0).max()), 1e-9, "max |T − 500 K| at rest")


def busy_scene() -> TestScene:
    """The harness's repeat-check scene; ``BUSY`` is its plug-in config."""
    return _scene(25, _init(power=400.0, sigma=2e-3, speed=(0.01, 0.0, 0.0), feed=0.05), start=(0.002, 0.004, 0.004),
                  worlds=3)


BUSY = _cfg(**tee(2e-3, 0.012), torch=TORCH, torch_axis=(0.0, -math.sqrt(0.5), -math.sqrt(0.5)), arc_sigma=2e-3)


def grid_laws() -> list[Law]:
    up = (0.0, 0.0, 0.01, 0.0, 0.0, -1.0)
    return [
        Law("slab_flux", _check_slab, scene=_scene(100, _init(power=Q0 * 2.0 * math.pi, sigma=1.0, pose=(1e-3, 1e-3,
            0.01, 0.0, 0.0, -1.0))), config=SLAB, doc="G-1: surface flux into a graded column vs ierfc, 1 %"),
        Law("stefan", _check_stefan, scene=_scene(100, _init()), config=STEFAN,
            doc="G-2: Neumann one-phase melting of a pure substance, front within 1 % (GA E1: 0.11 %)"),
        Law("ledger", _check_ledger, scene=_scene(150, _init(power=LEDGER_POWER, sigma=2e-3,
            speed=(LEDGER_SPEED, 0.0, 0.0)), start=(0.005, 0.01, 0.01)), config=LEDGER,
            doc="G-3: energy ledger closes within 1e-6 of the arc input (GA: 3.6e-8)"),
        Law("rosenthal", _check_rosenthal, scene=_scene(round(ROS_TIME / PERIOD), _init(power=ROS_POWER,
            sigma=ROS_SIGMA, speed=(ROS_SPEED, 0.0, 0.0)), start=(0.0, 0.0, 0.01)), config=ROSENTHAL,
            doc="G-4: moving Gaussian vs the exact transient field, >= 3 sigma from the source, 3 %"),
        Law("contact", _check_contact, scene=_scene(200, _init(temps=_split_temps)), config=CONTACT,
            doc="G-5: two parts relax with tau = C1 C2 / ((C1 + C2) h_c A) within 3 % (WP13)"),
        Law("null", _check_null, scene=_scene(10, _init(pose=up)), config=NULL, doc="G-6: uniform T stays put"),
    ]


__all__ = ["BUSY", "busy_scene", "grid_laws", "moving_gaussian", "neumann_lambda", "tee"]
