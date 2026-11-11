# SPDX-License-Identifier: Apache-2.0
"""Laws of the SPH-VBD coupler's light shells (engine item 3, P2; soft_inertia.py; handoff notes/wp9-p2-handoff.md).

Each runner builds its scene, runs it wet (the VBD <-> SPH Scheduler, staggered with Aitken) or dry (VBD alone, the
analytic fluid load as particle_f, the added mass in the plate) and returns a metrics dict; ``soft_inertia=False``
runs P1's prescribed walls for the P1 / P2 comparison. Both report passes per exchange (mean, max) and wall time.

- L6 ``spring_plate``: a rigid sheet on springs is the floor of a square channel under a liquid column of height L
  (free surface on top). The column moves with the sheet: its added mass is m_a = rho L A_p^2 / A_t (A_p the sheet's
  area, A_t the channel's; the column's kinetic energy at sheet speed v), so omega = sqrt(k / (m + m_a)). The springs
  hold the sheet's own weight at the start; the column's weight rho g L A_p arrives as a step load at t = 0, so the
  sheet oscillates about its loaded rest with amplitude rho g L A_p / k. Wet against the formula and against the dry
  twin (sheet mass m + m_a, the step load as particle_f); the start transient is the first half period, with F_used
  zero at t = 0 (soft_inertia.py). Ratios m_a / m = 1, 10, 50 at one period (``period``).
- L7 ``cloth_floor``: a clamped cloth (``n`` x ``n`` cells, areal density sigma) is the floor of a tank under a liquid
  column H (WP9's cloth probe, notes/wp9-runs/svbd/cloth_probe.py): m_a / m_s ~ rho H / sigma. Its centre sag over
  the second half, wet against a dry twin whose interior vertices carry rho g H s^2; liquid particles below the cloth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from alloy._vendor import newton
import numpy as np
import warp as wp

from .build import add_box_tank, add_fluid_points, lattice_points
from .config import DFSPHConfig, register_custom_attributes
from .soft_scenes import SOFT_CONTACT_GAP, SoftPool, soft_pipeline, vbd_sph_scheduler

RHO_F, G = 1000.0, 9.81


@dataclass
class ShellRig:
    """One L6 / L7 scene: the model, the VBD particles, the proxies, the liquid and the external forces."""

    model: newton.Model
    config: DFSPHConfig
    liquid: range
    vbd: list[int]
    proxies: list[int]
    probe: np.ndarray  # VBD particle ids whose mean z is the measured coordinate
    load: np.ndarray = None  # dry twin: (particle_count, 3) particle_f [N]
    springs: tuple | None = None  # L6: (vertex ids, anchor points (n, 3), stiffness each) as particle_f
    info: dict = field(default_factory=dict)

    def external(self, q: np.ndarray) -> np.ndarray | None:
        """particle_f for positions q: the dry twin's load plus the anchor springs, evaluated at the exchange start."""
        if self.load is None and self.springs is None:
            return None
        f = np.zeros((self.model.particle_count, 3), np.float32) if self.load is None else self.load.copy()
        if self.springs is not None:
            ids, anchor, ke = self.springs
            f[ids] += -ke * (q[ids].astype(np.float64) - anchor)
        return f


def _builder() -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    register_custom_attributes(b)
    b.default_shape_cfg.mu = 0.0
    return b


def _finalize(b: newton.ModelBuilder, device) -> newton.Model:
    b.end_world()
    b.color()
    m = b.finalize(device=device)
    m.soft_contact_mu = 0.0
    return m


def _vertex_shares(tri: np.ndarray, x: np.ndarray, count: int) -> np.ndarray:
    """Each vertex's share of a uniform pressure on the triangles (area / 3 per incident triangle), normalised."""
    area = 0.5 * np.linalg.norm(np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]]), axis=1)
    share = np.zeros(count)
    for c in range(3):
        np.add.at(share, tri[:, c], area / 3.0)
    return share / share.sum()


def spring_plate(ratio: float, *, wet: bool = True, n: int = 8, L: int = 8, period: float = 0.06, cells: int = 4,
                 device: str | None = None) -> ShellRig:
    """L6 scene (module doc): sheet mass m = m_a / ratio, spring constant k = (2 pi / period)^2 (m + m_a).

    The sheet is a cloth grid (``cells`` per side, membrane only, no pinned vertex) whose vertex k carries mass
    m share_k and a linear isotropic spring k share_k to a fixed point, share_k its part of a uniform pressure on the
    triangles (area / 3 per incident triangle). A uniform load then accelerates every vertex alike, so the sheet moves
    as a rigid plate with no elasticity to converge. A stiff tet plate does not work here: VBD's sweep moves a light
    stiff body only part of the way each step (its translation converged 42 % in 200 iterations at m_a / m = 10 with
    pinned spring anchors, which also block SolverVBDMomentum's exact translation step; free, its rotation still
    drifted into a 5 mm tilt in 30 ms). The springs act as particle_f at the exchange start (symplectic Euler), so the
    component is free. The sheet spans the channel (no gap for liquid to pass at its edges: a 0.5 mm gap let it
    through within 30 ms) and VBD gets no wall contacts (``info['contact_gap']`` None), so no record blocks the
    translation step; the lateral springs keep the sheet in place."""
    cfg = DFSPHConfig(particle_radius=1.0e-3)
    s = cfg.spacing
    c, radius = 0.0, 0.05 * s  # sheet clearance from the channel walls; VBD vertex radius
    side = n * s - 2.0 * c
    a_t, a_p = (n * s) ** 2, side**2
    m_a = RHO_F * L * s * a_p**2 / a_t
    m_sheet = m_a / float(ratio) + (0.0 if wet else m_a)
    k = (2.0 * np.pi / period) ** 2 * (m_a / float(ratio) + m_a)
    z_sheet = 3.0 * s
    b = _builder()
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (n * s, n * s, z_sheet + (L + 6) * s), wall=cfg.h)
    p0 = b.particle_count
    b.add_cloth_grid(pos=wp.vec3(c, c, z_sheet), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=cells,
                     dim_y=cells, cell_x=side / cells, cell_y=side / cells, mass=1.0, tri_ke=1.0e-3, tri_ka=1.0e-3,
                     tri_kd=0.0, edge_ke=0.0, particle_radius=radius)
    sheet = list(range(p0, b.particle_count))
    liq = add_fluid_points(b, cfg, lattice_points((0.0, 0.0, z_sheet), (n, n, L), s)) if wet else range(0)
    model = _finalize(b, device)
    q0 = model.particle_q.numpy().astype(np.float64)
    tri = model.tri_indices.numpy()
    share = _vertex_shares(tri - p0, q0[sheet], len(sheet))
    mass = model.particle_mass.numpy()
    mass[sheet] = m_sheet * share
    model.particle_mass.assign(mass)
    inv = model.particle_inv_mass.numpy()
    inv[sheet] = 1.0 / mass[sheet]
    model.particle_inv_mass.assign(inv)
    ids = np.asarray(sheet)
    anchor = q0[ids] + np.array([0.0, 0.0, m_sheet * G / k])  # the springs hold the sheet's weight at the start
    ke = (k * share)[:, None]
    load = None
    w_col = RHO_F * G * L * s * a_p
    if not wet:
        load = np.zeros((model.particle_count, 3), np.float32)
        load[ids, 2] = -w_col * share
    info = {"m_added": m_a, "m_plate": m_a / float(ratio), "k": k, "ratio": float(ratio),
            "omega_ref": float(np.sqrt(k / (m_a / float(ratio) + m_a))), "amp_ref": w_col / k, "a_p": a_p,
            "a_t": a_t, "z0": float(z_sheet), "vbd_iterations": 20, "contact_gap": None}
    return ShellRig(model, cfg, liq, sheet, sheet, ids, load, springs=(ids, anchor, ke), info=info)


def cloth_floor(sigma: float, *, wet: bool = True, n: int = 12, H: int = 8, tri_ke: float = 100.0,
                device: str | None = None) -> ShellRig:
    """L7 scene (module doc; WP9's probe): a clamped cloth of areal density sigma [kg/m^2] under H cells."""
    cfg = DFSPHConfig(particle_radius=1.0e-3)
    s = cfg.spacing
    z0 = 6 * s
    b = _builder()
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (n * s, n * s, z0 + (H + 8) * s), wall=cfg.h)
    p0 = b.particle_count
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, z0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=n,
                     dim_y=n, cell_x=s, cell_y=s, mass=sigma * s * s, fix_left=True, fix_right=True, fix_top=True,
                     fix_bottom=True, tri_ke=tri_ke, tri_ka=tri_ke, tri_kd=1.0e-3, edge_ke=1.0e-6,
                     particle_radius=0.25 * s)
    cloth = list(range(p0, b.particle_count))
    liq = add_fluid_points(b, cfg, lattice_points((0.0, 0.0, z0), (n, n, H), s)) if wet else range(0)
    model = _finalize(b, device)
    rest = model.particle_q.numpy()[cloth].astype(np.float64)
    centre = np.array([cloth[int(np.argmin(np.linalg.norm(rest[:, :2] - n * s / 2, axis=1)))]])
    load = None
    if not wet:
        load = np.zeros((model.particle_count, 3), np.float32)
        load[cloth, 2] = -RHO_F * G * H * s * s * s  # one cell per vertex (the edges are pinned)
    info = {"ratio": RHO_F * H * s / sigma, "z0": float(rest[centre - p0, 2].mean()), "sigma": sigma}
    return ShellRig(model, cfg, liq, cloth, cloth, centre, load, info=info)


def _stepper(rig: ShellRig, wet: bool, soft_inertia: bool, added_mass: float, half: float, iterations: int):
    from alloy.solvers.vbd import make_vbd

    if not wet:
        return make_vbd(rig.model, {"iterations": iterations})
    pool = SoftPool(rig.model, rig.config, rig.liquid, rig.vbd, rig.proxies)
    return vbd_sph_scheduler(pool, added_mass=added_mass, vbd_iterations=iterations,
                             sph_kwargs={"soft_inertia": soft_inertia, "soft_half_thickness": half})


def run_rig(rig: ShellRig, *, T: float, dt_x: float = 1.0e-3, soft_inertia: bool = True, added_mass: float = 0.0,
            half: float = 0.0, log=None, tag: str = "") -> dict:
    """Step the rig for T [s]; record the probe's mean z, passes per exchange and liquid below the shell."""
    wet = len(rig.liquid) > 0
    stepper = _stepper(rig, wet, soft_inertia, added_mass, half, int(rig.info.get("vbd_iterations", 20)))
    st, nx = rig.model.state(), rig.model.state()
    gap = rig.info.get("contact_gap", SOFT_CONTACT_GAP)
    pipe = soft_pipeline(rig.model, gap) if gap is not None else None
    con = pipe.contacts() if pipe is not None else None
    steps = int(round(T / dt_x))
    z, passes, below, t0 = [], [], 0, time.time()
    lo, hi = rig.liquid.start, rig.liquid.stop
    for k in range(steps):
        st.clear_forces()
        f = rig.external(st.particle_q.numpy())
        if f is not None:
            st.particle_f.assign(f)
        if pipe is not None:
            pipe.collide(st, con)
        stepper.step(st, nx, None, con, dt_x)
        st, nx = nx, st
        q = st.particle_q.numpy()
        z.append(float(q[rig.probe, 2].mean()))
        if wet:
            passes.append(int(stepper._loop.iters_used.numpy()[0]))
            below = max(below, int((q[lo:hi, 2] < q[rig.proxies, 2].min() - 1.0e-4).sum()))
        if log is not None and (k + 1) % max(steps // 10, 1) == 0:
            log(f"{tag} t={(k + 1) * dt_x:.3f} dz {(z[-1] - rig.info['z0']) * 1e3:+.4f} mm, passes "
                f"{np.mean(passes[-50:]) if passes else 0:.2f}, below {below}, wall {time.time() - t0:.0f}s")
    q = st.particle_q.numpy()
    return {"t": dt_x * np.arange(1, steps + 1), "dz": np.array(z) - rig.info["z0"],
            "passes_mean": float(np.mean(passes)) if passes else 0.0,
            "passes_max": int(np.max(passes)) if passes else 0, "below": below,
            "finite": bool(np.isfinite(q).all()), "wall_s": time.time() - t0}


def oscillation(t: np.ndarray, x: np.ndarray, omega_hint: float) -> tuple[float, float]:
    """(omega, first trough) of an oscillation about 0 sampled at t: crossings of 0 (linear interpolation; one closer
    than an eighth of the hinted period to the last is a wiggle of the plate's own vibration and skipped), omega from
    the mean spacing of crossings two apart (whole periods, so an offset of the rest cancels); |min x| before the
    second crossing."""
    gap = 0.25 * np.pi / omega_hint
    tc = []
    for j in np.nonzero(np.sign(x[1:]) != np.sign(x[:-1]))[0]:
        tj = t[j] + (t[j + 1] - t[j]) * x[j] / (x[j] - x[j + 1])
        if not tc or tj - tc[-1] > gap:
            tc.append(tj)
    if len(tc) < 3:
        return float("nan"), float("nan")
    tc = np.array(tc)
    period = float(np.mean(tc[2:] - tc[:-2]))
    return 2.0 * np.pi / period, float(abs(x[(t > tc[0]) & (t < tc[1])].min()))


def l6_added_mass(ratio: float, *, periods: float = 3.0, soft_inertia: bool = True, dry: bool = True,
                  device: str | None = None, log=None, **kw) -> dict:
    """L6 row for one ratio: omega and first peak, wet and dry, against sqrt(k / (m + m_a)) and rho g L A_p / k."""
    rig = spring_plate(ratio, wet=True, device=device, **kw)
    T = periods * 2.0 * np.pi / rig.info["omega_ref"]
    wet = run_rig(rig, T=T, soft_inertia=soft_inertia, added_mass=rig.info["m_added"], log=log, tag=f"L6 r{ratio}")
    out = {**{k: v for k, v in rig.info.items()}, "passes_mean": wet["passes_mean"],
           "passes_max": wet["passes_max"], "below": wet["below"], "finite": wet["finite"], "wall_s": wet["wall_s"]}
    hint = rig.info["omega_ref"]
    out["omega_wet"], peak = oscillation(wet["t"], wet["dz"] + rig.info["amp_ref"], hint)
    out["omega_ratio"], out["peak_ratio"] = out["omega_wet"] / rig.info["omega_ref"], peak / rig.info["amp_ref"]
    if dry:
        drig = spring_plate(ratio, wet=False, device=device, **kw)
        d = run_rig(drig, T=T)
        out["omega_dry"], dpeak = oscillation(d["t"], d["dz"] + rig.info["amp_ref"], hint)
        out["omega_wet_dry"] = out["omega_wet"] / out["omega_dry"]
        out["peak_wet_dry"] = peak / dpeak
        n = min(len(d["dz"]), len(wet["dz"]))
        out["track_err"] = float(np.max(np.abs(wet["dz"][:n] - d["dz"][:n])) / rig.info["amp_ref"])
    return out


def l7_cloth(sigma: float, *, T: float = 0.3, soft_inertia: bool = True, dry: bool = True, device: str | None = None,
             log=None, **kw) -> dict:
    """L7 row for one areal density: centre sag over the second half, wet and dry; liquid below the cloth."""
    rig = cloth_floor(sigma, wet=True, device=device, **kw)
    n, H, s = kw.get("n", 12), kw.get("H", 8), rig.config.spacing
    wet = run_rig(rig, T=T, soft_inertia=soft_inertia, added_mass=RHO_F * (n * s) ** 2 * H * s, log=log,
                  tag=f"L7 r{rig.info['ratio']:.0f}")
    h = len(wet["dz"]) // 2
    out = {"ratio": rig.info["ratio"], "sag_wet_mm": float(-wet["dz"][h:].mean() * 1e3),
           "passes_mean": wet["passes_mean"], "passes_max": wet["passes_max"], "below": wet["below"],
           "finite": wet["finite"], "wall_s": wet["wall_s"]}
    if dry:
        d = run_rig(cloth_floor(sigma, wet=False, device=device, **kw), T=T)
        out["sag_dry_mm"] = float(-d["dz"][h:].mean() * 1e3)
        out["sag_wet_dry"] = out["sag_wet_mm"] / out["sag_dry_mm"]
    return out
