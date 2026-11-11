# SPDX-License-Identifier: Apache-2.0
"""Laws of the SPH-VBD coupler (engine item 3; design [internal note], L1-L5).

Each runner builds its scene (soft_scenes.py), runs the VBD <-> SPH Scheduler eagerly and returns a metrics dict;
tests run them at quick size, the results rows at full size. Common monitors: passes per exchange (mean, max; WP4
review), liquid particles inside a soft body at the end (L5), max liquid speed.

- L1 ``floating_cube``: a soft tet box of density rho_s floats in a pool: draft vs m / (rho_f A), measured as
  A13.archimedes does (surface rise from the liquid volume, the bottom face's mean height); the harvested fluid force
  vs its weight; both averaged over the second half. A flat box (side >= 1.28 x height, as A13.archimedes): a uniform
  cube of density 0.21-0.79 floats on an edge, which the first runs did (tilt reported).
- L3 ``impact_momentum``: zero gravity, no walls; a liquid block at v0 hits a free soft cube. The total momentum of
  liquid and cube, which nothing outside acts on, against its start value over the run (and the momentum given to
  the cube, as a scale).
- L2 ``squeeze_slab``: a soft slab is the pool's floor (spans the tank, bottom pinned, frictionless side walls:
  laterally confined) under a water column H. Its top's mean compression, wet, against a dry twin whose top vertices
  carry rho g H as nodal forces (the coupling alone), and both against linear uniaxial strain
  (rho g H + rho_s g t / 2) t / M, M = E (1 - nu) / ((1 + nu)(1 - 2 nu)) (VBD's own accuracy, reported). Only the top
  face's vertices are proxies: the slab's sides lie on the tank walls, where a second coplanar wall would count the
  boundary volume twice.
- L4 ``drop_energy``: a soft box (rho_s 600) dropped into a closed pool; total mechanical energy (liquid and box:
  kinetic, gravity, and VBD's elastic energy) per exchange. Its largest rise over any window, as a fraction of the
  drop's energy E_scale = m g (z0 - z_float), against CPL-4's 5e-2 bound (the scene is dissipative: XSPH, implicit
  Euler).
- L5 ``sinking_box``: a dense soft box (rho_s 2000) released above the pool falls in, sinks and lands on the floor,
  squeezing the liquid out from under it. No liquid particle inside the box at any checked exchange; no active liquid
  particle lost (the solver's mass ledger).
"""

from __future__ import annotations

import time

from alloy._vendor import newton
import numpy as np

from .build import add_box_tank, add_fluid_block, add_fluid_points, lattice_points
from .config import DFSPHConfig, register_custom_attributes
from .soft_scenes import SoftPool, add_soft_box, elastic_energy, soft_pipeline, add_soft_cube, cube_pool, inside_box, tilt_deg, vbd_sph_scheduler


def floating_cube(*, rho_s: float = 600.0, T: float = 1.0, dt_x: float = 2.0e-3, mode: str = "staggered",
                  n: int = 16, depth: int = 12, box=(8, 8, 4), cells=(4, 4, 2), sph_rung: int = 2,
                  device: str | None = None, log=None) -> dict:
    """L1: draft and fluid force of a floating soft box (module doc); ``box`` in lattice cells, ``cells`` tet cells."""
    draft0 = rho_s / 1000.0 * box[2]
    x0, y0 = 0.5 * (n - box[0]), 0.5 * (n - box[1])
    pool = cube_pool("soft", n=n, depth=depth, cube_cells=box, cube_at=(x0, y0, depth - draft0), rho_s=rho_s,
                     ncell=cells, device=device)
    s = pool.config.spacing
    rest = pool.model.particle_q.numpy()[pool.soft].astype(np.float64)
    bottom_ids = np.asarray(pool.soft)[rest[:, 2] < rest[:, 2].min() + 1.0e-6]
    sched = vbd_sph_scheduler(pool, mode=mode, sph_rung=sph_rung,
                              added_mass=1000.0 * draft0 * s * pool.size[0] * pool.size[1])
    st, nx = pool.model.state(), pool.model.state()
    pipe = soft_pipeline(pool.model)
    con = pipe.contacts()
    mass = float(pool.model.particle_mass.numpy()[pool.soft].sum())
    a_tank, a_box = (n * s) ** 2, float(pool.size[0] * pool.size[1])
    v_liq = len(pool.liquid) * s**3
    steps = int(round(T / dt_x))
    bottom, fz, passes, vmax = [], [], [], 0.0
    t0 = time.time()
    for k in range(steps):
        st.clear_forces()
        pipe.collide(st, con)
        sched.step(st, nx, None, con, dt_x)
        st, nx = nx, st
        q = st.particle_q.numpy()
        bottom.append(float(q[bottom_ids, 2].mean()))
        fz.append(float(sched._proxy_particle_mappings[0].coupling_forces.numpy()[:, 2].sum()))
        passes.append(int(sched._loop.iters_used.numpy()[0]))
        vmax = max(vmax, float(np.abs(st.particle_qd.numpy()[pool.liquid.start:pool.liquid.stop]).max()))
        if log is not None and (k + 1) % max(steps // 20, 1) == 0:
            log(f"t={(k + 1) * dt_x:.3f} bottom {bottom[-1] / s:.3f} sp, Fz/W {fz[-1] / (mass * 9.81):.4f}, "
                f"passes {passes[-1]}, wall {time.time() - t0:.0f}s")
    h = steps // 2
    b = np.array(bottom[h:])
    draft = (v_liq / a_tank - b) / (1.0 - a_box / a_tank)
    ref = mass / (1000.0 * a_box)
    q = st.particle_q.numpy().astype(np.float64)
    return {"draft_ratio": float(draft.mean() / ref), "draft_ptp_ratio": float(np.ptp(draft) / ref),
            "force_ratio": float(np.mean(fz[h:]) / (mass * 9.81)), "passes_mean": float(np.mean(passes)),
            "passes_max": int(np.max(passes)),
            "inside": inside_box(q[pool.liquid.start:pool.liquid.stop], rest, q[pool.soft]),
            "tilt_deg": tilt_deg(rest, q[pool.soft]),
            "vmax_liquid": vmax, "draft_ref_m": float(ref), "draft_over_2r": float(ref / s), "mass_kg": mass,
            "finite": bool(np.isfinite(q).all()), "wall_s": time.time() - t0}


def impact_momentum(*, v0: float = 0.3, T: float = 0.15, dt_x: float = 1.0e-3, mode: str = "staggered",
                    block: int = 6, cube_cells: int = 6, gap: int = 2, rho_s: float = 1000.0, sph_rung: int = 2,
                    xsph_c: float | None = None, device: str | None = None, log=None) -> dict:
    """L3: momentum of a liquid block hitting a free soft cube in zero gravity (module doc). ``xsph_c`` 0 removes
    XSPH, whose m_j / rho_j weights are not pairwise symmetric (the liquid's own momentum drifts ~0.3 % after an
    impact, rigid walls too); the coupler alone is then measured."""
    cfg = DFSPHConfig(particle_radius=1.0e-3) if xsph_c is None else DFSPHConfig(particle_radius=1.0e-3,
                                                                                   xsph_c=float(xsph_c))
    s = cfg.spacing
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    off = 0.5 * (cube_cells - block) * s
    liq = add_fluid_block(b, cfg, (0.0, off, off), (block, block, block), velocity=(v0, 0.0, 0.0))
    lo = np.array([(block + gap) * s, 0.0, 0.0])
    soft = add_soft_cube(b, lo, cube_cells * s, 3, rho_s, radius=0.25 * s)
    b.end_world()
    b.color()
    model = b.finalize(device=device)
    tri = model.tri_indices.numpy()
    pool = SoftPool(model, cfg, liq, soft, sorted({int(v) for t in tri for v in t}), lo=lo,
                    size=np.full(3, cube_cells * s))
    rest = model.particle_q.numpy()[soft].astype(np.float64)
    sched = vbd_sph_scheduler(pool, mode=mode, sph_rung=sph_rung, added_mass=1000.0 * (cube_cells * s) ** 3 * 0.5)
    st, nx = model.state(), model.state()
    m = model.particle_mass.numpy().astype(np.float64)
    sel = np.r_[np.arange(liq.start, liq.stop), np.asarray(soft)]

    def momentum(state):
        v = state.particle_qd.numpy().astype(np.float64)
        return (m[sel, None] * v[sel]).sum(axis=0), (m[soft, None] * v[soft]).sum(axis=0)

    P0, _ = momentum(st)
    drift, p_soft, passes = [], [], []
    t0 = time.time()
    steps = int(round(T / dt_x))
    for k in range(steps):
        sched.step(st, nx, None, None, dt_x)
        st, nx = nx, st
        P, Ps = momentum(st)
        drift.append(np.linalg.norm(P - P0))
        p_soft.append(Ps[0])
        passes.append(int(sched._loop.iters_used.numpy()[0]))
        if log is not None and (k + 1) % max(steps // 15, 1) == 0:
            log(f"t={(k + 1) * dt_x:.3f} |P-P0|/|P0| {drift[-1] / np.linalg.norm(P0):.2e} "
                f"P_soft/P0 {Ps[0] / P0[0]:.4f} passes {passes[-1]}")
    q = st.particle_q.numpy()
    return {"momentum_drift": float(np.max(drift) / np.linalg.norm(P0)), "p_soft_ratio": float(p_soft[-1] / P0[0]),
            "passes_mean": float(np.mean(passes)), "passes_max": int(np.max(passes)),
            "inside": inside_box(q[liq.start:liq.stop].astype(np.float64), rest, q[soft].astype(np.float64)),
            "finite": bool(np.isfinite(q).all()),
            "wall_s": time.time() - t0}


def sinking_box(*, rho_s: float = 2000.0, T: float = 0.6, dt_x: float = 1.0e-3, n: int = 16, depth: int = 12,
                box=(6, 6, 4), cells=(3, 3, 2), drop: float = 2.0, check_every: int = 5, sph_rung: int = 2,
                device: str | None = None, log=None) -> dict:
    """L5: no liquid inside a dense soft box that falls into the pool and lands on the floor (module doc)."""
    x0, y0 = 0.5 * (n - box[0]), 0.5 * (n - box[1])
    pool = cube_pool("soft", n=n, depth=depth, cube_cells=box, cube_at=(x0, y0, depth + drop), rho_s=rho_s,
                     ncell=cells, device=device)
    s = pool.config.spacing
    rest = pool.model.particle_q.numpy()[pool.soft].astype(np.float64)
    sched = vbd_sph_scheduler(pool, sph_rung=sph_rung,
                              added_mass=1000.0 * float(np.prod(pool.size)))
    sph = sched._entries["sph"].solver
    st, nx = pool.model.state(), pool.model.state()
    pipe = soft_pipeline(pool.model)
    con = pipe.contacts()
    L = pool.liquid
    n_active0 = int((pool.model.particle_flags.numpy()[L.start:L.stop] & 1).sum())
    steps = int(round(T / dt_x))
    inside_max, passes, t0 = 0, [], time.time()
    for k in range(steps):
        st.clear_forces()
        pipe.collide(st, con)
        sched.step(st, nx, None, con, dt_x)
        st, nx = nx, st
        passes.append(int(sched._loop.iters_used.numpy()[0]))
        if (k + 1) % check_every == 0 or k == steps - 1:
            q = st.particle_q.numpy().astype(np.float64)
            inside_max = max(inside_max, inside_box(q[L.start:L.stop], rest, q[pool.soft]))
            if log is not None and (k + 1) % max(steps // 12, 1) == 0:
                log(f"t={(k + 1) * dt_x:.3f} box bottom {q[pool.soft, 2].min() / s:.3f} sp, inside {inside_max}, "
                    f"passes {passes[-1]}, wall {time.time() - t0:.0f}s")
    q = st.particle_q.numpy().astype(np.float64)
    flags = sph.model.particle_flags.numpy()[L.start:L.stop]
    return {"inside_max": int(inside_max), "active_lost": int(n_active0 - int((flags & 1).sum())),
            "bottom_sp": float(q[pool.soft, 2].min() / s), "tilt_deg": tilt_deg(rest, q[pool.soft]),
            "passes_mean": float(np.mean(passes)), "passes_max": int(np.max(passes)),
            "finite": bool(np.isfinite(q).all()), "wall_s": time.time() - t0}


def _slab_model(cfg: DFSPHConfig, n: int, t: int, H: int, cells, E: float, nu: float, rho_s: float, k_damp: float,
                wet: bool, device):
    s = cfg.spacing
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.default_shape_cfg.mu = 0.0  # frictionless side walls: the slab is confined laterally only
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (n * s, n * s, (t + H + 6) * s), wall=cfg.h)
    slab = add_soft_box(b, (0.0, 0.0, 0.0), (n * s, n * s, t * s), cells, rho_s, E=E, nu=nu, radius=0.25 * s,
                        k_damp=k_damp)
    for i in slab:
        if b.particle_q[i][2] < 1.0e-9:
            b.particle_mass[i] = 0.0  # pinned bottom (add_soft_grid's fix_bottom pins y = 0)
    liq = add_fluid_points(b, cfg, lattice_points((0.0, 0.0, t * s), (n, n, H), s)) if wet else range(0)
    b.end_world()
    b.color()
    model = b.finalize(device=device)
    model.soft_contact_mu = 0.0
    return model, slab, liq


def squeeze_slab(*, E: float = 2.0e3, nu: float = 0.3, rho_s: float = 1000.0, H: int = 12, t: int = 4, n: int = 12,
                 cells=(6, 6, 2), T: float = 1.0, dt_x: float = 2.0e-3, k_damp: float = 1.0e-2, sph_rung: int = 2,
                 device: str | None = None, log=None) -> dict:
    """L2: confined soft slab under a water column, wet vs dry twin vs linear theory (module doc)."""
    from alloy.solvers.vbd import make_vbd

    cfg = DFSPHConfig(particle_radius=1.0e-3)
    s = cfg.spacing
    g, p_load = 9.81, 1000.0 * 9.81 * H * s
    out, t0 = {}, time.time()
    for wet in (False, True):
        model, slab, liq = _slab_model(cfg, n, t, H, cells, E, nu, rho_s, k_damp, wet, device)
        rest = model.particle_q.numpy()[slab].astype(np.float64)
        top = np.asarray(slab)[rest[:, 2] > rest[:, 2].max() - 1.0e-9]
        z_top0 = model.particle_q.numpy()[top, 2].astype(np.float64)
        st, nx = model.state(), model.state()
        pipe = soft_pipeline(model)
        con = pipe.contacts()
        if wet:
            pool = SoftPool(model, cfg, liq, slab, sorted(int(i) for i in top))
            stepper = vbd_sph_scheduler(pool, sph_rung=sph_rung, added_mass=1000.0 * (n * s) ** 2 * H * s)
        else:
            stepper = make_vbd(model, {"iterations": 20})
            xy = model.particle_q.numpy()[top, :2] / ((n * s) / cells[0])
            edge = (np.isclose(xy, 0.0) | np.isclose(xy, cells[0])).sum(axis=1)  # corner 2, edge 1, interior 0
            area = ((n * s) / cells[0]) * ((n * s) / cells[1]) / 2.0**edge
            f = np.zeros((model.particle_count, 3), np.float32)
            f[top, 2] = -p_load * area
        steps = int(round(T / dt_x))
        dz, passes = [], []
        for k in range(steps):
            st.clear_forces()
            if not wet:
                st.particle_f.assign(f)
            pipe.collide(st, con)
            stepper.step(st, nx, None, con, dt_x)
            st, nx = nx, st
            dz.append(float((z_top0 - st.particle_q.numpy()[top, 2]).mean()))
            if wet:
                passes.append(int(stepper._loop.iters_used.numpy()[0]))
            if log is not None and (k + 1) % max(steps // 10, 1) == 0:
                log(f"{'wet' if wet else 'dry'} t={(k + 1) * dt_x:.3f} compression {dz[-1] * 1e3:.4f} mm, "
                    f"wall {time.time() - t0:.0f}s")
        key = "wet" if wet else "dry"
        out[f"delta_{key}_m"] = float(np.mean(dz[steps // 2:]))
        out[f"delta_{key}_ptp_m"] = float(np.ptp(dz[steps // 2:]))
        if wet:
            out["passes_mean"], out["passes_max"] = float(np.mean(passes)), int(np.max(passes))
            out["finite"] = bool(np.isfinite(st.particle_q.numpy()).all())
    M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lin = (p_load + rho_s * g * t * s / 2.0) * t * s / M
    out.update(delta_linear_m=lin, wet_over_dry=out["delta_wet_m"] / out["delta_dry_m"],
               dry_over_linear=out["delta_dry_m"] / lin, p_load_pa=p_load, wall_s=time.time() - t0)
    return out


def drop_energy(*, rho_s: float = 600.0, T: float = 0.6, dt_x: float = 2.0e-3, n: int = 16, depth: int = 12,
                box=(8, 8, 4), cells=(4, 4, 2), drop: float = 2.0, sph_rung: int = 2, device: str | None = None,
                log=None) -> dict:
    """L4: the largest rise of total energy when a soft box drops into the pool (module doc)."""
    x0, y0 = 0.5 * (n - box[0]), 0.5 * (n - box[1])
    pool = cube_pool("soft", n=n, depth=depth, cube_cells=box, cube_at=(x0, y0, depth + drop), rho_s=rho_s,
                     ncell=cells, device=device)
    s, g = pool.config.spacing, 9.81
    sched = vbd_sph_scheduler(pool, sph_rung=sph_rung,
                              added_mass=1000.0 * rho_s / 1000.0 * box[2] * s * pool.size[0] * pool.size[1])
    st, nx = pool.model.state(), pool.model.state()
    pipe = soft_pipeline(pool.model)
    con = pipe.contacts()
    m = pool.model.particle_mass.numpy().astype(np.float64)
    L = np.arange(pool.liquid.start, pool.liquid.stop)
    sel = np.r_[L, np.asarray(pool.soft)]
    mass_box = float(m[pool.soft].sum())

    def energy(state) -> float:
        q = state.particle_q.numpy().astype(np.float64)
        v = state.particle_qd.numpy().astype(np.float64)
        return float((m[sel] * (0.5 * (v[sel] ** 2).sum(axis=1) + g * q[sel, 2])).sum()) + elastic_energy(
            pool.model, q)

    E = [energy(st)]
    steps = int(round(T / dt_x))
    t0 = time.time()
    for k in range(steps):
        st.clear_forces()
        pipe.collide(st, con)
        sched.step(st, nx, None, con, dt_x)
        st, nx = nx, st
        E.append(energy(st))
        if log is not None and (k + 1) % max(steps // 12, 1) == 0:
            log(f"t={(k + 1) * dt_x:.3f} E-E0 {(E[-1] - E[0]) * 1e6:+.3f} uJ, wall {time.time() - t0:.0f}s")
    E = np.array(E)
    z_float = depth * s - rho_s / 1000.0 * box[2] * s  # the box's bottom when floating
    e_scale = mass_box * g * (depth * s + drop * s - z_float)
    rise = float(np.max(E - np.minimum.accumulate(E)))  # largest rise after any earlier minimum
    return {"rise_ratio": rise / e_scale, "step_rise_ratio": float(np.max(np.diff(E)) / e_scale),
            "dE_ratio": float((E[-1] - E[0]) / e_scale), "e_scale_J": e_scale, "finite": bool(np.isfinite(E).all()),
            "wall_s": time.time() - t0}
