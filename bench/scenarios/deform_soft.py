# SPDX-License-Identifier: Apache-2.0
"""D06 volumetric soft body: uniaxial Neo-Hookean squeeze of a 4 cm cube, and a cantilever block hanging in gravity.

FEM configs (vbd_tet, semi_implicit_fem, xpbd_tet) build tet grids with ModelBuilder.add_soft_grid (as
newton@ac7f7830 softbody/example_softbody_hanging.py). The cube is load controlled: a nominal compressive stress
staircase (the standard compressible Neo-Hookean stress at stretch 0.9, 0.8, 0.7 for the configured Lame
parameters) is applied as follower tractions on the top and bottom face triangles (reference area, current normal),
lateral faces free, gravity on the cube cancelled. With uniform traction a linear-tet mesh deforms homogeneously, so the law is exact up to solver
convergence. The block (20 x 4 x 4 cm, left face pinned) sags under eased gravity against Timoshenko beam theory.

implicit_mpm follows mpm/example_mpm_pinch_carry.py: particles squeezed between two frictionless kinematic pads
(displacement controlled, same strain levels) with the pad force from collect_collider_impulses().

Laws per solver energy (each solver implements a different "Neo-Hookean", see bench/notes/deform.md):
  std  Psi = mu/2 (I1-3) - mu ln J + lam/2 (ln J)^2                               (headline reference)
  vbd  P = mu F + (lam+mu)(J - alpha) cof F, alpha = 1 + mu/(lam+mu)             (particle_vbd_kernels.py:220)
  semi P = mu F (1 - 1/(I_C+1)) + lam (J - alpha) cof F, alpha = 1 + 3mu/(4lam)  (kernels_particle.py:250-410)
  xpbd energy V0 [k_mu (I_C-3)^2 + k_lam (J-1)^2] / 2                           (xpbd/kernels.py:505-640)
  mpm  Hencky: P = E ln(s) / s under uniaxial stress                             (implicit_mpm_solver_kernels.py:21)
"""
from __future__ import annotations

import math
from typing import Callable

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench.harness import Built, Config
from bench.scenarios import deform_laws as LAW
from bench.scenarios.deform_common import (G, DeformScenario, Driver, assemble, flag_blowup, law_record, pct,
                                           rec_rows, recorded_worlds, world_ids)

# ---- geometry, materials, schedule -------------------------------------------------------------------------------
CUBE, CUBE_N = 0.04, 4  # cube edge [m], cells per edge
CUBE_E, CUBE_NU, RHO = 3.0e4, 0.49, 1000.0
BLOCK = (0.20, 0.04, 0.04)  # cantilever length, width, height [m]
BLOCK_N = (20, 4, 4)
BLOCK_E, BLOCK_NU = 2.0e6, 0.3
STRETCHES = (0.9, 0.8, 0.7)  # load levels: 10, 20, 30 % compressive strain
T_GRAV = 0.5  # gravity eased in over [0, T_GRAV]
T_LOAD0, T_RAMP, T_HOLD = 0.5, 0.8, 0.5  # first ramp start, ramp duration, hold duration per level
CUBE_Y, BLOCK_Y = 0.0, 0.3  # world-local placement


def level_windows() -> list[tuple[float, float, float]]:
    """(ramp start, ramp end, hold end) per load level."""
    out, t = [], T_LOAD0
    for _ in STRETCHES:
        out.append((t, t + T_RAMP, t + T_RAMP + T_HOLD))
        t += T_RAMP + T_HOLD
    return out


HORIZON = level_windows()[-1][2]


def lame(E: float, nu: float) -> tuple[float, float]:
    return LAW.lame(E, nu)


# ---- uniaxial references per energy ----------------------------------------------------------------------------
def _p_std(a, s, mu, lam):
    J = a * a * s  # a in [0.5, 2] and s in (0.3, 1.5): J > 0
    return mu * (a - 1 / a) + lam * math.log(J) / a, mu * (s - 1 / s) + lam * math.log(J) / s


def _p_vbd(a, s, mu, lam):
    ln, J = lam + mu, a * a * s
    alpha = 1.0 + mu / ln
    return mu * a + ln * (J - alpha) * a * s, mu * s + ln * (J - alpha) * a * a


def _p_semi(a, s, mu, lam):
    J, ic = a * a * s, 2 * a * a + s * s
    alpha = 1.0 + mu / lam - mu / (4.0 * lam)
    dev = mu * (1.0 - 1.0 / (ic + 1.0))
    return dev * a + lam * (J - alpha) * a * s, dev * s + lam * (J - alpha) * a * a


def _p_xpbd(a, s, mu, lam):
    J, ic = a * a * s, 2 * a * a + s * s
    return 2 * mu * (ic - 3) * a + lam * (J - 1) * a * s, 2 * mu * (ic - 3) * s + lam * (J - 1) * a * a


ENERGY: dict[str, Callable] = {"std": _p_std, "vbd": _p_vbd, "semi_implicit": _p_semi, "xpbd": _p_xpbd}


def uniaxial(form: str, s: float, mu: float, lam: float) -> tuple[float, float]:
    """(nominal axial stress P [Pa], volume ratio J) at axial stretch s with traction-free lateral faces.

    NaN for a stretch outside (0.3, 1.5): a collapsed or inverted cube has no uniaxial reference.
    """
    if not (0.3 < s < 1.5):
        return float("nan"), float("nan")
    if form == "mpm":
        E, nu = mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))
        return E * math.log(s) / s, s ** (1.0 - 2.0 * nu)
    fn = ENERGY[form]
    lo, hi = 0.5, 2.0
    for _ in range(200):  # lateral traction is increasing in a
        mid = 0.5 * (lo + hi)
        if fn(mid, s, mu, lam)[0] > 0.0:
            hi = mid
        else:
            lo = mid
    a = 0.5 * (lo + hi)
    return fn(a, s, mu, lam)[1], a * a * s


def stretch_for_stress(form: str, P: float, mu: float, lam: float) -> float:
    """Axial stretch carrying nominal stress P (< 0) under the given energy (bisection on s in (0.3, 1))."""
    lo, hi = 0.3, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if uniaxial(form, mid, mu, lam)[0] < P:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- FEM world ---------------------------------------------------------------------------------------------------
def _face_tris(verts: np.ndarray, tris: np.ndarray, z: float, outward: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface triangles in the plane z, reoriented so their normal points `outward` (+1 = +z), and their areas."""
    on = np.all(np.abs(verts[tris][:, :, 2] - z) < 1e-9, axis=1)
    sel = tris[on].copy()
    a, b, c = verts[sel[:, 0]], verts[sel[:, 1]], verts[sel[:, 2]]
    n = np.cross(b - a, c - a)
    flip = n[:, 2] * outward < 0
    sel[flip] = sel[flip][:, [0, 2, 1]]
    return sel, 0.5 * np.linalg.norm(n, axis=1)


@wp.kernel
def k_follower_traction(counter: wp.array(dtype=wp.int32), h: float, t0: wp.array(dtype=float),
                        t1: wp.array(dtype=float), dP: wp.array(dtype=float), tris: wp.array2d(dtype=wp.int32),
                        area0: wp.array(dtype=float), particle_q: wp.array(dtype=wp.vec3),
                        particle_f: wp.array(dtype=wp.vec3)):
    """Nominal stress P(t) (eased staircase) on face triangles along their current outward normal.

    Force per triangle = P * A0 * n_hat, split equally over its vertices: the total is the nominal load P * A0 and it
    turns with the face, so a free cube under compression feels no couple when it rotates (dead loads make the
    compressed free cube tumble: all FEM configs inverted at about 7 % strain in a dead-load version).
    """
    i = wp.tid()
    t = float(counter[0]) * h
    P = float(0.0)
    for k in range(dP.shape[0]):
        u = wp.clamp((t - t0[k]) / (t1[k] - t0[k]), 0.0, 1.0)
        P = P + dP[k] * u * u * (3.0 - 2.0 * u)
    a = tris[i, 0]
    b = tris[i, 1]
    c = tris[i, 2]
    n = wp.normalize(wp.cross(particle_q[b] - particle_q[a], particle_q[c] - particle_q[a]))
    f = n * (P * area0[i] / 3.0)
    wp.atomic_add(particle_f, a, f)
    wp.atomic_add(particle_f, b, f)
    wp.atomic_add(particle_f, c, f)


def _fem_world(form: str, p: dict) -> tuple[newton.ModelBuilder, dict]:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    mu_c, lam_c = lame(CUBE_E, CUBE_NU)
    h = CUBE / CUBE_N
    s0 = b.particle_count
    b.add_soft_grid(pos=wp.vec3(-0.5 * CUBE, CUBE_Y - 0.5 * CUBE, 0.5), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                    dim_x=CUBE_N, dim_y=CUBE_N, dim_z=CUBE_N, cell_x=h, cell_y=h, cell_z=h, density=RHO, k_mu=mu_c,
                    k_lambda=lam_c, k_damp=p["damp_cube"], particle_radius=0.0, add_surface_mesh_edges=False,
                    label="cube")
    s1, t1 = b.particle_count, len(b.tri_indices)
    cube_v = np.asarray(b.particle_q[s0:s1], dtype=np.float64)
    tri = np.asarray(b.tri_indices, dtype=np.int64) - s0
    tri = tri[(tri >= 0).all(axis=1) & (tri < s1 - s0).all(axis=1)]
    z_lo, z_hi = cube_v[:, 2].min(), cube_v[:, 2].max()
    t_top, a_top = _face_tris(cube_v, tri, z_hi, +1.0)
    t_bot, a_bot = _face_tris(cube_v, tri, z_lo, -1.0)
    face_tris, face_area = np.concatenate([t_top, t_bot]) + s0, np.concatenate([a_top, a_bot])

    def node(x, y, z):
        return s0 + int(np.argmin(np.linalg.norm(cube_v - np.array([x, y, z]), axis=1)))

    cx, cy = 0.0, CUBE_Y
    marks = {"top": node(cx, cy, z_hi), "bot": node(cx, cy, z_lo), "xm": node(cx - 0.5 * CUBE, cy, 0.5 + 0.5 * CUBE),
             "xp": node(cx + 0.5 * CUBE, cy, 0.5 + 0.5 * CUBE), "ym": node(cx, cy - 0.5 * CUBE, 0.5 + 0.5 * CUBE),
             "yp": node(cx, cy + 0.5 * CUBE, 0.5 + 0.5 * CUBE)}
    # cantilever block, left face pinned
    mu_b, lam_b = lame(BLOCK_E, BLOCK_NU)
    hb = BLOCK[0] / BLOCK_N[0]
    b0 = b.particle_count
    b.add_soft_grid(pos=wp.vec3(0.0, BLOCK_Y - 0.5 * BLOCK[1], 0.6), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                    dim_x=BLOCK_N[0], dim_y=BLOCK_N[1], dim_z=BLOCK_N[2], cell_x=hb, cell_y=hb, cell_z=hb,
                    density=RHO, k_mu=mu_b, k_lambda=lam_b, k_damp=p["damp_block"], fix_left=True,
                    particle_radius=0.0, add_surface_mesh_edges=False, label="block")
    blk = np.asarray(b.particle_q[b0:], dtype=np.float64)
    tip = b0 + np.where(np.abs(blk[:, 0] - blk[:, 0].max()) < 1e-9)[0]
    if form == "vbd":
        b.color()
    info = {"cube": (s0, s1), "face_tris": face_tris, "face_area": face_area, "marks": marks, "tip": tip, "mu": mu_c, "lam": lam_c,
            "mass": np.asarray(b.particle_mass, dtype=np.float64)}
    return b, info


def _fem_solver(cfg: Config, model: newton.Model):
    S, p = newton.solvers, cfg.params
    if cfg.solver == "vbd":
        return S.SolverVBD(model, iterations=p["iterations"], particle_enable_self_contact=False,
                           particle_enable_tile_solve=False)
    if cfg.solver == "semi_implicit":
        return S.SolverSemiImplicit(model)
    if cfg.solver == "xpbd":
        return S.SolverXPBD(model, iterations=p["iterations"], soft_body_relaxation=p["relaxation"])
    raise ValueError(cfg.solver)


# ---- MPM world ---------------------------------------------------------------------------------------------------
PAD_HALF = (0.035, 0.035, 0.006)
MPM_PITCH = 0.2  # world spacing along x [m] (one shared grid; worlds never touch)


@wp.kernel
def k_mpm_finish(solver_dt: float, time: wp.array(dtype=float)):
    time[0] = time[0] + solver_dt


def _strain_schedule(t: np.ndarray) -> np.ndarray:
    """Compressive engineering strain vs time: eased steps to 0.1, 0.2, 0.3 with holds (same windows as FEM)."""
    eps = np.zeros_like(t)
    prev = 0.0
    for (t0, t1, _), s in zip(level_windows(), STRETCHES):
        u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        eps += (1.0 - s - prev) * u * u * (3 - 2 * u)
        prev = 1.0 - s
    return eps


class SoftSqueeze(DeformScenario):
    id = "D06_soft_squeeze"
    title = "4 cm Neo-Hookean cube squeezed to 30 % (traction / pad driven) + cantilever block hanging in gravity"
    horizon_s = HORIZON
    sample_every = 6  # 0.1 s at dt 1/60: MPM pad forces are sampled host side in the hold windows
    accuracy_unit = "pct"
    accuracy_ref = ("f20: nominal stress error at the measured stretch for the 20 % level vs uniaxial compressible "
                    "Neo-Hookean (mu, lam configured; A15); own-energy errors, f10/f30, small-strain slope vs E, "
                    "volume ratio vs the law, block sag vs Timoshenko in metrics")
    configs = (
        Config("vbd_tet", "vbd", dt=1 / 60, substeps=10,
               params={"iterations": 20, "damp_cube": 50.0, "damp_block": 2.0e3}),
        Config("vbd_tet_it100", "vbd", dt=1 / 60, substeps=10,
               params={"iterations": 100, "damp_cube": 50.0, "damp_block": 2.0e3}, max_nworld=1024),
        Config("semi_implicit_fem", "semi_implicit", dt=1 / 60, substeps=200,
               params={"damp_cube": 5.0, "damp_block": 5.0}, max_nworld=1024),
        Config("xpbd_tet", "xpbd", dt=1 / 60, substeps=20,
               params={"iterations": 10, "relaxation": 0.2, "damp_cube": 1.0, "damp_block": 10.0}),
        Config("implicit_mpm", "implicit_mpm", dt=1 / 60, substeps=10,
               params={"voxel": 0.005, "ppc": 2, "max_iterations": 250, "tolerance": 1e-5}, max_nworld=256),
        Config("implicit_mpm_v10", "implicit_mpm", dt=1 / 60, substeps=10,
               params={"voxel": 0.01, "ppc": 2, "max_iterations": 250, "tolerance": 1e-5}, max_nworld=1024),
    )

    # ---- build ------------------------------------------------------------------------------------------------
    def build(self, nworld: int, device: str, config: Config) -> Built:
        if config.solver == "implicit_mpm":
            return self._build_mpm(nworld, device, config)
        tmpl, info = _fem_world(config.solver, config.params)
        asm = assemble([tmpl], nworld)
        model = asm.builder.finalize(device=device)
        solver = _fem_solver(config, model)
        s0, s1, ctrl = model.state(), model.state(), model.control()
        h = config.dt / config.substeps
        drv = Driver(device, h)
        drv.gravity_ramp(model, 0.0, T_GRAV)
        worlds = list(range(nworld))
        c0, c1 = info["cube"]
        cube_ids = list(range(c0, c1))
        drv.particle_force(world_ids(asm.particle_start, cube_ids, worlds).ravel(),
                           np.tile(np.outer(info["mass"][c0:c1], [0.0, 0.0, G]), (nworld, 1)), 0.0, T_GRAV)
        loads = [uniaxial("std", s, info["mu"], info["lam"])[0] for s in STRETCHES]
        win = level_windows()
        sched = [wp.array(np.asarray(v, np.float32), dtype=float, device=device)
                 for v in ([w[0] for w in win], [w[1] for w in win], np.diff(np.concatenate([[0.0], loads])))]
        ftris = (asm.particle_start[:, None, None] + info["face_tris"][None]).reshape(-1, 3)
        ftris_w = wp.array(ftris.astype(np.int32), dtype=wp.int32, device=device)
        farea_w = wp.array(np.tile(info["face_area"], nworld).astype(np.float32), dtype=float, device=device)
        drv.pre.append(lambda st0, st1: wp.launch(k_follower_traction, dim=len(ftris), device=device,
                                                  inputs=[drv.counter, h, *sched, ftris_w, farea_w, st0.particle_q,
                                                          st0.particle_f]))
        rw = recorded_worlds(nworld)
        names = list(info["marks"])
        stride = max(1, round(0.01 / h))
        rows = rec_rows(HORIZON, h, stride)
        drv.record_particles("cube", world_ids(asm.particle_start, [info["marks"][k] for k in names], rw).ravel(),
                             stride, rows)
        drv.record_particles("tip", world_ids(asm.particle_start, list(info["tip"]), rw).ravel(), stride, rows)
        x0 = s0.particle_q.numpy().astype(np.float64)
        return Built(model, solver, s0, s1, ctrl, nworld=nworld,
                     extra={"driver": drv, "info": info, "names": names, "nrec": len(rw), "loads": loads,
                            "x0_marks": x0[[info["marks"][k] for k in names]], "x0_tip": x0[info["tip"]],
                            "asm": asm})

    def _build_mpm(self, nworld: int, device: str, config: Config) -> Built:
        from alloy._vendor.newton.solvers import SolverImplicitMPM

        p = config.params
        spacing = p["voxel"] / p["ppc"]
        n = int(round(CUBE / spacing))
        ax = (np.arange(n) + 0.5) * spacing - 0.5 * CUBE
        pts = np.stack(np.meshgrid(ax, ax, ax + 0.5 * CUBE, indexing="ij"), -1).reshape(-1, 3)
        main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverImplicitMPM.register_custom_attributes(main)
        pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
        pads, origins = [], []
        for w in range(nworld):
            wb = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverImplicitMPM.register_custom_attributes(wb)
            org = np.array([w * MPM_PITCH, 0.0, 0.0])
            q = pts + org
            wb.add_particles(pos=q.tolist(), vel=np.zeros_like(q).tolist(), mass=[RHO * spacing**3] * len(q),
                             radius=[0.5 * spacing] * len(q))
            ids = []
            for side, z in (("bot", -PAD_HALF[2]), ("top", CUBE + PAD_HALF[2])):
                body = wb.add_body(is_kinematic=True, xform=wp.transform(wp.vec3(*(org + [0.0, 0.0, z])), wp.quat_identity()),
                                   label=f"pad_{side}")
                wb.add_shape_box(body, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=pad_cfg)
                ids.append(body)
            b0 = main.body_count
            main.add_world(wb)
            pads.append([b0 + i for i in ids])
            origins.append(org)
        model = main.finalize(device=device)
        E = CUBE_E
        model.mpm.young_modulus.fill_(E)
        model.mpm.poisson_ratio.fill_(CUBE_NU)
        model.mpm.friction.fill_(0.0)
        model.mpm.yield_stress.fill_(1.0e12)
        model.mpm.yield_pressure.fill_(1.0e12)
        model.mpm.tensile_yield_ratio.fill_(1.0)
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = p["voxel"]
        cfg.max_iterations = p["max_iterations"]
        cfg.tolerance = p["tolerance"]
        cfg.grid_type = "sparse"
        cfg.max_active_cell_count = int(model.particle_count) * 2 + 256
        solver = SolverImplicitMPM(model, cfg)
        state = model.state()
        state.mpm.particle_Jp.fill_(1.0)
        h = config.dt / config.substeps
        drv = Driver(device, h)
        t = np.arange(0.0, HORIZON + 0.1, 0.005)
        half = 0.5 * CUBE * _strain_schedule(t)
        table = np.zeros((len(t), 2 * nworld, 7))
        table[..., 6] = 1.0
        for w in range(nworld):
            table[:, 2 * w, :3] = origins[w] + np.stack([np.zeros_like(t), np.zeros_like(t), -PAD_HALF[2] + half], 1)
            table[:, 2 * w + 1, :3] = origins[w] + np.stack([np.zeros_like(t), np.zeros_like(t),
                                                            CUBE + PAD_HALF[2] - half], 1)
        drv.body_pose_table([i for pr in pads for i in pr], table, 0.005)
        drv.run_pre(state, state)
        solver.reset(state)
        mu, lam = lame(CUBE_E, CUBE_NU)
        return Built(model, solver, state, state, None, nworld=nworld,
                     extra={"driver": drv, "mpm": True, "pads": pads, "samples": [], "h": h, "mu": mu, "lam": lam,
                            "npart": len(pts), "origins": np.array(origins)})

    # ---- stepping -----------------------------------------------------------------------------------------------
    def substep(self, b: Built, dt: float) -> None:
        if not b.extra.get("mpm"):
            return super().substep(b, dt)
        drv: Driver = b.extra["driver"]
        s = b.state_0
        drv.run_pre(s, s)
        b.solver.step(s, s, None, None, dt)
        b.solver.project_outside(s, s, dt)
        drv.run_post(s)

    def measure(self, b: Built, t: float, rec: dict) -> None:
        if not b.extra.get("mpm"):
            return
        imp, _, cid = b.solver.collect_collider_impulses(b.state_0)
        imp, cid = imp.numpy(), cid.numpy()
        cbi = b.solver.collider_body_index.numpy()
        body = np.where(cid >= 0, cbi[np.maximum(cid, 0)], -2)
        top = b.extra["pads"][0][1]
        fz = float(np.abs(imp[body == top, 2].sum())) / b.extra["h"]
        q = b.state_0.particle_q.numpy()[: b.extra["npart"]]
        b.extra["samples"].append((t, fz, float(q[:, 2].min()), float(q[:, 2].max())))

    # ---- laws ---------------------------------------------------------------------------------------------------
    def finish(self, b: Built, rec: dict) -> None:
        if b.extra.get("mpm"):
            return self._finish_mpm(b, rec)
        flag_blowup(b.state_0, rec)
        drv: Driver = b.extra["driver"]
        t, xyz = drv.trace("cube")
        names, info = b.extra["names"], b.extra["info"]
        xyz = xyz.reshape(len(t), b.extra["nrec"], len(names), 3)[:, 0]
        k = {n: i for i, n in enumerate(names)}
        x0 = b.extra["x0_marks"]
        h0 = x0[k["top"], 2] - x0[k["bot"], 2]
        w0x, w0y = x0[k["xp"], 0] - x0[k["xm"], 0], x0[k["yp"], 1] - x0[k["ym"], 1]
        s_t = (xyz[:, k["top"], 2] - xyz[:, k["bot"], 2]) / h0
        ax_t = (xyz[:, k["xp"], 0] - xyz[:, k["xm"], 0]) / w0x
        ay_t = (xyz[:, k["yp"], 1] - xyz[:, k["ym"], 1]) / w0y
        mu, lam = info["mu"], info["lam"]
        form = {"vbd": "vbd", "semi_implicit": "semi_implicit", "xpbd": "xpbd"}[rec["solver"]]
        out, per = {}, {}
        for lvl, ((t0, t1, t2), s_ref, P) in enumerate(zip(level_windows(), STRETCHES, b.extra["loads"])):
            win = (t >= t1 + 0.5 * (t2 - t1)) & (t <= t2)
            key = f"f{round(100 * (1 - s_ref))}"
            if not win.any():
                per[key] = {"note": "horizon ended before the hold"}
                continue
            s_m = float(s_t[win].mean())
            J_m = float((s_t[win] * ax_t[win] * ay_t[win]).mean())
            P_std, J_std = uniaxial("std", s_m, mu, lam)
            P_own, J_own = uniaxial(form, s_m, mu, lam)
            per[key] = {"stretch_meas": s_m, "stretch_target": s_ref, "P_applied_pa": P, "P_ref_pa": P_std,
                        "err_pct": pct(P, P_std), "own_err_pct": pct(P, P_own), "J_meas": J_m, "J_ref": J_std,
                        "volume_ratio": J_m / J_std, "own_volume_ratio": J_m / J_own,
                        "osc_mm": 1e3 * h0 * float(np.ptp(s_t[win]))}
            out[f"{key}_err_pct"] = per[key]["err_pct"]
            out[f"{key}_own_err_pct"] = per[key]["own_err_pct"]
        # small-strain slope from the first ramp: P(t) is the eased schedule; fit P = E_meas (s - 1) for |s-1| < 3 %
        t0, t1, _ = level_windows()[0]
        u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        P_t = b.extra["loads"][0] * u * u * (3 - 2 * u)
        eps = s_t - 1.0
        sel = (t <= t1) & (eps < -0.002) & (eps > -0.03)
        E_meas = float(np.sum(P_t[sel] * eps[sel]) / np.sum(eps[sel] ** 2)) if sel.sum() >= 3 else float("nan")
        out["slope_err_pct"] = pct(E_meas, CUBE_E)
        out["slope_E_meas_pa"] = E_meas
        # hanging block: tip-face mean sag over the last hold vs Timoshenko (small strain)
        tt, tip = drv.trace("tip")
        ntip = len(info["tip"])
        tip = tip.reshape(len(tt), b.extra["nrec"], ntip, 3)[:, 0]
        sag_t = b.extra["x0_tip"][:, 2].mean() - tip[:, :, 2].mean(axis=1)
        late = tt >= tt[-1] - 0.5
        sag = float(sag_t[late].mean()) if late.any() else float("nan")
        sag_ref = LAW.timoshenko_tip(BLOCK[0], BLOCK[1], BLOCK[2], RHO, G, BLOCK_E, BLOCK_NU)
        out.update(hang_sag_mm=1e3 * sag, hang_sag_ref_mm=1e3 * sag_ref, hang_err_pct=pct(sag, sag_ref),
                   hang_osc_mm=1e3 * float(np.ptp(sag_t[late])) if late.any() else float("nan"))
        out["inverted"] = bool(np.any(s_t <= 0.0))  # top face passed through the bottom: law cannot hold
        out["min_stretch"] = float(np.min(s_t)) if len(s_t) else float("nan")
        f20 = out.get("f20_err_pct", float("nan"))
        vr = per.get("f20", {}).get("volume_ratio", float("nan"))
        law_ok = bool(np.isfinite(f20) and abs(f20) <= 5.0 and 0.99 <= vr <= 1.01)
        law_record(rec, f20, volume_ratio=vr, per_condition=per, law_pass=law_ok,
                   law_pass_hang=bool(np.isfinite(out["hang_err_pct"]) and abs(out["hang_err_pct"]) <= 10.0),
                   energy_form=form, fit_dt_only=False, conditions_covered=True, **out)

    def _finish_mpm(self, b: Built, rec: dict) -> None:
        smp = np.asarray(b.extra["samples"]) if b.extra["samples"] else np.zeros((0, 4))
        mu, lam = b.extra["mu"], b.extra["lam"]
        out, per = {}, {}
        if len(smp):
            t, fz = smp[:, 0], smp[:, 1]
            eps = _strain_schedule(t)
            for (t0, t1, t2), s_ref in zip(level_windows(), STRETCHES):
                win = (t >= t1 + 0.5 * (t2 - t1) - 1e-9) & (t <= t2 + 1e-9)
                key = f"f{round(100 * (1 - s_ref))}"
                if not win.any():
                    per[key] = {"note": "horizon ended before the hold"}
                    continue
                s = float(1.0 - eps[win].mean())
                P_meas = -float(fz[win].mean()) / CUBE**2
                P_std = uniaxial("std", s, mu, lam)[0]
                P_own = uniaxial("mpm", s, mu, lam)[0]
                per[key] = {"stretch": s, "P_meas_pa": P_meas, "P_ref_pa": P_std, "err_pct": pct(P_meas, P_std),
                            "own_err_pct": pct(P_meas, P_own), "force_cv": float(np.std(fz[win]) / max(np.mean(fz[win]), 1e-12))}
                out[f"{key}_err_pct"] = per[key]["err_pct"]
                out[f"{key}_own_err_pct"] = per[key]["own_err_pct"]
            ramp = (t <= level_windows()[0][1]) & (eps > 0.002) & (eps < 0.03)
            if ramp.sum() >= 2:
                E_meas = float(np.sum(fz[ramp] / CUBE**2 * eps[ramp]) / np.sum(eps[ramp] ** 2))
                out.update(slope_E_meas_pa=E_meas, slope_err_pct=pct(E_meas, CUBE_E))
        f20 = out.get("f20_err_pct", float("nan"))
        law_record(rec, f20, per_condition=per, energy_form="mpm (Hencky)", volume_ratio=None,
                   law_pass=bool(np.isfinite(f20) and abs(f20) <= 5.0), fit_dt_only=False, conditions_covered=True,
                   samples=len(smp), note_hang="cantilever block not built for MPM (no clamp primitive)", **out)


SCENARIOS_SOFT = [SoftSqueeze()]
