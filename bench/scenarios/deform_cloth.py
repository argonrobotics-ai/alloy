# SPDX-License-Identifier: Apache-2.0
"""D03 Cusick drape, D04 strip stretch / ASTM D1388 cantilever / hanging strip, D05 parallel-jaw cloth hold.

Scenes follow newton@ac7f7830 examples cloth/example_cloth_hanging.py (four-solver switch),
vbd/example_cloth_stiff_material_stretch.py (VBD membrane: tri_ke = mu, tri_ka = lambda, its :38-56),
vbd/example_vbd_gripper_soft_grid.py (gantry + prismatic jaws, full-surface rigid-soft contact).

One cloth material (cotton-like) is mapped to each solver's parameters by `cloth_kwargs`; the mapping is part of
what the bench measures (bench/notes/deform.md lists the derivations).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench.harness import Built, Config
from bench.scenarios import deform_geom as GEO
from bench.scenarios import deform_laws as LAW
from bench.scenarios.deform_common import (G, DeformScenario, Driver, assemble, flag_blowup, law_record,
                                           make_particle_solver, pct, pin_particles, rec_rows, recorded_worlds,
                                           world_ids)


@dataclass(frozen=True)
class ClothMat:
    rho_s: float = 0.15  # areal density [kg/m^2] (150 gsm cotton)
    E_s: float = 1000.0  # membrane Young's modulus x thickness [N/m]
    nu: float = 0.3
    B: float = 2.0e-5  # bending rigidity per unit width [N m] -> bending length c = (B / W)^(1/3) = 2.39 cm

    @property
    def mu_s(self) -> float:
        return self.E_s / (2 * (1 + self.nu))

    @property
    def lam_s(self) -> float:
        return self.E_s * self.nu / (1 - self.nu**2)

    @property
    def W(self) -> float:
        return self.rho_s * G

    @property
    def c(self) -> float:
        return (self.B / self.W) ** (1.0 / 3.0)


def cloth_kwargs(solver: str, mat: ClothMat, h: float, tau: float) -> dict:
    """Material -> solver parameters on a square-grid mesh of cell h (bending about a grid axis).

    vbd: tri_ke = mu, tri_ka = lambda (StableNH adds mu internally, particle_vbd_kernels.py:581); dihedral energy
         0.5 k |e| theta^2 with theta = kappa h on a grid edge gives edge_ke = B / h.
    semi_implicit: Smith StableNH triangles (kernels_particle.py:100-140) linearise to lambda_lame = k_lambda - mu,
         so tri_ka = lambda + mu; bending force ke theta |e|^2 along n (not energy consistent) gives edge_ke = B / h^3.
    xpbd: no membrane FEM, springs on every mesh edge; a right-triangle lattice of springs k has E_s = 4k/3 and
         nu = 1/3, so spring_ke = 0.75 E_s; bending gradient magnitude |e| instead of 1/h gives edge_ke = B / h^2.
    style3d: Baraff-Witkin stretch tri_aniso_ke = (E_s, E_s, mu) (nu = 0 in that model), isometric bending
         edge_aniso_ke = B.
    Damping is stiffness proportional with time constant tau (explicit solvers need a tiny tau).
    """
    if solver == "vbd":
        return dict(tri_ke=mat.mu_s, tri_ka=mat.lam_s, tri_kd=mat.mu_s * tau, edge_ke=mat.B / h,
                    edge_kd=mat.B / h * tau)
    if solver == "semi_implicit":
        return dict(tri_ke=mat.mu_s, tri_ka=mat.lam_s + mat.mu_s, tri_kd=mat.mu_s * tau, edge_ke=mat.B / h**3,
                    edge_kd=mat.B / h**3 * tau)
    if solver == "xpbd":
        return dict(tri_ke=0.0, tri_ka=0.0, tri_kd=0.0, edge_ke=mat.B / h**2, edge_kd=0.0, add_springs=True,
                    spring_ke=0.75 * mat.E_s, spring_kd=0.0)
    if solver == "style3d":
        return dict(tri_aniso_ke=wp.vec3(mat.E_s, mat.E_s, mat.mu_s), tri_ka=mat.lam_s, tri_kd=mat.mu_s * tau,
                    edge_aniso_ke=wp.vec3(mat.B, mat.B, mat.B), edge_kd=mat.B * tau)
    raise ValueError(solver)


def new_builder(solver: str, gravity: float = -G) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    if solver == "style3d":
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
    return b


def register_for(solver: str):
    return newton.solvers.SolverStyle3D.register_custom_attributes if solver == "style3d" else None


def add_cloth(b: newton.ModelBuilder, solver: str, verts: np.ndarray, tris: np.ndarray, mat: ClothMat, h: float,
              tau: float, radius: float) -> int:
    """Add a cloth mesh (vertices already in world coordinates); returns the first particle index."""
    start = b.particle_count
    kw = cloth_kwargs(solver, mat, h, tau)
    common = dict(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                  vertices=[wp.vec3(*v) for v in verts], indices=tris.ravel().tolist(), density=mat.rho_s,
                  particle_radius=radius)
    if solver == "style3d":
        from alloy._vendor.newton.solvers import style3d

        panel = [wp.vec2(float(v[0]), float(v[1])) for v in _panel_coords(verts)]
        style3d.add_cloth_mesh(b, **common, panel_verts=panel, panel_indices=tris.ravel().tolist(), **kw)
    else:
        b.add_cloth_mesh(scale=1.0, **common, **kw)
    return start


def _panel_coords(verts: np.ndarray) -> np.ndarray:
    """Flat 2D rest coordinates of a planar mesh (its two largest-extent axes)."""
    ext = np.ptp(verts, axis=0)
    ax = np.argsort(ext)[::-1][:2]
    return verts[:, np.sort(ax)]


def finalize_cloth_model(asm, device: str, mu: float = 0.5, ke: float = 1.0e3, kd: float = 1.0e1):
    model = asm.builder.finalize(device=device)
    model.soft_contact_ke = ke
    model.soft_contact_kd = kd
    model.soft_contact_mu = mu
    return model


def max_speed(state) -> float:
    v = state.particle_qd.numpy()
    return float(np.max(np.linalg.norm(v, axis=1))) if len(v) else 0.0


# ---- D03 Cusick drape ---------------------------------------------------------------------------------------------
DR_R, DR_SUPPORT, DR_HS = 0.15, 0.09, (0.012, 0.008, 0.005)  # cloth radius, support radius, mesh cells [m]
DR_RADIUS = 0.002  # particle / contact radius
DR_MAT = ClothMat()


class ClothDrape(DeformScenario):
    id = "D03_cloth_drape_cusick"
    title = "Cusick drape: 30 cm cotton-like disc on an 18 cm support, 3 s settle, 3 mesh refinements"
    horizon_s = 3.0
    accuracy_unit = "dc"
    accuracy_ref = ("no closed form: |DC(finest) - DC(middle)| mesh convergence (DC = (As - Ad)/(AD - Ad), Cusick "
                    "1965, ISO 9073-9) when nworld >= 3, else NaN; strain, intersections, spread in metrics")
    configs = (
        Config("vbd", "vbd", dt=1 / 60, substeps=10, params={"iterations": 10, "self_contact": True, "tau": 0.02,
                                                             "self_contact_radius": 0.0015,
                                                             "self_contact_margin": 0.002}),
        Config("style3d", "style3d", dt=1 / 60, substeps=4, params={"iterations": 10, "tau": 0.02}),
        Config("xpbd", "xpbd", dt=1 / 60, substeps=20, params={"iterations": 10}),
        Config("semi_implicit", "semi_implicit", dt=1 / 600, substeps=32, params={"tau": 2.0e-5}, max_nworld=1024),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        tau = config.params.get("tau", 0.0)
        tmpls, meshes = [], []
        for hc in DR_HS:
            b = new_builder(config.solver)
            v, t = GEO.disc_mesh(DR_R, hc)
            v = v + np.array([0.0, 0.0, DR_RADIUS + 0.001])
            add_cloth(b, config.solver, v, t, DR_MAT, hc, tau, DR_RADIUS)
            if config.solver == "vbd":
                b.color(include_bending=True)
            tmpls.append(b)
            meshes.append((v, t))
        asm = assemble(tmpls, nworld, register=register_for(config.solver))
        cyl = newton.ModelBuilder.ShapeConfig(mu=0.5)
        asm.builder.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                                       radius=DR_SUPPORT, half_height=0.05, cfg=cyl)
        asm.builder.add_ground_plane(height=-0.4)
        model = finalize_cloth_model(asm, device)
        solver = make_particle_solver(config, model)
        pipe = newton.CollisionPipeline(model)
        s0, s1, ctrl = model.state(), model.state(), model.control()
        drv = Driver(device, config.dt / config.substeps)
        drv.gravity_ramp(model, 0.0, 0.5)
        return Built(model, solver, s0, s1, ctrl, pipe.contacts(), pipe, nworld=nworld,
                     extra={"driver": drv, "asm": asm, "meshes": meshes})

    def finish(self, b: Built, rec: dict) -> None:
        flag_blowup(b.state_0, rec)
        q = b.state_0.particle_q.numpy().astype(np.float64)
        asm, meshes = b.extra["asm"], b.extra["meshes"]
        n_sel = min(b.nworld, len(DR_HS))
        a_d = math.pi * DR_SUPPORT**2
        per, strain, pen = {}, [], []
        for w in range(n_sel):
            v0, t = meshes[asm.cond[w]]
            x = q[asm.particle_start[w]: asm.particle_start[w] + len(v0)]
            a_big = GEO.mesh_area(v0, t)
            a_s = GEO.silhouette_area(x[:, :2], t, 1e-3)
            dc = (a_s - a_d) / (a_big - a_d)
            st = 100 * GEO.max_edge_strain(v0, x, GEO.mesh_edges(t))
            on = np.linalg.norm(x[:, :2], axis=1) < DR_SUPPORT - 0.003
            p = np.clip(DR_RADIUS - x[on, 2], 0.0, None) * 1e3 if on.any() else np.zeros(1)
            per[f"h{DR_HS[asm.cond[w]] * 1e3:g}mm"] = {"dc": dc, "node_count": len(v0), "max_strain_pct": st,
                                                    "pen_max_mm": float(p.max())}
            strain.append(st)
            pen.append(p)
        inter = GEO.self_intersections(b.model, b.state_0)
        tri_world = b.model.particle_world.numpy()[b.model.tri_indices.numpy()[:, 0]]
        n_int = int(inter[tri_world < n_sel].sum())
        dcs = [per[f"h{hc * 1e3:g}mm"]["dc"] for hc in DR_HS[:n_sel]]
        conv = abs(dcs[2] - dcs[1]) if n_sel >= 3 else float("nan")
        rec["penetration_p99_mm"] = float(np.percentile(np.concatenate(pen), 99))
        law_record(rec, conv, dc=dcs[-1], dc_per_mesh=dcs, dc_convergence=conv, max_strain_pct=max(strain),
                   self_intersections=n_int, per_condition=per, conditions_covered=bool(b.nworld >= len(DR_HS)),
                   end_max_speed=max_speed(b.state_0),
                   law_pass=bool(np.isfinite(conv) and conv < 0.02 and max(strain) <= 1.0 and n_int == 0),
                   bending_length_cm=100 * DR_MAT.c, fit_dt_only=False)


# ---- D04 strip stretch, D1388 cantilever, hanging strip --------------------------------------------------------
S4_MAT = ClothMat(E_s=200.0)
ST_L, ST_W, ST_H, ST_EPS = 0.20, 0.05, 0.005, 0.05  # stretch strip length, width, cell, target strain
HG_L, HG_W, HG_H = 0.30, 0.05, 0.005  # hanging strip
CL_W, CL_N = 0.025, 20  # D1388 strip width, cells over the overhang
S4_RADIUS = 0.001
T_GRAV, T_PULL0, T_PULL1 = 0.5, 0.5, 2.0


def _strip_layout(mat: ClothMat) -> dict:
    l_over = 2.0 * mat.c  # D1388: tip reaches the 41.5 deg line when the overhang is twice the bending length
    h_cl = l_over / CL_N
    return {"l_over": l_over, "h_cl": h_cl}


def _d04_world(solver: str, tau: float) -> tuple[newton.ModelBuilder, dict]:
    b = new_builder(solver)
    mat = S4_MAT
    idx = {}
    # stretch strip in the plane z = 1, x along the strip, left edge pinned; gravity compensated (driver)
    nx, ny = round(ST_L / ST_H), round(ST_W / ST_H)
    v, t = GEO.grid_mesh(nx, ny, ST_H, ST_H)
    s = add_cloth(b, solver, v + np.array([0.0, 0.0, 1.0]), t, mat, ST_H, tau, S4_RADIUS)
    col = lambda i: [s + j * (nx + 1) + i for j in range(ny + 1)]  # noqa: E731
    pin_particles(b, col(0))
    idx.update(st_all=list(range(s, s + len(v))), st_right=col(nx), st_mid_right=s + (ny // 2) * (nx + 1) + nx,
               st_mid_left=s + (ny // 2) * (nx + 1), st_ctop=s + ny * (nx + 1) + nx // 2, st_cbot=s + nx // 2)
    # D1388 cantilever: two pinned columns on the platform (x <= 0), overhang along +x, at y = 0.3
    lay = _strip_layout(mat)
    h_cl = lay["h_cl"]
    ny_c = max(2, round(CL_W / h_cl))
    v, t = GEO.grid_mesh(CL_N + 2, ny_c, h_cl, CL_W / ny_c)
    s = add_cloth(b, solver, v + np.array([-2 * h_cl, 0.3, 1.0]), t, mat, h_cl, tau, S4_RADIUS)
    rowlen = CL_N + 3
    pin_particles(b, [s + j * rowlen + i for j in range(ny_c + 1) for i in (0, 1, 2)])
    idx.update(cl_tip=s + (ny_c // 2) * rowlen + CL_N + 2, cl_edge=s + (ny_c // 2) * rowlen + 2)
    # hanging strip in the xz plane at y = 0.6, top edge pinned
    nx, nz = round(HG_W / HG_H), round(HG_L / HG_H)
    v, t = GEO.grid_mesh(nx, nz, HG_H, HG_H)
    v = np.stack([v[:, 0], np.full(len(v), 0.6), 1.0 - HG_L + v[:, 1]], axis=1)
    s = add_cloth(b, solver, v, t, mat, HG_H, tau, S4_RADIUS)
    pin_particles(b, [s + nz * (nx + 1) + i for i in range(nx + 1)])
    idx.update(hg_bot=s + nx // 2, hg_top=s + nz * (nx + 1) + nx // 2)
    if solver == "vbd":
        b.color(include_bending=True)
    idx["mass"] = np.asarray(b.particle_mass, dtype=np.float64)
    return b, idx


class ClothStretchBend(DeformScenario):
    id = "D04_cloth_stretch_bend"
    title = "Strip stretch to 5 % (load controlled), ASTM D1388 41.5 deg cantilever, hanging strip"
    horizon_s = 3.0
    accuracy_unit = "pct"
    accuracy_ref = ("max of |stretch stiffness err|, |bending length err| (D1388: c = l/2 at 41.5 deg, inverted on "
                    "the heavy elastica), |hanging elongation err| (W L^2 / 2 E_s); Poisson in metrics")
    configs = (
        Config("style3d", "style3d", dt=1 / 60, substeps=4, params={"iterations": 10, "tau": 0.02}),
        Config("vbd", "vbd", dt=1 / 60, substeps=10, params={"iterations": 20, "tau": 0.02}),
        # realistic 150 gsm cloth at 5 mm cells is 57x stiffer than its inertia at h = 1/600 and VBD diverges there
        # (bench/notes/deform.md); this config shows the h = 1/6000 behaviour
        Config("vbd_s100", "vbd", dt=1 / 60, substeps=100, params={"iterations": 10, "tau": 0.02}, max_nworld=1024),
        Config("xpbd", "xpbd", dt=1 / 60, substeps=20, params={"iterations": 10}),
        Config("semi_implicit", "semi_implicit", dt=1 / 600, substeps=32, params={"tau": 2.0e-5}, max_nworld=1024),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        tmpl, idx = _d04_world(config.solver, config.params.get("tau", 0.0))
        asm = assemble([tmpl], nworld, register=register_for(config.solver))
        model = finalize_cloth_model(asm, device)
        solver = make_particle_solver(config, model)
        pipe = newton.CollisionPipeline(model)  # no shapes; SolverStyle3D.step requires a Contacts object
        s0, s1, ctrl = model.state(), model.state(), model.control()
        h = config.dt / config.substeps
        drv = Driver(device, h)
        drv.gravity_ramp(model, 0.0, T_GRAV)
        worlds = list(range(nworld))
        comp = world_ids(asm.particle_start, idx["st_all"], worlds).ravel()
        mass = np.tile(idx["mass"][idx["st_all"]], nworld)
        drv.particle_force(comp, np.outer(mass, [0.0, 0.0, G]), 0.0, T_GRAV)  # cancels the gravity ramp exactly
        F5 = S4_MAT.E_s * ST_W * ST_EPS
        right = idx["st_right"]
        wts = np.ones(len(right))
        wts[[0, -1]] = 0.5
        f = np.zeros((len(right), 3))
        f[:, 0] = F5 * wts / wts.sum()
        drv.particle_force(world_ids(asm.particle_start, right, worlds).ravel(), np.tile(f, (nworld, 1)), T_PULL0, T_PULL1)
        names = ["st_mid_right", "st_mid_left", "st_ctop", "st_cbot", "cl_tip", "cl_edge", "hg_bot", "hg_top"]
        rw = recorded_worlds(nworld)
        stride = max(1, round(0.01 / h))
        drv.record_particles("pts", world_ids(asm.particle_start, [idx[k] for k in names], rw).ravel(), stride,
                             rec_rows(self.horizon_s, h, stride))
        x0 = s0.particle_q.numpy()[[asm.particle_start[0] + idx[k] for k in names]].astype(np.float64)
        return Built(model, solver, s0, s1, ctrl, pipe.contacts(), pipe, nworld=nworld,
                     extra={"driver": drv, "names": names, "x0": x0, "nrec": len(rw), "F5": F5})

    def finish(self, b: Built, rec: dict) -> None:
        flag_blowup(b.state_0, rec)
        t, xyz = b.extra["driver"].trace("pts")
        names, x0 = b.extra["names"], b.extra["x0"]
        xyz = xyz.reshape(len(t), b.extra["nrec"], len(names), 3)[:, 0]
        k = {n: i for i, n in enumerate(names)}
        mat, F5 = S4_MAT, b.extra["F5"]
        # stretch: eps_x from the mid line, applied force from the eased schedule
        eps = (xyz[:, k["st_mid_right"], 0] - xyz[:, k["st_mid_left"], 0]) / ST_L - 1.0
        u = np.clip((t - T_PULL0) / (T_PULL1 - T_PULL0), 0, 1)
        F = F5 * u * u * (3 - 2 * u)
        sel = (F > 0.02 * F5) & (F <= 0.3 * F5) & (t <= T_PULL1)
        k_init = float(np.sum(F[sel] * eps[sel]) / np.sum(eps[sel] ** 2)) / ST_W if sel.sum() >= 2 else float("nan")
        eps_end = float(eps[-1])
        width0 = x0[k["st_ctop"], 1] - x0[k["st_cbot"], 1]
        eps_y = (xyz[-1, k["st_ctop"], 1] - xyz[-1, k["st_cbot"], 1]) / width0 - 1.0
        nu_meas = -eps_y / eps_end if abs(eps_end) > 1e-9 else float("nan")
        stretch_err = pct(k_init, mat.E_s)
        # D1388
        lay = _strip_layout(mat)
        d = xyz[-1, k["cl_edge"]] - xyz[-1, k["cl_tip"]]
        ang = math.atan2(d[2], -d[0])
        g_meas = LAW.gamma_for_chord_angle(ang) if ang > 0 else float("nan")
        c_meas = lay["l_over"] / g_meas ** (1 / 3) if np.isfinite(g_meas) else float("nan")
        bend_err = pct(c_meas, mat.c)
        # hanging strip
        L_now = xyz[-1, k["hg_top"], 2] - xyz[-1, k["hg_bot"], 2]
        L_rest = x0[k["hg_top"], 2] - x0[k["hg_bot"], 2]
        dl, dl_ref = L_now - L_rest, mat.W * HG_L**2 / (2 * mat.E_s)
        hang_err = pct(dl, dl_ref)
        errs = [abs(e) for e in (stretch_err, bend_err, hang_err)]
        head = max(errs) if all(np.isfinite(errs)) else float("nan")
        law_record(rec, head, stretch_k_err_pct=stretch_err, stretch_k_init=k_init, stretch_k_secant=F5 / ST_W / eps_end,
                   stretch_eps_end=eps_end, poisson_meas=nu_meas, poisson_ref=mat.nu, poisson_err=nu_meas - mat.nu,
                   d1388_angle_deg=math.degrees(ang), bend_length_cm=100 * c_meas, bend_length_ref_cm=100 * mat.c,
                   bend_length_err_pct=bend_err, flexural_rigidity_err_pct=pct(c_meas**3, mat.c**3),
                   hang_elong_mm=1e3 * dl, hang_elong_ref_mm=1e3 * dl_ref, hang_err_pct=hang_err,
                   end_max_speed=max_speed(b.state_0), law_pass=bool(np.isfinite(head) and head <= 5.0),
                   fit_dt_only=False)


# ---- D05 parallel-jaw cloth hold --------------------------------------------------------------------------------
GR_MU, GR_FACTORS = 0.5, (0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0)
GR_OBJECTS = ("cloth20", "grid1x1")
GR_MASS = 0.006  # both objects [kg] -> F* = m g / (2 mu) = 0.0589 N
GR_ZC, GR_PAD, GR_RADIUS, GR_GAP = 0.5, 0.01, 0.001, 0.0015  # cloth top, pad half extent, particle radius, initial gap
T_CLOSE, T_G0, T_G1, T_LIFT0, T_LIFT1, LIFT = 0.2, 0.3, 0.8, 1.2, 2.2, 0.05
GR_MAT = ClothMat(rho_s=GR_MASS / 0.04)


@wp.kernel
def k_gantry_lift(counter: wp.array(dtype=wp.int32), h: float, dofs: wp.array(dtype=wp.int32), z0: float,
                  lift: float, t0: float, t1: float, target: wp.array(dtype=float)):
    i = wp.tid()
    t = float(counter[0]) * h
    u = wp.clamp((t - t0) / (t1 - t0), 0.0, 1.0)
    target[dofs[i]] = z0 + lift * 0.5 * (1.0 - wp.cos(wp.pi * u))


def _d05_world(solver: str, obj: str, factor: float, full_surface: bool) -> tuple[newton.ModelBuilder, dict]:
    b = new_builder(solver)
    z_pinch = GR_ZC - GR_PAD - 0.002
    z_g = z_pinch + 0.05
    gantry = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z_g), wp.quat_identity()), mass=0.01,
                        inertia=wp.mat33(*(np.eye(3) * 1e-5).ravel()), label="gantry")
    jcfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=GR_MU)
    if full_surface:
        jcfg.configure_sdf(force_sdf=True)
    off = GR_RADIUS + GR_GAP + GR_PAD / 2
    jaws, joints = [], [b.add_joint_prismatic(-1, gantry, axis=wp.vec3(0.0, 0.0, 1.0), target_ke=5.0e4,
                                              target_kd=5.0e3, target_pos=z_g, label="gantry_z")]
    for side in (-1.0, 1.0):
        jaw = b.add_link(xform=wp.transform(wp.vec3(0.0, side * off, z_pinch), wp.quat_identity()), label=f"jaw{side:+.0f}")
        b.add_shape_box(jaw, hx=GR_PAD, hy=GR_PAD / 2, hz=GR_PAD, cfg=jcfg)
        joints.append(b.add_joint_prismatic(gantry, jaw, axis=wp.vec3(0.0, side, 0.0),
                                            parent_xform=wp.transform(wp.vec3(0.0, 0.0, z_pinch - z_g), wp.quat_identity()),
                                            child_xform=wp.transform_identity(),
                                            target_ke=0.0, target_kd=5.0, label=f"jaw_joint{side:+.0f}"))
        jaws.append(jaw)
    b.add_articulation(joints, label="gripper")
    b.joint_q[0] = z_g
    b.joint_q[1] = off
    b.joint_q[2] = off
    if obj == "cloth20":
        v, t = GEO.grid_mesh(20, 20, 0.01, 0.01)
        hc = 0.01
    else:
        v, t = GEO.grid_mesh(1, 1, 0.24, 0.09)
        hc = 0.09
    ext = np.ptp(v, axis=0)
    v = np.stack([v[:, 0] - 0.5 * ext[0], np.zeros(len(v)), GR_ZC - ext[1] + v[:, 1]], axis=1)
    if obj == "grid1x1":  # a single 24 x 9 cm quad (two triangles) centred on the pinch line
        v[:, 2] -= v[:, 2].mean() - z_pinch
    mat = ClothMat(rho_s=GR_MASS / GEO.mesh_area(v, t))
    s = add_cloth(b, solver, v, t, mat, hc, 0.02 if solver == "vbd" else 0.0, GR_RADIUS)
    near = s + int(np.argmin(np.linalg.norm(v - np.array([0.0, 0.0, z_pinch]), axis=1)))
    if solver == "vbd":
        b.color(include_bending=True)
    fstar = GR_MASS * G / (2 * GR_MU)
    return b, {"jaws": jaws, "gantry_dof": 0, "near": near, "n_cloth": len(v), "cloth_start": s, "fstar": fstar,
               "force": factor * fstar, "z_g": z_g}


class ClothGripperHold(DeformScenario):
    id = "D05_cloth_gripper_hold"
    title = "Parallel jaw pinch, lift and hold of a 20x20 cm cloth and a 1x1 soft grid; grip force sweep per world"
    horizon_s = 3.0
    accuracy_unit = "pct"
    accuracy_ref = ("Coulomb hold: held iff 2 mu Fn >= m g; onset grip factor vs mg/(2 mu) over the sweep "
                    "(0.5 .. 4), cloth20 object; slip after lift, intersections, penetration in metrics")
    configs = (
        Config("vbd_fullsurface", "vbd", dt=1 / 60, substeps=20,
               params={"iterations": 5, "full_surface": True, "self_contact": True, "self_contact_radius": 0.0008,
                       "self_contact_margin": 0.001}),
        Config("vbd_vertex_only", "vbd", dt=1 / 60, substeps=20,
               params={"iterations": 5, "full_surface": False, "self_contact": True, "self_contact_radius": 0.0008,
                       "self_contact_margin": 0.001}),
        Config("xpbd", "xpbd", dt=1 / 60, substeps=20, params={"iterations": 10, "full_surface": False}),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        fs = config.params.get("full_surface", False)
        conds = [(o, f) for o in GR_OBJECTS for f in GR_FACTORS]
        tmpls = [_d05_world(config.solver, o, f, fs) for o, f in conds]
        asm = assemble([t[0] for t in tmpls], nworld, gravity=-G)
        model = finalize_cloth_model(asm, device, mu=GR_MU)
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        S = newton.solvers
        if config.solver == "vbd":
            kw = dict(iterations=config.params["iterations"], rigid_body_contact_buffer_size=512)
            if config.params.get("self_contact"):
                kw.update(particle_enable_self_contact=True,
                          particle_self_contact_radius=config.params["self_contact_radius"],
                          particle_self_contact_margin=config.params["self_contact_margin"])
            solver = S.SolverVBD(model, **kw)
        else:
            solver = S.SolverXPBD(model, iterations=config.params["iterations"])
        pipe = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.005,
                                        enable_rigid_soft_full_surface_contact=fs)
        s0, s1, ctrl = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        wp.copy(s1.body_q, s0.body_q)
        h = config.dt / config.substeps
        drv = Driver(device, h)
        drv.gravity_ramp(model, T_G0, T_G1)
        jaw_ids, jaw_f, dofs = [], [], []
        for w in range(nworld):
            info = tmpls[asm.cond[w]][1]
            for side, jaw in zip((-1.0, 1.0), info["jaws"]):
                jaw_ids.append(asm.body_start[w] + jaw)
                jaw_f.append((0.0, -side * info["force"], 0.0))
            dofs.append(asm.dof_start[w] + info["gantry_dof"])
        drv.body_force(jaw_ids, np.asarray(jaw_f), 0.0, T_CLOSE)
        dofs_w = wp.array(np.asarray(dofs, np.int32), dtype=wp.int32, device=device)
        z_g = tmpls[0][1]["z_g"]
        drv.pre.append(lambda a, c: wp.launch(k_gantry_lift, dim=len(dofs), device=device,
                                              inputs=[drv.counter, h, dofs_w, z_g, LIFT, T_LIFT0, T_LIFT1,
                                                      ctrl.joint_target_q]))
        rw = recorded_worlds(nworld, len(conds))
        stride = max(1, round(0.01 / h))
        rows = rec_rows(self.horizon_s, h, stride)
        drv.record_particles("near", [asm.particle_start[w] + tmpls[asm.cond[w]][1]["near"] for w in rw], stride, rows)
        drv.record_bodies("jaw", [asm.body_start[w] + tmpls[asm.cond[w]][1]["jaws"][0] for w in rw],
                          np.zeros((len(rw), 3)), stride, rows)
        return Built(model, solver, s0, s1, ctrl, pipe.contacts(), pipe, nworld=nworld,
                     extra={"driver": drv, "asm": asm, "conds": conds, "tmpls": [t[1] for t in tmpls], "rw": rw})

    def finish(self, b: Built, rec: dict) -> None:
        flag_blowup(b.state_0, rec)
        t, near = b.extra["driver"].trace("near")
        _, jaw = b.extra["driver"].trace("jaw")
        asm, conds, tmpls, rw = b.extra["asm"], b.extra["conds"], b.extra["tmpls"], b.extra["rw"]
        rel = near[:, :, 2] - jaw[:, :, 2]
        i_ref = np.searchsorted(t, T_LIFT0 - 0.1)
        per, held = {}, {o: {} for o in GR_OBJECTS}
        for j, w in enumerate(rw[: len(conds)]):
            obj, f = conds[asm.cond[w]]
            if i_ref >= len(t):
                continue
            slip = 1e3 * float(rel[i_ref, j] - rel[-1, j])
            slip_ramp = 1e3 * float(rel[np.searchsorted(t, T_G0), j] - rel[i_ref, j]) if i_ref > 0 else float("nan")
            ok = abs(slip) < 2.0 and slip_ramp < 5.0
            per[f"{obj}_f{f}"] = {"slip_mm": slip, "slip_gravity_ramp_mm": slip_ramp, "held": bool(ok)}
            held[obj][f] = ok
        onset = {}
        for obj, d in held.items():
            fs = sorted(d)
            if len(fs) < len(GR_FACTORS):
                onset[obj] = float("nan")
                continue
            ok_from = [f for i, f in enumerate(fs) if all(d[g] for g in fs[i:])]
            onset[obj] = min(ok_from) if ok_from else float("nan")
        err = pct(onset["cloth20"], 1.0)
        pen = self._jaw_penetration(b, rw)
        inter = GEO.self_intersections(b.model, b.state_0)
        rec["penetration_p99_mm"] = float(np.percentile(pen, 99)) if pen.size else 0.0
        slips = [v["slip_mm"] for k, v in per.items() if k.startswith("cloth20") and v["held"]]
        law_record(rec, err, hold_onset_err_pct=err, onset_factor=onset, per_condition=per,
                   slip_mm=max(np.abs(slips)) if slips else float("nan"), intersections=int(inter.sum()),
                   pen_max_mm=float(pen.max()) if pen.size else 0.0,
                   conditions_covered=bool(b.nworld >= len(conds)),
                   law_pass=bool(np.isfinite(err) and abs(err) <= 5.0 and int(inter.sum()) == 0), fit_dt_only=False)

    @staticmethod
    def _jaw_penetration(b: Built, worlds) -> np.ndarray:
        """Depth [mm] of cloth particles (radius GR_RADIUS) inside the jaw boxes, recorded worlds."""
        q = b.state_0.particle_q.numpy().astype(np.float64)
        bq = b.state_0.body_q.numpy().astype(np.float64)
        asm, tmpls = b.extra["asm"], b.extra["tmpls"]
        out = []
        half = np.array([GR_PAD, GR_PAD / 2, GR_PAD])
        for w in worlds[: len(b.extra["conds"])]:
            info = tmpls[asm.cond[w]]
            s = asm.particle_start[w] + info["cloth_start"]
            x = q[s: s + info["n_cloth"]]
            for jaw in info["jaws"]:
                p = x - bq[asm.body_start[w] + jaw, :3]  # jaws stay axis aligned (prismatic joints)
                d = np.abs(p) - half
                sdf = np.linalg.norm(np.clip(d, 0, None), axis=1) + np.minimum(d.max(axis=1), 0.0)
                out.append(np.clip(GR_RADIUS - sdf, 0.0, None) * 1e3)
        return np.concatenate(out) if out else np.zeros(0)


SCENARIOS_CLOTH = [ClothDrape(), ClothStretchBend(), ClothGripperHold()]
