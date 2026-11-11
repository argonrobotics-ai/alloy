# SPDX-License-Identifier: Apache-2.0
"""D01 cable cantilever + hanging chain, D02 bend/twist boundary values + Michell ring + capstan.

Scenes follow newton@ac7f7830 examples: vbd/example_cable_bend_stiffness.py (kinematic root, add_rod),
vbd/example_cable_bend_twist_analytic.py, vbd/example_cable_michell_threshold.py,
tendon/example_tendon_capstan_friction.py; the cantilever material follows [internal reference]
(r = 3 mm, rho = 1200 kg/m^3, stiffness set by gamma = w L^3 / EI) at L = 0.5 m.
"""
from __future__ import annotations

import math

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench.harness import Built, Config
from bench.scenarios import deform_laws as LAW
from bench.scenarios.deform_common import (G, DeformScenario, Driver, assemble, law_record, make_kinematic_body, pct,
                                           period_from_crossings, rec_rows, recorded_worlds, world_ids)

# ---- D01 material and layout ------------------------------------------------------------------------------------
L, R, RHO, NU = 0.5, 0.003, 1200.0, 0.3
M_LIN = RHO * math.pi * R**2
W_LIN = M_LIN * G
AREA, I_SEC = math.pi * R**2, math.pi * R**4 / 4.0
DAMP_TIME = 0.1  # [s] stiffness-proportional damping on the static cables (bend/twist/stretch k * tau)
Z0 = 1.0  # cantilever root height [m]
CHAIN_TOP, CHAIN_Y, CHAIN_AMP = 1.2, 1.0, 0.01  # pin height, lateral offset, first-mode amplitude at the free end
# (name, gamma = w L^3 / EI, damped, y offset)
CANTILEVERS = (("static_small", 0.5, True, 0.0), ("static_large", 5.0, True, 0.25), ("pluck", 0.5, False, 0.5))
OBJ_NAMES = [c[0] for c in CANTILEVERS] + ["chain"]


def _ei(gamma: float) -> float:
    return W_LIN * L**3 / gamma


def _cable_props(gamma: float, nseg: int, damped: bool) -> dict:
    """Per-joint stiffness of a discretised rod: k_bend = EI/l, k_twist = GJ/l, k_stretch = EA/l (proto/cable)."""
    EI = _ei(gamma)
    E = EI / I_SEC
    l = L / nseg
    tau = DAMP_TIME if damped else 0.0
    return {"l": l, "k_bend": EI / l, "k_twist": EI / (1 + NU) / l, "k_stretch": E * AREA / l, "tau": tau}


def _seg_mass(l: float) -> tuple[float, float, float]:
    """Mass, transverse and axial inertia of a solid cylinder segment."""
    m = M_LIN * l
    return m, m * (3 * R**2 + l**2) / 12.0, 0.5 * m * R**2


def _vbd_rod_mass(b: newton.ModelBuilder, bodies, l: float) -> None:
    m, it, ia = _seg_mass(l)
    for body in bodies:  # body frame: local z along the segment (add_rod capsules)
        b.body_mass[body] = m
        b.body_inv_mass[body] = 1.0 / m
        b.body_inertia[body] = wp.mat33(it, 0, 0, 0, it, 0, 0, 0, ia)
        b.body_inv_inertia[body] = wp.mat33(1 / it, 0, 0, 0, 1 / it, 0, 0, 0, 1 / ia)


def _chain_angles(nseg: int) -> np.ndarray:
    """Absolute link angles of the hanging chain's first mode, free-end lateral offset = CHAIN_AMP."""
    _, shape = LAW.hanging_chain_modes(nseg, L, M_LIN, R, G)
    return CHAIN_AMP * shape


def _double_clamp_joint(b: newton.ModelBuilder, joint: int) -> None:
    """Ghost-segment clamp: the hinge at x = 0 carries half a Voronoi length, so bend/twist gains double."""
    d0 = b.joint_qd_start[joint]
    for off in (3, 4, 5):  # ROD dof layout: shear x, shear y, stretch, bend x, bend y, twist
        b.joint_target_ke[d0 + off] *= 2.0
        b.joint_target_kd[d0 + off] *= 2.0


def _d01_vbd_world(nseg: int) -> tuple[newton.ModelBuilder, list, np.ndarray]:
    """add_rod cables with one extra kinematic ghost segment on [-l, 0]; the clamp hinge sits at x = 0."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False, density=RHO)
    tips, local = [], []
    for name, gamma, damped, y in CANTILEVERS:
        p = _cable_props(gamma, nseg, damped)
        l = p["l"]
        rod = newton.Rod.create_straight(start=wp.vec3(-l, y, Z0), direction=wp.vec3(1.0, 0.0, 0.0), length=L + l,
                                         segment_count=nseg + 1, radius=R)
        bodies, joints = b.add_rod(rod=rod, cfg=cfg, stretch_stiffness=p["k_stretch"], bend_stiffness=p["k_bend"],
                                   twist_stiffness=p["k_twist"], stretch_damping=p["k_stretch"] * p["tau"],
                                   bend_damping=p["k_bend"] * p["tau"], twist_damping=p["k_twist"] * p["tau"],
                                   label=name)
        _vbd_rod_mass(b, bodies, l)
        make_kinematic_body(b, bodies[0])
        _double_clamp_joint(b, joints[0])
        tips.append(bodies[-1])
        local.append((0.0, 0.0, 0.5 * l))
    # hanging chain: one kinematic segment above the pin, nseg free segments on the first-mode shape, no bending
    l = L / nseg
    phi = _chain_angles(nseg)
    pts = [np.array([0.0, CHAIN_Y, CHAIN_TOP + l]), np.array([0.0, CHAIN_Y, CHAIN_TOP])]
    for a in phi:
        pts.append(pts[-1] + l * np.array([math.sin(a), 0.0, -math.cos(a)]))
    p = _cable_props(0.5, nseg, False)
    rod = newton.Rod(np.asarray(pts, np.float32), radius=R)
    bodies, _ = b.add_rod(rod=rod, cfg=cfg, stretch_stiffness=p["k_stretch"], bend_stiffness=0.0,
                          twist_stiffness=0.0, label="chain")
    _vbd_rod_mass(b, bodies, l)
    make_kinematic_body(b, bodies[0])
    tips.append(bodies[-1])
    local.append((0.0, 0.0, 0.5 * l))
    b.color()
    return b, tips, np.asarray(local)


def _joint_drive(flavor: str, k: float, kd: float) -> dict:
    """Rotational spring k [N m/rad] (+ damping kd) on a revolute joint for each joint-space solver."""
    if flavor == "mujoco":
        return {"custom_attributes": {"mujoco:dof_passive_stiffness": [k]}, "damping": kd}
    # Featherstone applies drives explicitly: damping on the high chain modes is unstable at any affordable step,
    # so its static cables run undamped and their equilibrium is the time average (finish()). XPBD maps ke to a
    # compliance 1/ke (xpbd/kernels.py:1475) and takes kd as constraint damping.
    return {"target_ke": k, "target_kd": kd if flavor == "xpbd" else 0.0, "target_pos": 0.0}


def _d01_joint_world(nseg: int, flavor: str) -> tuple[newton.ModelBuilder, list, np.ndarray]:
    """Rigid capsule chain, revolute (Y) joints at nodes 0..N-1; the joint at x = 0 is the clamp (2 EI / l)."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    if flavor == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    l = L / nseg
    m, it, ia = _seg_mass(l)
    tips, local = [], []
    for name, gamma, damped, y in CANTILEVERS:
        p = _cable_props(gamma, nseg, damped)
        parent, joints = -1, []
        for i in range(nseg):
            k = p["k_bend"] * (2.0 if i == 0 else 1.0)
            body = b.add_link(xform=wp.transform(wp.vec3((i + 0.5) * l, y, Z0), wp.quat_identity()), mass=m,
                              inertia=wp.mat33(ia, 0, 0, 0, it, 0, 0, 0, it), label=f"{name}_{i}")
            pxf = wp.transform(wp.vec3(0.0, y, Z0) if parent < 0 else wp.vec3(0.5 * l, 0.0, 0.0), wp.quat_identity())
            joints.append(b.add_joint_revolute(parent, body, parent_xform=pxf,
                                               child_xform=wp.transform(wp.vec3(-0.5 * l, 0.0, 0.0), wp.quat_identity()),
                                               axis=(0.0, 1.0, 0.0), **_joint_drive(flavor, k, k * p["tau"])))
            parent = body
        b.add_articulation(joints)
        tips.append(parent)
        local.append((0.5 * l, 0.0, 0.0))
    phi = _chain_angles(nseg)
    parent, joints = -1, []
    for i in range(nseg):
        body = b.add_link(xform=wp.transform(wp.vec3(0.0, CHAIN_Y, CHAIN_TOP - (i + 0.5) * l), wp.quat_identity()),
                          mass=m, inertia=wp.mat33(it, 0, 0, 0, it, 0, 0, 0, ia), label=f"chain_{i}")
        pxf = wp.transform(wp.vec3(0.0, CHAIN_Y, CHAIN_TOP) if parent < 0 else wp.vec3(0.0, 0.0, -0.5 * l),
                           wp.quat_identity())
        joints.append(b.add_joint_revolute(parent, body, parent_xform=pxf,
                                           child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * l), wp.quat_identity()),
                                           axis=(0.0, 1.0, 0.0)))
        parent = body
    b.add_articulation(joints)
    q0 = len(b.joint_q) - nseg
    rel = np.diff(np.concatenate([[0.0], -phi]))  # +Y rotation swings the link toward -x; mode offsets are +x
    for i in range(nseg):
        b.joint_q[q0 + i] = float(rel[i])
    tips.append(parent)
    local.append((0.0, 0.0, -0.5 * l))
    return b, tips, np.asarray(local)


XPBD_BEND_SCALE = 1e-4  # (i, i+2) spring stiffness / stretch stiffness; >= 1e-2 diverges, 1e-3 diverges by 4 s (notes)


def _d01_particle_world(nseg: int) -> tuple[newton.ModelBuilder, list, np.ndarray]:
    """Particle chains: stretch springs (i, i+1) and 'bending' springs (i, i+2); ghost node at x = -l and node 0 pinned.

    A spring across two segments has no linear-order bending stiffness, so this config cannot represent EI; it is
    kept as the 'springs only' baseline. Spring damping is off: kd/(ke dt) of order 60 diverges in XPBD (probe in
    bench/notes/deform.md), and the bending springs are 1e-4 of the stretch stiffness for the same reason.
    """
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    l = L / nseg
    tips = []
    for name, gamma, damped, y in CANTILEVERS:
        p = _cable_props(gamma, nseg, damped)
        ids = [b.add_particle(pos=wp.vec3((i - 1) * l, y, Z0), vel=wp.vec3(0.0),
                              mass=M_LIN * l * (0.5 if i == nseg + 1 else 1.0), radius=1e-4) for i in range(nseg + 2)]
        for i in (0, 1):
            b.particle_mass[ids[i]] = 0.0
        for i in range(nseg + 1):
            b.add_spring(ids[i], ids[i + 1], p["k_stretch"], 0.0, 0.0)
        for i in range(nseg):
            b.add_spring(ids[i], ids[i + 2], XPBD_BEND_SCALE * p["k_stretch"], 0.0, 0.0)
        tips.append(ids[-1])
    _, shape = LAW.hanging_lumped_modes(nseg, L, M_LIN, G)
    p = _cable_props(0.5, nseg, False)
    ids = [b.add_particle(pos=wp.vec3(0.0, CHAIN_Y, CHAIN_TOP), vel=wp.vec3(0.0), mass=0.0, radius=1e-4)]
    for i in range(1, nseg + 1):
        ids.append(b.add_particle(pos=wp.vec3(CHAIN_AMP * shape[i - 1], CHAIN_Y, CHAIN_TOP - i * l), vel=wp.vec3(0.0),
                                  mass=M_LIN * l * (0.5 if i == nseg else 1.0), radius=1e-4))
    for i in range(nseg):
        b.add_spring(ids[i], ids[i + 1], p["k_stretch"], 0.0, 0.0)
    tips.append(ids[-1])
    return b, tips, np.zeros((len(tips), 3))


class CableCantilever(DeformScenario):
    id = "D01_cable_cantilever"
    title = "0.5 m clamped cable: static tip drop (gamma 0.5 and 5), pluck from straight, hanging chain"
    horizon_s = 4.0
    accuracy_unit = "pct"
    accuracy_ref = ("static tip drop at gamma = wL^3/EI = 5 vs the Kirchhoff elastica (numerical, continuum); "
                    "exact discrete chain, wL^4/8EI, first cantilever mode and w1 = 1.2024 sqrt(g/L) in metrics")
    configs = tuple(
        c for n in (20, 40, 80) for c in (
            Config(f"vbd_rod_it10_n{n}", "vbd", dt=1 / 60, substeps=10, params={"nseg": n, "iterations": 10}),
            Config(f"vbd_rod_it50_n{n}", "vbd", dt=1 / 60, substeps=10, params={"nseg": n, "iterations": 50}),
            Config(f"joint_chain_mujoco_n{n}", "mujoco", dt=1 / 600, substeps=1,
                   params={"nseg": n, "integrator": "discrete", "solver": "newton", "iterations": 50}),
            Config(f"joint_chain_featherstone_n{n}", "featherstone", dt=1 / 600, substeps={20: 24, 40: 80, 80: 240}[n],
                   params={"nseg": n, "static_damping": 0.0}, max_nworld={20: 4096, 40: 1024, 80: 256}[n]),
            Config(f"joint_chain_xpbd_n{n}", "xpbd", dt=1 / 60, substeps=20,
                   params={"nseg": n, "iterations": 10, "model": "joint_chain"}),
            Config(f"xpbd_springs_n{n}", "xpbd", dt=1 / 60, substeps=10,
                   params={"nseg": n, "iterations": 10, "model": "particle_springs",
                           "bend_spring_scale": XPBD_BEND_SCALE, "spring_damping": 0.0}),
        ))

    def build(self, nworld: int, device: str, config: Config) -> Built:
        n = config.params["nseg"]
        particles = config.params.get("model") == "particle_springs"
        if config.solver == "vbd":
            tmpl, tips, local = _d01_vbd_world(n)
        elif particles:
            tmpl, tips, local = _d01_particle_world(n)
        else:
            tmpl, tips, local = _d01_joint_world(n, config.solver)
        reg = newton.solvers.SolverMuJoCo.register_custom_attributes if config.solver == "mujoco" else None
        asm = assemble([tmpl], nworld, register=reg)
        model = asm.builder.finalize(device=device)
        p = config.params
        S = newton.solvers
        if config.solver == "vbd":
            solver = S.SolverVBD(model, iterations=p["iterations"])
        elif config.solver == "mujoco":
            solver = S.SolverMuJoCo(model, integrator=p["integrator"], solver=p["solver"], iterations=p["iterations"],
                                    disable_contacts=True)
        elif config.solver == "featherstone":
            solver = S.SolverFeatherstone(model)
        else:
            solver = S.SolverXPBD(model, iterations=p["iterations"])
        s0, s1, ctrl = model.state(), model.state(), model.control()
        if model.joint_count:
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            wp.copy(s1.body_q, s0.body_q)
        h = config.dt / config.substeps
        drv = Driver(device, h)
        stride = max(1, round(2e-3 / h))
        rows = rec_rows(self.horizon_s, h, stride)
        worlds = recorded_worlds(nworld)
        if particles:
            ids = world_ids(asm.particle_start, tips, worlds).ravel()
            drv.record_particles("tips", ids, stride, rows)
        else:
            ids = world_ids(asm.body_start, tips, worlds).ravel()
            drv.record_bodies("tips", ids, np.tile(local, (len(worlds), 1)), stride, rows)
        return Built(model, solver, s0, s1, ctrl, nworld=nworld, extra={"driver": drv, "nseg": n, "nrec": len(worlds)})

    def finish(self, b: Built, rec: dict) -> None:
        n = b.extra["nseg"]
        t, xyz = b.extra["driver"].trace("tips")
        xyz = xyz.reshape(len(t), b.extra["nrec"], len(OBJ_NAMES), 3)
        last = xyz[-1]
        out, per_obj = {}, {}
        roots = np.array([[0.0, c[3], Z0] for c in CANTILEVERS] + [[0.0, CHAIN_Y, CHAIN_TOP]])
        reach = float(np.max(np.linalg.norm(xyz[:, 0] - roots[None], axis=-1)))
        if reach > 1.5 * L:  # a tip farther than 1.5 L from its root: the chain blew up (SCENARIOS.md 'stable')
            rec["stable"] = False
            out["blowup_reach_m"] = reach
        late = t >= 0.5 * t[-1]
        for k, (name, gamma, _, _) in enumerate(CANTILEVERS):
            drop = Z0 - last[:, k, 2]
            settled = late.any() and np.ptp(xyz[t >= t[-1] - 0.2, 0, k, 2]) < 1e-4
            if name != "pluck" and not settled and late.any():  # undamped or unsettled: equilibrium = time mean
                drop = Z0 - xyz[late, :, k, 2].mean(axis=0)
                out[f"{name}_from_time_mean"] = True
            per_obj[name] = drop
            EI = _ei(gamma)
            cont = LAW.elastica_tip(gamma, 1.0)[0] * L
            disc = LAW.cantilever_chain(n, L, W_LIN, EI)[0]
            if name != "pluck":
                out[f"{name}_drop_mm"] = 1e3 * drop[0]
                out[f"{name}_elastica_mm"] = 1e3 * cont
                out[f"{name}_discrete_mm"] = 1e3 * disc
                out[f"{name}_err_pct"] = pct(drop[0], cont)
                out[f"{name}_err_discrete_pct"] = pct(drop[0], disc)
                tail = t >= t[-1] - 0.2
                out[f"{name}_drift_mm_last_0p2s"] = 1e3 * float(np.ptp(xyz[tail, 0, k, 2])) if tail.any() else None
        out["static_small_err_linear_pct"] = pct(per_obj["static_small"][0], LAW.linear_tip(L, W_LIN, _ei(0.5)))
        # pluck: released from straight, undamped
        drop_t = Z0 - xyz[:, 0, 2, 2]
        ref = LAW.pluck_reference(n, L, M_LIN, _ei(0.5), R, G, t)
        T = period_from_crossings(t, drop_t)
        out.update(pluck_period_s=T, pluck_period_discrete_s=ref["T1"], pluck_period_continuum_s=ref["T1_continuum"],
                   period_err_pct=pct(T, ref["T1"]), period_err_continuum_pct=pct(T, ref["T1_continuum"]),
                   pluck_peak_mm=1e3 * float(drop_t.max()), pluck_peak_ref_mm=1e3 * float(ref["tip"].max()),
                   peak_drop_err_pct=pct(float(drop_t.max()), float(ref["tip"].max())),
                   pluck_traj_rms_mm=1e3 * float(np.sqrt(np.mean((drop_t - ref["tip"]) ** 2))))
        # hanging chain: free-end lateral swing in x
        x_t = xyz[:, 0, 3, 0]
        Tc = period_from_crossings(t, x_t)
        w1 = 2 * math.pi / Tc if np.isfinite(Tc) else float("nan")
        w1_ref = LAW.hanging_chain_w1(L, G)
        w1_disc = (LAW.hanging_lumped_modes(n, L, M_LIN, G)[0] if b.model.particle_count
                   else LAW.hanging_chain_modes(n, L, M_LIN, R, G)[0])
        hang = np.linalg.norm(xyz[:, 0, 3] - np.array([0.0, CHAIN_Y, CHAIN_TOP]), axis=1)
        out.update(omega1=w1, omega1_ref=w1_ref, omega1_discrete=w1_disc, omega1_err_pct=pct(w1, w1_ref),
                   omega1_err_discrete_pct=pct(w1, w1_disc), chain_stretch_pct=100 * (float(hang.mean()) / L - 1.0))
        out["world_spread_mm"] = 1e3 * float(max(np.ptp(v) for v in per_obj.values()))
        static_ok = abs(out["static_large_err_pct"]) <= 1.0 and abs(out["static_small_err_pct"]) <= 1.0
        w1_ok = np.isfinite(out["omega1_err_pct"]) and abs(out["omega1_err_pct"]) <= 2.0
        law_record(rec, out["static_large_err_pct"], tip_drop_err_pct=out["static_large_err_pct"],
                   law_pass=bool(static_ok and w1_ok), law_pass_static=bool(static_ok), law_pass_omega1=bool(w1_ok),
                   nseg=n, fit_dt_only=False, sim_time_s=float(t[-1]) if len(t) else 0.0, **out)


# ---- D02 bend / twist rows, Michell ring, capstan ---------------------------------------------------------------
BT_N, BT_SEG, BT_R, BT_STRETCH = 14, 0.10, 0.010, 1.0e6
BT_TARGETS = (math.radians(30.0), math.radians(60.0), math.radians(90.0))
BT_RAMP = 2.0
MI_N, MI_RING_R, MI_STRETCH, MI_BEND, MI_TWIST_TOTAL, MI_SEED = 48, 0.55, 1.0e7, 50.0, 2.0 * math.pi, 1.0e-3
MI_RATIOS = (1.5, 2.2, 2.7, 3.0, 3.3, 4.2, 6.0)  # twist/bend; factor = total / critical = 0.71 .. 1.41
MI_X = 3.0
CAP_MU = (0.1, 0.25, 0.4)
CAP_RATIOS = (1.25, 0.8)  # heavy / (light * exp(mu pi)): slides, holds
CAP_R, CAP_Z, CAP_M_LIGHT, CAP_HALF = 0.1, 2.0, 1.0, 0.04
CAP_RAMP = 0.5


def _q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _q_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _q_rot(q, v):
    return _q_mul(_q_mul(q, np.array([*v, 0.0])), _q_conj(q))[:3]


def _q_axis(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([*(a * math.sin(0.5 * ang)), math.cos(0.5 * ang)])


def _q_axis_angle(q):
    q = q if q[3] >= 0 else -q
    w = min(1.0, max(-1.0, q[3]))
    s = math.sqrt(max(0.0, 1 - w * w))
    return (np.array([1.0, 0, 0]), 0.0) if s < 1e-9 else (q[:3] / s, 2 * math.acos(w))


def _node(body_q, seg):  # start node of a COM-centred capsule (vbd/_viewer.py node_xyz)
    return body_q[:3] - _q_rot(body_q[3:7], (0.0, 0.0, 0.5 * seg))


def _ease_np(u):
    x = np.clip(u, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _ring_rod() -> newton.Rod:
    th = np.linspace(0.0, 2 * math.pi, MI_N + 1)
    pts = np.stack([MI_X + MI_RING_R * np.cos(th), MI_RING_R * np.sin(th), np.full_like(th, 0.6)], axis=1)
    quats = []
    for i in range(MI_N):
        mid = 0.5 * (th[i] + th[i + 1])
        Rm = np.column_stack([[-math.cos(mid), -math.sin(mid), 0.0], [0.0, 0.0, 1.0], [-math.sin(mid), math.cos(mid), 0.0]])
        quats.append(_mat_quat(Rm))
    return newton.Rod(pts.astype(np.float32), quaternions=np.asarray(quats, np.float32), radius=BT_R, closed=True)


def _mat_quat(Rm):
    tr = np.trace(Rm)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [(Rm[2, 1] - Rm[1, 2]) / s, (Rm[0, 2] - Rm[2, 0]) / s, (Rm[1, 0] - Rm[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(Rm)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + Rm[i, i] - Rm[j, j] - Rm[k, k]) * 2
    q = [0.0, 0.0, 0.0, (Rm[k, j] - Rm[j, k]) / s]
    q[i], q[j], q[k] = 0.25 * s, (Rm[j, i] + Rm[i, j]) / s, (Rm[k, i] + Rm[i, k]) / s
    return q


def _d02_vbd_world(ratio: float) -> tuple[newton.ModelBuilder, dict]:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    info = {"bend": [], "twist": []}
    for kind, y, kb, kt in (("bend", 0.72, 400.0, 1000.0), ("twist", -0.72, 2000.0, 120.0)):
        for i, target in enumerate(BT_TARGETS):
            rod = newton.Rod.create_straight(start=wp.vec3(0.0, y, (2 - i) * 0.42), direction=wp.vec3(1.0, 0.0, 0.0),
                                             length=BT_N * BT_SEG, segment_count=BT_N, radius=BT_R)
            bodies, joints = b.add_rod(rod=rod, stretch_stiffness=BT_STRETCH, bend_stiffness=kb, bend_damping=kb,
                                       twist_stiffness=kt, twist_damping=kt, wrap_in_articulation=False,
                                       label=f"{kind}_{i}")
            make_kinematic_body(b, bodies[0])
            make_kinematic_body(b, bodies[-1])
            b.add_articulation(joints)
            info[kind].append({"target": target, "bodies": list(bodies)})
    twist = MI_BEND * ratio
    bodies, _ = b.add_rod(rod=_ring_rod(), stretch_stiffness=MI_STRETCH, bend_stiffness=MI_BEND, twist_stiffness=twist,
                          label="michell")
    info["ring"] = {"bodies": list(bodies), "ratio": ratio,
                    "factor": MI_TWIST_TOTAL / LAW.michell_critical_twist(MI_BEND, twist)}
    b.color()
    return b, info


def _seed_ring(body_q: np.ndarray, bodies: list[int]) -> None:
    """Impose one material-frame turn and the example's deterministic seed (michell_threshold.py)."""
    n = len(bodies)
    for i, body in enumerate(bodies):
        q = body_q[body, 3:7].astype(np.float64)
        tangent = _q_rot(q, (0.0, 0.0, 1.0))
        body_q[body, 3:7] = _q_mul(_q_axis(tangent, MI_TWIST_TOTAL * i / n), q)
        body_q[body, 2] += MI_SEED * math.sin(6.0 * math.pi * i / n)


def _tip_table(body_q0: np.ndarray, info: dict, row_dt: float) -> tuple[list[int], np.ndarray]:
    """Kinematic tip pose rows (bend: on the uniform discrete arc; twist: rotated in place), eased over BT_RAMP."""
    times = np.arange(0.0, BT_RAMP + row_dt, row_dt)
    ids, table = [], []
    for kind in ("bend", "twist"):
        for case in info[kind]:
            tip = case["bodies"][-1]
            ids.append(tip)
            rest_q = body_q0[tip, 3:7].astype(np.float64)
            root = _node(body_q0[case["bodies"][0]], BT_SEG)
            rest_tip_node = _node(body_q0[tip], BT_SEG)
            col = []
            for t in times:
                tgt = case["target"] * float(_ease_np(t / BT_RAMP))
                if kind == "bend":
                    ang = np.arange(BT_N) / (BT_N - 1) * tgt
                    node = root + BT_SEG * np.array([np.cos(ang[:-1]).sum(), 0.0, -np.sin(ang[:-1]).sum()])
                    rot = _q_mul(_q_axis((0, 1, 0), tgt), rest_q)
                else:
                    node = rest_tip_node
                    rot = _q_mul(_q_axis((1, 0, 0), tgt), rest_q)
                col.append([*(node + _q_rot(rot, (0.0, 0.0, 0.5 * BT_SEG))), *rot])
            table.append(col)
    return ids, np.transpose(np.asarray(table), (1, 0, 2))


def _coplanarity(pts: np.ndarray) -> float:
    c = pts - pts.mean(axis=0)
    m = c.T @ c
    tr = float(np.trace(m))
    return 0.0 if tr <= 0 else float(3.0 * max(np.linalg.eigvalsh(m)[0], 0.0) / tr)


def _d02_capstan_world(mu: float, ratio: float) -> tuple[newton.ModelBuilder, list[int], float]:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    m_heavy = CAP_M_LIGHT * ratio * LAW.capstan_ratio(mu, math.pi)
    Dof = newton.ModelBuilder.JointDofConfig
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    bodies = []
    for x, m in ((-CAP_R, CAP_M_LIGHT), (CAP_R, m_heavy)):
        pos = wp.vec3(x, 0.0, CAP_Z - 1.0)
        body = b.add_link(xform=wp.transform(pos, wp.quat_identity()), mass=m,
                          inertia=wp.mat33(*(np.eye(3) * m * CAP_HALF**2 / 1.5).ravel()))
        b.add_shape_box(body, hx=CAP_HALF, hy=CAP_HALF, hz=CAP_HALF, cfg=cfg)
        j = b.add_joint_d6(parent=-1, child=body, linear_axes=[Dof(axis=newton.Axis.Z, limit_lower=-0.6, limit_upper=0.6)],
                           angular_axes=[], parent_xform=wp.transform(pos, wp.quat_identity()), child_xform=wp.transform())
        b.add_articulation([j])
        bodies.append(body)
    span = {"compliance": 1.0e-6, "damping": 5.0, "rest_length": -1.0}
    b.add_tendon([
        newton.TendonGuide(body=bodies[0], guide_type=newton.TendonGuideType.ANCHOR, offset=(0.0, 0.0, CAP_HALF),
                           axis=(0.0, 1.0, 0.0)),
        newton.TendonGuide(body=-1, guide_type=newton.TendonGuideType.ROLLER, radius=CAP_R, orientation=1, mu=mu,
                           offset=(0.0, 0.0, CAP_Z), axis=(0.0, 1.0, 0.0), **span),
        newton.TendonGuide(body=bodies[1], guide_type=newton.TendonGuideType.ANCHOR, offset=(0.0, 0.0, CAP_HALF),
                           axis=(0.0, 1.0, 0.0), **span),
    ])
    b.color()  # SolverVBD needs body colour groups; XPBD ignores them
    return b, bodies, m_heavy


class CableBendTwist(DeformScenario):
    id = "D02_cable_bend_twist"
    title = "Discrete bend/twist boundary values, Michell twisted ring, capstan exp(mu theta)"
    horizon_s = 6.0
    accuracy_unit = "deg"
    accuracy_ref = ("rod configs: max |theta_i - i/(N-1) theta_tip| over bend and twist rows [deg]; Michell critical "
                    "twist 2 pi sqrt(3B/C) classification in metrics. Capstan configs: max |T_h/T_l / exp(mu pi) - 1| "
                    "[pct] from the sliding acceleration (unit overridden per record)")
    configs = (
        Config("vbd_it10", "vbd", dt=1 / 60, substeps=10, params={"iterations": 10}),
        Config("vbd_it50", "vbd", dt=1 / 60, substeps=10, params={"iterations": 50}),
        Config("xpbd_tendon_capstan", "xpbd", dt=1 / 60, substeps=32, params={"iterations": 32, "capstan": True}),
        Config("vbd_tendon_capstan", "vbd", dt=1 / 60, substeps=32, params={"iterations": 10, "capstan": True}),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        if config.params.get("capstan"):
            return self._build_capstan(nworld, device, config)
        tmpls = [_d02_vbd_world(r) for r in MI_RATIOS]
        asm = assemble([t[0] for t in tmpls], nworld, gravity=0.0)
        model = asm.builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(model, iterations=config.params["iterations"])
        s0, s1, ctrl = model.state(), model.state(), model.control()
        q = s0.body_q.numpy()
        for w in range(nworld):
            info = tmpls[asm.cond[w]][1]
            _seed_ring(q, [asm.body_start[w] + i for i in info["ring"]["bodies"]])
        s0.body_q.assign(q)
        s1.body_q.assign(q)
        h = config.dt / config.substeps
        drv = Driver(device, h)
        ids_all, tab_all = [], []
        for w in range(nworld):
            info = tmpls[asm.cond[w]][1]
            off = asm.body_start[w]
            shifted = {k: ([{"target": c["target"], "bodies": [off + i for i in c["bodies"]]} for c in info[k]])
                       for k in ("bend", "twist")}
            ids, tab = _tip_table(q, shifted, row_dt=1 / 120)
            ids_all += ids
            tab_all.append(tab)
        drv.body_pose_table(ids_all, np.concatenate(tab_all, axis=1), row_dt=1 / 120)
        return Built(model, solver, s0, s1, ctrl, nworld=nworld,
                     extra={"driver": drv, "tmpls": [t[1] for t in tmpls], "asm": asm, "q0": q.copy()})

    def _build_capstan(self, nworld: int, device: str, config: Config) -> Built:
        conds = [(mu, r) for mu in CAP_MU for r in CAP_RATIOS]
        tmpls = [_d02_capstan_world(mu, r) for mu, r in conds]
        asm = assemble([t[0] for t in tmpls], nworld, gravity=0.0)
        model = asm.builder.finalize(device=device)
        p = config.params
        S = newton.solvers
        solver = (S.SolverXPBD(model, iterations=p["iterations"]) if config.solver == "xpbd"
                  else S.SolverVBD(model, iterations=p["iterations"]))
        s0, s1, ctrl = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        wp.copy(s1.body_q, s0.body_q)
        h = config.dt / config.substeps
        drv = Driver(device, h)
        drv.gravity_ramp(model, 0.0, CAP_RAMP)
        worlds = recorded_worlds(nworld, len(conds))
        stride = max(1, round(2e-3 / h))
        drv.record_bodies("masses", world_ids(asm.body_start, tmpls[0][1], worlds).ravel(), np.zeros((2 * len(worlds), 3)),
                          stride, rec_rows(self.horizon_s, h, stride))
        return Built(model, solver, s0, s1, ctrl, nworld=nworld,
                     extra={"driver": drv, "asm": asm, "conds": conds, "m_heavy": [t[2] for t in tmpls],
                            "nrec": len(worlds), "capstan": True})

    def finish(self, b: Built, rec: dict) -> None:
        if b.extra.get("capstan"):
            return self._finish_capstan(b, rec)
        q = b.state_0.body_q.numpy().astype(np.float64)
        q0, asm, tmpls = b.extra["q0"], b.extra["asm"], b.extra["tmpls"]
        ramp = float(_ease_np(b.extra["driver"].time() / BT_RAMP))  # tip angle reached so far (1 after BT_RAMP)
        bend_err, twist_err, shape_rms, per_cond = [], [], [], {}
        for w in range(b.nworld if b.nworld <= 64 else 64):
            info, off = tmpls[asm.cond[w]], asm.body_start[w]
            for kind, axis in (("bend", 1), ("twist", 0)):
                for case in info[kind]:
                    bodies = [off + i for i in case["bodies"]]
                    meas = np.array([(lambda aa: aa[0][axis] * aa[1])(_q_axis_angle(_q_mul(q[i, 3:7], _q_conj(q0[i, 3:7]))))
                                     for i in bodies])
                    exp = np.arange(BT_N) / (BT_N - 1) * case["target"] * ramp
                    if abs(meas[-1] + exp[-1]) < abs(meas[-1] - exp[-1]):
                        meas = -meas
                    err = math.degrees(float(np.max(np.abs(meas - exp))))
                    (bend_err if kind == "bend" else twist_err).append(err)
                    if kind == "bend":
                        pts = np.array([_node(q[i], BT_SEG) for i in bodies])
                        ref = [_node(q0[bodies[0]], BT_SEG)]
                        for a in exp[:-1]:
                            ref.append(ref[-1] + BT_SEG * np.array([math.cos(a), 0.0, -math.sin(a)]))
                        shape_rms.append(float(np.sqrt(np.mean(np.sum((pts - np.array(ref)) ** 2, axis=1)))))
            ring = info["ring"]
            pts = np.array([_node(q[off + i], 2 * MI_RING_R * math.sin(math.pi / MI_N)) for i in ring["bodies"]])
            cop = _coplanarity(pts)
            cls = "stable" if cop < 5e-3 else ("writhe" if cop > 5e-2 else "ambiguous")
            per_cond[f"{ring['factor']:.2f}"] = {"coplanarity": cop, "class": cls}
        decisive = {k: v for k, v in per_cond.items() if float(k) <= 0.9 or float(k) >= 1.35}
        correct = [(v["class"] == "stable") == (float(k) < 1.0) and v["class"] != "ambiguous" for k, v in decisive.items()]
        acc = float(np.mean(correct)) if correct else float("nan")
        err = max(bend_err + twist_err)
        law_record(rec, err, tip_ramp_fraction=ramp, bend_angle_err_deg=max(bend_err), twist_angle_err_deg=max(twist_err),
                   bend_shape_rms_mm=1e3 * max(shape_rms), michell_class_acc=acc, per_condition=per_cond,
                   conditions_covered=bool(b.nworld >= len(MI_RATIOS)),
                   law_pass=bool(err <= 0.1 and (not np.isfinite(acc) or acc == 1.0)), fit_dt_only=False)

    def _finish_capstan(self, b: Built, rec: dict) -> None:
        t, xyz = b.extra["driver"].trace("masses")
        conds, asm = b.extra["conds"], b.extra["asm"]
        xyz = xyz.reshape(len(t), b.extra["nrec"], 2, 3)
        per_cond, errs, held = {}, [], []
        for w in range(min(b.extra["nrec"], len(conds))):
            mu, ratio = conds[asm.cond[w]]
            m_h = b.extra["m_heavy"][asm.cond[w]]
            z = xyz[:, w, 1, 2]
            travel = z[0] - z
            win = (t >= CAP_RAMP + 0.1) & (travel < 0.35)
            key = f"mu{mu}_r{ratio}"
            if win.sum() < 10:
                per_cond[key] = {"note": "window too short"}
                continue
            c2, c1, c0 = np.polyfit(t[win], z[win], 2)
            a = -2.0 * c2  # downward acceleration of the heavy mass
            meas = m_h * (G - a) / (CAP_M_LIGHT * (G + a))
            ref = LAW.capstan_ratio(mu, math.pi)
            entry = {"accel": a, "slip_mm": 1e3 * float(np.ptp(z[win]))}
            if ratio > 1.0:
                entry["ratio_meas"], entry["ratio_ref"] = meas, ref
                entry["err_pct"] = pct(meas, ref)
                errs.append(abs(entry["err_pct"]))
            else:
                entry["held"] = bool(entry["slip_mm"] < 1.0)
                held.append(entry["held"])
            per_cond[key] = entry
        err = max(errs) if errs else float("nan")
        law_record(rec, err, unit="pct", capstan_err_pct=err, hold_ok=bool(all(held)) if held else None,
                   per_condition=per_cond, conditions_covered=bool(b.nworld >= len(conds)),
                   law_pass=bool(np.isfinite(err) and err <= 2.0 and all(held)), fit_dt_only=False)


SCENARIOS_CABLE = [CableCantilever(), CableBendTwist()]
