# SPDX-License-Identifier: Apache-2.0
"""Rigid bodies and articulations (R01-R06): baseline laws for Newton's stock rigid solvers.

    python -m bench.run --family rigid --device cpu --ladder 1 --horizon 0.5          # Mac CPU smoke test
    python -m bench.run --family rigid --device cuda:0 --ladder 1,16,256,1024,4096
    python -m bench.scenarios.rigid ref-r05                                           # offline CPU MuJoCo rows

Spec: bench/SCENARIOS.md §1.1. Design notes, CPU smoke results and known problems: bench/notes/rigid.md.
Newton sources are cited as newton:<path>:<line> at ac7f7830.

Conventions used by every scenario in this file
- Conditions tile over worlds (world w runs condition w % ncond). Per-world parameters (restitution, friction,
  initial angle, tilt) are written into the finalized Model arrays before the solver is created, so every world has
  the same structure (MuJoCo Warp needs that) and ModelBuilder.replicate stays the build path.
- Per-step quantities (impact, rebound apex, zero crossings, in-hand slip, tracking error) are tracked by Warp
  kernels launched inside step() and read in finish(). Scripted inputs (gravity tilt, velocity kicks, the Franka
  joint trajectory) are indexed by a device step counter. Nothing in step() syncs with the host, so CUDA graphs work.
- Contacts: solvers on Newton's CollisionPipeline get an explicit rigid_contact_max; MuJoCo-native configs use
  MuJoCo's narrowphase with shape gap 0 (MJCF semantics) and explicit nconmax/njmax. Overflow is accumulated every
  substep (pipeline counters, MuJoCo's OVERFLOW observable) and reported as metrics.contact_overflow.
- Penetration for MuJoCo-native configs comes from mjw_data.contact.dist; for the others from Newton contacts
  (metrics.rigid_penetration_mm).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import sys
import time
from pathlib import Path

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench import metrics as M
from bench.harness import Built, Config, Scenario

G = 9.81
DT600 = 1.0 / 600.0
SCENE_KEYS = frozenset({"n", "gap", "shape_ke", "shape_kd", "shape_kf", "noslip_iterations"})


# =================================================================================================================
# Shared plumbing: solver factory, builders, contacts, overflow, per-world energy
# =================================================================================================================
def native_contacts(cfg: Config) -> bool:
    """True when the solver runs its own narrowphase (SolverMuJoCo's default use_mujoco_contacts=True)."""
    return cfg.solver == "mujoco" and bool(cfg.params.get("use_mujoco_contacts", True))


def _mjw_has_noslip() -> bool:
    from alloy._vendor.mujoco_warp._src import types as mjw_types

    return any(f.name == "noslip_iterations" for f in dataclasses.fields(mjw_types.Option))


def _kamino_config(model: newton.Model, kw: dict) -> newton.solvers.SolverKamino.Config:
    from alloy._vendor.newton._src.solvers.kamino import config as kc

    classes = {"constraints": kc.ConstraintStabilizationConfig, "dynamics": kc.ConstrainedDynamicsConfig,
               "padmm": kc.PADMMSolverConfig, "dvi": kc.DVISolverConfig}
    kw = {k: (classes[k](**v) if k in classes and isinstance(v, dict) else v) for k, v in kw.items()}
    return newton.solvers.SolverKamino.Config.from_model(model, **kw)


def ref_mujoco_cpu(dt: float, params: dict, name: str = "ref_mujoco_cpu") -> Config:
    """Offline reference row: CPU MuJoCo 3.16 (float64 MjData) through SolverMuJoCo(use_mujoco_cpu=True), one world.

    update_data_interval 0 keeps MjData's own float64 state (no per-step copy from the float32 Newton state).
    Run with --device cpu; Newton's native CPU backend holds a single template world (max_nworld 1).
    """
    return Config(name, "mujoco", dt=dt, cuda_graph=False, max_nworld=1,
                  params={**params, "use_mujoco_cpu": True, "update_data_interval": 0})


def make_solver(cfg: Config, model: newton.Model, contact_max: int | None = None):
    """Solver from cfg.solver; every kwarg comes from cfg.params (minus the scene keys) so it lands in the record.
    contact_max: the pipeline's rigid_contact_max, published on the model first because Steel sizes its scratch from
    model.rigid_contact_max at construction (else R03 at 256 worlds: "Steel contact capacity mismatch")."""
    S = newton.solvers
    if contact_max is not None and cfg.solver == "steel":
        model.rigid_contact_max = int(contact_max)
    kw = {k: v for k, v in cfg.params.items() if k not in SCENE_KEYS}
    if cfg.solver == "mujoco":
        noslip = int(cfg.params.get("noslip_iterations", 0))
        if noslip and not _mjw_has_noslip():
            from importlib import metadata

            raise NotImplementedError(f"noslip: mujoco_warp {metadata.version('mujoco-warp')} has no "
                                      "Option.noslip_iterations (MuJoCo Warp does not implement noslip yet)")
        solver = S.SolverMuJoCo(model, **{"use_mujoco_cpu": False, **kw})
        if noslip:
            solver.mjw_model.opt.noslip_iterations = noslip
        return solver
    if cfg.solver == "kamino":
        return S.SolverKamino(model, config=_kamino_config(model, kw))
    if cfg.solver == "steel":
        mode = kw.pop("solve_mode", "auto")
        if mode == "auto":  # matrix_free is CUDA only (newton/tests/test_physics_verification.py:1540)
            mode = "matrix_free" if model.device.is_cuda else "split"
        return S.SolverSteel(model, solve_mode=mode, **kw)
    classes = {"featherstone": S.SolverFeatherstone, "xpbd": S.SolverXPBD, "semi_implicit": S.SolverSemiImplicit,
               "vbd": S.SolverVBD}
    if cfg.solver not in classes:
        raise ValueError(f"unknown solver {cfg.solver!r}")
    return classes[cfg.solver](model, **kw)


def new_builder(cfg: Config, gap: float | None = None) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    if cfg.solver == "kamino":
        newton.solvers.SolverKamino.register_custom_attributes(b)
    if gap is not None:
        b.default_shape_cfg.gap = gap
    return b


def replicate(scene: newton.ModelBuilder, nworld: int, cfg: Config, device: str) -> newton.Model:
    main = new_builder(cfg)
    # Kamino names one material per distinct (mu, restitution) after the shape label and rejects duplicates
    # (newton/_src/solvers/kamino/_src/core/conversions.py:2168-2180), so its worlds get unique label prefixes.
    prefixes = [f"w{i}" for i in range(nworld)] if cfg.solver == "kamino" else None
    main.replicate(scene, copy_count=nworld, label_prefixes=prefixes)
    if cfg.solver == "vbd":
        main.color()
    return main.finalize(device=device)


def contact_gap(cfg: Config, default: float) -> float:
    """Shape gap: 0 for MuJoCo's own narrowphase (MJCF semantics), else cfg.params['gap'] or the scenario default."""
    return 0.0 if native_contacts(cfg) else float(cfg.params.get("gap", default))


def apply_material_overrides(model: newton.Model, cfg: Config) -> None:
    for key, attr in (("shape_ke", "shape_material_ke"), ("shape_kd", "shape_material_kd"),
                      ("shape_kf", "shape_material_kf")):
        if key in cfg.params:
            getattr(model, attr).fill_(float(cfg.params[key]))


def assemble(model: newton.Model, cfg: Config, solver, nworld: int, rigid_contact_max: int | None,
             extra: dict | None = None) -> Built:
    """States (FK from model.joint_q), an explicit-capacity CollisionPipeline unless contacts are native, counters."""
    dev = model.device
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    pipe = contacts = None
    if rigid_contact_max and not native_contacts(cfg):
        pipe = newton.CollisionPipeline(model, rigid_contact_max=int(rigid_contact_max))
        contacts = pipe.contacts()
    ex = {"ctr": wp.zeros(1, dtype=wp.int32, device=dev), "pipe_overflow": wp.zeros(2, dtype=wp.int32, device=dev),
          "obs": None, "mj_overflow": None}
    if cfg.solver == "mujoco" and solver.use_mujoco_cpu:
        # reference rows run with update_data_interval 0 (float64 MjData never overwritten), so push the
        # scenario's initial state (per-condition q, qd) into MjData once here
        solver._update_mjc_data(solver.mj_data, model, s0)
    elif cfg.solver == "mujoco":
        ex["obs"] = solver.observables({solver.ObservableKind.OVERFLOW})
        ex["mj_overflow"] = wp.zeros(ex["obs"].overflow.shape[0], dtype=wp.int32, device=dev)
    ex.update(extra or {})
    return Built(model, solver, s0, s1, control, contacts, pipe, nworld=nworld, extra=ex)


@wp.kernel
def _inc_counter(ctr: wp.array[wp.int32]):
    ctr[0] = ctr[0] + 1


@wp.kernel
def _acc_pipeline_overflow(counters: wp.array[wp.int32], cmax: int, acc: wp.array[wp.int32]):
    over = counters[0] - cmax
    if over > 0:
        wp.atomic_max(acc, 0, over)
    if counters[2] > 0:
        acc[1] = 1


@wp.kernel
def _acc_bitmask(bits: wp.array[wp.int32], acc: wp.array[wp.int32]):
    w = wp.tid()
    acc[w] = acc[w] | bits[w]


def penetration_mm(b: Built) -> np.ndarray:
    """Depth [mm] of every active contact: Newton contacts, or MuJoCo's own contact.dist for native configs."""
    if b.contacts is not None:
        return M.rigid_penetration_mm(b.model, b.state_0, b.contacts)
    if getattr(b.solver, "use_mujoco_cpu", False):  # offline CPU MuJoCo reference: MjData, float64
        dc = b.solver.mj_data
        return np.clip(-np.asarray(dc.contact.dist[: dc.ncon], dtype=np.float64), 0.0, None) * 1e3
    d = getattr(b.solver, "mjw_data", None)
    if d is None:
        return np.zeros(0)
    n = min(int(d.nacon.numpy()[0]), int(d.naconmax))
    return np.clip(-d.contact.dist.numpy()[:n].astype(np.float64), 0.0, None) * 1e3 if n else np.zeros(0)


def overflow_metrics(b: Built) -> dict:
    po = b.extra["pipe_overflow"].numpy()
    out = {"contact_overflow": int(po[0]) + int(po[1])}
    flags = getattr(b.solver, "constraint_overflow", None)  # Steel: per-world row-capacity / dropped-contact flags
    if flags is not None:
        n_flag = int(np.count_nonzero(flags.numpy()))
        out["contact_overflow"] += n_flag
        out["steel_overflow_worlds"] = n_flag
    if b.extra.get("mj_overflow") is not None:
        bits = b.extra["mj_overflow"].numpy()
        out["contact_overflow"] += int(np.count_nonzero(bits))
        out["mujoco_overflow_bits"] = int(np.bitwise_or.reduce(bits)) if bits.size else 0
    return out


def body_energy(model: newton.Model, state) -> np.ndarray:
    """Per-body kinetic + gravitational energy [J] (PE = -m g.x_com, world gravity per body's world)."""
    q = state.body_q.numpy().astype(np.float64)
    qd = state.body_qd.numpy().astype(np.float64)
    mass = model.body_mass.numpy().astype(np.float64)
    inertia = model.body_inertia.numpy().astype(np.float64)
    com = M.transform_points(q, np.arange(len(q)), model.body_com.numpy())
    g = np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)
    gb = g[model.body_world.numpy().clip(min=0)] if g.shape[0] > 1 else np.broadcast_to(g[0], com.shape)
    w_body = M.quat_rotate(q[:, 3:7] * np.array([-1.0, -1.0, -1.0, 1.0]), qd[:, 3:6])
    ke = 0.5 * mass * np.sum(qd[:, :3] ** 2, axis=1) + 0.5 * np.einsum("bi,bij,bj->b", w_body, inertia, w_body)
    return ke - mass * np.sum(gb * com, axis=1)


def copy_spread(b: Built, ncond: int) -> dict:
    """Worlds running the same condition (w % ncond) should end in the same state. Max body-position deviation of
    each world from its condition's first copy (world w % ncond) [mm]; nonzero = run-to-run order effects (GPU
    atomics) amplified by the scene, or a bad world."""
    if b.nworld <= ncond:
        return {}
    q = b.state_0.body_q.numpy().astype(np.float64).reshape(b.nworld, -1, 7)[:, :, :3]
    dev = np.linalg.norm(q - q[np.arange(b.nworld) % ncond], axis=2).max(axis=1) * 1e3
    return {"copy_spread_mm": _f(dev.max(), 5), "copy_spread_p50_mm": _f(np.median(dev), 5),
            "copies_off_1mm": int(np.count_nonzero(dev > 1.0))}


def coverage(nworld: int, ncond: int) -> dict:
    return {"conditions_covered": int(min(nworld, ncond)), "ncond": int(ncond)}


def _f(x, nd: int = 4):
    """JSON-friendly rounding (None for NaN)."""
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


class RigidScenario(Scenario):
    """Base for R01-R06: device step counter, pre/post kernels, overflow accumulation, MuJoCo-native penetration."""

    family = "rigid"
    has_contacts = True
    ncond = 1  # conditions tiled over worlds; copies of one condition are compared by copy_spread()

    def pre_step(self, b: Built, config: Config) -> None:
        """Scripted inputs indexed by the device step counter (before the substeps)."""

    def post_step(self, b: Built, config: Config) -> None:
        """Per-step tracking kernels (after the substeps)."""

    def substep(self, b: Built, dt: float) -> None:
        dev = b.model.device
        b.state_0.clear_forces()
        if b.pipeline is not None:
            b.pipeline.collide(b.state_0, b.contacts)
            wp.launch(_acc_pipeline_overflow, dim=1, device=dev,
                      inputs=[b.contacts.contact_counters, int(b.contacts.rigid_contact_max), b.extra["pipe_overflow"]])
        obs = b.extra["obs"]
        if obs is not None:
            b.solver.step(b.state_0, b.state_1, b.control, b.contacts, dt, observables=obs)
            wp.launch(_acc_bitmask, dim=obs.overflow.shape[0], inputs=[obs.overflow, b.extra["mj_overflow"]], device=dev)
        else:
            b.solver.step(b.state_0, b.state_1, b.control, b.contacts, dt)
        b.state_0, b.state_1 = b.state_1, b.state_0

    def step(self, b: Built, config: Config) -> None:
        self.pre_step(b, config)
        super().step(b, config)
        self.post_step(b, config)
        wp.launch(_inc_counter, dim=1, inputs=[b.extra["ctr"]], device=b.model.device)

    def measure(self, b: Built, t: float, rec: dict) -> None:
        if self.has_contacts:
            pen = penetration_mm(b)
            if pen.size:
                b.extra.setdefault("_pen_mm", []).append(pen)
        self.sample(b, t, rec)

    def sample(self, b: Built, t: float, rec: dict) -> None:
        """Scenario-specific host sample (state already synced)."""

    def finish(self, b: Built, rec: dict) -> None:
        if self.has_contacts:
            rec["metrics"].update(overflow_metrics(b))
        rec["metrics"].update(copy_spread(b, self.ncond))
        self.evaluate(b, rec)

    def evaluate(self, b: Built, rec: dict) -> None:
        raise NotImplementedError


# =================================================================================================================
# R01: free fall / projectile (A1), then a drop onto a plane with restitution e (A6)
# =================================================================================================================
R01_R = 0.05  # sphere radius [m]
R01_H = 0.5  # apex height of the sphere bottom above the plane [m]
R01_KE = 1.0e4  # contact stiffness for the force-based contact models (MuJoCo solref, penalty solvers)
R01_GAP = 0.005
R01_CLEAR = 0.02  # A1 is checked while the sphere bottom is more than this above the plane [m]
# (e, launch, v0x, v0z): e-major so 3 worlds cover every e
R01_COND = tuple((e, name, vx, vz) for name, vx, vz in (("free_fall", 0.0, 0.0), ("projectile", 1.0, 2.0))
                 for e in (0.0, 0.5, 0.8))


@functools.lru_cache(maxsize=64)
def contact_restitution(zeta: float, law: str) -> float:
    """Restitution of a unit-mass, unit-frequency spring-damper contact entered at unit speed.

    law "half": damping only while compressing (Newton's penalty contact, fd = min(vn, 0) kd,
    newton/_src/solvers/semi_implicit/kernels_contact.py:520). law "unilateral": full Kelvin-Voigt damping with the
    contact force clamped at >= 0 (MuJoCo's soft contact). Semi-implicit Euler at 1/2000 of the natural period.
    """
    x, v, h = 0.0, 1.0, 2.0 * math.pi / 2000.0
    for _ in range(400000):
        damp = 2.0 * zeta * v if (law == "unilateral" or v > 0.0) else 0.0
        f = x + damp
        if law == "unilateral" and f <= 0.0 and v < 0.0:
            return -v  # separates with the force at zero
        v -= f * h
        x += v * h
        if x <= 0.0 and v < 0.0:
            return -v
    return 0.0


def damping_for_restitution(e: float, law: str, zeta_max: float) -> tuple[float, float]:
    """(zeta, e_model): damping ratio that gives restitution e under `law`, capped at zeta_max for step stability;
    e_model is the restitution the capped zeta actually gives."""
    lo, hi = 0.0, 50.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if contact_restitution(round(mid, 9), law) > e else (lo, mid)
    zeta = min(0.5 * (lo + hi), zeta_max)
    return zeta, contact_restitution(round(zeta, 9), law)


@wp.kernel
def _r01_track(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], ctr: wp.array[wp.int32],
               dt: float, h_sub: float, g: float, r: float, clear: float,
               z0: wp.array[float], vx0: wp.array[float], vz0: wp.array[float],
               err: wp.array[float], ratio: wp.array[float], phase: wp.array[wp.int32], vz_prev: wp.array[float],
               zmax0: wp.array[float], zmax1: wp.array[float]):
    w = wp.tid()
    t = float(ctr[0] + 1) * dt
    p = wp.transform_get_translation(body_q[w])
    vz = wp.spatial_top(body_qd[w])[2]
    ph = phase[w]
    if ph == 0:
        zmax0[w] = wp.max(zmax0[w], p[2])
        if p[2] > r + clear:
            e = wp.max(wp.abs(p[2] - (z0[w] + vz0[w] * t - 0.5 * g * t * t)), wp.abs(p[0] - vx0[w] * t))
            e = wp.max(e, wp.abs(p[1]))
            err[w] = wp.max(err[w], e)
            ratio[w] = wp.max(ratio[w], e / wp.max(g * h_sub * t, 1.0e-6))  # bound 2 * (g h t / 2)
        elif vz > vz_prev[w]:  # net upward acceleration near the plane: contact has started
            phase[w] = 1
    elif ph == 1:
        if vz > 0.0:
            phase[w] = 2
            zmax1[w] = p[2]
    elif ph == 2:
        zmax1[w] = wp.max(zmax1[w], p[2])
        if vz <= 0.0:
            phase[w] = 3
    vz_prev[w] = vz


class FreeFallBounce(RigidScenario):
    id = "R01_free_fall_bounce"
    ncond = len(R01_COND)
    title = "Sphere free fall / projectile vs the parabola, then a bounce with e in {0, 0.5, 0.8}"
    horizon_s = 1.0
    accuracy_unit = "mm"
    accuracy_ref = ("max |h1 - e^2 h0| over worlds (A6, tol 2% h + 5 dt; 5 mm at e=0); A1 z = z0 + v0z t - g t^2/2, "
                    "x = v0x t within 2 * g h t / 2 (newton/tests/test_physics_verification.py:85,665-760)")
    configs = (
        Config("mujoco", "mujoco", dt=DT600, params={"use_mujoco_contacts": True, "solver": "newton",
                                                     "integrator": "implicitfast", "cone": "elliptic",
                                                     "iterations": 50, "ls_iterations": 50, "nconmax": 8,
                                                     "njmax": 32}),
        Config("featherstone", "featherstone", dt=DT600, params={"angular_damping": 0.0}),
        Config("xpbd", "xpbd", dt=DT600, params={"iterations": 10, "angular_damping": 0.0,
                                                 "enable_restitution": True}),
        Config("semi_implicit", "semi_implicit", dt=DT600, params={"angular_damping": 0.0}),
        Config("kamino_dvi", "kamino", dt=DT600, params={"dynamics_solver": "dvi"}),
        Config("steel", "steel", dt=DT600, params={"solve_mode": "auto"}),
        # Steel restitution needs the CUDA-only matrix_free solve; this row keeps A1 measurable on CPU
        Config("steel_split_norest", "steel", dt=DT600, params={"solve_mode": "split", "enable_restitution": False}),
        ref_mujoco_cpu(DT600, {"use_mujoco_contacts": True, "solver": "newton", "integrator": "implicitfast",
                               "cone": "elliptic", "iterations": 50, "ls_iterations": 50, "nconmax": 8, "njmax": 32}),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        gap = contact_gap(config, R01_GAP)
        scene = new_builder(config, gap)
        # mu = 1e-3, not 0: MuJoCo Warp warns that mu < 1e-5 with condim 3 may give NaN; tangential impulses on a
        # sphere do not change its vertical motion
        mat = newton.ModelBuilder.ShapeConfig(mu=1.0e-3, restitution=0.0, ke=R01_KE, kd=0.0, kf=0.0, gap=gap)
        scene.add_ground_plane(cfg=mat)
        body = scene.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()), label="ball")
        scene.add_shape_sphere(body, radius=R01_R, cfg=mat)
        model = replicate(scene, nworld, config, device)

        cond = np.arange(nworld) % len(R01_COND)
        e_w = np.array([R01_COND[c][0] for c in cond])
        vx_w = np.array([R01_COND[c][2] for c in cond])
        vz_w = np.array([R01_COND[c][3] for c in cond])
        z0_w = R01_R + R01_H - vz_w**2 / (2.0 * G)
        mass = float(model.body_mass.numpy()[0])
        h = config.dt / config.substeps
        kd_w, e_model = self._contact_damping(config, e_w, mass, h)
        sw = model.shape_world.numpy()
        model.shape_material_restitution.assign(e_w[sw].astype(np.float32))
        model.shape_material_kd.assign(kd_w[sw].astype(np.float32))
        g = float(np.linalg.norm(model.gravity.numpy().reshape(-1, 3)[0]))
        jq = model.joint_q.numpy().reshape(nworld, 7)
        jq[:, 2] = z0_w
        jqd = model.joint_qd.numpy().reshape(nworld, 6)
        jqd[:, 0], jqd[:, 2] = vx_w, vz_w
        model.joint_q.assign(jq.ravel())
        model.joint_qd.assign(jqd.ravel())
        solver = make_solver(config, model, 4 * nworld)

        def arr(x, dtype=float):
            return wp.array(np.asarray(x, dtype=np.float32 if dtype is float else np.int32), dtype=dtype, device=device)

        extra = {"g": g, "h": h, "cond": cond, "e": e_w, "e_model": e_model, "z0": z0_w, "vz0": vz_w,
                 "restitution": bool(config.params.get("enable_restitution", True)),
                 "d_z0": arr(z0_w), "d_vx0": arr(vx_w), "d_vz0": arr(vz_w), "err": arr(np.zeros(nworld)),
                 "ratio": arr(np.zeros(nworld)), "phase": arr(np.zeros(nworld), wp.int32), "vz_prev": arr(vz_w),
                 "zmax0": arr(np.full(nworld, -1e9)), "zmax1": arr(np.full(nworld, -1e9))}
        return assemble(model, config, solver, nworld, 4 * nworld, extra)

    @staticmethod
    def _contact_damping(config: Config, e_w: np.ndarray, mass: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-world kd for the force-based contact models, chosen so their own contact law gives restitution e.

        Impulse solvers (xpbd, kamino, steel) read shape restitution directly and ignore kd. MuJoCo: Newton maps
        (ke, kd) to solref as acceleration-space stiffness and damping (newton/_src/solvers/mujoco/kernels.py:190),
        and the default solimp impedance d = 0.95 scales both, so omega = sqrt(d ke) and kd = 2 zeta omega / d,
        capped at kd <= 1/dt (MuJoCo clamps timeconst = 2/kd to >= 2 dt). Penalty solvers (featherstone,
        semi_implicit) apply kd as force per normal speed while compressing only: omega = sqrt(ke/m),
        kd = 2 zeta omega m, capped at 2 zeta omega h <= 1 for the explicit step.
        """
        if config.solver == "mujoco":
            d = 0.95
            omega = math.sqrt(d * R01_KE)
            law, zmax, scale = "unilateral", (1.0 / config.dt) * d / (2.0 * omega), 2.0 * omega / d
        elif config.solver in ("featherstone", "semi_implicit"):
            omega = math.sqrt(R01_KE / mass)
            law, zmax, scale = "half", 1.0 / (2.0 * omega * h), 2.0 * omega * mass
        else:
            return np.zeros_like(e_w), e_w.copy()
        sol = {e: damping_for_restitution(e, law, zmax) for e in set(e_w.tolist())}
        return np.array([sol[e][0] * scale for e in e_w]), np.array([sol[e][1] for e in e_w])

    def post_step(self, b: Built, config: Config) -> None:
        x = b.extra
        wp.launch(_r01_track, dim=b.nworld, device=b.model.device,
                  inputs=[b.state_0.body_q, b.state_0.body_qd, x["ctr"], config.dt, x["h"], x["g"], R01_R, R01_CLEAR,
                          x["d_z0"], x["d_vx0"], x["d_vz0"], x["err"], x["ratio"], x["phase"], x["vz_prev"],
                          x["zmax0"], x["zmax1"]])

    def evaluate(self, b: Built, rec: dict) -> None:
        x, n = b.extra, b.nworld
        err, ratio, phase = x["err"].numpy(), x["ratio"].numpy(), x["phase"].numpy()
        zmax0, zmax1 = x["zmax0"].numpy().astype(np.float64), x["zmax1"].numpy().astype(np.float64)
        h0 = zmax0 - R01_R  # measured drop apex (the integrator's own apex, not the analytic one)
        h1 = np.where(phase >= 2, zmax1 - R01_R, np.where(phase == 1, 0.0, np.nan))
        exp = x["e"] ** 2 * h0
        tol = np.where(x["e"] == 0.0, 0.005, 0.02 * exp + 5.0 * x["h"])
        berr = np.abs(h1 - exp)
        per = []
        for c, (e, name, _, _) in enumerate(R01_COND):
            sel = x["cond"] == c
            if not sel.any():
                continue
            per.append({"e": e, "launch": name, "traj_err_mm": _f(err[sel].max() * 1e3),
                        "traj_bound_ratio": _f(ratio[sel].max()), "h0_mm": _f(h0[sel].mean() * 1e3, 2),
                        "h1_mm": _f(np.nanmax(h1[sel]) * 1e3, 2), "h1_expected_mm": _f(exp[sel].mean() * 1e3, 2),
                        "bounce_err_mm": _f(np.nanmax(berr[sel]) * 1e3), "bounce_tol_mm": _f(tol[sel].min() * 1e3),
                        "e_contact_model": _f(x["e_model"][sel][0]), "impact_seen": bool((phase[sel] >= 1).all())})
        bounce_pct = {}
        for e in sorted(set(x["e"].tolist())):
            sel = x["e"] == e
            denom = np.where(e > 0, exp[sel], h0[sel])
            bounce_pct[str(e)] = _f(np.nanmax(100.0 * berr[sel] / denom), 3)
        a1 = bool(np.all(ratio <= 1.0))
        scored = np.ones(n, bool) if x["restitution"] else x["e"] == 0.0  # restitution off: A6 only at e = 0
        if not x["restitution"]:
            rec["notes"] += "restitution disabled in this config: A6 scored on e = 0 worlds only; "
        bs, ts_ = berr[scored], tol[scored]
        a6 = bool(np.all(bs <= ts_)) if bs.size and np.isfinite(bs).all() else False
        rec["accuracy_error"] = float(np.nanmax(bs) * 1e3) if np.isfinite(bs).any() else None
        rec["metrics"].update(traj_err_mm=_f(err.max() * 1e3), traj_bound_ratio=_f(ratio.max()), a1_pass=a1,
                              a6_pass=a6, law_pass=a1 and a6, bounce_err_pct=bounce_pct, per_condition=per,
                              gravity=x["g"], **coverage(n, len(R01_COND)))
        if not np.isfinite(berr).all():
            rec["notes"] += "no impact within the horizon for some worlds (bounce law not evaluated there); "


# =================================================================================================================
# R02: single pendulum at 5 and 90 deg (period, A2) and a double pendulum (energy, A3)
# =================================================================================================================
R02_L = 1.0
# Bob radius 0.1 m and 10 cm square double-pendulum links, as Newton's own semi_implicit pendulum test
# (newton/tests/test_physics_verification.py:1657-1658): thinner bodies leave too little rotational inertia for the
# maximal-coordinate joint springs of SolverSemiImplicit. The period law uses the exact I_pivot, so radius is free.
R02_R = 0.1
R02_DOUBLE_HY = 0.05
R02_PIVOT_Z = 2.0
R02_DOUBLE_Y = 1.0  # double pendulum pivot sits 1 m beside the single one (same world, no contact possible)
R02_DOUBLE_HX = 0.25
R02_AMPS_DEG = (5.0, 90.0)
R02_NCROSS = 8


def ellip_k(k: float) -> float:
    """Complete elliptic integral of the first kind K(k) (modulus k) via the arithmetic-geometric mean."""
    a, b = 1.0, math.sqrt(1.0 - k * k)
    for _ in range(40):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


@wp.kernel
def _r02_track(body_q: wp.array[wp.transform], ctr: wp.array[wp.int32], dt: float, bpw: int, pz: float,
               th_prev: wp.array[float], ncross: wp.array[wp.int32], tcross: wp.array2d[float]):
    w = wp.tid()
    p = wp.transform_get_translation(body_q[w * bpw])
    th = wp.atan2(-p[0], pz - p[2])
    tp = th_prev[w]
    if tp > 0.0 and th <= 0.0:
        k = ncross[w]
        if k < tcross.shape[1]:
            t = float(ctr[0] + 1) * dt
            tcross[w, k] = t - dt + dt * tp / (tp - th)
            ncross[w] = k + 1
    th_prev[w] = th


class Pendulum(RigidScenario):
    id = "R02_pendulum"
    ncond = len(R02_AMPS_DEG)
    title = "Single pendulum period at 5 and 90 deg (exact elliptic law) and double-pendulum energy"
    horizon_s = 6.0
    has_contacts = False
    sample_every = 25
    accuracy_unit = "pct"
    accuracy_ref = ("max period error vs T = 4 sqrt(I_p/(m g d)) K(sin(theta0/2)) (A2, 0.1 %); energy drift <= 0.5 % "
                    "of the swing energy over 2 s (A3, newton/tests/test_physics_verification.py:236)")
    _mj = {"solver": "newton", "integrator": "implicitfast", "cone": "elliptic", "iterations": 50,
           "ls_iterations": 50, "nconmax": 1, "njmax": 16}
    configs = (
        Config("mujoco", "mujoco", dt=1e-3, params=_mj),
        Config("featherstone", "featherstone", dt=1e-3, params={"angular_damping": 0.0}),
        Config("xpbd_s3", "xpbd", dt=1e-3, substeps=3, params={"iterations": 20, "angular_damping": 0.0}),
        Config("semi_implicit_s3", "semi_implicit", dt=1e-3, substeps=3,
               params={"angular_damping": 0.0, "joint_attach_ke": 1.0e5, "joint_attach_kd": 10.0}),
        Config("kamino_dvi", "kamino", dt=1e-3, params={"dynamics_solver": "dvi"}),
        Config("kamino_padmm", "kamino", dt=1e-3, params={"dynamics_solver": "padmm", "constraints": {"alpha": 0.1},
                                                          "padmm": {"rho_0": 0.01}}),
        Config("steel", "steel", dt=1e-3, params={"solve_mode": "auto"}),
        ref_mujoco_cpu(1e-3, _mj),
    )

    def _scene(self, config: Config) -> newton.ModelBuilder:
        s = new_builder(config)
        bob = s.add_link(label="bob")
        s.add_shape_sphere(bob, radius=R02_R)
        j0 = s.add_joint_revolute(parent=-1, child=bob, axis=wp.vec3(0.0, 1.0, 0.0), armature=0.0,
                                  parent_xform=wp.transform(wp.vec3(0.0, 0.0, R02_PIVOT_Z), wp.quat_identity()),
                                  child_xform=wp.transform(wp.vec3(0.0, 0.0, R02_L), wp.quat_identity()))
        s.add_articulation([j0], label="single")
        hx, hy = R02_DOUBLE_HX, R02_DOUBLE_HY
        la = s.add_link(label="double_a")
        s.add_shape_box(la, hx=hx, hy=hy, hz=hy)
        lb = s.add_link(label="double_b")
        s.add_shape_box(lb, hx=hx, hy=hy, hz=hy)
        ja = s.add_joint_revolute(parent=-1, child=la, axis=wp.vec3(0.0, 1.0, 0.0), armature=0.0,
                                  parent_xform=wp.transform(wp.vec3(0.0, R02_DOUBLE_Y, R02_PIVOT_Z),
                                                            wp.quat_identity()),
                                  child_xform=wp.transform(wp.vec3(-hx, 0.0, 0.0), wp.quat_identity()))
        jb = s.add_joint_revolute(parent=la, child=lb, axis=wp.vec3(0.0, 1.0, 0.0), armature=0.0,
                                  parent_xform=wp.transform(wp.vec3(hx, 0.0, 0.0), wp.quat_identity()),
                                  child_xform=wp.transform(wp.vec3(-hx, 0.0, 0.0), wp.quat_identity()))
        s.add_articulation([ja, jb], label="double")  # starts horizontal (q = 0), released from rest
        return s

    def build(self, nworld: int, device: str, config: Config) -> Built:
        model = replicate(self._scene(config), nworld, config, device)
        bpw = model.body_count // nworld
        cond = np.arange(nworld) % len(R02_AMPS_DEG)
        th0 = np.radians(np.array(R02_AMPS_DEG))[cond]
        q_start = model.joint_q_start.numpy()
        jpw = model.joint_count // nworld
        single_q = q_start[np.arange(nworld) * jpw]  # joint 0 of each world is the single pendulum
        jq = model.joint_q.numpy()
        # rest configuration (hanging) for the energy reference: single 0, double a = +90 deg (link along -z), b = 0
        q_rest = jq.copy()
        q_rest[single_q] = 0.0
        q_rest[q_start[np.arange(nworld) * jpw + 1]] = 0.5 * math.pi
        rest = model.state()
        newton.eval_fk(model, wp.array(q_rest, dtype=float, device=device), model.joint_qd, rest)
        e_rest = body_energy(model, rest).reshape(nworld, bpw)
        jq[single_q] = th0
        model.joint_q.assign(jq)
        solver = make_solver(config, model)
        mass = float(model.body_mass.numpy()[0])
        i_cm = model.body_inertia.numpy()[0]
        i_p = float(i_cm[1, 1]) + mass * R02_L**2
        g = float(np.linalg.norm(model.gravity.numpy().reshape(-1, 3)[0]))
        t_exact = np.array([4.0 * math.sqrt(i_p / (mass * g * R02_L)) * ellip_k(math.sin(a / 2.0)) for a in th0])
        extra = {"cond": cond, "bpw": bpw, "e_rest": e_rest, "t_exact": t_exact,
                 "t0_small": 2.0 * math.pi * math.sqrt(i_p / (mass * g * R02_L)),
                 "th_prev": wp.array(th0.astype(np.float32), dtype=float, device=device),
                 "ncross": wp.zeros(nworld, dtype=wp.int32, device=device),
                 "tcross": wp.zeros((nworld, R02_NCROSS), dtype=float, device=device), "samples": []}
        return assemble(model, config, solver, nworld, None, extra)

    def post_step(self, b: Built, config: Config) -> None:
        x = b.extra
        wp.launch(_r02_track, dim=b.nworld, device=b.model.device,
                  inputs=[b.state_0.body_q, x["ctr"], config.dt, x["bpw"], R02_PIVOT_Z, x["th_prev"], x["ncross"],
                          x["tcross"]])

    def _swing(self, b: Built) -> np.ndarray:
        """Per world [single, double] energy above the hanging rest state [J]."""
        e = body_energy(b.model, b.state_0).reshape(b.nworld, b.extra["bpw"]) - b.extra["e_rest"]
        return np.stack([e[:, 0], e[:, 1:].sum(axis=1)], axis=1)

    def on_start(self, b: Built, rec: dict) -> None:
        b.extra["samples"].append((0.0, self._swing(b)))  # t is approximate (after warm-up); drift is relative

    def sample(self, b: Built, t: float, rec: dict) -> None:
        b.extra["samples"].append((t, self._swing(b)))

    def energy(self, b: Built) -> float:
        return float(self._swing(b).sum())  # referenced to the rest state, so drift_pct is % of swing energy

    def evaluate(self, b: Built, rec: dict) -> None:
        x, n = b.extra, b.nworld
        nc, tc = x["ncross"].numpy(), x["tcross"].numpy().astype(np.float64)
        period = np.array([(tc[w, k - 1] - tc[w, 0]) / (k - 1) if k >= 2 else np.nan for w, k in enumerate(nc)])
        perr = 100.0 * np.abs(period - x["t_exact"]) / x["t_exact"]
        ts = np.array([s[0] for s in x["samples"]])
        es = np.stack([s[1] for s in x["samples"]])  # (nsample, nworld, 2)
        e0 = es[0]
        win = ts <= 2.0 + 1e-9
        drift = 100.0 * np.max(np.abs(es[win] - e0), axis=0) / np.abs(e0)  # (nworld, 2)
        drift_all = 100.0 * np.max(np.abs(es - e0), axis=0) / np.abs(e0)
        per = []
        for c, amp in enumerate(R02_AMPS_DEG):
            sel = x["cond"] == c
            if sel.any():
                pm = period[sel][np.isfinite(period[sel])]
                per.append({"amplitude_deg": amp, "period_s": _f(pm.mean(), 6) if pm.size else None,
                            "period_exact_s": _f(x["t_exact"][sel][0], 6),
                            "period_err_pct": _f(np.nanmax(perr[sel]), 4) if np.isfinite(perr[sel]).any() else None,
                            "periods_seen": int(nc[sel].min()) - 1,
                            "single_energy_drift_pct_2s": _f(drift[sel, 0].max(), 4)})
        a2 = bool(np.all(perr <= 0.1)) if np.isfinite(perr).all() else False
        # A3 on the 90 deg single pendulum (the analogue of Newton's 1 rad test) and on every double pendulum; the
        # 5 deg swing energy is too small for a relative drift to mean anything
        single90 = drift[x["cond"] == 1, 0]
        a3_single = bool(np.all(single90 <= 0.5)) if single90.size else None
        a3_double = bool(np.all(drift[:, 1] <= 0.5))
        e_law = np.concatenate([single90, drift[:, 1]])
        a3 = a3_double and a3_single is not False
        rec["accuracy_error"] = float(np.nanmax(perr)) if np.isfinite(perr).any() else None
        rec["metrics"].update(period_err_pct=_f(np.nanmax(perr), 4) if np.isfinite(perr).any() else None,
                              energy_drift_pct_2s=_f(e_law.max(), 4),
                              single90_energy_drift_pct_2s=_f(single90.max(), 4) if single90.size else None,
                              double_energy_drift_pct_2s=_f(drift[:, 1].max(), 4),
                              double_energy_drift_pct_horizon=_f(drift_all[:, 1].max(), 4),
                              period_small_angle_s=_f(x["t0_small"], 6), a2_pass=a2, a3_single_pass=a3_single,
                              a3_double_pass=a3_double, a3_pass=a3, law_pass=a2 and a3,
                              per_condition=per, **coverage(n, len(R02_AMPS_DEG)))
        if not np.isfinite(perr).all():
            rec["notes"] += "fewer than 2 same-direction zero crossings (horizon too short for the period law); "


# =================================================================================================================
# R03: box on a plane under an eased gravity tilt; Coulomb onset, acceleration, stop distance, creep, isotropy (A5)
# =================================================================================================================
R03_MU = (0.3, 0.6)
R03_ONSET = (-1.0, -0.1, -0.05, 0.05, 0.1)  # tilt offsets from atan(mu) [deg]; -1 deg is also the creep cell
R03_ACCEL = 5.0  # sliding cells sit 5 deg above atan(mu)
R03_NDIR = 16
R03_V0 = 1.0  # stop-distance kick [m/s]
R03_EASE = 0.5  # gravity tilt eased in over this [s] (CON-51)
R03_KICK_T = 0.5
R03_BOX = (0.1, 0.1, 0.05)  # flat slab: no tipping below atan(hx/hz) = 63 deg
R03_GAP = 0.005


def r03_conditions() -> list[dict]:
    conds = [{"kind": "onset", "mu": mu, "off": off, "phi": 0.0} for mu in R03_MU for off in R03_ONSET]
    conds += [{"kind": "accel", "mu": mu, "off": R03_ACCEL, "phi": 360.0 * k / R03_NDIR}
              for mu in R03_MU for k in range(R03_NDIR)]
    conds += [{"kind": "stop", "mu": mu, "off": None, "phi": 360.0 * k / R03_NDIR}
              for mu in R03_MU for k in range(R03_NDIR)]
    return conds


R03_COND = r03_conditions()


@wp.kernel
def _r03_control(ctr: wp.array[wp.int32], dt: float, t_ease: float, g: float, theta: wp.array[float],
                 phi: wp.array[float], kick_step: int, kick_v: wp.array[float], gravity: wp.array[wp.vec3],
                 mj_gravity: wp.array[wp.vec3], has_mj: int, body_q: wp.array[wp.transform],
                 body_qd: wp.array[wp.spatial_vector], joint_qd: wp.array[float], p_kick: wp.array[wp.vec3]):
    w = wp.tid()
    s = wp.clamp(float(ctr[0]) * dt / t_ease, 0.0, 1.0)
    th = (0.5 - 0.5 * wp.cos(wp.pi * s)) * theta[w]
    gv = g * wp.vec3(wp.sin(th) * wp.cos(phi[w]), wp.sin(th) * wp.sin(phi[w]), -wp.cos(th))
    gravity[w] = gv
    if has_mj != 0:
        mj_gravity[w] = gv
    if ctr[0] == kick_step and kick_v[w] > 0.0:
        v = kick_v[w] * wp.vec3(wp.cos(phi[w]), wp.sin(phi[w]), 0.0)
        body_qd[w] = wp.spatial_vector(v, wp.vec3(0.0))
        for k in range(3):
            joint_qd[w * 6 + k] = v[k]
            joint_qd[w * 6 + 3 + k] = 0.0
        p_kick[w] = wp.transform_get_translation(body_q[w])


class InclineFriction(RigidScenario):
    id = "R03_incline_friction"
    ncond = len(R03_COND)
    title = "Box under an eased gravity tilt: Coulomb onset, slide acceleration, stop distance, creep, isotropy"
    horizon_s = 4.0
    accuracy_unit = "deg"
    accuracy_ref = ("Coulomb (A5): onset |theta_s - atan mu| <= 0.1 deg (cells at -0.1/-0.05/+0.05/+0.1 deg); a = g(sin - "
                    "mu cos) 1 %; stop v0^2/(2 mu g) 1 %; creep at -1 deg <= 0.01 mm/10 s; isotropy 1 % over 16 dirs "
                    "(newton/tests/test_rigid_friction_ramp.py:68-88,391)")
    _mj = {"use_mujoco_contacts": True, "solver": "newton", "integrator": "implicitfast", "impratio": 10.0,
           "iterations": 200, "ls_iterations": 100, "nconmax": 16, "njmax": 96}
    configs = (
        Config("mujoco_elliptic", "mujoco", dt=DT600, params={**_mj, "cone": "elliptic"}),
        Config("mujoco_pyramidal", "mujoco", dt=DT600, params={**_mj, "cone": "pyramidal"}),
        Config("mujoco_noslip", "mujoco", dt=DT600, params={**_mj, "cone": "elliptic", "noslip_iterations": 10}),
        Config("kamino_dvi", "kamino", dt=DT600, params={"dynamics_solver": "dvi"}),
        Config("xpbd", "xpbd", dt=DT600, params={"iterations": 10}),
        Config("steel", "steel", dt=DT600, params={"solve_mode": "auto"}),
        Config("featherstone_s10", "featherstone", dt=DT600, substeps=10, params={}),
        Config("semi_implicit_s10", "semi_implicit", dt=DT600, substeps=10, params={}),
        # one condition per run on CPU; `python -m bench.scenarios._rigid_ref r03` runs all 74 and scores them together
        ref_mujoco_cpu(DT600, {**_mj, "cone": "elliptic"}, "ref_mujoco_cpu_elliptic"),
        ref_mujoco_cpu(DT600, {**_mj, "cone": "pyramidal"}, "ref_mujoco_cpu_pyramidal"),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        gap = contact_gap(config, R03_GAP)
        scene = new_builder(config, gap)
        mat = newton.ModelBuilder.ShapeConfig(mu=0.5, ke=1.0e5, kd=1.0e3, kf=1.0e3, gap=gap)
        scene.add_ground_plane(cfg=mat)
        hx, hy, hz = R03_BOX
        box = scene.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, hz), q=wp.quat_identity()), label="box")
        scene.add_shape_box(box, hx=hx, hy=hy, hz=hz, cfg=mat)
        model = replicate(scene, nworld, config, device)
        apply_material_overrides(model, config)
        cond = np.arange(nworld) % len(R03_COND)
        cs = [R03_COND[c] for c in cond]
        mu_w = np.array([c["mu"] for c in cs])
        theta = np.array([0.0 if c["off"] is None else math.atan(c["mu"]) + math.radians(c["off"]) for c in cs])
        phi = np.radians([c["phi"] for c in cs])
        kick = np.array([R03_V0 if c["kind"] == "stop" else 0.0 for c in cs])
        model.shape_material_mu.assign(mu_w[model.shape_world.numpy()].astype(np.float32))
        solver = make_solver(config, model, 8 * nworld)
        g = float(np.linalg.norm(model.gravity.numpy().reshape(-1, 3)[0]))

        def arr(v):
            return wp.array(np.asarray(v, dtype=np.float32), dtype=float, device=device)

        mjw = config.solver == "mujoco" and not solver.use_mujoco_cpu  # CPU reference: gravity copied in pre_step
        mjg = solver.mjw_model.opt.gravity if mjw else model.gravity
        extra = {"cond": cond, "cs": cs, "mu": mu_w, "theta": theta, "phi": phi, "g": g, "d_theta": arr(theta),
                 "d_phi": arr(phi), "d_kick": arr(kick), "mjg": mjg, "has_mj": int(mjw),
                 "cpu_mj": config.solver == "mujoco" and not mjw,
                 "kick_step": int(round(R03_KICK_T / config.dt)),
                 "p_kick": wp.zeros(nworld, dtype=wp.vec3, device=device), "ts": [], "pos": [], "vel": []}
        return assemble(model, config, solver, nworld, 8 * nworld, extra)

    def pre_step(self, b: Built, config: Config) -> None:
        x, s = b.extra, b.state_0
        wp.launch(_r03_control, dim=b.nworld, device=b.model.device,
                  inputs=[x["ctr"], config.dt, R03_EASE, x["g"], x["d_theta"], x["d_phi"], x["kick_step"], x["d_kick"],
                          b.model.gravity, x["mjg"], x["has_mj"], s.body_q, s.body_qd, s.joint_qd, x["p_kick"]])
        if x["cpu_mj"]:  # offline CPU MuJoCo reference (host arrays): gravity every step, the kick once
            b.solver.mj_model.opt.gravity[:] = b.model.gravity.numpy()[0]
            if int(x["ctr"].numpy()[0]) == x["kick_step"]:
                b.solver._update_mjc_data(b.solver.mj_data, b.model, s)

    def sample(self, b: Built, t: float, rec: dict) -> None:
        b.extra["ts"].append(t)
        b.extra["pos"].append(b.state_0.body_q.numpy()[:, :3].astype(np.float64))
        b.extra["vel"].append(b.state_0.body_qd.numpy()[:, :3].astype(np.float64))

    # no energy(): gravity is re-oriented during the run and sliding dissipates, so a drift % would mean nothing

    def evaluate(self, b: Built, rec: dict) -> None:
        x, n, g = b.extra, b.nworld, b.extra["g"]
        ts, pos, vel = np.array(x["ts"]), np.stack(x["pos"]), np.stack(x["vel"])
        t_end = ts[-1]
        kinds = np.array([c["kind"] for c in x["cs"]])
        res = {"onset": {}, "creep": {}, "accel": {}, "stop": {}}

        def first(t_min):
            idx = np.nonzero(ts >= t_min - 1e-9)[0]
            return int(idx[0]) if idx.size and idx[0] < len(ts) - 1 else None

        i_on, i_creep, i_acc = first(R03_EASE + 0.25), first(R03_EASE + 0.5), first(R03_EASE + 0.2)
        for mu in R03_MU:
            crit = math.atan(mu)
            # onset: classify cells by displacement over the hold window vs half the Coulomb distance at +0.05 deg
            cells = [(w, x["cs"][w]["off"]) for w in range(n) if kinds[w] in ("onset", "accel") and x["mu"][w] == mu
                     and x["cs"][w]["phi"] == 0.0]
            if i_on is not None and cells:
                tw = t_end - ts[i_on]
                th = crit + math.radians(0.05)
                a_min = g * (math.sin(th) - mu * math.cos(th))
                thr = 0.5 * (0.5 * a_min * tw * tw)
                slides = {}
                for w, off in cells:
                    d = float(np.linalg.norm(pos[-1, w] - pos[i_on, w]))
                    slides[off] = bool(slides.get(off, False) or d > thr)
                sl = sorted(o for o, s in slides.items() if s)
                st = sorted(o for o, s in slides.items() if not s)
                hi = sl[0] if sl else None
                lo = max((o for o in st if hi is None or o < hi), default=None)
                err = max(abs(lo) if lo is not None else abs(min(slides)), abs(hi) if hi is not None else abs(max(slides)))
                res["onset"][str(mu)] = {"bracket_deg": [lo, hi], "onset_err_deg": err,
                                         "open": lo is None or hi is None, "threshold_mm": thr * 1e3,
                                         "cells": {str(o): s for o, s in sorted(slides.items())}}
            wc = [w for w in range(n) if kinds[w] == "onset" and x["mu"][w] == mu and x["cs"][w]["off"] == -1.0]
            if i_creep is not None and wc:
                tw = t_end - ts[i_creep]
                d = max(float(np.linalg.norm(pos[-1, w] - pos[i_creep, w])) for w in wc)
                res["creep"][str(mu)] = d * 1e3 * 10.0 / tw
            wa = [w for w in range(n) if kinds[w] == "accel" and x["mu"][w] == mu]
            if i_acc is not None and wa and len(ts) - i_acc >= 2:
                th = crit + math.radians(R03_ACCEL)
                a_exp = g * (math.sin(th) - mu * math.cos(th))
                am = []
                for w in wa:
                    ph = x["phi"][w]
                    # the plane stays flat and gravity tilts, so the slide is in-plane along azimuth phi
                    sp = vel[i_acc:, w, :2] @ np.array([math.cos(ph), math.sin(ph)])
                    am.append(np.polyfit(ts[i_acc:], sp, 1)[0])
                am = np.array(am)
                res["accel"][str(mu)] = {"a_expected": a_exp, "a_mean": float(am.mean()),
                                         "accel_err_pct": float(np.max(np.abs(am - a_exp)) / a_exp * 100.0),
                                         "isotropy_pct": float((am.max() - am.min()) / abs(am.mean()) * 100.0)
                                         if len(am) > 1 else None, "ndir": len(am)}
            ws = [w for w in range(n) if kinds[w] == "stop" and x["mu"][w] == mu]
            d_exp = R03_V0**2 / (2.0 * mu * g)
            if ws and t_end >= R03_KICK_T + R03_V0 / (mu * g) + 0.1:
                pk = x["p_kick"].numpy().astype(np.float64)
                d = np.array([np.linalg.norm(pos[-1, w, :2] - pk[w, :2]) for w in ws])
                moving = np.array([np.linalg.norm(vel[-1, w, :2]) > 0.01 for w in ws])
                res["stop"][str(mu)] = {"d_expected_mm": d_exp * 1e3, "d_mean_mm": float(d.mean() * 1e3),
                                        "stop_err_pct": float(np.max(np.abs(d - d_exp)) / d_exp * 100.0),
                                        "isotropy_pct": float((d.max() - d.min()) / d.mean() * 100.0) if len(d) > 1
                                        else None, "still_moving": int(moving.sum()), "ndir": len(d)}

        def worst(group, key):
            vals = [v[key] if isinstance(v, dict) else v for v in res[group].values()]
            vals = [v for v in vals if v is not None]
            return max(vals) if vals else None

        onset = worst("onset", "onset_err_deg")
        accel, stop = worst("accel", "accel_err_pct"), worst("stop", "stop_err_pct")
        creep = max(res["creep"].values()) if res["creep"] else None
        iso = [v for v in (worst("accel", "isotropy_pct"), worst("stop", "isotropy_pct")) if v is not None]
        iso = max(iso) if iso else None
        checks = {"onset": (onset, 0.1), "accel": (accel, 1.0), "stop": (stop, 1.0), "creep": (creep, 0.01),
                  "isotropy": (iso, 1.0)}
        passed = {k: (bool(v <= lim) if v is not None else None) for k, (v, lim) in checks.items()}
        rec["accuracy_error"] = onset
        rec["metrics"].update(onset_err_deg=_f(onset, 3), accel_err_pct=_f(accel, 3), stop_err_pct=_f(stop, 3),
                              creep_mm_per_10s=_f(creep, 5), isotropy_pct=_f(iso, 3), checks_pass=passed,
                              law_pass=all(passed.values()) if all(v is not None for v in passed.values()) else None,
                              per_condition=json.loads(json.dumps(res, default=float)),
                              **coverage(n, len(R03_COND)))


# =================================================================================================================
# R04: tower of 10 cubes at rest (static equilibrium, creep, energy gain, penetration)
# =================================================================================================================
R04_N = 10
R04_HALF = 0.025
R04_GAP = 0.005


class BoxStack(RigidScenario):
    id = "R04_box_stack_10"
    title = "Tower of 10 x 5 cm cubes resting 10 s: top drift, tilt, energy gain, penetration"
    horizon_s = 10.0
    accuracy_unit = "mm_per_10s"
    accuracy_ref = ("top-cube drift after the 1 s settle, scaled to 10 s; static equilibrium: drift <= 0.1 mm/10 s "
                    "(bench.md acceptance #8), energy gain <= 1 mJ (CON-30), penetration <= 1 mm")
    _mj = {"use_mujoco_contacts": True, "integrator": "implicitfast", "cone": "elliptic", "ls_iterations": 50,
           "nconmax": 96, "njmax": 512}
    configs = (
        Config("mujoco_newton", "mujoco", dt=DT600, params={**_mj, "solver": "newton", "iterations": 50}),
        Config("mujoco_cg", "mujoco", dt=DT600, params={**_mj, "solver": "cg", "iterations": 100}),
        Config("kamino_dvi", "kamino", dt=DT600, params={"dynamics_solver": "dvi"}),
        Config("xpbd", "xpbd", dt=DT600, params={"iterations": 10}),
        Config("steel", "steel", dt=DT600, params={"solve_mode": "auto"}),
        # Penalty contacts on 0.125 kg cubes: kd and kf act per contact point (4 per face) on the explicit substep,
        # so kd = kf = 100 keeps 4 kd h / m ~ 0.5 < 2; the defaults (kd 100, kf 1000) and kd = 1e3 diverge.
        Config("featherstone_s10", "featherstone", dt=DT600, substeps=10,
               params={"shape_ke": 1.0e5, "shape_kd": 100.0, "shape_kf": 100.0}),
        Config("semi_implicit_s10", "semi_implicit", dt=DT600, substeps=10,
               params={"shape_ke": 1.0e5, "shape_kd": 100.0, "shape_kf": 100.0}),
        # VBD's compliant contact takes its stiffness from shape ke (default 2.5e3 sags 1.6 mm and creeps
        # 14-76 mm/10 s here); kd = 0 as newton/examples/vbd/example_vbd_rigid_rigid_contact.py:41
        Config("vbd", "vbd", dt=DT600, params={"iterations": 10, "shape_ke": 1.0e5, "shape_kd": 0.0}),
        ref_mujoco_cpu(DT600, {**_mj, "solver": "newton", "iterations": 50}),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        gap = contact_gap(config, R04_GAP)
        scene = new_builder(config, gap)
        scene.add_ground_plane()
        h = R04_HALF
        for i in range(R04_N):
            body = scene.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, h + 2.0 * h * i), q=wp.quat_identity()),
                                  label=f"cube{i}")
            scene.add_shape_box(body, hx=h, hy=h, hz=h)
        model = replicate(scene, nworld, config, device)
        apply_material_overrides(model, config)
        solver = make_solver(config, model, 12 * R04_N * nworld)
        b = assemble(model, config, solver, nworld, 12 * R04_N * nworld,
                     {"p0": None, "ts": [], "top": [], "e": [], "xy": []})
        b.extra["p0"] = b.state_0.body_q.numpy().reshape(nworld, R04_N, 7)[:, -1].astype(np.float64)
        return b

    def sample(self, b: Built, t: float, rec: dict) -> None:
        q = b.state_0.body_q.numpy().reshape(b.nworld, R04_N, 7).astype(np.float64)
        b.extra["ts"].append(t)
        b.extra["top"].append(q[:, -1])
        b.extra["xy"].append(np.max(np.linalg.norm(q[:, :, :2] - q[:, :1, :2], axis=2), axis=1))
        b.extra["e"].append(body_energy(b.model, b.state_0).reshape(b.nworld, R04_N).sum(axis=1))

    def on_start(self, b: Built, rec: dict) -> None:
        b.extra["e_start"] = body_energy(b.model, b.state_0).reshape(b.nworld, R04_N).sum(axis=1)

    def energy(self, b: Built) -> float:
        return float(body_energy(b.model, b.state_0).sum())

    def evaluate(self, b: Built, rec: dict) -> None:
        x = b.extra
        ts, top = np.array(x["ts"]), np.stack(x["top"])
        t_ref = min(1.0, 0.25 * ts[-1])
        i0 = int(np.nonzero(ts >= t_ref - 1e-9)[0][0])
        win = ts[-1] - ts[i0]
        drift = np.linalg.norm(top[-1, :, :3] - top[i0, :, :3], axis=1)
        rate = drift * 1e3 * 10.0 / win if win > 0 else np.full_like(drift, np.nan)
        total = np.linalg.norm(top[-1, :, :3] - x["p0"][:, :3], axis=1) * 1e3
        qw = np.abs(top[-1, :, 6]).clip(0.0, 1.0)
        tilt = np.degrees(2.0 * np.arccos(qw))
        gain = (np.max(np.stack(x["e"]), axis=0) - x["e_start"]) * 1e3
        spread = np.max(np.stack(x["xy"]), axis=0) * 1e3
        pen = rec.get("penetration_p99_mm")
        drift_ok = bool(np.all(rate <= 0.1))
        rec["accuracy_error"] = float(np.max(rate))
        rec["metrics"].update(worlds_drift_fail=int(np.count_nonzero(~(rate <= 0.1))),
                              worlds_collapsed=int(np.count_nonzero(spread > R04_HALF * 1e3)),
                              top_drift_p50_mm_per_10s=_f(np.median(rate), 5))
        rec["metrics"].update(top_drift_mm_per_10s=_f(np.max(rate), 5), top_drift_total_mm=_f(np.max(total), 4),
                              tilt_deg=_f(np.max(tilt), 4), energy_gain_mj=_f(np.max(gain), 4),
                              max_xy_offset_mm=_f(np.max(spread), 3), collapsed=bool(np.any(spread > R04_HALF * 1e3)),
                              drift_window_s=_f(win, 3), law_pass=drift_ok and float(np.max(gain)) <= 1.0
                              and (pen is None or pen <= 1.0))


# =================================================================================================================
# R05: 64 / 256 mixed bodies dropped into a bin (pa/bench/tools/bin_scene.py MJCF, same options)
# =================================================================================================================
R05_HALF, R05_PITCH = 0.015, 0.04


def bin_xml(n: int, dt: float = 0.002) -> tuple[str, float]:
    """The pa/bench/tools/bin_scene.py `mixed` bin (box/sphere/capsule cycling), returned with the bin half width."""
    k = math.ceil(n ** (1 / 3)) if n >= 27 else math.ceil(math.sqrt(n))
    w = max(0.12, k * R05_PITCH / 2 + 0.03)
    layers = math.ceil(n / (k * k))
    wall_h = 0.05 + 0.6 * layers * R05_PITCH
    geoms = (f'<geom type="box" size="{R05_HALF} {R05_HALF * 0.8} {R05_HALF * 0.6}"/>',
             f'<geom type="sphere" size="{R05_HALF}"/>', f'<geom type="capsule" size="{R05_HALF * 0.6} {R05_HALF}"/>')
    bodies = []
    for i in range(n):
        ix, iy, iz = i % k, (i // k) % k, i // (k * k)
        x, y, z = (ix - (k - 1) / 2) * R05_PITCH, (iy - (k - 1) / 2) * R05_PITCH, 0.05 + iz * R05_PITCH
        rot = (0.13 * i) % 3.14
        bodies.append(f'<body name="o{i}" pos="{x:.4f} {y:.4f} {z:.4f}" euler="{rot:.3f} {0.7 * rot:.3f} 0">'
                      f'<freejoint/>{geoms[i % 3]}</body>')
    t = 0.01
    walls = (f'<geom type="box" size="{w + t} {w + t} {t}" pos="0 0 {-t}"/>'
             f'<geom type="box" size="{t} {w + t} {wall_h / 2}" pos="{w + t} 0 {wall_h / 2}"/>'
             f'<geom type="box" size="{t} {w + t} {wall_h / 2}" pos="{-w - t} 0 {wall_h / 2}"/>'
             f'<geom type="box" size="{w + t} {t} {wall_h / 2}" pos="0 {w + t} {wall_h / 2}"/>'
             f'<geom type="box" size="{w + t} {t} {wall_h / 2}" pos="0 {-w - t} {wall_h / 2}"/>')
    xml = f"""<mujoco model="bin_mixed_{n}">
  <option timestep="{dt}" solver="Newton" iterations="50" ls_iterations="50" cone="elliptic" integrator="implicitfast"/>
  <default><geom condim="3" density="700" friction="0.5 0.005 0.0001" solref="0.004 1"/></default>
  <worldbody>{walls}{''.join(bodies)}</worldbody>
</mujoco>"""
    return xml, w


def bin_metrics(p_half: np.ndarray, p_end: np.ndarray, window_s: float, half_width: float) -> dict:
    """pa/bench/tools/fidelity.py bin_metrics on positions (nbody, 3): creep (median travel of bodies still in the
    bin, mm per 10 s), escaped count, pile top [m]."""
    out = (np.abs(p_end[:, 0]) > half_width + 0.02) | (np.abs(p_end[:, 1]) > half_width + 0.02) | (p_end[:, 2] < -0.05)
    travel = np.linalg.norm(p_end - p_half, axis=1)
    keep = ~out
    creep = float(np.median(travel[keep])) * 1e3 * 10.0 / window_s if keep.any() and window_s > 0 else None
    p90 = float(np.percentile(travel[keep], 90)) * 1e3 * 10.0 / window_s if keep.any() and window_s > 0 else None
    return {"creep_mm_per_10s": creep, "creep_mm_per_10s_p90": p90, "escaped": int(out.sum()),
            "pile_top_m": float(p_end[keep, 2].max()) if keep.any() else None}


def _r05_configs() -> tuple[Config, ...]:
    out = []
    for n in (64, 256):
        cap = 1024 if n == 64 else 256
        mj = {"n": n, "solver": "newton", "integrator": "implicitfast", "cone": "elliptic", "iterations": 50,
              "ls_iterations": 50, "nconmax": 8 * n, "njmax": 24 * n}
        out += [Config(f"mujoco_n{n}", "mujoco", dt=0.002, params={**mj, "use_mujoco_contacts": True}, max_nworld=cap),
                Config(f"mujoco_nc_n{n}", "mujoco", dt=0.002, params={**mj, "use_mujoco_contacts": False},
                       max_nworld=cap),
                Config(f"kamino_dvi_n{n}", "kamino", dt=0.002, params={"n": n, "dynamics_solver": "dvi",
                                                                       "sparse_dynamics": True, "sparse_jacobian": True}),
                Config(f"xpbd_n{n}", "xpbd", dt=0.002, params={"n": n, "iterations": 10}),
                # default free_body_constraint_max (512 rows/world) drops rows at n = 64 within 1 s
                Config(f"steel_n{n}", "steel", dt=0.002, params={"n": n, "solve_mode": "auto", "iterations": 12,
                                                                  "free_body_constraint_max": 24 * n}),
                # the MJCF solref (0.004, 1) imports as ke 6.25e4, kd 500, which are acceleration-space values;
                # as penalty forces on these 7-10 g bodies they diverge at once. 4 kd h / m ~ 0.2 here.
                Config(f"featherstone_s10_n{n}", "featherstone", dt=0.002, substeps=10,
                       params={"n": n, "shape_ke": 1.0e4, "shape_kd": 2.0, "shape_kf": 2.0})]
    return tuple(out)


class BoxPile(RigidScenario):
    id = "R05_box_pile"
    title = "64 / 256 mixed boxes, spheres and capsules dropped into a bin, 4 s (n in Config.params)"
    horizon_s = 4.0
    accuracy_unit = "mm_per_10s"
    accuracy_ref = ("median resting creep over the 2nd half (pa/bench/tools/fidelity.py bin_metrics), max over worlds; "
                    "no escapes, KE -> 0, rest penetration <= 1 mm, creep <= 0.1 mm/10 s; prior 200 boxes: Kamino "
                    "0.001, CPU MuJoCo 0.018, XPBD 53-94 mm/10 s (pa/findings/bench.md)")
    configs = _r05_configs()

    def build(self, nworld: int, device: str, config: Config) -> Built:
        n = int(config.params["n"])
        xml, half_width = bin_xml(n, config.dt)
        scene = new_builder(config, contact_gap(config, 0.005))  # 5 mm gap, 24 slots per body (pa gap5mm_cap24)
        scene.add_mjcf(xml)
        model = replicate(scene, nworld, config, device)
        apply_material_overrides(model, config)
        solver = make_solver(config, model, 24 * n * nworld)
        bpw = model.body_count // nworld
        return assemble(model, config, solver, nworld, 24 * n * nworld,
                        {"n": n, "bpw": bpw, "half_width": half_width, "ts": [], "ke": [], "p_half": None})

    def on_start(self, b: Built, rec: dict) -> None:
        b.extra["t_half"] = 0.5 * float(rec["horizon_s"])

    def _ke(self, b: Built) -> np.ndarray:
        qd = b.state_0.body_qd.numpy().astype(np.float64)
        m = b.model.body_mass.numpy().astype(np.float64)
        ke = 0.5 * m * np.sum(qd[:, :3] ** 2, axis=1)  # linear KE (as pa newton_bin.py)
        return ke.reshape(b.nworld, b.extra["bpw"]).sum(axis=1)

    def sample(self, b: Built, t: float, rec: dict) -> None:
        x = b.extra
        x["ts"].append(t)
        x["ke"].append(self._ke(b))
        if x["p_half"] is None and t >= x.get("t_half", np.inf) - 1e-9:
            x["p_half"] = (t, b.state_0.body_q.numpy()[:, :3].astype(np.float64))

    # no energy(): the drop dissipates almost all of it by design (KE -> 0 is checked in evaluate instead)

    def evaluate(self, b: Built, rec: dict) -> None:
        x, n, nw = b.extra, b.extra["n"], b.nworld
        ts, ke = np.array(x["ts"]), np.stack(x["ke"])  # (nsample, nworld)
        p_end = b.state_0.body_q.numpy()[:, :3].astype(np.float64).reshape(nw, x["bpw"], 3)
        thr = n * 1e-6  # world KE below 1 uJ per body counts as settled
        settled = np.all(ke < thr, axis=1)
        settle_t = None
        for i in range(len(ts)):
            if settled[i:].all():
                settle_t = float(ts[i])
                break
        per_world = []
        if x["p_half"] is not None:
            t_half, p_half = x["p_half"]
            p_half = p_half.reshape(nw, x["bpw"], 3)
            per_world = [bin_metrics(p_half[w], p_end[w], ts[-1] - t_half, x["half_width"]) for w in range(nw)]
        pen = penetration_mm(b)
        creeps = [m["creep_mm_per_10s"] for m in per_world if m["creep_mm_per_10s"] is not None]
        p90s = [m["creep_mm_per_10s_p90"] for m in per_world if m["creep_mm_per_10s_p90"] is not None]
        tops = [m["pile_top_m"] for m in per_world if m["pile_top_m"] is not None]
        escaped = sum(m["escaped"] for m in per_world) if per_world else None
        if escaped is None:  # horizon too short for a half-window sample: still count escapes at the end
            escaped = sum(bin_metrics(p_end[w], p_end[w], 1.0, x["half_width"])["escaped"] for w in range(nw))
        rest_pen = float(pen.max()) if pen.size else 0.0
        creep = max(creeps) if creeps else None
        rec["accuracy_error"] = creep
        rec["metrics"].update(n_bodies=n, escaped=int(escaped), settle_time_s=settle_t, ke_end_j=_f(ke[-1].max(), 8),
                              rest_pen_mm=_f(rest_pen, 4), creep_mm=_f(creep, 5), creep_p90_mm=_f(max(p90s), 4)
                              if p90s else None, pile_height_mm=_f(np.mean(tops) * 1e3, 2) if tops else None,
                              law_pass=bool(escaped == 0 and rest_pen <= 1.0 and creep is not None and creep <= 0.1
                                            and settle_t is not None))


# =================================================================================================================
# R06: Franka picks, lifts, carries and places a 5 cm cube from a precomputed IK joint trajectory
# =================================================================================================================
# Panda kinematics and inertias from MuJoCo Menagerie franka_emika_panda (Apache-2.0), as vendored in
# mujoco_warp@a9daa05f:benchmarks/franka_emika_panda/panda.xml. Link collision meshes are dropped (self collision
# is off and the links never reach the cube or ground); the hand capsule and the four fingertip pad boxes per finger
# are kept, which are the shapes that touch the cube. No actuators or equality: joint drives are set below.
PANDA_XML = """<mujoco model="panda">
  <compiler angle="radian" autolimits="true"/>
  <default><joint armature="0.1" damping="0" axis="0 0 1"/></default>
  <worldbody>
    <body name="link0">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974" fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762" fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="joint1" range="-2.8973 2.8973"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495" fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="joint2" range="-1.7628 1.7628"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2" fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <joint name="joint3" range="-2.8973 2.8973"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2" fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="joint4" range="-3.0718 -0.0698"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2" fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="joint5" range="-2.8973 2.8973"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2" fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="joint6" range="-0.0175 3.7525"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2" fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="joint7" range="-2.8973 2.8973"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="hand_capsule" type="capsule" size="0.04 0.06" quat="1 1 0 0" pos="0 0 0.03"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.011 0.0075 0.01" pos="0 0.0185 0.011"/>
                        <geom type="box" size="0.011 0.0044 0.0019" pos="0 0.0068 0.0022"/>
                        <geom type="box" size="0.00875 0.0035 0.01175" pos="0 0.0159 0.02835" quat="1 0.25 0 0"/>
                        <geom name="left_pad" type="box" size="0.00875 0.0076 0.00825" pos="0 0.00758 0.04525"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.011 0.0075 0.01" pos="0 0.0185 0.011"/>
                        <geom type="box" size="0.011 0.0044 0.0019" pos="0 0.0068 0.0022"/>
                        <geom type="box" size="0.00875 0.0035 0.01175" pos="0 0.0159 0.02835" quat="1 0.25 0 0"/>
                        <geom name="right_pad" type="box" size="0.00875 0.0076 0.00825" pos="0 0.00758 0.04525"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
# Drive gains, effort limits and armature from newton/examples/ik/example_ik_cube_stacking.py:316-325.
R06_KE = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0)
R06_KD = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0)
R06_EFFORT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0)
R06_ARMATURE = (0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11, 0.15, 0.15)
R06_CUBE = 0.05
R06_DENSITY = 400.0
R06_MU = 0.75
R06_PICK = (0.5, -0.15)
R06_PLACE = (0.5, 0.15)
R06_TCP = 0.1034  # grasp centre (fingertip pad centres) on the hand z axis [m]
R06_RATE = 100.0  # trajectory samples per second (linear interpolation in the replay kernel)
R06_OPEN = 0.04
# (t_end, tcp z above the cube centre, tcp y, finger target): piecewise segments with cosine easing
R06_PLAN = ((0.5, 0.10, R06_PICK[1], R06_OPEN),  # settle at pre-grasp
            (1.3, 0.0, R06_PICK[1], R06_OPEN),  # descend
            (1.8, 0.0, R06_PICK[1], 0.0),  # close
            (2.0, 0.0, R06_PICK[1], 0.0),  # hold
            (2.8, 0.10, R06_PICK[1], 0.0),  # lift 10 cm
            (4.0, 0.10, R06_PLACE[1], 0.0),  # carry 30 cm
            (4.8, 0.001, R06_PLACE[1], 0.0),  # lower to 1 mm above the ground
            (5.3, 0.001, R06_PLACE[1], R06_OPEN),  # open
            (6.0, 0.10, R06_PLACE[1], R06_OPEN))  # retract
R06_SLIP_WINDOW = (1.95, 4.0)  # in-hand slip measured over lift and carry, from the end of the hold
R06_MOTION_END = 6.0


def _panda_builder(cfg: Config | None) -> newton.ModelBuilder:
    b = new_builder(cfg) if cfg is not None else newton.ModelBuilder()
    b.add_mjcf(PANDA_XML, enable_self_collisions=False)
    from alloy._vendor.newton import JointTargetMode

    for i in range(9):
        b.joint_target_ke[i], b.joint_target_kd[i] = R06_KE[i], R06_KD[i]
        b.joint_effort_limit[i], b.joint_armature[i] = R06_EFFORT[i], R06_ARMATURE[i]
        # position + velocity targets: with POSITION only, kd pulls toward zero velocity and the arm lags ~3 deg
        b.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)
    return b


def _ease(a: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(a, 0.0), 1.0))


def r06_plan_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times, TCP targets (T, 3) and finger targets (T,) at R06_RATE from R06_PLAN."""
    ts = np.arange(0.0, R06_MOTION_END + 0.5 + 1e-9, 1.0 / R06_RATE)
    tcp, fing = np.zeros((len(ts), 3)), np.zeros(len(ts))
    h = 0.5 * R06_CUBE
    prev = (0.0,) + R06_PLAN[0][1:]
    for i, t in enumerate(ts):
        p0 = prev
        for seg in R06_PLAN:
            if t <= seg[0]:
                a = _ease((t - p0[0]) / max(seg[0] - p0[0], 1e-9))
                z = p0[1] + a * (seg[1] - p0[1])
                y = p0[2] + a * (seg[2] - p0[2])
                f = p0[3] + a * (seg[3] - p0[3])
                break
            p0 = seg
        else:
            z, y, f = R06_PLAN[-1][1:]
        tcp[i] = (R06_PICK[0], y, h + z)
        fing[i] = f
    return ts, tcp, fing


@functools.lru_cache(maxsize=2)
def r06_trajectory(device: str) -> dict:
    """Joint-target trajectory from Newton IK (one batched LM solve over all samples, seeded by a warm-started
    waypoint chain) plus gravity feed-forward torques g(q) (newton.eval_inverse_dynamics_passive). Cached per
    process, so ladder rungs and repeat runs reuse it."""
    import alloy._vendor.newton.ik as ik

    # plan on the CPU whatever the run device: the result is numpy, and on CUDA the tiled LM kernel asks for 235 KB of
    # shared memory (L4 has 101 KB: "_lm_solve_tiled ... requests 235936 bytes of dynamic shared memory")
    device = "cpu"
    ts, tcp, fing = r06_plan_samples()
    model = _panda_builder(None).finalize(device=device)
    hand = next(i for i, lab in enumerate(model.body_label) if lab.rsplit("/", 1)[-1] == "hand")  # MJCF path labels
    down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)  # hand z down, fingers close along world y
    seed = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785, R06_OPEN, R06_OPEN], dtype=np.float32)
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()

    def solve(targets: np.ndarray, q_init: np.ndarray, iters: int) -> np.ndarray:
        n = len(targets)
        pos = ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.0, 0.0, R06_TCP),
                                     target_positions=wp.array(targets.astype(np.float32), dtype=wp.vec3, device=device))
        rot = ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat_identity(),
                                     target_rotations=wp.array(np.tile(np.array(down)[None, :4], (n, 1)).astype(np.float32),
                                                               dtype=wp.vec4, device=device))
        lim = ik.IKObjectiveJointLimit(joint_limit_lower=wp.array(np.tile(lower, n), dtype=float, device=device),
                                       joint_limit_upper=wp.array(np.tile(upper, n), dtype=float, device=device))
        solver = ik.IKSolver(model=model, n_problems=n, objectives=[pos, rot, lim], lambda_initial=0.1,
                             jacobian_mode=ik.IKJacobianType.ANALYTIC)
        q = wp.array(q_init.astype(np.float32), dtype=float, device=device).reshape((n, -1))
        solver.step(q, q, iterations=iters)
        return q.numpy().reshape(n, -1)

    # warm-started chain over the plan's waypoints (keeps one IK branch), then a dense batched solve seeded by
    # linear interpolation of the chain
    knots = [0.0] + [seg[0] for seg in R06_PLAN] + [ts[-1]]
    kq, q_prev = [], seed
    for tk in knots:
        i = int(np.argmin(np.abs(ts - tk)))
        q_prev = solve(tcp[i:i + 1], q_prev[None, :], 200)[0]
        kq.append(q_prev)
    kq = np.array(kq)
    q_init = np.stack([np.interp(ts, knots, kq[:, j]) for j in range(9)], axis=1)
    q = solve(tcp, q_init, 60)
    q[:, 7] = q[:, 8] = fing
    qd = np.gradient(q, 1.0 / R06_RATE, axis=0)  # joint velocity targets (central differences)
    # feed-forward gravity torque along the plan (arm joints only; the cube load is not included)
    state = model.state()
    gf = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    tau = np.zeros_like(q)
    for i in range(len(ts)):
        state.joint_q.assign(q[i].astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        newton.eval_inverse_dynamics_passive(model, state, gravity_force=gf)
        tau[i, :7] = gf.numpy()[:7]
    # IK residual vs the plan
    res = np.zeros(len(ts))
    for i in range(0, len(ts), 10):
        state.joint_q.assign(q[i].astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        x = state.body_q.numpy()[hand]
        res[i] = np.linalg.norm(x[:3] + M.quat_rotate(x[3:7], np.array([0.0, 0.0, R06_TCP])) - tcp[i])
    return {"ts": ts, "tcp": tcp, "q": q, "qd": qd, "tau": tau, "hand": hand, "ik_residual_mm": float(res.max() * 1e3)}


@wp.kernel
def _r06_control(ctr: wp.array[wp.int32], dt: float, rate: float, q_traj: wp.array2d[float],
                 qd_traj: wp.array2d[float], tau_traj: wp.array2d[float], tgt_stride: int, f_stride: int,
                 joint_target_q: wp.array[float], joint_target_qd: wp.array[float], joint_f: wp.array[float]):
    w, j = wp.tid()
    x = float(ctr[0]) * dt * rate
    n = q_traj.shape[0]
    k0 = wp.clamp(int(x), 0, n - 1)
    k1 = wp.min(k0 + 1, n - 1)
    a = wp.clamp(x - float(k0), 0.0, 1.0)
    joint_target_q[w * tgt_stride + j] = (1.0 - a) * q_traj[k0, j] + a * q_traj[k1, j]
    joint_target_qd[w * f_stride + j] = (1.0 - a) * qd_traj[k0, j] + a * qd_traj[k1, j]
    joint_f[w * f_stride + j] = (1.0 - a) * tau_traj[k0, j] + a * tau_traj[k1, j]


@wp.kernel
def _r06_track(body_q: wp.array[wp.transform], ctr: wp.array[wp.int32], dt: float, rate: float, bpw: int,
               hand: int, cube: int, tcp_off: float, t_ref: float, t_end: float, t_motion: float,
               tcp_traj: wp.array[wp.vec3], ref_set: wp.array[wp.int32], rel0: wp.array[wp.vec3],
               slip: wp.array[float], ee_err: wp.array[float]):
    w = wp.tid()
    t = float(ctr[0] + 1) * dt
    xh = body_q[w * bpw + hand]
    tcp = wp.transform_point(xh, wp.vec3(0.0, 0.0, tcp_off))
    xc = wp.transform_get_translation(body_q[w * bpw + cube])
    rel = wp.quat_rotate_inv(wp.transform_get_rotation(xh), xc - tcp)
    if t >= t_ref and ref_set[w] == 0:
        rel0[w] = rel
        ref_set[w] = 1
    if ref_set[w] == 1 and t <= t_end:
        slip[w] = wp.max(slip[w], wp.length(rel - rel0[w]))
    if t <= t_motion:
        n = tcp_traj.shape[0]
        x = t * rate
        k0 = wp.clamp(int(x), 0, n - 1)
        k1 = wp.min(k0 + 1, n - 1)
        a = wp.clamp(x - float(k0), 0.0, 1.0)
        ee_err[w] = wp.max(ee_err[w], wp.length(tcp - ((1.0 - a) * tcp_traj[k0] + a * tcp_traj[k1])))


class FrankaPickPlace(RigidScenario):
    id = "R06_franka_pick_place"
    title = "Franka picks a 5 cm cube, lifts 10 cm, carries 30 cm and places it (IK trajectory replayed on device)"
    horizon_s = 6.5
    accuracy_unit = "mm"
    accuracy_ref = ("place error |cube_xy - place_xy| (<= 2 mm); Coulomb hold 2 mu F_n >= m g => in-hand slip <= 0.5 mm "
                    "over lift+carry; success 100 % (newton/examples/ik/example_ik_cube_stacking.py test_final)")
    _mj = {"solver": "newton", "integrator": "implicitfast", "cone": "elliptic", "impratio": 1000.0, "iterations": 20,
           "ls_iterations": 100, "nconmax": 64, "njmax": 256}
    configs = (
        Config("mujoco_nc", "mujoco", dt=DT600, params={**_mj, "use_mujoco_contacts": False}),
        Config("mujoco_mjc", "mujoco", dt=DT600, params={**_mj, "use_mujoco_contacts": True}),
        # penalty contact on a 50 g cube: kd, kf <= ~0.5 m / (4 h) ~ 37 for the explicit substep (kd 500 diverges),
        # so friction is viscous below mu F_n / kf ~ 6 cm/s and the hold is expected to slip
        Config("featherstone_s10", "featherstone", dt=DT600, substeps=10,
               params={"shape_ke": 5.0e4, "shape_kd": 20.0, "shape_kf": 30.0}),
        Config("kamino_dvi", "kamino", dt=DT600, params={"dynamics_solver": "dvi"}),
        Config("xpbd", "xpbd", dt=DT600, params={"iterations": 10}),
        ref_mujoco_cpu(DT600, {**_mj, "use_mujoco_contacts": True}),  # CPU MuJoCo runs its own narrowphase
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        traj = r06_trajectory(device)
        gap = contact_gap(config, 0.005)
        mat = newton.ModelBuilder.ShapeConfig(mu=R06_MU, ke=5.0e4, kd=5.0e2, kf=1.0e3, gap=gap, density=R06_DENSITY)
        scene = _panda_builder(config)
        for s in range(scene.shape_count):
            scene.shape_material_mu[s] = R06_MU
            scene.shape_gap[s] = gap
        robot_bodies = scene.body_count
        scene.add_ground_plane(cfg=mat)
        h = 0.5 * R06_CUBE
        cube = scene.add_body(xform=wp.transform(p=wp.vec3(R06_PICK[0], R06_PICK[1], h), q=wp.quat_identity()),
                              label="cube")
        scene.add_shape_box(cube, hx=h, hy=h, hz=h, cfg=mat)
        q0 = traj["q"][0]
        for i in range(9):
            scene.joint_q[i] = float(q0[i])
            scene.joint_target_q[i] = float(q0[i])
        model = replicate(scene, nworld, config, device)
        apply_material_overrides(model, config)
        solver = make_solver(config, model, 64 * nworld)
        bpw = model.body_count // nworld
        b = assemble(model, config, solver, nworld, 64 * nworld)
        mass = float(model.body_mass.numpy()[robot_bodies])
        b.extra.update(traj=traj, bpw=bpw, hand=traj["hand"], cube=robot_bodies, mass=mass,
                       q_traj=wp.array(traj["q"].astype(np.float32), dtype=float, device=device),
                       qd_traj=wp.array(traj["qd"].astype(np.float32), dtype=float, device=device),
                       tau_traj=wp.array(traj["tau"].astype(np.float32), dtype=float, device=device),
                       tcp_traj=wp.array(traj["tcp"].astype(np.float32), dtype=wp.vec3, device=device),
                       tgt_stride=b.control.joint_target_q.shape[0] // nworld,
                       f_stride=b.control.joint_f.shape[0] // nworld,
                       ref_set=wp.zeros(nworld, dtype=wp.int32, device=device),
                       rel0=wp.zeros(nworld, dtype=wp.vec3, device=device),
                       slip=wp.zeros(nworld, dtype=float, device=device),
                       ee_err=wp.zeros(nworld, dtype=float, device=device), jerr=[],
                       ik_q=wp.zeros_like(model.joint_q), ik_qd=wp.zeros_like(model.joint_qd))
        return b

    def pre_step(self, b: Built, config: Config) -> None:
        x = b.extra
        wp.launch(_r06_control, dim=(b.nworld, 9), device=b.model.device,
                  inputs=[x["ctr"], config.dt, R06_RATE, x["q_traj"], x["qd_traj"], x["tau_traj"], x["tgt_stride"],
                          x["f_stride"], b.control.joint_target_q, b.control.joint_target_qd, b.control.joint_f])

    def post_step(self, b: Built, config: Config) -> None:
        x = b.extra
        wp.launch(_r06_track, dim=b.nworld, device=b.model.device,
                  inputs=[b.state_0.body_q, x["ctr"], config.dt, R06_RATE, x["bpw"], x["hand"], x["cube"], R06_TCP,
                          R06_SLIP_WINDOW[0], R06_SLIP_WINDOW[1], R06_MOTION_END, x["tcp_traj"], x["ref_set"],
                          x["rel0"], x["slip"], x["ee_err"]])

    def sample(self, b: Built, t: float, rec: dict) -> None:
        x = b.extra
        if t > R06_MOTION_END:
            return
        newton.eval_ik(b.model, b.state_0, x["ik_q"], x["ik_qd"])
        q = x["ik_q"].numpy().reshape(b.nworld, -1)[:, :7]
        tgt = b.control.joint_target_q.numpy().reshape(b.nworld, -1)[:, :7]
        x["jerr"].append(float(np.degrees(np.max(np.abs(q - tgt)))))

    def evaluate(self, b: Built, rec: dict) -> None:
        x, n = b.extra, b.nworld
        q = b.state_0.body_q.numpy().reshape(n, x["bpw"], 7)[:, x["cube"]].astype(np.float64)
        h = 0.5 * R06_CUBE
        err_xy = np.linalg.norm(q[:, :2] - np.array(R06_PLACE), axis=1) * 1e3
        err_z = np.abs(q[:, 2] - h) * 1e3
        tilt = np.degrees(2.0 * np.arccos(np.abs(q[:, 6]).clip(0.0, 1.0)))
        success = (err_xy < 20.0) & (err_z < 10.0) & (tilt < 5.0)
        slip = x["slip"].numpy() * 1e3
        lifted = x["ref_set"].numpy() == 1
        complete = float(rec["horizon_s"]) >= R06_PLAN[-1][0]
        fn = R06_KE[7] * h  # finger drive force when closed on the cube (target 0, finger at the half width)
        hold = 2.0 * R06_MU * fn / (x["mass"] * G)
        rec["accuracy_error"] = float(err_xy.max()) if complete else None
        rec["metrics"].update(success_rate=_f(success.mean(), 4) if complete else None, place_err_mm=_f(err_xy.max(), 3),
                              place_err_z_mm=_f(err_z.max(), 3), cube_tilt_deg=_f(tilt.max(), 3),
                              slip_mm=_f(slip[lifted].max(), 4) if lifted.any() else None,
                              ee_track_err_mm=_f(x["ee_err"].numpy().max() * 1e3, 3),
                              joint_track_err_deg=_f(max(x["jerr"]), 4) if x["jerr"] else None,
                              hold_margin=_f(hold, 3), ik_residual_mm=_f(x["traj"]["ik_residual_mm"], 4),
                              plan_complete=complete,
                              law_pass=bool(complete and success.all() and err_xy.max() <= 2.0
                                            and lifted.any() and slip[lifted].max() <= 0.5))
        if not complete:
            rec["notes"] += f"horizon {rec['horizon_s']} s ends before the plan ({R06_PLAN[-1][0]} s); place not scored; "


SCENARIOS = [FreeFallBounce(), Pendulum(), InclineFriction(), BoxStack(), BoxPile(), FrankaPickPlace()]


# =================================================================================================================
# Offline reference: CPU MuJoCo 3.16 on the same bin MJCF (never in the simulation loop)
# =================================================================================================================
def cpu_mujoco_r05_reference(out_dir: Path, seconds: float = 4.0) -> list[dict]:
    """Run the R05 bin MJCF in CPU MuJoCo and append ref:mujoco_cpu rows (same metrics) to R05_box_pile.ref.jsonl."""
    import mujoco

    from bench.harness import versions

    rows = []
    for n in (64, 256):
        xml, half_width = bin_xml(n)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        steps = int(round(seconds / m.opt.timestep))
        t0 = time.perf_counter()
        p_half = None
        for i in range(steps):
            mujoco.mj_step(m, d)
            if i + 1 == steps // 2:
                p_half = d.xpos[1:n + 1].copy()
        wall = time.perf_counter() - t0
        p_end = d.xpos[1:n + 1].copy()
        bm = bin_metrics(p_half, p_end, seconds / 2, half_width)
        pen = float(max(0.0, -np.min(d.contact.dist[: d.ncon]))) * 1e3 if d.ncon else 0.0
        ke = 0.5 * float(np.sum(m.body_mass[1:n + 1] * np.sum(d.cvel[1:n + 1, 3:] ** 2, axis=1)))
        rows.append({"schema": "alloy.bench.run/1", "scenario": "R05_box_pile", "family": "rigid",
                     "config": f"ref_mujoco_cpu_n{n}", "solver": "ref:mujoco_cpu", "params": {"n": n},
                     "device": "cpu", "nworld": 1, "dt": m.opt.timestep, "substeps": 1, "horizon_s": seconds,
                     "steps": steps, "status": "ok", "ms_per_step": wall / steps * 1e3,
                     "single_world_realtime_factor": m.opt.timestep / (wall / steps), "stable": True,
                     "complete": True, "accuracy_error": bm["creep_mm_per_10s"], "accuracy_unit": "mm_per_10s",
                     "metrics": {"n_bodies": n, "escaped": bm["escaped"], "rest_pen_mm": pen,
                                 "creep_mm": bm["creep_mm_per_10s"], "creep_p90_mm": bm["creep_mm_per_10s_p90"],
                                 "pile_height_mm": bm["pile_top_m"] * 1e3 if bm["pile_top_m"] else None,
                                 "ke_end_j": ke},
                     "versions": versions(), "command": "python -m bench.scenarios.rigid ref-r05",
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    path = out_dir / "rigid" / "R05_box_pile.ref.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    if sys.argv[1:2] == ["ref-r05"]:
        from bench.harness import REPO

        for row in cpu_mujoco_r05_reference(REPO / "bench" / "results"):
            print(json.dumps({k: row[k] for k in ("config", "ms_per_step", "accuracy_error")} | row["metrics"]))
    else:
        print("usage: python -m bench.scenarios.rigid ref-r05")
