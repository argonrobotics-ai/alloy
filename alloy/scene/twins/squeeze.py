# SPDX-License-Identifier: Apache-2.0
"""R5 squeeze-stiffness twin (bench §4.4 R5): F at 10 % and 20 % strain (10 %), initial slope (10 %), hysteresis (± 0.05).

Rig: a soft block between plates at 1 mm/s to 30 % strain, 3 triangle cycles, the first dropped.

Twin: one VBD tet block per world (Kuhn box tets from WP2's `box_tets`, Lamé values from `derive.lame`, tet damping
k_damp = β·μ as `emit_soft` writes it). The plates are the block's top and bottom node layers, made heavy (100 kg
each) so they follow a velocity set every step: VBD integrates a particle by its velocity before iterating
(newton@ac7f7830:newton/_src/solvers/vbd/particle_vbd_kernels.py:1425-1437), so the plate nodes reach their target
inside the step and VBD's Green-strain-rate damping sees the plate motion (a zero-mass node would be pinned at its
step-start position, :1426, and its motion would bypass the damping, which differences against step-start
positions, :278-313). Plates are frictionless: their nodes follow the block's mean lateral stretch, measured on the
free nodes each step, so the block deforms homogeneously as in the analytic law. The reaction is the internal
force VBD itself evaluates on the top nodes (`evaluate_volumetric_neo_hookean_force_and_hessian`, :179-326, damping
included); the bottom reaction is recorded as an equilibrium check.

Speed-up: the plates run SPEEDUP x faster than the rig and the damping coefficient is divided by SPEEDUP, so the
viscous stress (∝ k_damp·strain rate) is the rig's at each strain. It stays quasi-static: the slowest axial mode of
the 50 mm foam cube has a period 2h/√(E/ρ) ≈ 3 ms against a 0.15 s half-cycle, and the measured top/bottom reaction
imbalance is reported in `monitors["imbalance"]`.

Synthetic truth (`dataset`): the closed form of the same constitutive law in homogeneous uniaxial compression:
VBD's stable neo-Hookean (μ, λ_NH = λ + μ, α = 1 + μ/λ_NH; :218-245) with P_visc = 2η·F·Ċ (η = k_damp, the
continuum form of the damping block, :278-313), lateral stretch from zero lateral traction, integrated at the rig
rate. Calibration (`bind_calib`, calib trials only): β from the mean calib hysteresis ratio (root of the closed
form), then E from the mean calib F10 (force is linear in E at fixed ν and β). Poisson's ratio comes from the
condition (a measured card value). The Shore 20A ball of the protocol is not covered: it needs a sphere tet mesh in
plate contact (Hertz, no displacement boundary) and ν ≈ 0.49 rubber, where linear tets lock.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
from alloy._vendor.newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian
from scipy.optimize import brentq

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigDataset, RigTrial, Tolerance
from alloy.core.errors import LawError
from alloy.scene import derive
from alloy.scene.emitters.deformable import box_tets
from alloy.testing.fakes import synthetic_rig

log = logging.getLogger("alloy.scene")

SPEEDUP = 100.0          # twin plate speed / rig plate speed
PLATE_MASS = 100.0       # [kg] per plate node: inertia-dominated, follows its set velocity to < 1e-8 m per step
DT = 1.0e-3              # [s]
ITERATIONS = 20          # VBD iterations per step
SLOPE_WINDOW = (0.01, 0.05)   # strain range of the initial-slope fit on the loading branch
METRICS = ("force_10", "force_20", "slope", "hysteresis")
_SRC = "bench §4.4 R5"


@dataclasses.dataclass(frozen=True)
class SqueezeSpec:
    """One specimen and protocol (SI): block lx × ly × h, plates at `speed` to `strain_max`, `cycles` cycles."""

    lx: float
    ly: float
    h: float
    poisson: float
    density: float
    strain_max: float = 0.30
    speed: float = 1.0e-3
    cycles: int = 3

    @staticmethod
    def of(cond: Condition) -> "SqueezeSpec":
        p = cond.params
        return SqueezeSpec(p["lx"], p["ly"], p["h"], p["poisson"], p["density"], p["strain_max"], p["speed"],
                           int(p["cycles"]))

    @property
    def half_period(self) -> float:
        """Rig time [s] of one loading (or unloading) ramp."""
        return self.strain_max * self.h / self.speed

    @property
    def area(self) -> float:
        return self.lx * self.ly


def twin_duration(specs: Sequence[SqueezeSpec]) -> float:
    return max(2 * s.cycles * s.half_period for s in specs) / SPEEDUP


CONDITIONS = (
    Condition("pu_foam_cube", {"lx": 0.05, "ly": 0.05, "h": 0.05, "poisson": 0.05, "density": 30.0,
                               "strain_max": 0.30, "speed": 1.0e-3, "cycles": 3}),
    Condition("sponge", {"lx": 0.06, "ly": 0.04, "h": 0.03, "poisson": 0.10, "density": 25.0,
                         "strain_max": 0.30, "speed": 1.0e-3, "cycles": 3}),
)
LAW = LawSpec(
    id="R5.squeeze_stiffness", suite="rig",
    metrics={"force_10": Tolerance(rel=0.10, source=f"{_SRC}: F at 10 % strain within 10 %"),
             "force_20": Tolerance(rel=0.10, source=f"{_SRC}: F at 20 % strain within 10 %"),
             "slope": Tolerance(rel=0.10, source=f"{_SRC}: initial slope within 10 %"),
             "hysteresis": Tolerance(abs=0.05, source=f"{_SRC}: hysteresis ratio +/- 0.05")},
    stat=dict.fromkeys(METRICS, "mean"), dt=DT, substeps=1,
    duration_s=twin_duration([SqueezeSpec.of(c) for c in CONDITIONS]), worlds_per_condition=1, solvers=("vbd",),
    producer="WP2", quick={"worlds": 2, "duration_s": twin_duration([SqueezeSpec.of(c) for c in CONDITIONS])},
)


# --- protocol and metrics (shared by the twin and the synthetic truth) --------------------------------------------

def strain_at(t_rig: np.ndarray, spec: SqueezeSpec) -> np.ndarray:
    """Prescribed plate strain of the triangle protocol at rig time `t_rig` (0 after the last cycle)."""
    th = spec.half_period
    phase = np.mod(t_rig, 2.0 * th) / th
    eps = spec.strain_max * np.where(phase < 1.0, phase, 2.0 - phase)
    return np.where(t_rig < 2 * spec.cycles * th, eps, 0.0)


def _branch(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo - 1e-9) & (t <= hi + 1e-9)


def _cycle_metrics(eps_l: np.ndarray, f_l: np.ndarray, eps_u: np.ndarray, f_u: np.ndarray, h: float
                   ) -> dict[str, float]:
    order = np.argsort(eps_l, kind="stable")
    e_l, fl = eps_l[order], f_l[order]
    win = (e_l >= SLOPE_WINDOW[0]) & (e_l <= SLOPE_WINDOW[1])
    slope = float(np.polyfit(e_l[win] * h, fl[win], 1)[0]) if win.sum() >= 2 else math.nan
    w_load = float(np.trapezoid(f_l, eps_l * h))
    w_unload = -float(np.trapezoid(f_u, eps_u * h))
    return {"force_10": float(np.interp(0.10, e_l, fl)), "force_20": float(np.interp(0.20, e_l, fl)),
            "slope": slope, "hysteresis": (w_load - w_unload) / w_load if w_load > 0 else math.nan}


def squeeze_metrics(t_rig: np.ndarray, eps: np.ndarray, force: np.ndarray, spec: SqueezeSpec) -> dict[str, float]:
    """Mean over cycles 2..n (the first is dropped) of F10, F20, initial slope [N/m] and hysteresis ratio
    (W_load − W_unload)/W_load, W = ∫F dδ, from compressive force `force` [N] vs strain `eps` at rig time `t_rig`."""
    th = spec.half_period
    rows = []
    for k in range(1, spec.cycles):
        load, unload = _branch(t_rig, 2 * k * th, (2 * k + 1) * th), _branch(t_rig, (2 * k + 1) * th, (2 * k + 2) * th)
        if load.sum() < 4 or unload.sum() < 4:
            raise LawError(f"R5: cycle {k + 1} has too few samples ({load.sum()}, {unload.sum()})")
        rows.append(_cycle_metrics(eps[load], force[load], eps[unload], force[unload], spec.h))
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


# --- closed form: homogeneous uniaxial compression of VBD's damped stable neo-Hookean --------------------------------

def _lateral_stretch(a0: float, lz: float, mu: float, lnh: float, alpha: float, eta_dt: float, a_prev: float) -> float:
    """Newton on zero lateral traction μa + λ_NH(a²λz − α)aλz + 2η a (a² − a_prev²)/Δt = 0 (η_dt = η/Δt)."""
    a = a0
    for _ in range(30):
        g = mu * a + lnh * (a * a * lz - alpha) * a * lz + 2.0 * eta_dt * a * (a * a - a_prev * a_prev)
        dg = mu + lnh * lz * (3.0 * a * a * lz - alpha) + 2.0 * eta_dt * (3.0 * a * a - a_prev * a_prev)
        step = g / dg
        a -= step
        if abs(step) < 1e-13:
            break
    return a


def homogeneous_response(spec: SqueezeSpec, youngs: float, beta: float, steps_per_half: int = 300
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, strain, compressive force [N]) of the protocol at the rig rate, homogeneous deformation (frictionless
    plates), P_zz = μλz + λ_NH(J − α)λt² + 2η λz Ċzz with η = β·μ, backward-difference rates as VBD takes them."""
    mu, lam = derive.lame(youngs, spec.poisson)
    lnh = lam + mu
    alpha = 1.0 + mu / lnh
    dt = spec.half_period / steps_per_half
    t = np.arange(2 * spec.cycles * steps_per_half + 1) * dt
    eps = strain_at(t, spec)
    eta_dt = beta * mu / dt
    force = np.zeros_like(t)
    a_prev, lz_prev = 1.0, 1.0
    for n in range(1, t.size):
        lz = 1.0 - eps[n]
        a = _lateral_stretch(a_prev, lz, mu, lnh, alpha, eta_dt, a_prev)
        pzz = mu * lz + lnh * (a * a * lz - alpha) * a * a + 2.0 * eta_dt * lz * (lz * lz - lz_prev * lz_prev)
        force[n] = -pzz * spec.area
        a_prev, lz_prev = a, lz
    return t, eps, force


def law_metrics(spec: SqueezeSpec, youngs: float, beta: float) -> dict[str, float]:
    """R5 scalars of the closed-form law for one specimen."""
    t, eps, force = homogeneous_response(spec, youngs, beta)
    return squeeze_metrics(t, eps, force, spec)


def fit_material(spec: SqueezeSpec, force_10: float, hysteresis: float) -> tuple[float, float]:
    """(E, β) reproducing the calib mean F10 and hysteresis: β by bracketing on the hysteresis ratio (E-free at fixed
    ν), then E by linear scaling of F10."""
    def resid(log_beta: float) -> float:
        return law_metrics(spec, 1.0, math.exp(log_beta))["hysteresis"] - hysteresis

    lo, hi = math.log(1e-4), math.log(1e2)
    if not resid(lo) < 0.0 < resid(hi):
        raise LawError(f"R5: hysteresis {hysteresis:.3f} outside the damped-neo-Hookean range for this specimen")
    beta = math.exp(brentq(resid, lo, hi, xtol=1e-10))
    return force_10 / law_metrics(spec, 1.0, beta)["force_10"], beta


# --- device kernels ---------------------------------------------------------------------------------------------

@wp.func
def _plate_strain(t: float, half: float, emax: float, cycles: float) -> float:
    """Device copy of `strain_at` (twin time)."""
    if t >= 2.0 * cycles * half:
        return 0.0
    phase = wp.mod(t, 2.0 * half) / half
    if phase < 1.0:
        return emax * phase
    return emax * (2.0 - phase)


@wp.kernel
def _drive_plates(q: wp.array(dtype=wp.vec3), rest: wp.array(dtype=wp.vec3), free_idx: wp.array(dtype=wp.int32),
                  free_off: wp.array(dtype=wp.int32), plate_idx: wp.array(dtype=wp.int32),
                  plate_top: wp.array(dtype=wp.int32), plate_off: wp.array(dtype=wp.int32),
                  sched: wp.array2d(dtype=float), t_next: float, dt: float, qd: wp.array(dtype=wp.vec3)):
    """Per world: mean lateral stretch of the free nodes, then plate-node velocities that reach the target in dt.
    `sched[w]` = (h, twin half-period, strain_max, cycles)."""
    w = wp.tid()
    num = float(0.0)
    den = float(0.0)
    for k in range(free_off[w], free_off[w + 1]):
        i = free_idx[k]
        num += q[i][0] * rest[i][0] + q[i][1] * rest[i][1]
        den += rest[i][0] * rest[i][0] + rest[i][1] * rest[i][1]
    s = num / den
    h = sched[w, 0]
    eps = _plate_strain(t_next, sched[w, 1], sched[w, 2], sched[w, 3])
    for k in range(plate_off[w], plate_off[w + 1]):
        i = plate_idx[k]
        z = -0.5 * h
        if plate_top[k] == 1:
            z = 0.5 * h - eps * h
        qd[i] = (wp.vec3(s * rest[i][0], s * rest[i][1], z) - q[i]) / dt


@wp.kernel
def _record_reaction(q: wp.array(dtype=wp.vec3), q_prev: wp.array(dtype=wp.vec3), tets: wp.array2d(dtype=wp.int32),
                     poses: wp.array(dtype=wp.mat33), mats: wp.array2d(dtype=float),
                     inc_tet: wp.array(dtype=wp.int32), inc_ord: wp.array(dtype=wp.int32),
                     inc_top: wp.array(dtype=wp.int32), inc_off: wp.array(dtype=wp.int32),
                     plate_idx: wp.array(dtype=wp.int32), plate_top: wp.array(dtype=wp.int32),
                     plate_off: wp.array(dtype=wp.int32), sched: wp.array2d(dtype=float), dt: float, col: int,
                     out_f: wp.array2d(dtype=float), out_fb: wp.array2d(dtype=float),
                     out_eps: wp.array2d(dtype=float)):
    """Per world: VBD's own tet force (elastic + damping) summed over the plate nodes, and the plate strain."""
    w = wp.tid()
    f_top = float(0.0)
    f_bot = float(0.0)
    for k in range(inc_off[w], inc_off[w + 1]):
        t = inc_tet[k]
        f, h = evaluate_volumetric_neo_hookean_force_and_hessian(t, inc_ord[k], q_prev, q, tets, poses[t],
                                                                 mats[t, 0], mats[t, 1], mats[t, 2], dt)
        if inc_top[k] == 1:
            f_top += f[2]
        else:
            f_bot -= f[2]
    zt = float(0.0)
    zb = float(0.0)
    nt = float(0.0)
    for k in range(plate_off[w], plate_off[w + 1]):
        if plate_top[k] == 1:
            zt += q[plate_idx[k]][2]
            nt += 1.0
        else:
            zb += q[plate_idx[k]][2]
    nb = float(plate_off[w + 1] - plate_off[w]) - nt
    out_f[w, col] = f_top
    out_fb[w, col] = f_bot
    out_eps[w, col] = 1.0 - (zt / nt - zb / nb) / sched[w, 0]


# --- twin --------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class _BlockMesh:
    """Global indices of one world's block: free nodes, plate nodes (+ top flag), plate-node tet incidences."""

    free: np.ndarray
    plate: np.ndarray
    plate_top: np.ndarray
    inc_tet: np.ndarray
    inc_ord: np.ndarray
    inc_top: np.ndarray


def block_mesh(spec: SqueezeSpec, p0: int, t0: int) -> tuple[np.ndarray, np.ndarray, _BlockMesh]:
    """Kuhn box tets (about h/3 per cell, >= 2 cells per axis), plate layers and their tet incidences."""
    full = np.array([spec.lx, spec.ly, spec.h])
    dims = np.maximum(2, np.round(full / (spec.h / 3.0))).astype(int)
    verts, tets = box_tets(full, dims)
    z = verts[:, 2]
    top, bot = np.isclose(z, z.max()), np.isclose(z, z.min())
    plate = np.flatnonzero(top | bot)
    on_plate = top | bot
    t_idx, order = np.nonzero(on_plate[tets])
    mesh = _BlockMesh(free=p0 + np.flatnonzero(~on_plate), plate=p0 + plate, plate_top=top[plate].astype(np.int32),
                      inc_tet=t0 + t_idx, inc_ord=order.astype(np.int32),
                      inc_top=top[tets[t_idx, order]].astype(np.int32))
    return verts, tets, mesh


def _csr(parts: Sequence[np.ndarray], device: Any) -> tuple[wp.array, wp.array]:
    off = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int32)
    flat = np.concatenate(parts).astype(np.int32)
    return wp.array(flat, dtype=wp.int32, device=device), wp.array(off, dtype=wp.int32, device=device)


def _flat(parts: Sequence[np.ndarray], device: Any) -> wp.array:
    return wp.array(np.concatenate(parts).astype(np.int32), dtype=wp.int32, device=device)


class SqueezeTwin(BaseTwin):
    """R5 twin: one VBD block per world, condition params = `SqueezeSpec` fields, material fitted on calib."""

    law_id = LAW.id
    units = {"force_10": "N", "force_20": "N", "slope": "N/m", "hysteresis": "1"}
    needs_contacts = False

    def __init__(self, iterations: int = ITERATIONS) -> None:
        super().__init__()
        self.settings = {"vbd": SolverSettings("vbd", {"iterations": int(iterations)})}
        self.fit: dict[str, tuple[float, float]] = {}
        self._mesh: list[_BlockMesh] = []

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        """(E, β) per condition from the calib means of F10 and hysteresis; geometry and ν from `card.objects`."""
        for cid in dict.fromkeys(t.condition_id for t in calib):
            rows = [t for t in calib if t.condition_id == cid]
            spec = SqueezeSpec.of(Condition(cid, card.objects[cid]))
            f10 = float(np.mean([t.scalars["force_10"] for t in rows]))
            hyst = float(np.mean([t.scalars["hysteresis"] for t in rows]))
            self.fit[cid] = fit_material(spec, f10, hyst)
        self.monitors["fit_calib"] = {c: {"youngs": e, "beta": b} for c, (e, b) in self.fit.items()}
        log.info("R5 calib fit: %s", self.monitors["fit_calib"])

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)  # the law has no self-weight (foam weighs < 0.1 % of F10)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        if cond.id not in self.fit:
            raise LawError(f"R5: no calib fit for condition {cond.id!r}")
        spec = SqueezeSpec.of(cond)
        youngs, beta = self.fit[cond.id]
        k_mu, k_lambda = derive.lame(youngs, spec.poisson)
        verts, tets, mesh = block_mesh(spec, b.particle_count, b.tet_count)
        b.add_soft_mesh(pos=wp.vec3(), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(),
                        vertices=verts.astype(np.float32).tolist(), indices=tets.reshape(-1).tolist(),
                        density=spec.density, k_mu=k_mu, k_lambda=k_lambda, k_damp=beta / SPEEDUP * k_mu,
                        label=f"block_{cond.id}")
        self._mesh.append(mesh)

    def on_model(self, model: newton.Model) -> None:
        dev = model.device
        plates = np.concatenate([m.plate for m in self._mesh])
        mass = model.particle_mass.numpy()
        mass[plates] = PLATE_MASS
        model.particle_mass.assign(mass)
        model.particle_inv_mass.assign(np.where(mass > 0, 1.0 / np.maximum(mass, 1e-30), 0.0).astype(np.float32))
        self._rest = wp.clone(model.particle_q)
        self._prev = wp.clone(model.particle_q)
        self._free, self._free_off = _csr([m.free for m in self._mesh], dev)
        self._plate, self._plate_off = _csr([m.plate for m in self._mesh], dev)
        self._plate_top = _flat([m.plate_top for m in self._mesh], dev)
        self._inc_tet, self._inc_off = _csr([m.inc_tet for m in self._mesh], dev)
        self._inc_ord = _flat([m.inc_ord for m in self._mesh], dev)
        self._inc_top = _flat([m.inc_top for m in self._mesh], dev)
        specs = [SqueezeSpec.of(c) for c in self.worlds]
        sched = [[s.h, s.half_period / SPEEDUP, s.strain_max, float(s.cycles)] for s in specs]
        self._sched = wp.array(np.asarray(sched, np.float32), dtype=float, device=dev)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_drive_plates, dim=len(self.worlds),
                  inputs=[state.particle_q, self._rest, self._free, self._free_off, self._plate, self._plate_top,
                          self._plate_off, self._sched, t + DT, DT], outputs=[state.particle_qd],
                  device=self.model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        col = log_.column(step)
        if col >= 0:
            m = self.model
            wp.launch(_record_reaction, dim=len(self.worlds),
                      inputs=[state.particle_q, self._prev, m.tet_indices, m.tet_poses, m.tet_materials,
                              self._inc_tet, self._inc_ord, self._inc_top, self._inc_off, self._plate,
                              self._plate_top, self._plate_off, self._sched, DT, col],
                      outputs=[log_.channel("force"), log_.channel("force_bottom"), log_.channel("strain")],
                      device=m.device)
        wp.copy(self._prev, state.particle_q)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        t_rig = log_.times() * SPEEDUP
        f, fb, eps = (log_.series(k).astype(np.float64) for k in ("force", "force_bottom", "strain"))
        rows, imbalance = [], []
        for w, cond in enumerate(self.worlds):
            spec = SqueezeSpec.of(cond)
            rows.append(squeeze_metrics(t_rig, eps[w], f[w], spec))
            live = t_rig <= 2 * spec.cycles * spec.half_period
            imbalance.append(float(np.max(np.abs(f[w, live] - fb[w, live])) / np.max(np.abs(f[w, live]))))
        self.monitors["imbalance"] = imbalance
        self.monitors["per_world"] = rows
        return {k: np.array([r[k] for r in rows]) for k in METRICS}


# --- synthetic dataset ---------------------------------------------------------------------------------------------

TRUTH = {"pu_foam_cube": {"youngs": 30.0e3, "beta": 0.5}, "sponge": {"youngs": 20.0e3, "beta": 0.6}}
NOISE_REL = 0.02         # specimen scatter of forces and slope (1 sigma, relative to the smallest specimen)
NOISE_HYST = 0.008       # hysteresis ratio scatter (1 sigma)
TRIALS = 20              # per specimen
INSTRUMENTS = {"force_10": {"u95": 0.05, "what": "5 kg HX711 cell, M1 masses"},
               "force_20": {"u95": 0.05, "what": "5 kg HX711 cell, M1 masses"},
               "slope": {"u95": 10.0, "what": "cell + 0.01 mm stage encoder"},
               "hysteresis": {"u95": 0.005, "what": "loop integral of cell x encoder"}}


def _truth(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    return law_metrics(SqueezeSpec.of(cond), params[f"{cond.id}.youngs"], params[f"{cond.id}.beta"])


def dataset(seed: int = 0) -> RigDataset:
    """TRIALS synthetic trials per specimen (half calib, half valid) from the closed-form law plus Gaussian scatter.

    The rig protocol has n = 5; with 2 % scatter, 5 valid trials give TUR 2-4 (t(4) = 2.78), so the synthetic set
    uses 10 per split to keep every metric judgeable (TUR >= 3) across seeds."""
    params = {f"{c}.{k}": v for c, m in TRUTH.items() for k, v in m.items()}
    nominal = {c.id: _truth(params, c) for c in CONDITIONS}
    noise = {m: NOISE_REL * min(nominal[c][m] for c in nominal) for m in ("force_10", "force_20", "slope")}
    noise["hysteresis"] = NOISE_HYST
    data = synthetic_rig(LAW.id, params, noise, conditions=CONDITIONS, trials_per_condition=TRIALS,
                         calib_per_condition=TRIALS // 2, seed=seed, model=_truth)
    card = dataclasses.replace(data.card, dataset_id="synthetic-v1", protocol_version="synthetic/1",
                               instruments=dict(INSTRUMENTS), objects={c.id: dict(c.params) for c in CONDITIONS})
    return dataclasses.replace(data, card=card)


__all__ = ["LAW", "SqueezeTwin", "SqueezeSpec", "dataset", "law_metrics", "fit_material", "homogeneous_response",
           "squeeze_metrics"]
