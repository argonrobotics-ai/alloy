# SPDX-License-Identifier: Apache-2.0
"""Law A9 (thread advance = p and run-down torque, SPEC §5.3 A9 amendment) and its MuJoCo twin; the shared M10 scene.

Scene per world (gravity off): a fixed M10x1.5 6g bolt (a d2 cylinder on the world body for MuJoCo; its thread is the
analytic partner), a DIN 934 M10 6H nut on a D6 joint (axial slide + spin about the bolt axis; lateral motion and
tilt locked, as by a socket drive) spun by a MuJoCo velocity actuator (kd 10 N·m·s/rad, implicitfast, target ramped
over 0.1 s) at ω, and a constant axial load F on the slide opposing the advance (the rig's 5 N spring; its rate over
a few mm is ignored). The bolt/nut pair is filtered from native collision and runs through ``analytic_pair`` (bolt
thread SDF vs nut flank samples) appended after ``AlloyContactLayer.collide`` on the ``native="newton"`` path (a
``native="mujoco"`` layer without precision pairs has no room for them yet). Friction: MuJoCo mixes geom μ by max,
so the nut geom has μ 1e-4 and the bolt μ_thread. Drive torque = kd·(ω_target - ω) after the step (the implicit
velocity actuator's torque; it matched the contact torque about the axis to 0.1 % on CPU). R2 seating uses a
modelled washer in the drive kernel (``ThreadRig.seat_gap``/``k_clamp``, see ``thread_rig``).

MuJoCo choices, each from a measured failure on CPU (μ 0.14):
- every thread row gets the pair K (``split_k=False``) with K set so its MuJoCo timeconst is 4·dt: K split over the
  ~60 rows of a flank made each row ~8x slower and the drive screwed the nut up to 90 µm into the flank;
- rows only within 2 µm (``PAIR_GAP``) and no Newton speculative rows: separated rows are frictionless speculative
  constraints there, and the nut then rests on them at +0.3 µm without friction (torque = lift only);
- nut samples on the middle 20 % of each flank: MuJoCo puts the load of several equal rows on one of them, so all
  rows sit near the mid-flank radius (sim torque radius 4.59 mm vs the formula's d2/2 = 4.51 mm: +1.3 %);
- spin armature 0.01 kg·m² (drive train): with 1e-4 the elliptic-cone friction under a kN clamp load drove the nut
  forward instead of resisting it (seating slope -78 %; -22 % at 3e-3, +2 % at 1e-2).
MuJoCo drops a contact row as soon as its distance is positive, so a stiff resting contact flickers; the remaining
chatter averages out over a turn (contact Fz 5.00 N, friction/normal 0.140). Axial armature 1 kg (A9; R2 30 kg for
its explicit washer spring); the run is quasi-static (0.75 mm/s). Nut mass 11.7 g: hex 17 mm AF x 8 mm, bore at D2,
steel 7850 kg/m³ (meshgen nut_m10_din934); spin inertia of the equal-area ring.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.bench import solvers as SV
from alloy.bench.analytic.common import BaseTwin, smoothstep
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.contact.analytic_pair import (AnalyticPair, ThreadFeature, analytic_collide, attach_analytic_pairs,
                                         torque_radius)
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.precision import pair_stiffness, size_l2
from alloy.contact.thread_sdf import parse_thread_id
from alloy.contact.types import PrecisionPairSpec
from alloy.core.log import get_logger

log = get_logger("contact.thread_twins")

BOLT_ID, NUT_ID = "iso_thread:M10x1.5:6g", "iso_thread:M10x1.5:6H"
# M10x1.5, the reference's own numbers: pitch (ISO 261) and basic pitch diameter d - 3/4 H, H = sqrt(3)/2 p (ISO 68-1;
# 9.026 mm in ISO 724's table)
A9_PITCH_M = 1.5e-3
A9_D2_M = 10.0e-3 - 0.75 * (math.sqrt(3.0) / 2.0) * A9_PITCH_M
BOLT_Z = (-0.025, 0.025)            # bolt thread [m]; the bolt shape frame is the world frame (thread phase 0 at z 0)
NUT_HALF = 4.0e-3                   # DIN 934 M10 height 8 mm
NUT_OUTER = 8.5e-3                  # inscribed radius of the 17 mm hex
BEARING_R_IN = 5.25e-3              # DIN 125-A M10 washer bore 10.5 mm
STOP_ZETA = 0.5                     # damping ratio of the modelled washer stop
OMEGA_SMOOTH = 0.05                 # rad/s: bearing friction ramps in as tanh(ω/OMEGA_SMOOTH)
HEX_AF, STEEL = 17e-3, 7850.0
ARM_AXIAL, ARM_SPIN, KD_SPIN = 1.0, 1.0e-2, 10.0
ROW_TAU_STEPS = 4.0                 # each thread row realises MuJoCo timeconst 4·dt (K_real = K, D22 clamp is 2·dt)
NUT_MU = 1.0e-4                     # below every partner μ: MuJoCo's max-mix then uses the partner's (> MJ_MINMU)
SAMPLE_SPACING, PAIR_GAP = 5.0e-5, 2.0e-6
CONTROL_DT, CONTACT_SPEED = 0.02, 0.01
SETTLE_S = 0.25                     # load and spin ramp (SPIN_RAMP_S) settle before measuring
SPIN_RAMP_S = 0.1
SPLIT_K = False                     # every thread row gets the pair K (module docstring)
TWIN_SECTORS = 12
NUT_FLANK_BAND = 0.2                # nut samples on the middle 20 % of each flank (module docstring)
ROWS_PER_WORLD = 256                # >= 168 thread + 12 face bins
T_SEAT_ON = 0.05                    # N·m: seating starts above max(this, 5x the run-down torque)
SEAT_FIT = (1.0, 8.0)               # N·m window of the seating-slope fit
THREAD_MUJOCO = SV.SolverSettings(
    "mujoco", {"cone": "elliptic", "impratio": 300.0, "iterations": 100, "ls_iterations": 50,
               "integrator": "implicitfast", "nconmax": ROWS_PER_WORLD, "njmax": 4 * ROWS_PER_WORLD,
               "use_speculative_contacts": False},
    mujoco_solimp=(0.99, 0.99, 0.001, 0.5, 2.0), newton_contacts=True)


def bearing_radius() -> float:
    """Uniform-pressure friction radius of the nut face on the washer (5.25..8.5 mm)."""
    return torque_radius(BEARING_R_IN, NUT_OUTER)


def nut_mass() -> float:
    d2 = parse_thread_id(NUT_ID).d2
    return (0.5 * math.sqrt(3.0) * HEX_AF**2 - math.pi * (0.5 * d2) ** 2) * 2.0 * NUT_HALF * STEEL


def mujoco_m_eff(armature: float = ARM_AXIAL) -> float:
    """1/body_invweight0 of the nut: MuJoCo averages the translational inverse mass over x, y, z (slide on z only)."""
    return 3.0 * (nut_mass() + armature)


def thread_spec(shape_a: int, shape_b: int, k: float | None = None, design_force: float = 5.0
                ) -> PrecisionPairSpec:
    """Bolt (partner) / nut (sampled) pair with the 6g/6H diametral clearance (max_penetration 7.9 µm, D2); K is
    ``k`` or 2·design_force/max_penetration (D22)."""
    clearance = parse_thread_id(NUT_ID).deviation - parse_thread_id(BOLT_ID).deviation
    spec = PrecisionPairSpec(shape_a, shape_b, mode="sdf", voxel_size=SAMPLE_SPACING, gap=PAIR_GAP, ccd="off",
                             clearance=clearance)
    return dataclasses.replace(spec, pair_stiffness=k if k is not None else pair_stiffness(spec, design_force))


def thread_dt(design_force: float, armature: float = ARM_AXIAL) -> float:
    """§3.5 L2 step for the thread pair at ``design_force`` (D4 grid under 20 ms)."""
    return size_l2(thread_spec(0, 1, design_force=design_force), m_eff=mujoco_m_eff(armature),
                   design_force=design_force, contact_speed=CONTACT_SPEED, control_dt=CONTROL_DT,
                   label="M10 thread").dt


def row_stiffness(dt: float, armature: float = ARM_AXIAL, imp: float = 0.99) -> float:
    """Per-row K whose MuJoCo timeconst is ROW_TAU_STEPS·dt: m_eff/((1-imp)·τ²) (precision.k_real_mujoco)."""
    return mujoco_m_eff(armature) / ((1.0 - imp) * (ROW_TAU_STEPS * dt) ** 2)


@dataclass(frozen=True)
class ThreadRig:
    """One world's settings [SI]. The washer is a modelled stop: once the nut has advanced ``seat_gap`` its face meets
    a spring ``k_clamp`` (the rig's bolt + joint stiffness) with bearing friction ``mu_bearing`` at r_b."""

    mu_thread: float = 0.14
    mu_bearing: float = 0.14
    axial_load: float = 5.0
    omega: float = math.pi
    seat_gap: float = math.inf
    k_clamp: float = 0.0
    axial_armature: float = ARM_AXIAL
    torque_stop: float = math.inf


def build_thread_world(b: newton.ModelBuilder, rig: ThreadRig) -> int:
    """Add bolt (template shape 0) and nut (1); returns the nut's D6 joint."""
    bolt = b.add_shape_cylinder(-1, radius=0.5 * parse_thread_id(BOLT_ID).d2, half_height=BOLT_Z[1],
                                cfg=newton.ModelBuilder.ShapeConfig(mu=rig.mu_thread, density=0.0), label="bolt")
    m = nut_mass()
    r_eq2 = 0.5 * math.sqrt(3.0) * HEX_AF**2 / math.pi
    i_z = 0.5 * m * (r_eq2 + (0.5 * parse_thread_id(NUT_ID).d2) ** 2)
    body = b.add_link(mass=m, inertia=wp.mat33(np.diag([i_z, i_z, i_z])), label="nut")
    nut = b.add_shape_cylinder(body, radius=NUT_OUTER, half_height=NUT_HALF,
                               cfg=newton.ModelBuilder.ShapeConfig(mu=NUT_MU, density=0.0), label="nut")
    dof = newton.ModelBuilder.JointDofConfig
    joint = b.add_joint_d6(-1, body, linear_axes=[dof(axis=(0, 0, 1), armature=rig.axial_armature)],
                           angular_axes=[dof(axis=(0, 0, 1), target_vel=rig.omega, target_kd=KD_SPIN,
                                             armature=ARM_SPIN)], label="nut_drive")
    b.add_articulation([joint])
    b.add_shape_collision_filter_pair(bolt, nut)
    return joint


def thread_pairs(k_row: float) -> list[AnalyticPair]:
    """The bolt (partner, template shape 0) / nut (sampled, 1) pair, every row ``k_row``."""
    return [AnalyticPair(thread_spec(0, 1, k_row), ThreadFeature(BOLT_ID, *BOLT_Z),
                         ThreadFeature(NUT_ID, -NUT_HALF, NUT_HALF, NUT_OUTER, flank_band=NUT_FLANK_BAND),
                         split_k=SPLIT_K)]


class ThreadPipeline:
    """``run_law`` contact source: ``layer.collide`` (with dt) then the analytic rows."""

    def __init__(self, model: newton.Model, pairs: list[AnalyticPair], dt: float) -> None:
        self.dt = dt
        self.layer = AlloyContactLayer(model, [], native="newton")
        self.rows = attach_analytic_pairs(self.layer, pairs, sectors=TWIN_SECTORS)

    def contacts(self) -> newton.Contacts:
        return self.layer.contacts()

    def collide(self, state: newton.State, contacts: newton.Contacts) -> None:
        self.layer.collide(state, contacts, dt=self.dt)
        analytic_collide(self.layer, state, contacts, self.dt)


@wp.struct
class DriveIn:
    """Per-world drive state (indices into joint_q / joint_qd, loads, the modelled washer)."""

    q_slide: wp.array(dtype=wp.int32)
    qd_spin: wp.array(dtype=wp.int32)
    load: wp.array(dtype=wp.float32)
    omega: wp.array(dtype=wp.float32)
    stop: wp.array(dtype=wp.float32)
    seat: wp.array(dtype=wp.float32)
    k_clamp: wp.array(dtype=wp.float32)
    c_clamp: wp.array(dtype=wp.float32)
    bearing: wp.array(dtype=wp.float32)          # μ_b·r_b [m]
    latched: wp.array(dtype=wp.int32)
    target: wp.array(dtype=wp.float32)


@wp.func
def _clamp_force(d: DriveIn, w: int, z: float, vz: float) -> float:
    """Washer force on the nut (>= 0, along -z) once it has advanced past the seat."""
    pen = z - d.seat[w]
    if pen <= 0.0:
        return 0.0
    return wp.max(d.k_clamp[w] * pen + d.c_clamp[w] * vz, 0.0)


@wp.kernel
def _drive(d: DriveIn, joint_q: wp.array[float], joint_qd: wp.array[float], kd: float, ramp: float,
           joint_f: wp.array[float], target_qd: wp.array[float]):
    w = wp.tid()
    i = d.qd_spin[w]
    omega = float(joint_qd[i])
    if kd * (d.target[w] - omega) >= d.stop[w]:
        d.latched[w] = 1
    d.target[w] = d.omega[w] * ramp * float(1 - d.latched[w])
    target_qd[i] = d.target[w]
    fc = _clamp_force(d, w, float(joint_q[d.q_slide[w]]), float(joint_qd[i - 1]))
    joint_f[i - 1] = -d.load[w] - fc
    joint_f[i] = -d.bearing[w] * fc * wp.tanh(omega / OMEGA_SMOOTH)


@wp.kernel
def _record(joint_q: wp.array[float], joint_qd: wp.array[float], q_slide: wp.array[wp.int32],
            qd_spin: wp.array[wp.int32], target: wp.array[wp.float32], kd: float, col: int,
            z: wp.array2d[wp.float32], theta: wp.array2d[wp.float32], tau: wp.array2d[wp.float32]):
    w = wp.tid()
    z[w, col] = float(joint_q[q_slide[w]])
    theta[w, col] = float(joint_q[q_slide[w] + 1])
    tau[w, col] = kd * (target[w] - float(joint_qd[qd_spin[w]]))


def thread_metrics(t: np.ndarray, theta: np.ndarray, z: np.ndarray, tau: np.ndarray, settle_s: float = SETTLE_S
                   ) -> dict[str, float]:
    """advance per rev and mean torque over the run-down (after ``settle_s``, ending a quarter turn before seating),
    and the seating slope dT/dθ fitted on T in ``SEAT_FIT`` up to the peak torque. Shared by twins and rig data."""
    out = {"advance_per_rev": math.nan, "rundown_torque": math.nan, "seating_slope": math.nan}
    i0 = int(np.searchsorted(t, settle_s))
    if i0 >= len(t) - 3:
        return out
    early = tau[i0: max(i0 + 2, int(np.searchsorted(theta, theta[i0] + 0.5 * math.pi)))]
    above = np.flatnonzero(tau[i0:] > max(T_SEAT_ON, 5.0 * abs(float(np.median(early)))))
    onset = i0 + int(above[0]) if above.size else len(t)
    end = int(np.searchsorted(theta[:onset], theta[onset - 1] - (0.5 * math.pi if above.size else 0.0)))
    end = min(end, onset - 1)
    if end > i0 + 2 and theta[end] > theta[i0]:
        out["advance_per_rev"] = 2.0 * math.pi * float(z[end] - z[i0]) / float(theta[end] - theta[i0])
        out["rundown_torque"] = float(np.mean(tau[i0: end + 1]))
    if above.size:
        peak = onset + int(np.argmax(tau[onset:]))
        sel = np.arange(onset, peak + 1)
        sel = sel[(tau[sel] >= SEAT_FIT[0]) & (tau[sel] <= SEAT_FIT[1])]
        if sel.size >= 3 and np.ptp(theta[sel]) > 0.0:
            out["seating_slope"] = float(np.polyfit(theta[sel], tau[sel], 1)[0])
    return out


class ThreadTwinBase(BaseTwin):
    """Shared hooks: build from ``rig_for``, the analytic pipeline, drive (load, spin, torque stop) and recording."""

    settings = {"mujoco": THREAD_MUJOCO}
    units = {"advance_per_rev": "m", "rundown_torque": "N*m", "seating_slope": "N*m/rad"}

    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.rigs: list[ThreadRig] = []

    def rig_for(self, cond: Condition, sample: Mapping[str, float]) -> ThreadRig:
        raise NotImplementedError

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rig = self.rig_for(cond, sample)
        self.rigs.append(rig)
        build_thread_world(b, rig)

    def on_model(self, model: newton.Model) -> None:
        model.rigid_contact_max = ROWS_PER_WORLD * model.world_count
        joints = [j for j, lab in enumerate(model.joint_label) if str(lab).endswith("nut_drive")]
        self._drive_in = self._drive_struct(model, joints)

    def _drive_struct(self, model: newton.Model, joints: list[int]) -> DriveIn:
        dev, r = model.device, self.rigs
        d = DriveIn()
        d.q_slide = wp.array(model.joint_q_start.numpy()[joints].astype(np.int32), dtype=wp.int32, device=dev)
        d.qd_spin = wp.array(model.joint_qd_start.numpy()[joints].astype(np.int32) + 1, dtype=wp.int32, device=dev)
        mass = [nut_mass() + x.axial_armature for x in r]
        cols = {"load": [x.axial_load for x in r], "omega": [x.omega for x in r], "stop": [x.torque_stop for x in r],
                "seat": [x.seat_gap for x in r], "k_clamp": [x.k_clamp for x in r],
                "c_clamp": [2.0 * STOP_ZETA * math.sqrt(x.k_clamp * m) for x, m in zip(r, mass)],
                "bearing": [x.mu_bearing * bearing_radius() for x in r]}
        for name, values in cols.items():
            setattr(d, name, wp.array(np.asarray(values, np.float32), dtype=wp.float32, device=dev))
        d.latched = wp.zeros(len(joints), dtype=wp.int32, device=dev)
        d.target = wp.zeros(len(joints), dtype=wp.float32, device=dev)
        return d

    def make_pipeline(self, model: newton.Model, solver: str) -> ThreadPipeline:
        armature = self.rigs[0].axial_armature if self.rigs else ARM_AXIAL
        return ThreadPipeline(model, thread_pairs(row_stiffness(self.dt, armature)), self.dt)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_drive, dim=len(self.rigs), device=self.model.device,
                  inputs=[self._drive_in, state.joint_q, state.joint_qd, KD_SPIN, smoothstep(t / SPIN_RAMP_S),
                          control.joint_f, control.joint_target_qd])

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        col = log_.column(step)
        if col < 0:
            return
        d = self._drive_in
        outs = [log_.channel(n) for n in ("z", "theta", "tau")]
        wp.launch(_record, dim=log_.n_worlds, device=log_.device,
                  inputs=[state.joint_q, state.joint_qd, d.q_slide, d.qd_spin, d.target, KD_SPIN, col],
                  outputs=outs)

    def per_world(self, log_: DeviceTwinLog) -> list[dict[str, float]]:
        t = log_.times()
        z, th, tau = (log_.series(n).astype(np.float64) for n in ("z", "theta", "tau"))
        rows = [thread_metrics(t, th[w], z[w], tau[w]) for w in range(log_.n_worlds)]
        self.monitors["per_world"] = rows
        return rows


# ---------------------------------------------------------------- law A9 --------------------------------------------

A9_DT = thread_dt(5.0)
A9_REVS = 1.0
A9_LAW = LawSpec(
    id="A9.thread_rundown", suite="analytic",
    metrics={"advance_per_rev": Tolerance(rel=0.005, source="bench §4 A9: advance = p within 0.5 %"),
             "rundown_torque": Tolerance(rel=0.05, source="bench §4 A9: power-screw torque within 5 %")},
    stat={"advance_per_rev": "mean", "rundown_torque": "mean"},
    dt=A9_DT, substeps=1, duration_s=SETTLE_S + A9_REVS * 2.0 + 0.05, worlds_per_condition=1, solvers=("mujoco",),
    producer="WP3", quick={"worlds": 2, "duration_s": SETTLE_S + A9_REVS * 2.0 + 0.05})


class ThreadA9Twin(ThreadTwinBase):
    """A9: fixed 6g bolt, 6H nut at π rad/s against 5 N; advance per revolution and mean drive torque over one turn
    after settling, judged against p and the power-screw formula (``reference``)."""

    law_id = A9_LAW.id

    def __init__(self) -> None:
        super().__init__(A9_LAW.dt)

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"mu{mu:g}", {"mu": mu, "F": 5.0, "omega": math.pi}) for mu in (0.14, 0.10))

    def rig_for(self, cond: Condition, sample: Mapping[str, float]) -> ThreadRig:
        return ThreadRig(mu_thread=cond.params["mu"], axial_load=cond.params["F"], omega=cond.params["omega"])

    def reference(self, cond: Condition) -> dict[str, float]:
        """Advance p and the power-screw torque F (d2/2) (p + pi mu d2 sec30) / (pi d2 - mu p sec30) from the law's
        own M10x1.5 numbers (audit AB-laws-2: parsed from BOLT_ID they came through the parser the thread SDF is
        built from, so a wrong pitch there moved the thread and the reference together)."""
        p, d2, k = A9_PITCH_M, A9_D2_M, cond.params["mu"] / math.cos(math.radians(30.0))
        return {"advance_per_rev": p,
                "rundown_torque": cond.params["F"] * 0.5 * d2 * (p + math.pi * k * d2) / (math.pi * d2 - k * p)}

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        rows = self.per_world(log_)
        return {m: np.array([r[m] for r in rows]) for m in ("advance_per_rev", "rundown_torque")}


__all__ = ["A9_LAW", "ThreadA9Twin", "ThreadTwinBase", "ThreadRig", "ThreadPipeline", "build_thread_world",
           "thread_pairs", "thread_metrics", "thread_dt", "thread_spec", "nut_mass", "mujoco_m_eff"]
