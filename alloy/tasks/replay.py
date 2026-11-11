# SPDX-License-Identifier: Apache-2.0
"""Replay a task trace under Newton ``SolverMuJoCo`` and measure object-pose deviation from the recording.

SPEC §7 WP13 acceptance: the converted scene, driven by the recorded inputs, reproduces the recorded MuJoCo
object poses within 1 mm over the first 2 s on CPU (or the deviation is reported with its cause). The trace binds
by name (:mod:`alloy.tasks.newton_map`), so the model can be the vendored MJCF imported directly or any model that
contains the same robot and objects. Per step it writes ``control.mujoco.ctrl`` (``ctrl_direct=True`` keeps the
MJCF actuator order), ``control.joint_f`` and the per-world ``mjw_model.jnt_actfrcrange`` (the recorder's gripper
force clip; batched per world, newton@ac7f7830:newton/_src/solvers/mujoco/solver_mujoco.py:453-471). Object error
is reported as origin distance, rotation angle and the point bound ``|Δp| + r·θ`` the 1 mm gate is applied to.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.core.errors import TraceError
from alloy.core.newton_compat import enable_mujoco_noslip
from alloy.tasks.attribution import pose_errors
from alloy.tasks.newton_map import body_index, initial_to_newton, joint_index, lookup, runs
from alloy.tasks.trace import ACTFRC, CTRL, JOINT_F, QPOS, WARM, TaskTrace

log = logging.getLogger("alloy.tasks")
REPO = Path(__file__).resolve().parents[2]  # trace meta "scene" paths are relative to the repo root


@dataclass(frozen=True)
class ReplayConfig:
    seconds: float = 2.0
    device: str = "cpu"
    multiccd: bool | None = None          # None: as recorded (trace meta "mujoco_multiccd")
    noslip: int | None = None             # None: as recorded (the vendored MJCF's <option noslip_iterations>); 0 off
    nativeccd: bool | None = False        # False: MuJoCo-Warp collides box-box with its primitive box_box, as MuJoCo C
                                          # always does; None: the scene's flag (Warp: GJK/EPA box-box). See make_solver
    nconmax: int = 1024
    njmax: int = 8192
    warmstart: bool = True                # seed qacc_warmstart from the recorded state
    gate_m: float = 1e-3


@dataclass(frozen=True, eq=False)
class ReplayResult:
    trace: str
    times: np.ndarray                     # [S] s after the window start
    objects: tuple[str, ...]
    pos_err: np.ndarray                   # [S, n_obj] m
    ang_err: np.ndarray                   # [S, n_obj] rad
    point_err: np.ndarray                 # [S, n_obj] m, |Δp| + r·θ
    wall_s: float
    steps: int
    overflow: int
    settings: dict = field(default_factory=dict)

    @property
    def max_point_err(self) -> float:
        return float(self.point_err.max(initial=0.0))

    def first_exceed(self, gate: float) -> float | None:
        """Time of the first sample whose worst object error exceeds ``gate`` (None if never)."""
        over = np.nonzero(self.point_err.max(axis=1, initial=0.0) > gate)[0]
        return float(self.times[over[0]]) if len(over) else None


@dataclass
class ReplayRig:
    tt: TaskTrace
    model: newton.Model
    solver: newton.solvers.SolverMuJoCo
    states: list[newton.State]
    control: newton.Control
    joint_f_dofs: np.ndarray              # Newton dof per JOINT_F column
    actfrc_joints: np.ndarray             # MuJoCo-Warp joint per ACTFRC column
    ref_bodies: np.ndarray                # Newton body per reference object


def import_scene(xml: Path, device: str) -> newton.Model:
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(str(xml), ctrl_direct=True, parse_visuals=False)
    return builder.finalize(device=device)


def recorded_noslip(tt: TaskTrace) -> tuple[int, float]:
    """``(noslip_iterations, noslip_tolerance)`` of the trace's vendored scene (``<option>``; MuJoCo defaults 0, 1e-6).

    Newton ac7f7830 drops both when it builds its MjSpec, so the replay sets them back on the built solver."""
    scene = tt.meta.get("scene")
    opt = ET.parse(REPO / scene).getroot().find("option") if scene else None
    if opt is None:
        return 0, 1e-6
    return int(opt.get("noslip_iterations", 0)), float(opt.get("noslip_tolerance", 1e-6))


def make_solver(model: newton.Model, tt: TaskTrace, cfg: ReplayConfig) -> newton.solvers.SolverMuJoCo:
    """MuJoCo C always collides box-box pairs with ``mjc_BoxBox``; MuJoCo-Warp uses its primitive ``box_box`` only with
    NATIVECCD disabled (GJK/EPA otherwise) and ignores the flag everywhere else (mujoco_warp@86e2fae0:
    mujoco_warp/_src/io.py:876-878, collision_driver.py:1046-1050). The replay therefore disables it by default to get
    the recorder's box-box (box drops: bin64box 21.9 mm with the scene's flag, 0.018 mm without). Noslip comes from the
    recording (every robot trace ran 50 iterations at 1e-10) unless ``cfg.noslip`` overrides it."""
    multiccd = bool(tt.meta.get("mujoco_multiccd", False)) if cfg.multiccd is None else cfg.multiccd
    if cfg.nativeccd is not None:
        model.mujoco.flag_nativeccd.assign(np.array([cfg.nativeccd], dtype=bool))
        log.info("replay %s: NATIVECCD %s", tt.trace.source, "on" if cfg.nativeccd else "off")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, enable_multiccd=multiccd,
                                         nconmax=cfg.nconmax, njmax=cfg.njmax)  # fmt: skip
    iterations, tolerance = recorded_noslip(tt)
    iterations = iterations if cfg.noslip is None else cfg.noslip
    if iterations:
        enable_mujoco_noslip(solver, iterations, tolerance)
        log.info("replay %s: noslip %d iterations, tolerance %g", tt.trace.source, iterations, tolerance)
    return solver


def _mjw_joint_of(solver: newton.solvers.SolverMuJoCo, newton_joints: np.ndarray) -> np.ndarray:
    to_newton = solver.mjc_jnt_to_newton_jnt.numpy()[0]
    inverse = {int(n): k for k, n in enumerate(to_newton) if n >= 0}
    return np.array([inverse[int(j)] for j in newton_joints], dtype=np.int64)


def _seed_warmstart(rig: ReplayRig) -> None:
    """Recorded ``qacc_warmstart`` (MuJoCo dof order) into the solver's MuJoCo-Warp data, joint by joint."""
    mjm = rig.solver.mj_model
    jindex = joint_index(rig.model)
    warm = rig.solver.mjw_data.qacc_warmstart.numpy()
    for name, start, width in runs(rig.tt.trace.columns[WARM]):
        k = int(_mjw_joint_of(rig.solver, lookup(jindex, (name,), "joint"))[0])
        adr = int(mjm.jnt_dofadr[k])
        warm[0, adr : adr + width] = rig.tt.trace.initial[WARM][start : start + width]
    rig.solver.mjw_data.qacc_warmstart.assign(warm)


def _set_initial_state(rig: ReplayRig, cfg: ReplayConfig) -> None:
    tt, model, s0 = rig.tt, rig.model, rig.states[0]
    q, qd = initial_to_newton(model, tt.trace.columns, tt.trace.initial[QPOS], tt.trace.initial["qvel"])
    s0.joint_q.assign(q.astype(np.float32))
    s0.joint_qd.assign(qd.astype(np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    if cfg.warmstart:
        _seed_warmstart(rig)


def build_rig(model: newton.Model, tt: TaskTrace, cfg: ReplayConfig = ReplayConfig()) -> ReplayRig:
    """Solver, name bindings and the recorded initial state for ``model``."""
    control = model.control()
    nu = tt.trace.inputs[CTRL].shape[1]
    have = control.mujoco.ctrl.shape[0] if nu else 0
    if have != nu * model.world_count:
        raise TraceError(f"model has {have} actuator controls, the trace {nu}")
    solver = make_solver(model, tt, cfg)
    jindex = joint_index(model)
    dofs = model.joint_qd_start.numpy()[lookup(jindex, tt.trace.columns[JOINT_F], "joint")]
    actfrc = _mjw_joint_of(solver, lookup(jindex, tt.trace.columns[ACTFRC], "joint"))
    bodies = lookup(body_index(model), tt.ref.names, "body")
    rig = ReplayRig(tt, model, solver, [model.state(), model.state()], control, dofs, actfrc, bodies)
    _set_initial_state(rig, cfg)
    return rig


def _apply_row(rig: ReplayRig, i: int, joint_f: np.ndarray, afr: np.ndarray) -> None:
    inputs = rig.tt.trace.inputs
    if inputs[CTRL].shape[1]:
        rig.control.mujoco.ctrl.assign(inputs[CTRL][i].astype(np.float32))
    if len(rig.joint_f_dofs):
        joint_f[rig.joint_f_dofs] = inputs[JOINT_F][i]
        rig.control.joint_f.assign(joint_f)
    if len(rig.actfrc_joints):
        afr[0, rig.actfrc_joints] = inputs[ACTFRC][i]
        rig.solver.mjw_model.jnt_actfrcrange.assign(afr)


def _sample(rig: ReplayRig, k: int) -> np.ndarray:
    body_q = rig.states[0].body_q.numpy().astype(np.float64)[rig.ref_bodies]
    ref = rig.tt.ref
    return pose_errors(body_q[:, :3], body_q[:, 3:], ref.pos[k], ref.quat[k], ref.radius, scalar_last=True)


def run_replay(rig: ReplayRig, cfg: ReplayConfig = ReplayConfig()) -> ReplayResult:
    """Step through the trace, sampling object poses where the recording has reference rows."""
    trace, every = rig.tt.trace, int(rig.tt.meta.get("ref_every", 50))
    steps = min(int(round(cfg.seconds / trace.control_dt)), trace.periods, len(rig.tt.ref.times) * every)
    obs = rig.solver.observables({rig.solver.ObservableKind.OVERFLOW})
    joint_f = np.zeros(rig.model.joint_dof_count, dtype=np.float32)
    afr = rig.solver.mjw_model.jnt_actfrcrange.numpy()
    samples, overflow, t0 = [], 0, time.perf_counter()
    for i in range(steps):
        _apply_row(rig, i, joint_f, afr)
        rig.solver.step(rig.states[0], rig.states[1], rig.control, None, trace.control_dt, observables=obs)
        rig.states.reverse()
        overflow |= int(obs.overflow.numpy().max())
        if (i + 1) % every == 0:
            samples.append(_sample(rig, (i + 1) // every - 1))
    wall = time.perf_counter() - t0
    arr = np.array(samples).reshape(-1, 3, len(rig.tt.ref.names))
    opt = rig.solver.mj_model.opt
    settings = {"device": cfg.device, "disableflags": int(opt.disableflags),
                "noslip_iterations": int(opt.noslip_iterations), "nconmax": cfg.nconmax, "njmax": cfg.njmax,
                "warmstart": cfg.warmstart}  # fmt: skip
    log.info("%s: %d steps in %.1f s, max point error %.3g m", trace.source, steps, wall, arr[:, 2].max(initial=0.0))
    return ReplayResult(trace.source, rig.tt.ref.times[: len(arr)], rig.tt.ref.names, arr[:, 0], arr[:, 1], arr[:, 2],
                        wall, steps, overflow, settings)  # fmt: skip


def replay_scene(xml: Path, tt: TaskTrace, cfg: ReplayConfig = ReplayConfig()) -> ReplayResult:
    """Import the vendored MJCF, then build and run the replay on ``cfg.device``."""
    with wp.ScopedDevice(cfg.device):
        return run_replay(build_rig(import_scene(xml, cfg.device), tt, cfg), cfg)


def replay_model(model: newton.Model, tt: TaskTrace, cfg: ReplayConfig = ReplayConfig()) -> ReplayResult:
    """Replay on an already built Model, e.g. ``SceneAssembler.assemble(scene).model`` (acceptance item 1 + 2): the
    trace binds by joint/body/actuator names, so the assembler's label prefixes and extra entities do not matter."""
    with wp.ScopedDevice(cfg.device):
        return run_replay(build_rig(model, tt, cfg), cfg)
