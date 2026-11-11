# SPDX-License-Identifier: Apache-2.0
"""Bully targets on the converted task scenes (WP13's vendored MJCF + recorded traces) under ``SolverMuJoCo``.

``task_t12`` runs the production route: WP2's ``SceneAssembler`` builds the ``task_t12`` scene card, and
``Simulation``'s stepper supplies the routed solver (the scene's MuJoCo options, Alloy's scoped noslip) and its clock
rule: the route's 1/1200 s snapped to the bully's 20 ms period gives 0.625 ms, what ``Simulation`` runs for a 50 Hz
controller. ``DirectSim`` steps that solver so the monitors see every step. ``task_t12_raw`` is
the earlier raw-Newton stand-in (``add_mjcf`` + ``SolverMuJoCo``: no noslip, 1.25 ms), kept for comparison.

``task_target("t12")``: the dual-arm block scene. Actions are the 14 position-actuator targets
(``control.mujoco.ctrl``, ``ctrl_direct``): the legal box is each actuated joint's range, the rate limit a joint
target speed (2 rad/s arms, 0.1 m/s fingers), the initial command the trace's first row, and the initial state the
recorded ``qpos``/``qvel`` mapped to Newton coordinates. MuJoCo-native contacts are exported each step
(``SolverMuJoCo.update_contacts``) so the monitors see robot contact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from alloy._vendor import newton

from alloy.bully.actions import MUJOCO_CTRL
from alloy.bully.direct import DirectSim
from alloy.bully.targets import register_target
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

ROOT = Path(__file__).resolve().parents[2]
SCENES = {"t12": ("library/robots/task_t12/scene.xml", "bench/traces/t12", "task_t12")}  # MJCF, trace, scene card
_FINGER_RANGE_MAX = 0.1  # [m] a slide joint with a range below this is a finger (rate in m/s, not rad/s)


@dataclass(frozen=True)
class TaskTargetSpec:
    scene: str = "t12"
    routed: bool = True  # the production route (scene card + routed solver and step); False: raw-Newton stand-in
    # WP5's ordered MuJoCo Warp route (main 60f64eb): bitwise repeat, and a world's trajectory does not depend on its
    # batch, so an SB-7 trip at batch index k replays with that world alone. Routed target only.
    deterministic: bool = True
    control_dt: float = 0.02
    substeps: int = 16  # raw stand-in only (1.25 ms, D4 grid); the routed target takes the route's step
    joint_speed: float = 2.0  # [rad/s] legal arm joint target rate
    finger_speed: float = 0.1  # [m/s] legal finger target rate
    # Per-world MuJoCo Warp capacities. SB-7 on T12 (5,120 episodes, L4) peaked at 627 rows (p99 522-618) and 134
    # contacts per world (98.5k in the pooled buffer); at njmax 512, 85 worlds dropped rows and 10 went non-finite,
    # at 1024 none did ([internal note]). 2x the old 128 / 512 covers 1.5x p99 (SPEC
    # §4.2) with headroom for B1; any overflow left is an S1 ``capacity_overflow`` finding.
    nconmax: int | None = 256
    njmax: int | None = 1024
    # MuJoCo Warp CCD scratch slots per world (SCR-WP8-7 M1, WP5's alloy.precision.mjw_ccd): T12 uses <= 6, so 32
    # instead of the default nconmax lets 16,384 worlds fit (epa_pr was 9.1 GB); an overflow is CONTACT_OVERFLOW
    # (INVALID) and the capacity_overflow row, so it fails closed. None keeps MuJoCo Warp's default (naconmax).
    ccd_slots: int | None = 32


def _template(xml: Path) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(xml), ctrl_direct=True, parse_visuals=False)
    return b


@dataclass(frozen=True)
class _Built:
    model: newton.Model
    solver: newton.solvers.SolverMuJoCo
    dt: float  # [s] solver step


def _raw_build(spec: TaskTargetSpec, worlds: int, device: str, init: _SceneInfo | None) -> _Built:
    """The raw-Newton stand-in: ``add_mjcf`` replicated, ``SolverMuJoCo`` with Newton's defaults, 1.25 ms."""
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(_template(ROOT / SCENES[spec.scene][0]), copy_count=worlds)
    model = scene.finalize(device=device)
    kw = {} if init is None else {"enable_multiccd": init.multiccd, "nconmax": spec.nconmax, "njmax": spec.njmax}
    if init is not None:
        _set_initial(model, init, worlds)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, **kw)
    if spec.ccd_slots is not None:
        from alloy.precision.mjw_ccd import set_ccd_slots

        set_ccd_slots(solver, spec.ccd_slots)  # before the first step (mjw_ccd)
    return _Built(model, solver, spec.control_dt / spec.substeps)


def _routed_build(spec: TaskTargetSpec, worlds: int, device: str, init: _SceneInfo | None) -> _Built:
    """The production route: the scene card assembled (capacities from ``spec``), the routed solver, and the step
    ``Simulation`` would take for ``spec.control_dt``."""
    import warp as wp

    from alloy.cards.library import CardLibrary
    from alloy.parallel.types import PackSpec
    from alloy.scene.assembler import SceneAssembler
    from alloy.sim import Simulation

    caps = None
    if spec.njmax is not None and spec.nconmax is not None:
        caps = PackSpec(device=device, world_ids=tuple(range(worlds)), n=worlds, stream_priority=0, kind="throughput",
                        rigid_contact_max=spec.nconmax * worlds, njmax=spec.njmax, nconmax=spec.nconmax,
                        shape_pairs_max=0, predicted_bytes=0)
    lib = CardLibrary()
    built = SceneAssembler(lib).assemble(lib.scene(SCENES[spec.scene][2]), device, world_count=worlds, capacities=caps)
    if spec.ccd_slots is not None:  # the MuJoCo entry factory sizes the pool right after construction (mjw_ccd)
        from alloy.precision.mjw_ccd import with_ccd_slots

        built = with_ccd_slots(built, spec.ccd_slots)
    if init is not None:  # before the solver exists: SolverMuJoCo copies the model's joint state at construction
        _set_initial(built.model, init, worlds)
    with wp.ScopedDevice(device):
        sim = Simulation(built)
    solvers = sim.stepper.solvers()
    if len(solvers) != 1 or not isinstance(solvers[0], newton.solvers.SolverMuJoCo):
        raise BullyError(f"scene {spec.scene!r} does not route to one SolverMuJoCo entry: {solvers}")
    if spec.deterministic:
        _enable_determinism(solvers[0], spec.scene)
    dt, _ = sim.stepper.choose_clock(spec.control_dt, None)  # Simulation's clock rule for this control period
    return _Built(built.model, solvers[0], dt)


def _enable_determinism(solver: newton.solvers.SolverMuJoCo, scene: str) -> None:
    """WP5's ordered route, before the first step (DirectSim builds no RUN_TO_RUN HealthMonitor that would do it)."""
    from alloy.precision.mjw_route import enable_route

    why = enable_route(solver)
    if why is not None:
        log.warning("task_%s: ordered MuJoCo Warp route stays off (%s); GPU trips will not replay exactly", scene, why)


def _set_initial(model: newton.Model, info: _SceneInfo, worlds: int) -> None:
    model.joint_q.assign(np.tile(info.joint_q, worlds).astype(np.float32))
    model.joint_qd.assign(np.tile(info.joint_qd, worlds).astype(np.float32))


def _build(spec: TaskTargetSpec, worlds: int, device: str, init: _SceneInfo | None) -> _Built:
    return (_routed_build if spec.routed else _raw_build)(spec, worlds, device, init)


@dataclass(frozen=True)
class _SceneInfo:
    joint_q: np.ndarray  # one world's Newton joint_q at the trace start
    joint_qd: np.ndarray
    low: np.ndarray  # per actuator
    high: np.ndarray
    ctrl0: np.ndarray
    multiccd: bool
    robot: tuple[int, ...]  # template bodies driven by the actuators (their articulations, free joints excluded)


@lru_cache(maxsize=4)
def _scene_info(spec: TaskTargetSpec) -> _SceneInfo:
    """One-world model + MuJoCo conversion: actuator joint ranges and the recorded initial state (host, once)."""
    from alloy.tasks.newton_map import initial_to_newton
    from alloy.tasks.trace import load_trace

    if spec.scene not in SCENES:
        raise BullyError(f"no task scene {spec.scene!r}; known: {sorted(SCENES)}")
    tt = load_trace(ROOT / SCENES[spec.scene][1])
    b = _build(spec, 1, "cpu", None)
    mj = b.solver.mj_model
    joints = mj.actuator_trnid[:, 0]
    robot = _actuated_bodies(b.model, b.solver.mjc_jnt_to_newton_jnt.numpy()[0][joints])
    low, high = (np.asarray(mj.jnt_range[joints, k], dtype=np.float64) for k in (0, 1))
    q, qd = initial_to_newton(b.model, tt.trace.columns, tt.trace.initial["qpos"], tt.trace.initial["qvel"])
    ctrl0 = np.clip(np.asarray(tt.trace.inputs[MUJOCO_CTRL][0], dtype=np.float64), low, high)
    return _SceneInfo(q, qd, low, high, ctrl0, bool(tt.meta.get("mujoco_multiccd", False)), robot)


def _actuated_bodies(model: newton.Model, newton_joints: np.ndarray) -> tuple[int, ...]:
    """Child bodies of every non-free joint in the articulations that hold an actuated joint (one-world model)."""
    art, jt, child = (model.joint_articulation.numpy(), model.joint_type.numpy(), model.joint_child.numpy())
    arts = {int(art[j]) for j in newton_joints if j >= 0 and art[j] >= 0}
    free = (int(newton.JointType.FREE), int(newton.JointType.DISTANCE))
    return tuple(sorted(int(child[j]) for j in range(jt.size) if int(art[j]) in arts and int(jt[j]) not in free))


def _substeps(control_dt: float, dt: float) -> int:
    k = int(round(control_dt / dt))
    if k < 1 or abs(k * dt - control_dt) > 1e-9 * control_dt:
        raise BullyError(f"control period {control_dt} s is not a whole number of solver steps of {dt} s")
    return k


def task_target(spec: TaskTargetSpec | None = None) -> BullyTarget:
    """The task scene as a bully target (see module doc)."""
    spec = spec or TaskTargetSpec()
    info = _scene_info(spec)

    def make_sim(worlds: int, device: str) -> DirectSim:
        b = _build(spec, worlds, device, info)
        sim = DirectSim(b.model, b.solver, control_dt=spec.control_dt, substeps=_substeps(spec.control_dt, b.dt),
                        collide=False, export_contacts=True)
        sim.control.mujoco.ctrl.assign(np.tile(info.ctrl0, worlds).astype(np.float32))
        return sim

    rate = 1.0 / spec.control_dt
    finger = (info.high - info.low) < _FINGER_RANGE_MAX
    speed = np.where(finger, spec.finger_speed, spec.joint_speed)
    actions = ActionSpace(low=info.low, high=info.high, rate_hz=rate, max_delta_per_tick=speed / rate,
                          initial=info.ctrl0)
    name = f"task_{spec.scene}" + ("" if spec.routed else "_raw")
    return BullyTarget(name=name, scene=None, make_sim=make_sim, actions=actions,
                       action_kind=MUJOCO_CTRL, action_index=tuple(range(info.low.size)), robot_bodies=info.robot)


register_target("task_t12", lambda: task_target(TaskTargetSpec(scene="t12")))
register_target("task_t12_raw", lambda: task_target(TaskTargetSpec(scene="t12", routed=False)))
