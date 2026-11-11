# SPDX-License-Identifier: Apache-2.0
"""Bully targets for the P3-2b embodiment scenes: ``emb_<key>`` runs ``library/robots/emb_<key>/scene.xml`` on the
Alloy route, as ``task_t12`` does for T12 (alloy/bully/tasks.py).

- Route: WP2's ``SceneAssembler`` builds the scene card ``task_emb_<key>``; ``Simulation``'s stepper supplies the
  routed ``SolverMuJoCo`` and its clock rule for the bully's 20 ms period. WP5's ordered MuJoCo Warp route is on, so
  a trip at batch index k replays with that world alone. ``DirectSim`` steps the solver so the monitors see every
  step, with MuJoCo-native contacts exported.
- Actions (``mujoco.ctrl``, every MJCF actuator in order): the legal box is each actuated joint's range intersected
  with the actuator's ctrl range (a continuous joint, the Duo's drive wheels: the ctrl range). Torque limits are the actuators' force ranges, applied by MuJoCo. Commands are held
  at 50 Hz and rate-limited per tick: 2 rad/s for hinges, 0.1 m/s for slides (fingers), as for T12. The initial
  command is the ``init`` keyframe's ctrl.
- Initial state: the ``init`` keyframe's qpos / qvel, mapped to Newton coordinates by joint name.
- Robot bodies: every body of the articulations that hold an actuated joint, the floating base of a legged robot
  included (it is the robot, not a passive object, for the monitors' energy ledger).

Registered on import of this module, which ``alloy.bully.targets`` lists as a provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from alloy._vendor import newton

from alloy.bully.actions import MUJOCO_CTRL
from alloy.bully.direct import DirectSim
from alloy.bully.targets import register_target
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.core.errors import BullyError
from alloy.tasks.emb_scene import card_name, scene_path
from alloy.tasks.emb_specs import EMBODIMENTS, Embodiment

log = logging.getLogger("alloy.tasks")
INIT_KEY = "init"


@dataclass(frozen=True)
class EmbTargetSpec:
    key: str
    control_dt: float = 0.02
    deterministic: bool = True
    joint_speed: float = 2.0        # [rad/s] legal hinge target rate
    slide_speed: float = 0.1        # [m/s] legal slide (finger) target rate
    nconmax: int = 256              # per-world MuJoCo Warp capacities (T12's); an overflow is an S1 finding
    njmax: int = 1024


@dataclass(frozen=True)
class _Info:
    joint_q: np.ndarray
    joint_qd: np.ndarray
    low: np.ndarray
    high: np.ndarray
    slide: np.ndarray               # bool per actuator: a slide joint (m, m/s)
    ctrl0: np.ndarray
    robot: tuple[int, ...]


@dataclass(frozen=True)
class _Built:
    model: newton.Model
    solver: newton.solvers.SolverMuJoCo
    dt: float


def _capacities(spec: EmbTargetSpec, worlds: int, device: str):  # noqa: ANN202 - PackSpec (lazy import)
    from alloy.parallel.types import PackSpec

    return PackSpec(device=device, world_ids=tuple(range(worlds)), n=worlds, stream_priority=0, kind="throughput",
                    rigid_contact_max=spec.nconmax * worlds, njmax=spec.njmax, nconmax=spec.nconmax,
                    shape_pairs_max=0, predicted_bytes=0)  # fmt: skip


def _build(spec: EmbTargetSpec, worlds: int, device: str, info: _Info | None) -> _Built:
    import warp as wp

    from alloy.cards.library import CardLibrary
    from alloy.precision.mjw_route import enable_route
    from alloy.scene.assembler import SceneAssembler
    from alloy.sim import Simulation

    e = EMBODIMENTS[spec.key]
    lib = CardLibrary()
    built = SceneAssembler(lib).assemble(lib.scene(card_name(e)), device, world_count=worlds,
                                         capacities=_capacities(spec, worlds, device))  # fmt: skip
    if info is not None:  # before the solver exists: SolverMuJoCo copies the joint state at construction
        built.model.joint_q.assign(np.tile(info.joint_q, worlds).astype(np.float32))
        built.model.joint_qd.assign(np.tile(info.joint_qd, worlds).astype(np.float32))
    with wp.ScopedDevice(device):
        sim = Simulation(built)
    solvers = sim.stepper.solvers()
    if len(solvers) != 1 or not isinstance(solvers[0], newton.solvers.SolverMuJoCo):
        raise BullyError(f"{e.name} does not route to one SolverMuJoCo entry: {solvers}")
    if spec.deterministic and (why := enable_route(solvers[0])) is not None:
        log.warning("%s: ordered MuJoCo Warp route stays off (%s); GPU trips will not replay exactly", e.name, why)
    dt, _ = sim.stepper.choose_clock(spec.control_dt, None)
    return _Built(built.model, solvers[0], dt)


def _robot_bodies(model: newton.Model, newton_joints: np.ndarray) -> tuple[int, ...]:
    """Child bodies of every joint in the articulations that hold an actuated joint (one-world model)."""
    art, child = model.joint_articulation.numpy(), model.joint_child.numpy()
    arts = {int(art[j]) for j in newton_joints if j >= 0 and art[j] >= 0}
    return tuple(sorted(int(child[j]) for j in range(art.size) if int(art[j]) in arts))


def action_box(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per actuator: legal low / high (joint range inside the ctrl range) and whether its joint slides."""
    if np.any(m.actuator_trntype != int(mujoco.mjtTrn.mjTRN_JOINT)):
        raise BullyError("embodiment actuators must drive joints (no tendon or site transmissions)")
    joints = m.actuator_trnid[:, 0]
    limited = m.jnt_limited[joints].astype(bool)
    low = np.where(limited, m.jnt_range[joints, 0], -np.inf).astype(np.float64)
    high = np.where(limited, m.jnt_range[joints, 1], np.inf).astype(np.float64)
    ctrl = m.actuator_ctrllimited.astype(bool)
    low[ctrl] = np.maximum(low[ctrl], m.actuator_ctrlrange[ctrl, 0])
    high[ctrl] = np.minimum(high[ctrl], m.actuator_ctrlrange[ctrl, 1])
    if not np.all(np.isfinite(low) & np.isfinite(high)) or np.any(high <= low):
        raise BullyError("every actuator needs a finite legal box (joint range or, for a continuous joint, ctrl range)")
    return low, high, m.jnt_type[joints] == int(mujoco.mjtJoint.mjJNT_SLIDE)


@lru_cache(maxsize=16)
def _info(spec: EmbTargetSpec) -> _Info:
    """Host, once: the MJCF's action box and ``init`` keyframe, mapped onto a one-world Newton model."""
    from alloy.tasks.mjmodel_info import element_columns
    from alloy.tasks.newton_map import initial_to_newton

    m = mujoco.MjModel.from_xml_path(str(scene_path(EMBODIMENTS[spec.key])))
    k = m.key(INIT_KEY).id
    b = _build(spec, 1, "cpu", None)
    columns = {"qpos": element_columns(m, 0), "qvel": element_columns(m, 1)}
    q, qd = initial_to_newton(b.model, columns, m.key_qpos[k].astype(np.float64), m.key_qvel[k].astype(np.float64))
    low, high, slide = action_box(m)
    ctrl0 = np.clip(m.key_ctrl[k].astype(np.float64), low, high)
    joints = b.solver.mjc_jnt_to_newton_jnt.numpy()[0][m.actuator_trnid[:, 0]]
    return _Info(q, qd, low, high, slide, ctrl0, _robot_bodies(b.model, joints))


def emb_target(spec: EmbTargetSpec) -> BullyTarget:
    info = _info(spec)

    def make_sim(worlds: int, device: str) -> DirectSim:
        b = _build(spec, worlds, device, info)
        k = int(round(spec.control_dt / b.dt))
        if k < 1 or abs(k * b.dt - spec.control_dt) > 1e-9 * spec.control_dt:
            raise BullyError(f"control period {spec.control_dt} s is not a whole number of steps of {b.dt} s")
        sim = DirectSim(b.model, b.solver, control_dt=spec.control_dt, substeps=k, collide=False,
                        export_contacts=True)  # fmt: skip
        sim.control.mujoco.ctrl.assign(np.tile(info.ctrl0, worlds).astype(np.float32))
        return sim

    rate = 1.0 / spec.control_dt
    speed = np.where(info.slide, spec.slide_speed, spec.joint_speed)
    actions = ActionSpace(low=info.low, high=info.high, rate_hz=rate, max_delta_per_tick=speed / rate,
                          initial=info.ctrl0)  # fmt: skip
    return BullyTarget(name=EMBODIMENTS[spec.key].name, scene=None, make_sim=make_sim, actions=actions,
                       action_kind=MUJOCO_CTRL, action_index=tuple(range(info.low.size)), robot_bodies=info.robot)


def _register(e: Embodiment) -> None:
    register_target(e.name, lambda: emb_target(EmbTargetSpec(e.key)))


for _e in EMBODIMENTS.values():
    if scene_path(_e).is_file():
        _register(_e)
