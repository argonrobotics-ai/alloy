# SPDX-License-Identifier: Apache-2.0
"""Any scene card as a bully target (``alloy.bully.audit``): the product loop and the scene's own legal inputs.

The scene is assembled by WP2's ``SceneAssembler`` and run by the ``IntegratedSimulation`` that
``build_simulation`` wires (``SceneSim``), with the card's ``control_dt`` replaced by the bully's command period
(``AuditOptions.control_dt``, 20 ms = SB-7's 50 Hz), so the engine step follows ``Simulation``'s clock rule for it.

Command channel (``AuditOptions.channel``; ``auto`` takes the first that exists):

- ``ctrl``: the scene's MuJoCo actuators (``control.mujoco.ctrl``), as the SB-7 robot targets. Legal box: the ctrl
  range, intersected with the joint range for position servos. Rate limit: 2 rad/s (hinge servos), 0.1 m/s (slide
  servos), the full range in 1 s for every other actuator. Initial command: a servo holds its joint's start
  position, any other actuator starts at 0 (clipped into its box). Actuators without a finite box are not commanded.
- ``push``: bounded external forces on every free body (no robot): each force axis in +-``push_g`` x the body's
  weight (0.2 by default, so a push never lifts a resting object), full range in ``push_rate_s``. The bench energy
  ledger books the work of these forces, as for any ``body_f`` target.
- ``passive``: nothing to command (a scene of particles only); every world runs the scene as built, so the
  episodes differ only by the card's randomized values.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bully.actions import MUJOCO_CTRL, PASSIVE
from alloy.bully.roles import free_bodies
from alloy.bully.scene_sim import SceneSim
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

CHANNELS = ("auto", "ctrl", "push", "passive")


@dataclass(frozen=True)
class AuditOptions:
    """How a scene card becomes a bully target (module doc)."""

    control_dt: float = 0.02  # [s] command period; replaces the card's control_dt
    channel: str = "auto"  # auto | ctrl | push | passive
    push_g: float = 0.2  # push box per force axis, in units of the body's weight
    push_rate_s: float = 0.25  # [s] a push crosses its whole box in this time (rate limit)
    joint_speed: float = 2.0  # [rad/s] hinge servo target rate (SB-7)
    slide_speed: float = 0.1  # [m/s] slide servo target rate (SB-7)
    other_rate_s: float = 1.0  # [s] any other actuator crosses its whole ctrl range in this time
    profile: str = "standard"  # alloy.integration profile (graphs are always off)
    mjw_route: bool = False  # build MuJoCo entries for WP5's ordered route (sparse Jacobian, ~1.3x the step): GPU
    #                          findings then replay alone on scenes the dense default keeps off it

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise BullyError(f"channel {self.channel!r}: pass one of {CHANNELS}")
        if not self.control_dt > 0.0 or not self.push_g >= 0.0:
            raise BullyError(f"control_dt {self.control_dt} must be > 0 and push_g {self.push_g} >= 0")


@dataclass(frozen=True, eq=False)
class Channel:
    """The legal inputs found in a scene: action kind, template indices and the box."""

    kind: str
    index: tuple[int, ...]
    low: np.ndarray
    high: np.ndarray
    max_delta: np.ndarray
    initial: np.ndarray
    robot_bodies: tuple[int, ...]
    description: str


def scene_target(scene: Any, options: AuditOptions | None = None, library: Any = None,
                 device: str = "cpu") -> tuple[BullyTarget, Channel]:
    """``scene`` (YAML path, library scene name, YAML text or ``SceneCard``, as ``alloy.check``) as a bully
    target, and the command channel found in it (a one-world build on ``device``)."""
    from alloy.check.source import load_source  # noqa: PLC0415 - alloy.check builds on the bully package

    opts = options or AuditOptions()
    src = load_source(scene, library)
    card = dataclasses.replace(src.card, control_dt=opts.control_dt)
    channel = discover(build_scene_sim(src.lib, card, 1, device, opts), opts)

    def make_sim(worlds: int, dev: str) -> SceneSim:
        return build_scene_sim(src.lib, card, worlds, dev, opts)

    rate = 1.0 / opts.control_dt
    actions = ActionSpace(low=channel.low, high=channel.high, rate_hz=rate, max_delta_per_tick=channel.max_delta,
                          initial=channel.initial)
    target = BullyTarget(name=f"scene:{src.label}", scene=card, make_sim=make_sim, actions=actions,
                         action_kind=channel.kind, action_index=channel.index, robot_bodies=channel.robot_bodies)
    log.info("audit target %s: %s", target.name, channel.description)
    return target, channel


def build_scene_sim(lib: Any, card: Any, worlds: int, device: str, opts: AuditOptions) -> SceneSim:
    """Assemble ``card`` for ``worlds`` worlds and wire the product simulation (graphs off)."""
    from alloy.integration.wiring import build_simulation  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    built = SceneAssembler(lib).assemble(card, device, world_count=worlds)
    with wp.ScopedDevice(device):
        sim = SceneSim(build_simulation(built, opts.profile, graph="off", mjw_route=opts.mjw_route))
    sim.ordered = ordered_route(sim)
    return sim


def ordered_route(sim: SceneSim) -> bool:
    """WP5's ordered MuJoCo Warp route on every ``SolverMuJoCo`` entry, before the first step (as the SB-7 robot
    targets): a world's trajectory then does not depend on its batch, so a replay handle reproduces a GPU finding
    alone. False when an entry stays off it (the reason is logged) or the scene has no such entry."""
    from alloy.precision.mjw_route import enable_route, route_enabled  # noqa: PLC0415

    mj = [s for s in sim.entry_solvers if type(s).__name__ == "SolverMuJoCo"]
    for s in mj:
        why = None if route_enabled(s) else enable_route(s)
        if why is not None:
            log.info("audit: ordered MuJoCo Warp route stays off (%s); GPU findings may not replay alone", why)
            return False
    return bool(mj)


def discover(sim: SceneSim, opts: AuditOptions) -> Channel:
    """The first command channel of ``opts.channel`` (``auto``: ctrl, then push, then passive) found in ``sim``."""
    order = ("ctrl", "push", "passive") if opts.channel == "auto" else (opts.channel,)
    for name in order:
        found = {"ctrl": ctrl_channel, "push": push_channel, "passive": lambda s, o: passive_channel()}[name](sim, opts)
        if found is not None:
            return found
    raise BullyError(f"channel {opts.channel!r}: the scene has no such inputs; pass channel='auto'")


def passive_channel() -> Channel:
    empty = np.zeros(0)
    return Channel(PASSIVE, (), empty, empty, empty, empty, (), "passive: no actuators and no free bodies")


def ctrl_channel(sim: SceneSim, opts: AuditOptions) -> Channel | None:
    """The scene's MuJoCo actuators, or None when the route has no ``SolverMuJoCo`` entry with actuators."""
    solver = sim.solver
    m = getattr(solver, "mj_model", None)
    ctrl = getattr(getattr(sim.control, "mujoco", None), "ctrl", None)
    if m is None or ctrl is None or int(m.nu) == 0:
        return None
    low, high, servo, slide, joint = ctrl_box(m)
    usable = np.isfinite(low) & np.isfinite(high) & (high > low)
    if not usable.any():
        return None
    rate = 1.0 / opts.control_dt
    speed = np.where(slide, opts.slide_speed, opts.joint_speed)
    other = np.where(usable, high - low, 0.0) / opts.other_rate_s
    max_delta = np.where(servo, speed, other) / rate
    newton_joint = np.where(joint >= 0, solver.mjc_jnt_to_newton_jnt.numpy()[0][np.maximum(joint, 0)], -1)
    q0 = _joint_positions(sim.model, newton_joint)
    initial = np.clip(np.where(servo, q0, 0.0), low, high)
    idx = np.nonzero(usable)[0]
    robot = _articulation_bodies(sim.model, newton_joint[newton_joint >= 0])  # every joint actuator's robot
    text = f"ctrl: {idx.size} of {int(m.nu)} MuJoCo actuators ({int(servo[idx].sum())} position servos)"
    return Channel(MUJOCO_CTRL, tuple(int(i) for i in idx), low[idx], high[idx], max_delta[idx], initial[idx],
                   robot, text)


def ctrl_box(m: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per actuator of a compiled ``MjModel``: legal low / high (ctrl range; a position servo also its joint
    range; +-inf where unbounded), whether it is a position servo on a joint, whether that joint slides, and the
    MuJoCo joint it drives (-1: tendon, site or body transmission)."""
    import mujoco  # noqa: PLC0415

    nu = int(m.nu)
    low, high = np.full(nu, -np.inf), np.full(nu, np.inf)
    lim = np.asarray(m.actuator_ctrllimited).astype(bool)
    low[lim], high[lim] = m.actuator_ctrlrange[lim, 0], m.actuator_ctrlrange[lim, 1]
    joint = np.asarray(m.actuator_trntype) == int(mujoco.mjtTrn.mjTRN_JOINT)
    jid = np.where(joint, m.actuator_trnid[:, 0], 0)
    servo = joint & (np.asarray(m.actuator_biastype) == int(mujoco.mjtBias.mjBIAS_AFFINE)) & \
        (m.actuator_biasprm[:, 1] < 0.0) & (m.actuator_gainprm[:, 0] > 0.0)
    jl = servo & np.asarray(m.jnt_limited)[jid].astype(bool)
    low[jl] = np.maximum(low[jl], m.jnt_range[jid[jl], 0])
    high[jl] = np.minimum(high[jl], m.jnt_range[jid[jl], 1])
    slide = servo & (np.asarray(m.jnt_type)[jid] == int(mujoco.mjtJoint.mjJNT_SLIDE))
    return low.astype(np.float64), high.astype(np.float64), servo, slide, np.where(joint, jid, -1)


def push_channel(sim: SceneSim, opts: AuditOptions) -> Channel | None:
    """Force axes on every free body of world 0 (``body_f`` components 0-2), or None when there is none."""
    model = sim.model
    if model.body_count == 0 or opts.push_g <= 0.0:
        return None
    world = model.body_world.numpy()
    bodies = np.nonzero(free_bodies(model) & (world == 0))[0]
    if bodies.size == 0:
        return None
    template = _template_ids(world, bodies)
    g = float(np.linalg.norm(np.asarray(model.gravity.numpy()).reshape(-1, 3)[0]))
    limit = np.repeat(opts.push_g * g * model.body_mass.numpy()[bodies], 3)
    index = tuple(int(6 * t + c) for t in template for c in range(3))
    max_delta = 2.0 * limit / (opts.push_rate_s / opts.control_dt)
    text = f"push: force +-{opts.push_g:g} x weight per axis on {bodies.size} free bodies"
    return Channel("body_f", index, -limit, limit, max_delta, np.zeros_like(limit), (), text)


def _template_ids(world: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Global ids of world-0 items -> their index among world 0's items (what ``template_to_global`` takes)."""
    own = np.nonzero(world == 0)[0]
    return np.searchsorted(own, ids)


def _joint_positions(model: newton.Model, joints: np.ndarray) -> np.ndarray:
    """``joint_q`` of each joint's first coordinate (NaN for -1)."""
    q, start = model.joint_q.numpy(), model.joint_q_start.numpy()
    return np.array([q[start[j]] if j >= 0 else np.nan for j in joints], dtype=np.float64)


def _articulation_bodies(model: newton.Model, joints: np.ndarray) -> tuple[int, ...]:
    """Template ids of the child bodies of every joint in the articulations holding ``joints`` (floating base
    included: a legged robot's base is the robot, not a passive object)."""
    if model.joint_count == 0 or model.joint_articulation is None:
        return ()
    art, child = model.joint_articulation.numpy(), model.joint_child.numpy()
    arts = {int(art[j]) for j in joints if j >= 0 and art[j] >= 0}
    bodies = np.array(sorted({int(child[j]) for j in range(art.size) if int(art[j]) in arts and child[j] >= 0}),
                      dtype=np.int64)
    return tuple(int(t) for t in _template_ids(model.body_world.numpy(), bodies))
