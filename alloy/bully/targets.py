# SPDX-License-Identifier: Apache-2.0
"""Bully target registry and the built-in raw-Newton targets.

A finding names its target, so ``Corpus.replay`` can rebuild it in a fresh process: targets are registered by
name with a zero-argument factory. Built in: ``xpbd_box_stack`` (the §7 WP7 GPU acceptance target: a stack of boxes,
a pusher on a three-axis prismatic D6 joint commanded through ``joint_target_q`` and tracked by a force- and
speed-limited servo (``alloy.bully.servo``), and a bystander box behind a 1 m wall whose displacement is the
non-target gate, VAL-116). Plug-ins register their B0 scenes from their own package
(``alloy.core.registry.load_plugins`` imports them all on the first unknown name);
``alloy.testing.bully`` registers the seeded FakeSolver targets.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.bully.servo import CartesianServo, ServoLimits
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.core.errors import RegistryError
from alloy.core.registry import load_plugins

log = logging.getLogger("alloy.bully")

_TARGETS: dict[str, Callable[[], BullyTarget]] = {}
_PROVIDERS = ("alloy.testing.bully", "alloy.bully.tasks", "alloy.tasks.emb_targets",  # register targets on import
              "alloy.topology.bully_target")  # WP10's cut_block_stroke (SB-7 cutting class)


def register_target(name: str, factory: Callable[[], BullyTarget]) -> None:
    """Register (or replace) a target factory under ``name``."""
    if name in _TARGETS:
        log.debug("bully target %s re-registered", name)
    _TARGETS[name] = factory


def _load_providers() -> None:
    """Import the WP7 providers, then every plug-in package incl. agent-written contrib ones (SCR-WP9-5): a plug-in
    registers its B0 scenes on import of ``alloy.solvers.<name>``."""
    for module in _PROVIDERS:
        try:
            importlib.import_module(module)
        except ImportError as exc:  # a provider may be absent in a trimmed install
            log.debug("bully target provider %s not importable: %s", module, exc)
    load_plugins(include_contrib=True)


def get_target(name: str) -> BullyTarget:
    """Build the target registered as ``name`` (imports the known provider modules once if needed)."""
    if name not in _TARGETS:
        _load_providers()
    if name not in _TARGETS:
        raise RegistryError(f"no bully target {name!r}; registered: {sorted(_TARGETS)}")
    return _TARGETS[name]()


def target_names() -> list[str]:
    _load_providers()
    return sorted(_TARGETS)


@dataclass(frozen=True)
class BoxStackSpec:
    """Geometry and drive of the XPBD box-stack target (SI)."""

    boxes: int = 3
    half: float = 0.05  # box half extent [m]
    box_mass: float = 0.2  # [kg]
    pusher_radius: float = 0.03
    pusher_mass: float = 1.0
    anchor: tuple[float, float, float] = (0.3, 0.0, 0.08)  # pusher joint anchor [m]
    reach: float = 0.4  # joint limits +-reach on x, y; z in [-0.05, 0.4]
    drive_ke: float = 2000.0  # [N/m] servo stiffness: gain = ke / kd [1/s]
    drive_kd: float = 100.0  # [N s/m]
    force_limit: float = 40.0  # [N] per axis (SolverXPBD ignores joint_effort_limit; the servo applies it)
    tcp_speed: float = 2.0  # max commanded and servo speed [m/s] -> max_delta_per_tick = tcp_speed / rate
    bystander_y: float = 1.0  # behind a static wall at y = wall_y, out of the pusher's reach (|y| <= reach + radius)
    wall_y: float = 0.8
    wall_height: float = 1.0  # [m] above the 0.82 m ballistic apex of a box thrown at 2 tcp_speed
    iterations: int = 8
    control_dt: float = 0.02
    substeps: int = 16  # 1.25 ms steps (D4 grid)


def _box_stack_template(spec: BoxStackSpec) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    ax = newton.ModelBuilder.JointDofConfig
    anchor = wp.transform(spec.anchor, wp.quat_identity())
    pusher = b.add_link(xform=anchor, mass=spec.pusher_mass)
    b.add_shape_sphere(pusher, radius=spec.pusher_radius)
    limits = ((-spec.reach, spec.reach), (-spec.reach, spec.reach), (-0.05, spec.reach))
    axes = [ax(axis=a, target_ke=0.0, target_kd=0.0, limit_lower=lo, limit_upper=hi)  # driven by CartesianServo
            for a, (lo, hi) in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), limits, strict=True)]
    joint = b.add_joint_d6(-1, pusher, linear_axes=axes, parent_xform=anchor)
    b.add_articulation([joint])
    for i in range(spec.boxes):
        z = spec.half + 2.0 * spec.half * i + 1e-3 * i
        body = b.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()), mass=spec.box_mass)
        b.add_shape_box(body, hx=spec.half, hy=spec.half, hz=spec.half)
    h = 0.5 * spec.wall_height
    b.add_shape_box(-1, xform=wp.transform((0.0, spec.wall_y, h), wp.quat_identity()), hx=1.0, hy=0.02, hz=h)
    bystander = b.add_body(xform=wp.transform((0.0, spec.bystander_y, spec.half), wp.quat_identity()),
                           mass=spec.box_mass)
    b.add_shape_box(bystander, hx=spec.half, hy=spec.half, hz=spec.half)
    b.add_ground_plane()
    return b


def xpbd_box_stack(spec: BoxStackSpec | None = None) -> BullyTarget:
    """Boxes stacked on the ground, a servo-driven spherical pusher and an untouched bystander, under ``SolverXPBD``."""
    spec = spec or BoxStackSpec()
    servo = ServoLimits(gain=spec.drive_ke / spec.drive_kd, kd=spec.drive_kd, v_lim=spec.tcp_speed,
                        f_max=spec.force_limit)

    def make_sim(worlds: int, device: str) -> DirectSim:
        scene = newton.ModelBuilder()
        scene.replicate(_box_stack_template(spec), copy_count=worlds)
        model = scene.finalize(device=device)
        solver = newton.solvers.SolverXPBD(model, iterations=spec.iterations)
        sim = DirectSim(model, solver, control_dt=spec.control_dt, substeps=spec.substeps)
        sim.pre_substep.append(CartesianServo(model, sim.state, sim.control, servo).apply)
        return sim

    rate = 1.0 / spec.control_dt
    low = np.array([-spec.reach, -spec.reach, -0.05])
    high = np.array([spec.reach, spec.reach, spec.reach])
    actions = ActionSpace(low=low, high=high, rate_hz=rate, max_delta_per_tick=np.full(3, spec.tcp_speed / rate),
                          initial=np.zeros(3))  # the pusher starts at its joint anchor
    return BullyTarget(name="xpbd_box_stack", scene=None, make_sim=make_sim, actions=actions,
                       non_target_bodies=(spec.boxes + 1,), robot_bodies=(0,))


register_target("xpbd_box_stack", xpbd_box_stack)
