# SPDX-License-Identifier: Apache-2.0
"""Bully B0 pour scene ``sph_pour`` (SPEC §3.8 item 7, §5.2 LIQ-8): a MuJoCo-owned cup on a carriage (slide x) and a
hinge (tilt about y at the rim) pours into a static receiver; SolverDFSPH is the proxy destination (lagged, the
proxy integrated inside SolverDFSPH at the source's effective mass, SCR-WP9-2).

Actions are joint position targets [slide m, tilt rad]. Near the +x end of the slide the cup's +x wall reaches the
receiver's -x wall (20 mm slide + 8 mm cup wall vs 22 mm to the receiver's outer face): MuJoCo collides the cup
with the receiver natively and can hold the two walls closer than 2r with liquid between them (the LIQ-8 slit case,
kept on purpose). Newton's pipeline is off (``collide=False``; the liquid is not in MuJoCo). Registered with
``alloy.bully.targets`` on import of ``alloy.solvers.sph`` (SCR-WP9-5 adds the package to the bully provider list).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alloy._vendor import newton
import numpy as np
import warp as wp

from .config import DFSPHConfig, register_custom_attributes
from .pour import PourGeometry, add_cup_and_liquid, add_receiver_and_floor, scene_bounds

TARGET_NAME = "sph_pour"


def _default_geometry() -> PourGeometry:
    return PourGeometry(liquid_mass=0.1, cup_h=0.07)


@dataclass(frozen=True)
class PourTargetSpec:
    geometry: PourGeometry = field(default_factory=_default_geometry)
    slide: float = 0.02  # +- [m] about the start; near +slide the cup meets the receiver wall (MuJoCo contact)
    tilt_lo: float = -0.3  # [rad]
    tilt_hi: float = 2.4
    slide_speed: float = 0.3  # max commanded speed [m/s]
    tilt_speed: float = 3.0  # [rad/s]
    slide_ke: float = 2000.0  # [N/m]
    slide_kd: float = 50.0
    tilt_ke: float = 50.0  # [N m/rad]
    tilt_kd: float = 2.0
    carriage_mass: float = 0.2  # [kg]
    wall_density: float = 400.0  # cup walls [kg/m^3]
    control_dt: float = 0.02
    substeps: int = 64  # 0.3125 ms (pour CFL, D4 grid)


def pour_template(spec: PourTargetSpec, cfg: DFSPHConfig) -> newton.ModelBuilder:
    """One world: carriage (prismatic x) -> cup (revolute y at the pivot), liquid, receiver, floor."""
    g = spec.geometry
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    pivot = wp.transform((0.0, 0.0, g.pivot_z), wp.quat_identity())
    inertia = wp.mat33(np.eye(3) * 1.0e-4 * spec.carriage_mass)  # a compact 0.2 kg sled, no shapes
    carriage = b.add_link(xform=pivot, mass=spec.carriage_mass, inertia=inertia, label="carriage")
    slide = b.add_joint_prismatic(-1, carriage, parent_xform=pivot, axis=(1.0, 0.0, 0.0), target_ke=spec.slide_ke,
                                  target_kd=spec.slide_kd, limit_lower=-spec.slide, limit_upper=spec.slide)
    cup = b.add_link(xform=pivot, label="cup")
    hinge = b.add_joint_revolute(carriage, cup, axis=(0.0, 1.0, 0.0), target_ke=spec.tilt_ke, target_kd=spec.tilt_kd,
                                 limit_lower=spec.tilt_lo, limit_upper=spec.tilt_hi)
    b.add_articulation([slide, hinge])
    add_cup_and_liquid(b, cfg, g, cup, wall_density=spec.wall_density)
    add_receiver_and_floor(b, cfg, g)
    return b


def make_pour_sim(spec: PourTargetSpec, worlds: int, device: str):
    """DirectSim over SolverCoupledProxy(MuJoCo source, SPH destination) for ``worlds`` copies of the template."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    from alloy.bully.direct import DirectSim
    from alloy.core.newton_compat import SolverCoupledProxy

    from .solver import SolverDFSPH

    r = spec.geometry.particle_radius
    cfg = DFSPHConfig(particle_radius=r, world_bounds=scene_bounds(spec.geometry, 4.0 * r))
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(scene)
    SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(pour_template(spec, cfg), copy_count=worlds)
    m = scene.finalize(device=device)
    cups = [i for i, lbl in enumerate(m.body_label) if str(lbl).endswith("cup")]
    solver = SolverCoupledProxy(
        model=m,
        entries=[
            SolverCoupledProxy.Entry(name="mujoco", bodies=list(range(m.body_count)), joints=list(range(m.joint_count)),
                                     solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=True)),
            SolverCoupledProxy.Entry(name="sph", solver=lambda v: SolverDFSPH(v, cfg),
                                     particles=list(range(m.particle_count))),
        ],
        coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(
            source="mujoco", destination="sph", bodies=cups, mode="lagged", collision_pipeline=lambda _m: None)],
            iterations=1),
    )
    return DirectSim(m, solver, control_dt=spec.control_dt, substeps=spec.substeps, collide=False)


def sph_pour(spec: PourTargetSpec | None = None):
    """The registered B0 target factory."""
    from alloy.bully.types import ActionSpace, BullyTarget

    spec = spec or PourTargetSpec()
    rate = 1.0 / spec.control_dt
    actions = ActionSpace(low=np.array([-spec.slide, spec.tilt_lo]), high=np.array([spec.slide, spec.tilt_hi]),
                          rate_hz=rate, max_delta_per_tick=np.array([spec.slide_speed, spec.tilt_speed]) / rate,
                          initial=np.zeros(2))
    return BullyTarget(name=TARGET_NAME, scene=None, make_sim=lambda worlds, device: make_pour_sim(spec, worlds, device),
                       actions=actions, action_kind="joint_target_q", action_index=(0, 1))


def register() -> None:
    try:
        from alloy.bully.targets import register_target
    except ImportError:  # bully package absent in a trimmed install
        return
    register_target(TARGET_NAME, sph_pour)


__all__ = ["PourTargetSpec", "TARGET_NAME", "make_pour_sim", "pour_template", "register", "sph_pour"]
