# SPDX-License-Identifier: Apache-2.0
"""Go2 port of mjlab's Go1 flat velocity task (OUR port, not mjlab's): MuJoCo Menagerie unitree_go2 MJCF
(BSD-3, fetched on the VM at the pinned commit by install.sh, never copied into the repo) with Go1's env cfg reused.

Changes against the Go1 task: robot MJCF; body ``base`` renamed ``trunk``; foot geoms renamed ``<leg>_foot_collision``;
foot sites ``<leg>``; IMU / angular-momentum sensors added with the names the task reads; joint damping and friction
loss zeroed (the position actuator supplies damping); position actuators at 10 Hz / damping ratio 2 on the
MJCF's armature (0.01), effort limits 23.7 (hip, thigh) and 45.43 (knee) N m, action scale 0.25 * effort / stiffness
as for Go1; only the feet collide (as mjlab's FEET_ONLY_COLLISION). Default pose is Go1's (the Menagerie home key).
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

MENAGERIE = Path(os.environ.get("MENAGERIE", Path.home() / "menagerie"))
GO2_XML = MENAGERIE / "unitree_go2" / "go2.xml"
LEGS = ("FR", "FL", "RR", "RL")
ARMATURE = 0.01
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    for obj in list(spec.actuators) + list(spec.keys):
        spec.delete(obj)
    for j in spec.joints:
        j.damping = [0.0, 0.0, 0.0]
        j.frictionloss = 0.0
    spec.body("base").name = "trunk"
    for leg in LEGS:
        spec.geom(leg).name = f"{leg}_foot_collision"
        spec.body(f"{leg}_calf").add_site(name=leg, pos=[-0.002, 0, -0.213], size=[0.022, 0, 0], group=5)
    _add_sensors(spec)
    return spec


def _add_sensors(spec: mujoco.MjSpec) -> None:
    S, O = mujoco.mjtSensor, mujoco.mjtObj
    spec.add_sensor(name="imu_ang_vel", type=S.mjSENS_GYRO, objtype=O.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_lin_vel", type=S.mjSENS_VELOCIMETER, objtype=O.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_lin_acc", type=S.mjSENS_ACCELEROMETER, objtype=O.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_upvector", type=S.mjSENS_FRAMEZAXIS, objtype=O.mjOBJ_BODY, objname="world",
                    reftype=O.mjOBJ_SITE, refname="imu")
    spec.add_sensor(name="root_angmom", type=S.mjSENS_SUBTREEANGMOM, objtype=O.mjOBJ_BODY, objname="trunk")


def get_go2_robot_cfg():
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.asset_zoo.robots.unitree_go1.go1_constants import FEET_ONLY_COLLISION, INIT_STATE
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    def act(names, effort):
        return BuiltinPositionActuatorCfg(target_names_expr=names, stiffness=STIFFNESS, damping=DAMPING,
                                          effort_limit=effort, armature=ARMATURE)

    articulation = EntityArticulationInfoCfg(
        actuators=(act((".*_hip_joint", ".*_thigh_joint"), 23.7), act((".*_calf_joint",), 45.43)),
        soft_joint_pos_limit_factor=0.9,
    )
    return EntityCfg(init_state=INIT_STATE, collisions=(FEET_ONLY_COLLISION,), spec_fn=get_spec,
                     articulation=articulation)


def go2_action_scale() -> dict[str, float]:
    return {".*_hip_joint": 0.25 * 23.7 / STIFFNESS, ".*_thigh_joint": 0.25 * 23.7 / STIFFNESS,
            ".*_calf_joint": 0.25 * 45.43 / STIFFNESS}


def go2_flat_env_cfg():
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    cfg = unitree_go1_flat_env_cfg()
    cfg.scene.entities = {"robot": get_go2_robot_cfg()}
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = go2_action_scale()
    return cfg
