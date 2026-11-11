# SPDX-License-Identifier: Apache-2.0
"""Protocol metrics per task, computed from simulator state (qpos / xpos / sensors), never from the reward.

Each ``*_metrics`` function runs inside the JAX env stack, vmapped over envs, on the state BEFORE auto-reset.
It returns {name: per-env scalar}. ``TaskSpec`` says how the runner averages each name over an iteration:
``step`` names are averaged over every env and step; ``end`` names over the episodes that ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jp


@dataclass(frozen=True)
class TaskSpec:
    key: str  # protocol task key
    env_name: str  # MuJoCo Playground registry name
    gamma: float
    entropy: float
    step_names: tuple[str, ...]
    end_names: tuple[str, ...]
    metric_fn: Callable  # (raw_env, prev_state, state) -> {name: (N,)}
    ref_envs: int  # env count the shipped Warp contact buffers are sized for
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _go1(env, prev, state):
    cmd = prev.info["command"]  # the command this step's reward used (info is not reset by the auto-reset)

    def one(data, c):
        lin = env.get_local_linvel(data)
        gyro = env.get_gyro(data)
        return jp.linalg.norm(c[:2] - lin[:2]), jp.abs(c[2] - gyro[2])

    xy, yaw = jax.vmap(one)(state.data, cmd)
    # cmd_xy / vel_xy: evidence that the error is read against a non-zero command and a moving robot
    cmd_xy = jp.linalg.norm(cmd[:, :2], axis=-1)
    vel_xy = jp.linalg.norm(jax.vmap(lambda d: env.get_local_linvel(d)[:2])(state.data), axis=-1)
    return {"vel_err_xy": xy, "vel_err_yaw": yaw, "cmd_xy": cmd_xy, "vel_xy": vel_xy}


def _panda(env, prev, state):
    target = state.info["target_pos"]
    box = state.data.xpos[:, env._obj_body]
    err = jp.linalg.norm(target - box, axis=-1)
    lifted = (box[:, 2] - env._init_obj_pos[2] > 0.04).astype(jp.float32)
    return {
        "pos_err_end": err,
        "success_2cm": (err < 0.02).astype(jp.float32),
        "success_5cm": (err < 0.05).astype(jp.float32),
        "lifted": lifted,
    }


def _leap(env, prev, state):
    def one(data, prev_data):
        # the env moves the goal after scoring: measure against the goal the step was scored with
        scored = data.replace(mocap_quat=prev_data.mocap_quat)
        return env._cube_orientation_error(scored)

    err = jax.vmap(one)(state.data, prev.data)
    return {"rot_err": err, "goal_hit": (err < env._config.success_threshold).astype(jp.float32)}


TASKS: dict[str, TaskSpec] = {
    "go2_flat": TaskSpec(
        "go2_flat", "Go1JoystickFlatTerrain", 0.99, 0.01, ("vel_err_xy", "vel_err_yaw", "cmd_xy", "vel_xy"), (), _go1, 8192,
        "Go1 substitutes for Go2 (Playground has no Go2 env)",
    ),
    "franka_lift": TaskSpec(
        "franka_lift", "PandaPickCube", 0.995, 0.005, (),
        ("pos_err_end", "success_2cm", "success_5cm", "lifted"), _panda, 2048,
    ),
    "cube_reorient": TaskSpec(
        "cube_reorient", "LeapCubeReorient", 0.998, 0.002, ("rot_err", "goal_hit"), (), _leap, 8192,
    ),
}
