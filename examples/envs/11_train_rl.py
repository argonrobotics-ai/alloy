# SPDX-License-Identifier: Apache-2.0
"""Train GripperLift with an RL library (rsl_rl, Stable-Baselines3 or skrl) and a critic that sees more than the policy.

The policy observes what a real gripper could measure: its joints, the cube's position and its last action. The critic
also gets the cube's orientation and velocities ("privileged" values that only a simulator knows), through
observation groups: Task(obs={"policy": [...], "critic": [...]}). rsl_rl and skrl give the critic group to the value
network; Stable-Baselines3 has one observation, so with it both networks see the policy group.

Run: python examples/envs/11_train_rl.py [lib] [num_envs] [iterations] [device]   lib: rsl_rl (default), sb3, skrl
     Quick check on the CPU: python examples/envs/11_train_rl.py rsl_rl 4 2 cpu    (device: the GPU if there is one)
Needs PyTorch and the library: pip install 'alloy-engine[envs]' rsl-rl-lib   (or stable-baselines3, or skrl).
Prints one JSON line per 10 iterations (episodes that ended in them, their success rate) and a summary line.
A ready-made env without groups, in one command: python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl
"""
import dataclasses
import importlib.util
import json
import sys

from alloy.envs import EnvError, VectorEnv
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task
from alloy.envs.wrappers.train import train

LIB = sys.argv[1] if len(sys.argv) > 1 else "rsl_rl"
NUM_ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
DEVICE = sys.argv[4] if len(sys.argv) > 4 else None         # None: cuda:0 when there is a GPU, else cpu

PACKAGES = {"rsl_rl": ("rsl_rl", "rsl-rl-lib"), "sb3": ("stable_baselines3", "stable-baselines3"),
            "skrl": ("skrl", "skrl")}                       # library -> (module, pip package)

GROUPS = {
    "policy": ["gripper", "cube.pos", "action"],            # joints (q, qd), cube position, last action
    "critic": ["gripper", "cube", "action"],                # + cube orientation, linear and angular velocity
}


def check_installed() -> None:
    """Stop with the pip command when PyTorch or the chosen library is missing."""
    if LIB not in PACKAGES:
        sys.exit(f"examples/envs/11_train_rl.py: unknown library {LIB!r}; use one of {sorted(PACKAGES)}")
    module, package = PACKAGES[LIB]
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec(module) is None:
        sys.exit(f"examples/envs/11_train_rl.py needs PyTorch and {package}: "
                 f"pip install 'alloy-engine[envs]' {package}")
    if LIB == "rsl_rl":
        import rsl_rl  # noqa: F401, PLC0415 - the default library
        import torch  # noqa: F401, PLC0415


def main() -> None:
    check_installed()
    task = dataclasses.replace(gripper_lift_task(), obs=GROUPS)
    try:
        env = VectorEnv(gripper_lift_scene(NUM_ENVS), task, worlds=NUM_ENVS, device=DEVICE, backend="torch")
        print(json.dumps({"groups": {g: len(env.obs_names(g)) for g in env.obs_groups}, "num_envs": env.num_envs,
                          "device": env.device}), flush=True)
        summary = train("GripperLift-v0", lib=LIB, env=env, num_envs=NUM_ENVS, iterations=ITERS)
    except EnvError as exc:
        sys.exit(f"examples/envs/11_train_rl.py: {exc.message}\nfix: {exc.fix}")
    print(json.dumps({"summary": summary}, default=str), flush=True)


if __name__ == "__main__":
    main()
