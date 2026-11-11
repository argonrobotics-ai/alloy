# SPDX-License-Identifier: Apache-2.0
"""An unofficial port of a Gym-Anything task as a Gym environment: run one episode on the port's own script (the
expert's actions), or on yours, and read the port's verdict at the end (Alloy's own checks: docs/guide/envs.md).

Each port env (``alloy.envs.ports.PORTS``: painting, rinsing, gift bags, porridge, forklift, concrete) runs its task's
scene and script one control period per ``step``. ``info["expert_action"]`` is what the script would do next;
stepping with it reproduces the port, any other action is the policy's. The observation is the proprio of what the
policy moves, the time, and a camera image framed on the scene (the camera does not draw paint or water films yet).
Run: python examples/envs/10_task_ports.py [env] [num_envs] [noise]
     env: ForkliftStartAndPark-v0 (default; about 40 s on a CPU), or e.g. RinseDrainWash-v0, GiftOpenFlaps-v0
     noise: added to the expert's action each step (0 = the script's own episode)
Prints one JSON line: steps, success per world, the reward at the end and the verdict's failed criteria.
"""
import json
import sys

import numpy as np

from alloy.envs import make

ENV = sys.argv[1] if len(sys.argv) > 1 else "ForkliftStartAndPark-v0"
NUM_ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NOISE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

env = make(ENV, num_envs=NUM_ENVS, camera=(64, 64), spp=1, autoreset="disabled")
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
steps = 0
while True:
    action = info["expert_action"] + NOISE * rng.standard_normal(info["expert_action"].shape).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(np.clip(action, env.single_action_space.low,
                                                                env.single_action_space.high))
    steps += 1
    if terminated.all():
        break
verdict = info["verdict"][0]
verdict = verdict[1] if isinstance(verdict, tuple) else verdict          # forklift: (episode, verdict)
failed = getattr(verdict, "failed", None)
print(json.dumps({"env": ENV, "steps": steps, "image": list(obs["image"].shape), "success": info["success"].tolist(),
                  "reward": reward.tolist(), "failed": failed() if callable(failed) else failed}, default=str))
env.close()
